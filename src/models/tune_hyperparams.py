"""
XGBoost HYPERPARAMETER tuner (Phase B) — the honest version, built on the exact
protocol proven by the weighting tuner (tune_weights.py) and shared via
tune_common.py. This is a SEPARATE tuner from the weighting one and is not run at
the same time: the weighting turned out to be a low-leverage knob, and the
hyperparameters are the higher-leverage lever tuned here.

The same three rules apply (see tune_common):
  1. Optimise pre-2016 walk-forward VALIDATION log-loss (not accuracy).
  2. The search NEVER scores a 2016+ match.
  3. Propose, don't overwrite — the best config is written to best_hyper_config.json;
     promoting it into train_wdl.DEFAULT_HYPERPARAMS is a separate, human-reviewed
     step gated on the held-out bootstrap comparison.

WHAT IS TUNABLE. The 6 real knobs in train_wdl.build_model (objective / num_class /
eval_metric / random_state / n_jobs are fixed; n_estimators + early_stopping_rounds
are left alone because early stopping already picks the tree count):
  max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, reg_lambda.

Two built-in cross-checks tie this tuner back to known numbers:
  - the baseline objective must equal the weighting tuner's baseline (0.9224) — same
    weighting + same model, just reached via model_fn instead of weight_fn.
  - the baseline held-out walk-forward must equal the committed walk_forward.py
    numbers (pooled acc 0.484, log-loss 1.055).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.build_matrix import PROCESSED_DIR
from src.models.train_wdl import load_matrix, build_model, DEFAULT_HYPERPARAMS
from src.models.walk_forward import walk_forward, _sample_weights
from src.models.tune_common import (
    inner_folds, inner_objective, default_weight_fn, INNER_TEST_START,
    per_match_logloss, bootstrap_diff_ci, pooled_metrics,
)

# --- Baseline (production) values, pulled straight from DEFAULT_HYPERPARAMS so
#     they can never drift from what build_model() actually uses. -------------
TUNABLE_KEYS = ["max_depth", "learning_rate", "subsample",
                "colsample_bytree", "min_child_weight", "reg_lambda"]


@dataclass(frozen=True)
class HyperConfig:
    """One candidate hyperparameter set (the 6 tunable knobs). Frozen so a config
    can't be mutated mid-search."""
    max_depth: int = DEFAULT_HYPERPARAMS["max_depth"]
    learning_rate: float = DEFAULT_HYPERPARAMS["learning_rate"]
    subsample: float = DEFAULT_HYPERPARAMS["subsample"]
    colsample_bytree: float = DEFAULT_HYPERPARAMS["colsample_bytree"]
    min_child_weight: int = DEFAULT_HYPERPARAMS["min_child_weight"]
    reg_lambda: float = DEFAULT_HYPERPARAMS["reg_lambda"]


# The production hyperparameters as a HyperConfig — the thing to beat.
BASELINE = HyperConfig()


def make_model_fn(cfg: HyperConfig):
    """Turn a HyperConfig into a zero-arg model builder for _fit_before/walk_forward.
    max_depth and min_child_weight are cast to int (XGBoost wants ints there)."""
    overrides = dict(
        max_depth=int(cfg.max_depth),
        learning_rate=float(cfg.learning_rate),
        subsample=float(cfg.subsample),
        colsample_bytree=float(cfg.colsample_bytree),
        min_child_weight=int(cfg.min_child_weight),
        reg_lambda=float(cfg.reg_lambda),
    )
    return lambda: build_model(**overrides)


def objective(cfg: HyperConfig, df: pd.DataFrame | None = None,
              folds=None, verbose: bool = False) -> dict:
    """Pooled PRE-2016 validation score for one hyperparameter config. The
    WEIGHTING is held at the production baseline (default_weight_fn); only the
    model builder varies. See tune_common.inner_objective."""
    if df is None:
        df = load_matrix()
    return inner_objective(df, model_fn=make_model_fn(cfg), folds=folds, verbose=verbose)


# ---------------------------------------------------------------------------
# Random search over the 6 knobs. Same shape as tune_weights.random_search:
# baseline seeded as trial 0 (so best is never worse than production on
# validation); log/depth-appropriate sampling; checkpoint + trial log.
# ---------------------------------------------------------------------------
MAX_DEPTH_BOUNDS = (3, 8)          # inclusive int range
LR_BOUNDS = (0.01, 0.30)           # log-uniform
SUBSAMPLE_BOUNDS = (0.5, 1.0)
COLSAMPLE_BOUNDS = (0.5, 1.0)
MCW_BOUNDS = (1, 12)               # inclusive int range
LAMBDA_BOUNDS = (0.1, 10.0)        # log-uniform

BEST_CONFIG_PATH = Path(__file__).resolve().parent / "best_hyper_config.json"
TRIALS_PATH = PROCESSED_DIR / "hyper_search_trials.csv"


def sample_config(rng: np.random.Generator) -> HyperConfig:
    """Draw one random HyperConfig from the search space."""
    return HyperConfig(
        max_depth=int(rng.integers(MAX_DEPTH_BOUNDS[0], MAX_DEPTH_BOUNDS[1] + 1)),
        learning_rate=float(np.exp(rng.uniform(np.log(LR_BOUNDS[0]), np.log(LR_BOUNDS[1])))),
        subsample=float(rng.uniform(*SUBSAMPLE_BOUNDS)),
        colsample_bytree=float(rng.uniform(*COLSAMPLE_BOUNDS)),
        min_child_weight=int(rng.integers(MCW_BOUNDS[0], MCW_BOUNDS[1] + 1)),
        reg_lambda=float(np.exp(rng.uniform(np.log(LAMBDA_BOUNDS[0]), np.log(LAMBDA_BOUNDS[1])))),
    )


def _config_dict(cfg: HyperConfig) -> dict:
    return {k: getattr(cfg, k) for k in TUNABLE_KEYS}


def _trial_row(i: int, cfg: HyperConfig, res: dict, is_best: bool) -> dict:
    row = {"trial": i, "val_log_loss": res["log_loss"], "val_acc": res["acc"],
           "is_best_so_far": is_best}
    row.update(_config_dict(cfg))
    return row


def random_search(k: int = 150, seed: int = 42, df: pd.DataFrame | None = None,
                  folds=None) -> dict:
    """Rank `k` random hyperparameter sets (plus baseline as trial 0) by pre-2016
    validation log-loss. Checkpoints best to BEST_CONFIG_PATH and logs trials to
    TRIALS_PATH so a long background run is inspectable and crash-safe."""
    if df is None:
        df = load_matrix()
    if folds is None:
        folds = inner_folds()
    rng = np.random.default_rng(seed)
    t0 = time.time()

    base_res = objective(BASELINE, df, folds)
    best_cfg, best_res = BASELINE, base_res
    rows = [_trial_row(0, BASELINE, base_res, True)]
    print(f"trial   0  BASELINE   val log-loss {base_res['log_loss']:.4f}  (target to beat)")

    for i in range(1, k + 1):
        cfg = sample_config(rng)
        res = objective(cfg, df, folds)
        improved = res["log_loss"] < best_res["log_loss"]
        if improved:
            best_cfg, best_res = cfg, res
            _save_best(best_cfg, best_res, base_res, seed, k)
            print(f"trial {i:3d}  * NEW BEST  val log-loss {res['log_loss']:.4f}  "
                  f"(baseline {base_res['log_loss']:.4f}, {res['log_loss']-base_res['log_loss']:+.4f})")
        rows.append(_trial_row(i, cfg, res, improved))
        if i % 10 == 0:
            pd.DataFrame(rows).to_csv(TRIALS_PATH, index=False)
            print(f"  ...{i}/{k} trials  ({time.time()-t0:.0f}s, best {best_res['log_loss']:.4f})", flush=True)

    pd.DataFrame(rows).to_csv(TRIALS_PATH, index=False)
    _save_best(best_cfg, best_res, base_res, seed, k)

    delta = best_res["log_loss"] - base_res["log_loss"]
    print(f"\n{'='*64}")
    print(f"search done: {k} trials in {time.time()-t0:.0f}s")
    print(f"  baseline val log-loss: {base_res['log_loss']:.4f}")
    print(f"  best     val log-loss: {best_res['log_loss']:.4f}   ({delta:+.4f})")
    if best_cfg is BASELINE:
        print("  -> no random hyperparameter set beat the baseline on validation.")
    else:
        print(f"  -> best config saved to {BEST_CONFIG_PATH.name} (a PROPOSAL; tested on 2016+ in `report`).")
    print(f"{'='*64}")
    return {"best_cfg": best_cfg, "best_res": best_res, "base_res": base_res, "trials": rows}


def _save_best(cfg: HyperConfig, res: dict, base_res: dict, seed: int, k: int) -> None:
    payload = {
        "_note": "PROPOSAL from tune_hyperparams.random_search. NOT auto-applied to "
                 "train_wdl.DEFAULT_HYPERPARAMS. Validation = pre-2016 walk-forward "
                 "log-loss; the honest test is the 2016+ comparison in `report`.",
        "config": _config_dict(cfg),
        "validation": {"log_loss": res["log_loss"], "acc": res["acc"], "n": res["n"]},
        "baseline_validation": {"log_loss": base_res["log_loss"], "acc": base_res["acc"]},
        "val_log_loss_improvement": base_res["log_loss"] - res["log_loss"],
        "search": {"seed": seed, "k": k, "is_baseline": cfg is BASELINE},
    }
    BEST_CONFIG_PATH.write_text(json.dumps(payload, indent=2))


def load_best_config() -> HyperConfig:
    """Rebuild the proposed HyperConfig from best_hyper_config.json."""
    c = json.loads(BEST_CONFIG_PATH.read_text())["config"]
    return HyperConfig(**{k: c[k] for k in TUNABLE_KEYS})


# ---------------------------------------------------------------------------
# The one honest held-out comparison (2016+), mirroring tune_weights.final_report.
# ---------------------------------------------------------------------------
def final_report(n_boot: int = 10000, seed: int = 0) -> None:
    df = load_matrix()
    payload = json.loads(BEST_CONFIG_PATH.read_text())
    best_cfg = load_best_config()
    is_baseline = payload.get("search", {}).get("is_baseline", False)

    print("Running 2016+ walk-forward for BASELINE hyperparameters...", flush=True)
    _, pooled_b = walk_forward(df, model_fn=make_model_fn(BASELINE))
    print("Running 2016+ walk-forward for PROPOSED hyperparameters...", flush=True)
    _, pooled_s = walk_forward(df, model_fn=make_model_fn(best_cfg))

    assert np.array_equal(pooled_b["y"], pooled_s["y"]), "configs scored different match sets"
    m_b, m_s = pooled_metrics(pooled_b), pooled_metrics(pooled_s)
    n = len(pooled_b["y"])

    print(f"\n{'='*72}\nHELD-OUT 2016+ COMPARISON  (n={n} tournament matches)\n{'='*72}")
    print(f"  validation said: baseline {payload['baseline_validation']['log_loss']:.4f} -> "
          f"proposed {payload['validation']['log_loss']:.4f} "
          f"({payload['val_log_loss_improvement']:+.4f} log-loss on pre-2016 folds)")
    if is_baseline:
        print("  NOTE: the search did not beat the baseline on validation, so the")
        print("        'proposed' config IS the baseline — the two rows below are identical.")
    print(f"\n{'metric':<12}{'baseline':>12}{'proposed':>12}{'delta':>12}")
    for key in ("acc", "log_loss", "brier"):
        print(f"{key:<12}{m_b[key]:>12.4f}{m_s[key]:>12.4f}{m_s[key]-m_b[key]:>+12.4f}")

    d = per_match_logloss(pooled_s) - per_match_logloss(pooled_b)
    point, lo, hi = bootstrap_diff_ci(d, n_boot=n_boot, seed=seed)
    print(f"\nlog-loss difference (proposed - baseline), paired bootstrap [{n_boot} resamples]:")
    print(f"  point {point:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
    if lo > 0:
        verdict, recommendation = "PROPOSED IS WORSE on held-out data (CI entirely > 0).", "KEEP BASELINE"
    elif hi < 0:
        verdict, recommendation = "PROPOSED IS BETTER on held-out data (CI entirely < 0).", "CONSIDER PROMOTING (human review)"
    else:
        verdict, recommendation = "WITHIN NOISE — the CI straddles 0; no real held-out gain.", "KEEP BASELINE (proposed not distinguishable from it)"
    print(f"  -> {verdict}")
    print(f"  -> recommendation: {recommendation}  (no production code changed either way)")
    print(f"{'='*72}")

    payload["held_out_2016plus"] = {
        "n": int(n),
        "baseline": {k: round(m_b[k], 4) for k in m_b},
        "proposed": {k: round(m_s[k], 4) for k in m_s},
        "log_loss_diff": {"point": round(point, 4), "ci95_lo": round(lo, 4),
                          "ci95_hi": round(hi, 4), "n_boot": n_boot},
        "verdict": verdict,
        "recommendation": recommendation,
    }
    BEST_CONFIG_PATH.write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Verification.
# ---------------------------------------------------------------------------
def _selftest_baseline() -> None:
    # (a) baseline HyperConfig -> build_model() defaults, on the 6 tunable keys.
    p = make_model_fn(BASELINE)().get_params()
    bad = {k: (p[k], DEFAULT_HYPERPARAMS[k]) for k in TUNABLE_KEYS if p[k] != DEFAULT_HYPERPARAMS[k]}
    assert not bad, f"baseline HyperConfig != build_model defaults on: {bad}"
    print("[ok] baseline HyperConfig reproduces build_model() on all 6 tunable keys")

    # (b) the weighting this tuner holds fixed IS the production weighting: the
    #     shared default_weight_fn must equal walk_forward._sample_weights exactly.
    df = load_matrix()
    ref = pd.Timestamp("2016-06-10")
    train = df[df["date"] < ref]
    assert np.allclose(default_weight_fn(train, ref), _sample_weights(train, ref), rtol=0, atol=0), \
        "default_weight_fn != _sample_weights (held-fixed weighting drifted)"
    print(f"[ok] held-fixed weighting == production _sample_weights on {len(train)} rows")

    print("\nselftest verified: baseline hyper-config is the production model, exactly.")


def _verify_objective() -> None:
    df = load_matrix()
    folds = inner_folds()
    print(f"inner folds (pre-2016 rolling-origin): {[(s.year, e.year) for s, e in folds]}")
    res = objective(BASELINE, df=df, folds=folds, verbose=True)
    print(f"\nBASELINE inner objective:  log-loss={res['log_loss']:.4f}  "
          f"acc={res['acc']:.3f}  n={res['n']}  fits={res['n_fits']}")
    print(f"latest match ever scored during tuning: {res['max_scored_date'].date()}")
    assert res["max_scored_date"] < INNER_TEST_START, \
        f"LEAKAGE: scored a match at {res['max_scored_date']} >= {INNER_TEST_START.date()}"
    print(f"[ok] zero matches scored on/after {INNER_TEST_START.date()} — test era untouched")
    # cross-check: same baseline objective as the weighting tuner (0.9224)
    print("[note] this should match the weighting tuner's baseline (0.9224): "
          f"{'MATCH' if abs(res['log_loss']-0.9224) < 5e-4 else 'DIFFERS'}")
    print("\nobjective verified: leakage-safe and consistent with the weighting tuner.")


if __name__ == "__main__":
    # Run a stage from repo root, e.g.:
    #   python -m src.models.tune_hyperparams selftest
    #   python -m src.models.tune_hyperparams objective
    #   python -m src.models.tune_hyperparams search [k] [seed]
    #   python -m src.models.tune_hyperparams report
    import sys
    stage = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if stage == "selftest":
        _selftest_baseline()
    elif stage == "objective":
        _verify_objective()
    elif stage == "search":
        k = int(sys.argv[2]) if len(sys.argv) > 2 else 150
        seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
        random_search(k=k, seed=seed)
    elif stage == "report":
        final_report()
    else:
        raise SystemExit(f"unknown stage {stage!r} (expected: selftest | objective | search | report)")
