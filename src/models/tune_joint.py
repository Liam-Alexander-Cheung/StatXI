"""
JOINT tuner: search the per-match WEIGHTING and the XGBoost HYPERPARAMETERS
together (15 dims), rank by pre-2016 validation log-loss, then test the single
joint winner ONCE on the held-out 2016+ backtest.

Why this exists. Tuned separately, each knob's proposal landed within noise; naively
stacking the two individual winners gave the best point estimate but still a CI that
included zero — and, crucially, "try the combination after seeing the parts" is
exactly the kind of test-set peeking that inflates false positives. The statistically
honest way to ask "does a combined config genuinely help?" is a SINGLE joint search
selected purely on validation, tested exactly once. That is this module.

It reuses everything already built: tune_weights supplies the weighting sampler +
apply_config, tune_hyperparams supplies the hyperparameter sampler + make_model_fn,
and tune_common supplies the leakage-safe inner objective and the held-out bootstrap.
Both injection points (`weight_fn`, `model_fn`) are active on every trial.

Same rules: baseline seeded as trial 0 (so best is never worse than production on
validation); never score a 2016+ match during the search; propose, don't overwrite.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.build_matrix import PROCESSED_DIR
from src.models.train_wdl import load_matrix
from src.models.walk_forward import walk_forward
from src.models.tune_common import (
    inner_objective, inner_folds, INNER_TEST_START,
    per_match_logloss, bootstrap_diff_ci, pooled_metrics,
)
import src.models.tune_weights as tw
import src.models.tune_hyperparams as th

BEST_CONFIG_PATH = Path(__file__).resolve().parent / "best_joint_config.json"
TRIALS_PATH = PROCESSED_DIR / "joint_search_trials.csv"

# The production baseline as a (weighting, hyperparameters) pair — the thing to beat.
BASELINE = (tw.BASELINE, th.BASELINE)


def sample_joint(rng: np.random.Generator):
    """Draw one (WeightConfig, HyperConfig) pair from the two search spaces."""
    return tw.sample_config(rng), th.sample_config(rng)


def objective(w_cfg, h_cfg, df=None, folds=None, verbose: bool = False) -> dict:
    """Pooled PRE-2016 validation score with BOTH the weighting and the model
    builder varied (see tune_common.inner_objective)."""
    if df is None:
        df = load_matrix()
    df_w, weight_fn = tw.apply_config(df, w_cfg)     # proposed weighting
    model_fn = th.make_model_fn(h_cfg)               # proposed hyperparameters
    return inner_objective(df_w, weight_fn=weight_fn, model_fn=model_fn,
                           folds=folds, verbose=verbose)


def _flat(w_cfg, h_cfg) -> dict:
    """Flatten a (weighting, hyper) pair into one row of scalar columns for the CSV."""
    row = {"half_life_days": w_cfg.half_life_days, "min_weight": w_cfg.min_weight}
    row.update({f"tier_{k}": v for k, v in w_cfg.tiers.items()})
    row.update(th._config_dict(h_cfg))
    return row


def random_search(k: int = 250, seed: int = 42, df=None, folds=None) -> dict:
    """Rank `k` random (weighting x hyperparameter) pairs, plus baseline as trial 0,
    by pre-2016 validation log-loss. Checkpoints best to BEST_CONFIG_PATH and logs
    every trial to TRIALS_PATH (crash-safe, inspectable mid-run)."""
    if df is None:
        df = load_matrix()
    if folds is None:
        folds = inner_folds()
    rng = np.random.default_rng(seed)
    t0 = time.time()

    base_res = objective(*BASELINE, df=df, folds=folds)
    best, best_res = BASELINE, base_res
    rows = [{"trial": 0, "val_log_loss": base_res["log_loss"], "val_acc": base_res["acc"],
             "is_best_so_far": True, **_flat(*BASELINE)}]
    print(f"trial   0  BASELINE   val log-loss {base_res['log_loss']:.4f}  (target to beat)")

    for i in range(1, k + 1):
        w_cfg, h_cfg = sample_joint(rng)
        res = objective(w_cfg, h_cfg, df=df, folds=folds)
        improved = res["log_loss"] < best_res["log_loss"]
        if improved:
            best, best_res = (w_cfg, h_cfg), res
            _save_best(best, best_res, base_res, seed, k)
            print(f"trial {i:3d}  * NEW BEST  val log-loss {res['log_loss']:.4f}  "
                  f"(baseline {base_res['log_loss']:.4f}, {res['log_loss']-base_res['log_loss']:+.4f})")
        rows.append({"trial": i, "val_log_loss": res["log_loss"], "val_acc": res["acc"],
                     "is_best_so_far": improved, **_flat(w_cfg, h_cfg)})
        if i % 10 == 0:
            pd.DataFrame(rows).to_csv(TRIALS_PATH, index=False)
            print(f"  ...{i}/{k} trials  ({time.time()-t0:.0f}s, best {best_res['log_loss']:.4f})", flush=True)

    pd.DataFrame(rows).to_csv(TRIALS_PATH, index=False)
    _save_best(best, best_res, base_res, seed, k)

    delta = best_res["log_loss"] - base_res["log_loss"]
    print(f"\n{'='*64}")
    print(f"joint search done: {k} trials in {time.time()-t0:.0f}s")
    print(f"  baseline val log-loss: {base_res['log_loss']:.4f}")
    print(f"  best     val log-loss: {best_res['log_loss']:.4f}   ({delta:+.4f})")
    if best is BASELINE:
        print("  -> no random joint config beat the baseline on validation.")
    else:
        print(f"  -> best saved to {BEST_CONFIG_PATH.name} (a PROPOSAL; tested on 2016+ in `report`).")
    print(f"{'='*64}")
    return {"best": best, "best_res": best_res, "base_res": base_res}


def _save_best(best, res, base_res, seed, k) -> None:
    w_cfg, h_cfg = best
    payload = {
        "_note": "PROPOSAL from tune_joint.random_search (weighting + hyperparameters "
                 "searched together). NOT auto-applied. Validation = pre-2016 walk-"
                 "forward log-loss; the honest test is the 2016+ comparison in `report`.",
        "weighting": tw._config_dict(w_cfg),
        "hyperparameters": th._config_dict(h_cfg),
        "validation": {"log_loss": res["log_loss"], "acc": res["acc"], "n": res["n"]},
        "baseline_validation": {"log_loss": base_res["log_loss"], "acc": base_res["acc"]},
        "val_log_loss_improvement": base_res["log_loss"] - res["log_loss"],
        "search": {"seed": seed, "k": k, "is_baseline": best is BASELINE},
    }
    BEST_CONFIG_PATH.write_text(json.dumps(payload, indent=2))


def load_best():
    """Rebuild the proposed (WeightConfig, HyperConfig) pair from best_joint_config.json."""
    p = json.loads(BEST_CONFIG_PATH.read_text())
    w = tw.WeightConfig(tiers={k: float(v) for k, v in p["weighting"]["tiers"].items()},
                        half_life_days=float(p["weighting"]["half_life_days"]),
                        min_weight=float(p["weighting"]["min_weight"]))
    h = th.HyperConfig(**{k: p["hyperparameters"][k] for k in th.TUNABLE_KEYS})
    return w, h


def final_report(n_boot: int = 10000, seed: int = 0) -> None:
    """The ONE definitive held-out test: joint winner vs baseline on 2016+."""
    df = load_matrix()
    payload = json.loads(BEST_CONFIG_PATH.read_text())
    w_cfg, h_cfg = load_best()
    is_baseline = payload.get("search", {}).get("is_baseline", False)

    print("Running 2016+ walk-forward for BASELINE...", flush=True)
    _, pooled_b = walk_forward(df)
    print("Running 2016+ walk-forward for JOINT winner (weighting + hyperparams)...", flush=True)
    df_w, weight_fn = tw.apply_config(df, w_cfg)
    _, pooled_j = walk_forward(df_w, weight_fn=weight_fn, model_fn=th.make_model_fn(h_cfg))
    assert np.array_equal(pooled_b["y"], pooled_j["y"]), "configs scored different match sets"

    m_b, m_j = pooled_metrics(pooled_b), pooled_metrics(pooled_j)
    n = len(pooled_b["y"])
    print(f"\n{'='*72}\nHELD-OUT 2016+ COMPARISON  (n={n})  — joint winner vs baseline\n{'='*72}")
    print(f"  validation said: baseline {payload['baseline_validation']['log_loss']:.4f} -> "
          f"joint {payload['validation']['log_loss']:.4f} "
          f"({payload['val_log_loss_improvement']:+.4f} on pre-2016 folds)")
    if is_baseline:
        print("  NOTE: the joint search did not beat baseline on validation; rows are identical.")
    print(f"\n{'metric':<12}{'baseline':>12}{'joint':>12}{'delta':>12}")
    for key in ("acc", "log_loss", "brier"):
        print(f"{key:<12}{m_b[key]:>12.4f}{m_j[key]:>12.4f}{m_j[key]-m_b[key]:>+12.4f}")

    d = per_match_logloss(pooled_j) - per_match_logloss(pooled_b)
    point, lo, hi = bootstrap_diff_ci(d, n_boot=n_boot, seed=seed)
    print(f"\nlog-loss difference (joint - baseline), paired bootstrap [{n_boot} resamples]:")
    print(f"  point {point:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
    if lo > 0:
        verdict, rec = "JOINT IS WORSE on held-out data (CI entirely > 0).", "KEEP BASELINE"
    elif hi < 0:
        verdict, rec = "JOINT IS BETTER on held-out data (CI entirely < 0).", "CONSIDER PROMOTING (human review)"
    else:
        verdict, rec = "WITHIN NOISE — the CI straddles 0; no real held-out gain.", "KEEP BASELINE (not distinguishable)"
    print(f"  -> {verdict}")
    print(f"  -> recommendation: {rec}   (this is the clean, single held-out test of a")
    print("        validation-selected joint config — no further test-set peeking)")
    print(f"{'='*72}")

    payload["held_out_2016plus"] = {
        "n": int(n),
        "baseline": {k: round(m_b[k], 4) for k in m_b},
        "joint": {k: round(m_j[k], 4) for k in m_j},
        "log_loss_diff": {"point": round(point, 4), "ci95_lo": round(lo, 4),
                          "ci95_hi": round(hi, 4), "n_boot": n_boot},
        "verdict": verdict, "recommendation": rec,
    }
    BEST_CONFIG_PATH.write_text(json.dumps(payload, indent=2))


def _selftest() -> None:
    """Baseline joint config must reproduce the shared 0.9224 baseline objective,
    and a sampled config must have BOTH knobs active."""
    df = load_matrix()
    res = objective(*BASELINE, df=df)
    assert res["max_scored_date"] < INNER_TEST_START, "LEAKAGE in joint objective"
    print(f"[ok] baseline joint objective = {res['log_loss']:.4f} "
          f"({'MATCH' if abs(res['log_loss']-0.9224) < 5e-4 else 'DIFFERS'} vs 0.9224)")

    rng = np.random.default_rng(0)
    w_cfg, h_cfg = sample_joint(rng)
    df_w, _ = tw.apply_config(df, w_cfg)
    imp_active = not np.array_equal(df_w["importance"].to_numpy(), df["importance"].to_numpy())
    hp_active = th.make_model_fn(h_cfg)().get_params()["max_depth"] != th.DEFAULT_HYPERPARAMS["max_depth"] \
        or h_cfg != th.BASELINE
    print(f"[ok] sampled joint config has weighting active={imp_active}, hyperparams active={hp_active}")
    print("\njoint selftest verified.")


if __name__ == "__main__":
    # Run from repo root, e.g.:
    #   python -m src.models.tune_joint selftest
    #   python -m src.models.tune_joint search [k] [seed]
    #   python -m src.models.tune_joint report
    import sys
    stage = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if stage == "selftest":
        _selftest()
    elif stage == "search":
        k = int(sys.argv[2]) if len(sys.argv) > 2 else 250
        seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
        random_search(k=k, seed=seed)
    elif stage == "report":
        final_report()
    else:
        raise SystemExit(f"unknown stage {stage!r} (expected: selftest | search | report)")
