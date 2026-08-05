"""
Weighting auto-tuner for the WDL model — the HONEST version.

The idea (Liam's): instead of hand-guessing the per-match training weights, search
for the weighting that generalises best. The trap to avoid: "a search is guaranteed
to raise accuracy" is only true for the metric measured on the data you optimise
against — optimise against the held-out test set and you manufacture a fake number.

So this module is built around three non-negotiable rules:
  1. Optimise VALIDATION log-loss (not accuracy). Log-loss is the project's real
     target (calibration), and accuracy is a noisy, argmax-discontinuous proxy that
     tempts the model to break its (already well-calibrated) draw probabilities.
  2. The search NEVER scores a 2016+ match. Candidate weightings are ranked by a
     walk-forward over PRE-2016 folds only (built in the A2 step). The 2016+
     tournaments stay pristine for one final honest comparison.
  3. The search PROPOSES a config; it does not silently overwrite the production
     weighting in train_wdl.py / data_pipeline.importance_weight. Promotion is a
     separate, human-reviewed step.

------------------------------------------------------------------------------
This file (A1): the parameterisation + a self-test proving the baseline config is
byte-identical to the current hand-tuned weighting. A2 (inner objective), A3
(random search) and A4 (final test comparison) build on top of it.
------------------------------------------------------------------------------

WHAT IS TUNABLE. The "manual weighting" is really three things:
  - the 8 IMPORTANCE-TIER VALUES (Friendly 0.3 ... World Cup 1.0 ... default 0.25).
    We vary the tier *values*, never the tier *membership* (which tournament sits
    in which tier) — that membership is imported straight from data_pipeline so it
    can't drift. NOTE the dual use: `importance` is BOTH a training-sample weight
    AND a model feature (it's in FEATURE_COLUMNS), so changing a tier value shifts
    both, consistently — apply_config handles that by remapping the whole column.
  - the recency HALF-LIFE in days (currently 3650 = 10 years).
  - the recency MIN-WEIGHT floor (currently 0.05).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Tier MEMBERSHIP is imported, not re-listed here, so it can never drift from the
# production definition. We only ever vary the tier VALUES.
from src.data_pipeline import (
    importance_weight,
    FIFA_MAJOR_FINALS,
    FIFA_MAJOR_QUALIFICATION,
    CONFEDERATION_SECOND_TIER,
    REGIONAL_CHAMPIONSHIPS,
    REGIONAL_QUALIFICATION,
    MULTISPORT_GAMES,
)
from src.models.build_matrix import PROCESSED_DIR
from src.models.train_wdl import load_matrix
from src.models.walk_forward import _sample_weights, walk_forward
# The leakage-safe inner objective + held-out comparison helpers are shared with
# the hyperparameter tuner — see tune_common.
from src.models.tune_common import (
    inner_folds, inner_objective, INNER_TEST_START,
    per_match_logloss, bootstrap_diff_ci, pooled_metrics,
)

# --- Tier names + the current (baseline) values -----------------------------
# These 8 names label the branches of data_pipeline.importance_weight. The values
# are the hand-chosen numbers currently in production — the thing the search will
# try to beat. Keeping them here (not just in importance_weight) is what lets a
# WeightConfig carry an *alternative* set of values through the pipeline.
TIER_NAMES = [
    "friendly", "major_finals", "major_qual", "confed_2nd",
    "regional_champ", "regional_qual", "multisport", "default",
]

BASELINE_TIERS: dict[str, float] = {
    "friendly":       0.30,
    "major_finals":   1.00,
    "major_qual":     0.60,
    "confed_2nd":     0.55,
    "regional_champ": 0.45,
    "regional_qual":  0.35,
    "multisport":     0.20,
    "default":        0.25,
}


def tier_of(tournament: str) -> str:
    """
    Which importance TIER a tournament belongs to. This reproduces the exact
    branch order of data_pipeline.importance_weight — same precedence, same sets —
    but returns the tier *name* instead of a hard-coded value, so the value can be
    supplied by a WeightConfig. The baseline self-test below guarantees this stays
    in lock-step with importance_weight.
    """
    if tournament == "Friendly":
        return "friendly"
    if tournament in FIFA_MAJOR_FINALS:
        return "major_finals"
    if tournament in FIFA_MAJOR_QUALIFICATION:
        return "major_qual"
    if tournament in CONFEDERATION_SECOND_TIER:
        return "confed_2nd"
    if tournament in REGIONAL_CHAMPIONSHIPS:
        return "regional_champ"
    if tournament in REGIONAL_QUALIFICATION:
        return "regional_qual"
    if tournament in MULTISPORT_GAMES:
        return "multisport"
    return "default"


@dataclass(frozen=True)
class WeightConfig:
    """
    One candidate weighting. `tiers` maps each TIER_NAME to its multiplier;
    `half_life_days` and `min_weight` parameterise the recency decay. Frozen so a
    config can't be mutated by accident mid-search (it's a fixed experiment point).
    """
    tiers: dict = field(default_factory=lambda: dict(BASELINE_TIERS))
    half_life_days: float = 3650.0
    min_weight: float = 0.05


# The production weighting, expressed as a WeightConfig — the thing to beat.
BASELINE = WeightConfig(tiers=dict(BASELINE_TIERS), half_life_days=3650.0, min_weight=0.05)


def importance_column(tournaments: pd.Series, tiers: dict) -> np.ndarray:
    """
    Vectorised tournament -> importance value under a given tier-value dict.
    Maps each distinct tournament once (there are ~149) and reuses the result, so
    this is cheap enough to call per search trial without a full matrix rebuild.
    """
    uniq = tournaments.unique()
    lookup = {t: tiers[tier_of(t)] for t in uniq}   # tier_of per distinct label only
    return tournaments.map(lookup).to_numpy()


def apply_config(df: pd.DataFrame, cfg: WeightConfig):
    """
    Turn a WeightConfig into the two things the backtest harness needs:

      df_trial : a copy of the matrix whose `importance` COLUMN is remapped to
                 cfg's tier values. Because `importance` is also a model feature,
                 remapping the column feeds the new values to BOTH the sample
                 weighting and the feature matrix — consistently, in one place.
      weight_fn: a (df, ref_date) -> weights function matching what walk_forward's
                 _fit_before expects, using cfg's recency half-life and floor. It
                 reads the (already remapped) `importance` column, so weight and
                 feature always agree.

    The weight math is a vectorised copy of data_pipeline.recency_weight
    (0.5 ** (days_ago / half_life), floored at min_weight) — identical numbers to
    the scalar version, just fast enough for thousands of refits. The baseline
    self-test asserts it matches walk_forward._sample_weights to full precision.
    """
    df_trial = df.copy()
    df_trial["importance"] = importance_column(df_trial["tournament"], cfg.tiers)

    hl, min_w = cfg.half_life_days, cfg.min_weight

    def weight_fn(sub: pd.DataFrame, ref_date: pd.Timestamp) -> np.ndarray:
        days_ago = (ref_date - sub["date"]).dt.days.to_numpy()
        recency = np.maximum(0.5 ** (days_ago / hl), min_w)
        return recency * sub["importance"].to_numpy()

    return df_trial, weight_fn


# The inner objective (leakage-safe pre-2016 walk-forward) lives in tune_common
# and is shared with the hyperparameter tuner. Here it is specialised to a
# WEIGHTING config: apply_config remaps importance + builds the weighting, then
# tune_common.inner_objective scores it with the production model builder.
def objective(cfg: WeightConfig, df: pd.DataFrame | None = None,
              folds=None, verbose: bool = False) -> dict:
    """Pooled PRE-2016 validation score for one weighting config (see
    tune_common.inner_objective). Returns {'log_loss','acc','n','n_fits',
    'max_scored_date'}; log_loss is the search objective."""
    if df is None:
        df = load_matrix()
    df_trial, weight_fn = apply_config(df, cfg)          # vary the weighting only
    return inner_objective(df_trial, weight_fn=weight_fn, folds=folds, verbose=verbose)


# ---------------------------------------------------------------------------
# A3: random search over the weighting, ranked by the A2 inner objective.
#
# Random search (not grid) because ~9 continuous dims make a grid explode, and
# random search is a strong, dependency-free baseline for that regime. Two design
# choices keep the results honest and interpretable:
#   - major_finals is ANCHORED at 1.0 (the scale reference "a World Cup match").
#     The other 7 tiers are sampled in [0.05, 1.0], i.e. "somewhere between nearly
#     ignored and as important as a World Cup match". This removes a redundant
#     global-scale degree of freedom (which would just interact with reg_lambda,
#     a Phase-B hyperparameter) and makes every tier readable relative to the anchor.
#   - trial 0 IS the baseline, so the best-found config is guaranteed no worse than
#     the current hand-tuned weighting ON THE VALIDATION folds. Whether that
#     validation edge survives on the held-out 2016+ test is the A4 question.
# ---------------------------------------------------------------------------

ANCHOR_TIER = "major_finals"          # fixed at 1.0; the scale reference
TIER_BOUNDS = (0.05, 1.0)             # range for the other 7 tiers
HALF_LIFE_BOUNDS = (365.0, 14600.0)   # 1 to 40 years, sampled LOG-uniform (scale param)
MIN_WEIGHT_BOUNDS = (0.0, 0.30)       # recency floor

# best_weight_config.json sits next to the model (small, chosen config — committed
# like a hyperparameter, NOT regenerable bulk data). The per-trial log is verbose
# and regenerable, so it goes in the gitignored data/processed/.
BEST_CONFIG_PATH = Path(__file__).resolve().parent / "best_weight_config.json"
TRIALS_PATH = PROCESSED_DIR / "weight_search_trials.csv"


def sample_config(rng: np.random.Generator) -> WeightConfig:
    """Draw one random WeightConfig from the search space described above."""
    tiers = {}
    for name in TIER_NAMES:
        tiers[name] = 1.0 if name == ANCHOR_TIER else float(rng.uniform(*TIER_BOUNDS))
    lo, hi = HALF_LIFE_BOUNDS
    half_life = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))   # log-uniform
    min_weight = float(rng.uniform(*MIN_WEIGHT_BOUNDS))
    return WeightConfig(tiers=tiers, half_life_days=half_life, min_weight=min_weight)


def _config_dict(cfg: WeightConfig) -> dict:
    return {"tiers": cfg.tiers, "half_life_days": cfg.half_life_days, "min_weight": cfg.min_weight}


def _trial_row(i: int, cfg: WeightConfig, res: dict, is_best: bool) -> dict:
    row = {"trial": i, "val_log_loss": res["log_loss"], "val_acc": res["acc"],
           "half_life_days": cfg.half_life_days, "min_weight": cfg.min_weight,
           "is_best_so_far": is_best}
    row.update({f"tier_{name}": cfg.tiers[name] for name in TIER_NAMES})
    return row


def random_search(k: int = 150, seed: int = 42, df: pd.DataFrame | None = None,
                  folds=None) -> dict:
    """
    Rank `k` random weightings (plus the baseline as trial 0) by pre-2016
    validation log-loss. Writes the running trial log to TRIALS_PATH and
    checkpoints the best config to BEST_CONFIG_PATH whenever it improves, so a
    long background run is inspectable and crash-safe. Returns a summary dict.
    """
    if df is None:
        df = load_matrix()
    if folds is None:
        folds = inner_folds()
    rng = np.random.default_rng(seed)
    t0 = time.time()

    # trial 0: the baseline (the thing to beat)
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
            _save_best(best_cfg, best_res, base_res, seed, k)   # checkpoint on every improvement
            print(f"trial {i:3d}  * NEW BEST  val log-loss {res['log_loss']:.4f}  "
                  f"(baseline {base_res['log_loss']:.4f}, {res['log_loss']-base_res['log_loss']:+.4f})")
        rows.append(_trial_row(i, cfg, res, improved))
        if i % 10 == 0:
            pd.DataFrame(rows).to_csv(TRIALS_PATH, index=False)   # periodic flush
            print(f"  ...{i}/{k} trials  ({time.time()-t0:.0f}s, best {best_res['log_loss']:.4f})", flush=True)

    pd.DataFrame(rows).to_csv(TRIALS_PATH, index=False)
    _save_best(best_cfg, best_res, base_res, seed, k)             # final save

    delta = best_res["log_loss"] - base_res["log_loss"]
    print(f"\n{'='*64}")
    print(f"search done: {k} trials in {time.time()-t0:.0f}s")
    print(f"  baseline val log-loss: {base_res['log_loss']:.4f}")
    print(f"  best     val log-loss: {best_res['log_loss']:.4f}   ({delta:+.4f})")
    if best_cfg is BASELINE:
        print("  -> no random config beat the hand-tuned baseline on validation.")
    else:
        print(f"  -> best config saved to {BEST_CONFIG_PATH.name} (a PROPOSAL; A4 tests it on 2016+).")
    print(f"{'='*64}")
    return {"best_cfg": best_cfg, "best_res": best_res, "base_res": base_res, "trials": rows}


def _save_best(cfg: WeightConfig, res: dict, base_res: dict, seed: int, k: int) -> None:
    """Persist the best config + its validation metrics. Explicitly a PROPOSAL:
    it is NOT wired into training until a human promotes it after seeing A4."""
    payload = {
        "_note": "PROPOSAL from tune_weights.random_search. NOT auto-applied to "
                 "training. Validation = pre-2016 walk-forward log-loss; the honest "
                 "test is the 2016+ comparison in tune_weights final_report (A4).",
        "config": _config_dict(cfg),
        "validation": {"log_loss": res["log_loss"], "acc": res["acc"], "n": res["n"]},
        "baseline_validation": {"log_loss": base_res["log_loss"], "acc": base_res["acc"]},
        "val_log_loss_improvement": base_res["log_loss"] - res["log_loss"],
        "search": {"seed": seed, "k": k, "is_baseline": cfg is BASELINE},
    }
    BEST_CONFIG_PATH.write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# A4: the ONE honest comparison. Take the search's proposed config and the
# baseline, run the REAL 2016+ walk-forward for each, and ask whether the
# validation edge survives on held-out data — with a paired bootstrap CI so we
# never over-claim a within-noise difference (the squad-feature precedent:
# +3.2pp val -> +1.1pp, inside the CI). This is the only place the test era is
# touched, and each config is scored exactly once.
# ---------------------------------------------------------------------------
def load_best_config() -> WeightConfig:
    """Rebuild the proposed WeightConfig from best_weight_config.json."""
    payload = json.loads(BEST_CONFIG_PATH.read_text())
    c = payload["config"]
    return WeightConfig(
        tiers={k: float(v) for k, v in c["tiers"].items()},
        half_life_days=float(c["half_life_days"]),
        min_weight=float(c["min_weight"]),
    )


def final_report(n_boot: int = 10000, seed: int = 0) -> None:
    """Run the 2016+ walk-forward for {baseline, proposed} and compare honestly."""
    df = load_matrix()
    payload = json.loads(BEST_CONFIG_PATH.read_text())
    best_cfg = load_best_config()
    is_baseline = payload.get("search", {}).get("is_baseline", False)

    print("Running 2016+ walk-forward for BASELINE weighting...", flush=True)
    df_b, wf_b = apply_config(df, BASELINE)
    rows_b, pooled_b = walk_forward(df_b, weight_fn=wf_b)

    print("Running 2016+ walk-forward for PROPOSED weighting...", flush=True)
    df_s, wf_s = apply_config(df, best_cfg)
    rows_s, pooled_s = walk_forward(df_s, weight_fn=wf_s)

    # both runs must score the exact same matches in the same order (only the
    # weighting changed) — otherwise the paired bootstrap is invalid.
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

    # paired bootstrap on the log-loss difference (proposed - baseline; negative = better)
    d = per_match_logloss(pooled_s) - per_match_logloss(pooled_b)
    point, lo, hi = bootstrap_diff_ci(d, n_boot=n_boot, seed=seed)
    print(f"\nlog-loss difference (proposed - baseline), paired bootstrap "
          f"[{n_boot} resamples]:")
    print(f"  point {point:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
    if lo > 0:
        verdict = "PROPOSED IS WORSE on held-out data (CI entirely > 0)."
        recommendation = "KEEP BASELINE"
    elif hi < 0:
        verdict = "PROPOSED IS BETTER on held-out data (CI entirely < 0)."
        recommendation = "CONSIDER PROMOTING (human review)"
    else:
        verdict = "WITHIN NOISE — the CI straddles 0; no real held-out gain."
        recommendation = "KEEP BASELINE (proposed not distinguishable from it)"
    print(f"  -> {verdict}")
    print(f"  -> recommendation: {recommendation}  (no production code changed either way)")
    print(f"{'='*72}")

    # Persist the held-out result + recommendation into the artifact so the JSON
    # tells the whole story (search proposal AND how it fared on the honest test).
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
# A1 verification: the baseline WeightConfig must reproduce the current weighting
# EXACTLY — both the importance column and the per-match sample weights. This is
# the regression guard that lets us trust every later (varied) config.
# ---------------------------------------------------------------------------
def _selftest_baseline() -> None:
    df = load_matrix()

    # (a) importance column: baseline tiers vs the production importance_weight(),
    #     checked on every distinct tournament label in the data.
    uniq = df["tournament"].unique()
    mismatches = [
        t for t in uniq
        if BASELINE_TIERS[tier_of(t)] != importance_weight(t)
    ]
    assert not mismatches, f"tier remap disagrees with importance_weight on: {mismatches}"
    print(f"[ok] baseline importance matches importance_weight on all {len(uniq)} tournaments")

    # (b) the remapped column equals the matrix's own precomputed `importance`.
    df_trial, weight_fn = apply_config(df, BASELINE)
    assert np.array_equal(df_trial["importance"].to_numpy(), df["importance"].to_numpy()), \
        "remapped importance column differs from the matrix's stored importance"
    print("[ok] baseline importance column is byte-identical to the stored matrix column")

    # (c) baseline weight_fn == walk_forward._sample_weights, to full float precision,
    #     on a real training slice (all matches before Euro 2016's start date).
    ref = pd.Timestamp("2016-06-10")
    train = df[df["date"] < ref]
    fast = weight_fn(train, ref)
    slow = _sample_weights(train, ref)
    assert np.allclose(fast, slow, rtol=0, atol=0), "vectorised weight_fn != _sample_weights"
    print(f"[ok] baseline weight_fn matches _sample_weights exactly on {len(train)} rows")

    print("\nA1 verified: the baseline config is the current weighting, exactly.")


def _verify_objective() -> None:
    """A2 verification: the inner objective runs, is sane, and — critically —
    never scores a 2016+ match."""
    df = load_matrix()
    folds = inner_folds()
    print(f"inner folds (pre-2016 rolling-origin): {[(s.year, e.year) for s, e in folds]}")

    res = objective(BASELINE, df=df, folds=folds, verbose=True)
    print(f"\nBASELINE inner objective:  log-loss={res['log_loss']:.4f}  "
          f"acc={res['acc']:.3f}  n={res['n']}  fits={res['n_fits']}")
    print(f"latest match ever scored during tuning: {res['max_scored_date'].date()}")

    # THE leakage guard: nothing scored on/after the 2016 wall.
    assert res["max_scored_date"] < INNER_TEST_START, \
        f"LEAKAGE: objective scored a match at {res['max_scored_date']} >= {INNER_TEST_START.date()}"
    print(f"[ok] zero matches scored on/after {INNER_TEST_START.date()} — test era untouched")

    print("\nA2 verified: inner objective is leakage-safe and returns a usable score.")


if __name__ == "__main__":
    # Run a stage from repo root, e.g.:
    #   python -m src.models.tune_weights selftest        # A1 regression check
    #   python -m src.models.tune_weights objective       # A2 leakage-safe objective
    #   python -m src.models.tune_weights search [k] [seed]  # A3 random search
    #   python -m src.models.tune_weights report           # A4 held-out comparison
    import sys
    stage = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if stage == "selftest":       # A1
        _selftest_baseline()
    elif stage == "objective":    # A2
        _verify_objective()
    elif stage == "search":       # A3
        k = int(sys.argv[2]) if len(sys.argv) > 2 else 150
        seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
        random_search(k=k, seed=seed)
    elif stage == "report":       # A4
        final_report()
    else:
        raise SystemExit(f"unknown stage {stage!r} (expected: selftest | objective | search | report)")
