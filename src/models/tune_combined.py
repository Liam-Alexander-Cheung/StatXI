"""
Combined evaluation: apply the proposed WEIGHTING and the proposed XGB
HYPERPARAMETERS AT THE SAME TIME, and ask whether together they beat the baseline
where neither did alone.

Each tuner selected its proposal on pre-2016 validation and, tested once on the
held-out 2016+ backtest, landed WITHIN NOISE. This script checks the natural
follow-up — do the two marginal nudges stack? It reuses both tuners' loaders and
the shared honest harness (tune_common); both injection points (`weight_fn` and
`model_fn`) are simply active at once.

HONESTY CAVEAT (printed at the end too): this is now the THIRD distinct config
tested against the same 281-match held-out set (weighting-only, hyperparams-only,
and now combined). Testing more candidates against one small test set inflates the
chance that one clears the CI by luck. The clean way to justify a combined config
would be a single JOINT pre-2016 search, then one held-out test. So we validate
first (pre-2016) and treat the held-out number as indicative, not a licence to
promote on its own.
"""

from __future__ import annotations

import json

import numpy as np

from src.models.train_wdl import load_matrix
from src.models.walk_forward import walk_forward
from src.models.tune_common import (
    inner_objective, inner_folds, INNER_TEST_START,
    per_match_logloss, bootstrap_diff_ci, pooled_metrics,
)
import src.models.tune_weights as tw
import src.models.tune_hyperparams as th


def _load_proposals():
    """The two saved proposals (raises if a search hasn't been run)."""
    return tw.load_best_config(), th.load_best_config()


def check_both_active(df, w_cfg, h_cfg) -> tuple[bool, dict]:
    """Confirm the combined config genuinely applies BOTH changes:
      - weighting active  -> the remapped `importance` column differs from baseline
      - hyperparams active -> the model's params differ from build_model()'s defaults
    Returns (importance_changed, {param: (baseline, proposed)})."""
    df_w, _ = tw.apply_config(df, w_cfg)
    importance_changed = not np.array_equal(
        df_w["importance"].to_numpy(), df["importance"].to_numpy()
    )
    base_params = th.make_model_fn(th.BASELINE)().get_params()
    prop_params = th.make_model_fn(h_cfg)().get_params()
    hp_changed = {k: (base_params[k], prop_params[k])
                  for k in th.TUNABLE_KEYS if base_params[k] != prop_params[k]}
    return importance_changed, hp_changed


def combined_injection(df, w_cfg, h_cfg):
    """(df_with_remapped_importance, weight_fn, model_fn) with BOTH proposals active."""
    df_w, weight_fn = tw.apply_config(df, w_cfg)   # proposed weighting
    model_fn = th.make_model_fn(h_cfg)             # proposed hyperparameters
    return df_w, weight_fn, model_fn


def main(n_boot: int = 10000, seed: int = 0) -> None:
    df = load_matrix()
    w_cfg, h_cfg = _load_proposals()

    # --- 1. are BOTH active? -------------------------------------------------
    importance_changed, hp_changed = check_both_active(df, w_cfg, h_cfg)
    print(f"{'='*72}\nBOTH-ACTIVE CHECK\n{'='*72}")
    print(f"  weighting active   : {importance_changed}  (importance column remapped)")
    print(f"  hyperparams active : {bool(hp_changed)}  changed {list(hp_changed)}")
    assert importance_changed and hp_changed, "combined config is not applying both changes"
    print("  [ok] both the proposed weighting AND the proposed hyperparameters are active")

    # --- 2. validation first: does combining STACK on pre-2016 folds? --------
    folds = inner_folds()
    df_w, weight_fn, model_fn = combined_injection(df, w_cfg, h_cfg)
    res = inner_objective(df_w, weight_fn=weight_fn, model_fn=model_fn, folds=folds)
    assert res["max_scored_date"] < INNER_TEST_START, "LEAKAGE in combined validation"

    # individual validation numbers come straight from each search's saved JSON
    base_val = 0.9224
    w_val = json.loads(tw.BEST_CONFIG_PATH.read_text())["validation"]["log_loss"]
    h_val = json.loads(th.BEST_CONFIG_PATH.read_text())["validation"]["log_loss"]
    print(f"\n{'='*72}\nVALIDATION (pre-2016 walk-forward log-loss, lower better)\n{'='*72}")
    print(f"  baseline            : {base_val:.4f}")
    print(f"  + weighting only    : {w_val:.4f}   ({w_val-base_val:+.4f})")
    print(f"  + hyperparams only  : {h_val:.4f}   ({h_val-base_val:+.4f})")
    print(f"  + BOTH (combined)   : {res['log_loss']:.4f}   ({res['log_loss']-base_val:+.4f})")
    additive = (w_val - base_val) + (h_val - base_val)
    print(f"  (if the two nudges were purely additive we'd expect {base_val+additive:.4f})")

    # --- 3. one held-out 2016+ comparison: combined vs baseline --------------
    print(f"\nRunning 2016+ walk-forward for BASELINE...", flush=True)
    _, pooled_b = walk_forward(df)                                   # default weight + model
    print("Running 2016+ walk-forward for COMBINED (both proposals)...", flush=True)
    _, pooled_c = walk_forward(df_w, weight_fn=weight_fn, model_fn=model_fn)
    assert np.array_equal(pooled_b["y"], pooled_c["y"]), "configs scored different match sets"

    m_b, m_c = pooled_metrics(pooled_b), pooled_metrics(pooled_c)
    n = len(pooled_b["y"])
    print(f"\n{'='*72}\nHELD-OUT 2016+ COMPARISON  (n={n})\n{'='*72}")
    print(f"{'metric':<12}{'baseline':>12}{'combined':>12}{'delta':>12}")
    for key in ("acc", "log_loss", "brier"):
        print(f"{key:<12}{m_b[key]:>12.4f}{m_c[key]:>12.4f}{m_c[key]-m_b[key]:>+12.4f}")

    d = per_match_logloss(pooled_c) - per_match_logloss(pooled_b)
    point, lo, hi = bootstrap_diff_ci(d, n_boot=n_boot, seed=seed)
    print(f"\nlog-loss difference (combined - baseline), paired bootstrap [{n_boot} resamples]:")
    print(f"  point {point:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
    if lo > 0:
        verdict = "COMBINED IS WORSE on held-out data (CI entirely > 0)."
    elif hi < 0:
        verdict = "COMBINED IS BETTER on held-out data (CI entirely < 0)."
    else:
        verdict = "WITHIN NOISE — the CI straddles 0; no real held-out gain."
    print(f"  -> {verdict}")
    print(f"\n  CAVEAT: this is the 3rd config tested on the same 281-match held-out set.")
    print("  A clean promotion would need a single JOINT pre-2016 search, then one test.")
    print(f"{'='*72}")


if __name__ == "__main__":
    # Run from repo root:  python -m src.models.tune_combined
    main()
