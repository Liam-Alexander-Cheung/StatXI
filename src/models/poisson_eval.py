"""
Poisson vs XGBoost vs bookmaker — the apples-to-apples scoreline-model benchmark.

Phase 4 of the Poisson build (see src/models/poisson.py). It reuses the existing
walk-forward harnesses VERBATIM for both models, differing only in what gets
handed to the model:
  - XGBoost: the 21 engineered FEATURE_COLUMNS (the default).
  - Poisson: POISSON_FEATURE_COLUMNS (match identity + goals), via
    model_fn=PoissonWDL.
Because both runs use the same leakage-safe walk-forward on the same matches, the
pooled truth / bookmaker / base-rate vectors are identical row-for-row — so the
two models can be paired match-by-match against each other AND against the market,
with a bootstrap CI on every difference.

Two views, both reported:
  1. BROAD covered block (broader_eval.annual_walk_forward): every odds-covered
     international (~2,200; qualifiers + Nations League + tournaments). Statistical
     power.
  2. TOURNAMENT FINALS only (walk_forward): the WC/Euro editions we ultimately
     claim to predict, on their odds-covered subset (~150). Smaller, but it is the
     unit the project is really about — does the Poisson edge hold there too?
"""

from __future__ import annotations

import numpy as np

from src.models.train_wdl import load_matrix
from src.models.walk_forward import walk_forward
from src.models.broader_eval import annual_walk_forward
from src.models.poisson import PoissonWDL, POISSON_FEATURE_COLUMNS
from src.models.tune_common import per_match_logloss, bootstrap_diff_ci, pooled_metrics


def _compare(y, proba_pois, proba_xgb, book, base, title):
    """Print the pooled Poisson / XGBoost / bookmaker / base-rate table plus paired
    bootstrap CIs. All arrays are row-aligned to the same matches `y`."""
    def metrics(proba):
        return pooled_metrics({"y": y, "proba": proba})

    rows = [
        ("Poisson (DC)", metrics(proba_pois)),
        ("XGBoost WDL", metrics(proba_xgb)),
        ("bookmaker", metrics(book)),
        ("base-rate", metrics(base)),
    ]
    print(f"\n{'='*72}\n{title}  (n={len(y)})\n{'='*72}")
    print(f"{'model':<16}{'accuracy':>12}{'log-loss':>12}{'brier':>12}")
    for name, m in rows:
        print(f"{name:<16}{m['acc']:>12.3f}{m['log_loss']:>12.3f}{m['brier']:>12.3f}")

    # Per-match log-loss vectors for the paired bootstrap.
    ll_pois = per_match_logloss({"y": y, "proba": proba_pois})
    ll_xgb = per_match_logloss({"y": y, "proba": proba_xgb})
    ll_book = per_match_logloss({"y": y, "proba": book})
    ll_base = per_match_logloss({"y": y, "proba": base})

    def line(label, d, note):
        point, lo, hi = bootstrap_diff_ci(d)
        print(f"{label:<24}{point:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  {note}")

    print(f"\n{'-'*72}\nPAIRED log-loss differences (negative => first model sharper)\n{'-'*72}")
    line("Poisson - XGBoost", ll_pois - ll_xgb, "(<0 => Poisson sharper than the ML model)")
    line("Poisson - bookmaker", ll_pois - ll_book, "(>0 => market sharper, expected)")
    line("Poisson - base-rate", ll_pois - ll_base, "(<0 => beats no-skill; MUST clear)")
    line("XGBoost - bookmaker", ll_xgb - ll_book, "(>0 => market sharper, expected)")


def main():
    df = load_matrix()

    # --- View 1: broad odds-covered block (annual walk-forward). ---------------
    print("Broad block: annual walk-forward for XGBoost then Poisson...", flush=True)
    _, pooled_xgb = annual_walk_forward(df)
    _, pooled_pois = annual_walk_forward(
        df, feature_columns=POISSON_FEATURE_COLUMNS, model_fn=PoissonWDL)
    assert np.array_equal(pooled_xgb["y"], pooled_pois["y"]), "broad: rows not aligned"
    _compare(pooled_pois["y"], pooled_pois["proba"], pooled_xgb["proba"],
             pooled_pois["book"], pooled_pois["base"],
             "BROAD BLOCK: POISSON vs XGBOOST vs BOOKMAKER")

    # --- View 2: tournament finals only (per-tournament walk-forward), on the
    #     odds-covered subset (Euro 2016 / WC 2018 have no odds -> excluded). -----
    print("\nFinals: per-tournament walk-forward for XGBoost then Poisson...", flush=True)
    _, pooled_xgb_f = walk_forward(df)
    _, pooled_pois_f = walk_forward(
        df, feature_columns=POISSON_FEATURE_COLUMNS, model_fn=PoissonWDL)
    assert np.array_equal(pooled_xgb_f["y"], pooled_pois_f["y"]), "finals: rows not aligned"

    # Mask to matches that actually have a de-vigged bookmaker price (log-loss
    # can't take NaN); the model is re-scored on that SAME subset, apples-to-apples.
    book_f = pooled_pois_f["book"]
    covered = ~np.isnan(book_f).any(axis=1)
    m = covered
    _compare(pooled_pois_f["y"][m], pooled_pois_f["proba"][m], pooled_xgb_f["proba"][m],
             book_f[m], pooled_pois_f["base"][m],
             "TOURNAMENT FINALS (odds-covered): POISSON vs XGBOOST vs BOOKMAKER")


if __name__ == "__main__":
    # Run from repo root:  python -m src.models.poisson_eval
    main()
