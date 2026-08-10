"""
Phase 5: does a rating_gap covariate earn its place on the Poisson rate?

The open design question (deferred from the plan, TESTED not assumed): the Poisson
attack/defence strengths already OPPONENT-ADJUST team quality from goals, and
rating_gap (home_pool_rating − away_pool_rating) is FIFA-rating team quality — the
two overlap heavily. So does adding gamma*rating_gap to the log-rate actually
sharpen the model, or is it redundant with what attack/defence already capture?

This runs plain Poisson vs Poisson+rating through the SAME leakage-safe annual
walk-forward on the SAME odds-covered block, then reports a paired-bootstrap CI on
the per-match log-loss difference — the identical honest pattern used for the WDL
rating_gap ablation in broader_eval.ablation_report. rating_gap is NaN for most
matches (only ~1k rated), so on unrated rows the two models predict identically and
the difference is exactly 0 there; the CI reflects the rated subset, which is the
only place the covariate can act.
"""

from __future__ import annotations

import numpy as np

from src.models.train_wdl import load_matrix
from src.models.broader_eval import annual_walk_forward
from src.models.poisson import (
    PoissonWDL,
    POISSON_FEATURE_COLUMNS,
    POISSON_RATING_COLUMNS,
)
from src.models.tune_common import per_match_logloss, bootstrap_diff_ci, pooled_metrics


def main():
    df = load_matrix()

    print("plain Poisson (attack/defence only)...", flush=True)
    _, pooled_plain = annual_walk_forward(
        df, feature_columns=POISSON_FEATURE_COLUMNS, model_fn=PoissonWDL)

    print("Poisson + rating_gap covariate...", flush=True)
    _, pooled_rating = annual_walk_forward(
        df, feature_columns=POISSON_RATING_COLUMNS,
        model_fn=lambda: PoissonWDL(use_rating=True))

    assert np.array_equal(pooled_plain["y"], pooled_rating["y"]), "rows not aligned"
    y = pooled_plain["y"]

    mp = pooled_metrics(pooled_plain)
    mr = pooled_metrics(pooled_rating)

    # Rows where the two models actually give a different prediction = rows that
    # carry a (non-NaN) rating_gap. Everywhere else the covariate contributes 0.
    diff_rows = int(np.any(~np.isclose(pooled_plain["proba"], pooled_rating["proba"]),
                           axis=1).sum())

    print(f"\n{'='*72}\nPHASE 5 ABLATION: Poisson  vs  Poisson + rating_gap  (covered block)\n{'='*72}")
    print(f"covered matches n={len(y)}   of which rating_gap-affected: {diff_rows}")
    print(f"\n{'model':<26}{'accuracy':>12}{'log-loss':>12}{'brier':>12}")
    print(f"{'Poisson (plain)':<26}{mp['acc']:>12.3f}{mp['log_loss']:>12.3f}{mp['brier']:>12.3f}")
    print(f"{'Poisson + rating_gap':<26}{mr['acc']:>12.3f}{mr['log_loss']:>12.3f}{mr['brier']:>12.3f}")

    # Paired bootstrap on per-match log-loss: (with rating) − (plain).
    # Negative => the covariate LOWERS log-loss = improves the model.
    d = per_match_logloss(pooled_rating) - per_match_logloss(pooled_plain)
    point, lo, hi = bootstrap_diff_ci(d)
    verdict = "within noise" if lo <= 0 <= hi else ("improves" if hi < 0 else "hurts")
    print(f"\n{'-'*72}")
    print(f"rating_gap marginal log-loss lift (with − without):"
          f"  {point:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  (<0 => covariate improves the Poisson model)   verdict: {verdict}")
    print(f"{'-'*72}")


if __name__ == "__main__":
    # Run from repo root:  python -m src.models.poisson_rating_ablation
    main()
