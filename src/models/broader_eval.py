"""
Broader covered-block evaluation: the project's headline model-vs-bookmaker
comparison, on EVERY international with a de-vigged bookmaker price — not just
the tournament finals.

WHY A SEPARATE HARNESS FROM walk_forward.py.
`walk_forward.py` backtests only the major-tournament FINALS (the rows with a
resolved squad `edition`). That is the right unit for "how well do we predict
the tournaments we claim", but it is a small sample (a few hundred matches) and
it excludes qualifiers / Nations League, where most of the bookmaker odds we
collected actually live. This module instead scores the whole odds-covered set
(~2,000 internationals across Euro 2020/2024 + World Cup 2022/2026 and their
qualifiers), which is what gives the model-vs-market claim real statistical
power — and, with a paired bootstrap CI, an honest read on whether any gap is
signal or noise. (Fulfils ToDo item 1's "broader-covered evaluation".)

LEAKAGE DISCIPLINE — annual walk-forward.
The bookmaker price is honest by construction (it existed pre-match). To make
the MODEL side equally honest, we never score a match with a model that has
seen its future: for each calendar year Y that has covered matches, we fit a
fresh model on all matches strictly before Y (early-stopping on the trailing
year), then predict the covered matches IN year Y. This reuses walk_forward's
`_fit_before` verbatim, so the fit/split logic can't drift from the finals
backtest.

WHAT IT REPORTS.
Model / bookmaker / form-favourite / base-rate on accuracy, log-loss, Brier —
all on the identical covered set — plus paired-bootstrap 95% CIs on the
per-match log-loss difference (model - bookmaker) and (model - base-rate). A CI
that straddles 0 means "within noise"; a positive (model - bookmaker) mean
means the market is sharper (expected — bookmakers are the ~52-58% skill
ceiling), the honest result the whole project is built to quantify.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from src.models.build_matrix import FEATURE_COLUMNS, LABEL_COLUMN
from src.models.train_wdl import load_matrix, build_model, LABEL_TO_INT
from src.models.walk_forward import _fit_before, VAL_DAYS, CLASSES, CLASS_NAMES
from src.models.evaluate_wdl import form_favourite_pred, multiclass_brier
from src.models.tune_common import per_match_logloss, bootstrap_diff_ci, pooled_metrics

BOOK_COLS = ["book_ph", "book_pd", "book_pa"]

# --- Ablation feature GROUPS -------------------------------------------------
# The per-feature-set ablation isolates how much each engineered group actually
# buys us on the broad covered block (ToDo item 2). The groups are listed by
# name (never sliced by index) and asserted to partition FEATURE_COLUMNS below,
# so if a new feature is ever added to build_matrix WITHOUT being placed in a
# group here, this module fails loudly rather than silently dropping it from the
# ablation — the same never-drift discipline the leakage guards use.
BASE_FEATURES = [
    "neutral", "importance",
    "home_form", "away_form",
    "home_gs", "home_gc", "away_gs", "away_gc",
    "h2h",
]
SQUAD_FEATURES = [
    "home_mean_age", "away_mean_age",
    "home_squad_size", "away_squad_size",
    "home_club_hhi", "away_club_hhi",
    "home_same_club_pair_ratio", "away_same_club_pair_ratio",
    "home_largest_club_bloc", "away_largest_club_bloc",
]
RATING_FEATURES = ["rating_gap"]

assert set(BASE_FEATURES + SQUAD_FEATURES + RATING_FEATURES) == set(FEATURE_COLUMNS), (
    "ablation feature groups don't partition FEATURE_COLUMNS — a feature was added "
    "to build_matrix but not assigned to a group in broader_eval.py"
)

# The four feature sets compared in the ablation. "full" reuses FEATURE_COLUMNS
# verbatim (same identity AND order) so its numbers match the headline report()
# and walk_forward's official model exactly, rather than an order-shuffled twin.
FEATURE_SETS = {
    "base":         BASE_FEATURES,
    "base+squad":   BASE_FEATURES + SQUAD_FEATURES,
    "base+rating":  BASE_FEATURES + RATING_FEATURES,
    "full":         FEATURE_COLUMNS,
}


def covered(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that have a full de-vigged bookmaker price (all three probs present).
    log-loss can't take NaN, so the whole comparison lives on this subset."""
    return df[df[BOOK_COLS].notna().all(axis=1)]


def annual_walk_forward(df: pd.DataFrame,
                        feature_columns: list[str] = FEATURE_COLUMNS,
                        val_days: int = VAL_DAYS,
                        model_fn=build_model):
    """
    Fit a fresh model before each calendar year and predict that year's
    odds-covered matches. Returns (per_year_rows, pooled) where pooled holds the
    concatenated truth / model-probs / bookmaker-probs / baselines across every
    covered match — all aligned row-for-row so the caller can pair them.
    """
    cov = covered(df)
    years = sorted(cov["date"].dt.year.unique())

    rows = []
    pool = {"y": [], "proba": [], "book": [], "form_fav": [], "base": []}

    for Y in years:
        t_start = pd.Timestamp(f"{Y}-01-01")
        T = cov[cov["date"].dt.year == Y]
        if T.empty:
            continue
        model, n_tr, n_val = _fit_before(df, t_start, feature_columns, val_days,
                                         model_fn=model_fn)

        proba = model.predict_proba(T[feature_columns])
        y = T[LABEL_COLUMN].to_numpy()
        y_int = T[LABEL_COLUMN].map(LABEL_TO_INT).to_numpy()
        book = T[BOOK_COLS].to_numpy(dtype=float)
        form_fav = form_favourite_pred(T)

        # No-skill vector = class frequencies of this year's own training block.
        train_block = df[df["date"] < t_start]
        base_vec = (train_block[LABEL_COLUMN].value_counts(normalize=True)
                    .reindex(CLASS_NAMES).to_numpy())
        base_proba = np.tile(base_vec, (len(T), 1))

        pred = np.array([CLASS_NAMES[i] for i in proba.argmax(axis=1)])
        predbk = np.array([CLASS_NAMES[i] for i in book.argmax(axis=1)])
        rows.append({
            "year": Y, "n": len(T), "n_train": n_tr,
            "acc_model": accuracy_score(y, pred),
            "acc_book": accuracy_score(y, predbk),
            "ll_model": log_loss(y_int, proba, labels=CLASSES),
            "ll_book": log_loss(y_int, book, labels=CLASSES),
        })
        pool["y"].append(y)
        pool["proba"].append(proba)
        pool["book"].append(book)
        pool["form_fav"].append(form_fav)
        pool["base"].append(base_proba)

    pooled = {
        "y": np.concatenate(pool["y"]),
        "proba": np.vstack(pool["proba"]),
        "book": np.vstack(pool["book"]),
        "form_fav": np.concatenate(pool["form_fav"]),
        "base": np.vstack(pool["base"]),
    }
    return rows, pooled


def summarize(rows, pooled) -> dict:
    """Reduce the broad-block backtest arrays to the numbers a reader (CLI or API)
    needs, as a plain dict — ONE source of truth so `report()` (below) and the
    webapp's /api/broad-eval endpoint can never drift from each other. Mirrors the
    same discipline as walk_forward.summarize().

    All values are full floats (no rounding); formatting is the caller's job.
    Structure:
      per_year : the per-calendar-year row dicts (model/book accuracy + log-loss),
                 exactly as annual_walk_forward produced them, chronological.
      pooled   : model vs bookmaker vs form-favourite vs base-rate across every
                 odds-covered match (accuracy / log-loss / brier).
      ci_book  : paired-bootstrap 95% CI on the per-match (model - bookmaker)
                 log-loss difference. mean>0 => the market is sharper (expected).
      ci_base  : same for (model - base-rate). mean<0 => the model beats no-skill
                 — the bar it MUST clear.
    """
    y = pooled["y"]
    y_int = pd.Series(y).map(LABEL_TO_INT).to_numpy()
    proba, book, base = pooled["proba"], pooled["book"], pooled["base"]

    pred = np.array([CLASS_NAMES[i] for i in proba.argmax(axis=1)])
    predbk = np.array([CLASS_NAMES[i] for i in book.argmax(axis=1)])

    # Paired bootstrap on the per-match log-loss difference. Positive mean =>
    # model's log-loss is higher => the reference is sharper than the model.
    # bootstrap_diff_ci is seeded (seed=0), so these are deterministic.
    d_book = per_match_logloss({"y": y, "proba": proba}) - \
             per_match_logloss({"y": y, "proba": book})
    d_base = per_match_logloss({"y": y, "proba": proba}) - \
             per_match_logloss({"y": y, "proba": base})
    pb, lob, hib = bootstrap_diff_ci(d_book)
    pa, loa, hia = bootstrap_diff_ci(d_base)

    return {
        "per_year": rows,
        "pooled": {
            "n": int(len(y)),
            "acc_model": float(accuracy_score(y, pred)),
            "acc_book": float(accuracy_score(y, predbk)),
            "acc_form_fav": float(accuracy_score(y, pooled["form_fav"])),
            "ll_model": float(log_loss(y_int, proba, labels=CLASSES)),
            "ll_book": float(log_loss(y_int, book, labels=CLASSES)),
            "ll_base": float(log_loss(y_int, base, labels=CLASSES)),
            "brier_model": float(multiclass_brier(y_int, proba)),
            "brier_book": float(multiclass_brier(y_int, book)),
            "brier_base": float(multiclass_brier(y_int, base)),
        },
        "ci_book": {"mean": float(pb), "lo": float(lob), "hi": float(hib)},
        "ci_base": {"mean": float(pa), "lo": float(loa), "hi": float(hia)},
    }


def report(rows, pooled):
    """Print the per-year table, the pooled aggregate vs bookmaker/baselines, and
    the two paired-bootstrap CIs. Every number comes from `summarize()` so the
    printed CLI report and the webapp's /api/broad-eval are identical by
    construction."""
    s = summarize(rows, pooled)
    p = s["pooled"]

    print(f"\n{'='*72}\nBROADER MODEL vs BOOKMAKER  (all odds-covered internationals)\n{'='*72}")
    print(f"{'year':<8}{'n':>6}{'train':>8}{'acc(mdl)':>10}{'acc(bk)':>10}"
          f"{'ll(mdl)':>10}{'ll(bk)':>10}")
    for r in s["per_year"]:
        print(f"{r['year']:<8}{r['n']:>6}{r['n_train']:>8}"
              f"{r['acc_model']:>10.3f}{r['acc_book']:>10.3f}"
              f"{r['ll_model']:>10.3f}{r['ll_book']:>10.3f}")

    print(f"\n{'-'*72}\nPOOLED  (n={p['n']})\n{'-'*72}")
    print(f"{'metric':<12}{'model':>12}{'bookmaker':>12}{'form-fav':>12}{'base-rate':>12}")
    print(f"{'accuracy':<12}{p['acc_model']:>12.3f}{p['acc_book']:>12.3f}"
          f"{p['acc_form_fav']:>12.3f}{'-':>12}")
    print(f"{'log-loss':<12}{p['ll_model']:>12.3f}"
          f"{p['ll_book']:>12.3f}{'-':>12}"
          f"{p['ll_base']:>12.3f}")
    print(f"{'brier':<12}{p['brier_model']:>12.3f}"
          f"{p['brier_book']:>12.3f}{'-':>12}"
          f"{p['brier_base']:>12.3f}")

    cb, ca = s["ci_book"], s["ci_base"]
    print(f"\nlog-loss diff (model - bookmaker):  {cb['mean']:+.4f}  95% CI [{cb['lo']:+.4f}, {cb['hi']:+.4f}]"
          f"\n   (>0 => bookmaker sharper; the expected, honest result)")
    print(f"log-loss diff (model - base-rate):  {ca['mean']:+.4f}  95% CI [{ca['lo']:+.4f}, {ca['hi']:+.4f}]"
          f"\n   (<0 => model beats no-skill; the bar the model MUST clear)")


def _paired_ci(pooled_added: dict, pooled_without: dict):
    """Paired-bootstrap CI on the per-match log-loss difference between two
    feature sets scored on the IDENTICAL covered block.

    `pooled_added` includes the feature group under test; `pooled_without` omits
    it. The mean of (added - without) is negative when the group LOWERS log-loss,
    i.e. improves the model. Because both runs scored the same covered rows in the
    same order (feature columns don't change which rows are scored), the per-match
    contributions pair 1:1 — asserted here so a future refactor that breaks the
    alignment can't silently produce a meaningless CI."""
    assert np.array_equal(pooled_added["y"], pooled_without["y"]), (
        "paired ablation CI requires row-aligned pooled results — the two feature "
        "sets scored different covered rows"
    )
    d = per_match_logloss(pooled_added) - per_match_logloss(pooled_without)
    return bootstrap_diff_ci(d)


def ablation_report(results: dict):
    """Per-feature-set accuracy/log-loss/brier table + paired-CI contrasts that
    isolate each engineered group's marginal contribution on the broad covered
    block. `results` maps feature-set name -> (rows, pooled) from
    annual_walk_forward, all on the same covered set."""
    print(f"\n{'='*72}\nFEATURE-SET ABLATION  (broad covered block, annual walk-forward)\n{'='*72}")
    print(f"{'feature set':<16}{'n':>7}{'acc':>10}{'log-loss':>12}{'brier':>10}")
    for name in FEATURE_SETS:
        _, pooled = results[name]
        m = pooled_metrics(pooled)
        print(f"{name:<16}{len(pooled['y']):>7}{m['acc']:>10.3f}"
              f"{m['log_loss']:>12.3f}{m['brier']:>10.3f}")

    # Each contrast = (set WITH the group) minus (same set WITHOUT it), so the
    # group's marginal value is measured both alone (on base) and on top of the
    # other group — a within-noise result on one but not the other is itself a
    # finding worth reporting.
    contrasts = [
        ("rating_gap | no squad",     "base+rating", "base"),
        ("rating_gap | squad present", "full",        "base+squad"),
        ("squad | no rating",         "base+squad",  "base"),
        ("squad | rating present",    "full",        "base+rating"),
    ]
    print(f"\n{'-'*72}\nMARGINAL LIFT  (paired per-match log-loss diff: with - without)\n{'-'*72}")
    print("(<0 => adding the feature group lowers log-loss = improves the model;\n"
          " a CI straddling 0 means the lift is within noise on this block)\n")
    for label, added, without in contrasts:
        _, p_add = results[added]
        _, p_wo = results[without]
        point, lo, hi = _paired_ci(p_add, p_wo)
        verdict = "within noise" if lo <= 0 <= hi else ("improves" if hi < 0 else "hurts")
        print(f"{label:<26}{point:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  ({verdict})")


def main():
    df = load_matrix()

    # One walk-forward per feature set, all on the same covered block. Reuse the
    # full-feature run for the headline model-vs-market report so it isn't fit
    # twice.
    results = {name: annual_walk_forward(df, feature_columns=cols)
               for name, cols in FEATURE_SETS.items()}

    rows, pooled = results["full"]
    report(rows, pooled)
    ablation_report(results)


if __name__ == "__main__":
    # Run from repo root:  python -m src.models.broader_eval
    main()
