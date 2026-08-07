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
from src.models.tune_common import per_match_logloss, bootstrap_diff_ci

BOOK_COLS = ["book_ph", "book_pd", "book_pa"]


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


def report(rows, pooled):
    y = pooled["y"]
    y_int = pd.Series(y).map(LABEL_TO_INT).to_numpy()
    proba, book, base = pooled["proba"], pooled["book"], pooled["base"]

    print(f"\n{'='*72}\nBROADER MODEL vs BOOKMAKER  (all odds-covered internationals)\n{'='*72}")
    print(f"{'year':<8}{'n':>6}{'train':>8}{'acc(mdl)':>10}{'acc(bk)':>10}"
          f"{'ll(mdl)':>10}{'ll(bk)':>10}")
    for r in rows:
        print(f"{r['year']:<8}{r['n']:>6}{r['n_train']:>8}"
              f"{r['acc_model']:>10.3f}{r['acc_book']:>10.3f}"
              f"{r['ll_model']:>10.3f}{r['ll_book']:>10.3f}")

    pred = np.array([CLASS_NAMES[i] for i in proba.argmax(axis=1)])
    predbk = np.array([CLASS_NAMES[i] for i in book.argmax(axis=1)])
    print(f"\n{'-'*72}\nPOOLED  (n={len(y)})\n{'-'*72}")
    print(f"{'metric':<12}{'model':>12}{'bookmaker':>12}{'form-fav':>12}{'base-rate':>12}")
    print(f"{'accuracy':<12}{accuracy_score(y, pred):>12.3f}{accuracy_score(y, predbk):>12.3f}"
          f"{accuracy_score(y, pooled['form_fav']):>12.3f}{'-':>12}")
    print(f"{'log-loss':<12}{log_loss(y_int, proba, labels=CLASSES):>12.3f}"
          f"{log_loss(y_int, book, labels=CLASSES):>12.3f}{'-':>12}"
          f"{log_loss(y_int, base, labels=CLASSES):>12.3f}")
    print(f"{'brier':<12}{multiclass_brier(y_int, proba):>12.3f}"
          f"{multiclass_brier(y_int, book):>12.3f}{'-':>12}"
          f"{multiclass_brier(y_int, base):>12.3f}")

    # Paired bootstrap on the per-match log-loss difference. Positive mean =>
    # model's log-loss is higher => the reference is sharper than the model.
    d_book = per_match_logloss({"y": y, "proba": proba}) - \
             per_match_logloss({"y": y, "proba": book})
    d_base = per_match_logloss({"y": y, "proba": proba}) - \
             per_match_logloss({"y": y, "proba": base})
    pb, lob, hib = bootstrap_diff_ci(d_book)
    pa, loa, hia = bootstrap_diff_ci(d_base)
    print(f"\nlog-loss diff (model - bookmaker):  {pb:+.4f}  95% CI [{lob:+.4f}, {hib:+.4f}]"
          f"\n   (>0 => bookmaker sharper; the expected, honest result)")
    print(f"log-loss diff (model - base-rate):  {pa:+.4f}  95% CI [{loa:+.4f}, {hia:+.4f}]"
          f"\n   (<0 => model beats no-skill; the bar the model MUST clear)")


def main():
    df = load_matrix()
    rows, pooled = annual_walk_forward(df)
    report(rows, pooled)


if __name__ == "__main__":
    # Run from repo root:  python -m src.models.broader_eval
    main()
