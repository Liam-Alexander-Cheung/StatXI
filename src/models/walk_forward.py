"""
Walk-forward backtest: the honest, never-stale evaluation of the WDL model.

`evaluate_wdl.py` scores ONE model (trained on <2014) across every 2016+ match.
That model is deliberately simple but, by construction, stale: by Euro 2024 it has
never seen a match after 2013. Walk-forward removes that staleness. For each
backtest tournament T it trains a FRESH model on every match strictly before T
kicks off, then predicts T. No tournament is ever scored by a model that has seen
data from its own future — the same leakage discipline as the temporal split, but
applied once per tournament so each prediction uses the maximum honest history.

Two reasons this is the right harness for the project going forward:
  1. It is the fairest possible backtest of what we actually claim — tournament
     prediction — because each tournament is forecast with all prior knowledge.
  2. It is the hard PREREQUISITE for the squad-rating features (TRACK 2). Under
     any single train/test cutoff every rating-covered tournament falls in the
     test block, so the model gets ZERO training rows for those columns. Retraining
     before each tournament is the only way those features ever appear in training.

Because it just consumes FEATURE_COLUMNS, adding new feature columns (ratings,
prodigy z-scores) needs no change here — they flow in automatically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from src.data_pipeline import recency_weight
from src.models.build_matrix import FEATURE_COLUMNS, LABEL_COLUMN
from src.models.train_wdl import load_matrix, build_model, LABEL_TO_INT
# Reuse the exact baselines evaluate_wdl uses, so numbers are comparable across
# the two scripts rather than re-implemented (and possibly drifting).
from src.models.evaluate_wdl import form_favourite_pred, multiclass_brier

CLASSES = [0, 1, 2]                 # H, D, A integer encoding (from train_wdl)
CLASS_NAMES = ["H", "D", "A"]

# Only tournaments from this year on are backtested. It matches the project's
# held-out boundary: earlier editions have little pre-history and no squad-feature
# training benefit, and are already the training bulk. Derived from the data (not
# a hard-coded list) so a future squad scrape of a new tournament is picked up.
BACKTEST_START_YEAR = 2016

# Days before a tournament reserved as the early-stopping validation slice. It is
# used ONLY to pick the tree count (never scored), and it is temporal — the year
# immediately before T — so it never contains data from T itself.
VAL_DAYS = 365


def backtest_editions(df: pd.DataFrame) -> list[tuple[str, pd.Timestamp]]:
    """
    (edition_label, start_date) for every major-tournament edition to backtest,
    chronological. `edition` was written into the matrix by build_matrix (the
    resolved label like "World Cup 2018"); only WC/Euro rows have it populated.
    A tournament's "start" is the date of its earliest match in the data.
    """
    ed = df.dropna(subset=["edition"])
    starts = ed.groupby("edition")["date"].min()
    out = [(name, start) for name, start in starts.items()
           if start.year >= BACKTEST_START_YEAR]
    return sorted(out, key=lambda x: x[1])   # chronological by start date


def _sample_weights(df: pd.DataFrame, ref_date: pd.Timestamp) -> np.ndarray:
    """Per-match training weight = recency (to this tournament) x importance —
    the same weighting train_wdl uses, but referenced to each tournament's start."""
    return df.apply(
        lambda r: recency_weight(r["date"], ref_date, half_life_days=3650) * r["importance"],
        axis=1,
    ).to_numpy()


def _fit_before(df: pd.DataFrame, t_start: pd.Timestamp, feature_columns: list[str],
                val_days: int):
    """Fit a fresh model on all matches before `t_start`, early-stopping on the
    trailing `val_days` window. Returns (model, n_train, n_val)."""
    val_start = t_start - pd.Timedelta(days=val_days)
    train = df[df["date"] < val_start]
    val = df[(df["date"] >= val_start) & (df["date"] < t_start)]

    model = build_model()
    model.fit(
        train[feature_columns], train[LABEL_COLUMN].map(LABEL_TO_INT),
        sample_weight=_sample_weights(train, t_start),
        eval_set=[(val[feature_columns], val[LABEL_COLUMN].map(LABEL_TO_INT))],
        verbose=False,
    )
    return model, len(train), len(val)


def walk_forward(df: pd.DataFrame, feature_columns: list[str] = FEATURE_COLUMNS,
                 val_days: int = VAL_DAYS):
    """
    Run the backtest. Returns (per_tournament, pooled) where per_tournament is a
    list of dicts (one per edition) and pooled holds the concatenated truth /
    model-probabilities / baseline predictions across every backtested match, so
    the caller can compute an exact pooled score.
    """
    rows = []
    pool = {"y": [], "proba": [], "form_fav": [], "base": []}

    for edition, t_start in backtest_editions(df):
        T = df[df["edition"] == edition]
        model, n_tr, n_val = _fit_before(df, t_start, feature_columns, val_days)

        proba = model.predict_proba(T[feature_columns])
        pred = np.array([CLASS_NAMES[i] for i in proba.argmax(axis=1)])
        y = T[LABEL_COLUMN].to_numpy()
        y_int = T[LABEL_COLUMN].map(LABEL_TO_INT).to_numpy()
        form_fav = form_favourite_pred(T)

        # No-skill probability vector for THIS tournament = class frequencies of
        # its own training block (never peeks at the tournament being scored).
        train_block = df[df["date"] < t_start]
        base_vec = (train_block[LABEL_COLUMN].value_counts(normalize=True)
                    .reindex(CLASS_NAMES).to_numpy())
        base_proba = np.tile(base_vec, (len(T), 1))

        rows.append({
            "edition": edition,
            "n": len(T),
            "n_train": n_tr,
            "trees": model.best_iteration,
            "acc": accuracy_score(y, pred),
            "log_loss": log_loss(y_int, proba, labels=CLASSES),
            "acc_form_fav": accuracy_score(y, form_fav),
        })
        pool["y"].append(y)
        pool["proba"].append(proba)
        pool["form_fav"].append(form_fav)
        pool["base"].append(base_proba)

    pooled = {
        "y": np.concatenate(pool["y"]),
        "proba": np.vstack(pool["proba"]),
        "form_fav": np.concatenate(pool["form_fav"]),
        "base": np.vstack(pool["base"]),
    }
    return rows, pooled


def report(rows, pooled):
    """Print the per-tournament table and the pooled aggregate vs baselines."""
    print(f"\n{'='*76}\nWALK-FORWARD BACKTEST  (fresh model trained before each tournament)\n{'='*76}")
    print(f"{'tournament':<16}{'n':>4}{'train':>8}{'trees':>7}{'acc':>8}{'form-fav':>10}{'log-loss':>10}")
    for r in rows:
        print(f"{r['edition']:<16}{r['n']:>4}{r['n_train']:>8}{r['trees']:>7}"
              f"{r['acc']:>8.3f}{r['acc_form_fav']:>10.3f}{r['log_loss']:>10.3f}")

    y, proba = pooled["y"], pooled["proba"]
    y_int = pd.Series(y).map(LABEL_TO_INT).to_numpy()
    pred = np.array([CLASS_NAMES[i] for i in proba.argmax(axis=1)])

    acc = accuracy_score(y, pred)
    acc_ff = accuracy_score(y, pooled["form_fav"])
    ll = log_loss(y_int, proba, labels=CLASSES)
    ll_base = log_loss(y_int, pooled["base"], labels=CLASSES)
    brier = multiclass_brier(y_int, proba)
    brier_base = multiclass_brier(y_int, pooled["base"])

    print(f"\n{'-'*76}\nPOOLED across all backtested tournaments   (n={len(y)})\n{'-'*76}")
    print(f"{'metric':<16}{'model':>12}{'form-fav':>12}{'base-rate':>12}")
    print(f"{'accuracy':<16}{acc:>12.3f}{acc_ff:>12.3f}{'-':>12}")
    print(f"{'log-loss':<16}{ll:>12.3f}{'-':>12}{ll_base:>12.3f}")
    print(f"{'brier':<16}{brier:>12.3f}{'-':>12}{brier_base:>12.3f}")
    print(f"\nmodel vs form-favourite:  accuracy {acc-acc_ff:+.3f}")
    print(f"model vs base-rate:       log-loss {ll-ll_base:+.3f}  (negative = better than no-skill)")


def main():
    df = load_matrix()
    rows, pooled = walk_forward(df)
    report(rows, pooled)


if __name__ == "__main__":
    # Run from repo root:  python -m src.models.walk_forward
    main()
