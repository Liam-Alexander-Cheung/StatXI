"""
Step 2 of the XGBoost Win/Draw/Loss model: train the classifier on the matrix
built in step 1 (src/models/build_matrix.py).

Two things here carry all the methodological weight:

1. TEMPORAL split, never a random one. The model is trained only on the past
   and tested only on the future. A random k-fold split would let the model
   learn from matches that happened *after* the ones it's scored on — the exact
   leakage that manufactures a fake 90% accuracy. We split by date instead:
       train  < 2014-01-01
       val    2014-01-01 .. 2016-01-01   (used only for early stopping)
       test   2016-01-01 ..              (held out entirely; scored in step 3)
   The 2016 boundary is chosen so the test block contains the tournaments we
   actually want to backtest: Euro 2016, WC 2018, Euro 2020 (played 2021),
   WC 2022, Euro 2024.

   Honesty caveat (documented in methodology.md): a model trained only on
   pre-2014 data is stale by Euro 2024. The rigorous fix — retrain freshly
   before each tournament ("walk-forward") — is a v2 upgrade; v1 keeps one
   clean cutoff so the pipeline is simple and obviously leak-free.

2. Probabilities, not hard labels. objective="multi:softprob" makes the model
   output a 3-number probability vector per match (P(home), P(draw), P(away)).
   That vector is how we "expect the unexpected": a well-calibrated 22% on an
   upset is the deliverable, not a lucky top-1 guess.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.data_pipeline import recency_weight
from src.models.build_matrix import MATRIX_PATH, FEATURE_COLUMNS, LABEL_COLUMN

# --- Split boundaries (dates are inclusive-left, exclusive-right blocks) -----
VAL_START = pd.Timestamp("2014-01-01")
TEST_START = pd.Timestamp("2016-01-01")

# XGBoost needs integer class labels. Fix an explicit, documented mapping so
# step 3 can translate predictions back to H/D/A unambiguously.
LABEL_TO_INT = {"H": 0, "D": 1, "A": 2}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}

# Half-life for the recency part of the per-match training weight. 10 years is
# deliberately gentle — we don't want to crush 1990s matches to near-nothing
# (recency_weight also floors at 0.05), just to lean the model toward recent,
# important games.
SAMPLE_WEIGHT_HALF_LIFE_DAYS = 3650

MODEL_PATH = Path(__file__).resolve().parent / "wdl_xgb.json"


def load_matrix() -> pd.DataFrame:
    """Load the cached training matrix and parse dates back to timestamps."""
    df = pd.read_csv(MATRIX_PATH, parse_dates=["date"])
    return df


def temporal_split(df: pd.DataFrame):
    """Split the matrix into train / val / test blocks by date (see module docstring)."""
    train = df[df["date"] < VAL_START]
    val = df[(df["date"] >= VAL_START) & (df["date"] < TEST_START)]
    test = df[df["date"] >= TEST_START]
    return train, val, test


def sample_weights(df: pd.DataFrame) -> np.ndarray:
    """
    Per-training-match weight = recency (relative to the test boundary) x the
    tournament importance already stored in the matrix. This is HOW MUCH each
    training example counts — distinct from the recency/importance weighting
    that lives *inside* the features themselves.
    """
    return df.apply(
        lambda r: recency_weight(r["date"], TEST_START, SAMPLE_WEIGHT_HALF_LIFE_DAYS)
        * r["importance"],
        axis=1,
    ).to_numpy()


def build_model() -> XGBClassifier:
    """
    XGBClassifier with parameters chosen for a small (~26k-row) tabular problem.
    Each hyperparameter, in plain terms:

      objective="multi:softprob"  -> output a probability per class (H/D/A)
      num_class=3                 -> three outcomes
      eval_metric="mlogloss"      -> optimise/early-stop on multiclass log-loss,
                                     the metric that punishes overconfident wrong
                                     calls (our real target, not raw accuracy)
      n_estimators=600            -> up to 600 boosting trees (early stopping
                                     will usually cut this far shorter)
      max_depth=4                 -> shallow trees. Depth caps how many features
                                     can interact in one tree; 4 is enough for
                                     "form x venue x importance" without letting
                                     the model memorise 26k rows
      learning_rate=0.05          -> shrink each tree's contribution, so no
                                     single tree dominates (more trees, each a
                                     gentler nudge = better generalisation)
      subsample=0.8               -> each tree sees 80% of rows (row bagging)
      colsample_bytree=0.8        -> each tree sees 80% of features
      min_child_weight=5          -> a split must keep >=~5 (weighted) samples
                                     per side; blocks tiny overfit leaves
      reg_lambda=1.0              -> L2 penalty on leaf weights (regularisation)
      early_stopping_rounds=40    -> stop if val log-loss doesn't improve for 40
                                     rounds; keeps the best iteration
    """
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_estimators=600,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        early_stopping_rounds=40,
        random_state=42,
        n_jobs=-1,
    )


def train():
    df = load_matrix()
    train_df, val_df, test_df = temporal_split(df)
    print(f"split sizes  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[LABEL_COLUMN].map(LABEL_TO_INT)
    X_val, y_val = val_df[FEATURE_COLUMNS], val_df[LABEL_COLUMN].map(LABEL_TO_INT)

    model = build_model()
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights(train_df),
        eval_set=[(X_val, y_val)],   # early stopping watches val log-loss
        verbose=False,
    )
    print(f"best iteration: {model.best_iteration}  (of up to {model.n_estimators})")
    print(f"best val mlogloss: {model.best_score:.4f}")

    model.save_model(MODEL_PATH)
    print(f"saved model -> {MODEL_PATH}")

    # --- sanity checks: does the model behave like football knowledge says? ---
    proba = model.predict_proba(X_val)  # rows sum to 1 by construction
    assert np.allclose(proba.sum(axis=1), 1.0), "probabilities must sum to 1"
    print("[ok] predicted probability vectors sum to 1")

    # a lopsided fixture: pick the val match with the largest home-vs-away form
    # gap and confirm the model heavily favours the stronger side
    gap = (val_df["home_form"] - val_df["away_form"])
    idx = gap.idxmax()
    p = model.predict_proba(val_df.loc[[idx], FEATURE_COLUMNS])[0]
    row = val_df.loc[idx]
    print(f"\nmost lopsided val fixture: {row['home_team']} (form {row['home_form']:.2f}) "
          f"vs {row['away_team']} (form {row['away_form']:.2f})")
    print(f"  P(home)={p[0]:.2f}  P(draw)={p[1]:.2f}  P(away)={p[2]:.2f}"
          f"   actual={row['result']}")

    return model


if __name__ == "__main__":
    train()
