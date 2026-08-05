"""
Shared honest-tuning protocol, used by BOTH tuners:
  - src/models/tune_weights.py      (varies the per-match training weighting)
  - src/models/tune_hyperparams.py  (varies the XGBoost hyperparameters)

Keeping this machinery in one place matters because it is the leakage-critical
core of the whole approach — the pre-2016 inner objective and the held-out
bootstrap comparison. Two tuners duplicating it would risk one silently drifting
out of leakage-safety. Each tuner supplies only its own injection point
(`weight_fn` or `model_fn`); everything about *how* candidates are scored honestly
lives here.

The rules (identical to what tune_weights established):
  1. Rank candidates by pre-2016 walk-forward VALIDATION log-loss (never accuracy).
  2. The inner objective NEVER scores a 2016+ match — those are reserved for one
     final held-out comparison.
  3. Report a paired bootstrap CI on the held-out difference, so a within-noise
     result is called what it is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score

from src.data_pipeline import recency_weight
from src.models.build_matrix import FEATURE_COLUMNS, LABEL_COLUMN
from src.models.train_wdl import LABEL_TO_INT, build_model
from src.models.walk_forward import _fit_before, VAL_DAYS, CLASSES, CLASS_NAMES
from src.models.evaluate_wdl import multiclass_brier

# The wall the inner search must never cross (matches train_wdl.TEST_START).
INNER_TEST_START = pd.Timestamp("2016-01-01")

# Rolling-origin scoring blocks, all ending on/before INNER_TEST_START. Two-year
# blocks from 2004 give 6 folds with ~1.5-2k matches each and >=14 years of
# training history before the first one. Fold k trains on everything before block
# k's start (expanding window) — the same discipline as the real backtest.
INNER_BLOCK_EDGES = [pd.Timestamp(f"{y}-01-01") for y in range(2004, 2018, 2)]


def inner_folds() -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """(block_start, block_end) pairs for the pre-2016 rolling-origin CV. Every
    block_end is <= INNER_TEST_START, so no fold ever scores a 2016+ match."""
    edges = INNER_BLOCK_EDGES
    folds = list(zip(edges[:-1], edges[1:]))
    assert all(end <= INNER_TEST_START for _, end in folds), "a fold crosses into the test era"
    return folds


def default_weight_fn(sub: pd.DataFrame, ref_date: pd.Timestamp) -> np.ndarray:
    """The production baseline weighting, vectorised: recency (half-life 3650d,
    floor 0.05) x the row's `importance`. Used as the default when a tuner varies
    only the MODEL and leaves the weighting alone. Numerically identical to
    walk_forward._sample_weights (asserted in tune_hyperparams' self-test)."""
    days_ago = (ref_date - sub["date"]).dt.days.to_numpy()
    recency = np.maximum(0.5 ** (days_ago / 3650.0), 0.05)
    return recency * sub["importance"].to_numpy()


def inner_objective(df: pd.DataFrame, weight_fn=default_weight_fn, model_fn=build_model,
                    folds=None, verbose: bool = False) -> dict:
    """
    Pooled PRE-2016 walk-forward validation score for one candidate. For each
    rolling block [start, end): fit a fresh model on all matches before `start`
    (via walk_forward._fit_before, with the injected weighting + model builder +
    early stopping), then predict the block. Pool truth + probabilities across
    blocks and score ONCE.

    A tuner injects EITHER a `weight_fn` (weighting search) OR a `model_fn`
    (hyperparameter search); the other stays at its production default.

    Returns {'log_loss', 'acc', 'n', 'n_fits', 'max_scored_date'}. `log_loss` is
    the objective (lower better); `max_scored_date` is the leakage guard — it must
    stay strictly before INNER_TEST_START.
    """
    if folds is None:
        folds = inner_folds()

    y_all, proba_all = [], []
    n_fits = 0
    max_scored = pd.Timestamp.min
    for start, end in folds:
        block = df[(df["date"] >= start) & (df["date"] < end)]
        if len(block) == 0:
            continue
        model, n_tr, _ = _fit_before(df, start, FEATURE_COLUMNS, VAL_DAYS, weight_fn, model_fn)
        proba = model.predict_proba(block[FEATURE_COLUMNS])
        y_all.append(block[LABEL_COLUMN].map(LABEL_TO_INT).to_numpy())
        proba_all.append(proba)
        n_fits += 1
        max_scored = max(max_scored, block["date"].max())
        if verbose:
            print(f"  fold {start.year}-{end.year}: score n={len(block):4d}  train n={n_tr:5d}  trees={model.best_iteration}")

    y = np.concatenate(y_all)
    proba = np.vstack(proba_all)
    return {
        "log_loss": float(log_loss(y, proba, labels=CLASSES)),
        "acc": float(accuracy_score(y, proba.argmax(axis=1))),
        "n": int(len(y)),
        "n_fits": n_fits,
        "max_scored_date": max_scored,
    }


# --- held-out (2016+) comparison helpers, shared by both tuners' final_report ---
def per_match_logloss(pooled: dict) -> np.ndarray:
    """Log-loss contribution of each individual test match, so two configs can be
    paired match-by-match for a bootstrap. Clipped to avoid log(0)."""
    y_int = pd.Series(pooled["y"]).map(LABEL_TO_INT).to_numpy()
    p_true = pooled["proba"][np.arange(len(y_int)), y_int]
    return -np.log(np.clip(p_true, 1e-15, 1.0))


def bootstrap_diff_ci(d: np.ndarray, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI for the MEAN of paired per-match differences d.
    Returns (point, lo, hi). A CI straddling 0 means the difference is within noise."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot_means = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(d.mean()), float(lo), float(hi)


def pooled_metrics(pooled: dict) -> dict:
    """Accuracy / log-loss / Brier over a pooled walk-forward result."""
    y = pooled["y"]
    y_int = pd.Series(y).map(LABEL_TO_INT).to_numpy()
    proba = pooled["proba"]
    pred = np.array([CLASS_NAMES[i] for i in proba.argmax(axis=1)])
    return {
        "acc": accuracy_score(y, pred),
        "log_loss": log_loss(y_int, proba, labels=CLASSES),
        "brier": multiclass_brier(y_int, proba),
    }
