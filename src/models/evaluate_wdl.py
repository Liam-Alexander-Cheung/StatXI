"""
Step 3 of the XGBoost Win/Draw/Loss model: score the trained model on the
held-out test block (2016+), honestly, against baselines.

The headline is NOT raw accuracy. Per the project's target reset, success means:
  (1) beat a no-skill baseline on BOTH accuracy AND log-loss, and
  (2) be reasonably calibrated (when it says 60%, it happens ~60% of the time).

Baselines we compare against:
  - "always Home"      : predict the majority class every time (accuracy floor).
  - "form favourite"   : predict whichever side has the higher rolling_form
                         (home breaks ties) — a stronger, fairer accuracy floor.
  - "base rates"       : assign every match the SAME probability vector, equal
                         to the training-set class frequencies. This is the
                         no-skill *probabilistic* forecast — the thing log-loss
                         and Brier must beat for the model to have added skill.

We also break out the major-tournament subset (World Cup + Euro) specifically,
since that is what the backtest actually claims.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, recall_score
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier

from src.models.build_matrix import FEATURE_COLUMNS, LABEL_COLUMN
from src.models.train_wdl import (
    load_matrix, temporal_split, MODEL_PATH,
    LABEL_TO_INT, INT_TO_LABEL,
)

CLASSES = [0, 1, 2]            # H, D, A  (integer encoding from train_wdl)
CLASS_NAMES = ["H", "D", "A"]
MAJOR_TOURNAMENTS = {"FIFA World Cup", "UEFA Euro"}


def multiclass_brier(y_int: np.ndarray, proba: np.ndarray) -> float:
    """
    Multiclass Brier score = mean squared error between the predicted
    probability vector and the one-hot truth. Lower is better; 0 is perfect.
    sklearn only ships the binary version, so we compute it directly.
    """
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y_int)), y_int] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def form_favourite_pred(df: pd.DataFrame) -> np.ndarray:
    """Predict the higher-rolling_form side; home wins ties / missing data."""
    # (NaN comparisons are False, so a NaN form never 'wins' the comparison —
    #  it falls through to the home-team default, a deliberate naive choice.)
    return np.where(df["away_form"] > df["home_form"], "A", "H")


def evaluate_block(name: str, df: pd.DataFrame, model: XGBClassifier,
                   base_rate_vec: np.ndarray):
    """Print the full metric comparison for one slice of matches."""
    y_str = df[LABEL_COLUMN].to_numpy()
    y_int = df[LABEL_COLUMN].map(LABEL_TO_INT).to_numpy()

    # --- model ---
    proba = model.predict_proba(df[FEATURE_COLUMNS])
    pred_int = proba.argmax(axis=1)
    pred_str = np.array([INT_TO_LABEL[i] for i in pred_int])

    # --- baselines ---
    always_home = np.full(len(df), "H")
    form_fav = form_favourite_pred(df)
    base_rate_proba = np.tile(base_rate_vec, (len(df), 1))  # same vector each row

    print(f"\n{'='*66}\n{name}   (n={len(df)})\n{'='*66}")
    print(f"{'metric':<16}{'model':>12}{'form-fav':>12}{'always-H':>12}{'base-rate':>12}")
    print(f"{'accuracy':<16}{accuracy_score(y_str, pred_str):>12.3f}"
          f"{accuracy_score(y_str, form_fav):>12.3f}"
          f"{accuracy_score(y_str, always_home):>12.3f}{'-':>12}")
    print(f"{'log-loss':<16}{log_loss(y_int, proba, labels=CLASSES):>12.3f}"
          f"{'-':>12}{'-':>12}"
          f"{log_loss(y_int, base_rate_proba, labels=CLASSES):>12.3f}")
    print(f"{'brier':<16}{multiclass_brier(y_int, proba):>12.3f}"
          f"{'-':>12}{'-':>12}{multiclass_brier(y_int, base_rate_proba):>12.3f}")

    # per-class recall — draws are the hard class; make that visible
    rec = recall_score(y_str, pred_str, labels=CLASS_NAMES, average=None, zero_division=0)
    print(f"\nper-class recall (model):  " +
          "  ".join(f"{c}={r:.3f}" for c, r in zip(CLASS_NAMES, rec)))
    # how often the model even *picks* each class as its top guess
    picked = pd.Series(pred_str).value_counts(normalize=True)
    print("model's predicted-class mix: " +
          "  ".join(f"{c}={picked.get(c,0):.3f}" for c in CLASS_NAMES))

    # --- bookmaker comparison, on the odds-covered subset of THIS slice -------
    # book_ph/pd/pa are metadata joined by build_matrix (NaN where no odds).
    # Compare model vs market on exactly the matches that have a de-vigged
    # price, re-scoring the model on that same subset for a fair head-to-head.
    if {"book_ph", "book_pd", "book_pa"}.issubset(df.columns):
        book = df[["book_ph", "book_pd", "book_pa"]].to_numpy(dtype=float)
        covered = ~np.isnan(book).any(axis=1)
        n_cov = int(covered.sum())
        if n_cov:
            yb, pb, bb = y_int[covered], proba[covered], book[covered]
            yb_str = y_str[covered]
            pred_m = np.array([INT_TO_LABEL[i] for i in pb.argmax(axis=1)])
            pred_bk = np.array([INT_TO_LABEL[i] for i in bb.argmax(axis=1)])
            print(f"\n--- vs BOOKMAKER (odds-covered subset, n={n_cov} of {len(df)}) ---")
            print(f"{'metric':<12}{'model':>12}{'bookmaker':>12}")
            print(f"{'accuracy':<12}{accuracy_score(yb_str, pred_m):>12.3f}"
                  f"{accuracy_score(yb_str, pred_bk):>12.3f}")
            print(f"{'log-loss':<12}{log_loss(yb, pb, labels=CLASSES):>12.3f}"
                  f"{log_loss(yb, bb, labels=CLASSES):>12.3f}")
            print(f"{'brier':<12}{multiclass_brier(yb, pb):>12.3f}"
                  f"{multiclass_brier(yb, bb):>12.3f}")

    return proba, y_str, y_int


def calibration_report(proba: np.ndarray, y_str: np.ndarray, n_bins: int = 10):
    """
    One-vs-rest calibration for the HOME-WIN probability: bin matches by the
    model's P(home win), and in each bin compare the mean predicted probability
    against the actual fraction of home wins. A well-calibrated model tracks the
    diagonal. Also reports ECE (expected calibration error) as a single number.
    """
    p_home = proba[:, 0]                 # column 0 = P(H)
    y_home = (y_str == "H").astype(int)
    frac_pos, mean_pred = calibration_curve(y_home, p_home, n_bins=n_bins, strategy="quantile")

    print(f"\n{'-'*46}\ncalibration of P(home win)  [quantile bins]\n{'-'*46}")
    print(f"{'predicted':>12}{'actual':>12}{'gap':>12}")
    for mp, fp in zip(mean_pred, frac_pos):
        print(f"{mp:>12.3f}{fp:>12.3f}{abs(mp-fp):>12.3f}")

    # ECE weights each bin by how many samples fall in it
    counts, edges = np.histogram(p_home, bins=n_bins)
    ece = 0.0
    for lo, hi, mp, fp in zip(edges[:-1], edges[1:], mean_pred, frac_pos):
        w = np.mean((p_home >= lo) & (p_home <= hi))
        ece += w * abs(mp - fp)
    print(f"\nECE (home-win prob): {ece:.3f}   (0 = perfectly calibrated)")


def upset_illustration(proba: np.ndarray, df: pd.DataFrame):
    """
    'Expect the unexpected' check: on matches the form-favourite baseline got
    WRONG (an upset), what probability did the model assign to the outcome that
    actually happened? Compare it to the base rate — if it's higher, the model
    was leaning toward the upset more than a no-skill guess would.
    """
    y_str = df[LABEL_COLUMN].to_numpy()
    form_fav = form_favourite_pred(df)
    upsets = form_fav != y_str
    if upsets.sum() == 0:
        return
    y_int = df[LABEL_COLUMN].map(LABEL_TO_INT).to_numpy()
    p_actual = proba[np.arange(len(df)), y_int]      # P the model gave the truth
    print(f"\n{'-'*46}\n'expect the unexpected' check\n{'-'*46}")
    print(f"form-favourite was wrong (an upset) in {upsets.mean():.1%} of these matches")
    print(f"mean P(actual outcome) the model assigned on those upsets: {p_actual[upsets].mean():.3f}")
    print(f"  (vs {p_actual[~upsets].mean():.3f} on non-upsets — lower is expected,")
    print("   the point is the model still gives upsets real, non-trivial probability)")


def main():
    df = load_matrix()
    train_df, _, test_df = temporal_split(df)

    # no-skill probability vector from TRAIN label frequencies (no test leakage),
    # ordered [H, D, A] to match the model's class order
    base_rate_vec = (train_df[LABEL_COLUMN].value_counts(normalize=True)
                     .reindex(CLASS_NAMES).to_numpy())
    print("train base rates  H/D/A:", np.round(base_rate_vec, 3))

    model = XGBClassifier()
    model.load_model(MODEL_PATH)

    # (a) all held-out matches
    proba_all, y_all, _ = evaluate_block("ALL TEST MATCHES (2016+)", test_df, model, base_rate_vec)
    calibration_report(proba_all, y_all)
    upset_illustration(proba_all, test_df)

    # (b) the backtest headline: World Cup + Euro matches only
    major = test_df[test_df["tournament"].isin(MAJOR_TOURNAMENTS)]
    evaluate_block("MAJOR TOURNAMENTS ONLY (World Cup + Euro, 2016+)", major, model, base_rate_vec)


if __name__ == "__main__":
    main()
