"""
Step 1 of the XGBoost Win/Draw/Loss model: turn the match history into a
*training matrix* — one row per historical match, columns = features, plus a
label (H / D / A).

The feature functions in src/features.py answer "what was team X's form as of
date D?" one call at a time. XGBoost instead needs a rectangular table: every
match as a row, every feature as a column. This module bridges the two by
walking the matches in date order and, for each one, calling those already-
verified feature functions "as of" that match's own date.

Why "as of that match's date" matters: it is the whole ballgame for avoiding
*data leakage* (letting the model peek at information it wouldn't have had at
kickoff). The feature functions already filter on `matches["date"] < as_of_date`
(strictly before), so a match can never see its own result or any future match.
That strict-before filter is what makes the resulting accuracy honest rather
than a fantasy 90%.
"""

# Python 3.9: makes all type annotations lazy strings, so newer union syntax
# like `int | None` (a 3.10+ feature) parses fine here instead of erroring.
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.data_pipeline import (
    load_raw_matches,
    clean_matches,
    importance_weight,
)
from src.features import rolling_form, goal_trend, head_to_head_record

# Where the cached matrix lands. data/processed/ is gitignored (regenerable
# derived data — same policy as data/raw/ and the .db file).
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MATRIX_PATH = PROCESSED_DIR / "training_matrix.csv"

# --- Single source of truth for which columns are model INPUTS. -------------
# This list is the leakage guard. The training step selects features by this
# name list ONLY — never "every column except the label" — so the metadata
# columns below (especially home_score / away_score, which literally *are* the
# answer) can sit in the same table for traceability without any risk of being
# fed to the model by accident.
FEATURE_COLUMNS = [
    "neutral",       # 1 = neutral venue (no real home advantage); essential —
                     #     28% of matches are neutral, and tournaments mostly are
    "importance",    # tournament-tier multiplier (friendly 0.3 ... World Cup 1.0)
    "home_form",     # rolling_form(home) as of match date  (NaN for debutants)
    "away_form",     # rolling_form(away)
    "home_gs",       # goal_trend(home).goals_scored
    "home_gc",       # goal_trend(home).goals_conceded
    "away_gs",       # goal_trend(away).goals_scored
    "away_gc",       # goal_trend(away).goals_conceded
    "h2h",           # head_to_head_record(home vs away)   (NaN if never met)
]

# Columns kept for traceability / verification / later scoreline work, but
# explicitly NOT features. home_score/away_score are the label's raw source.
METADATA_COLUMNS = ["date", "home_team", "away_team", "tournament",
                    "home_score", "away_score"]

LABEL_COLUMN = "result"  # "H" (home win) / "D" (draw) / "A" (away win)


def _result(home_score: float, away_score: float) -> str:
    """Map a final score to the 3-way label."""
    if home_score > away_score:
        return "H"
    if home_score == away_score:
        return "D"
    return "A"


def build_training_matrix(matches: pd.DataFrame, limit: int | None = None) -> pd.DataFrame:
    """
    Build the (features + label) table from cleaned match data.

    `matches` must be the FULL cleaned history — each feature call looks back
    from a match's date across all of it, so we can't pass a subset.

    `limit`: if set, only process the most recent N matches — used for fast
    correctness checks before paying the full ~10-minute build.

    Returns a DataFrame with METADATA_COLUMNS + FEATURE_COLUMNS + LABEL_COLUMN.
    """
    # Sort ascending by date so "progress" is chronological and reproducible.
    # reset_index gives clean 0..N-1 row numbers for the progress print.
    matches = matches.sort_values("date").reset_index(drop=True)

    # The rows we actually build a training example for. The feature *lookups*
    # below still use the full `matches` (all history), regardless of `limit`.
    target_rows = matches.tail(limit) if limit is not None else matches

    records = []
    t0 = time.time()
    total = len(target_rows)

    # iterrows() yields (index, row) pairs. It's the slow-but-readable way to
    # go row by row; at 32k rows and ~10 min total it's acceptable for a
    # one-time cached batch (we measured it before committing to this design).
    for n, (_, row) in enumerate(target_rows.iterrows(), start=1):
        d = row["date"]
        home, away = row["home_team"], row["away_team"]

        # goal_trend returns a dict; pull the two independent signals we keep
        # (differential is exactly scored - conceded, so it's redundant for a
        # tree and left out).
        home_gt = goal_trend(matches, home, d)
        away_gt = goal_trend(matches, away, d)

        records.append({
            # --- metadata (not features) ---
            "date": d,
            "home_team": home,
            "away_team": away,
            "tournament": row["tournament"],
            "home_score": row["home_score"],
            "away_score": row["away_score"],
            # --- features ---
            "neutral": int(row["neutral"]),
            "importance": importance_weight(row["tournament"]),
            "home_form": rolling_form(matches, home, d),
            "away_form": rolling_form(matches, away, d),
            "home_gs": home_gt["goals_scored"],
            "home_gc": home_gt["goals_conceded"],
            "away_gs": away_gt["goals_scored"],
            "away_gc": away_gt["goals_conceded"],
            "h2h": head_to_head_record(matches, home, away, d),
            # --- label ---
            "result": _result(row["home_score"], row["away_score"]),
        })

        # progress heartbeat every 2000 rows so a 10-min background run is
        # observably alive, not silently hung
        if n % 2000 == 0:
            elapsed = time.time() - t0
            print(f"  {n}/{total} rows  ({elapsed:.0f}s, "
                  f"~{elapsed/n*total:.0f}s projected total)", flush=True)

    # Column order: metadata, then features, then label — readable left-to-right
    return pd.DataFrame(records, columns=METADATA_COLUMNS + FEATURE_COLUMNS + [LABEL_COLUMN])


if __name__ == "__main__":
    # Run from repo root as:  python -m src.models.build_matrix
    # (the `-m` form puts the repo root on sys.path so `import src...` resolves —
    # same reason webapp/app.py must be run with -m, see methodology.md)
    print("Loading + cleaning matches (~14s)...", flush=True)
    m = clean_matches(load_raw_matches())
    print(f"cleaned matches: {m.shape}", flush=True)

    print("Building training matrix (this is the ~10-min step)...", flush=True)
    t0 = time.time()
    matrix = build_training_matrix(m)
    print(f"built matrix: {matrix.shape} in {time.time() - t0:.0f}s", flush=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(MATRIX_PATH, index=False)
    print(f"saved -> {MATRIX_PATH}", flush=True)

    # quick health readout: label balance + missingness (debutants / never-met)
    print("\nlabel balance:\n", matrix["result"].value_counts(normalize=True).round(3))
    print("\nNaN counts per feature:\n", matrix[FEATURE_COLUMNS].isna().sum())
