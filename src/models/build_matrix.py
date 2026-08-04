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
    load_squads,
    load_team_pool_ratings,
    load_tournament_editions,
    match_tournament_edition,
)
from src.features import (
    rolling_form,
    goal_trend,
    head_to_head_record,
    squad_age_depth,
    team_chemistry,
    build_pool_index,
    team_pool_rating,
)

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
    # --- v2 squad-quality features (NaN for ~99% of matches: only the ~875
    #     World Cup / Euro fixtures with a scraped squad edition get real
    #     values; XGBoost handles the NaNs natively). See match_tournament_edition
    #     for how a match row is tied to its tournament edition. Candidate set —
    #     any column that doesn't earn its keep on held-out log-loss is pruned. ---
    "home_mean_age",              # squad_age_depth(home).mean_age
    "away_mean_age",              # squad_age_depth(away).mean_age
    "home_squad_size",            # squad_age_depth(home).squad_size (depth: ~23/26)
    "away_squad_size",            # squad_age_depth(away).squad_size
    "home_club_hhi",              # team_chemistry(home).club_hhi (club concentration)
    "away_club_hhi",              # team_chemistry(away).club_hhi
    "home_same_club_pair_ratio",  # team_chemistry(home).same_club_pair_ratio
    "away_same_club_pair_ratio",  # team_chemistry(away).same_club_pair_ratio
    "home_largest_club_bloc",     # team_chemistry(home).largest_club_bloc
    "away_largest_club_bloc",     # team_chemistry(away).largest_club_bloc
    # --- v3 rating feature (TRACK 2). rating_gap = home_pool_rating -
    #     away_pool_rating, the RELATIVE form (beat two absolute columns 0.509 vs
    #     0.459). Pool ratings extend coverage from the 5 scraped tournaments to
    #     ~1,081 leakage-free international matches (2016+); NaN otherwise. ---
    "rating_gap",                 # home_pool_rating - away_pool_rating
]

# Columns kept for traceability / verification / later scoreline work, but
# explicitly NOT features. home_score/away_score are the label's raw source.
# `edition` is the resolved tournament-edition label (e.g. "World Cup 2014") or
# blank — kept so a squad-feature value in a row can be traced back to the exact
# squad it came from during verification.
METADATA_COLUMNS = ["date", "home_team", "away_team", "tournament", "edition",
                    "home_pool_rating", "away_pool_rating",  # absolute ratings, kept for traceability
                    "home_score", "away_score"]

LABEL_COLUMN = "result"  # "H" (home win) / "D" (draw) / "A" (away win)

# The five squad-quality scalars we keep, and the all-NaN placeholder used when a
# squad can't be located. Recording NaN (not 0) is deliberate: a squad_size of 0
# or largest_club_bloc of 0 would *falsely assert* an empty squad, when the truth
# is "unknown" (no edition, or a team-name mismatch like Germany@1990 = the old
# "West Germany" squad row). NaN routes it into XGBoost's missing-value handling,
# honouring the project's never-fabricate-data rule.
_SQUAD_FIELDS = ("mean_age", "squad_size", "club_hhi",
                 "same_club_pair_ratio", "largest_club_bloc")
_SQUAD_NAN = {field: float("nan") for field in _SQUAD_FIELDS}


def _squad_feature_row(squads: pd.DataFrame, team: str, edition: str | None) -> dict:
    """
    The five squad-quality scalars for one team at one tournament edition.

    Returns all-NaN (never a fabricated 0) when the match has no linkable
    edition (`edition is None`) or the team name isn't present in that edition's
    squad — the latter catches the ~27 historical name mismatches (Germany@1990,
    Russia/Soviet Union, Serbia/FR Yugoslavia, China/China PR).
    """
    if edition is None:
        return dict(_SQUAD_NAN)

    # Is this exact team name present in that edition's squad at all? If not,
    # we must NOT fall through to squad_age_depth/team_chemistry — they'd return
    # squad_size=0 / largest_club_bloc=0, i.e. fabricated zeros. .any() is a
    # cheap membership test before the heavier per-position / club-pair maths.
    present = (
        (squads["team"] == team) & (squads["tournament_name"] == edition)
    ).any()
    if not present:
        return dict(_SQUAD_NAN)

    ad = squad_age_depth(squads, team, edition)      # age / depth signal
    tc = team_chemistry(squads, team, edition)       # club-cohesion signal
    return {
        "mean_age": ad["mean_age"],
        "squad_size": ad["squad_size"],
        "club_hhi": tc["club_hhi"],
        "same_club_pair_ratio": tc["same_club_pair_ratio"],
        "largest_club_bloc": tc["largest_club_bloc"],
    }


def _result(home_score: float, away_score: float) -> str:
    """Map a final score to the 3-way label."""
    if home_score > away_score:
        return "H"
    if home_score == away_score:
        return "D"
    return "A"


def build_training_matrix(
    matches: pd.DataFrame,
    limit: int | None = None,
    squads: pd.DataFrame | None = None,
    editions: dict | None = None,
    pool_index: dict | None = None,
) -> pd.DataFrame:
    """
    Build the (features + label) table from cleaned match data.

    `matches` must be the FULL cleaned history — each feature call looks back
    from a match's date across all of it, so we can't pass a subset.

    `limit`: if set, only process the most recent N matches — used for fast
    correctness checks before paying the full ~10-minute build.

    `squads` / `editions` / `pool_index`: the squad table, tournament-edition
    lookup, and country pool-rating index used by the squad + rating features.
    Loaded once here if not supplied (tests can inject them). Loading happens
    outside the row loop — one DB read, not 32k.

    Returns a DataFrame with METADATA_COLUMNS + FEATURE_COLUMNS + LABEL_COLUMN.
    """
    # Load the squad-feature inputs once. All are small and constant across the
    # whole build, so they live outside the loop.
    if squads is None:
        squads = load_squads()
    if editions is None:
        editions = load_tournament_editions()
    if pool_index is None:
        pool_index = build_pool_index(load_team_pool_ratings())

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

        # Tie this match to its specific tournament edition (or None), then pull
        # the squad-quality scalars for each side. NaN for non-tournament matches
        # and unresolved name mismatches — see _squad_feature_row.
        edition = match_tournament_edition(row["tournament"], d, editions)
        home_sq = _squad_feature_row(squads, home, edition)
        away_sq = _squad_feature_row(squads, away, edition)

        # Date-scoped pool ratings — apply to ANY covered match, not just the 5
        # tournaments. rating_gap is the relative form (beat absolute columns).
        # NaN - NaN stays NaN, so a match needs BOTH teams rated to get a gap.
        home_pool = team_pool_rating(pool_index, home, d)
        away_pool = team_pool_rating(pool_index, away, d)
        rating_gap = home_pool - away_pool

        records.append({
            # --- metadata (not features) ---
            "date": d,
            "home_team": home,
            "away_team": away,
            "tournament": row["tournament"],
            "edition": edition,          # resolved edition label, or None
            "home_pool_rating": home_pool,   # absolute ratings (traceability, not fed to model)
            "away_pool_rating": away_pool,
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
            # --- v2 squad-quality features (home/away kept separate; XGBoost
            #     learns the home-vs-away interaction itself) ---
            "home_mean_age": home_sq["mean_age"],
            "away_mean_age": away_sq["mean_age"],
            "home_squad_size": home_sq["squad_size"],
            "away_squad_size": away_sq["squad_size"],
            "home_club_hhi": home_sq["club_hhi"],
            "away_club_hhi": away_sq["club_hhi"],
            "home_same_club_pair_ratio": home_sq["same_club_pair_ratio"],
            "away_same_club_pair_ratio": away_sq["same_club_pair_ratio"],
            "home_largest_club_bloc": home_sq["largest_club_bloc"],
            "away_largest_club_bloc": away_sq["largest_club_bloc"],
            # --- v3 rating feature (TRACK 2): relative pool-rating gap ---
            "rating_gap": rating_gap,
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
