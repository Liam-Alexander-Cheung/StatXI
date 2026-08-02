"""
Link Wikipedia-sourced `players` rows to `player_ratings` rows (sofifa/FIFA
data) — the blocking->score->tier matcher planned in reports/methodology.md
(Workstream B/C). The string half (normalize_name/name_similarity) already
lives in src/name_matching.py; this script is the rest of it.

Scope: only the five tournaments that actually fall inside the ratings
source's coverage window (FIFA 15-24, Sept 2014 - Sept 2023) are attempted.
A Euro 1996 squad has no FIFA video game rating to find, at all — this
script does not invent one for it, per the project's no-fabrication rule.

Method
------
BLOCK - cut the candidate set down before any string comparison, using two
cheap, near-certain filters: (1) the tournament's own era-matched
fifa_version (a Euro 2016 player is only ever compared against FIFA16 rows),
and (2) nationality (a Germany squad is never compared against Brazilian
ratings). This turns what would be an O(3,364 squad-players x 180,021
ratings) problem into ~O(squad-size x same-nationality-same-edition
candidates) - typically a few dozen to a few hundred per player, not 180,021.

SCORE - within a block, two signals: name_similarity() against both the
source's short_name and long_name (whichever scores higher — sofifa uses
short display names like "L. Messi", Wikipedia usually the fuller form), and
an exact date_of_birth match. Two different real people sharing both a
fuzzy-similar name AND an identical birthdate is vanishingly unlikely, so DOB
agreement is treated as close to decisive, not just another number to blend
in.

TIER - HIGH (name_score >= 0.90 AND dob matches) is auto-accepted and
written to players.sofifa_player_id. MEDIUM (strong on one signal, weak on
the other) is written to a review-queue CSV for a human to check, never
auto-linked. Anything weaker than that is left alone entirely — not even
logged, since below ~0.55 name-similarity is within the "genuinely different
people" noise floor already established in name_matching.py's own docstring
verification.
"""

from pathlib import Path

import pandas as pd

from src.database import get_connection
from src.name_matching import name_similarity

# Wikipedia's squads.country -> sofifa's player_ratings.nationality_name,
# only where they genuinely differ. Found by directly diffing the two
# sources' country-name sets across all 5 in-scope tournaments (checked, not
# assumed) - South Korea was the only mismatch. Grown as new mismatches
# appear, same spirit as name_matching.py's _LETTER_FOLD table.
NATIONALITY_ALIASES = {
    "South Korea": "Korea Republic",
}

# Each tournament maps to the closest FIFA/FC edition dated BEFORE it - the
# same "closest preceding snapshot" rule throughout, and the methodologically
# correct choice, not just the convenient one (see the Kimmich/Musiala
# finding: rating signal is tournament-date-sensitive).
TOURNAMENT_TO_FIFA_VERSION = {
    "Euro 2016": 16,        # 2015-09-21, ~9 months before
    "World Cup 2018": 18,   # 2017-09-18, ~9 months before. FIFA19 (2018-08-21)
                             # released AFTER the tournament - excluded.
    "Euro 2020": 21,        # 2020-09-23, ~9 months before the actual June
                             # 2021 play (COVID delay)
    "World Cup 2022": 23,   # 2022-09-26, ~2 months before
    "Euro 2024": 24,        # 2023-09-22, ~9 months before
}

HIGH_NAME_THRESHOLD = 0.90
MEDIUM_NAME_THRESHOLD = 0.55  # name_matching.py's own "genuinely different
                               # players" ceiling was 0.31-0.35; 0.55 stays
                               # well clear of that noise floor.

REVIEW_QUEUE_PATH = Path(__file__).resolve().parent / "ratings_match_review_queue.csv"


def load_target_players(conn):
    """Wikipedia players for the 5 in-scope tournaments, one row per
    squad appearance, tagged with the fifa_version to match against."""
    tournament_names = list(TOURNAMENT_TO_FIFA_VERSION)
    placeholders = ",".join("?" * len(tournament_names))
    df = pd.read_sql(
        f"""
        SELECT p.player_appearance_id, p.player_name, p.date_of_birth,
               p.club, s.country, t.name AS tournament_name
        FROM players p
        JOIN squads s ON p.squad_id = s.squad_id
        JOIN tournaments t ON s.tournament_id = t.tournament_id
        WHERE t.name IN ({placeholders})
        """,
        conn,
        params=tournament_names,
    )
    df["fifa_version"] = df["tournament_name"].map(TOURNAMENT_TO_FIFA_VERSION)
    df["block_nationality"] = df["country"].map(lambda c: NATIONALITY_ALIASES.get(c, c))
    return df


def load_candidate_pool(conn):
    """player_ratings rows for only the fifa_versions actually needed."""
    versions = sorted(set(TOURNAMENT_TO_FIFA_VERSION.values()))
    placeholders = ",".join("?" * len(versions))
    return pd.read_sql(
        f"""
        SELECT sofifa_player_id, fifa_version, short_name, long_name,
               date_of_birth, club_name, nationality_name
        FROM player_ratings
        WHERE fifa_version IN ({placeholders})
        """,
        conn,
        params=versions,
    )


def build_blocks(ratings_df):
    """Group candidates by (fifa_version, nationality_name) so each player
    only has to be compared against its own small block, not all 180k rows."""
    blocks = {}
    for (fv, nat), group in ratings_df.groupby(["fifa_version", "nationality_name"]):
        blocks[(fv, nat)] = group.to_dict("records")
    return blocks


def score_candidate(player_name, player_dob, candidate):
    """(name_score, dob_match) for one Wikipedia player against one rating
    candidate. name_score takes the better of short_name/long_name, since
    sofifa's short_name ("L. Messi") and Wikipedia's fuller name don't always
    score the same way against token_set_ratio."""
    name_score = max(
        name_similarity(player_name, candidate["short_name"]),
        name_similarity(player_name, candidate["long_name"]),
    )
    dob_match = player_dob == candidate["date_of_birth"]
    return name_score, dob_match


def classify(name_score, dob_match):
    if dob_match and name_score >= HIGH_NAME_THRESHOLD:
        return "high"
    if (dob_match and name_score >= MEDIUM_NAME_THRESHOLD) or (
        not dob_match and name_score >= HIGH_NAME_THRESHOLD
    ):
        return "medium"
    return None


def match_player(player_row, blocks):
    """Best candidate + tier for one player, or (None, None, None) if the
    block is empty or nothing clears even the medium floor."""
    block = blocks.get((player_row["fifa_version"], player_row["block_nationality"]), [])
    best_candidate, best_score, best_dob_match = None, -1.0, False
    for candidate in block:
        name_score, dob_match = score_candidate(
            player_row["player_name"], player_row["date_of_birth"], candidate
        )
        # dob_match ranks above name_score - an exact birthdate match with a
        # slightly lower name score beats a higher name score with no dob
        # agreement at all.
        if (dob_match, name_score) > (best_dob_match, best_score):
            best_candidate, best_score, best_dob_match = candidate, name_score, dob_match

    if best_candidate is None:
        return None, None, None
    tier = classify(best_score, best_dob_match)
    if tier is None:
        return None, None, None
    return best_candidate, best_score, tier


def main():
    conn = get_connection()
    conn.execute("PRAGMA foreign_keys = ON")

    players_df = load_target_players(conn)
    ratings_df = load_candidate_pool(conn)
    blocks = build_blocks(ratings_df)

    high_matches = []   # written straight to the DB
    review_queue = []   # printed to CSV, never auto-written
    high_ids = set()
    review_ids = set()

    for _, row in players_df.iterrows():
        candidate, score, tier = match_player(row, blocks)
        if tier == "high":
            high_matches.append((candidate["sofifa_player_id"], score, "high", row["player_appearance_id"]))
            high_ids.add(row["player_appearance_id"])
        elif tier == "medium":
            review_ids.add(row["player_appearance_id"])
            review_queue.append({
                "tournament": row["tournament_name"],
                "wiki_player_name": row["player_name"],
                "wiki_dob": row["date_of_birth"],
                "wiki_club": row["club"],
                "wiki_country": row["country"],
                "candidate_sofifa_player_id": candidate["sofifa_player_id"],
                "candidate_short_name": candidate["short_name"],
                "candidate_long_name": candidate["long_name"],
                "candidate_dob": candidate["date_of_birth"],
                "candidate_club": candidate["club_name"],
                "name_score": round(score, 3),
            })

    conn.executemany(
        "UPDATE players SET sofifa_player_id=?, rating_match_score=?, rating_match_tier=? "
        "WHERE player_appearance_id=?",
        high_matches,
    )
    conn.commit()

    if review_queue:
        pd.DataFrame(review_queue).to_csv(REVIEW_QUEUE_PATH, index=False)

    # Coverage report, per tournament - counts, not vibes.
    print(f"{'Tournament':<16} {'total':>6} {'high':>6} {'review':>7} {'none':>6} {'coverage':>9}")
    for tname in TOURNAMENT_TO_FIFA_VERSION:
        sub = players_df[players_df.tournament_name == tname]
        total = len(sub)
        high = sub["player_appearance_id"].isin(high_ids).sum()
        review = sub["player_appearance_id"].isin(review_ids).sum()
        none = total - high - review
        print(f"{tname:<16} {total:>6} {high:>6} {review:>7} {none:>6} {high/total:>8.1%}")

    print()
    print(f"Total: {len(players_df)} players, {len(high_matches)} auto-linked (high), "
          f"{len(review_queue)} flagged for manual review, "
          f"{len(players_df) - len(high_matches) - len(review_queue)} unmatched.")
    if review_queue:
        print(f"Review queue written to {REVIEW_QUEUE_PATH}")

    conn.close()


if __name__ == "__main__":
    main()
