"""
Build the `player_ratings` table from the raw FIFA/FC player-attribute CSV
(data/raw/fc24_male_players.csv — one snapshot per edition, FIFA 15 through
FC 24, sourced from stefanoleone992's Kaggle datasets, CC0-1.0 licence).

This table is standalone and source-agnostic: it has no foreign key into
tournaments/squads/players, because nothing has matched a rating row to a
specific squad player yet. That's the next, separate step (the
blocking->score->tier name-matcher — see src/name_matching.py and
reports/methodology.md, Workstream B/C). This script only gets the raw
ratings data into the database with real types, not guessed placeholders.
"""

from pathlib import Path

import pandas as pd

from src.database import get_connection

RATINGS_CSV = Path(__file__).resolve().parent / "data" / "raw" / "fc24_male_players.csv"

# The subset of the source CSV's 109 columns this project actually needs:
# identity (for the later name-matcher), the overall/potential rating signal,
# and the bio/club/league fields that feed squad-quality features. Skipping
# the ~90 granular sub-attribute columns (pace, dribbling, ...) - nothing in
# the v2 plan uses them, and they can be added later if that changes.
SOURCE_COLUMNS = [
    "player_id", "fifa_version", "update_as_of", "short_name", "long_name",
    "player_positions", "overall", "potential", "value_eur", "wage_eur",
    "age", "dob", "height_cm", "weight_kg", "club_name", "league_name",
    "league_level", "nationality_name",
]


def safe_int(value):
    """NaN-safe int conversion - same pattern as build_squad_schema.py's
    safe_int(), needed because pandas represents missing numbers as NaN
    (a float), which int() would happily mangle into a wrong number."""
    if pd.isna(value):
        return None
    return int(value)


def safe_float(value):
    if pd.isna(value):
        return None
    return float(value)


def safe_str(value):
    if pd.isna(value):
        return None
    return str(value)


def load_ratings_csv():
    df = pd.read_csv(RATINGS_CSV, usecols=SOURCE_COLUMNS)
    return df


def main():
    conn = get_connection()

    # CSV-backup before any DROP - same safety net as build_squad_schema.py.
    # On a first run player_ratings doesn't exist yet, so there's nothing to
    # lose; this guards a *re-run* after this table has accumulated any
    # future in-place work (e.g. match-confidence columns) that only lives
    # in the DB.
    existing = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='player_ratings'"
    ).fetchone()
    if existing:
        backup = pd.read_sql("SELECT * FROM player_ratings", conn)
        backup.to_csv("player_ratings_backup.csv", index=False)
        print(f"Backed up {len(backup)} existing player_ratings rows to player_ratings_backup.csv")

    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript("""
        DROP TABLE IF EXISTS player_ratings;

        CREATE TABLE player_ratings (
            rating_id INTEGER PRIMARY KEY AUTOINCREMENT,
            sofifa_player_id INTEGER NOT NULL,
            fifa_version INTEGER NOT NULL,
            rating_date DATE NOT NULL,
            short_name TEXT NOT NULL,
            long_name TEXT NOT NULL,
            player_positions TEXT,
            overall INTEGER NOT NULL,
            potential INTEGER NOT NULL,
            value_eur REAL,
            wage_eur REAL,
            age INTEGER NOT NULL,
            date_of_birth DATE NOT NULL,
            height_cm INTEGER,
            weight_kg INTEGER,
            club_name TEXT,
            league_name TEXT,
            league_level INTEGER,
            nationality_name TEXT NOT NULL,
            UNIQUE (sofifa_player_id, fifa_version)
        );
    """)

    df = load_ratings_csv()

    # Build one tuple per row with explicit, NaN-safe type conversion -
    # itertuples() is far faster than df.iterrows() for 180k+ rows.
    rows = [
        (
            safe_int(r.player_id), safe_int(r.fifa_version), safe_str(r.update_as_of),
            safe_str(r.short_name), safe_str(r.long_name), safe_str(r.player_positions),
            safe_int(r.overall), safe_int(r.potential), safe_float(r.value_eur),
            safe_float(r.wage_eur), safe_int(r.age), safe_str(r.dob),
            safe_int(r.height_cm), safe_int(r.weight_kg), safe_str(r.club_name),
            safe_str(r.league_name), safe_int(r.league_level), safe_str(r.nationality_name),
        )
        for r in df.itertuples(index=False)
    ]

    conn.executemany(
        """INSERT INTO player_ratings
           (sofifa_player_id, fifa_version, rating_date, short_name, long_name,
            player_positions, overall, potential, value_eur, wage_eur, age,
            date_of_birth, height_cm, weight_kg, club_name, league_name,
            league_level, nationality_name)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()

    # Verify against known facts about the source, same discipline as the
    # World Cup 2002 skip-count check in methodology.md: don't just trust
    # that the load "ran without an error".
    count = conn.execute("SELECT COUNT(*) FROM player_ratings").fetchone()[0]
    versions = conn.execute(
        "SELECT DISTINCT fifa_version FROM player_ratings ORDER BY fifa_version"
    ).fetchall()
    print(f"Loaded {count} player_ratings rows.")
    print(f"fifa_version coverage: {[v[0] for v in versions]}")
    assert count == len(df), f"row count mismatch: inserted {count}, source had {len(df)}"
    assert [v[0] for v in versions] == list(range(15, 25)), "expected editions 15-24"

    conn.close()
    print("Schema built: player_ratings.")


if __name__ == "__main__":
    main()
