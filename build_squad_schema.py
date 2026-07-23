import re
import sqlite3
from datetime import datetime

import pandas as pd
from src.database import get_connection

DOB_PATTERN = re.compile(r"^(.*?)\s*\(aged\s*(\d+)\)\s*$") # The regex
# re.compile breaks the wikipedia string into pieces
# ^ = start of string, $ = end of string
# . means "any character", * means "0 or more than",
#  ? makes it grab as few characters as possible    

TOURNAMENTS = [
    (1990, "World Cup", "World Cup 1990", "1990_FIFA_World_Cup_squads"),
    (1994, "World Cup", "World Cup 1994", "1994_FIFA_World_Cup_squads"),
    (1998, "World Cup", "World Cup 1998", "1998_FIFA_World_Cup_squads"),
    (2002, "World Cup", "World Cup 2002", "2002_FIFA_World_Cup_squads"),
    (2006, "World Cup", "World Cup 2006", "2006_FIFA_World_Cup_squads"),
    (2010, "World Cup", "World Cup 2010", "2010_FIFA_World_Cup_squads"),
    (2014, "World Cup", "World Cup 2014", "2014_FIFA_World_Cup_squads"),
    (2018, "World Cup", "World Cup 2018", "2018_FIFA_World_Cup_squads"),
    (2022, "World Cup", "World Cup 2022", "2022_FIFA_World_Cup_squads"),
    (1992, "Euro", "Euro 1992", "UEFA_Euro_1992_squads"),
    (1996, "Euro", "Euro 1996", "UEFA_Euro_1996_squads"),
    (2000, "Euro", "Euro 2000", "UEFA_Euro_2000_squads"),
    (2004, "Euro", "Euro 2004", "UEFA_Euro_2004_squads"),
    (2008, "Euro", "Euro 2008", "UEFA_Euro_2008_squads"),
    (2012, "Euro", "Euro 2012", "UEFA_Euro_2012_squads"),
    (2016, "Euro", "Euro 2016", "UEFA_Euro_2016_squads"),
    (2020, "Euro", "Euro 2020", "UEFA_Euro_2020_squads"),
    (2024, "Euro", "Euro 2024", "UEFA_Euro_2024_squads"),
]


def parse_dob_age(raw):
    """Parse "8 February 1995 (aged 29)" into (iso_date_or_None, age_or_None)."""
    if not isinstance(raw, str):
        return None, None
    match = DOB_PATTERN.match(raw)
    if not match:
        return None, None
    date_str, age_str = match.group(1), match.group(2)
    try:
        dob = datetime.strptime(date_str, "%d %B %Y").date().isoformat() # turning that regex match into real data
    except ValueError:
        dob = None
    return dob, int(age_str)


def clean_player_name(raw):
    """Strip footnote markers, extract captain status into its own flag."""
    if not isinstance(raw, str):  # guards against None/NaN player names instead of crashing
        return None, False
    is_captain = "(captain)" in raw
    name = raw.replace("(captain)", "").strip()
    name = re.sub(r"[\*\u2020]+$", "", name).strip()
    return name, is_captain


def safe_int(value): # a small defensive wrapper
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def main():
    conn = get_connection()

    # read the existing flat table BEFORE dropping anything
    flat = pd.read_sql("SELECT * FROM squads", conn)
    flat.to_csv("squads_flat_backup.csv", index=False)  
    # safety net independent of the DB and this process's memory

    ## The schema itself: executescript() and the three CREATE TABLE statements
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript("""
        DROP TABLE IF EXISTS players;
        DROP TABLE IF EXISTS squads;
        DROP TABLE IF EXISTS tournaments;

        CREATE TABLE tournaments (
            tournament_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            year INTEGER NOT NULL,
            competition TEXT NOT NULL,
            wiki_page TEXT NOT NULL
        );

        CREATE TABLE squads (
            squad_id INTEGER PRIMARY KEY AUTOINCREMENT,
            tournament_id INTEGER NOT NULL,
            country TEXT NOT NULL,
            FOREIGN KEY (tournament_id) REFERENCES tournaments(tournament_id)
        );

        CREATE TABLE players (
            player_appearance_id INTEGER PRIMARY KEY AUTOINCREMENT,
            squad_id INTEGER NOT NULL,
            squad_number INTEGER,
            position TEXT,
            player_name TEXT NOT NULL,
            is_captain BOOLEAN NOT NULL DEFAULT 0,
            date_of_birth DATE,
            age_at_tournament INTEGER,
            caps INTEGER,
            goals INTEGER,
            club TEXT,
            transfermarkt_player_id TEXT,
            FOREIGN KEY (squad_id) REFERENCES squads(squad_id)
        );
    """)

    cur = conn.cursor()

    tournament_ids = {}
    for year, competition, name, wiki_page in TOURNAMENTS:
        cur.execute(
            "INSERT INTO tournaments (name, year, competition, wiki_page) VALUES (?, ?, ?, ?)",
            (name, year, competition, wiki_page),
        )
        tournament_ids[name] = cur.lastrowid

    squad_ids = {}
    for (tournament_name, country), _ in flat.groupby(["tournament", "country"]):
        t_id = tournament_ids[tournament_name]
        cur.execute(
            "INSERT INTO squads (tournament_id, country) VALUES (?, ?)", (t_id, country)
        )
        squad_ids[(tournament_name, country)] = cur.lastrowid

    skipped = 0
    for _, row in flat.iterrows():
        name, is_captain = clean_player_name(row["Player"])
        if name is None:
            skipped += 1
            continue
        s_id = squad_ids[(row["tournament"], row["country"])]
        dob, age = parse_dob_age(row["Date of birth (age)"])
        cur.execute(
            """INSERT INTO players
               (squad_id, squad_number, position, player_name, is_captain,
                date_of_birth, age_at_tournament, caps, goals, club)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                s_id, safe_int(row["No."]), row["Pos."], name, is_captain,
                dob, age, safe_int(row.get("Caps")), safe_int(row.get("Goals")),
                row["Club"],
            ),
        )
    print(f"Skipped {skipped} rows with no player name.")

    conn.commit()
    conn.close()
    print("Schema built: tournaments, squads, players.")


if __name__ == "__main__":
    main()