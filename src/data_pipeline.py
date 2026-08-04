from pathlib import Path
# Optional[str] is a type hint meaning "either a str or None" — used below to
# signal that a match may legitimately have no linkable tournament edition.
from typing import Optional
# import pandas (open-source software library for data analysis and data manipulation).
import pandas as pd

# create directories and subdirectories using "(__file__)" that auto-populates the path to the current file.
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
# add cutoff year for data pipeline.
CUTOFF_YEAR = 1990

# defining non_fifa_tournements
NON_FIFA_TOURNAMENTS = {
    "CONIFA Africa Football Cup", "CONIFA Asia Cup", "CONIFA European Football Cup",
    "CONIFA South America Football Cup", "CONIFA World Cup qualification",
    "CONIFA World Football Cup", "CONIFA World Football Cup qualification",
    "ConIFA Challenger Cup", "Viva World Cup",
}

# The matches table and the tournaments table name the same competition
# differently: a match row's `tournament` column says "FIFA World Cup", while
# the tournaments table's `competition` column says "World Cup". This map
# bridges the two. Only these two competitions have scraped squad data, so any
# match whose tournament label is NOT a key here has no linkable squad edition.
COMPETITION_LABEL_MAP = {
    "FIFA World Cup": "World Cup",
    "UEFA Euro": "Euro",
}

# Era-correct FIFA/FC edition for each ratings-covered tournament — the closest
# game snapshot dated BEFORE the tournament, so a rating is a genuine pre-match
# prediction, never hindsight. Mirrors link_player_ratings.py's map (the source
# of truth for how ratings were linked); see methodology "Rating data acquired".
# Only these 5 tournaments fall inside the ratings source's window (FIFA 15-24),
# so only they can carry a rating feature; every other match is NaN.
TOURNAMENT_TO_FIFA_VERSION = {
    "Euro 2016": 16,
    "World Cup 2018": 18,   # FIFA 19 released after the tournament — excluded
    "Euro 2020": 21,        # postponed to 2021; FIFA 21 was the current edition
    "World Cup 2022": 23,
    "Euro 2024": 24,        # FC 24
}

# Approximate start date of each in-scope tournament. Used only to timestamp when
# a player first became a *known* international for a country (the earliest squad
# they appear in) — the cutoff that keeps the pool-rating feature leakage-free
# (a match may only use squads known on or before its own date). Kept conservative
# (tournament start, not the ~3-week-earlier squad announcement).
TOURNAMENT_START_DATE = {
    "Euro 2016": pd.Timestamp("2016-06-10"),
    "World Cup 2018": pd.Timestamp("2018-06-14"),
    "Euro 2020": pd.Timestamp("2021-06-11"),   # COVID-postponed to 2021
    "World Cup 2022": pd.Timestamp("2022-11-20"),
    "Euro 2024": pd.Timestamp("2024-06-14"),
}


def fifa_edition_for_date(date: pd.Timestamp) -> Optional[int]:
    """
    The era-correct FIFA/FC edition for a match date: the latest edition released
    on or before it (each edition ships in late September, e.g. FIFA 16 = Sep 2015,
    FC 24 = Sep 2023). So a match is only ever rated against a game that already
    existed — never hindsight.

    Returns None before FIFA 15 (Sep 2014, the start of the ratings window) and
    clamps to 24 afterwards (FC 24 is the latest edition we hold — a 2025+ match
    falls back to it, stale but not leaked). Reproduces TOURNAMENT_TO_FIFA_VERSION
    exactly for the five in-scope tournaments (a nice consistency check).
    """
    # Editions release in September, so from September a year's new edition is out.
    version = (date.year - 1999) if date.month >= 9 else (date.year - 2000)
    if version < 15:
        return None
    return min(version, 24)


from src.database import get_connection

def load_raw_matches(source: str = None) -> pd.DataFrame:
    """
    Load match data from the SQLite database. The `source` parameter is
    kept for backward compatibility with existing calls but is no longer
    used — data now comes from the matches table, not a named CSV file.
    """
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM matches", conn)
    conn.close()
    return df


def load_former_names() -> pd.DataFrame:
    """Load the team name history lookup table from the database."""
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM former_names", conn)
    conn.close()
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    return df


def load_squads() -> pd.DataFrame:
    """
    Load one row per player-tournament appearance, joined across the
    players/squads/tournaments tables into a single flat DataFrame — same
    shape as load_raw_matches: squad-level feature functions filter this
    DataFrame themselves rather than each writing their own SQL join.
    """
    conn = get_connection()
    df = pd.read_sql(
        """
        SELECT
            s.country AS team,
            t.name AS tournament_name,
            p.position,
            p.age_at_tournament,
            p.club
        FROM players p
        JOIN squads s ON p.squad_id = s.squad_id
        JOIN tournaments t ON s.tournament_id = t.tournament_id
        """,
        conn,
    )
    conn.close()
    return df


def load_tournament_editions() -> dict:
    """
    Load a lookup of {(competition, year): tournament_name} from the
    tournaments table — e.g. {("World Cup", 2018): "World Cup 2018"}.

    This is the key that lets a match row — which only knows a generic
    competition label ("FIFA World Cup") plus a date — be tied to the
    specific tournament *edition* the squad features are scoped to. The
    caller builds this dict once and passes it into match_tournament_edition
    for every match row, so we hit the database only once, not per-row.
    """
    conn = get_connection()
    df = pd.read_sql("SELECT competition, year, name FROM tournaments", conn)
    conn.close()
    # A dict comprehension: one entry per tournament edition. int() guards
    # against the year arriving as a numpy int / str from the DB driver.
    return {
        (row["competition"], int(row["year"])): row["name"]
        for _, row in df.iterrows()
    }


def match_tournament_edition(
    tournament_label: str, match_date: pd.Timestamp, editions: dict
) -> Optional[str]:
    """
    Map one match row's generic tournament label + date to the specific
    tournament-edition name the squad features use (e.g. "World Cup 2018"),
    or None when the match has no linkable squad edition.

    Returns None (never a guess — honouring the never-fabricate rule) for:
      - any competition without scraped squads (friendlies, qualifiers,
        Copa América, ...): its label is not in COMPETITION_LABEL_MAP
      - editions with no squad data: pre-1990 tournaments (before the scrape
        range) and not-yet-played ones (e.g. World Cup 2026) — the (competition,
        year) key simply isn't present in `editions`.
    """
    # Step 1: translate the match's competition label to the tournaments-table
    # spelling. .get returns None for anything that isn't a major tournament.
    competition = COMPETITION_LABEL_MAP.get(tournament_label)
    if competition is None:
        return None

    year = match_date.year

    # Step 2: Euro 2020 edge case. That tournament was postponed by COVID and
    # actually played in mid-2021, but its tournaments-table row keeps the
    # official name/year "Euro 2020". Match rows are dated 2021, so without this
    # correction every Euro 2020 fixture would fail to find its edition.
    if competition == "Euro" and year == 2021:
        year = 2020

    # Step 3: look up the edition. .get returns None when no such edition exists
    # (e.g. a 1986 World Cup match, before the 1990 squad-scrape range).
    return editions.get((competition, year))


def load_squad_ratings() -> pd.DataFrame:
    """
    One row per rated squad player: team, tournament_name, overall, potential —
    each taken at the tournament's era-correct FIFA edition. Same flat shape as
    load_squads, so a feature function can filter it by (team, tournament_name).

    Only the 5 ratings-covered tournaments appear, and only players that were
    linked to a rating row (players.sofifa_player_id populated by
    link_player_ratings.py) contribute. A squad player with no linked rating is
    simply absent — the feature function then works from whoever WAS rated.
    """
    conn = get_connection()
    # linked squad players for the 5 in-scope tournaments (one row per appearance)
    players = pd.read_sql(
        """
        SELECT t.name AS tournament_name, s.country AS team, p.sofifa_player_id
        FROM players p
        JOIN squads s ON p.squad_id = s.squad_id
        JOIN tournaments t ON s.tournament_id = t.tournament_id
        WHERE p.sofifa_player_id IS NOT NULL
          AND t.name IN ({})
        """.format(",".join("?" * len(TOURNAMENT_TO_FIFA_VERSION))),
        conn, params=list(TOURNAMENT_TO_FIFA_VERSION),
    )
    # every edition present, so a single query pulls all the ratings we need
    ratings = pd.read_sql(
        "SELECT sofifa_player_id, fifa_version, overall, potential FROM player_ratings",
        conn,
    )
    conn.close()

    # tag each appearance with its tournament's era-correct edition, then merge
    # THAT exact (player, version) rating — so a WC 2018 player gets his FIFA 18
    # overall, not a later hindsight rating.
    players["fifa_version"] = players["tournament_name"].map(TOURNAMENT_TO_FIFA_VERSION)
    merged = players.merge(ratings, on=["sofifa_player_id", "fifa_version"], how="left")
    return merged[["team", "tournament_name", "overall", "potential"]]


def load_team_pool_ratings() -> pd.DataFrame:
    """
    Build the country player-POOL rating table that lets a rating be assigned to
    ANY international match in the FIFA-era window, not just the 5 tournaments with
    a scraped squad. One row per (country, player, FIFA edition):
        team, sofifa_player_id, first_seen, fifa_version, overall

    `first_seen` is the start date of the earliest tournament in which that player
    appeared for that country — i.e. the first moment we can honestly say "this
    player is one of C's internationals". A match may then use only the players
    whose `first_seen` is on or before its own date, which is what keeps the
    downstream feature leakage-free (no crediting a country with a player known
    only from a *later* squad).

    Reuses the existing 2,851 player→rating links; no new matching. A player's
    rating at every edition (15–24) is carried, so their pre/post-tournament form
    is available for matches played years either side of the squad we know them from.
    """
    conn = get_connection()
    # every linked appearance, tagged with the tournament it came from
    appearances = pd.read_sql(
        """
        SELECT s.country AS team, p.sofifa_player_id, t.name AS tournament_name
        FROM players p
        JOIN squads s ON p.squad_id = s.squad_id
        JOIN tournaments t ON s.tournament_id = t.tournament_id
        WHERE p.sofifa_player_id IS NOT NULL
          AND t.name IN ({})
        """.format(",".join("?" * len(TOURNAMENT_START_DATE))),
        conn, params=list(TOURNAMENT_START_DATE),
    )
    ratings = pd.read_sql(
        "SELECT sofifa_player_id, fifa_version, overall FROM player_ratings", conn)
    conn.close()

    # first_seen = earliest squad date per (country, player)
    appearances["squad_date"] = appearances["tournament_name"].map(TOURNAMENT_START_DATE)
    first_seen = (appearances.groupby(["team", "sofifa_player_id"])["squad_date"]
                  .min().reset_index().rename(columns={"squad_date": "first_seen"}))

    # attach every edition's overall for each player (inner join drops editions a
    # player has no rating in — e.g. before they existed in the game)
    pool = first_seen.merge(ratings, on="sofifa_player_id", how="inner")
    return pool[["team", "sofifa_player_id", "first_seen", "fifa_version", "overall"]]


def resolve_team_name(name: str, date: pd.Timestamp, former_names: pd.DataFrame) -> str:
    """
    Given a team name and a match date, return the *current* name if the
    given name was a historical alias in use on that date, otherwise
    return the name unchanged (it's already current, or has no history).
    """
    matches = former_names[
        (former_names["former"] == name)
        & (former_names["start_date"] <= date)
        & (former_names["end_date"] >= date)
    ]
    if len(matches) == 0:
        return name
    return matches.iloc[0]["current"]

## normalises the team names although completely useless for this database as it already uses the same names everywhere. 
def normalize_team_names(df: pd.DataFrame, former_names: pd.DataFrame) -> pd.DataFrame:
    """
    Replace historical team names with their current equivalents,
    for both home_team and away_team columns.
    """
    df = df.copy()
    # .apply(axis=1) runs a funcion once per row rather than once per column.
    # -> since resolve_team_name needs both team name and date simultaneously.
    df["home_team"] = df.apply(
        lambda row: resolve_team_name(row["home_team"], row["date"], former_names), axis=1
    )
    df["away_team"] = df.apply(
        lambda row: resolve_team_name(row["away_team"], row["date"], former_names), axis=1
    )
    return df

# Wire it into clean_matches itself, so normalisation happens as part of the standard cleaning pipeline. 
# -> rather than being a separate manual step.
def clean_matches(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].dt.year >= CUTOFF_YEAR]
    df = df.dropna(subset=["home_score", "away_score"])
    former_names = load_former_names()
    df = normalize_team_names(df, former_names)
    # .isin(...) checks each row's tournament value against the set, returns True/False per row
    df = df[~df["tournament"].isin(NON_FIFA_TOURNAMENTS)]
    return df


## setting up a match weighting system 
FIFA_MAJOR_FINALS = {
    "FIFA World Cup", "UEFA Euro", "Copa América", "African Cup of Nations",
    "AFC Asian Cup", "Gold Cup", "Oceania Nations Cup",
}
FIFA_MAJOR_QUALIFICATION = {t + " qualification" for t in FIFA_MAJOR_FINALS}

CONFEDERATION_SECOND_TIER = {
    "UEFA Nations League", "CONCACAF Nations League", "Confederations Cup",
    "CONMEBOL–UEFA Cup of Champions",
}

REGIONAL_CHAMPIONSHIPS = {
    "AFF Championship", "ASEAN Championship", "Arab Cup", "CECAFA Cup",
    "CFU Caribbean Cup", "COSAFA Cup", "EAFF Championship", "Gulf Cup",
    "NAFU Championship", "SAFF Cup", "UNCAF Cup", "UNIFFAC Cup",
    "WAFF Championship", "Baltic Cup", "Nordic Championship",
    "AFC Challenge Cup", "AFC Solidarity Cup",
}
REGIONAL_QUALIFICATION = {t + " qualification" for t in REGIONAL_CHAMPIONSHIPS}

MULTISPORT_GAMES = {
    "Asian Games", "Pacific Games", "Pacific Mini Games", "South Pacific Games",
    "South Pacific Mini Games", "Southeast Asian Games", "South Asian Games",
    "East Asian Games", "Afro-Asian Games", "Island Games",
    "Indian Ocean Island Games",
}


def importance_weight(tournament: str) -> float:
    """
    Map a tournament name to an importance multiplier, using explicit
    allowlists per tier rather than keyword matching — this dataset has
    149 distinct tournament values including several near-duplicate names
    (e.g. "Nations Cup" vs "Four Nations' Cup" vs "Tri-Nations Cup"), which
    makes substring rules unreliable.
    """
    if tournament == "Friendly":
        return 0.3
    if tournament in FIFA_MAJOR_FINALS:
        return 1.0
    if tournament in FIFA_MAJOR_QUALIFICATION:
        return 0.6
    if tournament in CONFEDERATION_SECOND_TIER:
        return 0.55
    if tournament in REGIONAL_CHAMPIONSHIPS:
        return 0.45
    if tournament in REGIONAL_QUALIFICATION:
        return 0.35
    if tournament in MULTISPORT_GAMES:
        return 0.2

    return 0.25  # long tail: obscure invitationals — Merlion Cup, King's Cup, etc.

# adding the recency weight
def recency_weight(
    match_date: pd.Timestamp,
    reference_date: pd.Timestamp,
    half_life_days: int,
    # ad a minimum weight cap to the exponentiality to prevent weightings like 0.000002
    min_weight: float = 0.05,
) -> float:
    """
    Exponential recency decay, floored at min_weight. Without a floor,
    matches near the 1990 cutoff decay to near-zero regardless of the
    chosen half-life, which silently contradicts the earlier decision
    that 1990-onward data is statistically useful and worth including.
    """
    days_ago = (reference_date - match_date).days
    weight = 0.5 ** (days_ago / half_life_days)
    return max(weight, min_weight)

# combining both recency and importance weights into one comeplete weighting system. 
def add_match_weights(
    df: pd.DataFrame, reference_date: pd.Timestamp, half_life_days: int
) -> pd.DataFrame:
    """
    Add a match_weight column. Matches after reference_date are dropped —
    weighting a match that hasn't happened yet relative to your prediction
    point is meaningless, not just numerically unstable.
    """
    df = df.copy()
    df = df[df["date"] <= reference_date]
    df["match_weight"] = df.apply(
        lambda row: recency_weight(row["date"], reference_date, half_life_days)
        * importance_weight(row["tournament"]),
        axis=1,
    )
    return df





import requests

def fetch_market_value_history(player_id: str) -> list[dict]:
    """
    Fetch a player's full market-value-over-time history from
    Transfermarkt's internal (undocumented) API.

    Note: this hits an internal endpoint discovered via reverse-engineering,
    not a published API. No formal rate limit is documented, so this
    deliberately does not batch-call without a delay — see caller code for
    sleep/backoff when looping over many players.
    """
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    url = f"https://www.transfermarkt.com/ceapi/marketValueDevelopment/graph/{player_id}"
    resp = requests.get(url, headers=headers)
    # help with error messages
    resp.raise_for_status()
    return resp.json()["list"]



import re
from bs4 import BeautifulSoup


def search_player_id(name: str) -> list[dict]:
    """
    Search Transfermarkt's internal quick-search endpoint for a player
    name. Returns a list of candidate matches (player_id + display name)
    rather than a single "best guess" — common surnames return multiple
    results, and silently picking the first one risks matching the wrong
    player entirely.
    """
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    resp = requests.get(
        "https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche",
        params={"query": name},
        headers=headers,
    )
    resp.raise_for_status()
    # html parser
    soup = BeautifulSoup(resp.text, "html.parser")

    seen_ids = set()
    results = []
    # soup.select("a[href*='/profil/spieler/']") is a CSS selector
    for link in soup.select("a[href*='/profil/spieler/']"):
        match = re.search(r"/profil/spieler/(\d+)", link.get("href", ""))
        # link.get_text(strip=True) can pull the visible text 
        display_name = link.get_text(strip=True)
        if not match or not display_name:
            continue
        player_id = match.group(1)
        # seen_ids is our deduplication
        if player_id in seen_ids:
            continue
        seen_ids.add(player_id)
        results.append({"player_id": player_id, "name": display_name})

    return results