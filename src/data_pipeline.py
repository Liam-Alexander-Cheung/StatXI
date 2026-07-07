from pathlib import Path
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


def load_raw_matches(source: str) -> pd.DataFrame:
    """
    Load a raw match-results CSV from data/raw/.
    """
    # creates a filesystem path.
    path = RAW_DIR / source
    # "pd.read_csv(path)" is pandas' CSV reader.
    return pd.read_csv(path)

def clean_matches(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    #only takes games past the cutoff date.
    df = df[df["date"].dt.year >= CUTOFF_YEAR]
    # .dropna() removes rows containing missing values. 
    # subsets=[...] restricts the check to just the two columns.
    df = df.dropna(subset=["home_score", "away_score"])
    return df

# loads former names
def load_former_names() -> pd.DataFrame:
    """Load the team name history lookup table."""
    df = pd.read_csv(RAW_DIR / "former_names.csv")
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    return df

def resolve_team_name(name: str, date: pd.Timestamp, former_names: pd.DataFrame) -> str:
    """
    Given a team name and a match date, return the *current* name if the
    given name was a historical alias in use on that date, otherwise
    return the name unchanged (it's already current, or has no history).
    """
    matches = former_names[
        # is this a country that has historical aliases?
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