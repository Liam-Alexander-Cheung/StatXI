from pathlib import Path
# import pandas (open-source software library for data analysis and data manipulation).
import pandas as pd

# create directories and subdirectories using "(__file__)" that auto-populates the path to the current file.
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
# add cutoff year for data pipeline.
CUTOFF_YEAR = 1990


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
    return df

