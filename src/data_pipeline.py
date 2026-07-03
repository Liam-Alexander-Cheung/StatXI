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

