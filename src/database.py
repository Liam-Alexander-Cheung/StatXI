import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "statxi.db"


def get_connection() -> sqlite3.Connection:
    """Open a connection to the project's SQLite database."""
    return sqlite3.connect(DB_PATH)