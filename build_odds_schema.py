"""
Build the `match_odds` table from raw bookmaker-odds export(s) in data/raw/.

SOURCE. Historical 1X2 (home/draw/away) odds for international matches, from
Oddsportal (see reports/methodology.md for why: no free, reproducible
international-odds dataset exists — every Kaggle/football-data.co.uk odds set
is club-league only, The Odds API's international history is paid and only
reaches back to WC 2022, and Oddsportal's own feed is AES-encrypted behind
obfuscated JS, so an automated scrape is both against the reproducibility bar
and technically blocked in this environment).

HOW THE EXPORT WORKS, AND WHY IT'S A "FLAT PASTE". Oddsportal's results table
isn't real HTML `<table>` markup, so a plain browser select-and-copy off the
results page (Football > region > tournament > Results tab) doesn't paste into
spreadsheet columns — it dumps every cell as its OWN LINE, in strict reading
order, one value per row. That turns out to be reliably parseable because the
DOM order per match is fixed:
    "15 Jul 2024 - Play Offs"   <- date + stage (only on the FIRST match of a day)
    "1"                         <- \\
    "X"                         <-  | column-header labels, repeated per date group
    "2"                         <- /
    "Finished"                  <- match status ("After Pen." / "After ET" too)
    "Spain"                     <- home team
    "2"                         <- home score
    "–"                         <- EN DASH score separator (not a hyphen)
    "1"                         <- away score
    "England"                   <- away team
    "140"                       <- odds_home  (American; a bare positive value
    "179"                       <-    odds_draw   has its "+" stripped by the
    "260"                       <- odds_away      paste — see american_to_decimal)
A same-day second match skips the date+header lines and goes straight into
another 9-line block. `_parse_flat_paste` below is the state machine that
reverses this back into rows. Save the raw paste as .csv in data/raw/ (any
name containing "odds") and this script auto-detects the flat shape (its first
non-blank line matches a date pattern) vs. a conventional structured CSV.

WHAT IT STORES. Raw decimal odds and the team names *exactly as the source
wrote them* (`home_team_raw` / `away_team_raw`). The name crosswalk to our
canonical DB names, the de-vig, and the home/away re-orientation all happen
later, at load time, in src/odds.py — this script's only job is to get the raw
numbers into the database with real types (never a fabricated placeholder),
same discipline as build_ratings_schema.py. Scores are NOT trusted from the
source (unreliable for penalty/extra-time games) — our matches table is the
authoritative score.

STRUCTURED CSV FALLBACK (if a file's first line is a real header, not a date):
    date,home_team,away_team,odds_home,odds_draw,odds_away,competition
    2016-06-10,France,Romania,1.44,4.20,8.50,Euro 2016
"""

import re
from pathlib import Path
import glob

import pandas as pd

from src.database import get_connection
from src.odds import american_to_decimal

# Any CSV in data/raw/ whose name contains "odds" is ingested — so a per-
# tournament export (euro2016_odds.csv, wc2022_odds.csv, ...) or a single
# combined international_odds.csv both work with no code change.
RAW_DIR = Path(__file__).resolve().parent / "data" / "raw"
ODDS_GLOB = str(RAW_DIR / "*odds*.csv")

# Columns required by the STRUCTURED-CSV fallback path only.
REQUIRED_COLUMNS = ["date", "home_team", "away_team",
                    "odds_home", "odds_draw", "odds_away"]
OPTIONAL_COLUMNS = ["competition", "stage", "home_score", "away_score"]
FINAL_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS + ["source"]

# A flat-paste date line: "15 Jul 2024" or "15 Jul 2024 - Play Offs".
_DATE_LINE_RE = re.compile(r"^\d{1,2} [A-Za-z]{3} \d{4}(\s*-\s*(.+))?$")


def safe_float(value):
    """NaN-safe float conversion — pandas stores missing numbers as NaN; we
    persist a real NULL instead of a mangled value (same helper as
    build_ratings_schema.py)."""
    if pd.isna(value):
        return None
    return float(value)


def safe_str(value):
    if pd.isna(value):
        return None
    return str(value).strip()


def _parse_flat_paste(path: str) -> pd.DataFrame:
    """
    Reverse a flat, one-value-per-line Oddsportal copy-paste back into rows
    (see the module docstring for the exact 9-line match / date-group shape).
    Raises AssertionError immediately if a block doesn't match the expected
    shape — a loud failure here is far better than silently misaligning odds
    to the wrong team again (see reports/methodology.md for the earlier
    CSS-class table-scraper export that DID silently misalign, undetected
    until a favourite-accuracy sanity check caught it).
    """
    lines = [l.strip() for f in [path] for l in open(f, encoding="utf-8")]
    lines = [l for l in lines if l != ""]

    rows = []
    i, n = 0, len(lines)
    current_date = None
    while i < n:
        m = _DATE_LINE_RE.match(lines[i])
        if m:
            current_date = lines[i].split(" - ")[0].strip()
            current_stage = m.group(2)
            i += 1
            assert lines[i:i + 3] == ["1", "X", "2"], (
                f"{path}: expected the '1'/'X'/'2' header right after a date "
                f"line at position {i}, got {lines[i:i + 3]!r} instead — the "
                f"paste shape has changed; fix _parse_flat_paste before trusting it."
            )
            i += 3
            continue

        assert current_date is not None, (
            f"{path}: hit a match block before any date line was seen "
            f"(line {i}: {lines[i]!r})."
        )
        # A match block is variable-length. Normal (played) match, 9 lines:
        #     status, home, HOME_SCORE, "–", AWAY_SCORE, away, o1, oX, o2
        # A no-score match (e.g. the abandoned 2022 Brazil–Argentina WCQ, status
        # "Cancelled"), 7 lines — the "–" sits right after the home team with no
        # scores around it, and the odds are placeholder "-"s:
        #     status, home, "–", away, "-", "-", "-"
        # So the en dash position (i+2 vs i+3) tells the two apart. We locate it
        # rather than assume a fixed stride, so an odd row can never silently
        # shift every following match by a line.
        status, home = lines[i], lines[i + 1]
        if lines[i + 2] == "–":                         # no-score / cancelled match
            away = lines[i + 3]
            oh, od, oa = lines[i + 4], lines[i + 5], lines[i + 6]
            i += 7
        else:                                          # normal played match
            assert lines[i + 3] == "–", (
                f"{path}: expected an EN DASH ('–') score separator at "
                f"position {i + 2} or {i + 3}, got {lines[i + 2:i + 4]!r} — "
                f"block starts {lines[i:i + 9]!r}"
            )
            away = lines[i + 5]
            oh, od, oa = lines[i + 6], lines[i + 7], lines[i + 8]
            i += 9
        rows.append({
            "date": current_date, "home_team": home, "away_team": away,
            "odds_home": oh, "odds_draw": od, "odds_away": oa,
            "competition": pd.NA, "stage": current_stage,
            "home_score": pd.NA, "away_score": pd.NA,   # not trusted from source
        })

    out = pd.DataFrame(rows, columns=FINAL_COLUMNS[:-1])  # all but "source"
    parsed = pd.to_datetime(out["date"], format="%d %b %Y", errors="coerce")
    dropped = parsed.isna().sum()
    if dropped:
        print(f"  {path}: dropped {dropped} row(s) with an unparseable date")
    out = out[parsed.notna()].copy()
    out["date"] = parsed[parsed.notna()].dt.strftime("%Y-%m-%d")
    out["odds_home"] = out["odds_home"].map(american_to_decimal)
    out["odds_draw"] = out["odds_draw"].map(american_to_decimal)
    out["odds_away"] = out["odds_away"].map(american_to_decimal)
    return out


def _is_flat_paste(path: str) -> bool:
    """A flat paste's first non-blank line is a date; a structured CSV's is a
    header row (`date,home_team,...`), which never matches the date pattern."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return bool(_DATE_LINE_RE.match(line))
    return False


def _normalize_export(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise ONE raw Oddsportal export into the clean column set.

    The table-scraper writes the match date only on the FIRST row of each
    date-group (e.g. ``"15 Jul 2024  - Play Offs"``) and leaves it blank on the
    other matches that day; it also appends the round label after a dash, and
    writes odds in whatever format the account is set to (American here). This
    function, applied PER FILE (so a forward-fill never bleeds one file's last
    date into the next file's first rows):

      1. splits the date cell into the date and the trailing stage label,
      2. forward-fills both down each date-group,
      3. parses ``"DD Mon YYYY"`` -> ISO ``YYYY-MM-DD``,
      4. converts the three odds columns to decimal (see american_to_decimal),
      5. keeps only the canonical columns; every scraper junk column is dropped.

    Rows whose date cannot be parsed are dropped (never stored with a fabricated
    date), with a count printed.
    """
    out = pd.DataFrame()
    raw_date = df["date"].fillna("").astype(str).str.strip()

    # 1-2. stage = text after the dash; datepart = text before it. Both blank on
    #      the group's non-header rows, so forward-fill fills them in.
    stage = raw_date.str.extract(r"-\s*(.+?)\s*$")[0]
    datepart = raw_date.str.replace(r"\s*-\s*.+$", "", regex=True).str.strip()
    datepart = datepart.replace("", pd.NA).ffill()
    stage = stage.ffill()

    # 3. parse to ISO; unparseable -> NaT -> dropped below.
    parsed = pd.to_datetime(datepart, format="%d %b %Y", errors="coerce")
    out["date"] = parsed.dt.strftime("%Y-%m-%d")
    out["home_team"] = df["home_team"].astype(str).str.strip()
    out["away_team"] = df["away_team"].astype(str).str.strip()

    # 4. odds -> decimal (American "+140"/"-175" or already-decimal both handled).
    out["odds_home"] = df["odds_home"].map(american_to_decimal)
    out["odds_draw"] = df["odds_draw"].map(american_to_decimal)
    out["odds_away"] = df["odds_away"].map(american_to_decimal)

    out["competition"] = df["competition"] if "competition" in df.columns else pd.NA
    out["stage"] = stage
    # scores deliberately NOT taken from the source (unreliable for pen/ET games)
    out["home_score"] = pd.NA
    out["away_score"] = pd.NA

    dropped = out["date"].isna().sum()
    if dropped:
        print(f"  dropped {dropped} row(s) with an unparseable date")
    out = out.dropna(subset=["date", "home_team", "away_team"])
    return out


def load_odds_csvs(pattern: str = ODDS_GLOB) -> pd.DataFrame:
    """Read, normalise, and concatenate every odds CSV matching `pattern`,
    tagging each row with the file it came from (`source`). De-duplicates on
    (date, home_team, away_team) — if the same match appears in two exports, the
    first occurrence wins, so re-grabbing a tournament can't double-count it.

    Returns an EMPTY DataFrame (not an error) when no files match, so the caller
    can print a helpful "export the odds first" message rather than crash.
    """
    files = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame()

    frames = []
    for f in files:
        if _is_flat_paste(f):
            norm = _parse_flat_paste(f)
        else:
            # dtype=str: keep "+140"/"1.44" verbatim so american_to_decimal,
            # not pandas' number guesser, decides how to read each odd.
            df = pd.read_csv(f, dtype=str)
            missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
            if missing:
                raise ValueError(f"{f} is missing required column(s): {missing}. "
                                 f"Expected at least: {REQUIRED_COLUMNS}.")
            norm = _normalize_export(df)      # per-file (ffill must not cross files)
        norm["source"] = Path(f).name
        frames.append(norm[FINAL_COLUMNS])

    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(
        subset=["date", "home_team", "away_team"], keep="first"
    ).reset_index(drop=True)
    if before != len(combined):
        print(f"De-duplicated {before - len(combined)} repeated match row(s) "
              f"across {len(files)} file(s).")
    return combined


def build(pattern: str = ODDS_GLOB) -> int:
    """Create/replace the `match_odds` table from the matching CSV(s).
    Returns the number of rows inserted (0 if no source files were found)."""
    df = load_odds_csvs(pattern)
    if df.empty:
        print(f"No odds CSVs found matching {pattern}.")
        print("Export Oddsportal results to data/raw/international_odds.csv "
              "(see build_odds_schema.py docstring for the exact format), then "
              "re-run:  python build_odds_schema.py")
        return 0

    conn = get_connection()

    # Backup-before-DROP safety net (same precedent as build_ratings_schema.py):
    # protects any future in-DB-only work on this table across a re-run.
    existing = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='match_odds'"
    ).fetchone()
    if existing:
        backup = pd.read_sql("SELECT * FROM match_odds", conn)
        backup.to_csv("match_odds_backup.csv", index=False)
        print(f"Backed up {len(backup)} existing match_odds rows to match_odds_backup.csv")

    conn.executescript("""
        DROP TABLE IF EXISTS match_odds;

        CREATE TABLE match_odds (
            odds_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            match_date    DATE NOT NULL,   -- ISO YYYY-MM-DD, AS SCRAPED (may be
                                           -- +/-1 day off ours; resolved by the
                                           -- tolerant join in src/odds.py)
            home_team_raw TEXT NOT NULL,   -- team names AS THE SOURCE WROTE THEM;
            away_team_raw TEXT NOT NULL,   -- crosswalk to canonical happens in src/odds.py
            odds_home     REAL,            -- decimal 1 / X / 2 odds (raw, still carry the vig)
            odds_draw     REAL,
            odds_away     REAL,
            competition   TEXT,            -- optional source label (often NULL)
            stage         TEXT,            -- round label parsed from the date cell
            home_score    REAL,            -- NULL: scores come from our matches DB
            away_score    REAL,
            source        TEXT NOT NULL,   -- which raw file this row came from
            UNIQUE (match_date, home_team_raw, away_team_raw)
        );
    """)

    rows = [
        (
            safe_str(r.date), safe_str(r.home_team), safe_str(r.away_team),
            safe_float(r.odds_home), safe_float(r.odds_draw), safe_float(r.odds_away),
            safe_str(r.competition), safe_str(r.stage),
            safe_float(r.home_score), safe_float(r.away_score),
            safe_str(r.source),
        )
        for r in df.itertuples(index=False)
    ]
    conn.executemany(
        """INSERT INTO match_odds
           (match_date, home_team_raw, away_team_raw,
            odds_home, odds_draw, odds_away, competition, stage,
            home_score, away_score, source)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()

    # Verify — don't just trust "it ran". Confirm the round-trip count and show
    # coverage per source file + how many odds rows are unusable (odd <= 1.0,
    # which de-vig will reject), so a thin or dirty export is visible immediately.
    count = conn.execute("SELECT COUNT(*) FROM match_odds").fetchone()[0]
    assert count == len(df), f"row count mismatch: inserted {count}, source had {len(df)}"
    per_src = conn.execute(
        "SELECT source, COUNT(*) FROM match_odds GROUP BY source ORDER BY COUNT(*) DESC"
    ).fetchall()
    bad = conn.execute(
        "SELECT COUNT(*) FROM match_odds WHERE odds_home <= 1.0 OR odds_draw <= 1.0 "
        "OR odds_away <= 1.0 OR odds_home IS NULL OR odds_draw IS NULL OR odds_away IS NULL"
    ).fetchone()[0]
    span = conn.execute("SELECT MIN(match_date), MAX(match_date) FROM match_odds").fetchone()
    print(f"Loaded {count} match_odds rows spanning {span[0]} .. {span[1]}.")
    print("coverage per source file:")
    for src, n in per_src:
        print(f"  {src:<34}{n:>5}")
    if bad:
        print(f"note: {bad} row(s) have an invalid/missing odd (<=1.0) and will "
              f"de-vig to NaN (excluded from the comparison, never guessed).")

    conn.close()
    print("Schema built: match_odds.")
    return count


if __name__ == "__main__":
    build()
