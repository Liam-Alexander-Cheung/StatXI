"""
Bookmaker odds -> de-vigged 3-way probability vectors, and loading them aligned
to our match data.

WHY THIS EXISTS
The project's headline claim (agents.md) compares the WDL model against two
references: a naive baseline (already built) and *bookmaker implied
probabilities* (this file). A bookmaker's decimal odds encode an implied
probability, but they are inflated by the bookmaker's margin ("the vig" /
"overround") so that the three implied probabilities sum to MORE than 1 — that
excess is the bookmaker's built-in profit. To use the odds as an honest
probability forecast we must strip that margin back out ("de-vig") so the three
numbers sum to exactly 1. This module does that, and loads the odds already
oriented to our DB's home/away ordering.

This file is intentionally split into small, independently-verifiable pieces,
built one at a time (project rule):
  1. `devig()`            — the pure maths. Built + verified first (below).
  2. team-name crosswalk  — map Oddsportal spellings -> our canonical names.
  3. `load_match_odds()`  — read the `match_odds` table, de-vig, orient to DB
                            home/away, return one row per covered match.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from src.database import get_connection
from src.name_matching import name_similarity


# ---------------------------------------------------------------------------
# Team-name crosswalk: Oddsportal spelling -> our canonical DB team name.
#
# Built from REAL data (project rule: verify, don't guess). Across the first
# export only two names differed from our DB, and one of them is a genuine trap:
#   * "Bosnia & Herzegovina" — just "&" vs "and"; same country, safe.
#   * "Ireland" — Oddsportal's name for the REPUBLIC of Ireland. A fuzzy match
#     scores it 1.00 against "Northern Ireland" (substring), which would be
#     WRONG. It was resolved by checking who "Ireland" actually played
#     (France/Greece/Gibraltar/Netherlands = Republic of Ireland's real group).
# This is why we never auto-accept a fuzzy country match: an unrecognised name
# goes to a human review queue, never to a silent best-guess.
# ---------------------------------------------------------------------------
_TEAM_ALIASES = {
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Ireland": "Republic of Ireland",          # the trap: NOT Northern Ireland
    # Added once World Cup 2022/2026 qualifiers (global fields) were ingested.
    # Each verified: the canonical spelling on the right exists in the matches
    # table, and the tolerant team-pair join then confirms the mapping by only
    # matching if that team actually played those opponents on those dates.
    "USA": "United States",
    "D.R. Congo": "DR Congo",                  # fuzzy wrongly suggests "Congo"
    "Guinea Bissau": "Guinea-Bissau",          # fuzzy wrongly suggests "Guinea"
    "Central Africa": "Central African Republic",
    "Chinese Taipei": "Taiwan",
    "Curacao": "Curaçao",
    "Sao Tome and Principe": "São Tomé and Príncipe",
    "Trinidad & Tobago": "Trinidad and Tobago",
}

# How many days apart an odds row and a DB match may be and still be the same
# fixture. Oddsportal timestamps in the viewer's timezone, so its dates run ~1
# day off ours; 3 days is a safe cushion that still can't confuse the two legs
# of a home-and-away qualifier (those are months apart).
DATE_TOLERANCE_DAYS = 3

# Where the human-review queues land — gitignored derived data, same policy and
# precedent as ratings_match_review_queue.csv.
_QUEUE_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def canonical_team(name: str, db_names: set[str]) -> str | None:
    """Map an Oddsportal team name to our canonical DB name, or return None if
    it can't be resolved with confidence (caller queues it for human review).

    Resolution order — explicit and conservative:
      1. a hand-verified alias (`_TEAM_ALIASES`),
      2. an exact match to an existing DB name,
      3. otherwise unresolved -> None. We deliberately do NOT auto-accept a
         fuzzy match (the "Ireland" trap above shows a 1.00 fuzzy score can be
         the wrong country); fuzzy is only used to *suggest* in the review queue.
    """
    if name in _TEAM_ALIASES:
        return _TEAM_ALIASES[name]
    if name in db_names:
        return name
    return None


def devig(odds_home: float, odds_draw: float, odds_away: float
          ) -> tuple[float, float, float]:
    """
    Convert decimal 1X2 odds into a de-vigged (margin-free) probability vector
    ``(p_home, p_draw, p_away)`` that sums to 1.

    The maths, in two steps:

      1. **Implied probability of a decimal odd = 1 / odd.** Decimal odds are
         "total return per unit staked": odds of 4.0 pay 4 units back (3 profit
         + your 1 stake) for a win, which is the fair price of a 1/4 = 25%
         chance. So the raw implied probabilities are 1/odds_home, 1/odds_draw,
         1/odds_away.

      2. **Normalise out the overround.** Because of the vig, those three raw
         implied probabilities sum to some total > 1 (e.g. 1.06 => a 6% margin).
         Dividing each by that total rescales them to sum to exactly 1. This is
         the "multiplicative"/"basic normalisation" de-vig — the textbook
         default. It assumes the bookmaker spreads the margin proportionally
         across the three outcomes. (A more sophisticated alternative, Shin's
         method, models the margin as coming disproportionately from
         insider-driven favourites; it is documented in methodology.md as a
         considered-but-not-implemented option, since normalisation is the
         standard, defensible baseline for this project.)

    Missing or invalid odds (NaN, or any odd <= 1.0, which is impossible for a
    real market since a decimal odd of 1.0 implies a certain outcome with zero
    payout) return ``(nan, nan, nan)`` — never a fabricated guess, per the
    project's no-fabrication rule. Callers treat all-NaN as "no odds for this
    match" and exclude it from the comparison.
    """
    odds = (odds_home, odds_draw, odds_away)

    # Reject anything that isn't a usable decimal odd. `x != x` is the standard
    # NaN test (NaN is the only value not equal to itself). A real 1X2 market
    # always has each odd strictly above 1.0.
    for x in odds:
        if x is None or (isinstance(x, float) and math.isnan(x)) or x <= 1.0:
            return (float("nan"), float("nan"), float("nan"))

    # Step 1: raw implied probabilities (still carry the vig, so they sum > 1).
    inv = [1.0 / x for x in odds]

    # Step 2: normalise so the vector sums to exactly 1.
    total = sum(inv)
    return (inv[0] / total, inv[1] / total, inv[2] / total)


def american_to_decimal(value) -> float:
    """
    Convert an odds value to DECIMAL format, accepting either American/moneyline
    (``"+140"`` / ``"-175"`` / a bare ``"140"`` meaning ``"+140"``) or an
    already-decimal value (``"1.44"`` / 1.44).

    Why this exists: Oddsportal exports in whatever odds format the user's
    account is set to, and both real exports so far came back in American
    format. Rather than force a re-export, we normalise to decimal here so the
    rest of the pipeline only ever sees decimal odds. The conversion is exact:

      * American **positive** ``+A``  ->  ``1 + A/100``   (e.g. +140 -> 2.40:
        a $100 stake profits $140, total return $240 on $100 = 2.40 decimal).
      * American **negative** ``-A``  ->  ``1 + 100/A``   (e.g. -175 -> 1.5714:
        you must stake $175 to profit $100, total $275 on $175 = 1.5714).

    DISAMBIGUATING a bare, unsigned integer (``"260"``) — real, caught bug: a
    browser copy-paste can drop the leading "+" off positive American values
    while keeping "-" on negative ones, so an early version of this function
    treated a signless "260" as an already-decimal 260.0 (an impossible price)
    and silently crushed every favourite's implied probability toward 1.0.
    The fix relies on a fact true of every odds format we've seen: DECIMAL odds
    always carry a fractional point (``"1.44"``, never bare ``"144"``), while
    AMERICAN odds are always whole numbers with |value| >= 100 (the -99..+99
    band doesn't exist in moneyline pricing). So:
      * contains "."      -> decimal, parsed as-is.
      * signed integer     -> American with that sign (existing behaviour).
      * unsigned integer   -> American, treated as an implicit "+" (the only
        way a whole, sub-1000ish number could denote a real market price).

    Anything unparseable, blank, NaN, or zero returns ``nan`` (never a guess).
    """
    if value is None:
        return float("nan")
    s = str(value).strip().replace(",", "")   # tolerate "+1,200"
    if s == "" or s.lower() == "nan":
        return float("nan")
    try:
        if "." in s:
            return float(s)                   # decimal odds always have a point
        sign, digits = (s[0], s[1:]) if s[0] in "+-" else ("+", s)
        if not digits.isdigit():
            return float("nan")
        a = float(digits)
        if a == 0:
            return float("nan")               # "+0"/"-0"/"0" is not a real price
        return 1.0 + a / 100.0 if sign == "+" else 1.0 + 100.0 / a
    except (ValueError, IndexError):
        return float("nan")


def load_match_odds(matches: pd.DataFrame | None = None,
                    write_queues: bool = True) -> pd.DataFrame:
    """
    Load the `match_odds` table and return de-vigged bookmaker probabilities
    aligned to OUR match rows: one output row per resolved match, with columns
    ``date, home_team, away_team, book_ph, book_pd, book_pa`` where
    ``book_ph = P(our home_team wins)`` — i.e. already re-oriented to our DB's
    home/away ordering, not the source's.

    Three source quirks are handled here, each the reason a naive exact join
    would silently fail:
      * **names** — Oddsportal spellings are mapped via `canonical_team`;
        unresolved names are queued for review, never guessed.
      * **dates** — Oddsportal dates run ~1 day off ours (timezone), so matches
        are joined by the unordered team pair within `DATE_TOLERANCE_DAYS`, not
        on an exact date.
      * **orientation** — for neutral-venue ties the source's "home" need not be
        ours, so the (ph, pd, pa) vector is swapped to P(home)/P(draw)/P(away)
        in OUR ordering, decided by team identity.

    `matches` defaults to the cleaned match table. Rows whose odds are invalid
    (de-vig -> NaN), whose team can't be resolved, or which match no DB fixture
    are dropped (and, if `write_queues`, written to gitignored review CSVs) —
    never fabricated.
    """
    if matches is None:
        # Local import to avoid a module-level cycle (data_pipeline is heavy).
        from src.data_pipeline import load_raw_matches, clean_matches
        matches = clean_matches(load_raw_matches())

    conn = get_connection()
    odds = pd.read_sql("SELECT * FROM match_odds", conn)
    conn.close()
    odds["match_date"] = pd.to_datetime(odds["match_date"])

    db_names = set(matches["home_team"]) | set(matches["away_team"])

    # Index DB matches by the unordered team pair -> list of (date, home, away),
    # so each odds row is a cheap dict lookup + a small date-window filter.
    by_pair: dict[frozenset, list] = {}
    for m in matches.itertuples(index=False):
        by_pair.setdefault(frozenset((m.home_team, m.away_team)), []).append(
            (m.date, m.home_team, m.away_team)
        )

    resolved, unmatched_team, unmatched_fixture = [], [], []
    for r in odds.itertuples(index=False):
        ph, pd_, pa = devig(r.odds_home, r.odds_draw, r.odds_away)
        if math.isnan(ph):
            continue                                   # invalid/missing odds -> skip

        ch = canonical_team(r.home_team_raw, db_names)
        ca = canonical_team(r.away_team_raw, db_names)
        if ch is None or ca is None:
            unmatched_team.append((r.home_team_raw, r.away_team_raw, ch, ca))
            continue

        # candidate DB fixtures with this exact pair, within the date window
        cands = by_pair.get(frozenset((ch, ca)), [])
        cands = [c for c in cands
                 if abs((c[0] - r.match_date).days) <= DATE_TOLERANCE_DAYS]
        if not cands:
            unmatched_fixture.append((str(r.match_date.date()), ch, ca))
            continue
        # closest by date (guards the rare case two legs fall inside the window)
        db_date, db_home, db_away = min(cands, key=lambda c: abs((c[0] - r.match_date).days))

        # ORIENT to our home/away. Our home == source home -> keep; else swap
        # home/away probabilities (draw is unchanged).
        if db_home == ch:
            book = (ph, pd_, pa)
        else:                                          # our listing is reversed
            book = (pa, pd_, ph)

        resolved.append((db_date, db_home, db_away, *book))

    out = pd.DataFrame(resolved, columns=["date", "home_team", "away_team",
                                          "book_ph", "book_pd", "book_pa"])
    # A DB fixture could be reached from two odds rows only via bad data; keep
    # the first so the join key (date, home, away) stays unique.
    out = out.drop_duplicates(subset=["date", "home_team", "away_team"], keep="first")

    if write_queues:
        _QUEUE_DIR.mkdir(parents=True, exist_ok=True)
        # Write each queue when there's something to review, and REMOVE a stale
        # queue from a previous run when there isn't — so the files on disk
        # always reflect the current run, never a fixed-since-resolved leftover.
        team_q = _QUEUE_DIR / "odds_unmatched_teams.csv"
        fixture_q = _QUEUE_DIR / "odds_unmatched_fixtures.csv"
        if unmatched_team:
            pd.DataFrame(unmatched_team,
                         columns=["home_raw", "away_raw", "home_canon", "away_canon"]
                         ).to_csv(team_q, index=False)
        elif team_q.exists():
            team_q.unlink()
        if unmatched_fixture:
            pd.DataFrame(unmatched_fixture,
                         columns=["odds_date", "home_canon", "away_canon"]
                         ).to_csv(fixture_q, index=False)
        elif fixture_q.exists():
            fixture_q.unlink()

    print(f"load_match_odds: {len(odds)} odds rows -> {len(out)} resolved; "
          f"{len(unmatched_team)} unresolved name(s), "
          f"{len(unmatched_fixture)} with no DB fixture in +/-{DATE_TOLERANCE_DAYS}d.")
    return out


# ---------------------------------------------------------------------------
# Self-test: run `python -m src.odds` to verify the maths against
# hand-checkable cases (project rule: verify against real, checkable numbers,
# not "the code runs").
# ---------------------------------------------------------------------------
def _self_test() -> None:
    # Case 1 — FAIR odds (no vig): 2.0 / 4.0 / 4.0 imply 0.50 / 0.25 / 0.25,
    # which already sum to 1.0. De-vigging must return them essentially
    # unchanged (identity on a margin-free market).
    p = devig(2.0, 4.0, 4.0)
    assert math.isclose(sum(p), 1.0, abs_tol=1e-12), p
    assert math.isclose(p[0], 0.50, abs_tol=1e-9), p
    assert math.isclose(p[1], 0.25, abs_tol=1e-9), p
    print(f"[ok] fair odds 2.0/4.0/4.0 -> {tuple(round(x,4) for x in p)} (identity)")

    # Case 2 — a real, vig-carrying market. Raw implied sum > 1; output sums to
    # 1 and keeps the favourite as the largest probability.
    p = devig(1.90, 3.50, 4.20)
    raw_sum = 1/1.90 + 1/3.50 + 1/4.20
    assert raw_sum > 1.0, raw_sum                      # confirms there IS a margin
    assert math.isclose(sum(p), 1.0, abs_tol=1e-12), p # de-vigged sums to 1
    assert p[0] > p[1] and p[0] > p[2], p              # home is the favourite
    print(f"[ok] vig market 1.90/3.50/4.20 -> {tuple(round(x,4) for x in p)} "
          f"(raw sum {raw_sum:.4f}, margin {100*(raw_sum-1):.1f}%)")

    # Case 3 — a lopsided fixture (heavy favourite, e.g. a big nation vs a
    # minnow). The favourite's de-vigged probability should be very high.
    p = devig(1.04, 15.0, 41.0)
    assert p[0] > 0.90, p
    print(f"[ok] lopsided 1.04/15.0/41.0 -> {tuple(round(x,4) for x in p)} "
          f"(P(home)={p[0]:.3f})")

    # Case 4 — missing / invalid odds return all-NaN (never a guess).
    for bad in [devig(float("nan"), 3.5, 4.2), devig(1.0, 3.5, 4.2),
                devig(None, 3.5, 4.2)]:                 # type: ignore[arg-type]
        assert all(math.isnan(x) for x in bad), bad
    print("[ok] invalid/missing odds -> (nan, nan, nan)")

    # --- american_to_decimal ---
    assert math.isclose(american_to_decimal("+140"), 2.40, abs_tol=1e-9)
    assert math.isclose(american_to_decimal("-175"), 1.0 + 100/175, abs_tol=1e-9)
    assert math.isclose(american_to_decimal("+100"), 2.00, abs_tol=1e-9)   # even money
    assert math.isclose(american_to_decimal("1.44"), 1.44, abs_tol=1e-9)   # already decimal
    assert math.isclose(american_to_decimal("+1,200"), 13.0, abs_tol=1e-9) # comma tolerated
    assert math.isnan(american_to_decimal(""))
    assert math.isnan(american_to_decimal(None))
    assert math.isnan(american_to_decimal("nan"))
    # bare, unsigned integers (a browser paste can strip the leading "+") must
    # still resolve as positive American, not get mistaken for a decimal price —
    # this is the exact bug the flat Euro-2024 paste triggered originally.
    assert math.isclose(american_to_decimal("260"), 3.60, abs_tol=1e-9)
    assert math.isclose(american_to_decimal("425"), 5.25, abs_tol=1e-9)
    assert math.isclose(american_to_decimal("-1667"), 1.0 + 100/1667, abs_tol=1e-9)
    print("[ok] american_to_decimal: +140->2.40, -175->1.571, bare '260'->3.60, decimals pass through")

    # end-to-end on the real Euro 2024 final line (American): Spain +140 / +179 / +260
    p = devig(american_to_decimal("+140"), american_to_decimal("+179"),
              american_to_decimal("+260"))
    assert math.isclose(sum(p), 1.0, abs_tol=1e-12)
    assert p[0] > p[2]                                   # Spain favoured over England
    print(f"[ok] Euro2024 final +140/+179/+260 -> P(Spain)={p[0]:.3f} "
          f"P(draw)={p[1]:.3f} P(England)={p[2]:.3f}")

    print("\nall self-tests passed.")


if __name__ == "__main__":
    _self_test()
