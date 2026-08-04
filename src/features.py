import math  # math.comb(k, 2) counts unordered pairs — used for club-pair chemistry
import pandas as pd
#import recency_weight and imortance_weight from data_pipeline.
from src.data_pipeline import recency_weight, importance_weight, fifa_edition_for_date


# define rolling_form
def rolling_form(
    matches: pd.DataFrame,
    team: str,
    as_of_date: pd.Timestamp,
    window_years: int = 10,
    half_life_days: int = 730,
) -> float:
    """
    Weighted win rate for `team`, based on matches strictly before
    as_of_date, within the last `window_years`. Weighting combines
    recency (relative to as_of_date) and tournament importance — same
    two factors as add_match_weights, but recomputed per call since the
    reference point changes for every match being evaluated.

    Returns float('nan') if the team has no qualifying match history
    (e.g. a debutant nation) — deliberately not defaulting to 0.5 or 0,
    since "no data" and "known average form" are different things a
    model should be allowed to distinguish.
    """
    # Timedelta is pandas' duration type
    # 365 ignores leap years 
    window_start = as_of_date - pd.Timedelta(days=365 * window_years)

    team_matches = matches[
        ((matches["home_team"] == team) | (matches["away_team"] == team))
        & (matches["date"] < as_of_date)
        & (matches["date"] >= window_start)
    ]
 
    if len(team_matches) == 0:
        return float("nan")

    # defining outcome helps with with the asymmetricness of "home_score" and "away_score"
    def outcome(row) -> float:
        if row["home_team"] == team:
            team_score, opp_score = row["home_score"], row["away_score"]
        else:
            team_score, opp_score = row["away_score"], row["home_score"]
        if team_score > opp_score:
            return 1.0
        if team_score == opp_score:
            return 0.5
        return 0.0

    outcomes = team_matches.apply(outcome, axis=1)
    weights = team_matches.apply(
        lambda row: recency_weight(row["date"], as_of_date, half_life_days)
        * importance_weight(row["tournament"]),
        axis=1,
    )

    # the standard formula for a weighted mean
    return (outcomes * weights).sum() / weights.sum()


# define head to head record
def head_to_head_record(
    matches: pd.DataFrame,
    team_a: str,
    team_b: str,
    as_of_date: pd.Timestamp,
    half_life_days: int = 3650,  # 10-year half-life, much gentler than rolling_form's window
) -> float:
    """
    Weighted win rate for team_a against team_b specifically, across all
    historical meetings before as_of_date. No fixed window — since two
    teams may only meet once every several years, discarding meetings
    older than 10 years (as rolling_form does) would leave too many
    pairings with near-zero data. Returns float('nan') if the two teams
    have never met.
    """
    h2h = matches[
        (((matches["home_team"] == team_a) & (matches["away_team"] == team_b))
         | ((matches["home_team"] == team_b) & (matches["away_team"] == team_a)))
        & (matches["date"] < as_of_date)
    ]

    if len(h2h) == 0:
        # same float nan as in rolling form
        return float("nan")

    def outcome(row) -> float:
        if row["home_team"] == team_a:
            a_score, b_score = row["home_score"], row["away_score"]
        else:
            a_score, b_score = row["away_score"], row["home_score"]
        if a_score > b_score:
            return 1.0
        if a_score == b_score:
            return 0.5
        return 0.0

    # apply weights and read every line of outcomes (axis=1)
    outcomes = h2h.apply(outcome, axis=1)
    weights = h2h.apply(
        lambda row: recency_weight(row["date"], as_of_date, half_life_days)
        * importance_weight(row["tournament"]),
        axis=1,
    )
    # classic calculation like in rolling form
    return (outcomes * weights).sum() / weights.sum()


# defining goal_trend
def goal_trend(
    matches: pd.DataFrame,
    team: str,
    as_of_date: pd.Timestamp,
    window_years: int = 10,
    half_life_days: int = 730,
) -> dict:
    """
    Weighted average goals scored and conceded by `team`, based on matches
    strictly before as_of_date, within the last `window_years`. Same
    window/weighting convention as rolling_form, since both belong to the
    same "recent form" feature cluster.

    Returns {'goals_scored': nan, 'goals_conceded': nan} if no qualifying
    history exists — kept as two separate NaN-able values rather than one
    goal-difference scalar, since scoring and conceding are independent
    signals (a 3-3 team and a 0.5-0.5 team both average to zero difference
    but represent very different teams).
    """
    window_start = as_of_date - pd.Timedelta(days=365 * window_years)

    # defnining home and away team for the 5th time
    team_matches = matches[
        ((matches["home_team"] == team) | (matches["away_team"] == team))
        & (matches["date"] < as_of_date)
        & (matches["date"] >= window_start)
    ]

    if len(team_matches) == 0:
        return {"goals_scored": float("nan"), "goals_conceded": float("nan")}

    def scored(row) -> float:
        return row["home_score"] if row["home_team"] == team else row["away_score"]

    def conceded(row) -> float:
        return row["away_score"] if row["home_team"] == team else row["home_score"]

    # apply weights funcion
    weights = team_matches.apply(
        # lambda for multi-collumn reading
        lambda row: recency_weight(row["date"], as_of_date, half_life_days)
        * importance_weight(row["tournament"]),
        # go row by row
        axis=1,
    )
    scored_vals = team_matches.apply(scored, axis=1)
    conceded_vals = team_matches.apply(conceded, axis=1)

    return {
        # basic scored/conceded/differential calculations for trend
        "goals_scored": (scored_vals * weights).sum() / weights.sum(),
        "goals_conceded": (conceded_vals * weights).sum() / weights.sum(),
        "goal_differential": (
            (scored_vals * weights).sum() / weights.sum()
            - (conceded_vals * weights).sum() / weights.sum()
        ),
    }


# only 4 distinct position values exist in the whole players table (GK, DF,
# MF, FW) — checked directly against the database before relying on this,
# rather than assuming the scraped data was this clean
_POSITIONS = ["GK", "DF", "MF", "FW"]


def squad_age_depth(squads: pd.DataFrame, team: str, tournament_name: str) -> dict:
    """
    Age and positional-depth profile for `team`'s squad at one specific
    tournament. Squad-scoped, not date-scoped like the match-level features
    above — a tournament squad is a single fixed list, not something to
    filter "as of" a date.

    Returns mean age, squad size, and player counts *and* proportions per
    position. Both, not just one: counts are directly meaningful ("9
    defenders"), but proportions are what's actually comparable across
    squads, since official squad size varies (23-26 players depending on
    the tournament's own rules that year) — two squads with equal counts
    at a position can still differ in what share of the squad that is.

    Returns NaN-populated placeholders if `team`/`tournament_name` doesn't
    match any squad (typo, or a team that didn't compete). Position counts
    are correctly 0 in that case — that's a true fact about an empty
    result, not a fabrication — but proportions are NaN rather than a
    guessed 0.0, since "0 out of 0" is undefined, not zero.
    """
    squad = squads[
        (squads["team"] == team) & (squads["tournament_name"] == tournament_name)
    ]

    if len(squad) == 0:
        return {
            "mean_age": float("nan"),
            "squad_size": 0,
            "position_counts": {pos: 0 for pos in _POSITIONS},
            "position_proportions": {pos: float("nan") for pos in _POSITIONS},
        }

    squad_size = len(squad)

    # reindex against the known 4 positions (not just whatever positions
    # happen to appear in this particular squad) so every squad's dict has
    # the same 4 keys, even if e.g. a squad were missing an FW entirely
    counts = squad["position"].value_counts().reindex(_POSITIONS, fill_value=0)
    # cast numpy int64 -> plain int so this serializes cleanly to JSON
    # later without a custom encoder, same reasoning as elsewhere in this file
    position_counts = {pos: int(count) for pos, count in counts.items()}
    position_proportions = {
        pos: round(count / squad_size, 3) for pos, count in position_counts.items()
    }

    return {
        "mean_age": round(squad["age_at_tournament"].mean(), 3),
        "squad_size": squad_size,
        "position_counts": position_counts,
        "position_proportions": position_proportions,
    }


def team_chemistry(squads: pd.DataFrame, team: str, tournament_name: str) -> dict:
    """
    Club-cohesion profile for `team`'s squad at one specific tournament.
    Squad-scoped, exactly like squad_age_depth — a tournament squad is a
    single fixed list of players, each with the club they played for at
    that time (the DB stores the *as-of-tournament* club, so this is
    era-correct with zero extra work).

    The idea is borrowed, honestly, from video-game "chemistry": players
    who share a club understand each other on the pitch. For an
    international squad the nationality link is degenerate (everyone shares
    it), so the discriminating signal is club concentration — a squad built
    on one or two club spines (Germany 2014's Bayern core, Spain 2010's
    Barça+Real core) versus one scattered across many clubs.

    Metrics, where n = squad size and c_i = number of players at club i:
      largest_club_bloc     max(c_i)          — biggest single-club spine
      top2_club_bloc        two largest c_i   — two-club spine
      club_hhi              Σ (c_i/n)²        — Herfindahl concentration:
                                                1.0 = all one club,
                                                ≈1/n = all different clubs
      same_club_pairs       Σ C(c_i, 2)       — teammate pairs sharing a club
      same_club_pair_ratio  same_club_pairs / C(n, 2)  — normalized pair
                                                density (the main feature)
      n_distinct_clubs      count of clubs    — inverse breadth

    Missing-data rule (same discipline as squad_age_depth): if
    team/tournament_name matches no squad, counts are a true 0 but ratios
    are NaN, never a guessed value — "0 out of 0" is undefined, not zero.
    """
    squad = squads[
        (squads["team"] == team) & (squads["tournament_name"] == tournament_name)
    ]

    if len(squad) == 0:
        return {
            "largest_club_bloc": 0,
            "top2_club_bloc": 0,
            "club_hhi": float("nan"),
            "same_club_pairs": 0,
            "same_club_pair_ratio": float("nan"),
            "n_distinct_clubs": 0,
        }

    n = len(squad)
    # value_counts returns per-club counts already sorted high→low, so
    # .iloc[0] is the largest bloc and .iloc[:2] the two largest
    club_counts = squad["club"].value_counts()

    largest_club_bloc = int(club_counts.iloc[0])
    top2_club_bloc = int(club_counts.iloc[:2].sum())
    n_distinct_clubs = int(len(club_counts))

    # Herfindahl index: sum of squared club shares. Cast to plain float so
    # it serializes to JSON cleanly (value_counts / n gives numpy floats)
    club_hhi = float(((club_counts / n) ** 2).sum())

    # C(c_i, 2) = number of within-club teammate pairs for a club of size
    # c_i; summed over clubs gives total same-club pairs. math.comb(k, 2)
    # is "k choose 2" = k*(k-1)/2
    same_club_pairs = int(sum(math.comb(int(c), 2) for c in club_counts))

    # normalize by the total number of possible pairs in the squad, C(n, 2).
    # Guard n < 2 (no pairs possible) → NaN rather than a 0/0 crash; real
    # squads are ~23 players so this only guards a degenerate input
    total_pairs = math.comb(n, 2)
    same_club_pair_ratio = (
        same_club_pairs / total_pairs if total_pairs > 0 else float("nan")
    )

    return {
        "largest_club_bloc": largest_club_bloc,
        "top2_club_bloc": top2_club_bloc,
        "club_hhi": round(club_hhi, 3),
        "same_club_pairs": same_club_pairs,
        "same_club_pair_ratio": round(same_club_pair_ratio, 3),
        "n_distinct_clubs": n_distinct_clubs,
    }


from datetime import datetime

def transfer_value_delta_z(
    player_id: str, as_of_date: pd.Timestamp, months_back: int = 12
) -> tuple[float, float]:
    """
    Returns (value_at_as_of_date, value_months_back), both in raw euros,
    picked as the closest available data point to each target date. Not
    z-scored yet — that normalization happens across a whole player cohort,
    not per-player, so this function only returns raw values; the caller
    z-scores across the full squad list.
    """
    # fetch market value history from data pipeline
    from src.data_pipeline import fetch_market_value_history

    history = fetch_market_value_history(player_id)

    parsed = [
        # parse transfermarkkt's day/month/year format
        {"date": datetime.strptime(p["datum_mw"], "%d/%m/%Y"), "value": p["y"]}
        for p in history
    ]
    parsed.sort(key=lambda p: p["date"])

    target_recent = as_of_date
    target_past = as_of_date - pd.Timedelta(days=30 * months_back)

    # closest value takes the most recent valuation at or before the target date
    def closest_value(target: pd.Timestamp) -> float:
        candidates = [p for p in parsed if p["date"] <= target]
        if not candidates:
            return float("nan")
        return candidates[-1]["value"]

    return closest_value(target_recent), closest_value(target_past)


# A squad mean needs enough rated players to be trustworthy. Below a starting
# XI's worth (11), the mean is dominated by whoever happened to link, so we
# return NaN rather than a shaky number — same never-fabricate stance as the
# other squad features (missing -> NaN, never a guessed value).
_MIN_RATED_PLAYERS = 11


def squad_mean_rating(
    squad_ratings: pd.DataFrame, team: str, tournament_name: str
) -> dict:
    """
    Mean FIFA/FC `overall` (and `potential`) of a team's rated players at one
    tournament — the direct squad-quality signal validated in the TRACK 2
    correlation check (higher mean `overall` -> the team wins markedly more).

    `squad_ratings` is the flat frame from load_squad_ratings (already scoped to
    the era-correct edition per tournament). Returns NaN for both means when the
    tournament has no ratings coverage or fewer than _MIN_RATED_PLAYERS linked
    players — squad-scoped like squad_age_depth / team_chemistry, not date-scoped.
    """
    squad = squad_ratings[
        (squad_ratings["team"] == team)
        & (squad_ratings["tournament_name"] == tournament_name)
    ]
    # count only appearances that actually resolved to a rating value
    n_rated = int(squad["overall"].notna().sum())

    if n_rated < _MIN_RATED_PLAYERS:
        return {"mean_overall": float("nan"),
                "mean_potential": float("nan"),
                "n_rated": n_rated}

    return {
        "mean_overall": round(squad["overall"].mean(), 3),
        "mean_potential": round(squad["potential"].mean(), 3),
        "n_rated": n_rated,
    }


def build_pool_index(pool: pd.DataFrame) -> dict:
    """
    Group the flat pool table from load_team_pool_ratings into a fast lookup:
        {(team, fifa_version): [(first_seen, overall), ...]}
    Built once; lets team_pool_rating answer millions of per-match queries without
    re-filtering a 14k-row DataFrame each time (this feature, unlike the squad
    ones, fires on ~every match, not just the ~875 tournament rows).
    """
    index: dict = {}
    for team, _sid, first_seen, version, overall in pool.itertuples(index=False):
        index.setdefault((team, version), []).append((first_seen, overall))
    return index


def team_pool_rating(pool_index: dict, team: str, match_date: pd.Timestamp,
                     min_rated: int = 11) -> float:
    """
    Estimate a country's squad strength going INTO a match: the mean FIFA `overall`
    of its known internationals at the match's era-correct edition. This is the
    approximation that extends ratings from the 5 scraped tournaments to any
    international match in the FIFA-era window (we have squads, not match lineups,
    so a country's tournament-squad player pool stands in for its match-day XI).

    Leakage guard: only players whose `first_seen` is on or before `match_date`
    count — a match can never be credited with a player known only from a *later*
    squad. Returns NaN before FIFA 15, for uncovered teams, or when fewer than
    `min_rated` players are available (never a fabricated number).
    """
    edition = fifa_edition_for_date(match_date)
    if edition is None:
        return float("nan")
    entries = pool_index.get((team, edition), [])
    # keep only players already "known" as this country's internationals by now
    overalls = [overall for first_seen, overall in entries if first_seen <= match_date]
    if len(overalls) < min_rated:
        return float("nan")
    return round(sum(overalls) / len(overalls), 3)