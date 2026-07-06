import pandas as pd
#import recency_weight and imortance_weight from data_pipeline. 
from src.data_pipeline import recency_weight, importance_weight


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