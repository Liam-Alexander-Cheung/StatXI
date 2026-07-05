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