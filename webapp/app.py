from flask import Flask, jsonify, request
from flask import render_template
import pandas as pd
from src.data_pipeline import load_raw_matches, clean_matches, load_squads
from src.features import rolling_form, head_to_head_record, goal_trend, squad_age_depth

app = Flask(__name__)

_matches_cache = None
_squads_cache = None


def get_matches():
    global _matches_cache
    if _matches_cache is None:
        df = load_raw_matches()
        _matches_cache = clean_matches(df)
    return _matches_cache


def get_squads():
    global _squads_cache
    if _squads_cache is None:
        _squads_cache = load_squads()
    return _squads_cache

# warm both caches immediately at startup, not on the first request —
# otherwise whichever user happens to click first eats the load delay
get_matches()
get_squads()


@app.route("/api/rolling-form")
def api_rolling_form():
    team = request.args.get("team")
    if not team:
        return jsonify({"error": "missing 'team' query parameter"}), 400

    matches = get_matches()
    form = rolling_form(matches, team, pd.Timestamp.now())

    if pd.isna(form):
        return jsonify({"error": f"no recent match data for '{team}'"}), 404

    return jsonify({"team": team, "rolling_form": round(form, 3)})


@app.route("/api/h2h")
def api_h2h():
    # param names match head_to_head_record's own team_a/team_b names,
    # so app.py and features.py use the same vocabulary for the same idea
    team_a = request.args.get("team_a")
    team_b = request.args.get("team_b")
    if not team_a or not team_b:
        return jsonify({"error": "missing 'team_a' or 'team_b' query parameter"}), 400

    # head_to_head_record has no internal guard for team_a == team_b —
    # a team can never appear as both home and away in the same match,
    # so that case silently falls through to its "no matches found" path
    # and returns NaN. Catching it here instead of letting it fall through
    # matters because "you asked a nonsensical question" (400, client's
    # fault) and "these two teams just have no shared history" (404, not
    # the client's fault) are different failures and deserve different
    # status codes — collapsing them into one NaN->404 path would mislabel
    # a bad request as a data-availability gap.
    if team_a == team_b:
        return jsonify({"error": "'team_a' and 'team_b' must be different teams"}), 400

    matches = get_matches()
    win_rate = head_to_head_record(matches, team_a, team_b, pd.Timestamp.now())

    # NaN here means the two teams (as far as the dataset can tell) have
    # never played each other — could also mean one/both names are simply
    # misspelled or absent from the data. /api/rolling-form has the exact
    # same ambiguity for unknown teams and doesn't resolve it either, so
    # deliberately not solving it here keeps this change minimal and
    # consistent with existing precedent, rather than a silent inconsistency.
    if pd.isna(win_rate):
        return jsonify({"error": f"no head-to-head data for '{team_a}' vs '{team_b}'"}), 404

    # h2h_win_rate (not a bare "h2h") so this response's keys don't collide
    # if a caller ever stores both this and the rolling-form response
    # together in one object
    return jsonify({"team_a": team_a, "team_b": team_b, "h2h_win_rate": round(win_rate, 3)})


@app.route("/api/goal-trend")
def api_goal_trend():
    team = request.args.get("team")
    if not team:
        return jsonify({"error": "missing 'team' query parameter"}), 400

    matches = get_matches()
    trend = goal_trend(matches, team, pd.Timestamp.now())

    # goal_trend always returns all three keys, never a bare NaN like
    # rolling_form/head_to_head_record do — checking goals_scored alone is
    # enough, since all three come from the same underlying team_matches
    # filter and go NaN together (goal_differential is NaN - NaN, not a
    # separately-computed value that could disagree with the other two)
    if pd.isna(trend["goals_scored"]):
        return jsonify({"error": f"no recent match data for '{team}'"}), 404

    return jsonify({
        "team": team,
        "goals_scored": round(trend["goals_scored"], 3),
        "goals_conceded": round(trend["goals_conceded"], 3),
        "goal_differential": round(trend["goal_differential"], 3),
    })


@app.route("/api/squad-age-depth")
def api_squad_age_depth():
    team = request.args.get("team")
    tournament = request.args.get("tournament")
    if not team or not tournament:
        return jsonify({"error": "missing 'team' or 'tournament' query parameter"}), 400

    squads = get_squads()
    result = squad_age_depth(squads, team, tournament)

    # mean_age is the one always-NaN-together field when no squad matches
    # (same reasoning as goal_trend's goals_scored check above) — squad_size
    # would also work as the check, but mean_age matches the pattern used
    # by the other single-value endpoints (rolling_form, h2h) most closely
    if pd.isna(result["mean_age"]):
        return jsonify({"error": f"no squad data for '{team}' at '{tournament}'"}), 404

    return jsonify({"team": team, "tournament": tournament, **result})


@app.route("/api/teams")
def api_teams():
    matches = get_matches()
    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    return jsonify(teams)


@app.route("/api/tournaments")
def api_tournaments():
    squads = get_squads()
    tournaments = sorted(set(squads["tournament_name"]))
    return jsonify(tournaments)


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000)  # must be last — nothing after this line ever runs