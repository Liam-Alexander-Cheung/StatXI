from flask import Flask, jsonify, request
from flask import render_template
import pandas as pd
from src.data_pipeline import load_raw_matches, clean_matches
from src.features import rolling_form, head_to_head_record

app = Flask(__name__)

_matches_cache = None


def get_matches():
    global _matches_cache
    if _matches_cache is None:
        df = load_raw_matches()
        _matches_cache = clean_matches(df)
    return _matches_cache

# warm the cache immediately at startup, not on the first request —
# otherwise whichever user happens to click first eats a ~14s delay
get_matches()


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


@app.route("/api/teams")
def api_teams():
    matches = get_matches()
    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    return jsonify(teams)


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000)  # must be last — nothing after this line ever runs