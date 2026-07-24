from flask import Flask, jsonify, request
import pandas as pd
from src.data_pipeline import load_raw_matches, clean_matches
from src.features import rolling_form

app = Flask(__name__)

_matches_cache = None


def get_matches():
    global _matches_cache
    if _matches_cache is None:
        df = load_raw_matches()
        _matches_cache = clean_matches(df)
    return _matches_cache


@app.route("/")
def hello():
    return "Flask is running"


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


if __name__ == "__main__":
    app.run(debug=True, port=5000)  # must be last — nothing after this line ever runs