from flask import Flask, jsonify, request
from flask import render_template
import pandas as pd
from src.data_pipeline import load_raw_matches, clean_matches, load_squads
from src.features import rolling_form, head_to_head_record, goal_trend, squad_age_depth, team_chemistry
from src.models.poisson import fit_dixon_coles, predict_match
from src.models.build_matrix import FEATURE_COLUMNS, MATRIX_PATH
from src.models.walk_forward import _fit_before, VAL_DAYS, walk_forward, summarize

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


# Poisson attack/defence strengths are fit lazily per reference-date and cached.
# The fit is ~0.7s, so re-doing it for every prediction on the same date is waste;
# this dict keeps one fit per distinct "as of" date.
_strengths_cache = {}


def _ref_date():
    """Read the optional ?date=YYYY-MM-DD query param into a Timestamp; empty or
    absent means "now". Raises ValueError on a malformed date, which each route
    turns into a 400 rather than a 500."""
    d = request.args.get("date")
    return pd.Timestamp(d) if d else pd.Timestamp.now()


def get_strengths(ref_date):
    """Dixon-Coles strengths fit on ONLY the matches strictly before ref_date.

    This strict-before filter is the whole leakage guard: compute_weights weights
    by recency but does NOT drop future matches (a match after ref_date would even
    be up-weighted), so — exactly like walk_forward — the CALLER must pass a
    pre-cutoff slice. Cached per date. Returns None if there's no history yet."""
    key = ref_date.strftime("%Y-%m-%d")
    if key not in _strengths_cache:
        matches = get_matches()
        pre = matches[matches["date"] < ref_date]
        if pre.empty:
            return None
        _strengths_cache[key] = fit_dixon_coles(pre, reference_date=ref_date)
    return _strengths_cache.get(key)


# --- XGBoost Win/Draw/Loss: date-aware, fit on the cached feature matrix -------
# Loaded lazily. If the matrix (data/processed/training_matrix.csv) isn't built,
# XGBoost is unavailable and xgb_predict returns None — callers degrade gracefully.
_matrix_cache = None
_xgb_model_cache = {}


def get_matrix():
    global _matrix_cache
    if _matrix_cache is None:
        _matrix_cache = pd.read_csv(MATRIX_PATH, parse_dates=["date"])
    return _matrix_cache


def get_xgb_model(ref_date):
    """A fresh XGBoost model trained on the matrix rows before ref_date — the SAME
    leakage-free walk-forward fit the backtest uses, so it's directly comparable to
    the date-aware Poisson. Cached per date (~1.3s to fit)."""
    key = ref_date.strftime("%Y-%m-%d")
    if key not in _xgb_model_cache:
        model, _, _ = _fit_before(get_matrix(), ref_date, FEATURE_COLUMNS, VAL_DAYS)
        _xgb_model_cache[key] = model
    return _xgb_model_cache[key]


# --- Backtest scorecard: the walk-forward numbers, computed ONCE -------------
# walk_forward() refits a fresh XGBoost before each of the five backtested
# tournaments (~7s total), so we never redo it per request: the first visitor to
# the scorecard page triggers the compute, everyone after reads this cache. It's
# NOT warmed at startup on purpose — it would add ~7s to every server boot even
# for someone who only ever opens the predict page.
_scorecard_cache = None


def get_scorecard():
    """The walk-forward backtest reduced to display numbers (per-tournament +
    pooled + bookmaker), via the SAME summarize() the CLI report prints from — so
    the webapp and `make walk-forward` can never disagree. Computed lazily, cached."""
    global _scorecard_cache
    if _scorecard_cache is None:
        rows, pooled = walk_forward(get_matrix())
        _scorecard_cache = summarize(rows, pooled)
    return _scorecard_cache


def _xgb_row(matches, home, away, ref, neutral):
    """The one 21-feature row XGBoost needs, built with the SAME feature functions
    as build_matrix. Squad + rating features stay NaN (a pick-two-teams match has no
    tournament squad edition) — XGBoost handles missing values natively. `importance`
    defaults to top-tier (1.0), the assumed stakes of a marquee pick."""
    gt_h, gt_a = goal_trend(matches, home, ref), goal_trend(matches, away, ref)
    row = {c: float("nan") for c in FEATURE_COLUMNS}
    row["neutral"] = 1 if neutral else 0
    row["importance"] = 1.0
    row["home_form"] = rolling_form(matches, home, ref)
    row["away_form"] = rolling_form(matches, away, ref)
    row["home_gs"], row["home_gc"] = gt_h["goals_scored"], gt_h["goals_conceded"]
    row["away_gs"], row["away_gc"] = gt_a["goals_scored"], gt_a["goals_conceded"]
    row["h2h"] = head_to_head_record(matches, home, away, ref)
    return pd.DataFrame([row])[FEATURE_COLUMNS]


def xgb_predict(matches, home, away, ref, neutral):
    """XGBoost [home, draw, away] probabilities as a dict, or None if unavailable."""
    try:
        proba = get_xgb_model(ref).predict_proba(_xgb_row(matches, home, away, ref, neutral))[0]
        return {"home_win": round(float(proba[0]), 4),
                "draw": round(float(proba[1]), 4),
                "away_win": round(float(proba[2]), 4)}
    except Exception as e:
        app.logger.warning(f"xgb predict unavailable: {e}")
        return None


def bookmaker_probs(home, away, ref):
    """De-vigged bookmaker 1X2 probabilities for a REAL match, or None.

    Bookmaker odds exist only for matches actually played (they're joined per match
    in the training matrix as book_ph/pd/pa). A hypothetical pick has none. We match
    on (date, home, away); if the fixture is stored in the other orientation we swap
    home/away win (the draw is orientation-free). Returns None when there's no line."""
    try:
        df = get_matrix()
        day = ref.normalize()
        m = df[(df["date"] == day) & (df["home_team"] == home) & (df["away_team"] == away)]
        swap = False
        if m.empty:
            m = df[(df["date"] == day) & (df["home_team"] == away) & (df["away_team"] == home)]
            swap = True
        if m.empty or pd.isna(m.iloc[0]["book_ph"]):
            return None
        r = m.iloc[0]
        ph, pdr, pa = float(r["book_ph"]), float(r["book_pd"]), float(r["book_pa"])
        if swap:
            ph, pa = pa, ph
        return {"home_win": round(ph, 4), "draw": round(pdr, 4), "away_win": round(pa, 4)}
    except Exception:
        return None


def match_features(matches, home, away, ref):
    """The raw, human-readable signals feeding the models (for the detail page's
    self-analysis section). None where a value is unknown (NaN), never a fake 0."""
    gt_h, gt_a = goal_trend(matches, home, ref), goal_trend(matches, away, ref)
    def r(x):
        return None if pd.isna(x) else round(float(x), 3)
    return {
        "home_form": r(rolling_form(matches, home, ref)),
        "away_form": r(rolling_form(matches, away, ref)),
        "h2h_home": r(head_to_head_record(matches, home, away, ref)),
        "h2h_away": r(head_to_head_record(matches, away, home, ref)),
        "home_gs": r(gt_h["goals_scored"]), "home_gc": r(gt_h["goals_conceded"]),
        "away_gs": r(gt_a["goals_scored"]), "away_gc": r(gt_a["goals_conceded"]),
    }


@app.route("/api/rolling-form")
def api_rolling_form():
    team = request.args.get("team")
    if not team:
        return jsonify({"error": "missing 'team' query parameter"}), 400

    try:
        ref = _ref_date()
    except ValueError:
        return jsonify({"error": "bad 'date' (expected YYYY-MM-DD)"}), 400

    matches = get_matches()
    form = rolling_form(matches, team, ref)

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

    try:
        ref = _ref_date()
    except ValueError:
        return jsonify({"error": "bad 'date' (expected YYYY-MM-DD)"}), 400

    matches = get_matches()
    win_rate = head_to_head_record(matches, team_a, team_b, ref)

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


@app.route("/api/team-chemistry")
def api_team_chemistry():
    team = request.args.get("team")
    tournament = request.args.get("tournament")
    if not team or not tournament:
        return jsonify({"error": "missing 'team' or 'tournament' query parameter"}), 400

    squads = get_squads()
    result = team_chemistry(squads, team, tournament)

    # club_hhi is the field that goes NaN when no squad matches (same
    # reasoning as squad_age_depth's mean_age check) — the count fields are
    # a true 0 even for a real empty result, so they can't distinguish
    # "no squad found" from a squad that genuinely exists
    if pd.isna(result["club_hhi"]):
        return jsonify({"error": f"no squad data for '{team}' at '{tournament}'"}), 404

    return jsonify({"team": team, "tournament": tournament, **result})


@app.route("/api/predict")
def api_predict():
    # The headline endpoint: one Dixon-Coles fit powers the win/draw/loss split,
    # the most-likely scoreline, AND the expected goals (the two goal-rates lambda),
    # so a single call fills the prediction card and the xG box.
    home = request.args.get("home")
    away = request.args.get("away")
    if not home or not away:
        return jsonify({"error": "missing 'home' or 'away' query parameter"}), 400
    if home == away:
        return jsonify({"error": "'home' and 'away' must be different teams"}), 400
    try:
        ref = _ref_date()
    except ValueError:
        return jsonify({"error": "bad 'date' (expected YYYY-MM-DD)"}), 400

    strengths = get_strengths(ref)
    if strengths is None:
        return jsonify({"error": f"no match history before {ref.date()}"}), 404

    # Neutral venue by DEFAULT: picking two teams isn't at anyone's home ground, so
    # applying a home advantage to whoever sits in the 'home' slot would be a lie.
    # Pass ?neutral=0 to deliberately give `home` the home-field boost.
    neutral = request.args.get("neutral", "1") != "0"
    pred = predict_match(strengths, home, away, neutral=neutral)
    if pred is None:
        # predict_match returns None (never a fabricated guess) when a team has no
        # fitted strength — i.e. it played no matches before `ref`.
        return jsonify({"error": f"no data before {ref.date()} to rate '{home}' or '{away}'"}), 404

    i, j = pred["top_scoreline"]
    return jsonify({
        "home": home,
        "away": away,
        # both W/D/L models — the simple page shows their span as a range
        "poisson": {
            "home_win": round(pred["p_home"], 4),
            "draw": round(pred["p_draw"], 4),
            "away_win": round(pred["p_away"], 4),
        },
        "xgb": xgb_predict(get_matches(), home, away, ref, neutral),  # dict or None
        "scoreline": f"{i} – {j}",   # en-dash, matches the frontend's "1 – 1"
        "xg_home": round(pred["lambda_home"], 2),
        "xg_away": round(pred["lambda_away"], 2),
    })


@app.route("/api/detail")
def api_detail():
    # The "for the nerds" endpoint behind the detailed page: every W/D/L source
    # (Poisson, XGBoost, bookmaker) plus the scoreline grid and the raw feature
    # values, for one matchup.
    home = request.args.get("home")
    away = request.args.get("away")
    if not home or not away:
        return jsonify({"error": "missing 'home' or 'away' query parameter"}), 400
    if home == away:
        return jsonify({"error": "'home' and 'away' must be different teams"}), 400
    try:
        ref = _ref_date()
    except ValueError:
        return jsonify({"error": "bad 'date' (expected YYYY-MM-DD)"}), 400

    strengths = get_strengths(ref)
    if strengths is None:
        return jsonify({"error": f"no match history before {ref.date()}"}), 404
    neutral = request.args.get("neutral", "1") != "0"
    pred = predict_match(strengths, home, away, neutral=neutral)
    if pred is None:
        return jsonify({"error": f"no data before {ref.date()} to rate '{home}' or '{away}'"}), 404

    matches = get_matches()
    i, j = pred["top_scoreline"]
    grid = pred["grid"]  # full (max_goals+1) square; slice to 0–5 goals for display
    return jsonify({
        "home": home, "away": away, "date": ref.strftime("%Y-%m-%d"),
        "poisson": {"home_win": round(pred["p_home"], 4),
                    "draw": round(pred["p_draw"], 4),
                    "away_win": round(pred["p_away"], 4)},
        "xgb": xgb_predict(matches, home, away, ref, neutral),
        "bookmaker": bookmaker_probs(home, away, ref),
        "scoreline": f"{i} – {j}",
        "xg_home": round(pred["lambda_home"], 2),
        "xg_away": round(pred["lambda_away"], 2),
        "grid": [[round(float(x), 4) for x in row[:6]] for row in grid[:6]],
        "features": match_features(matches, home, away, ref),
    })


@app.route("/api/scorecard")
def api_scorecard():
    # The judges' hook made visible: the honest per-tournament backtest of the
    # WDL model against a naive favourite-picker and (where odds exist) the
    # bookmaker. No new modelling — it just surfaces walk_forward's numbers.
    s = get_scorecard()

    def r3(x):
        # round for display; None passes through untouched (never a fake 0)
        return None if x is None else round(float(x), 3)

    tournaments = [{
        "edition": t["edition"],
        "n": int(t["n"]),
        "acc": r3(t["acc"]),
        "acc_form_fav": r3(t["acc_form_fav"]),
        "log_loss": r3(t["log_loss"]),
    } for t in s["per_tournament"]]

    p = s["pooled"]
    pooled = {
        "n": int(p["n"]),
        "acc_model": r3(p["acc_model"]), "acc_form_fav": r3(p["acc_form_fav"]),
        "ll_model": r3(p["ll_model"]), "ll_base": r3(p["ll_base"]),
        "brier_model": r3(p["brier_model"]), "brier_base": r3(p["brier_base"]),
        "acc_vs_form_fav": r3(p["acc_vs_form_fav"]),
        "ll_vs_base": r3(p["ll_vs_base"]),
    }

    bk = s["bookmaker"]
    bookmaker = None if bk is None else {
        "n": int(bk["n"]), "n_total": int(bk["n_total"]),
        "acc_model": r3(bk["acc_model"]), "acc_book": r3(bk["acc_book"]),
        "ll_model": r3(bk["ll_model"]), "ll_book": r3(bk["ll_book"]),
        "brier_model": r3(bk["brier_model"]), "brier_book": r3(bk["brier_book"]),
    }

    return jsonify({"tournaments": tournaments, "pooled": pooled, "bookmaker": bookmaker})


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


@app.route("/legacy")
def legacy():
    # the previous feature-explorer UI, kept reachable as a fallback after the
    # SPA redesign took over "/". Self-contained template, uses the same API.
    return render_template("legacy_explorer.html")


if __name__ == "__main__":
    # port 5001, not 5000: on macOS the AirPlay Receiver (ControlCenter) squats
    # on 5000, and a stale Flask process from a previous run can hold it too —
    # both surface as "Address already in use". must be last — nothing after
    # this line ever runs
    app.run(debug=True, port=5001)