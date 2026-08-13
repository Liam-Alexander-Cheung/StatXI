/* api.js — THE DATA LAYER (live version).
 *
 * This is the real-backend twin of the mock in webapp/previews/app. Same function
 * names and same return shapes as the mock, so the views never changed — only the
 * bodies did: each function now calls the Flask API instead of inventing numbers.
 *
 * Design notes:
 *  - rolling form & h2h are BOTH-TEAMS metrics, but the backend endpoints are
 *    single-team, so we fire two requests and combine them (Promise.all).
 *  - a missing team returns null (shown as "—"), never a fabricated value.
 *  - the prediction (win/draw/loss + scoreline + expected goals) is ONE call:
 *    /api/predict does a single Poisson fit that yields all of it.
 */
window.StatXI = window.StatXI || {};

(function () {
  // Build a query string, dropping empty params so ?date= is omitted when blank.
  function qs(params){
    return Object.keys(params)
      .filter(function(k){ return params[k] !== undefined && params[k] !== null && params[k] !== ''; })
      .map(function(k){ return encodeURIComponent(k) + '=' + encodeURIComponent(params[k]); })
      .join('&');
  }

  // fetch JSON; on a non-2xx, reject with the server's own {error} message.
  function getJSON(url){
    return fetch(url).then(function(r){
      return r.json().then(function(body){
        if (!r.ok) throw new Error(body && body.error ? body.error : ('HTTP ' + r.status));
        return body;
      });
    });
  }
  // same, but swallow failures into null — for optional per-team lookups where one
  // side may simply have no data (e.g. a debutant with no rolling form yet).
  function tryJSON(url){ return getJSON(url).then(function(b){ return b; }, function(){ return null; }); }

  StatXI.api = {
    getTeams: function(){ return getJSON('/api/teams'); },

    // both teams' rolling form (two single-team calls, combined)
    getRollingForm: function(a, b, date){
      return Promise.all([
        tryJSON('/api/rolling-form?' + qs({ team: a, date: date })),
        tryJSON('/api/rolling-form?' + qs({ team: b, date: date }))
      ]).then(function(r){
        return { a: r[0] ? r[0].rolling_form : null, b: r[1] ? r[1].rolling_form : null };
      });
    },

    // both directions of the head-to-head win share (A-vs-B and B-vs-A)
    getH2H: function(a, b, date){
      return Promise.all([
        tryJSON('/api/h2h?' + qs({ team_a: a, team_b: b, date: date })),
        tryJSON('/api/h2h?' + qs({ team_a: b, team_b: a, date: date }))
      ]).then(function(r){
        return { aWin: r[0] ? r[0].h2h_win_rate : null, bWin: r[1] ? r[1].h2h_win_rate : null };
      });
    },

    // win/draw/loss (Poisson + XGBoost) + scoreline + expected goals, from one fit
    getPrediction: function(a, b, date){
      return getJSON('/api/predict?' + qs({ home: a, away: b, date: date }));
    },

    // the detailed page's everything: all W/D/L sources + bookmaker + grid + features
    getDetail: function(a, b, date){
      return getJSON('/api/detail?' + qs({ home: a, away: b, date: date }));
    },

    // the backtest scorecard: per-tournament + pooled + bookmaker accuracy/
    // log-loss/Brier. One call; the first one is slow (~7s) because the server
    // refits a fresh model before each tournament, then caches the result.
    getScorecard: function(){ return getJSON('/api/scorecard'); }
  };
})();
