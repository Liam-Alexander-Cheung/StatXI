/* views/predict.js — the main page (#/predict), live version.
 * Pick a date + two teams, hit Predict, and each box loads INDEPENDENTLY:
 * a spinner, then real data the moment its request resolves. The prediction
 * request fills TWO boxes (the card and the expected-goals tile) because a single
 * Poisson fit produces both. */
window.StatXI = window.StatXI || {};
StatXI.views = StatXI.views || {};

(function () {
  var api = StatXI.api;

  // The last match the user actually ran, remembered for the lifetime of the page.
  // The SPA re-renders every view from scratch on each navigation, so without this
  // a trip to the detail page and back (via the link OR the browser back button)
  // would drop you on an empty form. When set, mount() re-selects those teams/date
  // and re-runs the prediction so the results are already there on return.
  var lastRun = null;

  // --- formatting helpers (null-safe: missing data shows "—", never a fake 0) ---
  function num(x){ return (x === null || x === undefined || isNaN(x)) ? '—' : Number(x).toFixed(2); }
  // the RANGE spanning the two models (Poisson ↔ XGBoost), e.g. "34–40%".
  // Falls back to a single value when XGBoost is unavailable.
  function rangeStr(p, q){
    if (p === null || p === undefined || isNaN(p)) return '—';
    if (q === null || q === undefined || isNaN(q)) return Math.round(p * 100) + '%';
    var lo = Math.round(Math.min(p, q) * 100), hi = Math.round(Math.max(p, q) * 100);
    return lo === hi ? lo + '%' : lo + '–' + hi + '%';
  }
  // hover tooltip on the number: which model said what
  function rangeTitle(p, q){
    var pp = (p === null || p === undefined || isNaN(p)) ? '—' : Math.round(p * 100) + '%';
    if (q === null || q === undefined || isNaN(q)) return 'Poisson ' + pp;
    return 'Poisson ' + pp + '  ·  XGBoost ' + Math.round(q * 100) + '%';
  }
  function spinner(){ return '<div class="loading"><span class="spinner"></span> Loading…</div>'; }
  function errorBox(msg){ return '<div class="loading">⚠ ' + msg + '</div>'; }

  // one team+value row inside a two-team tile (NAME only, no flag)
  function duoItem(team, value){
    return '<div class="duo-item"><div class="who">' + team + '</div>' +
           '<div class="val">' + value + '</div></div>';
  }
  function fillDuoTile(el, title, sub, teamA, valA, teamB, valB){
    el.innerHTML = '<div class="t">' + title + '</div>' +
      '<div class="duo">' + duoItem(teamA, valA) + duoItem(teamB, valB) + '</div>' +
      '<div class="sub">' + sub + '</div>';
  }

  // p = Poisson prob, q = XGBoost prob; the cell shows their range, bar = midpoint
  function oddCol(cls, label, p, q){
    var mid = (q === null || q === undefined || isNaN(q)) ? (p || 0) : (p + q) / 2;
    return '<div class="odd ' + cls + '"><div class="lbl">' + label + '</div>' +
      '<div class="pct" title="' + rangeTitle(p, q) + '">' + rangeStr(p, q) + '</div>' +
      '<div class="bar"><i style="width:' + Math.round(mid * 100) + '%"></i></div></div>';
  }
  function fillPrediction(el, d, a, b){
    var P = d.poisson || {}, X = d.xgb || {};   // X absent if XGBoost unavailable
    el.innerHTML =
      '<div class="match">' +
        '<div class="team"><div class="flag">' + StatXI.flag(a) + '</div><div class="name">' + a + '</div></div>' +
        '<div class="vs">VS</div>' +
        '<div class="team"><div class="flag">' + StatXI.flag(b) + '</div><div class="name">' + b + '</div></div>' +
      '</div>' +
      '<div class="odds">' +
        oddCol('win',  a,      P.home_win, X.home_win) +
        oddCol('draw', 'Draw', P.draw,     X.draw) +
        oddCol('loss', b,      P.away_win, X.away_win) +
      '</div>' +
      '<div class="scoreline"><span class="k">Most likely scoreline</span>' +
        '<span class="v">' + d.scoreline + '</span></div>';
  }

  function run(root, a, b, date){
    var results = root.querySelector('#results');
    results.hidden = false;

    // point the "Full analysis" link at the detail page for THIS matchup
    var link = root.querySelector('#detailLink');
    if (link) link.href = '#/detail?home=' + encodeURIComponent(a) +
      '&away=' + encodeURIComponent(b) + '&date=' + encodeURIComponent(date);

    var boxPred = root.querySelector('#box-pred');
    var boxForm = root.querySelector('#box-form');
    var boxH2H  = root.querySelector('#box-h2h');
    var boxXg   = root.querySelector('#box-xg');
    [boxPred, boxForm, boxH2H, boxXg].forEach(function(el){ el.innerHTML = spinner(); });

    api.getRollingForm(a, b, date).then(function(d){
      fillDuoTile(boxForm, 'Rolling form', 'weighted win rate · last 10 yrs', a, num(d.a), b, num(d.b));
    });
    api.getH2H(a, b, date).then(function(d){
      fillDuoTile(boxH2H, 'Head-to-head', 'win share when these two meet', a, num(d.aWin), b, num(d.bWin));
    });
    // ONE prediction request fills BOTH the card and the expected-goals tile
    api.getPrediction(a, b, date).then(function(d){
      fillPrediction(boxPred, d, a, b);
      fillDuoTile(boxXg, 'Expected goals', 'goals the model expects per side', a, num(d.xg_home), b, num(d.xg_away));
    }).catch(function(err){
      boxPred.innerHTML = errorBox(err.message || 'No prediction available.');
      boxXg.innerHTML   = errorBox('No expected-goals data.');
    });
  }

  StatXI.views.predict = {
    render: function(){
      return '' +
      '<section class="predict">' +
        '<h1>Pick a match</h1>' +
        '<p class="lead">Choose a date and two teams. The model treats it as ' +
          '“predict this match using only what was known before that date”.</p>' +
        '<form class="controls" id="predictForm">' +
          '<label>Date<input type="date" id="matchDate" value="2024-06-14"></label>' +
          '<label>Team A<select id="teamA"></select></label>' +
          '<label>Team B<select id="teamB"></select></label>' +
          '<button class="btn" type="submit">Predict →</button>' +
        '</form>' +
        '<div id="results" class="results" hidden>' +
          '<div class="card" id="box-pred"></div>' +
          '<div class="tiles">' +
            '<div class="tile" id="box-form"></div>' +
            '<div class="tile" id="box-h2h"></div>' +
            '<div class="tile" id="box-xg"></div>' +
          '</div>' +
          '<div class="detail-cta"><a id="detailLink" href="#/detail">Full analysis (XGBoost · bookmaker · scoreline grid) →</a></div>' +
        '</div>' +
      '</section>';
    },
    mount: function(root){
      var selA = root.querySelector('#teamA');
      var selB = root.querySelector('#teamB');

      // fill both dropdowns from the real team list, then either restore the last
      // run (returning from detail) or fall back to a meaningful default matchup
      api.getTeams().then(function(teams){
        var opts = teams.map(function(t){ return '<option>' + t + '</option>'; }).join('');
        selA.innerHTML = opts; selB.innerHTML = opts;
        if (lastRun){
          selA.value = lastRun.a; selB.value = lastRun.b;
          root.querySelector('#matchDate').value = lastRun.date;
          run(root, lastRun.a, lastRun.b, lastRun.date);   // results already on screen
        } else {
          selA.value = teams.indexOf('Germany') >= 0 ? 'Germany' : teams[0];
          selB.value = teams.indexOf('France')  >= 0 ? 'France'  : teams[1];
        }
      });

      root.querySelector('#predictForm').addEventListener('submit', function(e){
        e.preventDefault();
        var a = selA.value, b = selB.value, date = root.querySelector('#matchDate').value;
        if (a === b){ alert('Pick two different teams.'); return; }
        lastRun = { a: a, b: b, date: date };   // remember it for the round-trip to detail
        run(root, a, b, date);
      });
    }
  };
})();
