/* views/predict.js — the main page (#/predict), live version.
 * Pick a date + two teams, hit Predict, and each box loads INDEPENDENTLY:
 * a spinner, then real data the moment its request resolves. The prediction
 * request fills TWO boxes (the card and the expected-goals tile) because a single
 * Poisson fit produces both. */
window.StatXI = window.StatXI || {};
StatXI.views = StatXI.views || {};

(function () {
  var api = StatXI.api;

  // --- formatting helpers (null-safe: missing data shows "—", never a fake 0) ---
  function pct(x){ return (x === null || x === undefined || isNaN(x)) ? '—' : Math.round(x * 100) + '%'; }
  function num(x){ return (x === null || x === undefined || isNaN(x)) ? '—' : Number(x).toFixed(2); }
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

  function oddCol(cls, label, p){
    return '<div class="odd ' + cls + '"><div class="lbl">' + label + '</div>' +
      '<div class="pct">' + pct(p) + '</div>' +
      '<div class="bar"><i style="width:' + Math.round((p || 0) * 100) + '%"></i></div></div>';
  }
  function fillPrediction(el, d, a, b){
    el.innerHTML =
      '<div class="match">' +
        '<div class="team"><div class="flag">' + api.flag(a) + '</div><div class="name">' + a + '</div></div>' +
        '<div class="vs">VS</div>' +
        '<div class="team"><div class="flag">' + api.flag(b) + '</div><div class="name">' + b + '</div></div>' +
      '</div>' +
      '<div class="odds">' +
        oddCol('win',  a,      d.home_win) +
        oddCol('draw', 'Draw', d.draw) +
        oddCol('loss', b,      d.away_win) +
      '</div>' +
      '<div class="scoreline"><span class="k">Most likely scoreline</span>' +
        '<span class="v">' + d.scoreline + '</span></div>';
  }

  function run(root, a, b, date){
    var results = root.querySelector('#results');
    results.hidden = false;

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
        '</div>' +
      '</section>';
    },
    mount: function(root){
      var selA = root.querySelector('#teamA');
      var selB = root.querySelector('#teamB');

      // fill both dropdowns from the real team list; default to a meaningful matchup
      api.getTeams().then(function(teams){
        var opts = teams.map(function(t){ return '<option>' + t + '</option>'; }).join('');
        selA.innerHTML = opts; selB.innerHTML = opts;
        selA.value = teams.indexOf('Germany') >= 0 ? 'Germany' : teams[0];
        selB.value = teams.indexOf('France')  >= 0 ? 'France'  : teams[1];
      });

      root.querySelector('#predictForm').addEventListener('submit', function(e){
        e.preventDefault();
        var a = selA.value, b = selB.value, date = root.querySelector('#matchDate').value;
        if (a === b){ alert('Pick two different teams.'); return; }
        run(root, a, b, date);
      });
    }
  };
})();
