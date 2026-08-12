/* views/detail.js — the detailed "for the nerds" page (#/detail?home=..&away=..&date=..).
 * Everything the simple card hides: all three W/D/L sources (Poisson, XGBoost,
 * bookmaker), the full scoreline probability heatmap, and the raw feature values.
 * Opened from the "Full analysis" link on a prediction. */
window.StatXI = window.StatXI || {};
StatXI.views = StatXI.views || {};

(function () {
  var api = StatXI.api;

  function pctOrDash(x){ return (x === null || x === undefined || isNaN(x)) ? '—' : Math.round(x * 100) + '%'; }
  function valOrDash(x){ return (x === null || x === undefined) ? '—' : x; }
  function spinner(){ return '<div class="loading"><span class="spinner"></span> Loading analysis…</div>'; }
  function errorBox(m){ return '<div class="loading">⚠ ' + m + '</div>'; }

  // --- 1) every W/D/L source as one comparison table ---
  function modelRow(name, w){
    if (!w) return '<tr class="muted"><td>' + name + '</td><td colspan="3">no line for this match</td></tr>';
    return '<tr><td>' + name + '</td><td>' + pctOrDash(w.home_win) + '</td>' +
           '<td>' + pctOrDash(w.draw) + '</td><td>' + pctOrDash(w.away_win) + '</td></tr>';
  }
  function cmpTable(d){
    return '<table class="cmp"><thead><tr><th>Model</th><th>' + d.home + '</th>' +
      '<th>Draw</th><th>' + d.away + '</th></tr></thead><tbody>' +
      modelRow('XGBoost (ML)', d.xgb) +
      modelRow('Poisson (stats)', d.poisson) +
      modelRow('Bookmaker', d.bookmaker) +
    '</tbody></table>';
  }

  // --- 2) scoreline heatmap: grid[homeGoals][awayGoals] ---
  function heatmap(d){
    var g = d.grid, max = 0, mi = 0, mj = 0;
    for (var i = 0; i < g.length; i++) for (var j = 0; j < g[i].length; j++)
      if (g[i][j] > max){ max = g[i][j]; mi = i; mj = j; }

    var cols = g[0].length;
    var html = '<div class="heat" style="grid-template-columns:auto repeat(' + cols + ',1fr)">';
    html += '<div class="heat-corner">' + d.home + ' ↓ / ' + d.away + ' →</div>';
    for (var a = 0; a < cols; a++) html += '<div class="heat-hdr">' + a + '</div>';
    for (var h = 0; h < g.length; h++){
      html += '<div class="heat-hdr">' + h + '</div>';
      for (var aw = 0; aw < g[h].length; aw++){
        var v = g[h][aw], alpha = max ? (v / max) : 0;
        var hi = (h === mi && aw === mj) ? ' hi' : '';
        var txt = v >= 0.01 ? Math.round(v * 100) : '';
        html += '<div class="heat-cell' + hi + '" style="background:rgba(16,185,140,' + alpha.toFixed(3) + ')" ' +
                'title="' + h + '–' + aw + ': ' + (Math.round(v * 1000) / 10) + '%">' + txt + '</div>';
      }
    }
    return html + '</div>';
  }

  // --- 3) the raw signals feeding the models ---
  function featRow(label, hv, av){
    return '<tr><td>' + valOrDash(hv) + '</td><th>' + label + '</th><td>' + valOrDash(av) + '</td></tr>';
  }
  function featTable(d){
    var f = d.features;
    return '<table class="feat"><thead><tr><th>' + d.home + '</th><th></th><th>' + d.away + '</th></tr></thead><tbody>' +
      featRow('Rolling form', f.home_form, f.away_form) +
      featRow('H2H win share', f.h2h_home, f.h2h_away) +
      featRow('Goals scored (avg)', f.home_gs, f.away_gs) +
      featRow('Goals conceded (avg)', f.home_gc, f.away_gc) +
    '</tbody></table>';
  }

  function renderDetail(d){
    return '<a class="back" href="#/predict">← back to predict</a>' +
      '<div class="detail-head">' +
        '<div class="team"><div class="flag">' + StatXI.flag(d.home) + '</div><div class="name">' + d.home + '</div></div>' +
        '<div class="vs">' + d.scoreline + '</div>' +
        '<div class="team"><div class="flag">' + StatXI.flag(d.away) + '</div><div class="name">' + d.away + '</div></div>' +
      '</div>' +
      '<div class="detail-date">as of ' + d.date + ' · neutral venue · xG ' + d.xg_home + ' / ' + d.xg_away + '</div>' +
      '<h3>Win / draw / loss — every source</h3>' + cmpTable(d) +
      '<h3>Scoreline probabilities</h3>' + heatmap(d) +
      '<h3>The signals behind it</h3>' + featTable(d) +
      '<div class="foot">bookmaker shown only where a real de-vigged odds line exists for this exact fixture</div>';
  }

  StatXI.views.detail = {
    render: function(){ return '<section class="detail"><div id="detail-body">' + spinner() + '</div></section>'; },
    mount: function(root){
      var p = StatXI.router.query();
      var body = root.querySelector('#detail-body');
      if (!p.home || !p.away){
        body.innerHTML = errorBox('Open this from a prediction — run one on the Predict page, then “Full analysis”.');
        return;
      }
      api.getDetail(p.home, p.away, p.date)
        .then(function(d){ body.innerHTML = renderDetail(d); })
        .catch(function(e){ body.innerHTML = errorBox(e.message || 'No analysis available.'); });
    }
  };
})();
