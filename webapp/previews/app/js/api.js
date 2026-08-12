/* api.js — THE DATA LAYER, and the only file that knows where numbers come from.
 *
 * Right now every function returns MOCK data after a fake network delay, so the
 * whole app runs by just double-clicking index.html (no server) and you can watch
 * the boxes load one-by-one. When we connect the real Flask backend, ONLY this
 * file changes: each function below becomes a real `fetch('/api/...')` that returns
 * the same shape. Every view keeps working untouched — that is the whole point of
 * keeping data access in one place.
 *
 * We attach everything to a single global object, window.StatXI, instead of using
 * ES `import`/`export`. Reason: ES modules are blocked over file:// by the browser,
 * and we want double-click-to-open to work. Classic <script> tags + one shared
 * namespace give us "multiple clean files" without needing a server.
 */
window.StatXI = window.StatXI || {};

(function () {
  // Display flags. Fallback is a football for anything unmapped.
  var FLAGS = {
    Germany:'🇩🇪', France:'🇫🇷', Spain:'🇪🇸', Italy:'🇮🇹', Portugal:'🇵🇹',
    Netherlands:'🇳🇱', Belgium:'🇧🇪', Croatia:'🇭🇷', Brazil:'🇧🇷', Argentina:'🇦🇷'
  };
  var TEAMS = Object.keys(FLAGS);

  // --- tiny deterministic pseudo-random (FNV-1a hash -> 0..1) ---
  // Same input string always gives the same number, so a given matchup produces
  // stable mock values instead of flickering on every click. Purely cosmetic —
  // it will be deleted the moment real data arrives.
  function seed(s){ var h=2166136261; for(var i=0;i<s.length;i++){ h^=s.charCodeAt(i); h=Math.imul(h,16777619); } return h>>>0; }
  function unit(s){ return (seed(s)%1000)/1000; }
  function between(s,lo,hi){ return lo + unit(s)*(hi-lo); }

  // --- fake latency: resolve `value` after `ms` milliseconds ---
  function later(ms,value){ return new Promise(function(res){ setTimeout(function(){ res(value); }, ms); }); }

  StatXI.api = {
    flag: function(team){ return FLAGS[team] || '⚽'; },

    // fast — just fills the dropdowns
    getTeams: function(){ return later(150, TEAMS.slice()); },

    /* The four INDEPENDENT box requests. The delays differ ON PURPOSE so the
       boxes finish at different times and pop in one after another. */

    // 700ms — usually first
    getRollingForm: function(a,b,date){
      return later(700, { a: between(a+'|form',0.40,0.80), b: between(b+'|form',0.40,0.80) });
    },
    // 1050ms
    getH2H: function(a,b,date){
      var aWin = between(a+'>'+b,0.30,0.55);
      var bWin = between(b+'>'+a,0.25,0.50);
      return later(1050, { aWin: aWin, bWin: bWin });
    },
    // 1500ms — the headline prediction
    getPrediction: function(a,b,date){
      var sa = between(a+'|str',0.35,0.75), sb = between(b+'|str',0.35,0.75), sd = 0.26;
      var tot = sa+sb+sd;
      var gh = Math.round(between(a+b+'|gh',0.6,2.2));
      var ga = Math.round(between(b+a+'|ga',0.4,2.0));
      return later(1500, { home_win: sa/tot, draw: sd/tot, away_win: sb/tot, scoreline: gh+' – '+ga });
    },
    // 1850ms — usually last. Expected goals = each side's Poisson λ for this
    // matchup (the unrounded goals behind the scoreline). Date-scoped, so unlike
    // squad chemistry it needs no tournament — which is why it replaced it.
    getExpectedGoals: function(a,b,date){
      return later(1850, { a: between(a+'|xg',0.70,2.40), b: between(b+'|xg',0.70,2.40) });
    }
  };
})();
