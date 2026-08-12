/* views/landing.js — the intro page (#/).
 * Pure markup, no data fetching. Explains what StatXI is, carries the honest
 * "reference only" warning, and links to the predict page. */
window.StatXI = window.StatXI || {};
StatXI.views = StatXI.views || {};

StatXI.views.landing = {
  render: function(){
    return ''+
    '<section class="hero">'+
      '<h1>The match, <span>before</span> it’s played.</h1>'+
      '<p class="lead">StatXI predicts football results from history — who wins, and what the score is — '+
        'and checks itself against real past tournaments instead of just sounding confident.</p>'+
      '<a class="cta" href="#/predict">Try a prediction →</a>'+
    '</section>'+

    '<div class="warn">'+
      '<span class="ico">⚠️</span>'+
      '<span><b>Reference model — not betting advice.</b> Every number here is a statistical / '+
        'machine-learning estimate produced for research and backtesting. It is not an official or live '+
        'forecast: a real Euro 2028 prediction is impossible before the squads exist (≈ 2028), so any '+
        'future-tournament run is a clearly-labelled hypothetical.</span>'+
    '</div>'+

    '<div class="sections">'+
      '<div class="section">'+
        '<div class="step">What</div>'+
        '<h2>What is StatXI?</h2>'+
        '<p>Two questions, two methods. <b>Who wins</b> is a machine-learning classifier '+
          '(win / draw / loss probabilities). <b>What’s the score</b> is a Poisson goals '+
          'simulation — the textbook model for counting goals. Right tool per problem.</p>'+
      '</div>'+
      '<div class="section">'+
        '<div class="step">How</div>'+
        '<h2>How does it work?</h2>'+
        '<p>Pick two teams and a date. The model looks only at what was known <i>before</i> that '+
          'date — recent form, head-to-head history, squad make-up — and turns it into a '+
          'probability and a most-likely scoreline. No peeking at the future.</p>'+
      '</div>'+
      '<div class="section">'+
        '<div class="step">Why</div>'+
        '<h2>Why we built it</h2>'+
        '<p>A Jugend forscht project. The point isn’t “more AI” — it’s rigour: '+
          'statistics where goals are count data, ML where patterns are too tangled to hand-write, and '+
          'an honest scoreboard against a naive favourite-picker and bookmaker odds.</p>'+
      '</div>'+
    '</div>'+

    '<div class="foot">PREVIEW SKELETON · numbers on the next page are mock data</div>';
  }
};
