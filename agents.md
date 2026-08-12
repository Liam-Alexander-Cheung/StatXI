# AGENTS.md

Guidance for Claude Code when working in this repository.

## What this project is

**StatXI** — a Jugend forscht 2026/27 entry: a general football
match-outcome prediction project, not tied to one tournament. UEFA Euro 2028
is the project's current milestone/goal, not its identity — the actual
deliverable is a general prediction approach, backtested against real past
tournaments, that happens to have Euro 2028 as its forward-looking test case
(see "Important scoping fact" below for why Euro 2028 specifically can only
ever be a hypothetical run, not a live prediction).
Repo: github.com/Liam-Alexander-Cheung/StatXI.

Two prediction targets, deliberately built with two different methods:

- **Win / Draw / Loss** — XGBoost classifier. This is the genuine ML part
  of the project: it learns nonlinear feature interactions (form × squad
  quality × tournament stage × home advantage) that can't be hand-specified.
- **Scoreline** — Poisson simulation (attack/defence strength fitting +
  Monte Carlo). This is classical statistics, not machine learning — goals
  are count data, Poisson is the textbook-correct distribution, and there's
  no shame in that half of the project not being "AI."

The judges' hook is comparing this model's accuracy against (1) a naive
baseline (always predict the favorite) and (2) bookmaker implied
probabilities, backtested against real historical tournaments. Using the
right tool per sub-problem (stats where stats is correct, ML where ML
earns its place) is the actual argument for methodological rigor — not
"more AI is more impressive."

**Team:** built by Liam, with a non-technical partner, Tomás, who is
deliberately kept off this repo (doesn't code). Tomás owns domain
sanity-checking, literature review, non-technical report sections, poster
design, manual data verification, and project-management tracking.

**Deadlines that actually matter:**
- Jugend forscht registration: November 2026
- Projektbeschreibung (written report) due: January 2027
- Regionalwettbewerb (judging): February/March 2027

**Important scoping fact, already settled — don't relitigate it:** real
Euro 2028 squad data won't exist until roughly May 2028 (major tournaments
announce final squads 3-4 weeks before kickoff), which is over a year
*after* this project's deadlines. The deliverable is NOT a live Euro 2028
prediction — that's structurally impossible on this timeline. It's (1)
backtested accuracy against real past tournaments (Euro 2016/2020/2024,
World Cup 2022), and (2) a clearly-labeled hypothetical run using
projected qualification scenarios, not presented as a final prediction.

## Architecture

**Data layer:** SQLite, `data/statxi.db` (never committed — regenerable,
see "Practical notes" below). Three areas:
- `matches` — cleaned historical match results (Kaggle source, 1990+)
- `former_names` — historical team-rename lookup table
- `tournaments` → `squads` → `players` — relational squad data scraped
  from Wikipedia, real foreign keys, 18 tournaments / 432 squads / 10,038
  player-tournament rows

**Prediction layer:** XGBoost W/D/L classifier + Poisson/Monte Carlo
simulator — both built and validated (see `src/models/` under "What's
built and verified").

**Web layer:** Flask backend (`webapp/`) exposing feature data via a
simple JSON API, consumed by a plain HTML/JS frontend. Built incrementally,
one feature at a time — five features live now (rolling form, h2h, goal
trend, squad age/depth, team chemistry), see "What's built and verified" below.

## What's built and verified (as of the last working session)

### `src/data_pipeline.py`
- `load_raw_matches`, `load_former_names` — query SQLite (migrated from
  CSV; migration verified byte-identical against the pre-migration
  pipeline before being trusted)
- `load_squads` — joins `players` → `squads` → `tournaments` into one flat
  DataFrame, same shape/pattern as `load_raw_matches`
- `clean_matches` — applies the 1990 cutoff (`CUTOFF_YEAR`), drops
  unplayed/future fixtures (`NaN` scores), team-name normalization
  (currently inert against this specific Kaggle source — it already
  pre-normalizes — but retained defensively for future data sources; see
  methodology), excludes CONIFA (non-FIFA) matches
- Match weighting: `importance_weight` (tournament-tier lookup, built from
  the dataset's actual 149 distinct tournament values, not guessed),
  `recency_weight` (exponential decay, floored at `min_weight=0.05` so
  matches near the 1990 cutoff aren't crushed to near-zero), combined in
  `add_match_weights`
- `fetch_market_value_history`, `search_player_id` — Transfermarkt
  scraping via its internal, undocumented `ceapi` and `schnellsuche`
  endpoints (reverse-engineered via browser dev tools — see methodology
  for the ToS gray-zone decision). **Currently blocked**: as of
  2026-07-27, Transfermarkt's WAF challenges every request from this
  environment (search, market-value, and plain profile-page fetches all
  return an empty `202` — see methodology before assuming these work)

### `src/features.py`
- `rolling_form` — weighted win rate, trailing 10-year window
- `head_to_head_record` — weighted win rate between two specific teams,
  full history (no fixed window — meetings are often too sparse for a
  10-year cutoff to be meaningful)
- `goal_trend` — weighted goals scored / conceded / differential, trailing
  10-year window
- `squad_age_depth` — mean age plus per-position (GK/DF/MF/FW) player
  counts and proportions for a team's squad at one tournament — squad-
  scoped, not date-scoped like the three above, since a tournament squad
  is a single fixed list
- `team_chemistry` — club-cohesion of a team's squad at one tournament
  (largest/two-club spine, Herfindahl concentration, same-club pair
  ratio, distinct-club count) from the already-populated `club` column —
  squad-scoped, era-correct with zero scraping (v2 Workstream A, done)
- `transfer_value_delta_z` — returns **raw** market values at two points
  in time, deliberately NOT z-scored inside the function; z-scoring
  belongs at the squad-cohort level, done by the caller

All six are tested against real, checkable football knowledge (Germany
vs. San Marino, Argentina vs. Brazil, Kimmich vs. Musiala market-value
trajectories, Germany's and France's actual tournament squads, Germany
2014's Bayern bloc vs. Cameroon 2022's 26-clubs-for-26-players scatter) —
not just "the code runs."

### `src/name_matching.py`
- `normalize_name` — diacritic/punctuation-free lowercase token string via
  Unicode NFKD (Müller→muller, İlkay→ilkay, Łukasz→lukasz), strips "(c)"
  captain markers, reorders "Surname, Forename". `name_similarity` —
  `rapidfuzz` token-set ratio in [0,1]. Verified on real DB names; source-
  agnostic string half of the FIFA-rating matching (v2 Workstream C). The
  blocking/scoring matcher is not built — it needs the rating dataset,
  which isn't acquired yet.
- **New dependency `rapidfuzz`** added with user sign-off (requirements.txt).

### `src/database.py` + schema
- `get_connection()` — SQLite connection helper
- Relational schema: `tournaments` (18 rows) → `squads` (432 rows) →
  `players` (10,038 rows), real `FOREIGN KEY` constraints
- **`PRAGMA foreign_keys = ON` must be set explicitly per connection** —
  SQLite does not enforce foreign keys by default
- `players.transfermarkt_player_id` column exists but is `NULL` for every
  row — the name-matching/disambiguation work below hasn't happened yet

### `webapp/`
- Flask backend: `/api/teams`, `/api/tournaments`, `/api/rolling-form`,
  `/api/h2h`, `/api/goal-trend`, `/api/squad-age-depth`,
  `/api/team-chemistry`
- Both match data and squad data are cached and **warmed at server
  startup**, not lazily on first request — the full clean/normalize
  pipeline takes ~14s, and a live user should never be the one who pays
  that cost
- `templates/index.html` — five features live: rolling form, head-to-head
  record, goal trend, squad age/depth, team chemistry. Each has its own dropdown(s), a
  one-line description of what it computes, and a result area, all
  dynamically populated from the real dataset (not hardcoded) and wired
  to the API via `fetch()`. No new model logic lives in the web layer —
  it's a thin display layer over functions already proven in
  `src/features.py`
- **Frontend redesign in progress (as of 2026-08-12):** the current
  `index.html` is the feature-explorer; a full visual redesign is
  underway. Static, self-contained style mockups live in
  `webapp/previews/` (open `previews/index.html`) — these are throwaway
  vibe comparisons with *mock* numbers, not wired to the API. Once a
  design is chosen, the chosen one gets built against the real endpoints
  (including a real prediction card now that the models below exist).

### `src/models/` — the prediction models (both real, not stubs)
Run any of these via the Makefile (`make <name>`) — see Practical notes.
- **XGBoost Win/Draw/Loss** — `build_matrix.py` → `train_wdl.py` →
  `evaluate_wdl.py`, with `multi:softprob` probability output and a strict
  **temporal** split (train `<2014`, val `2014–2016`, test `2016+`) so the
  backtest block covers Euro 2016 / WC 2018 / Euro 2020 / WC 2022 / Euro
  2024. Tuned configs saved (`wdl_xgb.json`, `best_*_config.json`);
  `walk_forward.py` is the retrain-before-each-tournament variant.
- **Poisson / Dixon-Coles scoreline model** — `poisson.py` (fits per-team
  attack/defence strengths by MLE, time-decay weighted, with the
  Dixon-Coles low-score `rho` correction) + `poisson_eval.py` (Phase 4:
  Poisson vs XGBoost vs bookmaker, CI-validated per-match). Phase 5's
  `rating_gap` covariate ablation was tried and **kept off** (redundant
  with existing signal) — don't re-add it without a backtest showing it
  helps.
- **Monte Carlo tournament simulator** — `montecarlo.py` +
  `montecarlo_eval.py`. Fits strengths strictly *before* a tournament,
  simulates group + knockout stages (20k sims ~0.5s, seeded/reproducible),
  returns per-team P(reach round). **WC 2022 backtested** and beats the
  no-skill base-rate baseline (Brier 0.105 vs 0.127, log-loss 0.325 vs
  0.401). Two honest limitations documented in methodology.md: only ≤4
  tournaments of evidence, and no outright-winner odds market to calibrate
  P(win trophy) against.

## What's NOT done yet

1. **`u21_weighted_minutes_z`** — second prodigy z-score. Needs
   minutes-played data weighted by opponent strength. Not started.
2. **`per90_vs_cohort_z`** — third prodigy z-score. Needs StatsBomb
   per-90 stats. Coverage gaps across leagues/seasons are a known,
   undocumented risk — audit before building on it.
3. **Tournament stage weighting** (group vs. knockout) — distinct from
   the competition-*type* importance tiers already built. Not built.
4. **Transfermarkt ↔ Wikipedia player linking** — `search_player_id`
   works and is proven to surface real ambiguity (a "Silva" search
   returns 8 distinct real players), but nothing yet cross-references a
   search result's club/nationality against what Wikipedia already
   provided to auto-resolve matches with confidence. **Paused**: blocked
   by Transfermarkt's WAF as of 2026-07-27 — see methodology.md and the
   `src/data_pipeline.py` note above before resuming. Until this exists,
   don't assume a name match is correct without a human checking it.
5. ✅ **XGBoost classifier — DONE.** Trained, tuned, temporally
   validated. See `src/models/` above.
6. ✅ **Poisson simulation — DONE** (Dixon-Coles, CI-validated per-match).
   The deferred squad-quality covariate was tried as Phase 5's
   `rating_gap` ablation and **kept off** — redundant with existing
   signal, exactly as the "don't build until a backtest shows it helps"
   note predicted. See `src/models/` above.
7. ✅ **Monte Carlo tournament simulator — DONE**, WC 2022 backtested and
   beating the base-rate baseline. See `src/models/` above.
8. **Backtesting — partially done.** Per-match (Poisson vs XGBoost vs
   bookmaker) is validated, and WC 2022 has a full pre-tournament
   tournament backtest. Still to do: the same tournament-level backtest
   run across Euro 2016/2020/2024 (only WC 2022 has a config so far —
   `src/tournaments/wc2022.json`), so the trophy-winner claim rests on
   more than one tournament.
9. **Frontend redesign + real prediction card.** The five feature tools
   are live; the visual redesign is underway (see `webapp/previews/`).
   The planned "AI insight" / prediction section can now show **real**
   model output (the models in #5–7 exist) — so it's no longer blocked on
   "don't fabricate numbers," but it must be wired to the actual endpoints,
   not hardcoded.
10. **Projektbeschreibung** — the actual 10-15 page written report, due
    January 2027. `reports/methodology.md` is real material for this, not
    a substitute for it.
11. **Poster & presentation** — Tomás's domain, blocked on real backtest
    results existing.
12. **Jugend forscht registration** — due November 2026. Just a form, but
    a real hard date.

## How to write code for this project

- **One step at a time.** Build one function, explain it, verify it
  against a real, checkable test case, then move to the next. Don't
  generate a wholesale finished module in one pass.
- **Explain every new concept as it's introduced** — a new library, a
  new Python idiom, a new SQL/regex concept — as if the reader is
  learning it, not just approving it.
- **Add inline code comments proactively.** Don't rely on chat-external
  explanation alone; comments should live with the code they explain.
- **Verify, don't assume.** Re-run known test cases after any refactor
  (e.g. the Germany/San Marino `rolling_form` sanity check) instead of
  trusting that a diff "looks right." Multiple real regressions in this
  project were only caught this way — including a script that silently
  deleted a function, and one that destroyed its own source table
  mid-run.
- **Never fabricate data.** Missing data returns `None`/`NaN` explicitly,
  never a guessed placeholder. Applies to unplayed matches, missing squad
  data, missing market values — everywhere.
- **When something breaks, read the actual error, don't paraphrase it.**
  Check the real state of a file/database/variable directly (`cat`,
  `grep`, a `SELECT`) rather than assuming based on what should be true —
  several real bugs this project hit only became findable this way (an
  accidentally-emptied CSV, an unsaved file, a stale server process).
- **Prefer concrete, numeric answers over vague reassurance.** Push back
  immediately on an incorrect assumption rather than deferring to it.
- **Document real dead ends, bugs, and findings in
  `reports/methodology.md`** as they happen. This project's write-up
  leans on an honest account of the actual engineering process — what
  broke, why, what was learned — not a highlights reel where nothing
  ever went wrong.

## Practical notes

- Python 3.9, venv at `./venv` — activate with `source venv/bin/activate`
  before running anything.
- **`make <name>` runs any script without activating the venv** (see the
  `Makefile`). Each recipe calls `venv/bin/python` directly and knows the
  right invocation — model/eval scripts need `python -m src.models.<x>`
  from the repo root, the webapp needs `python -m webapp.app`. Bare `make`
  prints the menu; `make webapp` starts Flask on :5001, `make kill` frees
  a stuck port. This is now the preferred way to run things.
- **Never commit derived/regenerable data**: `data/statxi.db`,
  `squads_flat_backup.csv`, and everything under `data/raw/` are
  gitignored on purpose. Rebuild via `migrate_to_db.py`,
  `scrape_all_squads.py`, and `build_squad_schema.py` — don't try to
  restore these from git history, they were never there.
- **Branch structure:** `main` (stable — data pipeline only) →
  `features` (in-progress feature engineering + squad schema work) →
  `webapp` (branched from `features`, since it imports functions that
  only exist there — needs periodic `git merge features` to avoid
  drifting stale).
- macOS, VS Code integrated terminal. The author is actively learning
  git, Python, and SQL through this project — prefers being walked
  through what a command does and why before running it, not having
  things run autonomously without explanation.
- `reports/methodology.md` is the single most useful file to read before
  touching related code — it contains the actual reasoning behind
  non-obvious decisions (why 1990 as a cutoff, why squad features are
  tournament-scoped, why the prodigy composite feeds XGBoost as three
  separate z-scores instead of one hand-weighted number, and more).
