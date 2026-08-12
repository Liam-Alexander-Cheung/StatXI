# Plan: Monte Carlo tournament simulator (StatXI)

*Plan for the next working session — written 2026-08-10, after Poisson Phases 1–5
landed. Start at Phase A (needs no data); Phase B is the gating source-data step.*

## Context

The Poisson/Dixon-Coles scoreline model (Phases 1–5) is complete: it predicts a
full scoreline distribution for any single match, opponent-adjusted, and beats the
XGBoost model on the broad block. The last piece of the classical-statistics half —
and ToDo items 7–8 — is the **Monte Carlo tournament simulator**: play a whole
tournament thousands of times to turn per-match scoreline distributions into
per-team probabilities of reaching each round and winning the trophy. This is what
lets the project answer "who wins the tournament?", the backtest deliverable, and
(eventually) the clearly-labelled hypothetical Euro 2028 run.

**The one hard new cost (confirmed by exploration): tournament STRUCTURE is not in
the data.** The `matches` table is only `date/teams/scores/tournament/neutral` — no
stage, group, or bracket column; the roster schema holds only squads; and penalty
knockouts are stored as draws with no winner recorded. So the group draw + bracket
wiring for any simulated tournament must be **hand-encoded** (small, checkable
against Wikipedia) and checked in as *source* data. Everything else reuses the
Poisson layer. **Decision (settled): build + validate on the FIFA World Cup 2022
first** — the clean 32-team format (8 groups of 4, top 2 advance, standard 16-team
bracket). Euro's messier 24-team "best-thirds" format is a later extension.

**Also settled by exploration (shapes validation):** there is **no outright /
tournament-winner odds market** anywhere in the project (odds are per-match 1X2
only). So the simulator's per-team "wins the tournament" probabilities can be
validated **only against actual outcomes**, never the market — and with just a
handful of past tournaments that is a genuine, honest limitation to state, not
paper over. The per-match 1X2 side is already validated (Poisson Phase 4).

### What exploration confirmed (reuse, don't reinvent)
- **The atomic match predictor exists.** `poisson.predict_match(strengths, home,
  away, neutral) -> {lambda_home, lambda_away, p_home/p_draw/p_away, grid, ...}` and
  `poisson.fit_dixon_coles(matches, reference_date=t_start)` give a fitted
  `strengths` dict; `scoreline_matrix` + `apply_dixon_coles` build the corrected
  grid. All in `src/models/poisson.py`.
- **"Before tournament T" fitting** is just `matches[matches.date < t_start]` with
  `reference_date=t_start` (the walk_forward pattern, `walk_forward.py:80-102`);
  `t_start` = earliest match date of that edition. Reuse verbatim so the forecast
  only ever uses pre-tournament knowledge.
- **A tournament's match set** is recoverable by tournament label + year
  (`match_tournament_edition`, `data_pipeline.py:151`); `neutral` is per-match. Team
  names must match the DB's normalised spelling — validate the config against
  `strengths["attack"]` keys and fail loudly on a miss (reuse `resolve_team_name` /
  `normalize_team_names` in `data_pipeline.py` if needed).
- **Randomness convention:** `rng = np.random.default_rng(seed)` threaded through,
  `rng.poisson(...)` / `rng.choice(...)`. numpy 2.0.2 confirmed. No existing sim
  code — greenfield. No new dependency.

## Phases (each small + verifiable, per the project's rules)

### Phase A — single-match Monte Carlo (the sampler; NO structure data needed)
New `src/models/montecarlo.py`. Prove the sampler reproduces the analytic model
before simulating anything bigger.
- `sample_scorelines(grid, n, rng)` — draw `n` (home_goals, away_goals) pairs by
  sampling CELLS from the flattened **Dixon-Coles-corrected** grid
  (`rng.choice(len, p=grid.ravel()/grid.sum())`). *Critical subtlety:* two
  independent `rng.poisson(λ_h)`, `rng.poisson(λ_a)` reproduce only the UNCORRECTED
  product — to match the real model you must sample the corrected grid.
- **Verify:** for a representative match (Spain vs Germany λ's from a real fit) the
  MC P(H/D/A) matches analytic `wdl_from_grid(grid)` within Monte-Carlo error
  (~2·√(p(1−p)/n), n=1e5) → agreement to ~2dp. Separately, `rho=0` sampled via
  `rng.poisson` must match `scoreline_matrix` (independence sanity).

### Phase B — hand-encode WC 2022 structure (the source-data step)
Check in `src/tournaments/wc2022.json` (SOURCE data, committed — unlike the
gitignored derived DB): `name`, `year`, `start_date`, `format:"wc32"`, `groups`
(8 × 4 team names, DB spelling), `qualify_per_group:2`, `bracket` (R16 slot wiring
1A–2B, 1C–2D, …), and `actual` (champion Argentina, runner-up France, SF Croatia +
Morocco, …) for validation only. A tiny loader validates every team resolves to a
`strengths` key.
- **Verify:** all 32 names resolve against a real fit (no unseen team); group/
  bracket counts are right (8×4=32, 16-team KO); Tomás can cross-check the draw
  against Wikipedia (his manual-verification domain).

### Phase C — group-stage simulator
- `simulate_group(teams, strengths, rng, n)` — play all C(4,2)=6 intra-group
  pairings (schedule irrelevant to final standings), neutral venue, 3/1/0 points,
  rank by points → goal difference → goals for (simplified vs FIFA's head-to-first;
  document the simplification), return the top-2 qualifiers per simulation.
  Vectorise across the `n` sims (fixed pairings ⇒ one batched sample per fixture).
- **Verify on WC 2022:** the sim's most-probable qualifiers per group broadly match
  who actually advanced (e.g. Netherlands/Senegal from A), and group-winner
  probability tracks team strength. Not every upset will be predicted — check the
  distribution is sane, not that it's clairvoyant.

### Phase D — knockout + full-tournament simulator
- `simulate_knockout(slots, strengths, rng, n)` — single elimination; each tie
  samples a scoreline, a drawn tie resolves by a **~50/50 shootout coin-flip**
  (documented placeholder; a small shootout model is a later refinement), winner
  advances per the bracket wiring.
- `simulate_tournament(config, strengths, n=10000, seed=0) -> DataFrame` of per-team
  P(reach R16 / QF / SF / final / win).
- **Verify:** P(win) ≤ P(final) ≤ P(SF) ≤ … (monotone by construction — assert it);
  champion probabilities sum to 1; Argentina/France/Brazil/England near the top;
  eyeball vs reputation and vs the actual result (Argentina won). Reproducible under
  a fixed seed.

### Phase E — honest backtest / validation
- Fit strengths **before** WC 2022 (`matches[date < t_start]`,
  `fit_dixon_coles(reference_date=t_start)`) so it's a true pre-tournament forecast,
  then simulate. Optional `src/models/montecarlo_eval.py`.
- Report: did the actual champion + finalists land in the sim's top tier? A
  per-team "reached-round" Brier/log-loss vs actual across all 32. **State the
  limitation plainly:** 1 tournament here (≤4 even with Euro 2016/2020/2024 + WC
  2022), and no outright market to benchmark against, so tournament-winner
  *calibration* cannot be strongly validated — the simulator's rigor rests on the
  per-match engine (already validated in Phase 4) plus round-level sanity.

### Later (explicitly OUT of this plan)
Euro 24-team "best-thirds" format + config; a fitted shootout model; and the
clearly-labelled **hypothetical Euro 2028** run (needs the Euro format + a projected
group draw). Scope these once WC 2022 A–E land.

## Files
- **New:** `src/models/montecarlo.py` (sampler + group + knockout + tournament),
  `src/tournaments/wc2022.json` (hand-encoded structure, **committed as source**),
  a small config loader (in `montecarlo.py` or `src/tournaments/__init__.py`), and
  optionally `src/models/montecarlo_eval.py` (Phase E backtest driver).
- **Reuse unchanged:** `src/models/poisson.py` (`fit_dixon_coles`, `predict_match`,
  `scoreline_matrix`, `apply_dixon_coles`, `_match_lambdas`),
  `src/data_pipeline.py` (`load_raw_matches`, `clean_matches`,
  `match_tournament_edition`, `resolve_team_name`), the `walk_forward.py` "before T"
  slice pattern, and the `np.random.default_rng(seed)` convention.
- **Docs:** `reports/methodology.md` (design + each phase's finding, incl. the
  structure-data gap and the no-outright-odds validation limit), `ToDo.txt`.

## Verification (end to end)
- Per-phase real-football / arithmetic checks above (project non-negotiable).
- `python -m src.models.montecarlo` runs the single-match cross-check and a seeded
  WC 2022 simulation, printing per-team round/win probabilities to eyeball against
  reputation and the actual 2022 result.
- Assertions baked in: MC ≈ analytic for a single match; reach-probabilities
  monotone; champion probabilities sum to 1; every config team resolves to a fitted
  strength.

## Branch
Recommend merging `poisson` → `main` first (Phases 1–5 are complete, verified, and
pushed), then a fresh `montecarlo` branch off `main` — mirrors how `poisson` was
started off the consolidated trunk. (Alternatively continue on `poisson`; the merge
is the cleaner checkpoint.)
