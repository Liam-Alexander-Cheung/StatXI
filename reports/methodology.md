## Team name normalization

International football has genuine historical name changes that break
naive team-identity tracking: Czechoslovakia's pre-war "Bohemia" era,
Zaïre → DR Congo, Swaziland → Eswatini, Macedonia → North Macedonia, among
others. Left unhandled, a model would treat these as unrelated teams with
artificially shortened histories.

Built `resolve_team_name()` — a date-range lookup against `former_names.csv`
(current name, former name, start/end dates) — plus `normalize_team_names()`
to apply it across the full match dataset. Verified correctness against the
Czechoslovakia/Bohemia case, which has three distinct former-name windows
(1903–1919, 1939–1945, 1993), confirming the resolver disambiguates by date
correctly, not just by name.

**Finding:** empirically testing this against the actual data source
(Kaggle, martj42 international-results dataset) showed the raw data already
applies current names retroactively across its entire history — confirmed
independently via three separate historical renames (Macedonia, Swaziland,
Zaïre), each returning zero matches under their *former* name anywhere in
the raw dataset. The normalization function is therefore currently inert
against this specific source.

**Decision:** retained rather than removed. Reasoning: (1) defensive
against any future data source that doesn't pre-normalize — e.g. if
StatsBomb or Transfermarkt data is added later for the prodigy signal;
(2) the resolution logic is independently validated and correct regardless
of whether this dataset currently exercises it; (3) documenting a
"built it, tested it, discovered it wasn't needed for this specific input,
kept it as a safeguard" reasoning chain is a more honest and complete
account of the engineering process than silently deleting evidence of the
investigation.

## Squad-based feature scoping

Match-level features (rolling form, h2h, goal trend) work across all 32,140
cleaned matches since squad composition doesn't matter for them. Squad-based
features (prodigy signals, squad age/depth) genuinely can't — squad lists
aren't reliably documented for routine matches since 1990, only for major
tournaments. Decision: these features are computed only at the
tournament-squad level (a team's announced squad for a specific Euro/World
Cup), shared across every match that team plays within that tournament,
rather than varying match-by-match like the other features. Roughly 300+
squad-tournament combinations across all Euros/World Cups since 1990,
versus 32,140 match rows — a fundamentally different scale of data problem.

## Data sourcing: Wikipedia + Transfermarkt

Squad lists (name, position, age-at-tournament, caps, club) are sourced from
Wikipedia's per-tournament "20XX squads" pages — clean, consistent structure
across decades, frozen at the tournament date by design (unlike Transfermarkt
profile pages, which only show current data), and no ToS concerns.

Market value history is sourced from Transfermarkt's internal, undocumented
`ceapi` endpoints — reverse-engineered via browser dev tools, not a published
API. This is a documented, conscious ToS gray-zone decision: Transfermarkt's
terms technically restrict automated access; the practical risk for a
non-commercial student research project pulling publicly visible data is low,
but not zero.

**Correction (caught during a later session, not fixed yet):** this section
previously claimed "every fetched page is cached locally to avoid redundant
re-scraping" as a mitigation. That was aspirational, not real — there is no
caching code anywhere in the repo; `search_player_id` and
`fetch_market_value_history` both call `requests.get()` directly, every time.
Leaving the incorrect claim in place would have been worse than flagging it:
this file is meant to be an honest account of the actual engineering process,
not a highlights reel. Actually building the cache is worth doing regardless
of the block below, since it would reduce redundant load on Transfermarkt
either way.

**Finding: Transfermarkt is currently blocking all requests from this
environment.** Checked directly (not assumed) on 2026-07-27: `search_player_id`,
`fetch_market_value_history`, and even a plain profile-page GET all returned
`HTTP 202` with an empty body and header `x-amzn-waf-action: challenge` —
AWS WAF (Transfermarkt's bot-protection layer) actively challenging every
request, not a bug in this project's code. This blocks both the planned
Transfermarkt↔Wikipedia player-linking work and any further
`transfer_value_delta_z` calls until it clears — unknown whether this is
IP-based, time-based, or a permanent tightening of their bot defenses; needs
re-checking before resuming that work, ideally from a different network as a
first diagnostic step.

## Name-matching risk (Transfermarkt search)

Player IDs are resolved from Wikipedia names via Transfermarkt's `schnellsuche`
search endpoint. Tested against "Silva" — a deliberately ambiguous case —
and confirmed the disambiguation problem is real: 10 distinct results
returned, correctly representing 8 different real players. Two results
(Bosingwa, Derlei) had no visible connection to "Silva" in their displayed
name at all — Transfermarkt's search appears to match against full/birth
names not shown in the short public display name, meaning a visual "does
this look right" check on a result is not sufficient. Planned mitigation:
cross-reference each search result's club/nationality against what
Wikipedia's squad table already provides, before accepting a match
automatically.

## Finding: prodigy signal magnitude is tournament-dependent

Tested `transfer_value_delta_z` against two contrasting cases ahead of
Euro 2024: Joshua Kimmich (established veteran, in a form dip) showed a
€-30m delta; Jamal Musiala (widely regarded as a generational talent)
showed only €+10m — a much smaller signal than expected. Investigation:
Musiala's actual breakout occurred earlier (2020-2022), and by the 12
months before Euro 2024 the market had already fully priced in his talent.
This is not a bug — it demonstrates the prodigy flag is sensitive to
*which* tournament a player's history is evaluated against, and a player
can move from "flagged" to "already-priced-in, no longer flagged" within a
single tournament cycle. Worth testing against Euro 2020 (Musiala's actual
breakout window) or a still-emerging talent (e.g. Lamine Yamal ahead of
Euro 2024) as a sharper positive control case.

## Migration from CSV to SQLite

Initial motivation was "CSVs are inefficient at scale" — worth correcting
that reasoning explicitly, since it isn't actually true at this size.
32,140 rows parses via `pd.read_csv` in well under a second; raw
performance was never the real bottleneck. The genuine reason to migrate
is relational structure: upcoming squad-level data (tournaments → squads →
players → market value history) has real foreign-key relationships that
flat CSVs handle poorly, requiring hand-written pandas joins in place of
what a database does natively. SQLite was chosen over Postgres/MySQL as
the right tool for this scale — single-file, serverless, part of Python's
standard library, no infrastructure to run or maintain.

Design choice: the migration only touches how data enters the pipeline
(`load_raw_matches`, `load_former_names` now query SQLite instead of
reading CSVs), while every downstream function (`clean_matches`,
`rolling_form`, `head_to_head_record`, `goal_trend`) is completely
untouched — they operate on the returned DataFrame regardless of its
source. This is deliberate: it contained the risk of the migration to two
functions instead of rewriting an entire proven pipeline at once.

**Caught during verification:** re-running the same Germany/San Marino
proof cases used throughout this project (rather than just checking that
the new code "looked right") surfaced a real regression — `resolve_team_name`
had been accidentally dropped entirely while rewriting the surrounding
functions, breaking `normalize_team_names` downstream with a `NameError`.
Fixed by restoring the function; identical output (`(32140, 9)`, Germany
0.7936, San Marino 0.0) confirmed against pre-migration numbers afterward.
Worth noting as a concrete example of why re-running known test cases after
a refactor matters more than reading a diff — the deletion wouldn't have
been visually obvious in a quick review, but it broke observable behavior
immediately.

The database file itself (`data/statxi.db`) is not committed to the
repository — same reasoning as the raw CSVs it replaced: it's regenerable
by running `migrate_to_db.py` against the source data, and derived data
doesn't belong in version control.

## Data currency: why there's no auto-refresh pipeline

Two data sources have fundamentally different lifecycles, and treating
them the same would be a mistake:

**Wikipedia tournament squad pages (1990-2024) are historical and frozen.**
These describe events that already happened; nothing meaningful changes
about Euro 2004's roster years later. Re-scraping only matters if a
structural error is suspected (Wikipedia's own markup has changed at least
once during this project — the `mw-headline` span disappeared at some
point, breaking an early heading selector) or a specific anomaly needs
re-checking. An occasional manual rerun is the right cadence, not a
scheduled job with nothing new to find.

**The Kaggle match-results dataset is genuinely live** — it already
contained unplayed July 2026 World Cup fixtures with null scores at the
time of the original download, confirmed and handled explicitly in
`clean_matches`. This data does warrant occasional manual refreshing
(before major project milestones: backtesting, final report), but not an
unattended automated scraper — a periodic manual redownload is simpler,
safer, and sufficient.

**The more important finding: there is no Euro 2028 squad data to
auto-refresh toward in the first place.** Major tournaments announce final
squads 3-4 weeks before kickoff — for Euro 2028 (June 2028), that means
roughly May 2028. This project's actual deadlines (Projektbeschreibung due
January 2027; Regionalwettbewerb February/March 2027) fall over a full
year *before* any real Euro 2028 squad will exist to scrape.

**Consequence for what this project can actually claim:** the deliverable
is not a live Euro 2028 prediction — that's structurally impossible on
this timeline. It is (1) backtested model accuracy against real historical
tournaments (Euro 2016/2020/2024, benchmarked against baseline and
bookmaker odds), and (2) a hypothetical Euro 2028 run using projected
qualification scenarios and provisional squad data, clearly labelled as
speculative rather than final. Tracking qualification scenarios as they
firm up is an explicit, ongoing task (see Tomás's workstream list).

## Squad-table anomaly: FR Yugoslavia at Euro 1992

The initial scrape of `squads` showed 9 countries for Euro 1992, when the
tournament actually had 8 participants. Investigation: Yugoslavia qualified
for Euro 1992 but was suspended from competing due to UN sanctions before
the tournament began; CIS was invited as a late replacement. Wikipedia's
page retains Yugoslavia's originally-qualified squad for historical
completeness, in a section with the same `No.`/`Pos.` table structure our
`fetch_tournament_squads` filter checks for — meaning the filter correctly
identified a real roster-shaped table, but couldn't distinguish "a squad
that actually competed" from "a squad that was named but never played,"
since that distinction isn't visible in table shape alone.

Fixed by explicitly removing the FR Yugoslavia rows for Euro 1992. Not
patched with a general rule, since this is a specific, known historical
case (late disqualification/replacement) rather than a systematic scraper
flaw — worth remembering that any tournament with a similar late
withdrawal could produce the same kind of entry, and it would only surface
if a row/country count looks suspicious enough to check by hand, as
happened here.

## Relational schema: tournaments → squads → players

Replaced the flat `squads` table (one row per player, country and
tournament as plain string columns) with a proper three-table relational
structure: `tournaments` (18 rows, one per competition), `squads` (~380
rows, one per team per tournament), `players` (10,058 rows, one per
roster entry), linked via foreign keys (`squads.tournament_id` →
`tournaments.tournament_id`, `players.squad_id` → `squads.squad_id`).

Also parsed two previously-raw string fields into real typed columns:
`"8 February 1995 (aged 29)"` split into an ISO date and an integer age;
player names had footnote markers (`*`, `†`) and `"(captain)"` suffixes
stripped, with captain status promoted to its own boolean column rather
than left as unstructured text.

Known limitation, left deliberately unresolved for now: a real person
(e.g. Kimmich) appears as multiple separate, unlinked `players` rows
across different tournaments — there is currently no way to say "these
rows are the same human." A `transfermarkt_player_id` column exists on
`players` for this purpose but is left NULL until the Transfermarkt
name-matching/disambiguation work (see "Name-matching risk" above) is
actually completed. Not faked with a guessed link.

Note: SQLite does not enforce foreign key constraints by default — `PRAGMA
foreign_keys = ON` must be set explicitly per connection, or invalid
references would be silently accepted rather than rejected.

## Bug: schema-building script destroyed its own source data mid-run

The first version of `build_squad_schema.py` read the existing flat
`squads` table into memory, then called `conn.executescript()` containing
both `DROP TABLE squads` and the new `CREATE TABLE squads` (empty,
relational). Python's `sqlite3.executescript()` commits before returning,
not after the whole calling function finishes — so the moment that call
ran, the old flat table was destroyed and replaced, permanently, before
the row-insertion loop (which uses the in-memory `flat` variable) ever
reached the data that mattered. A few lines later, the loop crashed on an
unrelated bug (a `None` player name), but the flat table was already gone
by that point regardless of the crash.

Recovered because the underlying source (Wikipedia) is still live and the
scrape is fully reproducible — re-running `scrape_all_squads.py` rebuilt
the flat table from scratch. Fix going forward: the script now writes an
independent CSV backup of `flat` (`squads_flat_backup.csv`) before running
any DROP/CREATE statements, so a mid-script crash can no longer mean data
only existed in one fragile, destroyable place. Also fixed the crash
itself: `clean_player_name` now returns `None` for non-string input
instead of raising, and the insertion loop skips and counts such rows
rather than crashing the whole run.

## Bug: Wikipedia footnote markers silently corrupted the World Cup 2002 scrape

After the fix above, the insertion loop reported skipping exactly 736
rows — suspicious specifically because 736 = 32 teams × 23 players,
matching World Cup 2002's own printed squad count exactly, suggesting one
entire tournament's data was affected rather than scattered individual
rows. Investigation confirmed: that Wikipedia page has per-country
footnote citations on its table headers (e.g. `Player[4]`, `Player[5]`,
each country citing a different footnote number), so `pd.read_html` gave
each country's table a uniquely-numbered `"Player[N]"` column instead of
a plain `"Player"` column. The roster-detection filter still correctly
identified these as real roster tables (`"No."` is a substring of
`"No.[4]"`), but `pd.concat` couldn't recognize `"Player"` and
`"Player[4]"` as the same field, silently producing ~30 near-duplicate
columns and leaving every row's plain `"Player"` column null.

Fixed by stripping trailing `[\d+]` footnote markers from every column
name immediately after parsing each table, before concatenation. Re-ran
the full 18-tournament scrape afterward — skip count dropped from 736 to
0, confirming the fix and, importantly, that no *other* tournament had a
milder, less-obvious version of the same bug that a smaller, less
suspicious skip count might have let slip past unnoticed.

## Web layer: exposing match-level features via Flask

Built incrementally, one feature at a time, per the project's stated
preference for small verified steps over a wholesale finished module:
`/api/rolling-form` first, then `/api/h2h`, then `/api/goal-trend`, each
backed by the already-proven functions in `src/features.py` and each
paired with a dropdown in `webapp/templates/index.html`. No new model
logic was written for the web layer — it's purely a thin routing/display
layer over match-level features that were already built and verified.

**Caching decision:** match data is loaded and cleaned once, at server
startup (`get_matches()` called eagerly at module load, not lazily on
first request), rather than on first request. The full clean/normalize
pipeline takes ~14 seconds; the alternative (lazy loading) would mean
whichever real user happens to load the page first eats that delay
instead of it being paid once, invisibly, before anyone connects.

**Finding: running `python webapp/app.py` directly fails, `python -m
webapp.app` from the repo root doesn't.** `app.py` imports `src.data_pipeline`
and `src.features`, but when a script is run directly, Python only adds
*that script's own directory* (`webapp/`) to `sys.path` — not the repo
root where the `src` package actually lives — so the import fails with
`ModuleNotFoundError: No module named 'src'`. Running it as a module
(`python -m webapp.app`, from the repo root) makes Python treat the
current directory as the import root instead, which resolves correctly.
Worth remembering since the failure mode gives no hint about *why* — it
looks like a missing dependency, not a module-resolution/working-directory
issue.

**API design consistency:** query parameter names deliberately match the
underlying function's own parameter names (`team_a`/`team_b` for h2h, not
`team1`/`team2` or similar) — so `app.py` and `features.py` use the same
vocabulary for the same concept, rather than translating between two
naming schemes for no reason.

**Edge case worth calling out explicitly: `team_a == team_b` in
`head_to_head_record`.** The function itself has no internal guard for a
team being compared against itself — the boolean row-mask
`(home==A & away==B) | (home==B & away==A)` can never match a real row
when `A == B` (a team can't play itself), so it silently falls through to
the same "no matches found" path as a genuinely never-met pair, returning
`NaN`. Handled at the API layer instead: `/api/h2h` checks for this
explicitly and returns a 400 ("must be different teams") rather than
letting it fall through to the existing NaN → 404 path, since "you asked
a nonsensical question" (the client's fault) and "these two teams simply
have no shared history" (not the client's fault, a real data-availability
gap) are different failures that deserve different HTTP status codes —
collapsing them into one path would mislabel a bad request as a missing-
data case.

**Verification, following the project's existing convention of checking
against real, checkable football knowledge rather than just "the code
runs":**
- Rolling form — Germany: 0.678 (a strong, plausible recent-form score).
- Head-to-head — Argentina vs Brazil: 0.525 (Argentina's win rate);
  querying the reverse direction (Brazil vs Argentina) returned 0.475 —
  confirms the two directions aren't silently symmetrized, and the two
  numbers sum to 1.0 as expected for a competitive, century-long rivalry
  with few draws skewing the total.
- Goal trend — Germany: scored 2.419, conceded 1.109, differential 1.311
  — internally consistent (2.419 − 1.109 = 1.310, matching within
  rounding).
- Error paths — missing query parameter, an unknown/never-met team, and
  identical `team_a`/`team_b` each independently confirmed to return the
  correct 400/404 and error message, never a silently fabricated number.

## Squad age & depth score

The first squad-level feature (as opposed to the match-level trio above).
"Depth" was never actually defined anywhere before this — `claude.md` and
this file both only ever mentioned it in passing (see "Squad-based feature
scoping" above) without saying what it measures. Checked the `players`
table directly before deciding: the `position` column turned out to only
have 4 distinct values across all 10,038 rows (`GK`, `DF`, `MF`, `FW`, no
NULLs, no free-text mess), which made a literal definition of depth —
how many players cover each position — both the most football-accurate
reading of the word and the cheapest to build, so that's what was built,
rather than a proxy like squad size or total caps.

**Design decision: return counts *and* proportions, not one or the
other.** Counts are directly meaningful ("Germany had 9 defenders").
Proportions (count ÷ squad size) are what's actually comparable across
squads, since official squad size isn't fixed — it's varied 23-26 players
depending on the tournament's own rules in a given year, so two squads
with equal defender *counts* can still differ in what share of the squad
that represents.

**Design decision: squad-scoped, not date-scoped.** Unlike `rolling_form`/
`goal_trend` (which filter matches by an `as_of_date`), a tournament squad
is a single fixed list — there's no "as of" question to ask about it — so
`squad_age_depth(squads, team, tournament_name)` takes a tournament name
directly instead of a date.

**Missing-squad handling:** if `team`/`tournament_name` doesn't match any
row (wrong name, or a team that simply didn't qualify for that
tournament), position counts are correctly `0` — that's a true fact about
an empty result, not a fabrication — but `mean_age` and the proportions
are `NaN`, since "0 out of 0" is undefined, not a guessed `0.0`. Confirmed
against a real case: San Marino has never qualified for a Euro, so
`squad_age_depth(squads, "San Marino", "Euro 2024")` correctly returns all
`NaN`/`0` rather than silently succeeding with garbage.

**New data-loading pattern:** added `load_squads()` to `data_pipeline.py`,
joining `players` → `squads` → `tournaments` into one flat DataFrame — the
same shape as `load_raw_matches`, so `squad_age_depth` filters a DataFrame
the same way `rolling_form` etc. filter `matches`, rather than each
squad-level feature writing its own SQL join.

**Verification**, against two real squads with known, checkable rosters:
- Germany, Euro 2024: mean age 28.538, squad size 26, GK 3 / DF 9 / MF 9 /
  FW 5 — cross-checked against a direct SQL `GROUP BY position` query
  before trusting the function, exact match.
- France, World Cup 2022 (the squad that reached the final): mean age
  26.538, GK 3 / DF 9 / MF 6 / FW 8 — a forward-heavy squad, consistent
  with that squad's actual makeup (Mbappé, Giroud, Griezmann, Dembélé
  among the front players).

**Web layer:** exposed the same way as the other three features —
`/api/squad-age-depth?team=...&tournament=...`, plus a new
`/api/tournaments` endpoint (mirrors `/api/teams`) so the frontend can
populate a tournament dropdown alongside the team one. Squad data is
cached and warmed at startup the same way match data is (`get_squads()`
alongside the existing `get_matches()`), for the same reason: nobody
should pay a load-time cost on their own first click.

## XGBoost Win/Draw/Loss classifier (v1)

The genuine ML half of the project: a gradient-boosted tree classifier that
outputs a probability over three outcomes (Home win / Draw / Away win) per
fixture. Built in three verified stages under `src/models/`: `build_matrix.py`
(assemble the training table), `train_wdl.py` (temporal split + fit),
`evaluate_wdl.py` (score against baselines).

### Target reset (the single most important decision in this section)
The originally-stated goals — ~90% winner accuracy and ~50% exact scoreline —
were re-scoped *before* any code was written, because neither is attainable and
carrying them into the report would actively damage its credibility:

- The best public models and bookmakers reach roughly **52–58%** top-1 accuracy
  on 3-way football, and **~10–12%** on exact scoreline. Football is
  low-scoring and high-variance — that is *why* upsets happen, and it caps how
  well any model can do.
- A WDL model reporting ~90% accuracy is, in almost every real case, a
  **data-leakage bug**, not a good model. An informed judge would (correctly)
  distrust the number on sight. So a high accuracy figure is treated in this
  project as a red flag to investigate, not a result to celebrate.
- "Expect the unexpected" (catching a Cape Verde-type upset) *contradicts*
  maximising accuracy — upsets are low-probability by definition. Its correct
  technical form is **calibrated probability**: a model that says "underdog
  22%" and is right 22% of the time. That is evaluated with **log-loss / Brier
  / calibration error**, not top-1 accuracy.

**Agreed success criterion:** beat a no-skill baseline on accuracy *and*
log-loss, and be well-calibrated. Not raw accuracy.

### Design decisions
- **Match-level features only, in v1** (available for all 32,140 cleaned
  matches): home & away `rolling_form`, `goal_trend` (goals scored/conceded per
  side), `head_to_head_record`, the `neutral`-venue flag, and the match's
  tournament `importance_weight`. Squad features exist only for 18 tournaments
  (NaN for ~99% of matches) and are deferred to v2 — a deliberate one-step-at-
  a-time choice, later *validated* by the tournament-level result below.
- **`neutral` is a first-class feature.** 48.5% of all matches are home wins,
  but 28% are neutral-venue where "home/away" is only listing order. Tournaments
  are largely neutral, so without this flag the model would export a home-bias
  learned from friendlies into exactly the setting we backtest.
- **A single explicit `FEATURE_COLUMNS` list is the leakage guard.** The
  training matrix keeps `home_score`/`away_score` for traceability, but the
  model is fed features by that name list *only* — never "all columns except
  the label" — so the raw scores (which *are* the answer) can never be fed in by
  accident. Verified on a sample: a built row's `home_form` equals an
  independent standalone `rolling_form` call to 6 decimals.
- **Temporal split, never random k-fold.** train `< 2014-01-01`; validation
  `2014–2016` (early-stopping only); test `2016+`. The 2016 boundary puts the
  tournaments we want to backtest (Euro 2016/2020/2024, WC 2018/2022) entirely
  in the held-out block. A random split would let the model learn from matches
  that happened *after* those it is scored on — the classic manufactured-90%
  failure mode.
- **Native NaN handling, no imputation.** XGBoost learns a per-split default
  direction for missing values, so debutant nations (Cape Verde →
  `rolling_form` returns NaN) are handled directly, honouring the project's
  never-fabricate-data rule. 152–167 matches have NaN form (first appearances);
  6,805 pairings have NaN h2h (never met).
- **Two distinct weightings, kept separate:** recency×importance *inside* the
  feature functions (part of the feature value); and `sample_weight` in `.fit()`
  (how much each training match counts — recency to the test boundary ×
  tournament importance, 10-year half-life).

### Results (held-out, 2016+)
Early stopping fitted 96 trees (of a 600 cap) — it stopped adding trees once
validation log-loss plateaued, the intended anti-overfitting behaviour.

*All test matches (n=9,904):* accuracy 0.579 (vs form-favourite 0.557,
always-home 0.477); log-loss 0.909 (vs no-skill base-rate 1.052); Brier 0.536
(vs 0.634); **calibration ECE 0.024** for P(home win). The model beats every
baseline on both accuracy and log-loss and is well-calibrated. 0.579 sits in
the believable 52–58% band — reassuring precisely *because* it is not 90%.

*Major tournaments only (World Cup + Euro, n=366) — the honest headline:*
accuracy 0.492 vs the naive form-favourite's **0.495** — a tie/slight loss;
log-loss 1.026 vs 1.088 — only a modest edge. **At the tournament level, where
teams are evenly matched on neutral ground, v1's match-level features add little
over "pick the higher-form team."** This is not hidden — it is the empirical
motivation for v2 (squad-quality features), and a far stronger scientific story
than an inflated number: *the model works broadly and is well-calibrated, but
the specific thing we claim (tournament prediction) needs the squad signal it
does not yet have.*

*Draw problem, stated plainly:* the model's argmax is "draw" only 0.2% of the
time (recall ≈ 0), despite ~24% of games drawing. This is inherent to argmax on
a class that is rarely the single most-likely outcome — not a bug. It is the
clearest demonstration that **accuracy is the wrong headline metric**: the model
assigns honest probability to draws (its log-loss and calibration are good), it
just never makes a draw its top pick.

*Upset behaviour:* on matches the form-favourite got wrong, the model still
assigned a mean 0.28 probability to what actually happened (vs 0.60 on
non-upsets) — real, non-trivial mass on the unexpected, which is the calibrated
form of "expect the unexpected."

### Next (deferred from v1)
Squad-quality features (v2, motivated directly by the tournament-level tie);
walk-forward retraining before each backtest tournament (removes the pre-2014
staleness the single cutoff imposes); tournament-stage weighting (group vs
knockout); and a bookmaker-odds benchmark once an odds source exists.

## Environment: XGBoost needs OpenMP (libomp) on macOS

`import xgboost` failed outright on this machine with
`Library not loaded: @rpath/libomp.dylib`. XGBoost parallelises the split-search
*within* each tree using OpenMP, so its compiled core (`libxgboost.dylib`) is
dynamically linked against the OpenMP runtime `libomp.dylib`. On Linux the GCC
toolchain ships one; Apple's Clang deliberately does not, and XGBoost's prebuilt
macOS wheel expects `libomp` to already exist on the system (it even hard-codes
`/opt/homebrew/opt/libomp/lib/`) rather than bundling it. Fixed with
`brew install libomp` (keg-only, ~1.8 MB); the smoke test — import, fit a
3-class model, `predict_proba` summing to 1 — passed immediately afterward.
Caught *before* it wasted the ~10-minute matrix build, by running a synthetic
XGBoost smoke test first rather than assuming the library worked. Worth
remembering: the failure message names OpenMP, not XGBoost, so it reads like an
unrelated system problem.

## Team chemistry (v2, Workstream A)

The first v2 squad-quality feature, and the one built first *because* it needs no
scraping: the `club` column is already 100% populated in the `players` table
(verified: 0 NULL/empty of 10,038), and it stores the *as-of-tournament* club, so
the feature is era-correct with zero extra work — no leakage risk, no
network dependency, no name-matching problem to solve first.

**The idea, borrowed honestly from video-game chemistry.** FIFA/FC chemistry
rewards links between players who share a club, league, or nationality. For an
international squad the **nationality link is degenerate** (everyone shares it),
so it carries no information here — the discriminating signal is **club
concentration**. Football history backs this: cohesive tournament sides often had
a dominant club spine (Spain 2010's Barça/Real core, Germany 2014's Bayern core),
while club-scattered squads less so. League-level chemistry is the same math on
leagues, but is **deferred to Workstream B** — we have no club→league map until a
ratings dataset (which carries `league`) is joined.

**Metrics** (`team_chemistry(squads, team, tournament_name)` in `features.py`,
squad-scoped exactly like `squad_age_depth`). With n = squad size and c_i = count
of players at club i: `largest_club_bloc` = max(c_i); `top2_club_bloc` = two
largest; `club_hhi` = Σ(c_i/n)² (Herfindahl concentration, 1.0 = all one club,
≈1/n = all different); `same_club_pairs` = Σ C(c_i, 2) (teammate pairs sharing a
club); `same_club_pair_ratio` = same_club_pairs / C(n, 2) (the main, size-
normalized feature); `n_distinct_clubs`.

**Missing-squad handling** (same discipline as `squad_age_depth`): counts are a
true `0`, but `club_hhi` and `same_club_pair_ratio` are `NaN` — "0 out of 0" is
undefined, not a guessed `0.0`.

**Verification** — computed metrics were cross-checked against a direct
`SELECT club, COUNT(*) ... GROUP BY club`, and the hand-computed formulas match
the function exactly:
- **Germany, World Cup 2014:** Bayern bloc of 7 (Neuer, Lahm, Boateng, Müller,
  Kroos, Schweinsteiger, Götze) → `largest_club_bloc` 7, `top2_club_bloc` 11
  (+ 4 Dortmund), `club_hhi` 0.161, `same_club_pairs` 31, `n_distinct_clubs` 11.
- **Spain, World Cup 2010:** Barça 7 + Real 5 → a *tighter two-club spine* than
  Germany (`top2_club_bloc` 12 vs 11, `club_hhi` 0.191 vs 0.161), exactly the
  distinction the metric is meant to capture.
- **Discrimination check across all 432 squads** (not just "it runs"): the most
  *scattered* squad is **Cameroon, World Cup 2022** — 26 players across 26
  different clubs (`largest_club_bloc` 1, `same_club_pairs` 0, `club_hhi` 0.038),
  alongside Ghana 2022, Nigeria/Algeria 2014. The most *concentrated* are host
  nations with domestic-league cores (Qatar & Saudi Arabia 2022) plus USA 1994
  and Egypt/Romania 1990. Both extremes are football-sensible, confirming the
  metric separates squads rather than returning noise.

**Loader change:** `load_squads()` now also selects `p.club` (previously only
team/tournament/position/age). Additive — every existing caller just filters the
frame, so nothing downstream broke.

**Web layer:** exposed as `/api/team-chemistry?team=...&tournament=...` mirroring
`/api/squad-age-depth` (missing-data check on `club_hhi`, the NaN-when-empty
field), plus a "Team Chemistry" card on the frontend. Warmed at startup via the
existing `get_squads()` cache — no new load-time cost.

**Caveat carried into Workstream E's ablation:** chemistry may be *confounded with
quality* (good players cluster at good clubs), so it must be shown to add signal
*beyond* ratings, not assumed to. That is the whole point of the planned ablation
(v1 vs v1+ratings vs v1+ratings+chemistry vs v1+chemistry). Also unresolved: the
squad-level metric may wash out because only ~11 of 23–26 players start — a
caps-weighted variant is flagged as a follow-up if the plain one underperforms.

## Name normalization for cross-source matching (v2, Workstream C — string half)

Built the source-agnostic string layer of the player→rating matching problem
ahead of the rating dataset itself, because it is testable against our *own*
player names with zero external data. Two functions in `src/name_matching.py`:

- `normalize_name(s)` — collapses a name to a diacritic-free, punctuation-free,
  lowercase token string. The core trick is **Unicode NFKD**: it splits an
  accented character into base letter + a separate combining mark (ü → u + ¨), so
  the mark can be deleted and accented/unaccented spellings collapse together.
  On top of NFKD it: strips `(c)`/footnote parentheticals (real rows carry a
  captain marker — "Lionel Messi (c)"), reorders "Surname, Forename", folds the
  handful of letters NFKD leaves alone (ø, ł, ß, æ, đ, Turkish dotless ı), drops
  apostrophes with no gap ("N'Golo" → "ngolo"), and turns other punctuation into
  spaces.
- `name_similarity(a, b)` — `rapidfuzz.fuzz.token_set_ratio` over the normalized
  names, scaled to [0, 1]. **New dependency `rapidfuzz` added with user sign-off**
  (recorded in requirements.txt); the plan's preferred choice over stdlib difflib
  for its token-*set* logic, which is robust to word order and to one name
  carrying extra tokens — the Spanish/Portuguese two-surname case.

**Verified against real DB names** (all 12 exact-match cases pass): İlkay
Gündoğan → "ilkay gundogan", Łukasz Fabiański → "lukasz fabianski", the "(c)"
marker stripped, surname-first reordered, empty/None → "". Similarity gives 1.00
for same-player/different-spelling ("İlkay Gündoğan" ~ "Ilkay Gundogan"; "Lionel
Messi" ~ "Lionel Andrés Messi Cuccittini" — the extra surnames don't hurt) and
0.31–0.35 for genuinely different players — a clean separation to threshold on.

**Known, accepted limitation:** NFKD maps ü → u, so the German transliteration
"Mueller" will *not* collapse to "Müller" → "muller". Such cases are left for the
matcher's DOB/club block + fuzzy score to catch, or the manual review queue —
never a silent wrong match (the no-fabrication rule).

**Still blocked (the rest of Workstream B/C/D/E):** the actual FIFA rating dataset
is not yet acquired — no Kaggle CLI/credentials in this environment. The
blocking→score→tier *matching algorithm*, the `player_ratings` schema, the
rating-based squad features, and the retrain/ablation all wait on that dataset
being placed in `data/raw/` (or Kaggle credentials provided). Source decided with
the user: **static Kaggle FIFA/FC dataset** (per-edition, carries `potential` —
the cold-start signal — plus club/league/DOB; no scraping, no anti-bot).

## Bug: two dependencies were merged onto one line in requirements.txt

Found while adding `rapidfuzz`: line 10 read `pytest>=8.0beautifulsoup4>=4.12` —
two requirements with no line break between them. pip parses one line as one
requirement, so this is not "pytest and beautifulsoup4"; it's a single invalid
specifier (`pytest>=8.0beautifulsoup4>=4.12`) that makes
`pip install -r requirements.txt` error out and install *nothing*. It survived
unnoticed because the working venv was populated incrementally with individual
`pip install` commands, so the broken file was never actually the install path.
A textbook "the file that documents how to reproduce the environment can't
reproduce the environment" bug — the kind only a fresh clone would have hit.
Fixed by splitting to two lines; `rapidfuzz>=3.0` added below.

## Verify-don't-assume: the squad count is 432, not "~380"

`agents.md` and the v2 plan both describe the squad data as "~380 squads". The
real number, checked directly (`SELECT COUNT(*) FROM squads`), is **432** — and
they are 432 *distinct* (country, tournament) pairs, so it isn't duplication. It
reconciles exactly with the tournaments' real team counts across the 18 editions:
160 Euro squads (8 in 1992, 16 through 2012, 24 from 2016) + 272 World Cup squads
(24 in 1990/1994, 32 from 1998) = 432, over 10,038 players. The "~380" was an
undocumented approximation; the exact figure is now corrected in `agents.md`. Not
a bug, but a reminder that a round-ish number carried in prose is worth checking
before quoting it in the write-up.

## Finding: "already-cleaned" scraped names still carry `(c)` captain markers

The schema notes describe `player_name` as having "diacritics, footnotes already
stripped". Verifying `normalize_name` against *real* rows (not invented test
strings) surfaced a counterexample the plan didn't anticipate: some names still
carry a trailing `(c)` captain marker — e.g. `Lionel Messi (c)`. Left unhandled,
that leaves a spurious one-letter `c` token that would quietly drag down every
similarity score for captains — precisely the star players (Messi, Ronaldo) whose
matches most need to be right. Caught only because the verification used the
database's own strings; this is exactly why the project tests features against
real, checkable data rather than "the code runs". `normalize_name` now strips any
`(...)` parenthetical before tokenizing.

## Rating data acquired: `player_ratings` schema (v2, Workstream B)

Unblocked the session's stated master blocker. Two corrections to the plan's
own assumptions surfaced along the way, both caught by checking directly
rather than trusting the starting description.

**Source correction:** the plan named Kaggle user "bryanb" as the source of
per-edition "complete" FIFA datasets sized 2017-2023. Checked directly
(`kaggle datasets list --user bryanb`): bryanb has exactly one FIFA dataset,
a single edition (FIFA23), not a multi-year family. The actual well-known
multi-edition "complete player dataset" series is published by
**stefanoleone992** — this is what was actually wanted and is what got used.

**Coverage correction, found by inspecting the file rather than assuming from
its name:** `stefanoleone992/ea-sports-fc-24-complete-player-dataset`'s
`male_players.csv` (96MB) turned out to be a strict superset of
`stefanoleone992/fifa-23-complete-player-dataset`'s `male_players (legacy).csv`
(91MB) — one clean snapshot per edition, **FIFA 15 through FC 24** (Sept 2014
- Sept 2023), 180,021 rows, 10 editions, in a single file. Verified, not
assumed: row counts per edition sum exactly to the total (161,583 + 18,350 =
180,021), and the two files' overlapping FIFA23 slice matched exactly - same
18,533 `player_id`s, zero `overall`-rating mismatches. The now-redundant
smaller legacy file was deleted; only `data/raw/fc24_male_players.csv` is
kept.

**Consequence: the plan's "2023 as baseline, fall back to older editions for
players missing from it" strategy is unnecessary.** One file already gives
every backtest tournament in agents.md its own era-correct edition, matched
to the closest snapshot *before* that tournament: Euro 2016 -> FIFA 16
(2015-09-21), WC 2018 -> FIFA 18/19, Euro 2020 (played 2021, COVID delay) ->
FIFA 21 (2020-09-23), WC 2022 -> FIFA 23 (2022-09-26, ~2 months out), Euro
2024 -> FC 24 (2023-09-22). This is also the methodologically better choice,
not just the simpler one - the earlier Kimmich/Musiala finding already
established that rating signal is tournament-date-sensitive, so a single
"2023 for everyone" baseline would have been the wrong design even if the
data had required it.

**Licence: CC0-1.0** (public domain dedication) on both datasets - confirmed
via `kaggle datasets metadata`, not assumed from the dataset page. Clears the
non-commercial-research-use gate cleanly; no restriction to record beyond
"there isn't one."

**Kaggle tooling finding:** the API token format Kaggle issued
(`KGAT_...`, via kaggle.com's newer "API Token" UI) is not supported by
either pip-installable client - `kaggle==1.7.4.5` or `kagglehub==0.3.13`,
both the current latest on PyPI at the time of checking. Grepped both
packages' full source for `access_token`/`api_token`/`KGAT`: zero hits in
either. Fell back to the older scheme (`kaggle.json`, username+key,
generated from Kaggle's legacy "Create New Token" button), which both
packages do support. Worth remembering next time a fresh Kaggle token is
needed: get the legacy `kaggle.json`, not the newer API-token-page format,
until client-library support catches up.

**Schema built** (`build_ratings_schema.py`, mirroring `build_squad_schema.py`'s
pattern): a standalone `player_ratings` table, 18 typed columns covering
identity (`sofifa_player_id` - named to match the existing
`transfermarkt_player_id` precedent on `players`), the rating signal
(`overall`, `potential`), and bio/club/league fields, `UNIQUE
(sofifa_player_id, fifa_version)`. Deliberately excludes the source CSV's
~90 granular sub-attribute columns (pace, dribbling, ...) - nothing in the
v2 plan calls for them; can be added later if that changes. No foreign key
into `tournaments`/`squads`/`players` yet, since nothing has matched a
rating row to a specific squad player - that's the next, separate step (the
blocking->score->tier matcher, still not built).

**Verification** against real, checkable football knowledge: Cristiano
Ronaldo's `overall` trajectory (92 -> 94 peak -> 86 by FC24) and club history
(Real Madrid -> Juventus -> Man Utd -> Al Nassr) both match reality exactly
across all 10 editions. Messi initially returned zero rows under a naive
`LIKE '%Lionel Messi%'` query - not a data bug, but the query being too
naive: his `long_name` is "Lionel Andrés Messi Cuccittini", and the middle
name breaks a literal substring match. Re-querying on `%Messi%` found him
correctly (93 -> 94 peak -> 90 by FC24) - a real, live example of exactly the
problem `name_matching.py`'s token-set similarity exists to solve, caught
during this step even though the full matcher isn't built yet. The "Silva"
ambiguity check was also re-run against this new source (270 matches at
`fifa_version=16`, mixing real Silvas with Brazilian "da Silva" surnames like
Neymar and Willian) - confirms the same disambiguation problem found earlier
against Transfermarkt search is real here too, not an artifact of that one
source.

## Player-rating matcher: blocking -> score -> tier (v2, Workstream B/C)

The step planned in ToDo.txt after `player_ratings` existed: link a specific
Wikipedia `players` row (a real human's appearance in a real squad at a real
tournament) to a specific `player_ratings` row (that same human's FIFA/FC
attributes), so squad-quality features can actually use `overall`/`potential`
rather than just club/age/position. Built as `link_player_ratings.py`.

**Scope, stated honestly:** only 5 of the 18 tournaments in `tournaments`
fall inside the ratings source's actual coverage window (FIFA 15-24, Sept
2014 - Sept 2023): Euro 2016, World Cup 2018, Euro 2020, World Cup 2022,
Euro 2024. The other 13 (every Euro 1992-2012, every World Cup 1990-2014)
get no rating link at all, full stop - there is no FIFA video game rating
for a 1996 squad to find, and nothing here pretends otherwise. This is also
exactly the backtest tournament list from agents.md, which is not a
coincidence worth taking credit for - it's the reason those tournaments were
chosen as the backtest set in the first place.

**Era-matching, made concrete in code** (previously only reasoned about by
hand in this file): each tournament maps to the closest FIFA/FC edition
dated *before* it - Euro 2016->FIFA16, World Cup 2018->FIFA18, Euro
2020->FIFA21 (the actual June 2021 COVID-delayed date, not the "2020" name),
World Cup 2022->FIFA23, Euro 2024->FC24. World Cup 2018 was the one real
decision: FIFA19 (released 2018-08-21) is technically closer in calendar
time to the tournament's aftermath but was released *after* the World Cup
already finished (14 June - 15 July 2018) - using it would mean rating
players partly on hindsight from a tournament the rating is meant to help
predict. FIFA18 (2017-09-18, ~9 months prior) is the last edition that
predates the tournament entirely, so it was chosen for consistency with
every other tournament's "closest *preceding* snapshot" rule.

**Nationality-name mismatch, found by diffing, not assumed:** compared every
Euro/World Cup squad's `country` value against `player_ratings.nationality_name`
for all 5 in-scope tournaments (136 country-tournament pairs checked). Found
exactly one mismatch: Wikipedia's "South Korea" vs sofifa's "Korea Republic"
(sofifa distinguishes "Korea Republic" from "Korea DPR" - North Korea). A
single hardcoded alias (`NATIONALITY_ALIASES`) fixes it; everything else,
including harder cases like "Czech Republic" and "Republic of Ireland",
already matched verbatim.

**Method - blocking, score, tier:**
- *Block*: cut the ~180,021-row candidate pool down before any string
  comparison, using the tournament's own fifa_version and the (aliased)
  nationality. Turns an O(3,364 squad-players x 180,021 ratings) problem
  into O(squad-size x same-nationality-same-edition candidates) - typically
  a few dozen to a few hundred per player.
- *Score*: `name_similarity()` (already built, `src/name_matching.py`)
  against both the source's `short_name` and `long_name`, taking whichever
  scores higher - sofifa's short display names ("L. Messi") and Wikipedia's
  fuller forms don't always score identically against `token_set_ratio`.
  Combined with an exact `date_of_birth` match, treated as close to decisive
  on its own: two different real people sharing both a fuzzy-similar name
  *and* an identical birthdate is vanishingly unlikely.
- *Tier*: **high** (name_score >= 0.90 AND dob matches) is auto-written to
  `players.sofifa_player_id`/`rating_match_score`/`rating_match_tier`.
  **medium** (strong on one signal, weak on the other) is written to
  `ratings_match_review_queue.csv` (gitignored, regenerable) for a human to
  check - never auto-linked. Anything weaker than medium is left alone
  entirely, not even logged - below name_score 0.55 is within the
  "genuinely different people" noise floor `name_matching.py`'s own
  docstring already established (0.31-0.35 for real non-matches).

**Schema note:** `sofifa_player_id`, `rating_match_score`, `rating_match_tier`
were added directly to `players` (`ALTER TABLE` on the live DB, and to
`build_squad_schema.py`'s own `CREATE TABLE` statement so a future full squad
rebuild doesn't silently drop the columns) - following the exact precedent
already set by the existing, still-unused `transfermarkt_player_id` column,
rather than a separate bridge table. A full squad rebuild would still *reset*
the linked values to NULL (same as it already does for
`transfermarkt_player_id` today), requiring a re-run of the matcher
afterward - an accepted, documented characteristic, not a surprise for a
later session to rediscover.

**Coverage** (players matched at "high" tier / total, per tournament):

| Tournament | Total | High | Review | Unmatched | Coverage |
|---|---|---|---|---|---|
| Euro 2016 | 552 | 454 | 25 | 73 | 82.2% |
| World Cup 2018 | 736 | 592 | 55 | 89 | 80.4% |
| Euro 2020 | 623 | 558 | 12 | 53 | 89.6% |
| World Cup 2022 | 831 | 677 | 49 | 105 | 81.5% |
| Euro 2024 | 622 | 570 | 12 | 40 | 91.6% |
| **Total** | **3,364** | **2,851** | **153** | **360** | **84.7%** |

**Verification against real, checkable football knowledge:**
- **Ronaldo** (Portugal, present in all 5 in-scope tournaments): the *same*
  `sofifa_player_id` (20801) linked at "high" tier, score 1.0, in every one -
  and the linked `overall`/`club_name` values (93/Real Madrid -> 94/Real
  Madrid -> 92/Juventus -> 90/Man Utd -> 86/Al Nassr) match exactly what was
  already independently verified straight from `player_ratings` when the
  schema was first built. Pre-2016 tournaments (Euro 2004 through World Cup
  2014) correctly returned no match at all - outside the ratings window, as
  designed, not a bug.
- **Messi** (Argentina, World Cup 2018 + 2022): same `sofifa_player_id`
  (158023) both times, score 1.0, `club_name` correctly moving
  FC Barcelona -> Paris Saint Germain between the two snapshots - matching
  his actual August 2021 transfer.
- **"(c)" captain-marker leak, found as a side effect, not part of this
  step's own work:** Portugal's Ronaldo rows for World Cup 2010/2014 still
  show `"Cristiano Ronaldo (c)"` in `players.player_name`, meaning
  `build_squad_schema.py`'s `clean_player_name()` only strips the literal
  string `"(captain)"`, not the abbreviated `"(c)"` some tournament pages
  use - so `is_captain` is silently wrong (False when it should be True) for
  whichever historical rows use that convention. Checked directly whether
  this reaches any of the 5 in-scope tournaments: **zero rows affected** -
  it's a pre-2016 artifact only, so it did not corrupt today's matching
  (`name_matching.py`'s `normalize_name()` strips *any* parenthetical
  content generically, independent of this). Left unfixed for now - a real,
  documented gap for a future session, not today's task.
- **Silva disambiguation - a harder test than the original Transfermarkt
  case, because these all share one nationality:** six distinct Portuguese
  "Silva"s (Adrien, André, António, Bernardo, Rafa, Rui) appear across the 5
  tournaments. Blocking by nationality alone cannot separate same-country
  namesakes, so this exercises the name+DOB scoring specifically, not just
  the blocking step. Every one that matched got a distinct, internally
  consistent `sofifa_player_id` (Bernardo Silva: 218667, identically across
  all 4 tournaments he appears in). Thiago Silva (Brazil) and David Silva
  (Spain) were also correctly kept apart from all six Portuguese Silvas. The
  two genuine misses in this group (António Silva - a 2022-breakout young
  defender; Martín Silva - a lower-profile Uruguayan goalkeeper) read as
  real sofifa coverage gaps rather than matcher failures, consistent with
  the ~85% overall coverage rate.

**What's still open:** the 153-row review queue needs a human pass (Liam's
call on each) before any of those links are trusted; the 360 unmatched
players are simply not linked (no guess made, per the no-fabrication rule) -
some genuine sofifa coverage gaps are expected (lower-profile players,
younger debutants not yet rated at that edition), but the review queue and
unmatched set haven't been audited for a systematic pattern (e.g. one
specific nationality or position underperforming) the way the World Cup 2002
footnote bug was eventually traced to one specific cause.

## Is v1 retrain-ready with the new data? Checked before assuming yes

After the matcher above, the natural next question: does team_chemistry
(built earlier) plus the newly-linked ratings mean v1 is ready for a retrain.
Checked directly against the actual code rather than answered from
impression - the honest answer is a two-track "partially", not a yes.

**Shared prerequisite, found while checking, that blocks BOTH tracks:**
`src/models/build_matrix.py`'s `FEATURE_COLUMNS` is currently 100%
match-level (`neutral`, `importance`, form, goal trend, h2h) - neither
`team_chemistry` nor `squad_age_depth` has ever been wired into the training
matrix, despite both being built and verified. The reason: `matches.tournament`
is a generic competition-*type* label ("FIFA World Cup", "UEFA Euro") shared
across every edition/year, confirmed by direct query - there is currently no
code linking a specific match row to a specific `tournaments.name`/`squad_id`
(that would need deriving from the match's `date`, e.g. year + competition
type -> `tournaments` row). Squad-level features were always going to need
this glue; it just hadn't been needed yet because v1 never used any of them.

**The two tracks, once that glue exists, are genuinely different -
verified with real counts, not assumed identical:**

Counted major-tournament (`FIFA World Cup/UEFA Euro`) match rows directly
against `train_wdl.py`'s existing split boundaries (`VAL_START=2014-01-01`,
`TEST_START=2016-01-01`):

| | `team_chemistry` / `squad_age_depth` | ratings (`overall`/`potential`) |
|---|---|---|
| Tournament coverage | all 18 (1990-2024) | only 5 (Euro16/20/24, WC18/22) |
| Real **training-set** rows | **530** (pre-2014 major-tournament matches) | **0** |
| Test-set rows | 281 | 281 |

The zero is not a guess: every tournament the ratings source covers
(FIFA 15-24, Sept 2014 - Sept 2023) maps to a *tournament* dated 2016 or
later (Euro 2016 onward) - so by construction, every match row with a
non-null rating-derived feature falls in the **test** block under the
existing single-cutoff split, never train or val. XGBoost cannot learn a
real relationship for a column it never observes populated during training;
with zero training examples it can only ever learn a meaningless
missing-value default direction. `team_chemistry`/`squad_age_depth` don't
have this problem - Wikipedia squad data goes back to 1990, giving 530 real
pre-2014 training examples.

**Consequence:** this is exactly why v1's own "Next" section (see the
XGBoost v1 section above) already flagged walk-forward retraining as
future work - that was framed as a staleness fix. This finding upgrades it
specifically for ratings features: walk-forward retraining (or some other
split redesign) is now a **hard prerequisite** for ratings features to
contribute anything at all, not merely a nice-to-have improvement.

**Recommendation logged** (Liam's call on sequencing, not yet started):
two separate tracks rather than one retrain. Track 1 - build the match ->
tournament-edition glue, wire in `team_chemistry`/`squad_age_depth` under
the *existing* split, retrain; real signal on both sides of the split,
directly tests what v1's tournament-level result (0.492 vs 0.495 baseline,
effectively a tie) already flagged as the open question. Track 2 - walk-
forward retraining, needed before ratings features are usable at all; a
cheap non-XGBoost correlation check (mean squad `overall` vs actual outcome
across the 281 covered tournament matches) was suggested as a first step, to
confirm the signal is worth that larger engineering investment before
building it.

## Environment: rating-acquisition tooling absent, CSV caching kept as default

For the record, so a later session doesn't rediscover it: this environment has
**no `kaggle` CLI, no `kaggle` Python package, and no `~/.kaggle/kaggle.json`
credentials**, and neither `rapidfuzz` nor `pyarrow` was installed at the start of
the session. `rapidfuzz` is now installed (user-approved). `pyarrow` was
deliberately *not* added — the plan's default is CSV caching for any downloaded
rating files, and no caching has happened yet (no dataset), so there is nothing to
decide until the data actually lands. PyPI itself is reachable (pip works), so the
only blocker to Workstream B is the dataset/credentials, not general network.

## TRACK 1 executed: squad features wired into the WDL model (v2)

This is the execution of TRACK 1 from the retrain-readiness analysis above —
building the match→tournament-edition glue, wiring `team_chemistry` and
`squad_age_depth` into the XGBoost matrix, and measuring whether squad quality
closes the tournament-level tie (v1: 0.492 vs the form-favourite's 0.495).

### The glue: `match_tournament_edition` (`src/data_pipeline.py`)
The single missing prerequisite. `matches.tournament` is a generic competition
label (`"FIFA World Cup"` / `"UEFA Euro"`) with no year and no foreign key, while
the squad features key off an edition label (`"World Cup 2018"`). The bridge maps
a match's (label, date) to the edition, and three things had to be handled — none
guessed, all confirmed against the live DB:
- **Label mismatch:** the two tables spell the competition differently —
  `"FIFA World Cup"` (matches) vs `"World Cup"` (tournaments). A hard-coded
  `COMPETITION_LABEL_MAP` bridges the two; any label not in it (friendlies,
  qualifiers, Copa América, …) returns `None`.
- **Euro 2020 was played in 2021** (COVID postponement) but its tournaments-table
  row keeps the official name/year "Euro 2020". Match rows are dated 2021, so a
  naive year-join would drop *every* Euro 2020 fixture. Corrected explicitly.
- **Editions with no squad data** (pre-1990, and World Cup 2026 which now appears
  in the data as played matches but has no scraped squad) return `None` → NaN,
  never a fabricated squad.

Coverage on the cleaned data: 875 of 960 major-tournament match rows resolve to
an edition (the 85 unresolved are all WC 2026, no squad yet). Of the resolved
rows, 1,723 of 1,750 team-slots find their squad. The **27 misses are all real
historical name changes**, not bugs: Germany@1990 (competed as *West Germany*),
Russia@1990/Euro1992 (*Soviet Union* / *CIS*), Serbia@1998/2000/2006 (*FR
Yugoslavia* / *Serbia and Montenegro*), China@2002 (*China PR*). All fall pre-2014
and all become NaN — resolving them needs a *reverse* name lookup (current name →
the historical name as-of the match date), which `resolve_team_name` does not do
(it maps the other direction). Logged as a known, low-priority gap.

### Features added (`src/models/build_matrix.py`)
Ten columns, home/away each of: `mean_age`, `squad_size`, `club_hhi`,
`same_club_pair_ratio`, `largest_club_bloc`. Home/away kept separate (XGBoost
learns the interaction). A team present in the match but absent from that
edition's squad (the 27 mismatches) records **NaN, not 0** — a `squad_size` of 0
would falsely assert an empty squad rather than "unknown". Verified the usual way:
a built matrix row's `home_club_hhi` for Germany@WC2014 equals a standalone
`team_chemistry` call to 6 dp (0.161), and the existing 9 features' NaN counts are
byte-identical to v1 (no regression). Squad columns populate on exactly 861 rows,
all major tournaments (0 leakage into friendlies).

### Result: A/B under two splits (same matrix, only the columns/boundary change)
| split | major test n | v1 acc (9 feats) | v2 acc (+squad) | Δ acc | v1→v2 log-loss |
|---|---|---|---|---|---|
| **original** (train<2014, test 2016+) | 366 | 0.492 | **0.500** | +0.008 | 1.026 → 1.020 |
| **end-of-2020** (train<2019, val 19-20, test 2021+) | 251 | 0.514 | **0.546** | +0.032 | 1.003 → 0.998 |

Under the original split, squad features flip the headline: v1 *lost* to the
form-favourite baseline (0.492 < 0.495); v2 *beats* it (0.500 > 0.495), and
improves log-loss — calibration held (ECE 0.025, unchanged). The all-test numbers
barely move (0.579 → 0.580) because squad features touch only ~3% of matches (only
tournaments) — the model is not harmed elsewhere, as intended by NaN-native design.

The **end-of-2020 split shows a larger benefit** (+3.2pp accuracy on major
tournaments; v2 0.546 vs baseline 0.514). This was requested specifically to give
the squad features real *training* representation: training through 2020 raises
populated squad-feature training rows from 516 to 695 and adds recent tournaments
(WC 2014, Euro 2016, WC 2018) close to the test distribution. A convenient
property confirmed here: carving 2019–2020 as the early-stopping validation slice
costs **zero** major-tournament training rows (no Euro/WC was held in 2019–2020),
so "everything up to 2020" genuinely feeds the model all 15 pre-test tournaments.
On the 166 test rows that actually *have* squad data (Euro 2020 + WC 2022 + Euro
2024; WC 2026 is NaN), v2 beats v1 by +2.4pp.

### Honesty caveats
The accuracy gains are **modest and within sampling noise** at these sizes:
+3.2pp on 251 matches ≈ 8 games, +2.4pp on 166 ≈ 4 games. The reassurance is
that the *direction is consistent* — v2 ≥ v1 on both accuracy and log-loss in
both splits, and v2 beats the naive baseline where v1 did not — and that log-loss
(a more stable metric than top-1 accuracy) improves in the same direction, if
only slightly. This is a suggestive positive signal, not a decisive one.

### Pruning check
`squad_size` is near-constant (23, or 26 for 2022+ tournaments) and had the lowest
importance (`away_squad_size` gain was literally 0.00 in the 2020 split). Tested
dropping both `squad_size` columns: it made major-tournament accuracy *worse*
(0.546 → 0.530 all-251; 0.542 → 0.536 covered-166) and log-loss no better, so the
full 10-feature set was kept — pruning decided empirically, not by importance
alone.

### Implication for TRACK 2
The end-of-2020 result is effectively a one-step walk-forward: the squad signal is
strongest when the model trains on data close to the test period. That is the
direct empirical argument for full walk-forward retraining (train fresh before
each backtest tournament), which the retrain-readiness analysis flagged as a hard
prerequisite for the *ratings* features (0 training rows under the single 2014
cutoff). TRACK 1 confirms the mechanism on features that already had training
representation, de-risking that larger investment.

### Walk-forward reality check — the single-split gain does not fully survive
Before reading too much into the +3.2pp, a proper walk-forward backtest was run
(now a committed module, `src/models/walk_forward.py`): for each of the five
backtestable tournaments (Euro 2016, WC 2018, Euro 2020, WC 2022, Euro 2024),
train on ALL matches strictly before it, early-stop on the trailing 365 days,
predict that tournament, then pool the 281 predictions. It derives its tournament
set from the matrix's `edition` column (self-updating for future squad scrapes)
and consumes `FEATURE_COLUMNS`, so the TRACK 2 ratings columns will flow into it
with no change. Result:

| | pooled acc (n=281) | pooled log-loss |
|---|---|---|
| v1 (no squad) | 0.488 | **1.042** |
| v2 (+squad) | **0.498** | 1.047 |
| form-favourite baseline | 0.491 | — |

So under the most rigorous split, the squad features give **+1.1pp accuracy
(≈3 matches) and a *marginally worse* log-loss** — i.e. the benefit largely
washes into noise. Per-tournament it is genuinely mixed: squad features help in
WC 2018 (+6pp) and WC 2022 (+1.6pp), hurt in Euro 2016 (−2pp) and Euro 2020
(−2pp), tie in Euro 2024. The clean +3.2pp end-of-2020 figure was therefore
**partly a favourable-split artifact** (that split trains through 2018 and tests
on a specific recent 4-tournament window). Honest conclusion: at the ~281
tournament matches available across football's backtestable window, both the
squad features and the split choice move accuracy by only a handful of games —
within sampling noise. The defensible claim is **methodological** (a leakage-free,
never-stale backtest harness that also unblocks the ratings features), not a
headline accuracy jump. This is logged rather than buried precisely because the
project's write-up leans on an honest process, not a highlights reel.

## TRACK 2 de-risked: mean squad `overall` strongly predicts outcomes

Before paying the walk-forward engineering to wire FIFA/FC ratings into the model,
the cheap non-model check the ToDo flagged (`scratchpad/ratings_correlation.py`):
does a squad's **mean `overall`** actually relate to results? For each of the 5
ratings-covered tournaments, each squad's mean `overall` was computed from its
linked players at the era-correct edition (Euro 2016→FIFA16, WC 2018→FIFA18, Euro
2020→FIFA21, WC 2022→FIFA23, Euro 2024→FC24 — the same map the linker used), then
compared against actual outcomes. No model, no split — a pure descriptive check,
and leakage-free by construction (each rating snapshot predates its tournament).

**Sanity first:** the means rank exactly as football knowledge demands — strongest
squads Spain/Germany WC 2018, Brazil WC 2018/2022, France Euro 2020 (~84);
weakest Panama, Iceland, Qatar, Saudi Arabia, Finland (~67). A ~17-point spread.

**The signal is strong and unambiguous** (256 of 281 matches had ≥11 rated players
both sides; the 25 dropped were low-coverage minnow squads):
- correlation of rating gap with home result: **Pearson r=+0.42, Spearman ρ=+0.44**,
  both p≈10⁻¹² — not marginal.
- **the higher-rated team wins 75% of decisive (non-draw) matches** (n=188).
- "predict the higher mean `overall`" scores **0.551** accuracy vs the
  form-favourite's 0.480 on the same matches (**+7pp**) — and it even beats the
  full walk-forward XGBoost that has chemistry/age but no ratings (0.498).
- restricted to clear favourites (|gap|≥3, n=173) accuracy rises to **0.618**.

**Conclusion:** ratings carry materially more signal than the chemistry/age squad
features did — expected, since `overall` is a direct expert quality assessment,
not an indirect proxy. This is a clear green light to add ratings columns to
`FEATURE_COLUMNS` and re-run `walk_forward` (which finally gives them training
rows). **Honest caveat to carry into that build:** under walk-forward the earliest
tournaments have thin ratings *training* coverage — the Euro 2016 model trains on
zero prior ratings-covered tournaments, WC 2018 on one, etc. — so the in-model
walk-forward gain will be smaller than this raw predictor's +7pp and will grow
with each later tournament. The correlation justifies the build; it does not
pre-promise the final number.

## TRACK 2 in-model: a strong raw signal that first HURT, then was fixed by data

Wiring the ratings into the model produced one of the project's most instructive
results — a feature that predicts strongly *in isolation* can *hurt* inside the
model, and the reason turns out to be data scarcity, not the feature.

**First attempt (tournament-only ratings, 281 matches).** Absolute per-squad
`overall` columns dropped walk-forward tournament accuracy to 0.459 (below the
0.488 no-ratings model). Re-encoding as a single *relative* `rating_gap`
(home−away) recovered it to the project-best 0.509 — but only when paired with
the bare match-level features. Adding `rating_gap` on *top* of the squad
chemistry/age features gave 0.477: the two feature groups appeared to "cloud"
each other.

**Root cause — checked directly, two intuitive explanations ruled OUT.**
- *Not redundancy.* A squad's mean `overall` correlates with its chemistry/age
  features at only **r≈0.07** — they measure different things, so it is not two
  quality proxies conflicting.
- *Not cross-edition rating drift.* The `overall` distribution is stable across
  FIFA 15–24 (mean ~65–66), so a model trained on FIFA 16/18 is not misreading a
  FIFA 21 scale.
- *The actual cause: scarcity → variance.* Ratings existed on only 281 tournament
  matches, with **0 training rows under any single cutoff** (a fitted model's
  `rating_gap` gain was literally 0.00 there). The 10 chemistry/age features are
  individually weak (TRACK 1: within noise); adding them plus a data-starved
  rating column just gave a greedy booster more axes to overfit a tiny sample.
  And bootstrap CIs later showed the whole "clouding" (~9 matches on 281) sat
  *inside* the noise band — it was never a robust effect.

**The fix (the partner's insight): stop starving the feature.** The
`player_ratings` table covers all players, not just 5 tournaments — the *feature*
was limited only because we have tournament **squads**, not match **lineups**. So
a country's strength going into *any* international match is approximated by its
tournament-squad **player pool's** mean `overall` at the match's era-correct
edition (`load_team_pool_ratings` + `team_pool_rating`, `fifa_edition_for_date`
mapping match date → FIFA edition). A strict `first_seen` cutoff (only players
known as internationals *on or before* the match date) keeps it leakage-free.
This lifts rating coverage from **281 → 1,081** leakage-free matches (friendlies,
qualifiers, Nations League, tournaments) across 56 nations, and the raw signal
holds: **r=0.42** on the expanded set (higher than the tournament-only 0.37,
because the leakage-free pool is more era-appropriate).

**Result — the clouding dissolves and ratings help (measured with bootstrap CIs).**
On the 281-match tournament backtest every feature set is *statistically
indistinguishable* (95% CIs all ≈[0.43, 0.56]) — the test is simply too small to
resolve a few points; but the previously catastrophic all-features model rose
0.459 → 0.484, most of the "clouding" gone. On a **broader, higher-power test**
(738 covered matches from 2021+, train <2021):

| feature set | acc | 95% CI | log-loss |
|---|---|---|---|
| match-level (9) | 0.505 | [0.469, 0.542] | 1.007 |
| **+ rating_gap (10)** | **0.528** | [0.491, 0.564] | **0.996** |
| + squad (19) | 0.505 | [0.469, 0.542] | 1.007 |
| all (20) | 0.520 | [0.482, 0.557] | 0.997 |

`rating_gap` adds **+2.3pp** (and **+2.5pp** on competitive-only, 0.542 vs 0.517),
with the best log-loss in every cut — a consistent, corroborated lift rather than
a noise blip (though the accuracy CIs still overlap, so not 95%-significant on
accuracy alone). Crucially the **all-features model (0.520) now matches
+rating_gap (0.528)** — with real training coverage the squad and rating features
coexist. The clouding was a scarcity artifact, and adding data (not manual
weighting) resolved it.

**Decision:** keep all 20 features. `rating_gap` earns its place; the squad
features are now neutral (no longer clouding); nothing is removed. The honest
headline for the write-up: on the small tournament backtest all approaches tie
within noise, but on the larger competitive-match test the FIFA-rating gap is a
real, if modest, source of skill — and the methodological point (a strong
isolated predictor can be useless in-model until it has enough training data) is
itself a result.

**Caveats carried forward:** the pool rating approximates a country's strength,
not its actual match-day XI (weakened friendly lineups add noise); only the 56
nations with a tournament squad are covered; ~500 of the covered matches still
lack ≥11 rated players at their edition and stay NaN; and `train_wdl`'s single
2014 cutoff still can't use `rating_gap` at all (0 training rows) — walk-forward
or a late split is required, exactly as the retrain-readiness analysis predicted.

### Reproducing the ablation on the broad covered block (does the +2.3pp survive?)

The `+2.3–2.5pp` above came from a scratchpad run on a **738-match** competitive
block. To make it reproducible rather than a one-off, the per-feature-set
ablation was promoted into `src/models/broader_eval.py` (`ablation_report`):
three name-listed feature **groups** — `base` (9 match-level), `squad` (10),
`rating` (`rating_gap`) — asserted to *partition* `FEATURE_COLUMNS` (the module
fails loudly if a future feature is added without being assigned to a group), a
loop running the leakage-safe annual walk-forward for four sets (base, +squad,
+rating, full) on the identical odds-covered block, and **paired-bootstrap CIs**
on the per-match log-loss difference. The pairing is exact because a feature set
changes only the *model*, never *which* covered rows are scored or their order —
asserted with `np.array_equal` on the pooled truth vectors.

**The lift does not reproduce as a significant margin on the wide block.** On the
full odds-covered set (**n = 2,184**, dominated by qualifiers / Nations League),
every contrast is within noise:

| contrast (with − without) | Δ log-loss | 95% CI | verdict |
|---|---|---|---|
| `rating_gap` \| no squad | −0.0025 | [−0.0069, +0.0020] | within noise |
| `rating_gap` \| squad present | +0.0010 | [−0.0031, +0.0052] | within noise |
| squad \| no rating | −0.0021 | [−0.0053, +0.0010] | within noise |
| squad \| rating present | +0.0014 | [−0.0015, +0.0044] | within noise |

Both groups lean the right way *in isolation* (each ≈ −0.002 log-loss on top of
`base`) but flip to a tiny positive on top of *each other*, and every CI straddles
0. This is the honest correction to the 738-match headline: that block was
favourable, not general. The wide-block result is instead **consistent with the
tournament backtest** — the rating and squad features are marginal, and on a large
enough, qualifier-heavy sample their lift is indistinguishable from noise (partly
because a large share of qualifiers have no `rating_gap` at all, diluting it). The
decision to keep all 20 features stands — none of them *hurt*, and `rating_gap`
still carries the strongest isolated signal (r = 0.42) — but the write-up claim is
now the sober one: *the engineered squad/rating features are a real but small,
block-dependent source of skill, not a reliable multi-point accuracy gain.* That
the peeked-block number shrank to noise under a wider, honestly-paired test is
itself the kind of result the project's methodology exists to surface.

## Draw prediction: a well-calibrated model that (correctly) never picks "draw"

A natural question — "why does the model almost never predict a draw?" — turned
into one of the clearest teaching results in the project, and a reminder that
*accuracy* is the wrong lens.

**The model's draw probabilities are already ~perfect.** On the 2016+ test the mean
predicted `P(draw)` is **0.233** against an actual draw rate of **0.233**, and the
draw column tracks the diagonal per bin (predicted 0.31 → actual 0.31). Nothing is
mis-estimated. The reason draw is almost never the top-1 pick is purely structural:
a correctly-calibrated `P(draw)` **tops out at ~0.389**, so it essentially never
exceeds both `P(home)` and `P(away)` — draw is the argmax only **19 / 9904** times.
Even in coin-flip fixtures (`|P(H)−P(A)| < 3%`) the actual split is **H .336 / D .303
/ A .361** — draw is never the plurality *anywhere* in probability space. So a
"lean toward draw on close games" rule cannot raise accuracy; it would trade correct
H/A calls for wrong D calls and *lower* it. Forcing draws (class weights) would only
inflate `P(draw)` above the true 24% and worsen log-loss/calibration.

**Penalty knockouts are already draws.** A separate proposal — "discard draws for
matches that go to penalties" — rests on a data misconception: the score column is
the full-time (post-extra-time, pre-shootout) result, so every shootout game is
stored as a **draw** (verified: England 1-1 Italy, Argentina 3-3 France, Russia 1-1
Spain). There is also no pre-match way to know which games go to penalties (it is a
*consequence* of the draw, not a predictor). Under the 1X2 target those labels are
correct.

**Decision:** do nothing to the WDL model on draws. The right home is the (still-stub)
Poisson scoreline model, where draws fall out of the score distribution and the
Dixon-Coles low-score correction exists precisely because independent Poisson
under-predicts 0-0/1-1. A separate 2-way "who advances" target (with a ~50/50
shootout) is where knockout resolution belongs, for the Monte Carlo simulator.
Deferred in ToDo.txt next to the Poisson item.

## Weighting auto-tuner: is the hand-tuned match weighting beatable? (honestly)

The per-match training weight is `recency(date) × importance(tournament)` — a
hand-chosen recency half-life (3650 days), a min-weight floor (0.05), and 8 tiered
importance multipliers (Friendly 0.3 … World Cup 1.0 … default 0.25). The idea was
to stop hand-guessing these and *search* for better ones. The stated intuition —
"a search has a ~100% chance of raising accuracy" — is the overfitting trap: a
search only guarantees improving the metric on the data it optimises against. Done
against the held-out test that would silently manufacture a fake number. So the
tuner (`src/models/tune_weights.py`) is built around three rules: optimise
**validation log-loss** (not accuracy); the search **never scores a 2016+ match**
(candidates are ranked by a leakage-safe walk-forward over pre-2016 rolling-origin
folds, 11,505 matches); and the search **proposes**, it does not silently overwrite
the production weights. The harness is the existing `walk_forward.py`, minimally
refactored so the weighting is an injected `weight_fn` (verified byte-identical to
the old output for the default; the baseline `WeightConfig` reproduces
`importance_weight` on all 140 tournaments and the sample weights to full precision).

**Result 1 — the validation objective is nearly flat.** 200 random configs
(baseline seeded as trial 0; `major_finals` anchored at 1.0 as the scale reference)
moved validation log-loss by **−0.0002** at best (0.9224 → 0.9222). Wildly different
weightings — the winner had Friendly 0.69, Confederations 0.91, default 0.15,
min-weight 0.29 vs the baseline's 0.30 / 0.55 / 0.25 / 0.05 — produce near-identical
pre-2016 log-loss. The weighting is a **low-leverage knob**: the model barely cares
how the training data is weighted.

**Result 2 — the apparent test gain is noise.** Run once on the held-out 2016+
tournament backtest (n=281), the "winning" config *looks* better:

| metric | baseline | proposed | Δ |
|---|---|---|---|
| accuracy | 0.484 | 0.509 | **+0.025** |
| log-loss | 1.055 | 1.045 | −0.010 |
| Brier | 0.633 | 0.629 | −0.005 |

But a paired (per-match) bootstrap on the log-loss difference gives a 95% CI of
**[−0.025, +0.005]** — it straddles zero. The +2.5pp accuracy is **within noise**.
Tellingly, the *test* improvement (−0.010 log-loss) is larger than the *validation*
improvement that selected the config (−0.0002): the signature of a lucky draw, not
skill. This is the same lesson as the squad features (val +3.2pp → +1.1pp within
noise) — and exactly what the "100% chance of raising accuracy" framing misses. Yes,
the accuracy number went up; the bootstrap shows it isn't real.

**Decision:** keep the hand-tuned baseline weighting. It is already near-optimal and
the proposed alternative is (a) statistically indistinguishable on held-out data and
(b) less interpretable (it inverts sensible tier orderings). No production code was
changed; the proposal + its held-out verdict are recorded in
`src/models/best_weight_config.json`. The result *is* the finding: a systematic,
leakage-safe search plus an honest CI shows the weighting isn't a reproducible source
of skill — which is a stronger claim for the write-up than a cherry-picked +2.5pp.

**Caveats.** Validation is *general* pre-2016 match log-loss (the weighting is a
global training-data choice), not major-tournament-specific — a design trade to keep
the 2016+ majors fully held out. It is random search (not exhaustive) over ~9 dims.
And this only tuned the *weighting*: the XGBoost hyperparameters are a separate,
likely higher-leverage knob, deliberately left as the next tuner (Phase B) so any
gain there can be attributed cleanly rather than tangled with the weighting.

## Hyperparameter tuner (Phase B): the higher-leverage knob, same honest verdict

The weighting turned out to be low-leverage, so Phase B tuned the actual XGBoost
hyperparameters — the more plausible source of a real gain. Rather than duplicate
the leakage-safe machinery, the pre-2016 inner objective and the held-out bootstrap
comparison were first extracted into `src/models/tune_common.py` (verified inert:
the weighting tuner still reproduces its 0.9224 baseline exactly afterward), and
both tuners now inject through the same core — the weighting tuner varies a
`weight_fn`, the hyperparameter tuner a `model_fn` (`build_model` gained keyword
overrides; `_fit_before`/`walk_forward` gained a `model_fn` argument, both
byte-identical by default). A built-in cross-check ties the second tuner to the
first: its baseline objective is **also 0.9224**, and its baseline held-out
walk-forward reproduces the committed 0.484 acc / 1.055 log-loss — same model,
reached a different way.

Six knobs were searched (`max_depth`, `learning_rate`, `subsample`,
`colsample_bytree`, `min_child_weight`, `reg_lambda`; `n_estimators` is left to
early stopping), 120 random trials, same rules: rank by pre-2016 validation
log-loss, never touch 2016+, propose-don't-overwrite.

**Result — a slightly larger validation edge that still doesn't survive.** The best
config (a *more* conservative model: depth 3, learning-rate 0.016, `min_child_weight`
9, `reg_lambda` 0.62) improved validation log-loss by **−0.0011** — five times the
weighting's −0.0002, but still tiny. On the held-out 2016+ backtest:

| metric | baseline | proposed | Δ |
|---|---|---|---|
| accuracy | 0.484 | 0.491 | +0.007 |
| log-loss | 1.055 | 1.048 | −0.007 |
| Brier | 0.633 | 0.630 | −0.004 |

Paired bootstrap on the log-loss difference: 95% CI **[−0.026, +0.012]** — straddles
zero, **within noise**. Keep the baseline hyperparameters; nothing promoted (recorded
in `src/models/best_hyper_config.json`).

**The combined finding (both tuners).** Neither the match weighting nor the XGBoost
hyperparameters is a reproducible source of skill on this backtest — both hand-chosen
sets are already near-optimal, and model performance is dominated by the *features* and
the intrinsic difficulty of international-match prediction, not by tuning knobs. A
telling detail confirms the discipline: the weighting's apparent test-accuracy bump
(+2.5pp) was *larger* than the hyperparameters' (+0.7pp) despite a *smaller* validation
gain — the two "improvements" don't even rank-order consistently, which is exactly what
noise looks like. The deliverable of this workstream is therefore methodological: a
reusable, leakage-safe, propose-don't-overwrite tuning harness (`tune_common` +
`tune_weights` + `tune_hyperparams`) that measures whether a proposed change is real
and, here, correctly refuses to over-claim. That is a stronger result for the write-up
than a cherry-picked accuracy bump would have been.

**Combining the knobs — and a clean demonstration of test-set peeking.** The obvious
follow-up: each knob was within noise alone, but do they *stack*? Two ways to ask,
with opposite honesty:

- *Peeked* (`tune_combined.py`): apply the two individually-best proposals together and
  test. On validation the weighting added ~nothing beyond the hyperparameters
  (combined 0.9212 vs hyper-only 0.9213); on the held-out set this gave the best point
  estimate yet — accuracy **0.484 → 0.495 (+1.1pp)**, log-loss −0.0157 — but the
  bootstrap CI **[−0.0375, +0.0062]** still included zero. Tempting, but this config was
  chosen *after* seeing test results for its parts.
- *Clean* (`tune_joint.py`): one leakage-safe JOINT search over all 15 dims (250 trials),
  winner selected purely on pre-2016 validation, tested exactly once. Two things fell
  out. First, random search *dilutes* in higher dimensions: the joint search's best
  validation edge (−0.0005) was **smaller** than the focused 6-dim hyperparameter search
  (−0.0011) — 250 trials spread over 15 dims cover the good region less densely than 120
  over 6. Second, and decisively, on the held-out set the joint winner's accuracy went
  **0.484 → 0.473 (−1.1pp, *worse*)**, log-loss −0.0128, CI **[−0.0302, +0.0045]** — within
  noise.

The same style of config swung from **+1.1pp** (peeked) to **−1.1pp** (clean) on
held-out accuracy purely because of whether the test set was allowed to influence the
choice. That reversal is the clearest evidence in the project of why the honest
protocol — select on validation, test once, report a CI — is not pedantry: it is what
stops a small (n=281) held-out set from being gradually fit by repeated candidate
testing. Final decision: **keep the baseline weighting and hyperparameters**; nothing
promoted. Proposals + held-out verdicts are recorded in `best_{weight,hyper,joint}_config.json`.

**Next lever.** Both marginals and their honest combination are flat, so more tuning
(Optuna, more trials) is low-value; the far higher-leverage work is better *features*
and the bookmaker-odds benchmark, not squeezing the knobs.

## Bookmaker-odds benchmark: the model-vs-market comparison

This is the second half of the judges' hook (agents.md): the WDL model compared not
only to naive baselines but to **bookmaker implied probabilities** — the sharpest
publicly available forecast, and the honest ceiling for any 3-way football model. It
took a long, instructive data-acquisition detour to build, and the detour itself is
good material: it is a clean case study in "the data you need is not the data you can
easily get", handled without fabricating anything.

### Data sourcing: four dead ends before a workable one
No free, reproducible source of historical **international** 1X2 odds exists. Checked,
in order:
- **Kaggle** (the project's usual channel): every odds dataset found is **club-league
  only** (football-data.co.uk-style leagues, Oddsportal club scrapes). Zero coverage of
  Euro/World Cup fixtures. So the ToDo's original football-data.co.uk plan was a
  non-starter — that site has no internationals at all.
- **FiveThirtyEight SPI** (`spi_matches_intl.csv`): would have been a clean *model*
  benchmark (not the market), but 538 was shut down by Disney in 2023; the live API
  redirects to ABC News and the GitHub CSV 404s. Dead.
- **The Odds API**: real bookmaker odds, ToS-clean — but historical access is paid and
  international coverage only reaches back to ~WC 2022 (misses Euro 2016/2020, WC 2018).
- **Oddsportal** (full coverage, free): its results feed is **AES-encrypted** (base64
  `ciphertext:iv`, key hidden in an obfuscated JS bundle). An automated scrape would
  mean reverse-engineering that key — against the project's reproducibility bar, and the
  environment's safety tooling blocked the key-extraction step anyway. Not pursued.

**What worked:** a human browser export from Oddsportal's results pages. In a real
browser the page decrypts client-side and renders normally, so a plain
select-and-copy captures the odds — sidestepping the WAF, the encryption, and the
key-extraction question entirely. Coverage obtained: **Euro 2020, Euro 2024, World Cup
2022, World Cup 2026** plus their qualifiers (Euro 2016 and WC 2018 simply had no
Oddsportal data). This is a human-in-the-loop acquisition step, documented as such.

### Two real bugs the export exposed (both caught by sanity checks, not by "it ran")
1. **CSS-class table-scraper misaligned odds and teams.** The first export used a
   scraper extension that named columns by CSS class (`flex-center`, `ml-auto`, …) and
   pulled the odds cells in a *different DOM order* than the team cells. Teams and scores
   were correct; the two win-odds were scrambled relative to them, inconsistently. It
   looked plausible until a **favourite-accuracy** check: the bookmaker's implied
   favourite matched the actual winner only ~50% of the time on decisive games (should
   be ~65%+), and San Marino showed as a 99% favourite at home to Denmark. Tested all 6
   column permutations — none was consistent (the only high-accuracy one implied a mean
   draw probability of 60%, nonsense). Verdict: not code-fixable; re-export needed.
2. **A plain copy-paste fixed it, but the odds arrived American with the `+` stripped.**
   Oddsportal's results table isn't real `<table>` markup, so a copy dumps one value per
   line in strict reading order — reliably parseable by a small state machine
   (`_parse_flat_paste`), and crucially the odds stay glued to the right team. But the
   paste dropped the leading `+` on positive American prices (`+260` → `260`), and
   `american_to_decimal` initially read a bare `260` as an already-decimal odd of 260.0,
   silently crushing every favourite's probability toward 1.0 and erasing draws. Caught
   by a **draw-calibration** check (predicted mean draw prob 7% vs actual 21%). The fix
   keys on a fact true of every format seen: decimal odds always carry a `.`, so a bare
   integer is unambiguously American with an implicit `+`. After the fix, predicted vs
   actual draw rate matched to within 0.3pp.

Both bugs share a lesson already central to this project: a data pipeline that "runs"
can still be silently wrong; the guard is a **numeric sanity check against reality**
(here, favourite accuracy and draw calibration), not a green checkmark.

### De-vig, crosswalk, and orientation (`src/odds.py`, `build_odds_schema.py`)
- **De-vig = multiplicative normalisation.** Implied prob = 1/decimal-odds; the three
  sum to >1 by the bookmaker's margin (the "overround"/vig), so divide each by the sum
  to rescale to 1. Textbook default; Shin's method (margin weighted toward favourites)
  noted as a heavier alternative, not needed for a baseline.
- **Team-name crosswalk** (`_TEAM_ALIASES`): only ~10 Oddsportal spellings differ from
  our canonical names, but two are traps that a fuzzy match gets **wrong** — "Ireland"
  scores 1.00 against "Northern Ireland" (substring) yet means the **Republic**
  (confirmed by checking who "Ireland" actually played: France/Greece/Gibraltar =
  Republic's group); "D.R. Congo" fuzzy-matches "Congo". So aliases are hand-verified
  and anything unrecognised goes to a review queue — never an auto-accepted best guess,
  the same discipline as the player-rating matcher.
- **Date-tolerant join + orientation.** Oddsportal timestamps in the viewer's timezone,
  so its dates run ~1 day off ours; matches are joined by the unordered team pair within
  ±3 days, not on an exact date. For neutral-venue ties the source's "home" need not be
  ours, so the (P_home, P_draw, P_away) vector is re-oriented to our home/away by team
  identity. Result: **2,184 of 2,205** odds rows resolve to a DB fixture; 0 unresolved
  names, and the only unmatched fixtures are genuinely absent from our data (the
  abandoned/awarded Russia–Poland 2022 WCQ, one Congo–Niger date gap). Odds are stored
  as **metadata** in the training matrix (`book_ph/pd/pa`), never as features — feeding
  the market's own probability to the model would be circular.

### WC 2026 data refresh
The user asked to include WC 2026 (the most recent tournament). Our `results.csv`
(martj42) had WC 2026 only as unplayed future fixtures. Refreshed it (85 → 102 scored
WC 2026 matches), but the martj42 snapshot (2026-07-17) still had the **final and
3rd-place** unscored. Filled just those two from a daily-updated mirror
(`patateriedata`): final **Spain 1–0 Argentina**, 3rd place **France 4–6 England**
(the 6-4 flagged for human verification). Did *not* switch sources wholesale — the
mirror labels the World Cup "World Cup" not "FIFA World Cup" and lacks the `city`
column, which would break the tournament-label logic and importance weights built on
martj42's vocabulary.

### Results — the honest headline
Two evaluation surfaces, both scoring the model against the market on exactly the
matches that have a de-vigged price (`src/models/broader_eval.py` for the broad set;
`walk_forward.py` / `evaluate_wdl.py` gained a bookmaker column for the finals/cutoff
views). The model side is always scored leakage-safely (annual walk-forward: a fresh
model fit before each year, predicting that year's covered matches).

**Broad covered set (n=2,184 internationals, Euro 2020/2024 + WC 2022/2026 + qualifiers):**

| metric | model | bookmaker | form-fav | base-rate |
|---|---|---|---|---|
| accuracy | 0.630 | **0.654** | 0.623 | — |
| log-loss | 0.826 | **0.762** | — | 1.054 |
| Brier | 0.482 | **0.443** | — | 0.636 |

Paired per-match bootstrap on the log-loss difference:
- **model − bookmaker = +0.0645, 95% CI [+0.053, +0.076]** — the bookmaker is sharper,
  and the gap is real (CI clears 0), not noise. This is the *expected* result:
  bookmakers are the ~52–58% skill ceiling, and beating them is not the goal.
- **model − base-rate = −0.2273, 95% CI [−0.248, −0.206]** — the model beats the
  no-skill forecast decisively. This is the bar the model *must* clear, and does.

**Finals-only walk-forward (odds-covered subset n=153 of the 281 backtested finals):**
model accuracy 0.575 vs bookmaker 0.569 (model marginally ahead on raw accuracy), but
log-loss 1.015 vs 0.970 and Brier 0.606 vs 0.575 (bookmaker better calibrated). On the
small, evenly-matched finals set the model is competitive on top-1 picks; the market's
edge is in calibration.

**The story for the write-up:** the model is well-calibrated (held-out ECE 0.026),
beats every naive baseline on both accuracy and log-loss, and lands a small but
statistically real margin *behind* the bookmaker on probabilistic sharpness — while
being competitive on raw accuracy over the finals. That is exactly the credible,
defensible position a rigorous model should report: near the market, clearly above
no-skill, with the gap honestly quantified by a bootstrap CI rather than hand-waved.

## Poisson / Dixon-Coles scoreline model (the classical-statistics half)

The WDL classifier predicts Win/Draw/Loss directly with machine learning. The
scoreline model is deliberately the *other* thing: classical statistics, no ML.
Goals are count data, so the textbook-correct tool is the Poisson distribution —
fit each team an attack and a defence strength, turn those into an expected
goal-rate λ per side, and read a full distribution over every scoreline off the
two Poissons. It is built in `src/models/poisson.py`, incrementally, one
verifiable step at a time (the reason each phase below has its own sanity check).

**Why a separate method at all, not "more ML".** The judges' hook is
methodological rigor: use the right tool per sub-problem. WDL has nonlinear
feature interactions ML earns its keep on; a scoreline is count data where Poisson
is simply *correct*, and draws (which the WDL argmax structurally never picks —
see the draw-prediction section above) fall out of the score distribution for
free. Using classical stats here is the honest choice, not a weaker one.

### The model
Per team i: `attack[i]` (scoring) and `defence[i]` (prevention — HIGHER = better
here). Two globals: `base` (log average goals in a neutral average matchup) and
`home_adv` (extra log-rate at home). Everything is in log space so rates stay
positive and parameters add linearly:

```
log λ_home = base + home_adv·(1 − neutral) + attack[home] − defence[away]
log λ_away = base +                          attack[away] − defence[home]
```

### Phase 1 — prediction machinery (proved before any fitting)
`scoreline_matrix(λ_home, λ_away)` is the outer product of two Poisson PMFs (the
independence assumption); `wdl_from_grid` sums the lower-triangle / diagonal /
upper-triangle into P(H)/P(D)/P(A), renormalising for the tiny truncated tail so
they sum to exactly 1. Verified against hand arithmetic: `P(0-0)` matches
`e^-λ_h·e^-λ_a` to 6dp; a strong-vs-weak pair (2.8 vs 0.4) gives P(H)=0.87; equal
λ gives *exactly* symmetric P(H)=P(A) — a structural check that the triangle
summing isn't biased.

### Phase 2 — fitting attack/defence by maximum likelihood
`fit_strengths` minimises the negative Poisson log-likelihood (dropping the
constant `log y!`) over ~624 parameters (2 per team, 311 teams) + `base` +
`home_adv`, using `scipy.optimize` L-BFGS-B with an **analytic gradient** — the
residual `(λ − y)` scattered back onto each match's parameters via `np.bincount`
(numeric gradients would cost ~600 passes/step over 32k matches). A tiny ridge
(1e-3) both resolves a harmless additive redundancy — you can add a constant to
every attack and every defence without changing any rate — and mildly shrinks
low-data minnows; strengths are then centred to mean 0 (the shift folded into
`base`) so they read as deviations from an average international side.

**Verified against real football:** attack top-10 = Brazil/Germany/Spain/
Argentina/Netherlands/France/England/Portugal/Belgium/Croatia (the elite set);
bottom = San Marino/Bhutan/Guam/Cook Islands/Tonga (the genuine minnows). The
sharpest validation is **Italy landing top-6 in DEFENCE but outside the attack
top-10** — the model independently reproduced Italy's catenaccio signature, i.e.
it separated attack from defence rather than learning one "good team" axis.
`home_adv` came out ×1.31–1.35, matching the well-documented football home edge.

### Phase 3 — time weighting + the Dixon-Coles low-score correction
Two refinements, in `fit_dixon_coles`:
- **Time weighting** (`compute_weights`): each match's likelihood term is weighted
  by `recency_weight × importance_weight` — the *same* recipe the WDL model uses,
  so both halves of the project age history identically (10-year half-life; the
  Phase-4 backtest passes each fold's cutoff as the reference date so a fold never
  weights toward its own future).
- **Dixon-Coles ρ**: independent Poisson under-predicts 0-0/1-1 and over-predicts
  1-0/0-1. One parameter ρ scales the four low-score cells by the DC τ factors.
  Fitted in a **second stage** by 1-D search holding strengths fixed (a profile
  estimate): ρ only touches four cells so it is near-orthogonal to the strengths,
  and splitting the fit avoids hand-deriving the τ-gradient into the 624-parameter
  problem — standard practice, far less bug-prone.

**Verified:** ρ = **−0.047** (negative as theory requires; smaller than
Dixon-Coles' club-football −0.13, sensible for lopsided international fixtures).
On Spain-vs-Germany the correction moved exactly the right way — P(0-0)
0.044→0.048, P(1-1) 0.096→0.100 (up), P(1-0)/P(0-1) down, P(draw) +0.9pp. Time
weighting also visibly surfaced current form: **Spain rose to #1 in both attack
and defence** (Euro 2024 winners) and **Morocco entered the top-10 defence** (2022
WC semifinalists) — neither was there unweighted — while the elite ordering held.

**Out-of-sample check (fit pre-2016, score 2016+ WDL, n=9,889; 33 unseen-team
matches honestly skipped, never fabricated):** plain Poisson log-loss 0.9027 →
+time weighting 0.8926 (−0.0101) → +Dixon-Coles 0.8921 (−0.0107). Both refinements
help and neither hurts; **time weighting is the larger contributor**, Dixon-Coles
adds a small further gain on the *collapsed* WDL (its real value is in the
scoreline distribution, which matters for the Monte-Carlo work, not the H/D/A
collapse). All clear the ~1.05 base-rate comfortably.

### Phase 4 — the headline: Poisson vs XGBoost vs bookmaker, same honest harness
The scoreline model rides the *existing* walk-forward harnesses with zero changes
to them, via a thin adapter `PoissonWDL` (fit/predict_proba/best_iteration). The
trick: the harness hands the model `df[feature_columns]`, so for the Poisson run
we simply pass a different column list (`POISSON_FEATURE_COLUMNS` = match identity
+ goals) instead of the 21 engineered features. The harness-supplied `sample_weight`
(recency × importance) becomes the Dixon-Coles time weighting; `predict_proba`
reads only team names + neutral, never the score columns present in the frame (that
would leak the answer). `src/models/poisson_eval.py` runs both models through the
identical leakage-safe walk-forward, so their pooled truth / bookmaker / base-rate
vectors are row-aligned and every difference gets a paired bootstrap CI. **Sanity
anchor:** the XGBoost column reproduces the committed `broader_eval` numbers to 3dp
(acc 0.630, log-loss 0.826), so only the Poisson column is new.

**Broad odds-covered block (n=2,184 — qualifiers + Nations League + tournaments):**

| model | accuracy | log-loss | brier |
|---|---|---|---|
| **Poisson (Dixon-Coles)** | **0.643** | **0.795** | **0.464** |
| XGBoost WDL | 0.630 | 0.826 | 0.482 |
| bookmaker | 0.654 | 0.762 | 0.443 |
| base-rate | 0.469 | 1.054 | 0.636 |

Poisson − XGBoost = **−0.0314, 95% CI [−0.046, −0.018]** (Poisson significantly
sharper); Poisson − bookmaker = **+0.033** vs XGBoost − bookmaker = **+0.065** (the
Poisson model roughly HALVES the gap to the market); Poisson − base-rate = −0.259
(clears no-skill). The result is trustworthy precisely because it still *loses* to
the bookmaker — a leaking model would beat the market too; this has the right shape.

**Tournament finals only (n=153, odds-covered — Euro 2020/2024 + WC 2022/2026):**

| model | accuracy | log-loss | brier |
|---|---|---|---|
| Poisson (Dixon-Coles) | 0.536 | **0.986** | **0.584** |
| XGBoost WDL | **0.575** | 1.015 | 0.606 |
| bookmaker | 0.569 | 0.970 | 0.575 |

On the finals the Poisson still edges XGBoost on log-loss/brier and is
statistically *indistinguishable from the bookmaker* (Poisson − bookmaker +0.0155,
CI [−0.016, +0.046]), but Poisson − XGBoost = −0.029, CI [−0.098, +0.039] is now
**within noise** (n=153 can't resolve it), and XGBoost wins on finals *accuracy*
(0.575 vs 0.536, the usual argmax-vs-calibration tension). Both beat base-rate.

**Why the edge is huge on the broad block but shrinks on finals — an honest,
coherent reason.** The two models' *information* differs by block. Poisson's key
advantage is **opponent adjustment**: San Marino's goals-conceded looks less
catastrophic once the `defence[opponent]` term nets out that it plays Germany and
Spain, whereas XGBoost's `goal_trend` is a raw 10-year average with no opponent
correction. On the broad block XGBoost's squad/`rating_gap` features are mostly NaN
(they only populate on finals), so it runs on form alone and Poisson's strength
model dominates. On the **finals** XGBoost has its full feature set, so the two
converge to within noise. That is exactly the pattern the "right tool per
sub-problem" thesis predicts — and the write-up headline: *the classical-statistics
model is significantly sharper on the broad set and fully competitive with both the
ML model and the market on the tournaments themselves, demonstrated with paired
bootstrap CIs rather than asserted.*

### Phase 5 — does a rating_gap covariate earn its place? (no — and for a reason)
The deferred design question, tested not assumed: add `gamma·rating_gap` to the
Poisson rate (log λ_home += gamma·gap, log λ_away −= gamma·gap, one fitted
coefficient) vs plain attack/defence. `src/models/poisson_rating_ablation.py` runs
both variants through the identical annual walk-forward on the covered block and
pairs them with a bootstrap CI (the same pattern as the WDL ablation).

**Result: no effect, well-powered.** Plain 0.795 log-loss / 0.643 acc vs +rating
0.795 / 0.642 — the paired lift is **+0.0001, 95% CI [−0.0006, +0.0009]**, dead on
zero (point estimate marginally *positive*, i.e. a hair worse). This is not a
coverage artefact: **2,182 of 2,184** covered rows carry a non-NaN `rating_gap`, so
the covariate genuinely acts almost everywhere and still moves nothing. (The plain
column also reproduces Phase 4 to 3dp — a free regression check that adding the
covariate machinery didn't disturb the base fit.)

**Why redundant here when it wasn't clearly so for WDL — the coherent reason.** The
Poisson strengths ALREADY opponent-adjust team quality straight from goals (the
whole point of the `defence[opponent]` term). A FIFA-rating quality gap is just a
noisier proxy for the same thing, so it adds nothing on top. The WDL model's
`goal_trend` does NOT opponent-adjust, which is exactly why `rating_gap` was at
least a candidate there. Same feature, opposite conclusion, one mechanism — and it
reinforces the Phase 4 story that opponent adjustment is the Poisson model's real
edge. **Decision:** keep the covariate code (tested, `use_rating` flag, reproducible
ablation) but leave it OFF by default; plain Dixon-Coles is the model.

### Design idea (NOT yet built — hypothesis for the webapp "AI insight")
Phase 4 shows model strength is *block-dependent*: Poisson dominates the broad
block, the two converge on finals where XGBoost's squad/`rating_gap` features
populate. That suggests a **stage-aware router** for the webapp's planned "AI
insight" panel: serve the Poisson WDL on most matches (qualifiers, group stage,
friendlies — where it is significantly sharper), and hand over to XGBoost on the
late-knockout matches (finals, semis) and any other category where it empirically
exceeds Poisson. Rather than one model, present the *better* model per match type.

**Why this is a research stub and not a build item — three honesty constraints:**
1. **The routing boundary must be a leakage-safe, pre-match rule** (tournament
   stage, or — more to the point — *whether the row even has squad/rating features
   populated*), fixed on a validation split. It must NEVER be "pick whichever model
   turned out to win this specific match": that is exactly the test-set peeking that
   produced the +1.1pp→−1.1pp reversal documented in the tuning section. The router
   decision has to be reproducible from information available at kickoff.
2. **The current evidence is thin and metric-dependent.** On finals (n=153) XGBoost
   wins *accuracy* (0.575 vs 0.536) but Poisson still leads (within noise) on
   *log-loss/brier*, and the Poisson−XGBoost difference there is within noise. So
   "XGBoost handles finals" is true for the top-1 pick but not clearly for the
   probability quality — which model to route to depends on which metric the panel
   optimises. That needs a bigger finals sample and a pre-registered metric.
3. **A hard router is only one option.** A **calibrated blend** (a weighted average
   of the two probability vectors, weight fit on validation) frequently beats either
   model alone and sidesteps a brittle either/or boundary — it should be tested as
   the baseline the router must beat.

The real driver to investigate is probably **feature availability**, not "finals"
per se: XGBoost closes the gap exactly when its squad/rating columns are non-NaN.
So the cleanest router might key on that directly. Open questions to settle before
building: which stages/categories (group vs QF vs SF vs final; favourite vs
underdog; feature-populated vs not) each model actually wins, on what metric, with
paired CIs — and whether a blend beats routing. Until then the "AI insight" panel
should show a single honest model (Poisson, the broad-block winner), never a
routed/blended number dressed up as validated. Ties into the unbuilt "AI insight"
frontend item — build it only with real, validated numbers.

## Monte Carlo tournament simulator (Poisson Phases A–E)

The Poisson/Dixon-Coles model predicts one match's full scoreline distribution. A
*tournament* winner is not a formula over those — it branches combinatorially (who
you might meet depends on who they beat, and so on). The textbook tool is **Monte
Carlo**: play the whole tournament thousands of times, rolling each match on its
Poisson distribution, and read each team's win probability off the frequency it
lifts the trophy. Built in `src/models/montecarlo.py` (+ `src/tournaments/` config),
five verifiable phases, reusing the Phase 1–5 Poisson layer unchanged (numpy only,
no new dependency). Decision (from the plan): build + validate on **WC 2022** — the
clean 32-team format — first; Euro's 24-team "best-thirds" is a later extension.

### The one hard new cost: tournament STRUCTURE is not in the data
The `matches` table is only `date/teams/scores/tournament/neutral` — no stage,
group, or bracket column, and penalty knockouts are stored as draws with no winner.
So the group draw and knockout wiring must be **hand-encoded as SOURCE data**
(`src/tournaments/wc2022.json`, committed — unlike the regenerable DB). Team names
use the DB's spelling, verified against the `matches` table up front (the trap:
**Iran** not "IR Iran", **United States** not "USA", **South Korea** not "Korea
Republic"). The loader (`src/tournaments/__init__.py`) validates the config's own
arithmetic (8×4=32 teams, 8+4+2+1 bracket, every group-qualifier slot used once) and
asserts every team resolves to a fitted strength — failing LOUDLY, never letting a
typo become a silent league-average phantom (the no-fabrication rule).

### Phase A — the sampler, and one genuinely subtle correctness point
`sample_scorelines(grid, n, rng)` draws random `(home, away)` scores by sampling
CELLS from the flattened grid (`rng.choice(size, p=grid.ravel()/sum)`). The subtlety
that justifies sampling the grid rather than two `rng.poisson` draws: the Dixon-Coles
correction makes the two goal counts slightly *dependent* in the four low-score
cells. Independent Poisson draws would silently reproduce only the *uncorrected*
product model. Verified two ways at n=1e6: MC P(H/D/A) matches analytic
`wdl_from_grid` to within ~1·(MC std err ≈ 5e-4) on a DC-corrected grid; and with
rho=0 the grid-sampler matches naive independent `rng.poisson` (the independence
sanity — they must agree exactly there).

### Phases C–D — group + knockout simulators (vectorised across sims)
`simulate_group` plays all C(4,2)=6 fixtures at neutral venues, tallies 3/1/0 points
and ranks by points → goal difference → goals for. Two documented **simplifications**
vs FIFA's real rules: (1) FIFA breaks ties on head-to-head *before* overall goal
difference — fiddly to vectorise, rarely changes who advances; (2) exact remaining
ties are split at random (a tiny per-team random key), standing in for
fair-play/drawing-of-lots, so no team gains a systematic edge from list position.
The composite ranking key packs (points, gd, gf, random) into one float with scaled
gaps so plain descending order reproduces the lexicographic tie-break.

`_play_ties` plays a whole knockout round across all `n` sims at once. The wrinkle:
different sims send different teams to the same bracket slot, so a "match" is really
up to (#teams)² distinct matchups. It groups sims by unique (home, away) ROW
(`np.unique(axis=0)` — a string separator is unsafe, numpy trims embedded null
bytes, a bug caught in testing) and batch-samples each matchup once, caching grids
across rounds. A drawn tie resolves by a **~50/50 shootout coin flip** — a documented
placeholder (shootouts ≈ coin tosses and are stored as draws with no winner; a fitted
shootout model is a later refinement). `simulate_tournament` returns per-team
P(reach R16/QF/SF/final/win) with three assertions baked in: reach-probs monotone by
round, champion probs sum to 1, and each round's reach-probs sum to its slot count
(16/8/4/2). 20,000 sims run in ~0.5s; reproducible under a fixed seed.

### Phase E — honest backtest on WC 2022, and its two real limitations
Fitting strengths strictly **before** WC 2022 (`matches[date < t_start]`,
`reference_date=t_start` — the walk_forward "before T" slice, so it is a true
pre-tournament forecast), the seeded simulation is sane and matches reputation:
Brazil / Spain / Argentina / England / Netherlands / France top the board. The actual
champion **Argentina ranks 3rd** by P(win) (0.106), finalist **France 6th**; both
firmly top-tier. The two famous overachievers land honestly low — Croatia (SF) 12th,
Morocco (SF) 20th — the model does not, and should not, retro-predict those runs.
Round-reach scoring across all 32 teams × 5 stages (160 predictions) **beats the
no-skill base-rate baseline** on both metrics: Brier 0.105 vs 0.127, log-loss 0.325
vs 0.401 (`src/models/montecarlo_eval.py`).

**Two limitations stated plainly, not papered over:** (1) this is ONE tournament (≤4
even adding Euro 2016/2020/2024) — the champion is a single Bernoulli draw, so
tournament-winner *calibration* cannot be strongly validated; the 160-prediction
Brier is a sanity check, not a calibration proof. (2) There is **no outright-winner
odds market** anywhere in this project (odds are per-match 1X2 only), so P(win
trophy) has no bookmaker benchmark — unlike the per-match engine, which IS CI-
validated (Phase 4). The simulator's rigor rests on that validated per-match engine
plus round-level sanity.

### Later (explicitly out of scope here)
Euro's 24-team "best-thirds" format + config; a fitted shootout model in place of the
coin flip; the single-match MC as an explicit analytic cross-check inside the
pipeline; and the clearly-labelled hypothetical Euro 2028 run (needs the Euro format
+ a projected group draw). `python -m src.models.montecarlo` runs the Phase-A
cross-check and a seeded WC 2022 simulation; `python -m src.models.montecarlo_eval`
runs the Phase-E backtest.