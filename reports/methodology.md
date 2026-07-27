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

The database file itself (`data/euro2028.db`) is not committed to the
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