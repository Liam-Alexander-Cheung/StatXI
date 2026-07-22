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
but not zero. Mitigations: requests are rate-limited, and every fetched page
is cached locally to avoid redundant re-scraping.

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