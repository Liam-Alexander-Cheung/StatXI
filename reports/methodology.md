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