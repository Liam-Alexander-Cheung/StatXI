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