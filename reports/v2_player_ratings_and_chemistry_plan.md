# v2 Plan: Player Ratings, Cross-Source Name Matching & Team Chemistry

> **Status:** PLAN ONLY — nothing here is built yet. Written 2026-07-27 to be
> executed in a later session. Do the work **one step at a time, verifying each
> against a real checkable case before the next**, per `agents.md`.

---

## 0. READ THIS FIRST (onboarding for a fresh session)

You are picking up a Jugend forscht Euro-2028 match-prediction project. Before
touching anything, read, in order:

1. `agents.md` (project rules — one-step-at-a-time, never fabricate data, verify
   against real cases, document dead ends in methodology.md).
2. `reports/methodology.md`, especially the newest sections:
   - **"XGBoost Win/Draw/Loss classifier (v1)"** — the model this plan feeds.
   - **"Name-matching risk (Transfermarkt search)"** and **"Relational schema"**
     — the matching problem this plan finally solves.
   - **"Environment: XGBoost needs OpenMP"** — libomp is already installed.
3. The v1 model modules: `src/models/build_matrix.py`, `train_wdl.py`,
   `evaluate_wdl.py`. **This plan's features get added to those.**

**Why this work exists (the one-sentence motivation):** v1's XGBoost model beats
naive baselines *overall* (accuracy 0.579, log-loss 0.909, ECE 0.024) but at the
**major-tournament level it only tied the naive "pick the higher-form team"
baseline** (accuracy 0.492 vs 0.495; log-loss 1.026 vs 1.088). The diagnosis:
tournaments are evenly-matched neutral-venue games where *recent form barely
separates teams* — the missing signal is **squad quality** (how good are the
players) and **squad cohesion** (do they play together). This plan adds both.

**Environment facts you'll need (don't rediscover them):**
- Python 3.9, venv at `./venv` — `source venv/bin/activate` first.
- Run modules from repo root as `python -m src.models.build_matrix` etc. (the
  `-m` form puts repo root on `sys.path` so `import src...` resolves).
- Put `from __future__ import annotations` at the top of any new module using
  `X | None` type hints — 3.9 errors on them otherwise.
- **No parquet engine installed** — cache to CSV (or `pip install pyarrow` first
  if you want parquet; that's a decision to raise with the user).
- SQLite FK constraints are OFF by default — `PRAGMA foreign_keys = ON` per
  connection (see `src/database.py`).
- **Never commit derived/regenerable data** — add any new scraped/cached/DB
  artifacts to `.gitignore` (see how `data/processed/*`, `data/statxi.db`,
  `src/models/wdl_xgb.json` are handled).
- **No self-directed browser testing** — the user runs visual/browser checks.

---

## 1. Goal & success criteria

**Goal:** give the model (and later the Poisson scoreline sim) two new feature
families computed at the tournament-squad level:
- **Player-rating features** — squad quality from an external per-player rating.
- **Team-chemistry features** — squad cohesion, modelled on video-game chemistry
  (players who share a club/league understand each other on the pitch).

**Definition of done (measurable, not vibes):** re-running `evaluate_wdl.py`
after adding these features shows, **on the MAJOR-TOURNAMENT subset**, an
improvement over the v1 numbers above — ideally accuracy that *beats* the naive
0.495 baseline and a log-loss below 1.026. If a feature family does **not**
help, that is a finding to document, not to hide (see the ablation in §7).

**Non-negotiable discipline for every new feature — leakage:** a feature for a
match may only use information available **before kickoff**. For squad features
this means **era-correct data**: a player's rating/club for Euro 2016 must be the
FIFA 16 rating and his 2016 club — *never* today's. This is automatic for `club`
(the DB already stores the as-of-tournament club) and must be enforced manually
for ratings by matching each tournament to its rating edition (§4).

---

## 2. What we already have locally (don't re-scrape this)

`data/statxi.db`, table `players` (10,038 rows, one per player-per-tournament):

| column | notes |
|---|---|
| `player_appearance_id` | PK — **one row = one player at one tournament** (already era-scoped) |
| `squad_id` → `squads.squad_id` | `squads` has `country` + `tournament_id` |
| `player_name` | raw Wikipedia name (diacritics, footnotes already stripped) |
| `position` | GK/DF/MF/FW only |
| `date_of_birth` | **full DOB** — a near-unique match key |
| `age_at_tournament`, `caps`, `goals` | |
| `club` | **100% populated**, as-of-tournament — powers chemistry with zero scraping |
| `transfermarkt_player_id` | **100% NULL** — the column §4 fills |

`tournaments`: `name`, `year`, `competition`, `wiki_page` — `year` gives the
rating edition to use (§4). 18 tournaments (Euro 1992–2024, WC 1990–2022).

**Key consequence:** because ratings attach to `player_appearance_id` (a single
tournament), we **never need to solve "is Messi@2010 the same human as
Messi@2014"** — each appearance is matched independently to its era's rating.
This deliberately sidesteps the unresolved cross-tournament identity limitation
documented in methodology.md.

---

## 3. Workstream A — Team Chemistry (build FIRST; needs no scraping)

Do this first: it's fully local, unblocks a real model improvement immediately,
and is independent of the scraping/matching risk in later workstreams.

### 3.1 Concept (adapted from video-game chemistry, honestly)
FIFA chemistry rewards links between players who share a **club**, **league**, or
**nationality**. For an international squad the **nationality link is degenerate**
(everyone is the same nationality) — so it carries no information here. The
discriminating signals are **club** and (once ratings give us leagues) **league**
concentration. Football intuition + history back this: cohesive tournament sides
often had a dominant club spine (Spain 2010 — Barça/Real core; Germany 2014 —
Bayern core), while club-scattered squads (many England sides) less so.

### 3.2 Metrics — `src/features.py::team_chemistry(squads, team, tournament_name)`
Squad-scoped (like `squad_age_depth`), returns a dict. Given the squad's list of
`club` values, let `n` = squad size and `c_i` = count of players at distinct club
*i*:

| feature | formula | meaning |
|---|---|---|
| `largest_club_bloc` | `max(c_i)` | biggest single-club spine |
| `top2_club_bloc` | sum of two largest `c_i` | two-club spine (e.g. Barça+Real) |
| `club_hhi` | `Σ (c_i/n)²` | Herfindahl concentration; 1.0 = all one club, ≈1/n = all different |
| `same_club_pairs` | `Σ C(c_i, 2)` | number of teammate pairs sharing a club |
| `same_club_pair_ratio` | `same_club_pairs / C(n,2)` | normalized pair density (main feature) |
| `n_distinct_clubs` | count of distinct clubs | inverse breadth |

- **Missing-data rule:** if `team`/`tournament_name` matches no squad → all NaN /
  0 exactly like `squad_age_depth` does (counts 0, ratios NaN; never a guessed
  value).
- **Optional refinement (note, don't build yet):** weight players by likely
  starting status. We have no lineup data, but `caps` is a rough proxy (more caps
  → more likely a starter); a caps-weighted `same_club_pair_ratio` could matter
  more than the raw squad-level one. Flag as a follow-up ablation.
- **League-level chemistry** (`league_hhi`, `same_league_ratio`) is the same math
  on leagues instead of clubs — **deferred to Workstream B**, since we have no
  club→league map until the ratings dataset (which carries `league`) is joined.

### 3.3 Verification (real, checkable — required before trusting it)
- **Germany, World Cup 2014:** should show a large Bayern Munich bloc (Neuer,
  Lahm, Boateng, Müller, Kroos, Schweinsteiger, Götze) → high `largest_club_bloc`
  and `club_hhi`. Cross-check by a direct `SELECT club, COUNT(*) ... GROUP BY
  club` on that squad.
- **Spain, Euro 2012 or WC 2010:** Barça + Real spine → high `top2_club_bloc`.
- **A minnow / broadly-scattered squad** (e.g. an African or Asian debutant whose
  players are spread across many domestic/lower clubs) → low concentration,
  high `n_distinct_clubs`. Confirms the metric discriminates, not just "runs".
- Print the `GROUP BY club` breakdown next to the computed metrics for 2–3
  squads and eyeball that the numbers correspond.

### 3.4 Web layer (optional, matches existing pattern)
If desired, expose `/api/team-chemistry?team=&tournament=` in `webapp/app.py`
mirroring `/api/squad-age-depth`, warmed at startup. Not required for the model.

---

## 4. Workstream B — Player-rating data acquisition

### 4.1 Source decision (recommended order)
1. **EA/FIFA ratings (PRIMARY).** Overall (0–99) + **potential** + club + league +
   nationality + DOB/age, published **per yearly edition** (FIFA 92→FC), so
   historical editions map cleanly to our tournaments (FIFA 16 → Euro 2016,
   FIFA 22 → WC 2022, etc.). Subjective (EA's assessment) but a well-established
   team-strength proxy in football analytics, and **its `potential` rating
   directly captures quality for young players who haven't played much** — the
   exact cold-start case that motivated this plan.
   - **Easiest acquisition:** a pre-scraped historical dataset (e.g. Kaggle
     "complete FIFA player datasets" published per edition) → a static download,
     **no scraping, no anti-bot, no ToS gray-zone**. Verify licensing allows
     non-commercial research use; record it in methodology.md.
   - **Fallback:** scrape SoFIFA (hosts current + historical editions via version
     URLs). More permissive than Transfermarkt but still rate-limited — **be
     polite: throttle + actually cache** (see §4.3).
2. **Transfermarkt market value (SECONDARY / contingency).** Code already exists
   (`fetch_market_value_history`, `search_player_id`) but is **WAF-blocked as of
   2026-07-27** (`202` + `x-amzn-waf-action: challenge`). Re-test from a different
   network first (see methodology). Market value is another quality proxy; keep
   as a cross-check or a second `source` row, not the primary path.
3. **Team-level Elo (OPTIONAL, trivial).** eloratings.net / a Kaggle Elo dataset
   gives a per-date team strength number — cheap to add as its own feature, but
   it needs match history so it does **not** solve the debutant cold-start; treat
   as a bonus baseline-strength feature, not part of the squad-rating work.

### 4.2 Storage — new table `player_ratings` (source-agnostic)
Add via a new `build_ratings_schema.py` (follow the safe pattern in
methodology.md: **CSV-backup before any DROP**, `clean_player_name` tolerates
`None`, `PRAGMA foreign_keys = ON`).

```
player_ratings(
  rating_id            INTEGER PRIMARY KEY,
  player_appearance_id INTEGER,   -- FK -> players (era-correct by construction)
  source               TEXT,      -- 'fifa' | 'transfermarkt' | ...
  source_player_id     TEXT,      -- external id, for re-fetch/audit
  rating_edition       TEXT,      -- e.g. 'FIFA16' — MUST match the tournament year
  overall              INTEGER,   -- 0-99 (fifa)
  potential            INTEGER,   -- 0-99 (fifa) — prodigy signal
  value_eur            REAL,      -- market value (transfermarkt)
  league               TEXT,      -- enables league-chemistry in §3.2
  match_confidence     REAL,      -- 0..1 from §5
  match_method         TEXT,      -- 'exact_dob_club' | 'name_club_age' | 'manual'
  FOREIGN KEY(player_appearance_id) REFERENCES players(player_appearance_id)
)
```
Keying to `player_appearance_id` means one rating row per (player, tournament),
inherently era-correct. Gitignore any downloaded raw rating files under
`data/raw/`.

### 4.3 Caching (build it for real this time)
methodology.md flags that a previous "everything is cached" claim was aspirational
and never true. **Do not repeat that.** Any network fetch must write to a local
cache (e.g. `data/raw/ratings_cache/<source>/<edition>/…`) and read from it before
hitting the network. A static-dataset download is inherently cached; a scraper is
not — give it a real on-disk cache + a throttle.

---

## 5. Workstream C — Name normalization & cross-source matching (the hard core)

This is the genuinely difficult part (see methodology "Name-matching risk"). The
job: for each of the 10,038 `players` rows, find the correct rating row for that
player **in that tournament's edition**, or leave it unmatched (NULL) — **never a
guessed match** (fabrication rule).

### 5.1 New module `src/name_matching.py`
- `normalize_name(s) -> str`: Unicode **NFKD** decomposition + strip combining
  diacritics (Müller→muller, İlkay→ilkay), lowercase, drop punctuation, collapse
  whitespace, handle `"Last, First"` → `"First Last"`. Explain NFKD in a comment
  (it splits an accented char into base char + combining mark so the mark can be
  removed). Handle single-name players (Brazilian "Hulk", "Fred") and
  two-surname Spanish/Portuguese names (keep all tokens; match on token *sets*).
- `name_similarity(a, b) -> float` in `[0,1]`: token-set similarity. **Preferred:**
  `rapidfuzz.fuzz.token_set_ratio` (fast, robust to word order/extra tokens) —
  but that's a **new dependency**; raise `pip install rapidfuzz` with the user, or
  fall back to stdlib `difflib.SequenceMatcher` / a token Jaccard if they'd rather
  not add one.

### 5.2 Matching algorithm (blocking → score → tier)
For each `players` row (with its `country` via `squads`, `date_of_birth`, `club`,
`age_at_tournament`, and its tournament's `rating_edition`):
1. **Block** the candidate rating rows to the same `rating_edition` **and** same
   nationality. This cuts the search space massively and kills most false
   positives (the two "Silva" results with no name overlap could never survive a
   nationality+DOB block).
2. **Score** each surviving candidate with a weighted combination:
   - **DOB exact match** (if the source has DOB): near-decisive — huge weight.
     Otherwise `age` within ±1 year.
   - **club match** (normalized): strong. Note club names differ across sources
     ("Man Utd" vs "Manchester United") → normalize/alias clubs too (a small
     `club_aliases` lookup, grown as mismatches appear — same spirit as
     `former_names`).
   - **name_similarity**: the tie-breaker/confirmation, not the sole key
     (methodology showed search matches birth names not shown publicly).
3. **Confidence tiers** (never auto-accept weak matches):
   - **auto-accept** (`match_method='exact_dob_club'` or high combined score):
     write the rating row.
   - **review queue** (medium): write to `data/processed/match_review_queue.csv`
     (player, tournament, candidate, scores) — this is **Tomás's manual-
     verification workstream** (agents.md: he owns manual data verification).
   - **reject/NULL** (low): leave unmatched; the model handles NaN natively.
4. Record `match_confidence` + `match_method` on every written row for audit.

### 5.3 Verification (real, checkable)
- **Exact positive controls:** Lionel Messi (DOB 1987-06-24, Barcelona) in the
  2010 & 2014 editions; Cristiano Ronaldo; a couple of GKs. Confirm exact
  DOB+club matches land with top confidence.
- **The "Silva" ambiguity:** confirm nationality+DOB blocking now resolves cases
  that bare-name search could not — i.e. the disambiguation the methodology said
  was unsolved is now handled, *with the evidence* (before: 8 candidates; after:
  1 within the block).
- **Coverage report:** print matched / review-queued / unmatched counts per
  tournament. Expect worse coverage for older tournaments (1990s) and minnows
  (lower-league players missing from rating sets) — **document the coverage gaps
  honestly**; they are a real limitation, not a bug to paper over.

---

## 6. Workstream D — Rating-based squad features

Once `player_ratings` is populated, add to `src/features.py` (squad-scoped, NaN
where coverage is missing — **never impute**):

- `squad_rating(squads_or_db, team, tournament_name) -> dict`:
  - `mean_overall`, `top11_mean_overall` (mean of the 11 highest — proxies the
    likely starting XI without lineup data), `max_overall`, `std_overall`
    (spread — a top-heavy vs balanced squad).
  - `mean_potential`, and `mean(potential - overall)` — the **prodigy/upside**
    signal (young squads with headroom); ties into the deferred prodigy z-scores
    in agents.md.
  - `coverage` = fraction of the squad that got matched — **carry this as its own
    feature** so the model can discount squads whose rating data is thin, instead
    of being misled by an average over 3 of 23 players.
- **League chemistry** now becomes buildable — extend `team_chemistry` (§3.2)
  with `league_hhi` / `same_league_ratio` using the joined `league` column.

Normalization note (from agents.md): raw values may be z-scored **at the squad-
cohort level by the caller**, not inside the feature (same decision as
`transfer_value_delta_z`).

---

## 7. Workstream E — Integrate, retrain, ablate, backtest

1. **Extend the training matrix.** In `build_matrix.py`, add the new squad
   features to each match's row by looking up the home & away teams' squads *for
   that match's tournament*. **Caveat:** squad features only exist for the 18
   major tournaments, so they'll be NaN for the ~99% of matches that are
   friendlies/qualifiers — that's fine (XGBoost handles NaN), and the payoff is
   precisely on the tournament rows we care about. Add every new column name to
   `FEATURE_COLUMNS` (the leakage guard) — and keep raw scores out of it.
   - Re-verify no leakage: a built row's new feature equals a standalone feature
     call for that team+tournament (same check pattern used in v1).
2. **Retrain** (`train_wdl.py`, unchanged logic) and **re-evaluate**
   (`evaluate_wdl.py`).
3. **Ablation — this is the scientific core, not optional.** Chemistry may be
   *confounded with quality* (good players cluster at good clubs), so it must be
   shown to add signal **beyond** ratings. Compare held-out (esp.
   major-tournament) log-loss across: (a) v1 features only; (b) v1 + ratings;
   (c) v1 + ratings + chemistry; (d) v1 + chemistry only. This mirrors the
   methodology's standing rule ("don't add a squad-quality covariate until a
   backtest shows the existing feature doesn't already capture it"). Whatever the
   result — including "chemistry didn't help beyond ratings" — **write it up**.
4. **Feature importance / direction check:** confirm the model uses the new
   features sensibly (e.g. higher `top11_mean_overall` → higher win prob for the
   stronger side) and that calibration (ECE) doesn't degrade.

---

## 8. Suggested execution order (each an independently verifiable checkpoint)

1. **A — Team chemistry from existing data** (`team_chemistry` in features.py) +
   its verification (Germany 2014 Bayern bloc, Spain spine, a scattered minnow).
   *Ship & verify before anything network-dependent.*
2. **B1 — Acquire one FIFA edition** (start with one, e.g. FIFA 16) as a static
   dataset; inspect columns/coverage/licensing; design `player_ratings` schema;
   build `build_ratings_schema.py` (with CSV backup + FK pragma).
3. **C — `name_matching.py`** + match that one edition to Euro 2016's squads;
   verify positive controls + the Silva block; print a coverage report.
4. **B2 — Scale acquisition + matching to all editions** that map to a backtest
   tournament (FIFA 16/18/20/22/24 → Euro 2016, WC 2018, Euro 2020, WC 2022,
   Euro 2024), plus older editions as coverage allows.
5. **D — Rating features + league chemistry** in features.py, verified on real
   squads.
6. **E — Integrate into build_matrix, retrain, evaluate, ablate**; update
   methodology.md with results (including any negative/coverage findings).

Stop and involve the user at: the `pip install rapidfuzz`/`pyarrow` decisions,
the dataset-licensing check, and any auto-accept confidence threshold (it decides
how much lands in Tomás's manual review queue).

---

## 9. Risks, open questions, contingencies

- **Matching precision vs coverage trade-off** — a loose threshold fabricates
  wrong links (violates the no-fabrication rule); a tight one leaves many NaN.
  Default to *tight + review queue*; let the ablation tell you if coverage is too
  thin to help.
- **Historical coverage decay** — 1990s editions and minnow players may be absent
  from rating sets. Ratings may only meaningfully improve *recent* tournaments;
  that's an acceptable, documentable outcome.
- **Chemistry ≠ quality must be proven**, not assumed (see §7.3). Also: does the
  squad-level metric wash out because only ~11 of 23 start? Test the caps-weighted
  variant if the plain one underperforms.
- **Club-name normalization across sources** is a mini name-matching problem of
  its own — budget for a `club_aliases` lookup.
- **Transfermarkt WAF** may still block the secondary source — the plan does not
  depend on it (FIFA static dataset is the primary path).
- **New dependencies** (`rapidfuzz`, maybe `pyarrow`) — get user sign-off; stdlib
  fallbacks exist for matching.
- **ToS/licensing** — SoFIFA scraping is a lighter gray-zone than Transfermarkt
  but still real; a static licensed dataset is cleanest. Document whichever is
  chosen in methodology.md, same as the existing Transfermarkt ToS note.

## 10. Files this plan will create/modify (map for the executor)
- **new:** `src/name_matching.py`, `build_ratings_schema.py`,
  `src/models/` (no new model file — reuse v1), `data/raw/ratings_cache/…`
  (gitignored), `data/processed/match_review_queue.csv` (gitignored).
- **modify:** `src/features.py` (+`team_chemistry`, +`squad_rating`),
  `src/data_pipeline.py` (loader for `player_ratings` join if needed),
  `src/models/build_matrix.py` (new columns + `FEATURE_COLUMNS`),
  `.gitignore` (new derived artifacts), `reports/methodology.md` (findings),
  `agents.md` (mark items done / update "What's NOT done yet").
- **DB:** new `player_ratings` table; `players.transfermarkt_player_id` finally
  populated (for `source='transfermarkt'` matches).
