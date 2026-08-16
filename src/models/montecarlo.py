"""
Monte Carlo tournament simulator — the last piece of the classical-statistics half.

The Poisson/Dixon-Coles model (src/models/poisson.py) predicts, for ONE match, a
full probability distribution over exact scorelines (0-0, 1-0, 2-1, ...). That is
enough to say "how likely is each result of Spain vs Germany". It is NOT enough to
answer "who WINS the tournament", because a tournament is a whole structure of
matches — a group stage that decides who advances, then a knockout bracket where the
winner of each tie plays the winner of the next.

There is no closed-form formula for "P(Argentina lifts the trophy)": it depends on
who they might meet, who those teams might have beaten, and so on, branching
combinatorially. The standard tool for that is MONTE CARLO SIMULATION — literally
play the whole tournament thousands of times, each time rolling the dice on every
match according to its Poisson scoreline distribution, and count how often each team
reaches each round. The fraction of simulations a team wins IS its estimated
probability of winning.

BUILT INCREMENTALLY (see reports/methodology.md and reports/montecarlo_plan.md):
  PHASE A — the SAMPLER: turn one match's scoreline grid into random (h, a) scores,
            and prove the sampler reproduces the analytic model before building on it.
  (Phases B-E — tournament structure, group sim, knockout sim, backtest — to follow.)

Everything reuses src/models/poisson.py unchanged; no new dependency (numpy only).
"""

from __future__ import annotations

import numpy as np

from itertools import combinations

import pandas as pd

from src.tournaments import validate_teams
from src.models.poisson import (
    scoreline_matrix,
    apply_dixon_coles,
    wdl_from_grid,
    predict_match,
)


# ============================================================================
# PHASE A — the single-match sampler
# ============================================================================

def sample_scorelines(grid: np.ndarray, n: int, rng: np.random.Generator
                      ) -> tuple[np.ndarray, np.ndarray]:
    """
    Draw `n` random scorelines (home_goals, away_goals) from a scoreline `grid`.

    `grid[i, j]` is the probability of the exact scoreline (home i, away j). To draw
    a random scoreline we treat the whole grid as a single categorical ("pick one
    cell") distribution: flatten the 2-D grid into a 1-D list of cells, sample cell
    indices with probability proportional to each cell's value, then translate each
    chosen flat index back into its (row, column) = (home goals, away goals).

    *Critical subtlety, and the whole reason this function samples the GRID rather
    than two independent Poisson draws:* the Dixon-Coles correction makes the two
    teams' goal counts slightly DEPENDENT in the four low-score cells. If we sampled
    `rng.poisson(lambda_home)` and `rng.poisson(lambda_away)` independently we would
    reproduce only the UNCORRECTED product model — silently throwing away the
    correction the model was built to include. Sampling cells from the corrected grid
    keeps whatever correction is baked into `grid`. (With rho=0 the grid IS the
    independent product, so this also covers the plain-Poisson case.)

    Returns two int arrays (home_goals, away_goals), each length `n`.
    """
    n_rows, n_cols = grid.shape
    # Flatten to 1-D. .ravel() reads the grid in row-major (C) order, so flat index
    # f corresponds to (row, col) = (f // n_cols, f % n_cols) — recovered below.
    flat = grid.ravel()
    # The grid sums to ~0.999999 (truncated at max_goals, and DC re-weighted), so we
    # normalise to a proper probability vector. This is the same honest
    # renormalisation of retained mass that wdl_from_grid does.
    p = flat / flat.sum()

    # rng.choice with p= draws categorical outcomes: each is a flat cell index in
    # [0, n_rows*n_cols), chosen with probability p. One vectorised call for all n.
    flat_idx = rng.choice(flat.size, size=n, p=p)

    # Translate flat indices back to (home goals, away goals). np.divmod does both
    # the integer-divide (row) and remainder (column) at once.
    home_goals, away_goals = np.divmod(flat_idx, n_cols)
    return home_goals, away_goals


def _wdl_from_samples(home_goals: np.ndarray, away_goals: np.ndarray
                      ) -> tuple[float, float, float]:
    """Empirical (P(home win), P(draw), P(away win)) from sampled scorelines —
    the Monte Carlo counterpart of poisson.wdl_from_grid. Just the fraction of
    sampled matches in which the home side scored more / equal / fewer goals."""
    n = len(home_goals)
    p_home = float(np.mean(home_goals > away_goals))
    p_draw = float(np.mean(home_goals == away_goals))
    p_away = float(np.mean(home_goals < away_goals))
    return p_home, p_draw, p_away


# ============================================================================
# PHASE C — the group-stage simulator
# ============================================================================

def _sample_fixture(strengths: dict, home: str, away: str, n: int,
                    rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample `n` scorelines for one NEUTRAL-venue fixture. Reuses predict_match to
    build the (Dixon-Coles-corrected) grid — exactly the model the per-match backtest
    validated — then draws from it with the Phase-A sampler. Group and knockout games
    at a World Cup are all at neutral venues, so home_adv is switched off."""
    pred = predict_match(strengths, home, away, neutral=True)
    if pred is None:  # should never happen: validate_teams runs first
        raise ValueError(f"No fitted strength for {home!r} or {away!r}")
    return sample_scorelines(pred["grid"], n, rng)


def simulate_group(teams: list[str], strengths: dict, rng: np.random.Generator,
                   n: int) -> dict:
    """
    Simulate a 4-team group `n` times and return, for each simulation, which two
    teams qualify (as group winner and runner-up).

    A group is a round-robin: every team plays every other once — C(4,2)=6 fixtures.
    The match SCHEDULE is irrelevant to the final table (all six are played), so we
    just sample every pairing `n` times and tally standings across all `n` parallel
    simulations at once (vectorised — one batched draw per fixture, not per sim).

    Points: 3 for a win, 1 for a draw, 0 for a loss (standard). Teams are then ranked
    by (1) points, (2) goal difference, (3) goals for. This is a documented
    SIMPLIFICATION of FIFA's real tie-break order, which puts head-to-head results
    before overall goal difference; head-to-head is fiddly to vectorise and rarely
    changes who advances, so we use the simpler global order and note it here. Exact
    remaining ties (equal on all three) are broken at RANDOM (a tiny per-team random
    key), standing in for FIFA's fair-play/drawing-of-lots final steps, so no team
    gets a systematic edge from its position in the list.

    Returns {"winners": (n,) array of team names ranked 1st,
             "runners_up": (n,) array ranked 2nd,
             "thirds": (n,) array ranked 3rd,
             "third_key": (n,) ranking-key VALUE of that 3rd-placed team}.
    The last two are for the Euro 24-team "best thirds" format, where the four best
    third-placed teams across the six groups also advance; the WC (top-2) path simply
    ignores them.
    """
    T = len(teams)                       # 4
    points = np.zeros((n, T), dtype=np.int32)
    gf = np.zeros((n, T), dtype=np.int32)   # goals for
    ga = np.zeros((n, T), dtype=np.int32)   # goals against

    for i, j in combinations(range(T), 2):          # the 6 fixtures
        hg, ag = _sample_fixture(strengths, teams[i], teams[j], n, rng)
        i_win = hg > ag
        j_win = ag > hg
        draw = hg == ag
        points[:, i] += 3 * i_win + draw            # bool -> 0/1 in arithmetic
        points[:, j] += 3 * j_win + draw
        gf[:, i] += hg; ga[:, i] += ag
        gf[:, j] += ag; ga[:, j] += hg

    gd = gf - ga
    # Composite ranking key, one number per (sim, team), engineered so a strict
    # lexicographic order (points, then gd, then gf) is preserved as plain
    # descending numeric order. Each field is scaled to sit strictly below the next:
    #   points in [0,9]; gd shifted to >=0 then *1000; gf in [0, ~30]; +random<1
    #   final tiebreak. Chosen gaps (1e6, 1e3) exceed the max of the field below.
    key = (points.astype(np.float64) * 1_000_000
           + (gd + 100) * 1_000
           + gf
           + rng.random((n, T)))                    # random split of exact ties
    # Rank teams within each simulation, best first. argsort is ascending, so negate.
    order = np.argsort(-key, axis=1)                # (n, T) team indices, best->worst
    teams_arr = np.asarray(teams)
    winners = teams_arr[order[:, 0]]
    runners_up = teams_arr[order[:, 1]]
    # 3rd place too, plus that team's key VALUE. Every group scores its teams with the
    # identical key formula, so a group's 3rd-placed key is directly comparable to
    # another group's — which is exactly what the caller needs to rank the six thirds
    # against each other and pick the four best (the Euro format). take_along_axis
    # gathers key[sim, order[sim, 2]] for every sim at once.
    thirds = teams_arr[order[:, 2]]
    third_key = np.take_along_axis(key, order[:, 2:3], axis=1)[:, 0]
    return {"winners": winners, "runners_up": runners_up,
            "thirds": thirds, "third_key": third_key}


def group_qualification_probs(teams: list[str], sim: dict) -> dict:
    """Turn one simulate_group result into per-team P(win group), P(runner-up),
    P(qualify) — for eyeballing against who actually advanced (Phase C verify)."""
    n = len(sim["winners"])
    out = {}
    for t in teams:
        p_win = float(np.mean(sim["winners"] == t))
        p_run = float(np.mean(sim["runners_up"] == t))
        out[t] = {"win": p_win, "runner_up": p_run, "qualify": p_win + p_run}
    return out


# ============================================================================
# PHASE D — knockout + full-tournament simulator
# ============================================================================

def _play_ties(home: np.ndarray, away: np.ndarray, strengths: dict,
               rng: np.random.Generator, grid_cache: dict) -> np.ndarray:
    """
    Play one knockout ROUND across all `n` simulations and return the (n,) array of
    winners' names. `home`/`away` are (n,) team-name arrays — DIFFERENT sims can have
    different teams in the same bracket slot (that is the whole point of simulating),
    so this is not a single fixture but up to (#teams)^2 distinct matchups.

    A knockout tie is decided by:
      1. sample a scoreline from the (neutral, DC-corrected) grid;
      2. if it is a DRAW, resolve it by a ~50/50 shootout COIN FLIP.
    The coin flip is a documented placeholder: penalty shootouts are close to a
    coin toss and the data stores them as draws with no winner (see methodology), so
    a fitted shootout model is a deliberate later refinement, not part of this phase.

    Efficiency: we group sims by their unique (home, away) pair and batch-sample each
    matchup once, rather than looping over all `n` sims in Python. Grids are cached
    across rounds in `grid_cache` (keyed by the ordered pair) since the venue is
    always neutral, so each distinct matchup builds its grid at most once.
    """
    n = len(home)
    winners = np.empty(n, dtype=object)
    # Stack the two name arrays into an (n, 2) grid and group identical ROWS: each
    # distinct row is one unique matchup. (A string separator is unsafe here — numpy
    # fixed-width strings silently trim embedded/trailing null bytes.)
    pairs = np.stack([home.astype(str), away.astype(str)], axis=1)   # (n, 2)
    uniq_pairs, inv = np.unique(pairs, axis=0, return_inverse=True)

    for u_i in range(len(uniq_pairs)):
        idxs = np.where(inv == u_i)[0]           # the sims with this exact matchup
        h, a = uniq_pairs[u_i]
        h, a = str(h), str(a)
        if (h, a) not in grid_cache:
            grid_cache[(h, a)] = predict_match(strengths, h, a, neutral=True)["grid"]
        hg, ag = sample_scorelines(grid_cache[(h, a)], len(idxs), rng)

        draw = hg == ag
        # Shootout coin flip only matters on drawn ties; True => home side wins it.
        coin = rng.random(len(idxs)) < 0.5
        home_advances = (hg > ag) | (draw & coin)
        winners[idxs] = np.where(home_advances, h, a)
    return winners


def _fill_group_slots(config: dict, strengths: dict, rng: np.random.Generator,
                      n: int) -> dict:
    """Simulate every group and return the bracket's group-derived slots as (n,)
    name arrays.

    Both formats fill the winner/runner-up slots ("1A".."2H" for the World Cup,
    "1A".."2F" for the Euro). The 24-team Euro format ALSO fills the four best-third
    slots (3vB/3vC/3vE/3vF): it ranks the six third-placed teams across groups, takes
    the best four, and routes them to slots via UEFA's fixed anti-rematch table. The
    World Cup path consumes the rng identically to before this helper existed (same
    groups, same order), so its results are unchanged."""
    group_letters = list(config["groups"].keys())
    group_sims: dict[str, dict] = {}
    slot: dict[str, np.ndarray] = {}
    for g in group_letters:
        sim = simulate_group(config["groups"][g], strengths, rng, n)
        group_sims[g] = sim
        slot[f"1{g}"] = sim["winners"]
        slot[f"2{g}"] = sim["runners_up"]

    if config["format"] == "euro24":
        _fill_third_slots(config, group_sims, slot, n)
    return slot


def _fill_third_slots(config: dict, group_sims: dict, slot: dict, n: int) -> None:
    """Fill the four Euro best-third slots (in-place on `slot`).

    Across the six groups, rank the third-placed teams by their (cross-group
    comparable) key, take the best four, and assign each to a third-slot by UEFA's
    table. The table is keyed by the SET of the four qualifying-third groups (there
    are 15 possible sets), and the assignment is rematch-free by construction. This
    is done with ONE mask per combination (≤15), not a Python loop over the n sims."""
    group_letters = list(config["groups"].keys())              # A..F
    letters = np.array(group_letters)
    # (n, 6): each group's 3rd-placed team name, and that team's ranking-key value.
    thirds_names = np.stack([group_sims[g]["thirds"] for g in group_letters], axis=1)
    thirds_keys = np.stack([group_sims[g]["third_key"] for g in group_letters], axis=1)
    # The four best thirds per sim = the four highest keys; the combination is the
    # SET of their group letters (sorted -> the canonical key into the UEFA table).
    top4 = np.argsort(-thirds_keys, axis=1)[:, :4]             # (n, 4) group-col indices
    combo_keys = np.array(["".join(sorted(row)) for row in letters[top4]])   # (n,)

    # The config carries its own UEFA table (editions differ — see src/tournaments):
    #   third_slots  e.g. ["3vB","3vC","3vE","3vF"]
    #   thirds_table {combo -> [group filling each slot, positional to third_slots]}
    third_slots = config["third_slots"]
    table = config["thirds_table"]
    gi = {g: i for i, g in enumerate(group_letters)}
    for s in third_slots:
        slot[s] = np.empty(n, dtype=thirds_names.dtype)
    filled = np.zeros(n, dtype=bool)
    for combo, row in table.items():
        mask = combo_keys == combo
        if not mask.any():
            continue
        filled |= mask
        for s, grp in zip(third_slots, row):
            slot[s][mask] = thirds_names[mask, gi[grp]]
    # Every sim resolves to exactly one of the 15 combos (4 distinct of 6 groups) —
    # fail loudly if any slot went unfilled rather than carry an empty string forward.
    assert filled.all(), f"{int((~filled).sum())} sims matched no thirds combination"


def simulate_tournament(config: dict, strengths: dict, n: int = 10000,
                        seed: int = 0) -> pd.DataFrame:
    """
    Simulate a whole tournament `n` times and return a per-team DataFrame of
    P(reach R16 / QF / SF / final / win the tournament).

    Flow: simulate the 8 groups (Phase C) to fill the bracket's group slots
    (1A, 2B, ...), then play the knockout bracket round by round (Phase D), each tie
    resolved by _play_ties. A team's probability of reaching a round = the fraction
    of the `n` simulations in which it appears among that round's participants; its
    win probability = the fraction in which it wins the final.

    Deterministic under a fixed `seed` (one rng threaded through groups then knockout).
    """
    validate_teams(config, strengths)                 # fail loudly on any unknown team
    rng = np.random.default_rng(seed)

    # --- Group stage: fill the bracket's group-derived slots ----------------------
    # Winner/runner-up slots ("1A".."2H"), plus the four best-third slots for the
    # 24-team Euro format. _fill_group_slots handles both (World Cup path unchanged).
    slot = _fill_group_slots(config, strengths, rng, n)

    # --- Knockout bracket, round by round -----------------------------------------
    match_winner: dict[int, np.ndarray] = {}          # match id -> (n,) winner names
    grid_cache: dict[tuple, np.ndarray] = {}
    # participants[stage] = list of (n,) name arrays that played in that stage.
    participants: dict[str, list[np.ndarray]] = {}

    def resolve(label: str) -> np.ndarray:
        """A bracket slot label -> the (n,) name array filling it: a group slot
        ("1A"/"2B") or a prior match winner ("W53")."""
        if label in slot:
            return slot[label]
        if label.startswith("W"):
            return match_winner[int(label[1:])]
        raise ValueError(f"Unrecognised bracket label {label!r}")

    for stage in ["R16", "QF", "SF", "F"]:
        stage_parts: list[np.ndarray] = []
        for match in config["bracket"][stage]:
            home = resolve(match["home"])
            away = resolve(match["away"])
            stage_parts += [home, away]               # both sides reached this stage
            match_winner[match["id"]] = _play_ties(home, away, strengths, rng, grid_cache)
        participants[stage] = stage_parts

    # The final has exactly one match; its winner is the champion in each sim.
    final_id = config["bracket"]["F"][0]["id"]
    champion = match_winner[final_id]

    # --- Turn simulation counts into per-team probabilities -----------------------
    all_teams = [t for teams in config["groups"].values() for t in teams]
    # A team reaches a stage in a sim if it appears in ANY participant array of it.
    stage_reached = {"R16": participants["R16"], "QF": participants["QF"],
                     "SF": participants["SF"], "final": participants["F"]}
    rows = []
    for t in all_teams:
        row = {"team": t}
        for label, parts in stage_reached.items():
            # OR across the stage's participant arrays: did team t play in this stage?
            reached = np.zeros(n, dtype=bool)
            for arr in parts:
                reached |= (arr == t)
            row[f"p_{label}"] = float(np.mean(reached))
        row["p_win"] = float(np.mean(champion == t))
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("p_win", ascending=False).reset_index(drop=True)

    # --- Assertions the plan requires (fail loudly if the sim is internally wrong) -
    # 1. Monotone by construction: winning implies reaching each earlier round.
    mono = (df["p_win"] <= df["p_final"] + 1e-12) & (df["p_final"] <= df["p_SF"] + 1e-12) \
        & (df["p_SF"] <= df["p_QF"] + 1e-12) & (df["p_QF"] <= df["p_R16"] + 1e-12)
    assert mono.all(), "reach-probabilities are not monotone by round"
    # 2. Exactly one champion per sim => champion probabilities sum to 1.
    assert abs(df["p_win"].sum() - 1.0) < 1e-9, df["p_win"].sum()
    # 3. Each knockout round has a fixed number of slots => reach-probs sum to that
    #    many teams (16 reach R16, 8 QF, 4 SF, 2 final).
    for label, k in [("R16", 16), ("QF", 8), ("SF", 4), ("final", 2)]:
        assert abs(df[f"p_{label}"].sum() - k) < 1e-9, (label, df[f"p_{label}"].sum())
    return df


# ============================================================================
# Phase A verification — run directly:  python -m src.models.montecarlo
# ============================================================================

if __name__ == "__main__":
    # We need one real fitted match to sample from. Rather than pay the ~14s data
    # load for a self-check, we use a representative pair of goal-rates and a real
    # Dixon-Coles rho, then confirm the SAMPLER matches the ANALYTIC model on the
    # same grid. (The full pipeline fit is exercised in later phases.)
    rng = np.random.default_rng(0)

    # Representative international rates: a stronger home side ~1.6 goals, weaker
    # away side ~1.1, and a realistic small negative rho (the value fit_dixon_coles
    # returns on this project's data is ~-0.047).
    lam_h, lam_a, rho = 1.6, 1.1, -0.047

    # --- Check 1: MC on the Dixon-Coles-corrected grid ~= analytic wdl_from_grid ---
    grid = scoreline_matrix(lam_h, lam_a)
    grid = apply_dixon_coles(grid, lam_h, lam_a, rho)
    a_home, a_draw, a_away = wdl_from_grid(grid)          # analytic truth

    n = 1_000_000
    hg, ag = sample_scorelines(grid, n, rng)
    m_home, m_draw, m_away = _wdl_from_samples(hg, ag)    # Monte Carlo estimate

    # Monte-Carlo standard error for a proportion p is sqrt(p(1-p)/n). At n=1e6 and
    # p~0.4 that is ~5e-4, so agreement should hold to ~3 decimal places.
    se = max(np.sqrt(p * (1 - p) / n) for p in (a_home, a_draw, a_away))
    print(f"Dixon-Coles grid (lambda {lam_h}-{lam_a}, rho={rho}):  n={n:,}")
    print(f"  analytic  P(H/D/A) = {a_home:.4f} / {a_draw:.4f} / {a_away:.4f}")
    print(f"  MonteCarlo P(H/D/A) = {m_home:.4f} / {m_draw:.4f} / {m_away:.4f}")
    print(f"  max |MC - analytic| = {max(abs(m_home-a_home), abs(m_draw-a_draw), abs(m_away-a_away)):.5f}"
          f"   (MC std err ~ {se:.5f}; expect agreement within ~3-4 std errs)")
    assert abs(m_home - a_home) < 5 * se
    assert abs(m_draw - a_draw) < 5 * se
    assert abs(m_away - a_away) < 5 * se

    # --- Check 2: independence sanity. With rho=0 the grid is the independent
    #     product, so sampling the grid must match TWO independent rng.poisson draws
    #     (the naive sampler) to within Monte-Carlo error. This confirms the grid
    #     sampler and the textbook independent-Poisson agree exactly when they should.
    grid0 = scoreline_matrix(lam_h, lam_a)               # rho=0, plain product
    hg0, ag0 = sample_scorelines(grid0, n, rng)
    g0_home, g0_draw, g0_away = _wdl_from_samples(hg0, ag0)
    # Naive independent-Poisson draws:
    ihg = rng.poisson(lam_h, size=n)
    iag = rng.poisson(lam_a, size=n)
    i_home, i_draw, i_away = _wdl_from_samples(ihg, iag)
    print(f"\nIndependence check (rho=0):  grid-sampler vs independent rng.poisson")
    print(f"  grid-sample P(H/D/A) = {g0_home:.4f} / {g0_draw:.4f} / {g0_away:.4f}")
    print(f"  indep-pois  P(H/D/A) = {i_home:.4f} / {i_draw:.4f} / {i_away:.4f}")
    print(f"  max |grid - indep|  = {max(abs(g0_home-i_home), abs(g0_draw-i_draw), abs(g0_away-i_away)):.5f}")
    assert abs(g0_draw - i_draw) < 5 * se

    print("\nPhase A OK: the sampler reproduces the analytic model within MC error.")

    # --- Phases C-D end-to-end: a seeded WC 2022 simulation ----------------------
    # This pays the ~14s data load + fit, so it runs after the cheap Phase-A checks.
    from src.data_pipeline import load_raw_matches, clean_matches
    from src.models.poisson import fit_dixon_coles
    from src.tournaments import load_tournament_config

    cfg = load_tournament_config("wc2022")
    t_start = pd.Timestamp(cfg["start_date"])
    print(f"\nFitting Dixon-Coles before {t_start.date()} for a WC2022 simulation "
          f"(~14s)...", flush=True)
    matches = clean_matches(load_raw_matches())
    strengths = fit_dixon_coles(matches[matches["date"] < t_start], reference_date=t_start)

    print("Simulating WC 2022 (n=20,000, seed=0)...", flush=True)
    result = simulate_tournament(cfg, strengths, n=20000, seed=0)
    print("\n--- Top 8 by simulated P(win the tournament) ---")
    print(result.head(8).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    a = cfg["actual"]
    order = list(result["team"])
    print(f"\nActual champion {a['champion']} ranked {order.index(a['champion'])+1}/32 by "
          f"P(win); runner-up {a['runner_up']} ranked {order.index(a['runner_up'])+1}/32.")
    print("Phases C-D OK: monotone reach-probs, champion probs sum to 1 (asserted "
          "inside simulate_tournament).")
