"""
Phase E — honest backtest of the Monte Carlo tournament simulator, across the four
tournaments we can encode: FIFA World Cup 2022 + UEFA Euro 2016 / 2020 / 2024.

For each, fit the Dixon-Coles strengths on matches STRICTLY BEFORE the tournament
(the exact walk_forward "before T" slice, referenced to the tournament start),
simulate the whole tournament thousands of times, and ask: did the model's per-team
round probabilities line up with what actually happened? `run(name)` reports one
tournament; `main()` runs all four and POOLS the round-reach score.

TWO honest caveats, both stated up front rather than papered over:
  1. FOUR tournaments is better than one but still FEW. You cannot strongly validate
     tournament-WINNER *calibration* off four trophies — each champion is one
     Bernoulli draw. What we CAN check is whether the actual champion / finalists /
     semi-finalists landed in the model's top tier, and a round-reach Brier/log-loss
     across every team x 5 rounds (160 predictions per World Cup, 120 per Euro; 520
     pooled — a real, if correlated, sample). Honestly, the model nailed Euro 2024
     (Spain #1) but rated the 2016/2020 underdog champions (Portugal, Italy) low, as
     any pre-tournament model would — it is not tuned to always crown the favourite.
  2. There is NO outright / tournament-winner ODDS market anywhere in this project
     (odds are per-match 1X2 only — confirmed by exploration), so unlike the
     per-match model there is no bookmaker benchmark to compare P(win trophy)
     against. The simulator's rigor rests on the per-match engine (already validated
     with bootstrap CIs in Poisson Phase 4) plus this round-level sanity check.

Run from repo root:  python -m src.models.montecarlo_eval   (all four, pooled)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_pipeline import load_raw_matches, clean_matches
from src.models.poisson import fit_dixon_coles
from src.models.montecarlo import simulate_tournament
from src.tournaments import load_tournament_config, all_config_teams


# The five nested reach targets, from easiest (survive the group) to hardest (win it).
# Column names must match simulate_tournament's output.
STAGES = ["p_R16", "p_QF", "p_SF", "p_final", "p_win"]


def actual_reach(config: dict) -> pd.DataFrame:
    """Build the ground-truth 0/1 reach table from the config's `actual` block: for
    every team, did it actually reach R16 / QF / SF / final / win? Uses only recorded
    tournament outcomes (no model), so it is the honest target the sim is scored on."""
    a = config["actual"]
    # R16 = the twelve group qualifiers PLUS (Euro 24-team format) the four best
    # third-placed teams that advanced. wc32 configs have no "third_qualifiers" key,
    # so this is a no-op there and reached_r16 stays the sixteen top-two finishers.
    reached_r16 = (set(a["group_winners"].values())
                   | set(a["group_runners_up"].values())
                   | set(a.get("third_qualifiers", [])))
    reached_qf = set(a["quarter_finalists"])
    reached_sf = set(a["semi_finalists"])
    reached_final = {a["champion"], a["runner_up"]}
    won = {a["champion"]}

    # Sanity on the hand-encoded actuals (counts must match the bracket shape).
    assert len(reached_r16) == 16 and len(reached_qf) == 8
    assert len(reached_sf) == 4 and len(reached_final) == 2

    rows = []
    for t in all_config_teams(config):
        rows.append({
            "team": t,
            "p_R16": float(t in reached_r16),
            "p_QF": float(t in reached_qf),
            "p_SF": float(t in reached_sf),
            "p_final": float(t in reached_final),
            "p_win": float(t in won),
        })
    return pd.DataFrame(rows)


def score_reach(pred: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """Brier score and log-loss of the model's reach probabilities vs the actual 0/1
    outcomes, pooled over all N teams x 5 stages (N=32 -> 160 predictions for the
    World Cup, N=24 -> 120 for the Euro). Brier = mean((p - y)^2); log-loss =
    mean(-[y log p + (1-y) log(1-p)]) with p clipped off 0/1 so a confident miss is
    heavily but finitely penalised. Lower is better for both; we report against the
    reach BASE RATES (16/N, 8/N, 4/N, 2/N, 1/N — sixteen of N teams reach the R16, and
    so on) as the no-skill reference a useful model must beat."""
    m = pred.merge(truth, on="team", suffixes=("_pred", "_true"))
    p = m[[s + "_pred" for s in STAGES]].to_numpy()
    y = m[[s + "_true" for s in STAGES]].to_numpy()

    brier = float(np.mean((p - y) ** 2))
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    logloss = float(np.mean(-(y * np.log(pc) + (1 - y) * np.log(1 - pc))))

    # No-skill baseline: predict each stage's base rate for every team. The field
    # size N is however many teams are actually scored (32 World Cup / 24 Euro), so
    # the rates are 16/N ... 1/N — for the World Cup this is exactly the previous /32.
    n_teams = len(m)
    base_rates = np.array([16, 8, 4, 2, 1]) / n_teams
    pb = np.broadcast_to(base_rates, p.shape)
    base_brier = float(np.mean((pb - y) ** 2))
    pbc = np.clip(pb, 1e-6, 1 - 1e-6)
    base_logloss = float(np.mean(-(y * np.log(pbc) + (1 - y) * np.log(1 - pbc))))

    return {"brier": brier, "logloss": logloss,
            "base_brier": base_brier, "base_logloss": base_logloss}


def evaluate(name: str = "wc2022", n: int = 20000, seed: int = 0,
             matches: pd.DataFrame | None = None) -> dict:
    """Pure backtest of the Monte Carlo simulator: fit Dixon-Coles strictly BEFORE
    the tournament, simulate it n times, and score the per-team round-reach
    probabilities against what actually happened. Returns everything both the CLI
    report (`run()`) and the webapp's /api/montecarlo need — no printing, so the two
    can never drift (the same discipline as walk_forward.summarize).

    `matches`: optionally pass an already-cleaned match frame (the webapp keeps one
    cached) to skip the ~14s clean; None => load + clean here so the CLI stays
    self-contained.

    Returns:
      pred    : the full simulate_tournament DataFrame (per-team P(reach round)),
                ordered by P(win) desc — the CLI formats head(8) from this, the
                endpoint serialises head(8) to records.
      landing : champion / runner_up / semi_finalists, each with model rank
                (1 = highest P(win)) and the relevant probability — "did the deep
                runs land in the model's top tier?"
      reach   : the score_reach dict (Brier / log-loss + no-skill base-rate refs).
      reach_arrays : raw per-(team,stage) p / y / base arrays, so main() can pool
                across tournaments (ignored by run() and the webapp endpoint).
      meta    : name, year, n, seed, n_train, n_teams, start_date (display + provenance).
    """
    config = load_tournament_config(name)
    t_start = pd.Timestamp(config["start_date"])

    if matches is None:
        matches = clean_matches(load_raw_matches())
    before = matches[matches["date"] < t_start]
    strengths = fit_dixon_coles(before, reference_date=t_start)

    pred = simulate_tournament(config, strengths, n=n, seed=seed)
    truth = actual_reach(config)

    # --- Did the actual deep runs land in the model's top tier? -------------------
    a = config["actual"]
    order = list(pred["team"])
    rank = {t: i + 1 for i, t in enumerate(order)}     # 1 = highest P(win)

    def _prob(team: str, col: str) -> float:
        return float(pred.loc[pred.team == team, col].iloc[0])

    landing = {
        "champion":  {"team": a["champion"],  "rank": rank[a["champion"]],
                      "p_win": _prob(a["champion"], "p_win")},
        "runner_up": {"team": a["runner_up"], "rank": rank[a["runner_up"]],
                      "p_final": _prob(a["runner_up"], "p_final")},
        "semi_finalists": [{"team": t, "rank": rank[t]} for t in a["semi_finalists"]],
    }
    reach = score_reach(pred, truth)
    # Raw per-(team, stage) arrays too, so a multi-tournament driver can POOL across
    # tournaments (each keeping its own base rate) without re-fitting. Ignored by the
    # single-tournament run() and the webapp endpoint.
    mrg = pred.merge(truth, on="team", suffixes=("_pred", "_true"))
    reach_arrays = {
        "p": mrg[[s + "_pred" for s in STAGES]].to_numpy(),
        "y": mrg[[s + "_true" for s in STAGES]].to_numpy(),
        "base": np.broadcast_to(np.array([16, 8, 4, 2, 1]) / len(mrg), (len(mrg), 5)).copy(),
    }
    meta = {
        "name": config["name"], "year": config["year"],
        "n": int(n), "seed": int(seed), "n_train": int(len(before)),
        "n_teams": int(len(pred)), "start_date": str(t_start.date()),
    }
    return {"pred": pred, "landing": landing, "reach": reach,
            "reach_arrays": reach_arrays, "meta": meta}


def run(name: str = "wc2022", n: int = 20000, seed: int = 0) -> None:
    """Print the Monte Carlo backtest report. Every number comes from `evaluate()`
    so this CLI report and the webapp's /api/montecarlo are identical by
    construction (verified byte-identical, the project's standard anti-drift check)."""
    d = evaluate(name, n=n, seed=seed)
    m, land, sc = d["meta"], d["landing"], d["reach"]

    print(f"=== Monte Carlo backtest: {m['name']} {m['year']} "
          f"(n={m['n']:,}, seed={m['seed']}) ===\n")
    print(f"Fitting Dixon-Coles on matches BEFORE {m['start_date']} "
          f"(true pre-tournament forecast)...", flush=True)
    print(f"  training matches: {m['n_train']:,}")
    print(f"\nSimulating {m['year']} {m['n']:,} times...", flush=True)

    print("\n--- Top 8 by simulated P(win) ---")
    print(d["pred"].head(8).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    c, r = land["champion"], land["runner_up"]
    print(f"\nActual champion : {c['team']:<12} model P(win) rank {c['rank']}/{m['n_teams']}, "
          f"P(win)={c['p_win']:.3f}")
    print(f"Actual runner-up: {r['team']:<12} model P(win) rank {r['rank']}/{m['n_teams']}, "
          f"P(final)={r['p_final']:.3f}")
    sf_ranks = {sf["team"]: sf["rank"] for sf in land["semi_finalists"]}
    print(f"Actual semi-finalists and their P(SF) ranks: {sf_ranks}")

    # --- Round-reach Brier / log-loss vs actual (all teams) -----------------------
    print(f"\n--- Round-reach scoring ({m['n_teams']} teams x 5 stages = {m['n_teams'] * 5} predictions) ---")
    print(f"  Brier   : model {sc['brier']:.4f}   vs base-rate {sc['base_brier']:.4f}   "
          f"({'beats' if sc['brier'] < sc['base_brier'] else 'WORSE than'} no-skill)")
    print(f"  Log-loss: model {sc['logloss']:.4f}   vs base-rate {sc['base_logloss']:.4f}   "
          f"({'beats' if sc['logloss'] < sc['base_logloss'] else 'WORSE than'} no-skill)")

    print("\n--- Honest limitations ---")
    print("  * ONE tournament: champion is a single Bernoulli draw; winner CALIBRATION")
    print(f"    cannot be strongly validated. Round-reach Brier pools {m['n_teams'] * 5} (correlated)")
    print("    predictions — a sanity check, not a calibration proof.")
    print("  * NO outright-winner odds market exists in this project, so P(win trophy)")
    print("    has no bookmaker benchmark. Per-match engine is CI-validated (Phase 4).")


def main(names=("wc2022", "euro2016", "euro2020", "euro2024"),
         n: int = 20000, seed: int = 0) -> None:
    """Backtest the simulator across EVERY configured tournament and POOL the result.

    The single-tournament run() answers "does the sim work on this event"; this
    answers the project's real question — "does it work across the tournaments we
    have" — with the honest caveat that four is still few. Fits each tournament
    strictly pre-kickoff (reusing one cleaned-match load), scores round-reach against
    that format's own no-skill base rate, and pools every team-round prediction into a
    single Brier / log-loss so no one event dominates the headline."""
    matches = clean_matches(load_raw_matches())
    print(f"{'=' * 78}\nMONTE CARLO TOURNAMENT BACKTEST — {len(names)} tournaments "
          f"(n={n:,}, seed={seed})\n{'=' * 78}")
    print("each: fit Dixon-Coles strictly BEFORE kickoff, simulate, score per-team "
          "round-reach\nvs that format's no-skill base rate (16/N .. 1/N reach each "
          "round; N = field size).\n")
    print(f"{'tournament':<20}{'teams':>6}{'train':>9}{'champion (rank)':>20}"
          f"{'Brier':>9}{'base':>8}{'logloss':>9}{'base':>8}")

    P, Y, B = [], [], []
    for name in names:
        d = evaluate(name, n=n, seed=seed, matches=matches)
        m, land, sc = d["meta"], d["landing"], d["reach"]
        champ = f"{land['champion']['team']} ({land['champion']['rank']}/{m['n_teams']})"
        print(f"{m['name'] + ' ' + str(m['year']):<20}{m['n_teams']:>6}{m['n_train']:>9,}"
              f"{champ:>20}{sc['brier']:>9.4f}{sc['base_brier']:>8.4f}"
              f"{sc['logloss']:>9.4f}{sc['base_logloss']:>8.4f}")
        a = d["reach_arrays"]
        P.append(a["p"]); Y.append(a["y"]); B.append(a["base"])

    p, y, b = np.vstack(P), np.vstack(Y), np.vstack(B)

    def brier(pp):
        return float(np.mean((pp - y) ** 2))

    def logloss(pp):
        pc = np.clip(pp, 1e-6, 1 - 1e-6)
        return float(np.mean(-(y * np.log(pc) + (1 - y) * np.log(1 - pc))))

    print(f"\n{'-' * 78}\nPOOLED  ({len(y) * 5} team-round predictions: {len(y)} teams "
          f"x 5 rounds across {len(names)} tournaments)\n{'-' * 78}")
    print(f"  Brier   : model {brier(p):.4f}   vs base-rate {brier(b):.4f}   "
          f"({'beats' if brier(p) < brier(b) else 'WORSE than'} no-skill)")
    print(f"  Log-loss: model {logloss(p):.4f}   vs base-rate {logloss(b):.4f}   "
          f"({'beats' if logloss(p) < logloss(b) else 'WORSE than'} no-skill)")

    print("\n--- Honest limitations ---")
    print(f"  * {len(names)} tournaments is still FEW: a champion is one Bernoulli draw, so")
    print("    tournament-WINNER calibration can't be strongly validated even pooled. The")
    print("    round-reach score is a real (if correlated) sanity check, not a proof.")
    print("  * NO outright-winner odds market exists in this project, so P(win trophy) has")
    print("    no bookmaker benchmark; the per-match engine underneath is CI-validated.")
    print("  * Euro games are simulated neutral (documented), and a drawn knockout tie")
    print("    uses a ~50/50 shootout coin-flip placeholder, not a fitted model.")


if __name__ == "__main__":
    # Run from repo root:  python -m src.models.montecarlo_eval   (all tournaments)
    #   single event:  python -c "from src.models.montecarlo_eval import run; run('euro2024')"
    main()
