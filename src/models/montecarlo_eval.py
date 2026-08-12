"""
Phase E — honest backtest of the Monte Carlo tournament simulator (WC 2022).

Fit the Dixon-Coles strengths on matches STRICTLY BEFORE the tournament (the exact
walk_forward "before T" slice, referenced to the tournament start), simulate the
whole tournament thousands of times, and ask: did the model's per-team round
probabilities line up with what actually happened?

TWO honest caveats, both stated up front rather than papered over:
  1. This is ONE tournament (at most ~4 even with Euro 2016/2020/2024 added). You
     cannot strongly validate tournament-WINNER *calibration* off a single trophy —
     the champion is one Bernoulli draw. What we CAN check is whether the actual
     champion/finalists/semi-finalists landed in the model's top tier, and a
     round-reach Brier/log-loss across all 32 teams (160 team-round predictions,
     which is a real, if correlated, sample).
  2. There is NO outright / tournament-winner ODDS market anywhere in this project
     (odds are per-match 1X2 only — confirmed by exploration), so unlike the
     per-match model there is no bookmaker benchmark to compare P(win trophy)
     against. The simulator's rigor rests on the per-match engine (already validated
     with bootstrap CIs in Poisson Phase 4) plus this round-level sanity check.

Run from repo root:  python -m src.models.montecarlo_eval
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
    reached_r16 = set(a["group_winners"].values()) | set(a["group_runners_up"].values())
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
    outcomes, pooled over all 32 teams x 5 stages = 160 predictions. Brier =
    mean((p - y)^2); log-loss = mean(-[y log p + (1-y) log(1-p)]) with p clipped off
    0/1 so a confident miss is heavily but finitely penalised. Lower is better for
    both; we report against the reach BASE RATES (16/32, 8/32, 4/32, 2/32, 1/32) as
    the no-skill reference a useful model must beat."""
    m = pred.merge(truth, on="team", suffixes=("_pred", "_true"))
    p = m[[s + "_pred" for s in STAGES]].to_numpy()
    y = m[[s + "_true" for s in STAGES]].to_numpy()

    brier = float(np.mean((p - y) ** 2))
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    logloss = float(np.mean(-(y * np.log(pc) + (1 - y) * np.log(1 - pc))))

    # No-skill baseline: predict each stage's base rate for every team.
    base_rates = np.array([16, 8, 4, 2, 1]) / 32.0
    pb = np.broadcast_to(base_rates, p.shape)
    base_brier = float(np.mean((pb - y) ** 2))
    pbc = np.clip(pb, 1e-6, 1 - 1e-6)
    base_logloss = float(np.mean(-(y * np.log(pbc) + (1 - y) * np.log(1 - pbc))))

    return {"brier": brier, "logloss": logloss,
            "base_brier": base_brier, "base_logloss": base_logloss}


def run(name: str = "wc2022", n: int = 20000, seed: int = 0) -> None:
    config = load_tournament_config(name)
    t_start = pd.Timestamp(config["start_date"])

    print(f"=== Monte Carlo backtest: {config['name']} {config['year']} "
          f"(n={n:,}, seed={seed}) ===\n")
    print(f"Fitting Dixon-Coles on matches BEFORE {t_start.date()} "
          f"(true pre-tournament forecast)...", flush=True)
    matches = clean_matches(load_raw_matches())
    before = matches[matches["date"] < t_start]
    print(f"  training matches: {len(before):,}")
    strengths = fit_dixon_coles(before, reference_date=t_start)

    print(f"\nSimulating {config['year']} {n:,} times...", flush=True)
    pred = simulate_tournament(config, strengths, n=n, seed=seed)
    truth = actual_reach(config)

    # --- Did the actual deep runs land in the model's top tier? -------------------
    a = config["actual"]
    order = list(pred["team"])
    rank = {t: i + 1 for i, t in enumerate(order)}     # 1 = highest P(win)
    print("\n--- Top 8 by simulated P(win) ---")
    print(pred.head(8).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nActual champion : {a['champion']:<12} model P(win) rank {rank[a['champion']]}/32, "
          f"P(win)={pred.loc[pred.team==a['champion'],'p_win'].iloc[0]:.3f}")
    print(f"Actual runner-up: {a['runner_up']:<12} model P(win) rank {rank[a['runner_up']]}/32, "
          f"P(final)={pred.loc[pred.team==a['runner_up'],'p_final'].iloc[0]:.3f}")
    sf_ranks = {t: rank[t] for t in a["semi_finalists"]}
    print(f"Actual semi-finalists and their P(SF) ranks: {sf_ranks}")

    # --- Round-reach Brier / log-loss vs actual (all 32 teams) --------------------
    sc = score_reach(pred, truth)
    print("\n--- Round-reach scoring (32 teams x 5 stages = 160 predictions) ---")
    print(f"  Brier   : model {sc['brier']:.4f}   vs base-rate {sc['base_brier']:.4f}   "
          f"({'beats' if sc['brier'] < sc['base_brier'] else 'WORSE than'} no-skill)")
    print(f"  Log-loss: model {sc['logloss']:.4f}   vs base-rate {sc['base_logloss']:.4f}   "
          f"({'beats' if sc['logloss'] < sc['base_logloss'] else 'WORSE than'} no-skill)")

    print("\n--- Honest limitations ---")
    print("  * ONE tournament: champion is a single Bernoulli draw; winner CALIBRATION")
    print("    cannot be strongly validated. Round-reach Brier pools 160 (correlated)")
    print("    predictions — a sanity check, not a calibration proof.")
    print("  * NO outright-winner odds market exists in this project, so P(win trophy)")
    print("    has no bookmaker benchmark. Per-match engine is CI-validated (Phase 4).")


if __name__ == "__main__":
    run()
