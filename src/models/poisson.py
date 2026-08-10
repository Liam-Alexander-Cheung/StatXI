"""
Poisson scoreline model — the classical-statistics half of StatXI.

The XGBoost model (src/models/train_wdl.py) predicts Win/Draw/Loss directly. This
module instead predicts the *full distribution over scorelines* (0-0, 1-0, 2-1,
...) and lets Win/Draw/Loss fall out of that distribution. Goals are count data,
and the textbook distribution for "how many independent rare-ish events happen in
a fixed window" is the Poisson distribution — so that is what we use.

BUILT INCREMENTALLY (see reports/methodology.md). Contents so far:
  PHASE 1 — the *prediction* machinery: two goal-rates -> scoreline grid -> H/D/A.
  PHASE 2 — *fitting* those rates: per-team attack/defence strengths by maximum
            likelihood over the whole match history (plain independent Poisson).
  PHASE 3 — two refinements that make it a proper Dixon-Coles model:
            (a) TIME-DECAY WEIGHTING — recent matches count for more (reusing the
                project's recency_weight x importance_weight), so strengths track
                current form, not 1990s history.
            (b) the DIXON-COLES low-score correction — one parameter rho that
                nudges the four low-score cells (0-0, 1-0, 0-1, 1-1), because
                independent Poisson is known to under-predict tight draws.

THE ONE FORMULA. For a Poisson rate lambda, the probability of seeing exactly k
events is:

    P(k) = exp(-lambda) * lambda**k / k!

We don't hand-code that — scipy.stats.poisson.pmf is exactly it, and scipy is
already a project dependency (scipy 1.13).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize, minimize_scalar

from src.data_pipeline import (
    load_raw_matches,
    clean_matches,
    recency_weight,
    importance_weight,
)


# ============================================================================
# PHASE 1 — prediction: goal-rates -> scoreline grid -> Win/Draw/Loss
# ============================================================================

def scoreline_matrix(lambda_home: float, lambda_away: float,
                     max_goals: int = 10) -> np.ndarray:
    """
    The full grid of exact-scoreline probabilities for one match.

    We assume — for now — that the two teams' goal counts are INDEPENDENT. Under
    independence, the probability of an exact scoreline (home scores i, away
    scores j) is just the product of the two separate Poisson probabilities:

        P(home=i, away=j) = P_poisson(i; lambda_home) * P_poisson(j; lambda_away)

    The Dixon-Coles correction (apply_dixon_coles, below) relaxes this
    independence for the four low scores where it is known to be slightly wrong.

    Returns a (max_goals+1) x (max_goals+1) numpy array `grid` where
    `grid[i, j]` = P(home scores i AND away scores j). Row index = home goals,
    column index = away goals.

    `max_goals=10` truncates the (in principle infinite) grid at 10-10. For
    international football rates (lambda ~ 0.5-3) the mass beyond 10 goals is
    vanishingly small (~1e-6), so this is a safe, cheap approximation; the tiny
    lost mass is handled honestly in wdl_from_grid by normalising.
    """
    # goals = [0, 1, 2, ..., max_goals] — the goal counts we evaluate the PMF at.
    goals = np.arange(max_goals + 1)

    # poisson.pmf(goals, lambda) is VECTORISED: it returns an array
    # [P(0), P(1), ..., P(max_goals)] in one call, no Python loop needed.
    home_probs = poisson.pmf(goals, lambda_home)   # shape (max_goals+1,)
    away_probs = poisson.pmf(goals, lambda_away)    # shape (max_goals+1,)

    # The independence assumption = an OUTER PRODUCT. np.outer(a, b)[i, j] = a[i]*b[j],
    # which is precisely P(home=i) * P(away=j) for every (i, j) cell at once.
    grid = np.outer(home_probs, away_probs)
    return grid


def wdl_from_grid(grid: np.ndarray) -> tuple[float, float, float]:
    """
    Collapse a scoreline grid into (P(home win), P(draw), P(away win)).

    In `grid[i, j]` (i = home goals, j = away goals):
      - home WINS  when i > j  -> the cells BELOW the diagonal (lower triangle)
      - DRAW       when i == j -> the cells ON the diagonal
      - away WINS  when i < j  -> the cells ABOVE the diagonal (upper triangle)

    np.tril/np.triu/np.diag pick those cells out; summing each region gives the
    three outcome probabilities.

    Because the grid is truncated at max_goals (and, after Dixon-Coles, slightly
    re-weighted), it does not sum to exactly 1. We divide by the grid's actual
    total so the three returned numbers sum to EXACTLY 1 — an honest
    renormalisation of the retained mass, not a fabricated value.
    """
    total = grid.sum()  # ~0.999999 for football rates; the retained mass

    # np.tril(grid, k=-1): keep only cells strictly below the diagonal (i > j).
    # k=-1 excludes the diagonal itself. That is exactly the home-win region.
    p_home = np.tril(grid, k=-1).sum() / total
    # np.diag(grid): the i == j cells -> draws.
    p_draw = np.diag(grid).sum() / total
    # np.triu(grid, k=1): strictly above the diagonal (i < j) -> away wins.
    p_away = np.triu(grid, k=1).sum() / total

    return float(p_home), float(p_draw), float(p_away)


# ============================================================================
# PHASE 3a — the Dixon-Coles low-score correction
# ============================================================================
#
# Independent Poisson gets the low scores slightly wrong: real football has MORE
# 0-0 and 1-1 draws, and FEWER 1-0 / 0-1 results, than independence predicts.
# Dixon-Coles (1997) fix this with a single parameter rho that multiplies just the
# four affected cells by a factor tau:
#
#     tau(0,0) = 1 - lambda_home * lambda_away * rho
#     tau(0,1) = 1 + lambda_home * rho
#     tau(1,0) = 1 + lambda_away * rho
#     tau(1,1) = 1 - rho
#     tau(x,y) = 1   for every other scoreline
#
# With rho < 0 this RAISES the 0-0 and 1-1 cells and LOWERS the 1-0/0-1 cells —
# exactly the empirical correction — so we expect the fitted rho to be negative.


def dixon_coles_tau(home_goals, away_goals, lambda_home, lambda_away, rho):
    """Vectorised Dixon-Coles tau factor for arrays of (home_goals, away_goals)
    and their matching lambda_home / lambda_away, at a given rho. Returns 1.0 for
    every scoreline outside the four low-score cells. Used inside the rho fit."""
    hg = np.asarray(home_goals)
    ag = np.asarray(away_goals)
    tau = np.ones(np.broadcast(hg, ag, lambda_home, lambda_away).shape, dtype=float)

    m00 = (hg == 0) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0)
    m11 = (hg == 1) & (ag == 1)

    lh = np.asarray(lambda_home)
    la = np.asarray(lambda_away)
    tau[m00] = 1.0 - lh[m00] * la[m00] * rho
    tau[m01] = 1.0 + lh[m01] * rho
    tau[m10] = 1.0 + la[m10] * rho
    tau[m11] = 1.0 - rho
    return tau


def apply_dixon_coles(grid: np.ndarray, lambda_home: float, lambda_away: float,
                      rho: float) -> np.ndarray:
    """Return a copy of a scoreline `grid` with the four low-score cells scaled by
    the Dixon-Coles tau factors. Leaves every other cell untouched. rho=0 is the
    identity (falls back to plain independent Poisson)."""
    if not rho:
        return grid
    g = grid.copy()
    g[0, 0] *= 1.0 - lambda_home * lambda_away * rho
    g[0, 1] *= 1.0 + lambda_home * rho
    g[1, 0] *= 1.0 + lambda_away * rho
    g[1, 1] *= 1.0 - rho
    return g


# ============================================================================
# PHASE 2 — fitting: per-team attack/defence strengths by maximum likelihood
# ============================================================================
#
# THE MODEL. Each team i gets two numbers: attack[i] (how many it tends to
# score) and defence[i] (how well it prevents goals — HIGHER = better defence
# here). Two shared numbers set the backdrop: `base` (the log of the average
# goals a team scores in a neutral, average matchup) and `home_adv` (the extra
# log-rate a team enjoys when playing at home). The expected goals for a match:
#
#     log(lambda_home) = base + home_adv*(1 - neutral) + attack[home] - defence[away]
#     log(lambda_away) = base +                          attack[away] - defence[home]
#
# We work in LOG space (then exp back) so the rates can never go negative and so
# the parameters add up linearly — which makes the gradient below clean.
#
# MAXIMUM LIKELIHOOD. "Likelihood" = the probability the model assigns to the
# scores that ACTUALLY happened. We search for the parameters that make the real
# history most probable. Equivalently (and more stably) we MINIMISE the negative
# log-likelihood (NLL). For a Poisson with rate lambda and observed goals y, the
# log-probability is  -lambda + y*log(lambda) - log(y!). The log(y!) term does not
# depend on any parameter, so it is a constant we drop — it shifts the NLL by a
# fixed amount but never moves the minimum.
#
# WHY AN ANALYTIC GRADIENT. There are ~2 numbers per team (a couple hundred teams
# -> ~600 parameters). Letting the optimiser estimate the gradient numerically
# would cost ~600 full passes over 32k matches per step — far too slow. Instead we
# hand the optimiser the exact gradient. The derivative of one term
# (lambda - y*log lambda) w.r.t. any parameter p is  (lambda - y) * d(log lambda)/dp,
# and because log(lambda) is linear in the parameters, d(log lambda)/dp is just
# +/-1 for the handful of parameters that appear in that match. So the gradient is
# the "residual" (lambda - y) scattered back onto each team's attack/defence slot —
# done in bulk with np.bincount.


def fit_strengths(matches, weights=None, reg: float = 1e-3, covariate=None,
                  verbose: bool = False) -> dict:
    """
    Fit per-team attack/defence strengths + global base & home_adv by maximum
    likelihood over `matches` (plain independent Poisson — the Dixon-Coles rho is
    fitted separately in fit_dixon_coles).

    `matches` must have columns: home_team, away_team, home_score, away_score,
    neutral. Both `clean_matches(load_raw_matches())` and the harness's fold
    frames satisfy this, so the same fitter serves the standalone run and the
    Phase-4 backtest adapter.

    `weights`: optional per-match weight (one per row of `matches`). None = all 1
    (the Phase-2 unweighted fit). fit_dixon_coles passes recency x importance here
    so recent, important matches shape the strengths more (Phase 3a).

    `covariate`: optional per-match number (Phase 5 = rating_gap) added to the home
    log-rate and subtracted from the away log-rate, with a single fitted coefficient
    gamma: log λ_home += gamma*cov, log λ_away −= gamma*cov. NaN entries become 0
    (no adjustment), so matches without a rating just get the plain prediction.
    None = no covariate (the vector has no gamma term, behaviour is byte-identical
    to the pre-Phase-5 fit).

    `reg` is a small L2 (ridge) penalty on the attack/defence values. It does two
    honest jobs at once: (1) it resolves a harmless redundancy in the model — you
    can add a constant to every attack and every defence without changing any
    predicted rate, so without a pin the raw numbers are only defined up to that
    shift; the penalty selects the smallest-norm solution. (2) It mildly shrinks
    the strengths of teams with little data toward average. It is NOT a tuned knob;
    1e-3 is deliberately tiny (predictions are near-identical for reg in [0, 1e-2]).

    Returns a dict:
      {"attack": {team: float}, "defence": {team: float},
       "home_adv": float, "base": float, "rho": 0.0, "rating_coef": float,
       "teams": [team, ...], "match_counts": {team: int},
       "n_matches": int, "converged": bool}
    where attack/defence are CENTRED (mean 0) so they read as deviations from an
    average international side. `rho` is 0.0 here (set by fit_dixon_coles);
    `rating_coef` is the fitted gamma (0.0 when no covariate was passed).
    """
    # --- Encode teams as integer indices (arrays are far faster than dict lookups
    #     inside the objective, which runs hundreds of times). ------------------
    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    T = len(teams)

    home_idx = matches["home_team"].map(idx).to_numpy()
    away_idx = matches["away_team"].map(idx).to_numpy()
    hg = matches["home_score"].to_numpy(dtype=float)   # home goals actually scored
    ag = matches["away_score"].to_numpy(dtype=float)   # away goals actually scored
    # home_adv multiplier: 1 on a real home match, 0 at a neutral venue.
    nz = (1 - matches["neutral"].to_numpy()).astype(float)

    # Per-match weights (recency x importance, or all-ones if unweighted).
    w = np.ones(len(matches)) if weights is None else np.asarray(weights, dtype=float)

    # Optional per-match covariate (Phase 5: rating_gap). NaN -> 0 = no adjustment.
    has_cov = covariate is not None
    cov = (np.nan_to_num(np.asarray(covariate, dtype=float), nan=0.0)
           if has_cov else None)

    # Per-team match counts (home + away appearances) — for honest display later
    # (rank only teams with enough games) and returned for the caller.
    counts = np.bincount(home_idx, minlength=T) + np.bincount(away_idx, minlength=T)

    # --- Parameter vector: [base, home_adv, attack[0..T-1], defence[0..T-1], (gamma?)]
    def unpack(theta):
        base, home_adv = theta[0], theta[1]
        A = theta[2:2 + T]
        D = theta[2 + T:2 + 2 * T]
        gamma = theta[2 + 2 * T] if has_cov else 0.0   # covariate coef, if any
        return base, home_adv, A, D, gamma

    def nll_and_grad(theta):
        """Weighted negative log-likelihood (dropping the constant log(y!)) AND
        its exact gradient, both vectorised over all matches. Returned together
        because the optimiser (jac=True) wants both and they share the work."""
        base, home_adv, A, D, gamma = unpack(theta)
        extra = gamma * cov if has_cov else 0.0        # +on home rate, −on away rate

        # Log-rates, then rates, for every match at once.
        loglam_h = base + home_adv * nz + A[home_idx] - D[away_idx] + extra
        loglam_a = base + A[away_idx] - D[home_idx] - extra
        lam_h = np.exp(loglam_h)
        lam_a = np.exp(loglam_a)

        # NLL = weighted sum over both goal terms of (lambda - y*log lambda), + ridge.
        nll = (np.sum(w * (lam_h - hg * loglam_h))
               + np.sum(w * (lam_a - ag * loglam_a))
               + 0.5 * reg * (A @ A + D @ D))

        # Weighted residuals w*(lambda - y). The gradient w.r.t. each parameter is
        # this scattered onto the teams/params that appear in each match.
        r_h = w * (lam_h - hg)
        r_a = w * (lam_a - ag)

        g_base = r_h.sum() + r_a.sum()                 # base appears in every term
        g_home_adv = np.sum(r_h * nz)                  # home_adv only in home term, only when nz=1
        # attack[t]: +1 in the home term when t is home, +1 in the away term when t is away.
        g_A = (np.bincount(home_idx, weights=r_h, minlength=T)
               + np.bincount(away_idx, weights=r_a, minlength=T)
               + reg * A)
        # defence[t]: -1 in the home term when t is the AWAY side, -1 in the away
        # term when t is the HOME side (defence enters with a minus sign).
        g_D = (-np.bincount(away_idx, weights=r_h, minlength=T)
               - np.bincount(home_idx, weights=r_a, minlength=T)
               + reg * D)

        parts = [[g_base], [g_home_adv], g_A, g_D]
        if has_cov:
            # covariate enters +extra on home, −extra on away:
            #   d(nll)/d(gamma) = sum cov*(r_h − r_a)
            parts.append([np.sum(cov * (r_h - r_a))])
        return nll, np.concatenate(parts)

    # --- Sensible starting point: base = log(overall mean goals per side),
    #     home_adv a small positive number, all strengths (and gamma) 0. ---------
    mean_goals = (hg.sum() + ag.sum()) / (2 * len(matches))
    theta0 = np.concatenate([[np.log(mean_goals)], [0.2], np.zeros(2 * T)]
                            + ([[0.0]] if has_cov else []))

    # L-BFGS-B: a quasi-Newton optimiser that handles hundreds of parameters
    # efficiently using our analytic gradient (jac=True). No bounds needed —
    # the ridge keeps the problem well-posed.
    res = minimize(nll_and_grad, theta0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 1000})
    if verbose:
        tail = f"  rating_coef={res.x[-1]:+.4f}" if has_cov else ""
        print(f"  strengths: success={res.success}  nll={res.fun:.1f}  "
              f"iters={res.nit}  params={len(theta0)}  teams={T}{tail}")

    base, home_adv, A, D, gamma = unpack(res.x)

    # --- Centre attack & defence to mean 0 so they read as deviations, folding
    #     the shift into `base` so every predicted rate is unchanged (see the
    #     redundancy noted in the docstring). base' = base + mean(A) - mean(D). ---
    a_mean, d_mean = A.mean(), D.mean()
    A_c = A - a_mean
    D_c = D - d_mean
    base_c = base + a_mean - d_mean

    return {
        "attack": {t: float(A_c[i]) for t, i in idx.items()},
        "defence": {t: float(D_c[i]) for t, i in idx.items()},
        "home_adv": float(home_adv),
        "base": float(base_c),
        "rho": 0.0,  # set by fit_dixon_coles; 0 = plain independent Poisson
        "rating_coef": float(gamma),  # fitted covariate coef; 0.0 if none (Phase 5)
        "teams": teams,
        "match_counts": {t: int(counts[i]) for t, i in idx.items()},
        "n_matches": int(len(matches)),
        "converged": bool(res.success),
    }


# ============================================================================
# PHASE 3 — weighting + rho: the full Dixon-Coles fit
# ============================================================================

def compute_weights(matches, reference_date=None, half_life_days: int = 3650):
    """
    Per-match training weight = recency x importance — the SAME recipe the WDL
    model uses (src/models/train_wdl.py), so the two halves of the project weight
    history identically. Reproduces recency_weight's formula vectorised (0.5 **
    (days_ago / half_life), floored at 0.05) and multiplies by importance_weight.

    `reference_date`: the "now" that recency is measured back from. Defaults to the
    latest match date (fit current strengths). The Phase-4 backtest passes each
    fold's cutoff so a fold never up-weights toward the future it must predict.
    `half_life_days=3650` (10 years) matches the project's existing default;
    international football is sparse, so a long half-life keeps enough data alive.
    """
    if reference_date is None:
        reference_date = matches["date"].max()
    days_ago = (reference_date - matches["date"]).dt.days.to_numpy()
    # Vectorised copy of recency_weight(..., min_weight=0.05).
    recency = np.maximum(0.5 ** (days_ago / half_life_days), 0.05)
    importance = matches["tournament"].map(importance_weight).to_numpy(dtype=float)
    return recency * importance


def _match_lambdas(matches, strengths):
    """Vectorised (lambda_home, lambda_away) for every row of `matches` under the
    fitted `strengths` — the same two model equations as predict_match, applied in
    bulk so fit_rho can score all matches at once. Assumes every team in `matches`
    is present in `strengths` (true when both come from the same fit)."""
    base = strengths["base"]
    home_adv = strengths["home_adv"]
    a_home = matches["home_team"].map(strengths["attack"]).to_numpy(dtype=float)
    a_away = matches["away_team"].map(strengths["attack"]).to_numpy(dtype=float)
    d_home = matches["home_team"].map(strengths["defence"]).to_numpy(dtype=float)
    d_away = matches["away_team"].map(strengths["defence"]).to_numpy(dtype=float)
    nz = (1 - matches["neutral"].to_numpy()).astype(float)
    lam_h = np.exp(base + home_adv * nz + a_home - d_away)
    lam_a = np.exp(base + a_away - d_home)
    # Apply the Phase-5 rating covariate if this fit used one (else gamma=0 -> skip),
    # so the rho fit sees the same lambdas the model actually predicts with.
    gamma = strengths.get("rating_coef", 0.0)
    if gamma and "rating_gap" in matches.columns:
        cov = np.nan_to_num(matches["rating_gap"].to_numpy(dtype=float), nan=0.0)
        lam_h = lam_h * np.exp(gamma * cov)
        lam_a = lam_a * np.exp(-gamma * cov)
    return lam_h, lam_a


def fit_rho(matches, strengths, weights=None, bound: float = 0.2) -> float:
    """
    Fit the Dixon-Coles rho by 1-D maximum likelihood, holding the attack/defence
    strengths fixed (a profile estimate). rho only reweights four low-score cells,
    so it is near-orthogonal to the strengths — fitting it separately gives
    essentially the joint-MLE answer with far less machinery and no risk of a
    subtle gradient bug in the 600-parameter fit.

    Only the tau factor depends on rho, so the objective is just the weighted sum
    of -log(tau) over the matches whose actual score lands in a low-score cell;
    every other match contributes a constant and is ignored. rho is bounded to
    [-bound, bound] to keep every tau strictly positive (a negative tau would be a
    negative probability). Returns the fitted rho (expected to be small, < 0).
    """
    lam_h, lam_a = _match_lambdas(matches, strengths)
    hg = matches["home_score"].to_numpy(dtype=float)
    ag = matches["away_score"].to_numpy(dtype=float)
    w = np.ones(len(matches)) if weights is None else np.asarray(weights, dtype=float)

    def neg_loglik(rho):
        tau = dixon_coles_tau(hg, ag, lam_h, lam_a, rho)
        if np.any(tau <= 0):          # negative probability — infeasible rho
            return 1e12
        return -np.sum(w * np.log(tau))

    res = minimize_scalar(neg_loglik, bounds=(-bound, bound), method="bounded")
    return float(res.x)


def fit_dixon_coles(matches, reference_date=None, half_life_days: int = 3650,
                    reg: float = 1e-3, verbose: bool = False) -> dict:
    """
    The full Phase-3 fit: time-weighted attack/defence strengths PLUS the
    Dixon-Coles rho. This is the model predict_match should be given.

    Two stages (see fit_rho for why splitting them is legitimate):
      1. fit_strengths on recency x importance weights (Phase 3a weighting).
      2. fit_rho on the resulting lambdas (Phase 3b low-score correction).
    Returns the fit_strengths dict with `rho` filled in.
    """
    weights = compute_weights(matches, reference_date, half_life_days)
    strengths = fit_strengths(matches, weights=weights, reg=reg, verbose=verbose)
    strengths["rho"] = fit_rho(matches, strengths, weights=weights)
    if verbose:
        print(f"  rho: {strengths['rho']:+.4f}  (negative => more 0-0/1-1 draws)")
    return strengths


def predict_match(strengths: dict, home: str, away: str,
                  neutral: bool = False, max_goals: int = 10) -> dict | None:
    """
    Predict one match from fitted `strengths`. Returns the two goal-rates, the
    (Dixon-Coles-corrected) scoreline grid, the collapsed W/D/L probabilities, and
    the single most-likely scoreline.

    Applies the Dixon-Coles correction whenever `strengths["rho"]` is non-zero
    (i.e. the model came from fit_dixon_coles); with rho=0 it is plain Poisson.

    Returns None (never a fabricated guess) if either team was not in the fitting
    data — honouring the project's never-fabricate rule.
    """
    if home not in strengths["attack"] or away not in strengths["attack"]:
        return None

    base = strengths["base"]
    home_adv = 0.0 if neutral else strengths["home_adv"]

    # The two model equations, exactly as fitted.
    loglam_h = base + home_adv + strengths["attack"][home] - strengths["defence"][away]
    loglam_a = base + strengths["attack"][away] - strengths["defence"][home]
    lam_h, lam_a = float(np.exp(loglam_h)), float(np.exp(loglam_a))

    grid = scoreline_matrix(lam_h, lam_a, max_goals=max_goals)
    grid = apply_dixon_coles(grid, lam_h, lam_a, strengths.get("rho", 0.0))
    p_home, p_draw, p_away = wdl_from_grid(grid)

    # argmax over the grid -> the most probable exact scoreline.
    i, j = np.unravel_index(np.argmax(grid), grid.shape)

    return {
        "lambda_home": lam_h,
        "lambda_away": lam_a,
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "top_scoreline": (int(i), int(j)),
        "grid": grid,
    }


# ============================================================================
# PHASE 4 — adapter: ride the existing WDL backtest harness
# ============================================================================
#
# walk_forward.py / broader_eval.py fit and score ANY object with two methods:
#   model.fit(X, y, sample_weight=..., eval_set=..., verbose=...)
#   model.predict_proba(X) -> N x 3  in H/D/A order
# They slice `df[feature_columns]` before handing X over. The XGBoost model gets
# the 21 engineered features; the Poisson model instead needs the raw match
# identity + goals, so we pass a DIFFERENT feature_columns list for the Poisson
# run (POISSON_FEATURE_COLUMNS). Nothing in the harness changes — the same
# leakage-safe per-fold split, bookmaker join, metrics, and bootstrap CIs all
# apply, so the comparison is genuinely apples-to-apples.

# The columns the adapter needs handed to it (in place of the 21 model features).
# home_score/away_score are needed at FIT time (the model learns from goals) but
# are deliberately IGNORED at predict time — see predict_proba.
POISSON_FEATURE_COLUMNS = ["home_team", "away_team", "home_score", "away_score", "neutral"]
# Phase 5 variant: same, plus the rating_gap covariate column.
POISSON_RATING_COLUMNS = POISSON_FEATURE_COLUMNS + ["rating_gap"]


class PoissonWDL:
    """
    A thin scikit-learn-shaped wrapper around the Dixon-Coles fit so it can be
    dropped into walk_forward / broader_eval as `model_fn=PoissonWDL`.

    Deliberately minimal: `fit` learns attack/defence + rho from the goals in X
    (using the harness-supplied recency x importance `sample_weight`), and
    `predict_proba` turns each match into H/D/A via the Phase-1/3 machinery. It is
    NOT a general classifier — it only understands the POISSON_FEATURE_COLUMNS.

    `use_rating` (Phase 5): if True, also fit a rating_gap covariate on the rate —
    the harness must then be given feature_columns=POISSON_RATING_COLUMNS so the
    `rating_gap` column is present. Default False = plain attack/defence.
    """

    def __init__(self, reg: float = 1e-3, use_rating: bool = False):
        self.reg = reg
        self.use_rating = use_rating
        self.strengths_ = None
        # The harness reads model.best_iteration for its per-tournament table
        # (XGBoost's chosen tree count). A Poisson MLE has no boosting rounds, so
        # this is None; the comparison driver doesn't print it (avoids the format).
        self.best_iteration = None

    def fit(self, X, y=None, sample_weight=None, eval_set=None, verbose=None):
        """Fit from the goals in X. `y` (the mapped H/D/A label) is ignored — the
        scoreline model learns from raw goals, not the collapsed outcome.
        `sample_weight` (recency x importance, from the harness) becomes the
        Dixon-Coles time weighting. `eval_set`/`verbose` are XGBoost early-stopping
        arguments with no Poisson analogue, accepted and ignored."""
        w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        cov = X["rating_gap"].to_numpy(dtype=float) if self.use_rating else None
        self.strengths_ = fit_strengths(X, weights=w, reg=self.reg, covariate=cov)
        self.strengths_["rho"] = fit_rho(X, self.strengths_, weights=w)
        return self

    def predict_proba(self, X) -> np.ndarray:
        """N x 3 array of [P(home win), P(draw), P(away win)] — the exact H/D/A=0/1/2
        order the harness and metrics expect. Reads ONLY home_team/away_team/neutral
        (and rating_gap when use_rating); the home_score/away_score columns present
        in X are the match result and are never touched here (that would leak the
        answer). A team unseen in training is treated as league-average
        (attack=defence=0) — an honest 'unknown' default, not a fabricated strength."""
        s = self.strengths_
        homes = X["home_team"].to_numpy()
        aways = X["away_team"].to_numpy()
        neutrals = X["neutral"].to_numpy()
        gamma = s.get("rating_coef", 0.0)
        if self.use_rating and "rating_gap" in X.columns:
            cov = np.nan_to_num(X["rating_gap"].to_numpy(dtype=float), nan=0.0)
        else:
            cov = np.zeros(len(X))

        out = np.empty((len(X), 3), dtype=float)
        base, home_adv, rho = s["base"], s["home_adv"], s.get("rho", 0.0)
        atk, dfn = s["attack"], s["defence"]
        for k in range(len(X)):
            ha = 0.0 if neutrals[k] else home_adv
            adj = gamma * cov[k]   # rating covariate: +on home rate, −on away rate
            # .get(team, 0.0): unseen team -> average strength (never fabricated).
            lam_h = np.exp(base + ha + atk.get(homes[k], 0.0) - dfn.get(aways[k], 0.0) + adj)
            lam_a = np.exp(base + atk.get(aways[k], 0.0) - dfn.get(homes[k], 0.0) - adj)
            grid = scoreline_matrix(lam_h, lam_a)
            grid = apply_dixon_coles(grid, lam_h, lam_a, rho)
            out[k] = wdl_from_grid(grid)
        return out


def _rank_table(strengths: dict, key: str, min_matches: int = 50):
    """Helper for the demo: sorted (team, value, count) rows for `key`
    ('attack' or 'defence'), restricted to teams with >= min_matches games so the
    ranking reflects established sides, not a handful of flukey fixtures. Callers
    slice the returned list for the top/bottom N."""
    vals = strengths[key]
    counts = strengths["match_counts"]
    rows = [(t, vals[t], counts[t]) for t in vals if counts[t] >= min_matches]
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows


if __name__ == "__main__":
    # Run from repo root:  python -m src.models.poisson
    print("Loading + cleaning matches (~14s)...", flush=True)
    m = clean_matches(load_raw_matches())
    print(f"cleaned matches: {len(m):,}", flush=True)

    print("\nFitting Dixon-Coles (time-weighted strengths + rho)...", flush=True)
    s = fit_dixon_coles(m, verbose=True)
    print(f"base={s['base']:.3f}  (=> avg goals/side {np.exp(s['base']):.2f})   "
          f"home_adv={s['home_adv']:.3f}  (=> home rate x{np.exp(s['home_adv']):.2f})   "
          f"rho={s['rho']:+.4f}")

    atk = _rank_table(s, "attack")
    dfc = _rank_table(s, "defence")
    print("\n--- Top 10 ATTACK (>=50 matches) ---")
    for t, v, c in atk[:10]:
        print(f"  {t:<18}{v:+.3f}   ({c} matches)")
    print("\n--- Top 10 DEFENCE (>=50 matches, higher = concedes fewer) ---")
    for t, v, c in dfc[:10]:
        print(f"  {t:<18}{v:+.3f}   ({c} matches)")
    print("--- Bottom 5 DEFENCE (leakiest) ---")
    for t, v, c in dfc[-5:]:
        print(f"  {t:<18}{v:+.3f}   ({c} matches)")

    # --- Show the Dixon-Coles correction at work on the four low-score cells ----
    print("\n--- Dixon-Coles low-score shift (Spain vs Germany, home) ---")
    p_dc = predict_match(s, "Spain", "Germany")
    s_plain = dict(s); s_plain["rho"] = 0.0          # same strengths, no correction
    p_plain = predict_match(s_plain, "Spain", "Germany")
    lh, la = p_dc["lambda_home"], p_dc["lambda_away"]
    print(f"  lambda {lh:.2f}-{la:.2f}   rho={s['rho']:+.4f}")
    for (i, j), name in [((0, 0), "0-0"), ((1, 1), "1-1"), ((1, 0), "1-0"), ((0, 1), "0-1")]:
        # normalise each grid so the cells are comparable probabilities
        pl = p_plain["grid"][i, j] / p_plain["grid"].sum()
        dc = p_dc["grid"][i, j] / p_dc["grid"].sum()
        arrow = "up" if dc > pl else "down"
        print(f"  P({name}): plain {pl:.4f} -> DC {dc:.4f}  ({arrow})")
    print(f"  P(draw): plain {p_plain['p_draw']:.4f} -> DC {p_dc['p_draw']:.4f}")

    print("\n--- Example predictions (with Dixon-Coles) ---")
    for home, away, neutral in [("Germany", "San Marino", False),
                                ("Brazil", "Argentina", True),
                                ("Spain", "Germany", False)]:
        p = predict_match(s, home, away, neutral=neutral)
        print(f"  {home} vs {away} (neutral={neutral}): "
              f"lambda {p['lambda_home']:.2f}-{p['lambda_away']:.2f}  "
              f"P(H/D/A)={p['p_home']:.2f}/{p['p_draw']:.2f}/{p['p_away']:.2f}  "
              f"most likely {p['top_scoreline'][0]}-{p['top_scoreline'][1]}")
