"""
Hand-encoded tournament STRUCTURE — the one thing the Monte Carlo simulator needs
that is not in the data.

The `matches` table stores only date / teams / scores / tournament / neutral. It has
NO column for stage, group, or bracket position, and penalty-shootout knockouts are
stored as draws with no winner. So to simulate a whole tournament we must supply its
structure (which teams are in which group, and how the knockout bracket is wired)
ourselves. That structure is checked in as SOURCE data (e.g. wc2022.json), small and
cross-checkable against Wikipedia — unlike the regenerable derived DB, it IS
committed.

This module loads such a config and validates it: the counts are internally
consistent, and (given a fitted `strengths` dict from poisson.fit_dixon_coles) every
team name resolves to a real fitted strength — failing LOUDLY on any miss, so a
typo'd or renamed team can never silently become a league-average phantom.
"""

from __future__ import annotations

import json
from pathlib import Path

_TOURNAMENT_DIR = Path(__file__).resolve().parent


def load_tournament_config(name: str) -> dict:
    """Load a hand-encoded tournament config by file stem (e.g. "wc2022" ->
    wc2022.json in this directory) and validate its internal structure. Does NOT
    check team names against a fit — that needs a `strengths` dict, see
    validate_teams. Raises FileNotFoundError / ValueError on a malformed config."""
    path = _TOURNAMENT_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"No tournament config at {path}")
    with open(path) as f:
        cfg = json.load(f)

    _validate_structure(cfg)
    return cfg


def _validate_structure(cfg: dict) -> None:
    """Check the config's own arithmetic is right (independent of any fit), so a
    hand-encoding slip is caught immediately, not deep inside a simulation."""
    fmt = cfg.get("format")
    if fmt != "wc32":
        raise ValueError(f"Only the 32-team World Cup format ('wc32') is supported "
                         f"so far, got {fmt!r}")

    groups = cfg["groups"]
    # 8 groups of exactly 4 = 32 teams.
    if len(groups) != 8:
        raise ValueError(f"wc32 needs 8 groups, got {len(groups)}")
    for g, teams in groups.items():
        if len(teams) != 4:
            raise ValueError(f"Group {g} has {len(teams)} teams, expected 4")
    all_teams = [t for teams in groups.values() for t in teams]
    if len(set(all_teams)) != 32:
        raise ValueError(f"Expected 32 distinct teams across groups, "
                         f"got {len(set(all_teams))} (duplicate team name?)")

    # Bracket: 16-team single elimination = 8 R16 + 4 QF + 2 SF + 1 F.
    bracket = cfg["bracket"]
    expected = {"R16": 8, "QF": 4, "SF": 2, "F": 1}
    for rnd, n in expected.items():
        got = len(bracket.get(rnd, []))
        if got != n:
            raise ValueError(f"Bracket round {rnd} has {got} matches, expected {n}")

    # Every R16 slot label must be a valid "<position><group>" like "1A" / "2B",
    # referencing a real group and a qualifying position.
    qpg = cfg["qualify_per_group"]
    valid_slots = {f"{pos}{g}" for g in groups for pos in range(1, qpg + 1)}
    r16_slots = [m["home"] for m in bracket["R16"]] + [m["away"] for m in bracket["R16"]]
    if sorted(r16_slots) != sorted(valid_slots):
        raise ValueError("R16 slot labels do not use each group-qualifier slot "
                         f"exactly once.\n  expected: {sorted(valid_slots)}\n"
                         f"  got:      {sorted(r16_slots)}")


def all_config_teams(cfg: dict) -> list[str]:
    """Flat list of the 32 team names in a validated config (group order)."""
    return [t for teams in cfg["groups"].values() for t in teams]


def validate_teams(cfg: dict, strengths: dict) -> None:
    """Assert every team in the config resolves to a fitted attack/defence strength.
    Raises ValueError listing any unresolved names (a typo, or a team with no
    post-1990 matches) — honouring the never-fabricate rule: an unknown team must
    fail loudly, never silently default to average inside a simulation."""
    known = strengths["attack"].keys()
    missing = [t for t in all_config_teams(cfg) if t not in known]
    if missing:
        raise ValueError(f"{len(missing)} config team(s) not in the fitted strengths "
                         f"(typo or no matches?): {missing}")
