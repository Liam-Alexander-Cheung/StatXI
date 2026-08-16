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


# ===========================================================================
# Euro 24-team format: the "best third-placed teams" knockout wiring
# ===========================================================================
# In the 24-team Euro format (2016, 2020, 2024) the six group winners (1A-1F) and
# six runners-up (2A-2F) advance, PLUS the four best of the six third-placed teams.
# Four group winners host a third-placed team in the Round of 16; WHICH group's third
# each faces is fixed by the SET of the four groups whose thirds qualified — UEFA's
# published anti-rematch table (a group's winner is never paired with its own group's
# third). There are C(6,4)=15 possible sets.
#
# The bracket layout is NOT the same across editions: Euro 2020 and 2024 share one
# layout (winners 1B/1C/1E/1F host the thirds), but Euro 2016 used a different one
# (winners 1A/1B/1C/1D). So the table lives IN each config next to its bracket, not
# as a shared constant. A euro24 config therefore carries two extra fields:
#   "third_slots"  : the ordered labels of the four best-third R16 slots, e.g.
#                    ["3vB","3vC","3vE","3vF"] — "3vB" is the third that plays the
#                    winner of group B, so its own host group is the letter after "3v".
#   "thirds_table" : {four-group combo -> [group filling each third-slot, positional
#                    to "third_slots"]}, one row per the 15 combos. Cross-checkable
#                    against Wikipedia's "Combinations of matches in the round of 16"
#                    (Tomas's manual-verification domain); validated on load by
#                    _validate_euro24_thirds below (bijection + no same-group rematch).
EURO_GROUP_LETTERS = "ABCDEF"


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
    hand-encoding slip is caught immediately, not deep inside a simulation. Both
    supported formats end in the same 16-team knockout bracket; they differ in the
    group count and in how the Round-of-16 slots are labelled (the 24-team Euro adds
    the four best-third slots)."""
    fmt = cfg.get("format")
    if fmt == "wc32":
        _validate_groups(cfg, n_groups=8)
        _validate_bracket_counts(cfg)
        _validate_wc32_r16_slots(cfg)
    elif fmt == "euro24":
        _validate_groups(cfg, n_groups=6)
        _validate_bracket_counts(cfg)
        _validate_euro24_thirds(cfg)
        _validate_euro24_r16_slots(cfg)
    else:
        raise ValueError(f"Unsupported tournament format {fmt!r} "
                         f"(expected 'wc32' or 'euro24')")


def _validate_groups(cfg: dict, n_groups: int) -> None:
    """`n_groups` groups of exactly 4, with all team names distinct."""
    groups = cfg["groups"]
    if len(groups) != n_groups:
        raise ValueError(f"{cfg['format']} needs {n_groups} groups, got {len(groups)}")
    for g, teams in groups.items():
        if len(teams) != 4:
            raise ValueError(f"Group {g} has {len(teams)} teams, expected 4")
    all_teams = [t for teams in groups.values() for t in teams]
    n_expected = n_groups * 4
    if len(set(all_teams)) != n_expected:
        raise ValueError(f"Expected {n_expected} distinct teams across groups, "
                         f"got {len(set(all_teams))} (duplicate team name?)")


def _validate_bracket_counts(cfg: dict) -> None:
    """A 16-team single-elimination bracket: 8 R16 + 4 QF + 2 SF + 1 F (both formats)."""
    bracket = cfg["bracket"]
    expected = {"R16": 8, "QF": 4, "SF": 2, "F": 1}
    for rnd, n in expected.items():
        got = len(bracket.get(rnd, []))
        if got != n:
            raise ValueError(f"Bracket round {rnd} has {got} matches, expected {n}")


def _r16_slot_labels(cfg: dict) -> list[str]:
    """Every home/away label used across the eight R16 matches."""
    r16 = cfg["bracket"]["R16"]
    return [m["home"] for m in r16] + [m["away"] for m in r16]


def _validate_wc32_r16_slots(cfg: dict) -> None:
    """Each R16 label is a '<position><group>' like '1A'/'2B', using each
    group-qualifier slot exactly once (32-team World Cup: top-2 of eight groups)."""
    groups = cfg["groups"]
    qpg = cfg["qualify_per_group"]
    valid_slots = {f"{pos}{g}" for g in groups for pos in range(1, qpg + 1)}
    if sorted(_r16_slot_labels(cfg)) != sorted(valid_slots):
        raise ValueError("R16 slot labels do not use each group-qualifier slot "
                         f"exactly once.\n  expected: {sorted(valid_slots)}\n"
                         f"  got:      {sorted(_r16_slot_labels(cfg))}")


def _validate_euro24_thirds(cfg: dict) -> None:
    """Validate the config's own best-thirds allocation table (its copy of UEFA's).

    `third_slots`: four labels "3v<G>" where <G> is a real group — the host group
    <G> is the winner that slot's third plays. `thirds_table`: one row per the 15
    four-of-six group combinations, each a list (positional to `third_slots`) of the
    group whose third fills each slot. Every row must be a bijection of the combo's
    four groups onto the four slots, with NO slot given its own host group (the whole
    point of the anti-rematch table). Fails loudly on any hand-encoding slip."""
    from itertools import combinations
    groups = set(cfg["groups"])
    third_slots = cfg["third_slots"]
    if len(third_slots) != 4:
        raise ValueError(f"euro24 needs 4 third_slots, got {third_slots}")
    hosts = []
    for s in third_slots:
        if not (s.startswith("3v") and s[2:] in groups):
            raise ValueError(f"third-slot {s!r} must be '3v<group>' with a real group")
        hosts.append(s[2:])

    table = cfg["thirds_table"]
    combos = {"".join(c) for c in combinations(sorted(groups), 4)}
    if set(table) != combos:
        raise ValueError("thirds_table keys must be exactly the 15 four-of-six group "
                         f"combinations.\n  missing: {sorted(combos - set(table))}"
                         f"\n  extra:   {sorted(set(table) - combos)}")
    for combo, row in table.items():
        if len(row) != 4 or set(row) != set(combo):
            raise ValueError(f"thirds_table[{combo}]={row} must be the four groups "
                             f"{list(combo)} in some order (positional to third_slots)")
        for host, grp in zip(hosts, row):
            if grp == host:
                raise ValueError(f"thirds_table[{combo}]: group {grp}'s third plays its "
                                 f"own winner 1{host} (same-group rematch)")


def _validate_euro24_r16_slots(cfg: dict) -> None:
    """The 24-team Euro R16 uses the twelve group-qualifier slots (1A-1F, 2A-2F)
    PLUS the four best-third slots (cfg['third_slots']), each exactly once."""
    groups = cfg["groups"]
    third_slots = cfg["third_slots"]
    valid_slots = {f"{pos}{g}" for g in groups for pos in (1, 2)} | set(third_slots)
    if sorted(_r16_slot_labels(cfg)) != sorted(valid_slots):
        raise ValueError("Euro R16 slot labels must be the 12 group-qualifier slots "
                         f"plus {third_slots}, each once.\n  expected: {sorted(valid_slots)}"
                         f"\n  got:      {sorted(_r16_slot_labels(cfg))}")


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
