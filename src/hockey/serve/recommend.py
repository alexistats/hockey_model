"""A shortlist, with the reasoning attached and the rules priced.

Two things this deliberately does not do.

It does not return one answer. A draft decision between comparable players is
close by construction - that is what makes it a decision - and handing back a
single name throws away the information that would let a human overrule it in
the four seconds they have. It returns a few, ranked, with what separates them.

And it does not hide what a rule costs. A constraint like "no goalie before
round 7" is a real instruction and is followed, but it is followed *visibly*:
the unconstrained best pick is computed too, and the gap between them is
reported. A bot that logs "took Swayman, rule bound, cost 96 points against
Vasilevskiy" can be audited after a mock draft. One that logs "took Swayman"
cannot, and the rule can never be evaluated.
"""

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

# Rounds are 1-indexed and inclusive: `min_round` 7 means the first goalie may
# be taken in round 7 or later.
DEFAULT_RULES = {
    "goalie_min_round": 7,
    "goalie_second_min_round": 14,
}


@dataclass
class Candidate:
    player_id: int
    player: str
    slot: str
    eligible: list[str]
    team: str
    mean: float
    floor: float
    ceiling: float
    vorp: float
    tier: int | None
    replacement_is_lower_bound: bool
    fills_a_need: bool
    beats_next: float | None = None
    blocked_by: str | None = None
    notes: list[str] = field(default_factory=list)


def assign_roster(mine: pd.DataFrame, shape: dict[str, int], bench: int) -> dict:
    """Put my drafted players into my roster slots, then say what is still open.

    Counting a roster by reading each player's position is wrong for the same
    reason pooling by position is wrong for replacement level: eligibility
    overlaps. Four right-wing-eligible players are not four right wings when
    two of them are also centres and the centre slots are empty. Reading a
    static column gets this backwards in both directions - it reported five
    left wings and no right wings on a roster that had four RW-eligible
    players - and a bot told it still needs two right wings will spend picks
    fixing a shortage it does not have.

    So the roster is filled the same way the league is: best first, into an
    open slot the player is eligible for, preferring the slot with the most
    room. Whoever does not fit a starting slot is bench depth, which is real
    but is not a need.
    """
    if mine.empty:
        return {
            "lineup": {},
            "filled": {},
            "needs": dict(shape),
            "bench": [],
            "bench_slots_left": bench,
        }
    from hockey.export.replacement import fill_slots

    assigned, leftover = fill_slots(mine, {p: float(n) for p, n in shape.items()})
    names = mine.set_index(mine["player_id"].astype(int))["player"].to_dict()
    lineup: dict[str, list[str]] = {}
    filled: dict[str, int] = {}
    for pid, slot in assigned.items():
        lineup.setdefault(slot, []).append(str(names.get(int(pid), pid)))
        filled[slot] = filled.get(slot, 0) + 1
    return {
        "lineup": lineup,
        "filled": filled,
        "needs": {p: max(0, n - filled.get(p, 0)) for p, n in shape.items()},
        "bench": [str(r.player) for r in leftover.itertuples()],
        "bench_slots_left": max(0, bench - len(leftover)),
    }


def _roster_needs(mine: pd.DataFrame, shape: dict[str, int]) -> dict[str, int]:
    """What is still open, after assigning who I have under the real slots."""
    return assign_roster(mine, shape, bench=0)["needs"]


def _rule_block(row, state, rules: dict, mine: pd.DataFrame) -> str | None:
    """Why this player may not be taken yet, or None."""
    if str(row.slot) != "G":
        return None
    goalies_held = (
        sum(
            1
            for r in mine.itertuples()
            if "G"
            in (r.eligible if isinstance(getattr(r, "eligible", None), tuple | list) else (r.slot,))
        )
        if len(mine)
        else 0
    )
    if goalies_held == 0:
        floor_round = rules.get("goalie_min_round")
        if floor_round and state.current_round < floor_round:
            return f"no goalie before round {floor_round} (currently round {state.current_round})"
        return None
    second = rules.get("goalie_second_min_round")
    if second and state.current_round < second:
        return f"second goalie not before round {second} (currently round {state.current_round})"
    return None


def shortlist(
    board,
    state,
    valued: pd.DataFrame,
    levels: pd.DataFrame,
    draws_for,
    limit: int = 4,
    rules: dict | None = None,
) -> dict:
    """The top candidates, what separates them, and what a rule cost."""
    rules = {**DEFAULT_RULES, **(rules or {})}
    mine = board.players[board.players["player_id"].astype(int).isin(state.mine)]
    needs = _roster_needs(mine, board.roster_shape)
    lower_bound = {
        str(r.position): bool(getattr(r, "extrapolated", False)) for r in levels.itertuples()
    }

    ranked = valued.sort_values("vorp", ascending=False)

    def make(row, blocked: str | None) -> Candidate:
        eligible = list(row.eligible) if isinstance(row.eligible, tuple | list) else [row.slot]
        return Candidate(
            player_id=int(row.player_id),
            player=str(row.player),
            slot=str(row.slot),
            eligible=eligible,
            team=str(row.team),
            mean=float(row.mean),
            floor=float(row.floor),
            ceiling=float(row.ceiling),
            vorp=float(row.vorp),
            tier=None if pd.isna(row.tier) else int(row.tier),
            replacement_is_lower_bound=lower_bound.get(str(row.slot), False),
            fills_a_need=needs.get(str(row.slot), 0) > 0,
            blocked_by=blocked,
        )

    allowed: list[Candidate] = []
    blocked: list[Candidate] = []
    for row in ranked.itertuples():
        why = _rule_block(row, state, rules, mine)
        if why is None:
            if len(allowed) < limit:
                allowed.append(make(row, None))
        elif len(blocked) < 2:
            blocked.append(make(row, why))
        if len(allowed) >= limit and len(blocked) >= 2:
            break

    # How often each candidate outscores the one behind it. Close pairs are the
    # only place this matters, and the number says whether the ordering is a
    # real preference or a coin flip dressed as one.
    for a, b in zip(allowed, allowed[1:], strict=False):
        x, y = draws_for(a.player_id), draws_for(b.player_id)
        if x is not None and y is not None:
            wins = float((x > y).mean() + (x == y).mean() / 2)
            a.beats_next = round(wins, 4)
            if wins < 0.55:
                a.notes.append(f"barely separated from {b.player} ({wins:.0%})")

    for c in allowed:
        if c.replacement_is_lower_bound:
            c.notes.append(
                f"{c.slot} has no freely available player left, so this value is a "
                f"lower bound and real scarcity is higher"
            )
        if not c.fills_a_need:
            c.notes.append(f"my {c.slot} slots are already filled; this is bench depth")

    # What the rules cost, priced rather than asserted.
    rule_cost = None
    if blocked and allowed:
        best_blocked = max(blocked, key=lambda c: c.vorp)
        if best_blocked.vorp > allowed[0].vorp:
            rule_cost = {
                "would_have_taken": best_blocked.player,
                "player_id": best_blocked.player_id,
                "blocked_by": best_blocked.blocked_by,
                "vorp_forgone": round(best_blocked.vorp - allowed[0].vorp, 1),
                "instead": allowed[0].player,
            }

    return {
        "round": state.current_round,
        "overall_pick": state.current_pick,
        "on_the_clock": state.on_the_clock,
        "picks_until_my_turn": state.picks_until_my_turn(),
        "needs": needs,
        "candidates": [c.__dict__ for c in allowed],
        "blocked_by_rules": [c.__dict__ for c in blocked],
        "rule_cost": rule_cost,
    }


def cost_of_waiting(valued: pd.DataFrame, positions: list[str], picks: int) -> dict:
    """Best available now against best available after `picks` more selections.

    Assumes those picks come off the top of the value board. The room will not
    do exactly that, so this is the direction and rough size of what passing
    costs, not a forecast - and it is reported as such rather than as a number
    that looks more certain than it is.
    """
    pool = valued.sort_values("vorp", ascending=False)
    taken = set(pool.head(picks)["player_id"].astype(int))
    out = {}
    for position in positions:
        eligible = pool[
            [
                position in (r.eligible if isinstance(r.eligible, tuple | list) else (r.slot,))
                for r in pool.itertuples()
            ]
        ]
        if eligible.empty:
            out[position] = None
            continue
        now = float(eligible.iloc[0]["vorp"])
        later = eligible[~eligible["player_id"].astype(int).isin(taken)]
        then = float(later.iloc[0]["vorp"]) if len(later) else None
        out[position] = {
            "best_now": round(now, 1),
            "best_after": None if then is None else round(then, 1),
            "cost": None if then is None else round(max(0.0, now - then), 1),
        }
    return out
