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
from types import SimpleNamespace

import pandas as pd

logger = logging.getLogger(__name__)

# Rounds are 1-indexed and inclusive: `min_round` 7 means the first goalie may
# be taken in round 7 or later.
#
# `rank_by` "risk" orders the shortlist by a floor-to-ceiling blend that moves
# from cautious in the first round to ambitious in the last; "vorp" orders it by
# the mean alone. `risk_start` and `risk_end` are the weight on the optimistic
# quantile at each end of the draft.
DEFAULT_RULES = {
    "goalie_min_round": 7,
    "goalie_second_min_round": 14,
    # Two start. Without a cap, the late ambitious rounds fill the bench with
    # backup goalies, whose p80 sits far above the goalie replacement level.
    "goalie_max": 3,
    "rank_by": "risk",
    "risk_start": 0.2,
    "risk_end": 0.8,
    # A starting slot left open is a logjam waiting to happen: take one
    # defenceman, then forwards all night, and the last rounds are spent filling
    # blue line with whatever is left. So a candidate who fills an open slot is
    # preferred over one who does not, unless the bench player is better by more
    # than this. "Unless the forwards are insanely better" priced, not asserted.
    "need_first_margin": 25.0,
    # How many rounds this draft actually runs. Defaults to the roster: starting
    # slots plus bench. Set it when the league drafts fewer, because every
    # deadline calculation below is measured against it.
    "draft_rounds": None,
    # Once every starting slot is filled, the remaining picks are bench. The
    # last `plan_window` of them are planned: the final `schedule_picks` go to
    # players who would actually be in the lineup in the opening weeks, and the
    # rest swing for the ceiling.
    "plan_window": 5,
    "schedule_picks": 2,
    "opening_weeks": 2,
}

# Statuses that mean the player cannot play now. A schedule pick is about games
# in the next fortnight, so one of these makes the whole point moot.
CANNOT_PLAY = ("Out", "Injured Reserve", "Suspension")

# Below this gap between the two costliest open positions, neither is a reason
# to reach. The page uses the same two numbers and the same room model, but not
# yet the blocked-position and lone-position rules in `positional_read`, so the
# two can differ there.
NO_CLEAR_CALL = 8.0
URGENT = 20.0


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
    p20: float
    p80: float
    risk_score: float
    tier: int | None
    replacement_is_lower_bound: bool
    replacement_free_below: int
    fills_a_need: bool
    need_slot: str | None = None
    injury: dict | None = None
    schedule_opening: dict | None = None
    schedule_season: dict | None = None
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


def draft_rounds(board, rules: dict) -> int:
    """How many rounds this draft runs: `draft_rounds` when set, else the roster.

    One copy, because three things are measured against it - the deadline, the
    plan for the last picks and the risk ramp - and they used to disagree. In a
    16-round mock with a 17-slot roster the deadline counted 16 while the plan
    and the ramp counted 17, so the two schedule picks started a round late and
    the ramp never reached its ambitious end.
    """
    return int(rules.get("draft_rounds") or (sum(board.roster_shape.values()) + board.bench))


def risk_weight(current_round: int, n_rounds: int, start: float, end: float) -> float:
    """Weight on the optimistic quantile at this round, ramped linearly.

    Early picks are expensive and a bust there cannot be replaced, so they lean
    on the cautious outcome. Late picks are cheap and a miss costs a waiver
    claim, so they lean on the upside. A ramp rather than a switch, because a
    board that reorders abruptly at one round is a rule nobody can reason about.
    """
    if n_rounds <= 1:
        return end
    t = min(1.0, max(0.0, (current_round - 1) / (n_rounds - 1)))
    return start + t * (end - start)


def add_risk_score(valued: pd.DataFrame, weight: float) -> pd.DataFrame:
    """Blend p20 and p80, each measured over the player's own replacement level.

    Over replacement, not raw. Raw floors rank nearly every defenceman below
    the forwards, and raw ceilings rank goalies above everyone, because those
    are the lowest-scoring and the most volatile positions - a bot that ranked
    by either drafted forwards, then goalies, then defence. Subtracting the
    replacement level the same way `vorp` does keeps positions comparable.

    p20 and p80 rather than the floor and ceiling: at the extremes the goalies'
    spread swamps everything else.
    """
    out = valued.copy()
    out["risk_score"] = (1 - weight) * (out["p20"] - out["replacement"]) + weight * (
        out["p80"] - out["replacement"]
    )
    return out


def positional_read(
    cost: dict, needs: dict[str, int], picks: int, blocked: frozenset[str] = frozenset()
) -> dict:
    """Whether a position is draining fast enough to be worth reaching for.

    Only open starting slots count - a position I have filled costs me nothing
    to pass on, however fast it is going - and only positions a rule lets me
    take now, since "take a G" in round 1 is advice I cannot follow. The costs
    are expected losses over simulated rooms, so a reach here is a bet that
    pays on average, not a certainty about the next pick.
    """
    open_ = {
        p: cost[p]["cost"]
        for p, n in needs.items()
        if n > 0 and p not in blocked and cost.get(p) and cost[p]["cost"] is not None
    }
    base = {
        "picks_until_my_turn": picks,
        "open_costs": open_,
        "urgent": sorted(p for p, c in open_.items() if c >= URGENT),
        "assumption": (
            "the other teams draft in the room's own order (average pick over saved "
            "mock rooms) toward their open slots; each cost is an expected loss"
        ),
    }
    if not open_:
        waiting_on = sorted(p for p, n in needs.items() if n > 0 and p in blocked)
        unpriced = sorted(
            p
            for p, n in needs.items()
            if n > 0 and p not in blocked and (not cost.get(p) or cost[p]["cost"] is None)
        )
        return {
            **base,
            "call": "best_available",
            "position": None,
            "reason": (
                f"no cost of waiting at the open slots ({', '.join(unpriced)}) - there is no "
                f"room model to say what the other teams will take"
                if unpriced
                else f"the open slots ({', '.join(waiting_on)}) are blocked by a rule for now"
                if waiting_on
                else "every starting slot is filled; the bench is positionless"
            ),
        }
    ranked = sorted(open_, key=open_.get, reverse=True)
    first = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    # A lone open position still has to clear the bar: waiting on it for
    # nothing is not a reason to reach.
    runner_up = open_[second] if second is not None else 0.0
    if open_[first] - runner_up < NO_CLEAR_CALL:
        return {
            **base,
            "call": "no_clear_call",
            "position": None,
            "reason": (
                f"passing on {first} costs about {open_[first]:.0f}"
                + (f" and on {second} about {runner_up:.0f}" if second else "")
                + "; not enough to reach, take the better player"
            ),
        }
    return {
        **base,
        "call": "position",
        "position": first,
        "reason": (
            f"waiting {picks} picks costs about {open_[first]:.0f} at {first}"
            + (f", against {open_[second]:.0f} at {second}" if second else "")
        ),
    }


def roster_pressure(board, state, rules: dict, needs: dict[str, int]) -> dict:
    """How much room is left to defer a starting slot.

    `cost_of_waiting` is a value question: what the best player left at a
    position will be worth at my next turn. What it cannot see is that the draft
    ends. An unfilled starting slot scores zero for the season, and the cost of
    deferring it is not the drop in the best available - it is the risk of
    running out of picks. That cost is zero for most of the draft and then
    enormous, which is a shape no value curve has.

    So it is counted instead of priced. `slack` is the picks I have beyond the
    slots I still must fill. While it is large a better bench player is worth
    taking; as it shrinks the bar rises; at zero every remaining pick is spoken
    for and bench depth is not on the table at any gap.

    This is what let two mocks finish with three empty defence slots and seven
    forwards on the bench: each individual override cleared the flat 25-point
    bar, and nothing was counting the picks left to fill them.
    """
    picks_left = max(0, draft_rounds(board, rules) - len(state.mine))
    slots_open = sum(needs.values())
    slack = picks_left - slots_open
    base = float(rules["need_first_margin"])
    if slots_open == 0:
        margin = base
    elif slack <= 0:
        margin = None  # nothing is worth a starting slot now
    else:
        # Rises as the room disappears: base when picks are plentiful, and
        # steeply higher as the last few are claimed by open slots.
        margin = base * picks_left / slack
    return {
        "picks_left": picks_left,
        "slots_open": slots_open,
        "open": {p: n for p, n in needs.items() if n},
        "slack": slack,
        "need_first_margin": None if margin is None else round(margin, 1),
        "reason": (
            f"{picks_left} pick(s) left for {slots_open} open starting slot(s)"
            + ("; every remaining pick is spoken for" if slack <= 0 and slots_open else "")
        ),
    }


def draft_plan(board, state, rules: dict, needs: dict[str, int]) -> dict:
    """What this pick is for, once the starting lineup is complete.

    The late picks were the iffy ones: with every slot filled, value over
    replacement barely separates anybody and the board drifts into whoever is
    nominally highest. So the last few are given jobs. The final
    `schedule_picks` go to players who would actually be in the lineup in the
    opening weeks - a fourth centre whose games land on nights my first three
    already cover is worth nothing in week one - and the picks before them swing
    for the ceiling, where a bust costs a waiver claim and a hit wins a week.
    """
    left = max(0, draft_rounds(board, rules) - len(state.mine))
    starters_open = sum(needs.values())
    if starters_open:
        return {
            "pick_kind": "starter",
            "picks_left": left,
            "reason": f"{starters_open} starting slot(s) still open",
        }
    if left <= int(rules["schedule_picks"]):
        return {
            "pick_kind": "schedule",
            "picks_left": left,
            "reason": (
                f"last {rules['schedule_picks']} pick(s): take someone who would be in "
                f"the lineup in the opening {rules['opening_weeks']} week(s)"
            ),
        }
    if left <= int(rules["plan_window"]):
        return {
            "pick_kind": "ceiling",
            "picks_left": left,
            "reason": "bench pick inside the planned window: swing for the ceiling",
        }
    return {"pick_kind": "bench", "picks_left": left, "reason": "bench depth, best available"}


def _windows(board, rules: dict) -> tuple[tuple | None, tuple | None]:
    """The opening fortnight and the whole season, as calendar spans."""
    from hockey.serve.schedule import week_window

    opening = week_window(board.weeks, 1, int(rules["opening_weeks"])) if board.weeks else None
    days = [d for days in board.calendar.values() for d in days]
    season = (min(days), max(days)) if days else None
    return opening, season


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
    cap = rules.get("goalie_max")
    if cap and goalies_held >= cap:
        return f"already holding {goalies_held} goalies (cap {cap})"
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
    limit: int = 6,
    rules: dict | None = None,
) -> dict:
    """The top candidates, what separates them, and what a rule cost."""
    rules = {**DEFAULT_RULES, **(rules or {})}
    if rules["rank_by"] not in ("risk", "vorp"):
        raise ValueError(f"rank_by must be 'risk' or 'vorp', not {rules['rank_by']!r}")
    mine = board.players[board.players["player_id"].astype(int).isin(state.mine)]
    needs = _roster_needs(mine, board.roster_shape)
    lower_bound = {
        str(r.position): bool(getattr(r, "extrapolated", False)) for r in levels.itertuples()
    }
    # How many genuinely free players set each baseline. Under a handful, the
    # baseline is noise and a small edge at that position is not a reason.
    free_below = {str(r.position): int(getattr(r, "free_below", 99)) for r in levels.itertuples()}

    weight = risk_weight(
        state.current_round,
        draft_rounds(board, rules),
        float(rules["risk_start"]),
        float(rules["risk_end"]),
    )
    valued = add_risk_score(valued, weight)
    key = "risk_score" if rules["rank_by"] == "risk" else "vorp"
    ranked = valued.sort_values(key, ascending=False)

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
            p20=float(row.p20),
            p80=float(row.p80),
            risk_score=round(float(row.risk_score), 1),
            tier=None if pd.isna(row.tier) else int(row.tier),
            replacement_is_lower_bound=lower_bound.get(str(row.slot), False),
            replacement_free_below=free_below.get(str(row.slot), 99),
            # Any eligible position with an opening counts, not just the slot
            # he happens to be valued at: a centre-and-right-wing who is valued
            # at centre still fills an open right wing, and reading the single
            # valued slot says he fills nothing.
            fills_a_need=any(needs.get(p, 0) > 0 for p in eligible),
            need_slot=next((p for p in eligible if needs.get(p, 0) > 0), None),
            injury=board.injuries.get(int(row.player_id)),
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
                f"{c.slot} has only {c.replacement_free_below} freely available player(s) "
                f"left, so the baseline reaches up into the cheapest starters: this value "
                f"is a lower bound, real scarcity is higher, and a small edge over a "
                f"candidate at another position is not a reason to prefer this one"
            )
        if not c.fills_a_need:
            c.notes.append(f"my {c.slot} slots are already filled; this is bench depth")

    # Whatever else is on the shortlist, the best player who fills an open
    # starting slot has to be on it. Otherwise a deadline can arrive with
    # nothing to satisfy it: the top six were all bench depth, and the rule that
    # must take a defenceman has no defenceman to take.
    if any(needs.values()) and not any(c.fills_a_need for c in allowed):
        filler = next(
            (
                row
                for row in ranked.itertuples()
                if _rule_block(row, state, rules, mine) is None
                and any(
                    needs.get(p, 0) > 0
                    for p in (
                        row.eligible if isinstance(row.eligible, tuple | list) else (row.slot,)
                    )
                )
            ),
            None,
        )
        if filler is not None:
            extra = make(filler, None)
            extra.notes.append(
                f"on the shortlist because he fills an open {extra.need_slot} slot, "
                f"not because he is in the top {limit} by score"
            )
            allowed.append(extra)

    # What the rules cost, priced rather than asserted.
    rule_cost = None
    if blocked and allowed:
        best_blocked = max(blocked, key=lambda c: getattr(c, key))
        if getattr(best_blocked, key) > getattr(allowed[0], key):
            rule_cost = {
                "would_have_taken": best_blocked.player,
                "player_id": best_blocked.player_id,
                "blocked_by": best_blocked.blocked_by,
                "vorp_forgone": round(best_blocked.vorp - allowed[0].vorp, 1),
                "score_forgone": round(best_blocked.risk_score - allowed[0].risk_score, 1),
                "instead": allowed[0].player,
            }

    # What each candidate would actually start, against the roster I hold. Only
    # asked once the lineup is full or the plan calls for it: mid-draft the
    # answer is dominated by slots that are still empty, where everyone starts.
    plan = draft_plan(board, state, rules, needs)
    opening, season = _windows(board, rules)
    if board.calendar and plan["pick_kind"] in ("schedule", "ceiling", "bench"):
        from hockey.serve.schedule import marginal_starts

        for c in allowed:
            row = valued[valued["player_id"].astype(int) == c.player_id].iloc[0]
            if opening:
                c.schedule_opening = marginal_starts(
                    row, mine, board.calendar, board.roster_shape, opening
                ).as_dict()
            if season:
                c.schedule_season = marginal_starts(
                    row, mine, board.calendar, board.roster_shape, season
                ).as_dict(with_dates=False)

    # The position read is on the mean, not the risk blend: it asks what a
    # position will be worth on average at my next turn, which is a question
    # about the room, not about my appetite for variance.
    picks = state.picks_until_my_turn()
    positions = [p for p in board.roster_shape if p in board.slots]
    waiting, room = cost_of_waiting(board, state, valued)
    blocked_now = frozenset(
        p for p in positions if _rule_block(SimpleNamespace(slot=p), state, rules, mine)
    )
    read = positional_read(waiting, needs, picks, blocked_now)
    if read["position"] is not None:
        best_there = next(
            (
                row
                for row in ranked.itertuples()
                if read["position"]
                in (row.eligible if isinstance(row.eligible, tuple | list) else (row.slot,))
                and _rule_block(row, state, rules, mine) is None
            ),
            None,
        )
        read["take"] = None if best_there is None else make(best_there, None).__dict__

    pressure = roster_pressure(board, state, rules, needs)
    pick = _recommend(allowed, read, plan, rules, key, pressure)

    return {
        "round": state.current_round,
        "overall_pick": state.current_pick,
        "on_the_clock": state.on_the_clock,
        "picks_until_my_turn": picks,
        "ranked_by": key,
        "risk_weight": round(weight, 3),
        "needs": needs,
        "candidates": [c.__dict__ for c in allowed],
        "blocked_by_rules": [c.__dict__ for c in blocked],
        "rule_cost": rule_cost,
        "cost_of_waiting": waiting,
        "room": room,
        "positional_read": read,
        "plan": plan,
        "roster_pressure": pressure,
        "recommendation": pick,
        "context": dict(board.context_from),
    }


def _recommend(
    allowed: list[Candidate],
    read: dict,
    plan: dict,
    rules: dict,
    key: str,
    pressure: dict | None = None,
) -> dict | None:
    """One pick, with the rule that produced it named.

    The bot used to assemble this itself from three separate fields, which is
    how "take the best score" quietly became "take a forward again". The order
    below is the whole strategy, and it is here so there is one copy of it:

      1. an open starting slot with no picks left to spare,
      2. a position draining fast enough to reach for,
      3. an open starting slot, unless a bench player beats it by the margin,
      4. the job this pick was given by the plan,
      5. the best score left.

    The deadline rule is first because it is the only one that knows the draft
    ends. Everything below it compares value, and on value a deep position is
    always worth deferring - which is how a blue line stays empty while the
    bench fills with wingers.
    """
    if not allowed:
        return None
    best = allowed[0]
    pressure = pressure or {}
    margin = pressure.get("need_first_margin", float(rules["need_first_margin"]))
    fillers = [c for c in allowed if c.fills_a_need]

    # No slack: every remaining pick is claimed by a slot that would otherwise
    # score zero all season, and no gap in projected points buys one back.
    if margin is None and fillers:
        pick = max(fillers, key=lambda c: getattr(c, key))
        return {
            "player_id": pick.player_id,
            "player": pick.player,
            "why": f"needs first -> {pick.need_slot}",
            "detail": (
                f"{pressure.get('reason', '')}, so this pick has to fill one. "
                f"{pick.player} is the best of them"
            ),
        }

    if read.get("position") and read.get("take"):
        return {
            "player_id": read["take"]["player_id"],
            "player": read["take"]["player"],
            "why": f"positional_read -> {read['position']}",
            "detail": read["reason"],
        }

    # Needs first. A starting slot left open while the bench fills up is the
    # logjam: the blue line ends up drafted from whatever is left in the last
    # rounds. The margin is what "unless the forwards are insanely better"
    # means as a number.
    if not best.fills_a_need:
        filler = fillers[0] if fillers else None
        if filler is not None and margin is not None:
            gap = getattr(best, key) - getattr(filler, key)
            if gap < margin:
                return {
                    "player_id": filler.player_id,
                    "player": filler.player,
                    "why": f"needs first -> {filler.need_slot}",
                    "detail": (
                        f"{best.player} scores {gap:.0f} more but fills no open slot; "
                        f"{filler.player} fills {filler.need_slot}. Over {margin:.0f} "
                        f"the bench player would win."
                    ),
                }
            return {
                "player_id": best.player_id,
                "player": best.player,
                "why": "need overridden",
                "detail": (
                    f"{best.player} fills no open slot but beats the best filler "
                    f"({filler.player}) by {gap:.0f}, over the {margin:.0f} margin "
                    f"({pressure.get('reason', 'no deadline pressure')})"
                ),
            }

    if plan["pick_kind"] == "schedule":
        # A player who cannot play cannot start, so the one job this pick has is
        # one he cannot do.
        fit = [
            c
            for c in allowed
            if c.schedule_opening and not (c.injury and c.injury.get("status") in CANNOT_PLAY)
        ]
        if fit:
            pick = max(fit, key=lambda c: (c.schedule_opening["starts"], getattr(c, key)))
            if pick.schedule_opening["starts"] > 0:
                return {
                    "player_id": pick.player_id,
                    "player": pick.player,
                    "why": "plan -> schedule",
                    "detail": (
                        f"{pick.schedule_opening['starts']} start(s) of "
                        f"{pick.schedule_opening['games']} game(s) in the opening weeks, "
                        f"against my roster as it stands"
                    ),
                }

    if plan["pick_kind"] == "ceiling":
        pick = max(allowed, key=lambda c: c.p80 - (c.mean - c.vorp))
        return {
            "player_id": pick.player_id,
            "player": pick.player,
            "why": "plan -> ceiling",
            "detail": f"highest p80 over replacement left ({pick.p80:.0f} at the 80th)",
        }

    return {
        "player_id": best.player_id,
        "player": best.player,
        "why": "best score",
        "detail": f"top of the shortlist by {key}",
    }


def cost_of_waiting(board, state, valued: pd.DataFrame) -> tuple[dict, dict]:
    """What passing on each position costs by my next turn, and how that was read.

    It used to assume the next picks come straight off the top of our value
    board. The room does not draft off our board - it has never seen it - and
    measured against the saved mock rooms that assumption priced waiting on C,
    LW, RW and G 25-50 points too high on average, always on the same side, while
    reading defence as free for thirteen rounds because no defenceman sat in the
    top dozen. `hockey.serve.room` has the measurement and the model that
    replaced it: the room drafts in its own order toward its open slots, and
    each cost is the expected drop over a few hundred simulated rooms.

    Returns the per-position costs and a description of the room they came from,
    including whether each team's open slots were read. Without a room model
    every cost is None, which `positional_read` treats as no reason to reach.
    """
    from hockey.serve.room import SIMS, expected_after, open_slots

    positions = [p for p in board.roster_shape if p in board.slots]
    room = getattr(board, "room", None)
    if room is None:
        return dict.fromkeys(positions), {
            "source": None,
            "reason": "no room model loaded, so what the room takes cannot be predicted",
        }
    seats = state.seats_before_my_turn()
    check, detail = state.order_check()
    openings = None
    if check != "inconsistent":
        rosters = state.rosters_by_seat()
        players = board.players.set_index(board.players["player_id"].astype(int))
        openings = {}
        for seat in set(seats):
            held = [
                (pid, float(players.at[pid, "mean"]), _eligible(players.loc[pid]))
                for pid in rosters.get(seat, [])
                if pid in players.index
            ]
            openings[seat] = open_slots(held, board.roster_shape)
    costs = expected_after(
        valued, positions, seats, room, openings, sims=SIMS, seed=state.total_picks
    )
    return costs, {
        **room.describe(),
        "sims": SIMS,
        "picks_until_my_turn": len(seats),
        "seats_before_my_turn": seats,
        "needs": (
            f"read from each team's roster ({check}: {detail})"
            if openings is not None
            else f"not read - the observed order is {check} ({detail})"
        ),
    }


def _eligible(row) -> tuple[str, ...]:
    got = row["eligible"]
    return tuple(got) if isinstance(got, tuple | list) else (str(row["slot"]),)
