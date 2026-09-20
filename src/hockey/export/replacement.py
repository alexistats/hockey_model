"""Positional strength: what a player is worth *given* what else is available.

Fantasy points rank players. They do not rank draft picks, because a pick is
not a claim on a player's total - it is a claim on the difference between that
player and whoever would fill the slot otherwise. Under this league's roster
the pools are wildly uneven: 14 teams start four defencemen each and two
centres each, so the 56th-best defenceman is a starter and the 40th-best centre
is not. A 480-point centre and a 480-point defenceman are the same projection
and not remotely the same pick.

This module is deliberately outside the model. Nothing here changes a
posterior; it re-reads one against the shape of the league. That separation is
what lets the roster change - a keeper rule, a different bench depth, a plan to
stream one position - without refitting anything.

One limitation to state plainly, because it is invisible in the output: the
position used here is the player's NHL primary position, not their Yahoo
eligibility. Yahoo lists many players at two positions, and a centre who is
also left-wing eligible is genuinely worth more than this reckons. When the
Yahoo crosswalk is working, eligibility should replace position and the same
functions apply unchanged.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Bench slots are not tied to a position, so charging them to one is a
# judgement. Teams bench roughly in proportion to how many of that position
# they start - a manager carrying four defence slots carries spare defencemen -
# so the default spreads the bench across the starting skater slots. It is a
# parameter because a drafter who plans to stream one position should say so.
SKATER_POSITIONS = ("C", "LW", "RW", "D")


def replacement_slots(
    roster: list[dict], n_teams: int, bench_to_skaters: bool = True
) -> dict[str, float]:
    """How many players at each position the league absorbs before replacement.

    The answer is not the starting slots alone. A league that starts 28 centres
    and benches 70 more players does not have its replacement centre at 28; the
    drafted pool runs deeper than the starting lineup, and a player just outside
    it is what "freely available" actually means.
    """
    starting = {r["position"]: r["count"] for r in roster if r.get("starting")}
    bench = sum(r["count"] for r in roster if r["position"] == "BN")
    slots = {p: float(n) for p, n in starting.items()}
    if bench_to_skaters and bench:
        skater_starts = sum(slots.get(p, 0.0) for p in SKATER_POSITIONS)
        for p in SKATER_POSITIONS:
            if p in slots:
                slots[p] += bench * slots[p] / skater_starts
    return {p: n * n_teams for p, n in slots.items()}


def _eligible(row, fallback: str) -> tuple[str, ...]:
    got = getattr(row, "eligible", None)
    return tuple(got) if got else (fallback,)


def fill_slots(
    board: pd.DataFrame, slots: dict[str, float], column: str = "mean"
) -> tuple[dict[int, str], pd.DataFrame]:
    """Hand out the league's slots best first, and return who is left over.

    Dual eligibility cannot be handled by asking each position who could fill it
    and counting to the cutoff. Those pools overlap, so every one of them counts
    players the others will take: ask who could play right wing and you get 106
    names, but 42 right-wing slots are not filled from the best 42 of them,
    because most are away filling centre and left wing. Reading replacement off
    that inflated pool understates value at exactly the positions where dual
    eligibility is commonest - measured here, right wing's baseline came out 100
    points high and defencemen took 10 of the top 25 as a result.

    Nor can each player simply be valued at whichever position has the lowest
    baseline. That is the same answer for everybody, so every dual-eligible
    player stampedes into one position and the others are left short.

    Capacity is what both miss, so this spends it: walk the board best first,
    give each player an open slot they are eligible for, and prefer the position
    with the most still open, which keeps a position from being starved by a
    run on it. When the slots are gone, whoever remains is by definition freely
    available - and that is what replacement level means.
    """
    openings = {p: int(round(n)) for p, n in slots.items()}
    assigned: dict[int, str] = {}
    for row in board.sort_values(column, ascending=False).itertuples():
        open_to = [p for p in _eligible(row, row.position) if openings.get(p, 0) > 0]
        if not open_to:
            continue
        pick = max(open_to, key=lambda p: openings[p])
        openings[pick] -= 1
        assigned[int(row.player_id)] = pick
    leftover = board[~board["player_id"].astype(int).isin(assigned)]
    return assigned, leftover


def replacement_levels(
    board: pd.DataFrame, slots: dict[str, float], window: int = 6, column: str = "mean"
) -> pd.DataFrame:
    """The baseline at each position: what the next player off the board scores.

    Averaged over a window of players rather than read off the single player at
    the cutoff, because one player's projection is noisy and the baseline it
    sets would propagate that noise into every value at the position.
    """
    assigned, leftover = fill_slots(board, slots, column)
    leftover = leftover.sort_values(column, ascending=False)
    taken = pd.Series(assigned)

    rows = []
    for position in sorted(slots):
        free = leftover[[position in _eligible(r, r.position) for r in leftover.itertuples()]]
        if free.empty:
            # Every eligible player was absorbed, so the baseline would have to
            # be extrapolated past the end of the pool. Say so rather than
            # quietly inventing one, which would overstate value here.
            logger.warning(
                "every %s-eligible player fits in a slot; value over replacement "
                "at this position is a lower bound",
                position,
            )
            free = board[[position in _eligible(r, r.position) for r in board.itertuples()]]
            free = free.sort_values(column, ascending=False).tail(window)
        head = free.head(window)
        filled = taken[taken == position]
        starters = board[board["player_id"].astype(int).isin(filled.index)]
        rows.append(
            {
                "position": position,
                "drafted": len(filled),
                "pool": len(filled) + len(free),
                "replacement": float(head[column].mean()),
                "starter_cutoff": (
                    float(starters[column].min()) if len(starters) else float(head[column].mean())
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("replacement", ascending=False).reset_index(drop=True)


def add_value_over_replacement(
    board: pd.DataFrame, levels: pd.DataFrame, column: str = "mean"
) -> pd.DataFrame:
    """`vorp` and `pos_rank` on the board, re-ranked by value rather than points.

    `vorp` is the column a pick is actually made from. `mean` still orders the
    board by raw production, and the gap between the two orderings is exactly
    the positional scarcity a points-only list would miss.
    """
    baseline = dict(zip(levels["position"], levels["replacement"], strict=True))
    out = board.copy()
    missing = set(out["position"]) - set(baseline)
    if missing:
        raise KeyError(
            f"no replacement level for position(s) {sorted(missing)}; the roster "
            f"config has no slot for them, so their value cannot be measured "
            f"against anything"
        )
    # A player is worth what they are worth in the scarcest slot they can fill,
    # so value is taken at their best eligible position and `slot` records which
    # - a value with no position attached cannot be checked. This is a per-player
    # maximum, unlike the capacity-constrained pass that set the baselines, and
    # that is the point: the baseline is what the league can get instead, while
    # this is the best use this particular player can be put to.
    out["slot"] = [
        min(
            (p for p in _eligible(r, r.position) if p in baseline),
            key=lambda p: baseline[p],
            default=r.position,
        )
        for r in out.itertuples()
    ]
    out["replacement"] = out["slot"].map(baseline)
    out["vorp"] = out[column] - out["replacement"]
    out["pos_rank"] = out.groupby("slot")[column].rank(ascending=False, method="min").astype(int)
    return out.sort_values("vorp", ascending=False).reset_index(drop=True)


def scarcity(board: pd.DataFrame, levels: pd.DataFrame, depth: int = 60) -> pd.DataFrame:
    """The drop-off curve: projected points by rank within each position.

    This is the shape a drafter reads to decide when a run starts. A position
    whose curve falls off a cliff at rank 12 has to be taken early; one that
    stays flat to rank 40 can wait, however good its best player is.
    """
    # `slot` when eligibility assigned one, position otherwise - the curve has
    # to be the pool the drafter is choosing from, not the NHL's labelling.
    key = "slot" if "slot" in board else "position"
    frames = []
    for position, group in board.groupby(key):
        ordered = group.sort_values("mean", ascending=False).head(depth).reset_index(drop=True)
        frames.append(
            pd.DataFrame(
                {
                    "position": position,
                    "pos_rank": np.arange(1, len(ordered) + 1),
                    "mean": ordered["mean"].to_numpy(),
                    "floor": ordered["floor"].to_numpy(),
                    "ceiling": ordered["ceiling"].to_numpy(),
                }
            )
        )
    curve = pd.concat(frames, ignore_index=True)
    baseline = dict(zip(levels["position"], levels["replacement"], strict=True))
    curve["above_replacement"] = curve["mean"] - curve["position"].map(baseline)
    return curve


def tiers(board: pd.DataFrame, draws: np.ndarray, ids: list[int], threshold: float = 0.40):
    """Tier breaks within each position, from the distributions rather than gaps.

    A tier holds while the next player still has a real chance of outscoring the
    player who opened it. When that probability falls below `threshold` the
    remaining names are a genuine step down, and a drafter who waits through the
    break gets something materially worse. Point gaps alone cannot say this:
    two players five points apart are a coin flip or not depending entirely on
    how wide they are.
    """
    key = "slot" if "slot" in board else "position"
    position_of = dict(zip(board["player_id"], board[key], strict=True))
    mean_of = dict(zip(board["player_id"], board["mean"], strict=True))
    column_of = {pid: i for i, pid in enumerate(ids)}
    assignment = {}
    for position in sorted(set(position_of.values())):
        members = [p for p in ids if position_of.get(p) == position]
        members.sort(key=lambda p: mean_of[p], reverse=True)
        tier, anchor = 1, None
        for pid in members:
            column = draws[:, column_of[pid]]
            if anchor is None:
                anchor = column
            elif float((column > anchor).mean()) < threshold:
                tier += 1
                anchor = column
            assignment[pid] = tier
    return assignment
