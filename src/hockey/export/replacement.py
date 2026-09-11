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


def replacement_levels(
    board: pd.DataFrame, slots: dict[str, float], window: int = 6, column: str = "mean"
) -> pd.DataFrame:
    """The baseline at each position: what the next player off the board scores.

    Averaged over a window of players rather than read off the single player at
    the cutoff, because one player's projection is noisy and the baseline it
    sets would propagate that noise into every value at the position.
    """
    rows = []
    for position, group in board.groupby("position"):
        ordered = group.sort_values(column, ascending=False).reset_index(drop=True)
        cutoff = slots.get(position)
        if cutoff is None:
            continue
        start = int(round(cutoff))
        window_rows = ordered.iloc[start : start + window]
        if window_rows.empty:
            # The pool does not reach replacement at this position, so the
            # baseline would be extrapolated. Say so rather than quietly using
            # the worst player in the pool, which would understate the gap.
            logger.warning(
                "pool of %d %s ends before replacement level (%d); "
                "value over replacement at this position is a lower bound",
                len(ordered),
                position,
                start,
            )
            window_rows = ordered.tail(window)
        rows.append(
            {
                "position": position,
                "drafted": start,
                "pool": len(ordered),
                "replacement": float(window_rows[column].mean()),
                "starter_cutoff": float(ordered[column].iloc[: start or 1].min()),
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
    out["replacement"] = out["position"].map(baseline)
    out["vorp"] = out[column] - out["replacement"]
    out["pos_rank"] = (
        out.groupby("position")[column].rank(ascending=False, method="min").astype(int)
    )
    return out.sort_values("vorp", ascending=False).reset_index(drop=True)


def scarcity(board: pd.DataFrame, levels: pd.DataFrame, depth: int = 60) -> pd.DataFrame:
    """The drop-off curve: projected points by rank within each position.

    This is the shape a drafter reads to decide when a run starts. A position
    whose curve falls off a cliff at rank 12 has to be taken early; one that
    stays flat to rank 40 can wait, however good its best player is.
    """
    frames = []
    for position, group in board.groupby("position"):
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
    position_of = dict(zip(board["player_id"], board["position"], strict=True))
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
