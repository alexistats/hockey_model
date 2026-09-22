"""Positional strength. The point of these is that a pick is worth the gap to
replacement, not the raw total, and the gap depends on the roster."""

import numpy as np
import pandas as pd
import pytest

from hockey.export import (
    add_value_over_replacement,
    replacement_levels,
    replacement_slots,
    scarcity,
    tiers,
)

ROSTER = [
    {"position": "C", "count": 2, "starting": True},
    {"position": "LW", "count": 2, "starting": True},
    {"position": "RW", "count": 2, "starting": True},
    {"position": "D", "count": 4, "starting": True},
    {"position": "G", "count": 2, "starting": True},
    {"position": "BN", "count": 5, "starting": False},
    {"position": "IR+", "count": 2, "starting": False},
]


def board_of(position: str, means: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [hash((position, m)) % 10**6 for m in means],
            "player": [f"{position}{i}" for i in range(len(means))],
            "position": position,
            "mean": means,
            "floor": [m - 50 for m in means],
            "ceiling": [m + 50 for m in means],
        }
    )


def test_bench_slots_deepen_the_drafted_pool():
    """Replacement is not the last starter. A league that benches five more
    players per team drafts well past its starting lineup, and a baseline set
    at the starter cutoff would overstate every player's value."""
    slots = replacement_slots(ROSTER, n_teams=14)
    assert slots["C"] > 2 * 14
    assert slots["D"] > 4 * 14
    # Bench is charged in proportion to starts, so D absorbs twice C's share.
    assert slots["D"] - 4 * 14 == pytest.approx(2 * (slots["C"] - 2 * 14))
    # Goalies keep exactly their starting slots: the bench went to skaters.
    assert slots["G"] == 2 * 14


def test_streaming_one_position_changes_nothing_but_the_argument():
    without = replacement_slots(ROSTER, n_teams=14, bench_to_skaters=False)
    assert without["C"] == 2 * 14
    assert without["D"] == 4 * 14


def test_the_same_projection_is_worth_more_at_the_scarce_position():
    """The whole reason this module exists. Two players project identically;
    one plays a position whose pool runs dry earlier."""
    deep = board_of("C", list(np.linspace(500, 300, 80)))
    thin = board_of("D", list(np.linspace(500, 420, 80)))
    board = pd.concat([deep, thin], ignore_index=True)
    slots = {"C": 40, "D": 40}
    levels = replacement_levels(board, slots, window=6)
    valued = add_value_over_replacement(board, levels)

    c_top = valued[(valued["position"] == "C") & (valued["mean"] == 500)].iloc[0]
    d_top = valued[(valued["position"] == "D") & (valued["mean"] == 500)].iloc[0]
    assert c_top["mean"] == d_top["mean"]
    assert c_top["vorp"] > d_top["vorp"]


def test_value_reorders_the_board_against_raw_points():
    board = pd.concat([board_of("C", [600, 500]), board_of("D", [560, 400])], ignore_index=True)
    levels = pd.DataFrame({"position": ["C", "D"], "replacement": [450.0, 300.0]})
    valued = add_value_over_replacement(board, levels)
    # By points the 560 defenceman is second; by value he is first.
    assert list(valued["mean"])[0] == 560
    assert list(valued["vorp"]) == sorted(valued["vorp"], reverse=True)


def test_a_position_with_no_roster_slot_is_an_error_not_a_zero():
    """Silently valuing an unrostered position against nothing would produce a
    complete, plausible board with a whole position mispriced."""
    board = board_of("W", [500, 400])
    levels = pd.DataFrame({"position": ["C"], "replacement": [300.0]})
    with pytest.raises(KeyError, match="no replacement level"):
        add_value_over_replacement(board, levels)


def test_a_pool_too_shallow_to_reach_replacement_warns(caplog):
    board = board_of("G", [500, 480, 460])
    with caplog.at_level("WARNING"):
        levels = replacement_levels(board, {"G": 28}, window=6)
    assert "lower bound" in caplog.text
    assert levels["replacement"].iloc[0] == pytest.approx(480.0)


def test_scarcity_curves_measure_the_drop_off_per_position():
    board = pd.concat(
        [board_of("C", [600, 550, 500]), board_of("D", [500, 495, 490])],
        ignore_index=True,
    )
    levels = pd.DataFrame({"position": ["C", "D"], "replacement": [400.0, 450.0]})
    curve = scarcity(board, levels, depth=3)
    c = curve[curve["position"] == "C"].sort_values("pos_rank")
    d = curve[curve["position"] == "D"].sort_values("pos_rank")
    assert list(c["pos_rank"]) == [1, 2, 3]
    # Centre falls 100 points over three ranks, defence falls 10.
    assert c["mean"].iloc[0] - c["mean"].iloc[-1] > d["mean"].iloc[0] - d["mean"].iloc[-1]
    assert c["above_replacement"].iloc[0] == pytest.approx(200.0)


def test_tiers_break_on_overlap_not_on_point_gaps():
    """Two players far apart in points but hugely uncertain stay in one tier;
    two players close in points but tight break into two. A gap-based tiering
    gets both backwards."""
    rng = np.random.default_rng(0)
    wide_a = rng.normal(500, 120, 4000)
    wide_b = rng.normal(460, 120, 4000)
    tight_a = rng.normal(500, 8, 4000)
    tight_b = rng.normal(480, 8, 4000)

    board = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4],
            "position": ["C", "C", "D", "D"],
            "mean": [500.0, 460.0, 500.0, 480.0],
        }
    )
    draws = np.column_stack([wide_a, wide_b, tight_a, tight_b])
    assigned = tiers(board, draws, [1, 2, 3, 4])
    assert assigned[1] == assigned[2] == 1  # 40 points apart, still one tier
    assert assigned[3] == 1 and assigned[4] == 2  # 20 apart, a real break


# --- Yahoo position eligibility -------------------------------------------
#
# The league fills a slot from everyone eligible for it, not from everyone the
# NHL happens to list there. Two ways of getting that wrong are easy and both
# were written before these tests existed: pooling every eligible player per
# position double-counts the ones another position will take, and valuing each
# player at whatever position has the lowest baseline sends all of them to the
# same one. Capacity is what rules both out.


def eligible_board(rows: list[tuple[str, tuple[str, ...], float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": range(1, len(rows) + 1),
            "player": [f"p{i}" for i in range(len(rows))],
            "position": [r[0] for r in rows],
            "eligible": [r[1] for r in rows],
            "mean": [r[2] for r in rows],
            "floor": [r[2] - 50 for r in rows],
            "ceiling": [r[2] + 50 for r in rows],
        }
    )


def test_fill_slots_never_exceeds_the_slots_that_exist():
    from hockey.export.replacement import fill_slots

    # Everyone can play both wings, so a naive pool would count each of them
    # twice and both positions would look twice as deep as the league is.
    board = eligible_board([("LW", ("LW", "RW"), 500 - 10 * i) for i in range(40)])
    assigned, leftover = fill_slots(board, {"LW": 10, "RW": 10})

    from collections import Counter

    filled = Counter(assigned.values())
    assert filled == {"LW": 10, "RW": 10}
    assert len(assigned) + len(leftover) == len(board)


def test_replacement_reads_off_who_is_actually_left_over():
    from hockey.export.replacement import replacement_levels as levels_of

    # 10 slots each, 30 players, all dual eligible. 20 get taken, so the
    # baseline must come from the 21st onward - not from the 11th, which is
    # what pooling each position separately would have used.
    board = eligible_board([("LW", ("LW", "RW"), 300 - 10 * i) for i in range(30)])
    levels = levels_of(board, {"LW": 10, "RW": 10}, window=2)
    baseline = dict(zip(levels["position"], levels["replacement"], strict=True))
    # ranks 21 and 22 are means 100 and 90
    assert baseline["LW"] == pytest.approx(95.0)
    assert baseline["RW"] == pytest.approx(95.0)


def test_dual_eligibility_is_worth_at_least_single_eligibility():
    from hockey.export.replacement import replacement_levels as levels_of

    # Same projection, but one of them can also fill the scarcer position.
    shared = [("RW", ("RW",), 400 - i) for i in range(30)]
    scarce = [("C", ("C",), 200 - i) for i in range(30)]
    rows = [*shared, *scarce, ("RW", ("RW",), 350.0), ("RW", ("RW", "C"), 350.0)]
    board = eligible_board(rows)
    levels = levels_of(board, {"C": 10, "RW": 10}, window=3)
    valued = add_value_over_replacement(board, levels)

    single = valued[valued["player"] == f"p{len(rows) - 2}"].iloc[0]
    dual = valued[valued["player"] == f"p{len(rows) - 1}"].iloc[0]
    assert dual["mean"] == single["mean"]
    assert dual["vorp"] >= single["vorp"]
    # and the slot it was valued at is the one that earned the extra
    assert dual["slot"] == "C"


def test_a_board_without_eligibility_is_unchanged():
    # The column is optional, and a board that predates it must value exactly
    # as it did before - otherwise every saved run silently changes meaning.
    board = pd.concat(
        [
            board_of("C", [500 - 10 * i for i in range(30)]),
            board_of("D", [400 - 10 * i for i in range(30)]),
        ],
        ignore_index=True,
    )
    slots = {"C": 10, "D": 10}
    levels = replacement_levels(board, slots, window=3)
    valued = add_value_over_replacement(board, levels)
    assert list(valued["slot"]) == list(valued["position"])
    baseline = dict(zip(levels["position"], levels["replacement"], strict=True))
    # ranks 11-13 at C are 400, 390, 380
    assert baseline["C"] == pytest.approx(390.0)


def test_a_shallow_pool_averages_across_the_cutoff_rather_than_one_survivor():
    """The bug this was written for.

    There are 42 centre slots in this league and about 51 centre-eligible
    players, so centre runs out of freely available players mid-draft. The
    baseline was then read off however few survived - at one point a single
    player - and a three-point edge built on a one-player baseline decided a
    third-round pick. The window now reaches back across the cutoff into the
    cheapest starters, so it always averages `window` players.
    """
    board = board_of("C", [100 - i for i in range(12)])  # 100 down to 89
    levels = replacement_levels(board, {"C": 10}, window=4)
    row = levels.iloc[0]
    assert row["free_below"] == 2  # only 90 and 89 are genuinely free
    # 92 and 91 are the cheapest starters, so the window is 92, 91, 90, 89.
    assert row["replacement"] == pytest.approx(90.5)
    assert bool(row["extrapolated"])


def test_the_baseline_never_jumps_up_as_the_board_drains():
    """It used to. When the last free player was absorbed the estimator
    switched to the six worst players in the whole pool, which were better
    than the survivors it replaced: the centre baseline rose 23 points
    mid-draft and then froze, so every centre's value fell for no reason on
    the board."""
    board = board_of("C", [100 - i for i in range(20)])
    slots = {"C": 12}
    seen = []
    for gone in range(0, 12):
        live = board.sort_values("mean", ascending=False).iloc[gone:]
        seen.append(float(replacement_levels(live, slots, window=4)["replacement"].iloc[0]))
    for before, after in zip(seen, seen[1:], strict=False):
        assert after <= before + 1e-9, f"baseline rose from {before} to {after}"


def test_free_below_says_how_many_players_actually_set_the_baseline():
    deep = replacement_levels(board_of("C", [100 - i for i in range(20)]), {"C": 4}, window=4)
    assert deep["free_below"].iloc[0] == 16
    assert not bool(deep["extrapolated"].iloc[0])
