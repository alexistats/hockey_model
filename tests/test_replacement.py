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
