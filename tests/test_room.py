"""The room model behind cost_of_waiting.

The first version assumed the other managers draft straight down our value
board. In a real mock that read the cost of waiting on defence as exactly 0.0
for thirteen rounds - no defenceman sat in the top dozen of our board, so a
straight run could never take one - while pricing waiting on forwards and
goalies 25-50 points too high, because the room does not draft off a board it
has never seen. These pin both halves of that, and the pieces the fix rests on.
"""

import json
import math

import numpy as np
import pandas as pd
import pytest

from hockey.serve.room import (
    Room,
    RoomModel,
    _newton,
    average_draft_position,
    expected_after,
    rooms_from_logs,
)
from hockey.serve.state import DraftState

# Slot 8 of 14 in round 1: the twelve picks between my first turn and my second.
ROUND_ONE_GAP = [9, 10, 11, 12, 13, 14, 14, 13, 12, 11, 10, 9]


def pool_of(rows: list[tuple[int, str, float]]) -> pd.DataFrame:
    """(player_id, position, vorp) -> the available pool the server holds."""
    return pd.DataFrame(
        {
            "player_id": [r[0] for r in rows],
            "player": [f"player {r[0]}" for r in rows],
            "eligible": [(r[1],) for r in rows],
            "slot": [r[1] for r in rows],
            "vorp": [r[2] for r in rows],
            "mean": [r[2] + 250.0 for r in rows],
        }
    )


def room_of(adp: dict[int, float], need_weight: float = 0.7) -> RoomModel:
    # Coefficients near the fitted ones; off the board effectively never, so
    # every pick in these tests is somebody on the board.
    return RoomModel(
        order_weight=7.5, need_weight=need_weight, off_board=-80.0, adp=adp, undrafted=240.0
    )


def forwards_over_defence():
    """Our board: fourteen wingers on top, twenty defencemen below them all.

    The room usually takes these defencemen around picks 100-138 and these
    wingers from 150 on - the room's order, not ours.
    """
    wingers = [(i, "LW", 270.0 - 5 * i) for i in range(14)]
    defence = [(100 + i, "D", 160.0 - 5 * i) for i in range(20)]
    adp = {i: 150.0 + i for i in range(14)} | {100 + i: 100.0 + 2 * i for i in range(20)}
    return pool_of(wingers + defence), adp


def test_a_position_below_the_top_of_our_board_still_costs_something_to_wait_on():
    """The flat zero, pinned so it cannot come back.

    Every defenceman is below the fourteenth winger on our board, so twelve
    picks straight off the top take none of them and waiting on D reads as
    free. But the twelve teams picking next have filled their wings and need
    defence, and the room takes these defencemen now.
    """
    pool, adp = forwards_over_defence()
    top_twelve = set(pool.sort_values("vorp", ascending=False).head(12)["player_id"])
    assert not top_twelve & set(pool.loc[pool["slot"] == "D", "player_id"])  # the old trap

    openings = {s: {"LW": 0, "D": 4} for s in set(ROUND_ONE_GAP)}
    out = expected_after(pool, ["LW", "D"], ROUND_ONE_GAP, room_of(adp), openings)

    assert out["D"]["cost"] > 10
    assert out["D"]["p_best_survives"] < 0.5
    assert out["D"]["best_after"] < out["D"]["best_now"]


def test_the_top_of_our_board_is_not_gone_when_the_room_does_not_want_it():
    """The other half: the straight run had our best wingers gone - a 65-point
    cost of waiting here - when the room was never going to take them yet."""
    pool, adp = forwards_over_defence()
    openings = {s: {"LW": 2, "D": 4} for s in set(ROUND_ONE_GAP)}
    out = expected_after(pool, ["LW", "D"], ROUND_ONE_GAP, room_of(adp), openings)

    assert out["LW"]["cost"] < 3
    assert out["LW"]["p_best_survives"] > 0.8


def test_a_team_missing_a_position_drafts_it():
    """The same room drains defence faster when the teams picking need it."""
    wingers = [(i, "LW", 200.0 - 3 * i) for i in range(20)]
    defence = [(100 + i, "D", 150.0 - 3 * i) for i in range(20)]
    adp = {i: 100.0 + 2 * i for i in range(20)} | {100 + i: 101.0 + 2 * i for i in range(20)}
    pool = pool_of(wingers + defence)
    need_d = {s: {"LW": 0, "D": 4} for s in set(ROUND_ONE_GAP)}
    need_w = {s: {"LW": 2, "D": 0} for s in set(ROUND_ONE_GAP)}
    room = room_of(adp, need_weight=1.5)

    wants_d = expected_after(pool, ["LW", "D"], ROUND_ONE_GAP, room, need_d)
    wants_w = expected_after(pool, ["LW", "D"], ROUND_ONE_GAP, room, need_w)

    assert wants_d["D"]["cost"] > wants_w["D"]["cost"] + 5
    assert wants_w["LW"]["cost"] > wants_d["LW"]["cost"] + 5


def test_one_board_state_gives_one_answer():
    # A pick decided on one reading must be reproducible from the log.
    pool, adp = forwards_over_defence()
    room = room_of(adp)
    first = expected_after(pool, ["LW", "D"], ROUND_ONE_GAP, room, None, seed=42)
    again = expected_after(pool, ["LW", "D"], ROUND_ONE_GAP, room, None, seed=42)
    assert first == again


def test_no_picks_in_between_costs_nothing_and_an_empty_position_is_none():
    # Slot 14 picks twice at the turn, so there can be nothing in between.
    pool, adp = forwards_over_defence()
    out = expected_after(pool, ["LW", "D", "G"], [], room_of(adp))
    assert out["D"]["cost"] == 0.0 and out["D"]["p_best_survives"] == 1.0
    assert out["G"] is None  # nobody on the board plays it


# --- fitting the room ---------------------------------------------------------


def test_the_average_pick_leaves_out_what_the_bot_took():
    """A player the bot took was never offered to the rest of that room.

    Counting him as undrafted there would push the room's order for every
    player our board likes later than the room actually takes them.
    """
    rooms = [
        Room("a", [10, 20, 30, 40], bot_seat=1),  # the bot took 10 (pick 1) and 40 (pick 4)
        Room("b", [20, 10, 30, 50], bot_seat=None),
    ]
    adp, undrafted, _ = average_draft_position(rooms, n_teams=2)
    assert undrafted == 6.0  # a round past the longest room
    assert adp[10] == 2.0  # room b only, not (undrafted + 2) / 2
    assert adp[20] == 1.5
    assert adp[50] == (undrafted + 4) / 2  # never taken in room a, which saw him


def test_the_solver_recovers_the_room_it_was_given():
    rng = np.random.default_rng(7)
    truth = np.array([2.0, 0.8, -3.0])
    situations = []
    for _ in range(1000):
        n = 25
        x = np.zeros((n + 1, 3))
        x[:-1, 0] = -np.log(rng.uniform(1, 200, n))
        x[:-1, 1] = rng.integers(0, 2, n)
        x[-1, 2] = 1.0
        u = x @ truth + rng.gumbel(size=n + 1)
        situations.append((x, int(u.argmax())))
    theta, loglik = _newton(situations)
    assert math.isfinite(loglik)
    assert np.allclose(theta, truth, atol=0.25), theta


def _log(directory, names, mine):
    directory.mkdir(parents=True)
    lines = [{"path": "/draft/observed", "payload": {"names": names[:10]}, "response": {}}]
    lines.append({"path": "/draft/observed", "payload": {"names": names}, "response": {}})
    for pid in mine:
        lines.append({"path": "/draft/mine", "payload": {}, "response": {"player_id": pid}})
    (directory / "api.jsonl").write_text("\n".join(json.dumps(x) for x in lines), "utf-8")


# Letters only: names are folded to letters before matching, and two capitals
# at the end would be read as a team code and stripped.
NAMES = [f"Skater {chr(65 + i // 26)}{chr(97 + i % 26)}" for i in range(40)]


def _board():
    from hockey.serve.identity import Resolver

    board = pd.DataFrame({"player_id": range(1, 41), "player": NAMES, "slot": "C"})
    return Resolver(board)


def test_saved_rooms_are_read_checked_and_deduplicated(tmp_path):
    names = list(NAMES)  # player i + 1 went at pick i + 1
    # Four teams, seat 2: picks 2, 7, 10, 15, ...
    mine = [2, 7, 10, 15]
    _log(tmp_path / "20260101-live", names, mine)
    _log(tmp_path / "20260102-live", names[:36], mine)  # the same room, a shorter read
    _log(tmp_path / "20260103-sim-slot2", names, mine)
    _log(tmp_path / "20260104-live", names[1:] + names[:1], mine)  # out of order

    rooms, skipped = rooms_from_logs(tmp_path, _board(), n_teams=4)

    assert [r.name for r in rooms] == ["20260101-live"]
    assert rooms[0].bot_seat == 2
    why = dict(skipped)
    assert "simulated" in why["20260103-sim-slot2"]
    assert "same room" in why["20260102-live"]
    assert "seats" in why["20260104-live"]


def test_a_name_that_will_not_resolve_is_a_pick_off_the_board_not_a_guess(tmp_path):
    names = list(NAMES)
    names[4] = "A. Nobody"
    _log(tmp_path / "20260101-live", names, [])
    rooms, _ = rooms_from_logs(tmp_path, _board(), n_teams=4)
    assert rooms[0].picks[4] is None
    assert rooms[0].picks[5] == 6  # everyone after keeps his place


# --- the state the room model reads -----------------------------------------


def test_an_ordered_observation_numbers_every_pick_and_seats_it():
    """Pick 3 could not be named, and still holds its place in the order."""
    state = DraftState(slot=2, n_teams=4)
    order = {11: 1, 12: 2, 14: 4, 15: 5, 13: 6}
    state.reconcile([11, 12, 14, 15, 13], mine=[12], unidentified=1, order=order)

    assert state.rosters_by_seat() == {1: [11], 2: [12], 4: [14, 15], 3: [13]}
    assert state.order_check()[0] == "consistent"


def test_an_order_that_puts_my_pick_on_another_seat_is_not_believed():
    state = DraftState(slot=2, n_teams=4)
    state.reconcile([11, 12], mine=[12], order={12: 1, 11: 2})
    status, detail = state.order_check()
    assert status == "inconsistent" and "seat" in detail


def test_the_seats_ahead_follow_the_snake():
    state = DraftState(slot=8, n_teams=14)
    state.reconcile(list(range(100, 107)))  # seven gone: pick 8, mine, on the clock
    assert state.on_the_clock
    assert state.seats_before_my_turn() == ROUND_ONE_GAP

    state.reconcile(list(range(100, 103)))  # three gone: pick 4, not mine
    assert state.seats_before_my_turn() == [4, 5, 6, 7]


def test_without_a_room_model_there_is_no_cost_and_no_reach():
    """No model of the room is no claim about the room - not a quiet fallback
    to the straight run that was measured wrong."""
    from hockey.serve.recommend import cost_of_waiting, positional_read

    class Board:
        roster_shape = {"C": 2, "D": 4}
        slots = {"C": 42.0, "D": 84.0}
        room = None

    costs, room = cost_of_waiting(Board(), DraftState(), pool_of([(1, "C", 100.0)]))
    assert costs == {"C": None, "D": None}
    assert room["source"] is None
    read = positional_read(costs, {"C": 1, "D": 4}, 12)
    assert read["position"] is None and "no room model" in read["reason"]


@pytest.mark.parametrize("gap", [ROUND_ONE_GAP, [1, 2, 3]])
def test_the_expected_cost_is_never_negative(gap):
    pool, adp = forwards_over_defence()
    out = expected_after(pool, ["LW", "D"], gap, room_of(adp))
    assert all(v["cost"] >= 0 for v in out.values())
