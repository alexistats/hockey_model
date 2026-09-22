"""The draft API's guardrails.

These cover the places where a mistake would be silent during a live draft and
uncorrectable afterwards: a name matched to the wrong player, a pick counted
twice, a state that cannot recover from a missed poll.
"""

import pandas as pd
import pytest

from hockey.serve.identity import Resolver, clean, fold
from hockey.serve.state import DraftState


def board_of(rows: list[tuple[int, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [r[0] for r in rows],
            "player": [r[1] for r in rows],
            "slot": [r[2] for r in rows],
        }
    )


# --- identity ---------------------------------------------------------------


def test_yahoo_decoration_is_stripped_before_matching():
    # The draft board renders a name with team, position and status attached.
    assert clean("Connor McDavid EDM - C") == "Connor McDavid"
    assert clean("Jack Hughes NJ - C, LW") == "Jack Hughes"
    assert clean("Nikita Kucherov (Q)") == "Nikita Kucherov"
    assert clean("  Cale  Makar  ") == "Cale Makar"


def test_accents_fold():
    assert fold("Tim Stützle") == fold("Tim Stutzle")
    assert fold("Juraj Slafkovský") == fold("Juraj Slafkovsky")


def test_an_unknown_name_is_a_miss_not_a_guess():
    r = Resolver(board_of([(1, "Connor McDavid", "C")]))
    got = r.resolve("Fakey McNotaplayer")
    assert not got.ok
    assert got.method == "unmatched"


def test_two_players_with_one_name_need_a_position():
    """Vancouver has had two Elias Petterssons, a centre and a defenceman.

    Resolving that by picking either one is how a complete, plausible
    projection gets attached to the wrong player, and nothing downstream can
    tell. Without a position it must refuse.
    """
    r = Resolver(board_of([(1, "Elias Pettersson", "C"), (2, "Elias Pettersson", "D")]))
    blind = r.resolve("Elias Pettersson")
    assert not blind.ok
    assert blind.method == "ambiguous"
    assert set(blind.candidates) == {1, 2}

    with_position = r.resolve("Elias Pettersson", position="D")
    assert with_position.player_id == 2
    assert with_position.method == "position_tiebreak"


# --- draft state ------------------------------------------------------------


def test_the_same_player_observed_twice_is_one_pick():
    """The bug this was written for.

    Two spellings of one name both resolve, or two polls overlap, and the same
    id arrives twice in one observation. Appending a pick per occurrence
    inflates the pick count - and the pick count is what the round number and
    every "picks until my turn" answer derive from, so one duplicate read
    shifts the whole draft clock and never announces itself.
    """
    state = DraftState(slot=8, n_teams=14)
    result = state.reconcile([100, 200, 200, 300])
    assert result["total"] == 3
    assert len(state.picks) == 3
    assert state.current_pick == 4
    assert [p.overall for p in state.picks] == [1, 2, 3]


def test_a_missed_poll_heals_on_the_next_observation():
    # Deltas would be permanently wrong here; a full observation is not.
    state = DraftState()
    state.reconcile([1, 2])
    state.reconcile([1, 2, 3, 4, 5])  # picks 3 and 4 were never seen alone
    assert state.drafted == {1, 2, 3, 4, 5}
    assert len(state.picks) == 5


def test_a_player_that_disappears_is_dropped():
    # A misread name, or a pick that was undone. Silently keeping it would
    # remove a real player from the board for the rest of the draft.
    state = DraftState()
    state.reconcile([1, 2, 3])
    result = state.reconcile([1, 3])
    assert result["removed"] == [2]
    assert state.drafted == {1, 3}


def test_my_picks_survive_reconciliation():
    state = DraftState()
    state.take(42, by_me=True)
    state.reconcile([42, 7, 8])
    assert state.mine == [42]


def test_snake_turns_alternate_and_are_lopsided_at_the_ends():
    middle = DraftState(slot=8, n_teams=14)
    assert middle.my_pick_numbers()[:4] == [8, 21, 36, 49]
    turns = middle.my_pick_numbers()[:5]
    gaps = [b - a for a, b in zip(turns, turns[1:], strict=False)]
    # Alternating, because the order reverses every round. A fixed "two rounds"
    # is wrong in both directions.
    assert gaps == [13, 15, 13, 15]

    end = DraftState(slot=1, n_teams=14)
    assert end.my_pick_numbers()[:3] == [1, 28, 29]


def test_picks_until_my_turn_measures_to_the_following_pick_when_on_the_clock():
    """On the clock, the question is what passing costs - so the gap that
    matters is to my *next* pick, not zero."""
    state = DraftState(slot=1, n_teams=14)
    assert state.on_the_clock
    assert state.picks_until_my_turn() == 26  # pick 1, then pick 28

    state.reconcile(list(range(100, 100 + 5)))
    assert not state.on_the_clock
    assert state.picks_until_my_turn() == 22  # pick 6 now, mine is 28


def test_round_advances_with_the_pick_count():
    state = DraftState(slot=8, n_teams=14)
    assert state.current_round == 1
    state.reconcile(list(range(1000, 1000 + 14)))
    assert state.current_round == 2
    state.reconcile(list(range(1000, 1000 + 84)))
    assert state.current_round == 7


# --- rules ------------------------------------------------------------------


def test_the_goalie_rule_blocks_early_and_releases_on_schedule():
    from hockey.serve.recommend import DEFAULT_RULES, _rule_block

    class Row:
        slot = "G"

    empty = pd.DataFrame({"slot": []})
    early = DraftState(slot=8, n_teams=14)
    assert _rule_block(Row(), early, DEFAULT_RULES, empty) is not None

    late = DraftState(slot=8, n_teams=14)
    late.reconcile(list(range(2000, 2000 + 14 * 6)))  # into round 7
    assert late.current_round == 7
    assert _rule_block(Row(), late, DEFAULT_RULES, empty) is None


def test_a_second_goalie_is_blocked_until_the_late_rounds():
    from hockey.serve.recommend import DEFAULT_RULES, _rule_block

    class Row:
        slot = "G"

    holding_one = pd.DataFrame({"slot": ["G", "C"]})
    state = DraftState(slot=8, n_teams=14)
    state.reconcile(list(range(3000, 3000 + 14 * 7)))  # round 8
    assert state.current_round == 8
    assert _rule_block(Row(), state, DEFAULT_RULES, holding_one) is not None

    state.reconcile(list(range(3000, 3000 + 14 * 13)))  # round 14
    assert _rule_block(Row(), state, DEFAULT_RULES, holding_one) is None


@pytest.mark.parametrize("slot", [1, 5, 8, 14])
def test_every_pick_is_claimed_by_exactly_one_seat(slot):
    """A slot's turns must never collide with another's, or the bot would
    believe it is on the clock when it is not."""
    n = 14
    seats = {
        s: set(DraftState(slot=s, n_teams=n).my_pick_numbers(rounds=10)) for s in range(1, n + 1)
    }
    mine = seats[slot]
    for other, theirs in seats.items():
        if other != slot:
            assert not (mine & theirs), f"slot {slot} collides with slot {other}"
    assert len(mine) == 10


# --- risk posture and the positional read -----------------------------------


def test_the_risk_weight_ramps_from_cautious_to_ambitious():
    from hockey.serve.recommend import risk_weight

    assert risk_weight(1, 17, 0.2, 0.8) == pytest.approx(0.2)
    assert risk_weight(17, 17, 0.2, 0.8) == pytest.approx(0.8)
    assert risk_weight(9, 17, 0.2, 0.8) == pytest.approx(0.5)
    # Past the last round (a longer draft than the roster) it holds, not overshoots.
    assert risk_weight(25, 17, 0.2, 0.8) == pytest.approx(0.8)


def test_the_risk_score_is_measured_over_replacement_not_raw():
    """The bug this was written for: ranked by raw floor, a bot took every
    forward before any defenceman, because defencemen score fewer raw points.
    Over replacement, a steady defenceman can out-floor a volatile forward."""
    from hockey.serve.recommend import add_risk_score

    board = pd.DataFrame(
        {
            "player": ["steady D", "volatile C"],
            "p20": [330.0, 350.0],
            "p80": [420.0, 560.0],
            "replacement": [209.0, 281.0],
        }
    )
    early = add_risk_score(board, 0.2).set_index("player")["risk_score"]
    late = add_risk_score(board, 0.8).set_index("player")["risk_score"]
    assert early["steady D"] > early["volatile C"]  # raw p20 says the opposite
    assert late["volatile C"] > late["steady D"]


def test_the_positional_read_names_a_position_only_when_the_gap_is_real():
    from hockey.serve.recommend import positional_read

    def cost(**kw):
        return {p: {"best_now": 0, "best_after": 0, "cost": c} for p, c in kw.items()}

    clear = positional_read(cost(C=177, LW=64, D=30), {"C": 1, "LW": 1, "D": 2}, 7)
    assert clear["call"] == "position" and clear["position"] == "C"
    assert clear["urgent"] == ["C", "D", "LW"]

    close = positional_read(cost(C=40, LW=35), {"C": 1, "LW": 1}, 7)
    assert close["call"] == "no_clear_call" and close["position"] is None


def test_a_filled_position_costs_nothing_to_pass_on():
    from hockey.serve.recommend import positional_read

    costs = {"C": {"best_now": 0, "best_after": 0, "cost": 177.0}}
    assert positional_read(costs, {"C": 0}, 7)["call"] == "best_available"


def test_a_position_i_may_not_take_is_not_a_reason_to_reach():
    # Round 1: goalies drain, but the rule forbids one until round 7.
    from hockey.serve.recommend import positional_read

    costs = {
        "G": {"best_now": 0, "best_after": 0, "cost": 60.0},
        "D": {"best_now": 0, "best_after": 0, "cost": 41.0},
        "C": {"best_now": 0, "best_after": 0, "cost": 10.0},
    }
    read = positional_read(costs, {"G": 2, "D": 4, "C": 2}, 12, frozenset({"G"}))
    assert read["position"] == "D"
    assert "G" not in read["open_costs"]


def test_a_lone_open_position_that_costs_nothing_is_no_call():
    from hockey.serve.recommend import positional_read

    costs = {"G": {"best_now": 80, "best_after": 80, "cost": 0.0}}
    assert positional_read(costs, {"G": 1}, 14)["call"] == "no_clear_call"


def test_goalies_are_capped():
    from hockey.serve.recommend import DEFAULT_RULES, _rule_block

    class Row:
        slot = "G"

    three = pd.DataFrame({"slot": ["G", "G", "G"], "eligible": [("G",)] * 3})
    state = DraftState(slot=8, n_teams=14)
    state.reconcile(list(range(4000, 4000 + 14 * 15)))  # round 16
    assert "cap" in _rule_block(Row(), state, DEFAULT_RULES, three)
    assert _rule_block(Row(), state, DEFAULT_RULES, three.head(2)) is None
