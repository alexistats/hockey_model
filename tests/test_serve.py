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


# --- the plan, and what the pick rule prefers ------------------------------


class _Board:
    """Just enough board for the plan: the roster shape and the bench."""

    roster_shape = {"C": 2, "LW": 2, "RW": 2, "D": 4, "G": 2}
    bench = 5
    injuries: dict = {}
    calendar: dict = {}
    weeks: list = []


def _state_with(mine: int):
    state = DraftState(slot=8, n_teams=14)
    for pid in range(9000, 9000 + mine):
        state.take(pid, by_me=True)
    return state


def test_the_plan_fills_starters_before_it_plans_anything():
    from hockey.serve.recommend import DEFAULT_RULES, draft_plan

    plan = draft_plan(_Board(), _state_with(15), DEFAULT_RULES, {"D": 1})
    assert plan["pick_kind"] == "starter"  # two picks left, but a slot is open


def test_the_last_two_picks_are_schedule_picks_and_the_ones_before_swing():
    from hockey.serve.recommend import DEFAULT_RULES, draft_plan

    full = dict.fromkeys(["C", "LW", "RW", "D", "G"], 0)
    kinds = {
        17 - n: draft_plan(_Board(), _state_with(n), DEFAULT_RULES, full)["pick_kind"]
        for n in range(12, 17)
    }
    # picks_left 5,4,3 swing for the ceiling; the final 2 are schedule picks
    assert kinds[5] == "ceiling" and kinds[4] == "ceiling" and kinds[3] == "ceiling"
    assert kinds[2] == "schedule" and kinds[1] == "schedule"


def _cand(pid, name, score, *, need=False, slot="C", starts=None, injury=None):
    from hockey.serve.recommend import Candidate

    return Candidate(
        player_id=pid,
        player=name,
        slot=slot,
        eligible=[slot],
        team="BOS",
        mean=400.0,
        floor=300.0,
        ceiling=500.0,
        vorp=score,
        p20=350.0,
        p80=450.0,
        risk_score=score,
        tier=1,
        replacement_is_lower_bound=False,
        replacement_free_below=20,
        fills_a_need=need,
        need_slot=slot if need else None,
        injury=injury,
        schedule_opening=None if starts is None else {"games": 5, "starts": starts},
    )


def test_an_open_slot_beats_a_slightly_better_bench_player():
    """The logjam this was written for: one defenceman, then forwards all night,
    and the blue line drafted from leftovers in the last rounds."""
    from hockey.serve.recommend import DEFAULT_RULES, _recommend

    no_read = {"position": None, "reason": ""}
    plan = {"pick_kind": "starter"}
    shortlist = [_cand(1, "bench forward", 200.0), _cand(2, "open D", 180.0, need=True, slot="D")]
    got = _recommend(shortlist, no_read, plan, DEFAULT_RULES, "risk_score")
    assert got["player"] == "open D" and got["why"] == "needs first -> D"


def test_a_far_better_bench_player_still_wins():
    # "Unless the forwards are insanely better" - priced at the margin, not
    # asserted, so the rule can be argued with rather than believed.
    from hockey.serve.recommend import DEFAULT_RULES, _recommend

    shortlist = [_cand(1, "much better", 260.0), _cand(2, "open D", 180.0, need=True, slot="D")]
    got = _recommend(
        shortlist, {"position": None}, {"pick_kind": "starter"}, DEFAULT_RULES, "risk_score"
    )
    assert got["player"] == "much better" and "margin" in got["detail"]


def test_a_schedule_pick_skips_a_player_who_cannot_play():
    from hockey.serve.recommend import DEFAULT_RULES, _recommend

    shortlist = [
        _cand(1, "injured", 200.0, starts=5, injury={"status": "Out"}),
        _cand(2, "available", 150.0, starts=3),
    ]
    got = _recommend(
        shortlist, {"position": None}, {"pick_kind": "schedule"}, DEFAULT_RULES, "risk_score"
    )
    assert got["player"] == "available"
    assert got["why"] == "plan -> schedule"


def test_a_schedule_pick_with_no_starts_falls_back_rather_than_forcing_one():
    # Everyone left is blocked by my own roster that fortnight, so there is no
    # schedule pick to make and the best player is the honest answer.
    from hockey.serve.recommend import DEFAULT_RULES, _recommend

    shortlist = [_cand(1, "blocked", 200.0, starts=0), _cand(2, "also blocked", 150.0, starts=0)]
    got = _recommend(
        shortlist, {"position": None}, {"pick_kind": "schedule"}, DEFAULT_RULES, "risk_score"
    )
    assert got["player"] == "blocked" and got["why"] == "best score"


# --- the clock counts picks it cannot name ---------------------------------


def test_a_pick_the_server_cannot_identify_still_moves_the_clock():
    """The bug this was written for.

    A live draft takes players who are not in the modelled pool at all - the
    board holds 295 skaters and 66 goalies. Those picks resolved to nothing and
    were dropped, so the server believed the draft was several picks behind:
    the round number gates the goalie rules and the risk ramp, and
    picks_until_my_turn is what the cost of waiting is measured over.
    """
    state = DraftState(slot=8, n_teams=14)
    state.reconcile([101, 102, 103], unidentified=2)
    assert state.total_picks == 5
    assert state.current_pick == 6
    assert len(state.picks) == 3  # still only three players named
    assert state.drafted == {101, 102, 103}


def test_an_unidentified_pick_is_never_attached_to_a_player():
    state = DraftState()
    state.reconcile([101], unidentified=3)
    assert state.drafted == {101}
    assert state.mine == []


def test_a_name_that_resolves_later_is_not_counted_twice():
    # The observation is a complete statement, so the count is set rather than
    # accumulated: the bot sending a fuller name next poll must not leave the
    # clock permanently ahead.
    state = DraftState()
    state.reconcile([101, 102], unidentified=1)
    assert state.total_picks == 3
    state.reconcile([101, 102, 103], unidentified=0)
    assert state.total_picks == 3
    assert state.drafted == {101, 102, 103}


def test_unidentified_picks_move_the_round_and_my_turn():
    state = DraftState(slot=8, n_teams=14)
    state.reconcile(list(range(200, 200 + 13)), unidentified=1)  # 14 picks: round 2
    assert state.current_round == 2
    # My pick is 8; with 14 on the board the next is 21, so 6 picks away.
    assert state.picks_until_my_turn() == 6


def test_reset_clears_the_unidentified_count_too():
    state = DraftState()
    state.reconcile([1], unidentified=4)
    state.reset()
    assert state.total_picks == 0 and state.current_round == 1


# --- the draft ends, which no value curve knows ----------------------------


def test_the_margin_rises_as_the_picks_run_out():
    from hockey.serve.recommend import DEFAULT_RULES, roster_pressure

    board, rules = _Board(), {**DEFAULT_RULES, "draft_rounds": 16}
    open_d = {"C": 0, "LW": 0, "RW": 0, "D": 3, "G": 1}
    early = roster_pressure(board, _state_with(4), rules, open_d)
    late = roster_pressure(board, _state_with(9), rules, open_d)
    assert early["slack"] == 8 and late["slack"] == 3
    assert late["need_first_margin"] > early["need_first_margin"]


def test_with_no_slack_no_gap_buys_a_bench_player():
    from hockey.serve.recommend import DEFAULT_RULES, roster_pressure

    rules = {**DEFAULT_RULES, "draft_rounds": 16}
    # four picks left, four slots open: every one of them is spoken for
    pressure = roster_pressure(_Board(), _state_with(12), rules, {"D": 3, "G": 1})
    assert pressure["slack"] == 0
    assert pressure["need_first_margin"] is None


def test_the_reported_mock_now_takes_the_defenceman():
    """The case from the bot's report, at its own numbers.

    Round 10 of 16, forwards full and three defence slots open. The best bench
    winger scored 195.2 against 148.6 for the best defenceman - a 46.6 gap, past
    the flat 25 bar, so the override fired and the blue line stayed empty. With
    seven picks left for four slots the bar is 58, and the defenceman wins.
    """
    from hockey.serve.recommend import DEFAULT_RULES, _recommend, roster_pressure

    rules = {**DEFAULT_RULES, "draft_rounds": 16}
    needs = {"C": 0, "LW": 0, "RW": 0, "D": 3, "G": 1}
    pressure = roster_pressure(_Board(), _state_with(9), rules, needs)
    shortlist = [
        _cand(1, "Kiefer Sherwood", 195.2, slot="LW"),
        _cand(2, "best defenceman", 148.6, need=True, slot="D"),
    ]
    got = _recommend(
        shortlist, {"position": None}, {"pick_kind": "starter"}, rules, "risk_score", pressure
    )
    assert got["player"] == "best defenceman"
    assert got["why"] == "needs first -> D"

    # The same two players in round 4, with room to spare, still go the other
    # way: this is a deadline, not a preference for defencemen.
    roomy = roster_pressure(_Board(), _state_with(3), rules, needs)
    got = _recommend(
        shortlist, {"position": None}, {"pick_kind": "starter"}, rules, "risk_score", roomy
    )
    assert got["player"] == "Kiefer Sherwood" and got["why"] == "need overridden"


def test_the_deadline_outranks_the_positional_reach():
    """Reach is a value argument and cannot see the draft ending, so with every
    remaining pick spoken for it would still spend one on whichever forward
    position happens to be draining."""
    from hockey.serve.recommend import DEFAULT_RULES, _recommend, roster_pressure

    rules = {**DEFAULT_RULES, "draft_rounds": 16}
    pressure = roster_pressure(_Board(), _state_with(12), rules, {"D": 4})
    read = {
        "position": "RW",
        "reason": "waiting 7 picks costs about 16 at RW, against 0 at D",
        "take": {"player_id": 1, "player": "a winger"},
    }
    shortlist = [
        _cand(1, "a winger", 200.0, slot="RW"),
        _cand(2, "a defenceman", 120.0, need=True, slot="D"),
    ]
    got = _recommend(shortlist, read, {"pick_kind": "starter"}, rules, "risk_score", pressure)
    assert got["player"] == "a defenceman"


def test_why_stays_inside_the_agreed_vocabulary():
    from hockey.serve.recommend import DEFAULT_RULES, _recommend, roster_pressure

    allowed_whys = {
        "best score",
        "need overridden",
        "plan -> ceiling",
        "plan -> schedule",
    }
    rules = {**DEFAULT_RULES, "draft_rounds": 16}
    cases = [
        ([_cand(1, "a", 200.0, need=True)], {"position": None}, {"pick_kind": "starter"}, {"D": 1}),
        (
            [_cand(1, "a", 200.0), _cand(2, "b", 100.0, need=True, slot="D")],
            {"position": None},
            {"pick_kind": "starter"},
            {"D": 1},
        ),
        ([_cand(1, "a", 200.0, starts=4)], {"position": None}, {"pick_kind": "schedule"}, {}),
        ([_cand(1, "a", 200.0)], {"position": None}, {"pick_kind": "ceiling"}, {}),
    ]
    for shortlist, read, plan, needs in cases:
        pressure = roster_pressure(_Board(), _state_with(3), rules, needs)
        why = _recommend(shortlist, read, plan, rules, "risk_score", pressure)["why"]
        assert why in allowed_whys or why.startswith(("needs first -> ", "positional_read -> ")), (
            why
        )
