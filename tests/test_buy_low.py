"""The Season tab's trade targets: buyLow, run under Node from the page."""


def _buy_low(page_js, players, **opts):
    return page_js("buylow", "buyLow(x.players, x.opts)", {"players": players, "opts": opts})


def _p(i, gp, pg, rate, pre):
    return {"i": i, "gp": gp, "pg": pg, "rate": rate, "pre": pre}


def test_a_rate_that_held_while_the_points_lagged_is_a_target(page_js):
    # Cozens's shape on Oct 7: no points yet, and the update nudged his rate up.
    got = _buy_low(page_js, [_p(1, 3, 0.0, 5.81, 5.71)])
    assert [g["i"] for g in got] == [1]
    assert abs(got[0]["gap"] - 5.81) < 1e-9
    assert abs(got[0]["up"] - 0.10) < 1e-9


def test_a_hot_start_the_model_does_not_buy_is_not(page_js):
    # Tolvanen's shape: 6.25 a game on three goals, and the rate slipped.
    got = _buy_low(page_js, [_p(1, 4, 6.25, 4.47, 4.55)])
    assert got == []


def test_a_rate_that_fell_is_out_even_with_a_big_gap(page_js):
    # Raddysh's shape on Oct 7: 1 point in 4 games on 11 shots, the rate 5.30 -> 5.08.
    # Kept out on purpose; a dip that size can be a player falling for real.
    got = _buy_low(page_js, [_p(1, 4, 2.5, 5.08, 5.30)])
    assert got == []


def test_one_game_is_too_few_and_a_small_gap_is_not_a_target(page_js):
    got = _buy_low(page_js, [_p(1, 1, 0.0, 5.0, 4.9), _p(2, 3, 4.7, 5.0, 4.9)])
    assert got == []


def _lookup(page_js, players, q, top=12):
    return page_js(
        "buylow", "lookupMatch(x.players, x.q, x.top)", {"players": players, "q": q, "top": top}
    )


def test_the_lookup_ignores_accents_and_case(page_js):
    players = [
        {"i": 1, "n": "Tim Stützle", "t": "OTT"},
        {"i": 2, "n": "Darren Raddysh", "t": "TOR"},
    ]
    assert [p["i"] for p in _lookup(page_js, players, "stutzle")] == [1]
    assert [p["i"] for p in _lookup(page_js, players, "RADDY")] == [2]


def test_the_lookup_takes_a_whole_team_and_needs_two_letters(page_js):
    players = [
        {"i": 1, "n": "Auston Matthews", "t": "TOR"},
        {"i": 2, "n": "Torey Krug", "t": "STL"},
    ]
    assert [p["i"] for p in _lookup(page_js, players, "tor")] == [1, 2]  # Torey by name
    assert [p["i"] for p in _lookup(page_js, players, "t")] == []


def test_the_lookup_sorts_by_name_and_stops_at_top(page_js):
    players = [
        {"i": i, "n": n, "t": "BOS"} for i, n in enumerate(["Zacha", "Lindholm", "Pastrnak"])
    ]
    assert [p["n"] for p in _lookup(page_js, players, "bos", top=2)] == ["Lindholm", "Pastrnak"]


def test_targets_sort_by_the_gap_and_stop_at_top(page_js):
    players = [_p(1, 3, 3.0, 4.0, 4.0), _p(2, 3, 1.0, 5.0, 4.9), _p(3, 3, 2.0, 4.5, 4.4)]
    got = _buy_low(page_js, players, top=2)
    assert [g["i"] for g in got] == [2, 3]
