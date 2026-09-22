"""Starts, not games.

The question a late pick answers is "how many nights would he be in my lineup",
and the answer depends on who I already have. These cover the cases where the
two readings disagree, because that disagreement is the only reason the feature
exists.
"""

from datetime import date

import pandas as pd

from hockey.serve.schedule import marginal_starts, week_window

SHAPE = {"C": 2, "LW": 2, "RW": 2, "D": 4, "G": 2}
MON = date(2026, 10, 12)
TUE = date(2026, 10, 13)
WED = date(2026, 10, 14)
THU = date(2026, 10, 15)
WINDOW = (MON, THU)


def player(pid: int, name: str, team: str, positions: tuple[str, ...], mean: float) -> dict:
    return {
        "player_id": pid,
        "player": name,
        "team": team,
        "position": positions[0],
        "eligible": positions,
        "mean": mean,
    }


def roster(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def test_a_player_behind_two_better_centres_starts_only_on_their_off_nights():
    """The whole point. Both candidates play four nights; one of them is on the
    bench for most of his, because my centre slots are already full those
    nights."""
    mine = roster(
        player(1, "my first C", "BOS", ("C",), 500.0),
        player(2, "my second C", "BOS", ("C",), 480.0),
    )
    calendar = {
        "BOS": [MON, TUE, WED, THU],  # my two centres play every night
        "UTA": [MON, TUE, WED, THU],  # same nights: he sits behind them
        "SEA": [TUE, THU],  # nights my centres also play, so still blocked
        "NSH": [MON, WED],
    }
    clash = pd.Series(player(3, "same nights", "UTA", ("C",), 300.0))
    got = marginal_starts(clash, mine, calendar, SHAPE, WINDOW)
    assert got.games == 4
    assert got.starts == 0
    assert got.blocked == 4

    # Free up the calendar: my centres only play Monday and Wednesday.
    calendar["BOS"] = [MON, WED]
    got = marginal_starts(clash, mine, calendar, SHAPE, WINDOW)
    assert got.starts == 2  # Tuesday and Thursday are his
    assert got.dates == [TUE.isoformat(), THU.isoformat()]


def test_a_better_player_takes_the_slot_from_the_one_i_hold():
    # He is not queued behind my roster, he competes with it: the night is
    # filled best first, so a stronger candidate starts and mine sits.
    mine = roster(player(1, "my weak C", "BOS", ("C",), 200.0))
    calendar = {"BOS": [MON], "UTA": [MON]}
    strong = pd.Series(player(2, "better", "UTA", ("C",), 600.0))
    assert marginal_starts(strong, mine, calendar, {"C": 1}, (MON, MON)).starts == 1


def test_dual_eligibility_finds_the_open_slot():
    # Centre is full on Monday, but he is also a winger and left wing is empty,
    # so he starts. Counting by his primary position alone would miss this.
    mine = roster(
        player(1, "C one", "BOS", ("C",), 500.0),
        player(2, "C two", "BOS", ("C",), 490.0),
    )
    calendar = {"BOS": [MON], "UTA": [MON]}
    flex = pd.Series(player(3, "C/LW", "UTA", ("C", "LW"), 300.0))
    assert marginal_starts(flex, mine, calendar, SHAPE, (MON, MON)).starts == 1


def test_games_outside_the_window_do_not_count():
    mine = roster()
    calendar = {"UTA": [MON, date(2026, 11, 20)]}
    cand = pd.Series(player(9, "x", "UTA", ("C",), 300.0))
    got = marginal_starts(cand, mine, calendar, SHAPE, WINDOW)
    assert got.games == 1


def test_a_team_with_no_games_in_the_window_is_zero_not_missing():
    got = marginal_starts(
        pd.Series(player(9, "x", "VGK", ("C",), 300.0)), roster(), {"UTA": [MON]}, SHAPE, WINDOW
    )
    assert (got.games, got.starts, got.blocked) == (0, 0, 0)


def test_the_window_comes_from_the_league_weeks_not_seven_day_arithmetic():
    """Yahoo's first week is short when the season opens midweek, so counting
    14 days from opening night measures the wrong fortnight."""
    weeks = [
        {"week": 1, "start": "2026-10-07", "end": "2026-10-11"},  # short opening week
        {"week": 2, "start": "2026-10-12", "end": "2026-10-18"},
        {"week": 3, "start": "2026-10-19", "end": "2026-10-25"},
    ]
    assert week_window(weeks, 1, 2) == (date(2026, 10, 7), date(2026, 10, 18))
    assert week_window(weeks, 2, 1) == (date(2026, 10, 12), date(2026, 10, 18))


def test_asking_for_more_weeks_than_exist_warns_and_shortens(caplog):
    weeks = [{"week": 1, "start": "2026-10-07", "end": "2026-10-11"}]
    with caplog.at_level("WARNING"):
        got = week_window(weeks, 1, 2)
    assert got == (date(2026, 10, 7), date(2026, 10, 11))
    assert "schedule window is short" in caplog.text
    assert week_window([], 1, 2) is None


def test_the_fast_lineup_fill_matches_fill_slots():
    """The season window fills ~180 lineups per candidate, so it does not build
    a DataFrame per night. That is a second implementation of the assignment
    rule, and this project has been bitten by those, so it is pinned to the
    original over random rosters."""
    import random

    from hockey.export.replacement import fill_slots
    from hockey.serve.schedule import _fill_night

    positions = ["C", "LW", "RW", "D", "G"]
    rng = random.Random(7)
    for _ in range(200):
        rows = []
        for pid in range(rng.randint(1, 14)):
            eligible = tuple(rng.sample(positions, rng.randint(1, 2)))
            rows.append(player(pid, f"p{pid}", "BOS", eligible, rng.uniform(100, 600)))
        frame = pd.DataFrame(rows)
        shape = {p: rng.randint(0, 3) for p in positions}
        want, _ = fill_slots(frame, {p: float(n) for p, n in shape.items()})
        got = _fill_night(
            [(int(r["player_id"]), float(r["mean"]), tuple(r["eligible"])) for r in rows], shape
        )
        assert got == want


def test_the_season_window_answers_a_different_question_from_the_fortnight():
    """A third centre can look fine over two weeks and poor over a season, or
    the reverse, which is why the depth picks ask about the whole schedule."""
    mine = roster(
        player(1, "C one", "BOS", ("C",), 500.0),
        player(2, "C two", "BOS", ("C",), 490.0),
    )
    # My centres play Mondays. The candidate plays Mondays in the opening
    # fortnight and Tuesdays for the rest of the season.
    mondays = [date(2026, 10, 12), date(2026, 10, 19)]
    tuesdays = [date(2026, 11, 3), date(2026, 11, 10), date(2026, 11, 17)]
    calendar = {"BOS": mondays, "UTA": mondays + tuesdays}
    cand = pd.Series(player(3, "late bloomer", "UTA", ("C",), 300.0))
    fortnight = marginal_starts(cand, mine, calendar, SHAPE, (mondays[0], mondays[1]))
    season = marginal_starts(cand, mine, calendar, SHAPE, (mondays[0], tuesdays[-1]))
    assert (fortnight.games, fortnight.starts) == (2, 0)
    assert (season.games, season.starts) == (5, 3)
    assert season.as_dict(with_dates=False)["start_share"] == 0.6
