"""Starts, not games.

The question a late pick answers is "how many nights would he be in my lineup",
and the answer depends on who I already have. These cover the cases where the
two readings disagree, because that disagreement is the only reason the feature
exists.
"""

from datetime import date

import pandas as pd
import pytest

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


# --- a night's lineup, and the games a player adds ---------------------------


def test_the_lineup_fills_the_slot_the_greedy_leaves_open():
    """Best first into the slot with the most room puts the C/LW on the wing,
    and the pure winger behind him has nowhere to go while centre sits open. A
    manager moves the C/LW to centre and starts all three."""
    from hockey.serve.schedule import _fill_night, lineup

    tonight = [(1, 400.0, ("C", "LW")), (2, 300.0, ("LW",)), (3, 200.0, ("LW",))]
    shape = {"C": 1, "LW": 2}
    assert len(_fill_night(tonight, shape)) == 2  # the league's roster rule
    assert lineup(tonight, shape) == {1: "C", 2: "LW", 3: "LW"}


def test_the_lineup_is_as_full_as_possible_with_the_best_players_in_it():
    """Held to a brute force over every set of players that could start
    together: the most of them, and of those the best by mean."""
    import itertools
    import random

    from hockey.serve.schedule import lineup

    positions = ["C", "LW", "RW", "D"]
    rng = random.Random(3)
    for _ in range(300):
        shape = {p: rng.randint(0, 2) for p in positions}
        units = [p for p, n in shape.items() for _ in range(n)]
        players = [
            (k, float(m), tuple(rng.sample(positions, rng.randint(1, 2))))
            for k, m in enumerate(rng.sample(range(100, 900), rng.randint(0, 7)))
        ]

        def fits(group, units=units, players=players):
            return any(
                all(units[u] in players[i][2] for i, u in zip(group, seats, strict=True))
                for seats in itertools.permutations(range(len(units)), len(group))
            )

        groups = [
            g
            for r in range(len(players) + 1)
            for g in itertools.combinations(range(len(players)), r)
        ]
        size = max(len(g) for g in groups if fits(g))
        best = max(
            (g for g in groups if len(g) == size and fits(g)),
            key=lambda g: sorted((players[i][1] for i in g), reverse=True),
        )
        got = lineup(players, shape)
        assert len(got) == size
        assert set(got) == {players[i][0] for i in best}
        assert all(got[pid] in dict((p[0], p[2]) for p in players)[pid] for pid in got)


def test_a_third_centre_adds_only_the_nights_one_of_mine_is_off():
    """The case that exposed the difference. With two centres on my roster a
    third one better than both starts every night he plays - but on nights all
    three play he only bumps one of mine. Those nights add nothing to my lineup,
    and a player who adds more games is the better schedule pick."""
    from hockey.serve.schedule import added_games

    mine = roster(
        player(1, "my first C", "BOS", ("C",), 500.0),
        player(2, "my second C", "NYR", ("C",), 480.0),
    )
    calendar = {"BOS": [MON, TUE, WED], "NYR": [MON, TUE, THU], "UTA": [MON, TUE, WED, THU]}
    better = pd.Series(player(3, "better than both", "UTA", ("C",), 900.0))
    assert marginal_starts(better, mine, calendar, SHAPE, WINDOW).starts == 4
    got = added_games(better, mine, calendar, SHAPE, WINDOW)
    assert (got.games, got.added) == (4, 2)  # Wednesday and Thursday, when one of mine sits
    assert got.dates == [WED.isoformat(), THU.isoformat()]


def test_a_player_adds_a_game_by_freeing_a_slot_for_someone_else():
    # Centre is held by my C/LW, and left wing is open. A pure centre still adds
    # a game: my C/LW moves to the wing and he takes centre.
    from hockey.serve.schedule import added_games

    mine = roster(player(1, "my C/LW", "BOS", ("C", "LW"), 500.0))
    calendar = {"BOS": [MON], "UTA": [MON]}
    centre = pd.Series(player(2, "pure centre", "UTA", ("C",), 300.0))
    assert added_games(centre, mine, calendar, {"C": 1, "LW": 1}, (MON, MON)).added == 1
    assert added_games(centre, mine, calendar, {"C": 1}, (MON, MON)).added == 0


# --- the page's copy ---------------------------------------------------------
#
# The draft page counts games added and starts itself, live as my roster fills,
# because it has no server behind it. That is a second copy of `lineup`,
# `added_games` and `marginal_starts`, so it is held to them here, under Node,
# on random rosters.


def _random_league(seed: int):
    """A random calendar, roster and field of candidates, with no tied means."""
    import random

    rng = random.Random(seed)
    teams = [f"T{k}" for k in range(8)]
    days = [date(2026, 10, 1 + k) for k in range(28)]
    calendar = {t: sorted(rng.sample(days, rng.randint(6, 18))) for t in teams}
    positions = ["C", "LW", "RW", "D", "G"]
    means = rng.sample(range(100, 700), 60)
    players = [
        player(k, f"p{k}", rng.choice(teams), tuple(rng.sample(positions, rng.randint(1, 2))), m)
        for k, m in enumerate(means)
    ]
    size = rng.randint(0, 17)
    return calendar, players[:size], players[size : size + 25], days


def _as_page(calendar, rows):
    iso = {t: [d.isoformat() for d in ds] for t, ds in calendar.items()}
    entries = [
        {"id": r["player_id"], "team": r["team"], "mean": r["mean"], "elig": list(r["eligible"])}
        for r in rows
    ]
    return iso, entries


def _frame(rows):
    return pd.DataFrame(rows, columns=list(player(0, "", "", ("C",), 0.0)))


def test_the_page_schedule_block_is_self_contained(page_source):
    block = page_source("schedule")
    assert "function addsCounter" in block
    for page_global in ("DATA", "state.", "P[", "document"):
        assert page_global not in block, page_global


@pytest.mark.parametrize("seed", range(6))
def test_the_page_counts_games_added_as_the_api_does(page_js, seed):
    from hockey.serve.schedule import added_games

    calendar, mine, field, days = _random_league(seed)
    window = (days[3], days[-4])
    iso, roster = _as_page(calendar, mine)
    _, candidates = _as_page(calendar, field)
    got = page_js(
        "schedule",
        "x.who.map((w) => addsCounter(x.roster, x.calendar, x.shape)(w, x.first, x.last))",
        {
            "roster": roster,
            "calendar": iso,
            "shape": SHAPE,
            "who": candidates + roster,
            "first": window[0].isoformat(),
            "last": window[1].isoformat(),
        },
    )
    frame = _frame(mine)
    for row, page in zip(field + mine, got, strict=True):
        want = added_games(pd.Series(row), frame, calendar, SHAPE, window)
        assert (page["games"], page["added"]) == (want.games, want.added), row["player"]


@pytest.mark.parametrize("seed", range(6))
def test_the_page_counts_my_players_starts_as_the_api_does(page_js, seed):
    calendar, mine, _, days = _random_league(seed)
    window = (days[2], days[-3])
    iso, roster = _as_page(calendar, mine)
    got = page_js(
        "schedule",
        "rosterStarts(x.roster, x.calendar, x.shape, x.first, x.last)",
        {
            "roster": roster,
            "calendar": iso,
            "shape": SHAPE,
            "first": window[0].isoformat(),
            "last": window[1].isoformat(),
        },
    )
    frame = _frame(mine)
    for row in mine:
        rest = frame[frame["player_id"] != row["player_id"]]
        want = marginal_starts(pd.Series(row), rest, calendar, SHAPE, window)
        assert got[str(row["player_id"])] == want.starts, row["player"]


def test_the_page_counts_off_nights_as_nights_with_few_games(page_js):
    # Monday: one game. Tuesday: two. Wednesday: three.
    calendar = {
        "A": ["mon", "tue", "wed"],
        "B": ["mon", "wed"],
        "C": ["tue", "wed"],
        "D": ["tue", "wed"],
        "E": ["wed"],
        "F": ["wed"],
    }
    one = page_js("schedule", "offNightGames(x.calendar, 1)", {"calendar": calendar})
    two = page_js("schedule", "offNightGames(x.calendar, 2)", {"calendar": calendar})
    assert one == {"A": 1, "B": 1, "C": 0, "D": 0, "E": 0, "F": 0}
    assert two == {"A": 2, "B": 1, "C": 1, "D": 1, "E": 0, "F": 0}


@pytest.mark.parametrize("seed", range(4))
def test_the_page_fills_my_lineup_night_by_night_as_the_api_would(page_js, seed):
    from hockey.serve.schedule import lineup

    calendar, mine, _, days = _random_league(seed)
    iso, roster = _as_page(calendar, mine)
    got = page_js(
        "schedule",
        "lineupNights(x.roster, x.calendar, x.shape, x.first, x.last)",
        {
            "roster": roster,
            "calendar": iso,
            "shape": SHAPE,
            "first": days[0].isoformat(),
            "last": days[-1].isoformat(),
        },
    )
    want = dict.fromkeys(SHAPE, 0)
    for day in days:
        tonight = [
            (r["player_id"], r["mean"], tuple(r["eligible"]))
            for r in mine
            if day in calendar[r["team"]]
        ]
        for slot in lineup(tonight, SHAPE).values():
            want[slot] += 1
    assert got == want
