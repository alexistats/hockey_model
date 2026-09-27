"""The season's league file, and what a move is worth to my lineup.

Two things the Season tab stands on: whose players are whose, kept as ids and
never as guesses; and a week's expected lineup points, which is what an add or
a drop changes - not games, and not starts.
"""

import itertools
import random
from datetime import date

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from hockey.serve import season
from hockey.serve.app import create_app
from hockey.serve.identity import Resolution
from hockey.serve.schedule import night_points, window_points

SHAPE = {"C": 2, "LW": 2, "RW": 2, "D": 4, "G": 2}

# --- the league file ---------------------------------------------------------


def _resolver(known: dict[str, int]):
    def resolve(name, position=None):
        pid = known.get(name)
        return Resolution(name, pid, "exact" if pid else "unmatched")

    return resolve


def test_the_draft_seeds_mine_theirs_and_the_names_it_could_not_read():
    rows = [
        {"team": "Me", "player": "Ada Alpha", "positions": "C"},
        {"team": "Them", "player": "Bea Beta", "positions": "LW,RW"},
        {"team": "Me", "player": "A Rookie", "positions": "D"},
    ]
    got = season.seed_from_draft(rows, "Me", _resolver({"Ada Alpha": 1, "Bea Beta": 2}))
    assert (got.mine, got.taken, got.me) == ([1], [2], "Me")
    # Kept by name where the page can show it - never turned into someone's id.
    assert got.unresolved == ["A Rookie (Me): unmatched"]


def test_a_saved_league_reads_back_and_my_players_are_never_also_theirs(tmp_path):
    path = tmp_path / "league.json"
    league = season.League(mine=[1, 2], taken=[2, 3, 3], out={1: "2026-11-15", 2: None}, me="Me")
    saved = season.save(league, path)
    assert saved.taken == [3]
    back = season.load(path)
    assert (back.mine, back.taken, back.out, back.me) == (
        [1, 2],
        [3],
        {1: "2026-11-15", 2: None},
        "Me",
    )
    assert back.updated_at is not None


def test_a_malformed_return_date_is_an_error_not_out_forever(tmp_path):
    with pytest.raises(ValueError):
        season.save(season.League(mine=[1], out={1: "mid-November"}), tmp_path / "league.json")


PLAYERS = [(1001, "Ada Alpha", "C"), (1002, "Bea Beta", "LW"), (1003, "Cy Gamma", "D")]


@pytest.fixture
def board_dir(tmp_path):
    ids = [p for p, _, _ in PLAYERS]
    pd.DataFrame(
        {
            "player": [n for _, n, _ in PLAYERS],
            "player_id": ids,
            "slot": [s for _, _, s in PLAYERS],
            "position": [s for _, _, s in PLAYERS],
            "team": "EDM",
            "mean": [500.0, 450.0, 400.0],
            "floor": 300.0,
            "p20": 350.0,
            "p80": 550.0,
            "ceiling": 600.0,
            "exp_games": 80.0,
            "age": 27,
            "vorp": [200.0, 150.0, 100.0],
            "pos_rank": 1,
            "tier": 1,
        }
    ).to_csv(tmp_path / "value_board.csv", index=False)
    draws = np.random.default_rng(0).normal(400, 50, size=(100, len(ids)))
    np.savez(tmp_path / "draws_batch_01.npz", player_ids=np.array(ids), draws=draws)
    return tmp_path


def test_the_page_reads_and_writes_the_league_file(board_dir, tmp_path):
    path = tmp_path / "season" / "league.json"
    client = TestClient(create_app(board_dir, n_teams=4, slot=3, league=path))
    assert client.get("/season/league").status_code == 404

    season.save(season.League(mine=[1001], unresolved=["A Rookie (Me): unmatched"], me="Me"), path)
    put = client.put(
        "/season/league", json={"mine": [1001, 1002], "taken": [1003], "out": {"1002": None}}
    )
    assert put.status_code == 200
    got = client.get("/season/league").json()
    assert (got["mine"], got["taken"], got["out"]) == ([1001, 1002], [1003], {"1002": None})
    # What the page cannot see, it cannot erase: the names and my team survive a write.
    assert (got["unresolved"], got["me"]) == (["A Rookie (Me): unmatched"], "Me")


def test_an_id_the_board_does_not_know_is_refused_not_stored(board_dir, tmp_path):
    path = tmp_path / "league.json"
    season.save(season.League(mine=[1001]), path)
    client = TestClient(create_app(board_dir, n_teams=4, slot=3, league=path))
    bad = client.put("/season/league", json={"mine": [1001, 424242]})
    assert bad.status_code == 422
    assert season.load(path).mine == [1001]


# --- a week's expected lineup points -------------------------------------------


def _goalie(pid, rate, share, team="T0"):
    return {"id": pid, "team": team, "rate": rate, "elig": ("G",), "share": share}


def _skater(pid, rate, elig, team="T0", start=None):
    e = {"id": pid, "team": team, "rate": rate, "elig": tuple(elig)}
    if start is not None:
        e["from"] = start
    return e


def test_goalie_slots_are_the_expectation_over_who_starts():
    goalies = [_goalie(1, 8.0, 0.6), _goalie(2, 7.0, 0.5), _goalie(3, 6.0, 0.4)]
    want = 0.0
    for starts in itertools.product([0, 1], repeat=3):
        chance = np.prod(
            [g["share"] if s else 1 - g["share"] for g, s in zip(goalies, starts, strict=True)]
        )
        up = sorted((g["rate"] for g, s in zip(goalies, starts, strict=True) if s), reverse=True)
        want += chance * sum(up[:2])
    assert night_points(goalies, SHAPE) == pytest.approx(want)
    # A third goalie adds something: nights one of the first two sits.
    assert night_points(goalies, SHAPE) > night_points(goalies[:2], SHAPE)


def test_skaters_score_the_lineup_a_manager_would_set():
    # Three centres, two slots, and a C/LW who moves to the wing so all three start.
    tonight = [_skater(1, 5.0, "C"), _skater(2, 4.0, "C"), _skater(3, 3.0, ("C", "LW"))]
    assert night_points(tonight, {"C": 2, "LW": 1}) == pytest.approx(12.0)
    assert night_points(tonight, {"C": 2}) == pytest.approx(9.0)


def test_a_player_out_until_a_date_counts_only_from_it():
    mon, tue, wed = date(2026, 10, 12), date(2026, 10, 13), date(2026, 10, 14)
    calendar = {"T0": [mon, tue, wed]}
    back_tuesday = _skater(1, 5.0, "C", start=tue)
    got = window_points([back_tuesday], calendar, SHAPE, mon, wed)
    assert got == {tue: 5.0, wed: 5.0}


# --- the page's copy, held to this one under Node --------------------------------


def _random_week(seed: int):
    rng = random.Random(seed)
    teams = [f"T{k}" for k in range(6)]
    days = [date(2026, 10, 1 + k).isoformat() for k in range(14)]
    calendar = {t: sorted(rng.sample(days, rng.randint(3, 9))) for t in teams}
    entries = []
    for k in range(rng.randint(6, 18)):
        if rng.random() < 0.2:
            entries.append(
                _goalie(
                    k,
                    round(rng.uniform(5, 9), 2),
                    round(rng.uniform(0.3, 0.7), 2),
                    rng.choice(teams),
                )
            )
        else:
            elig = tuple(rng.sample(["C", "LW", "RW", "D"], rng.randint(1, 2)))
            start = rng.choice(days) if rng.random() < 0.2 else None
            entries.append(_skater(k, round(rng.uniform(1, 6), 2), elig, rng.choice(teams), start))
    return calendar, entries, days


@pytest.mark.parametrize("seed", range(8))
def test_the_page_scores_a_window_as_the_api_does(page_js, seed):
    calendar, entries, days = _random_week(seed)
    first, last = days[2], days[-3]
    want = window_points(entries, calendar, SHAPE, first, last)
    page = [{**e, "elig": list(e["elig"])} for e in entries]
    got = page_js(
        "schedule",
        "[...windowPoints(x.entries, x.calendar, x.shape, x.first, x.last)]",
        {"entries": page, "calendar": calendar, "shape": SHAPE, "first": first, "last": last},
    )
    assert [d for d, _ in got] == list(want)
    for (d, points), expected in zip(got, want.values(), strict=True):
        assert points == pytest.approx(expected), d
