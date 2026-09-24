"""The draft page following the draft server: the pick list it reads, and how it marks it."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from hockey.serve.app import create_app

PLAYERS = [(1001, "Ada Alpha", "C"), (1002, "Bea Beta", "LW"), (1003, "Cy Gamma", "D")]


@pytest.fixture
def board_dir(tmp_path):
    """A three-player board: enough for the server to load and resolve names."""
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


def test_the_state_lists_every_pick_in_draft_order_with_mine_marked(board_dir):
    client = TestClient(create_app(board_dir, n_teams=4, slot=3))
    client.post(
        "/draft/observed",
        json={"names": ["Bea Beta", "Nobody Known", "Ada Alpha"], "mine": ["Ada Alpha"]},
    )
    state = client.get("/draft/state").json()
    assert [(p["player_id"], p["by_me"], p["number"]) for p in state["picks"]] == [
        (1002, False, 1),
        (1001, True, 3),
    ]
    assert state["picks"][0]["player"] == "Bea Beta"
    # The name nobody matched reaches the page, and still counts as a pick.
    assert state["unresolved"] == ["Nobody Known"]
    assert state["picks_on_the_board"] == 3


def test_a_reset_forgets_what_the_last_observation_could_not_read(board_dir):
    client = TestClient(create_app(board_dir, n_teams=4, slot=3))
    client.post("/draft/observed", json={"names": ["Nobody Known"]})
    client.post("/draft/reset")
    state = client.get("/draft/state").json()
    assert state["unresolved"] == [] and state["picks"] == []


def test_the_page_is_served_from_the_same_origin_as_the_api(board_dir, tmp_path):
    page = tmp_path / "draft_board.html"
    page.write_text("<html>the draft page</html>", encoding="utf-8")
    client = TestClient(create_app(board_dir, ui=page))
    got = client.get("/ui")
    assert got.status_code == 200
    assert got.headers["content-type"].startswith("text/html")
    assert "the draft page" in got.text
    assert client.get("/", follow_redirects=False).headers["location"] == "/ui"


def test_no_page_to_serve_says_how_to_build_one(board_dir, tmp_path):
    client = TestClient(create_app(board_dir, ui=tmp_path / "missing.html"))
    got = client.get("/ui")
    assert got.status_code == 404
    assert "build_draft_ui" in got.json()["detail"]


# --- the page's side, run under Node ---


def test_the_follow_block_is_self_contained(page_source):
    block = page_source("follow")
    assert "function followBoard" in block
    for page_global in ("DATA", "state.", "P[", "document", "fetch("):
        assert page_global not in block, page_global


ROWS = {"11": 0, "12": 1, "13": 2}


def _follow(page_js, server, extra=(), extra_mine=()):
    return page_js(
        "follow",
        "followBoard(x.server, x.rowOf, x.extra, x.extraMine)",
        {"server": server, "rowOf": ROWS, "extra": list(extra), "extraMine": list(extra_mine)},
    )


def test_the_page_marks_the_servers_picks_in_draft_order(page_js):
    server = {
        "picks": [
            {"player_id": 13, "by_me": True, "number": 3},
            {"player_id": 11, "by_me": False, "number": 1},
        ],
        "picks_on_the_board": 3,
    }
    # The page takes the server's order as sent; the server sorts by number.
    got = _follow(page_js, {**server, "picks": sorted(server["picks"], key=lambda p: p["number"])})
    assert got["order"] == [0, 2]
    assert got["roster"] == [2]
    assert got["numbers"] == {"0": 1, "2": 3}
    # Pick 2 went to someone the page cannot show; the clock still counts it.
    assert got["offBoard"] == 1


def test_a_hand_mark_fills_an_unnamed_pick_without_counting_it_twice(page_js):
    server = {"picks": [{"player_id": 11, "by_me": False, "number": 1}], "picks_on_the_board": 2}
    got = _follow(page_js, server, extra=[1])
    assert got["order"] == [0, 1]
    assert got["offBoard"] == 0
    # A hand mark has no draft number, so no team's roster is guessed from it.
    assert "1" not in got["numbers"]


def test_a_hand_mark_the_server_later_names_becomes_the_servers(page_js):
    server = {
        "picks": [
            {"player_id": 11, "by_me": False, "number": 1},
            {"player_id": 12, "by_me": False, "number": 2},
        ],
        "picks_on_the_board": 2,
    }
    got = _follow(page_js, server, extra=[1])
    assert got["order"] == [0, 1]
    assert got["numbers"]["1"] == 2
    assert got["fromServer"] == [0, 1]


def test_my_own_hand_mark_goes_on_my_roster(page_js):
    got = _follow(page_js, {"picks": [], "picks_on_the_board": 1}, extra=[2], extra_mine=[2])
    assert got["roster"] == [2]


def test_a_pick_the_server_knows_and_the_page_does_not_still_moves_the_clock(page_js):
    server = {"picks": [{"player_id": 99, "by_me": False, "number": 1}], "picks_on_the_board": 1}
    got = _follow(page_js, server)
    assert got["order"] == [] and got["unknown"] == [99] and got["offBoard"] == 1
