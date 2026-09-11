"""Stat posteriors -> fantasy points -> a draft decision.

The step this covers is the one that makes modelling stats rather than points
worth doing, so the properties that justify it are tested directly: the
correlation survives, a missing category is refused rather than zeroed, and
re-scoring under different settings needs no refit.
"""

import numpy as np
import pandas as pd
import pytest

from hockey.export import beats, draft_board, fantasy_points, rank, what_if
from hockey.model.forecast import Projection
from hockey.scoring import build_scoring

SCORING = build_scoring("test", [("G", 5.0), ("A", 3.0), ("SOG", 0.5)])


def projection_from(totals: dict[str, np.ndarray], names: list[str]) -> Projection:
    players = list(range(len(names)))
    return Projection(
        totals=totals,
        players=players,
        player_names=dict(enumerate(names)),
        games_per_player=dict.fromkeys(players, 84),
    )


@pytest.fixture
def simple():
    # Two players, ten draws each, constructed so the answer is arithmetic.
    return projection_from(
        {
            "goals": np.array([[10.0, 20.0]] * 10),
            "assists": np.array([[30.0, 10.0]] * 10),
            "sog": np.array([[200.0, 100.0]] * 10),
        },
        ["Playmaker", "Sniper"],
    )


def test_fantasy_points_are_the_weighted_sum_of_the_draw(simple):
    points = fantasy_points(simple, SCORING)
    # Playmaker: 5*10 + 3*30 + 0.5*200 = 240. Sniper: 5*20 + 3*10 + 0.5*100 = 180.
    assert points[0, 0] == pytest.approx(240.0)
    assert points[0, 1] == pytest.approx(180.0)


def test_the_board_reports_floor_ceiling_and_spread(simple):
    board = draft_board(simple, SCORING)
    assert list(board["player"]) == ["Playmaker", "Sniper"]  # sorted by mean
    row = board.iloc[0]
    assert row["mean"] == pytest.approx(240.0)
    assert row["spread"] == pytest.approx(row["ceiling"] - row["floor"])
    # The one-in-five pair sits strictly inside the one-in-ten pair.
    assert row["floor"] <= row["p20"] <= row["mean"] <= row["p80"] <= row["ceiling"]


def test_a_missing_category_is_refused_not_scored_as_zero(simple):
    """A category absent from the projection understates every player by a
    similar amount, so the ranking still looks reasonable and is wrong."""
    partial = projection_from(
        {"goals": simple.totals["goals"], "assists": simple.totals["assists"]},
        ["Playmaker", "Sniper"],
    )
    with pytest.raises(KeyError, match="missing scored categories"):
        fantasy_points(partial, SCORING)


def test_correlation_between_categories_survives_the_sum():
    """The reason for scoring draw by draw. Two players with identical means
    per category, one whose categories move together and one whose move
    against each other, must not get the same spread."""
    rng = np.random.default_rng(0)
    n = 20000
    shared = rng.normal(0, 1, n)
    together_g = 30 + 5 * shared
    together_a = 60 + 8 * shared
    opposed_g = 30 + 5 * shared
    opposed_a = 60 - 8 * shared
    proj = projection_from(
        {
            "goals": np.column_stack([together_g, opposed_g]),
            "assists": np.column_stack([together_a, opposed_a]),
            "sog": np.full((n, 2), 200.0),
        },
        ["Together", "Opposed"],
    )
    board = draft_board(proj, SCORING).set_index("player")
    assert board.loc["Together", "mean"] == pytest.approx(board.loc["Opposed", "mean"], rel=0.02)
    # Same mean, very different risk - which averaging would have hidden.
    assert board.loc["Together", "sd"] > 3 * board.loc["Opposed", "sd"]


def test_head_to_head_uses_the_draws_not_the_means():
    rng = np.random.default_rng(1)
    n = 10000
    steady = np.full(n, 40.0)
    swingy = rng.normal(40.0, 15.0, n)
    proj = projection_from(
        {
            "goals": np.column_stack([steady, swingy]),
            "assists": np.full((n, 2), 50.0),
            "sog": np.full((n, 2), 200.0),
        },
        ["Steady", "Swingy"],
    )
    matrix = beats(proj, SCORING)
    assert matrix.loc["Steady", "Swingy"] == pytest.approx(0.5, abs=0.02)
    assert np.isnan(matrix.loc["Steady", "Steady"])


def test_risk_appetite_reorders_the_board():
    """Same mean, different shape: the safe pick and the upside pick are
    different players, and only a distribution can say so."""
    rng = np.random.default_rng(2)
    n = 20000
    proj = projection_from(
        {
            "goals": np.column_stack([rng.normal(40, 2, n), rng.normal(40, 18, n)]),
            "assists": np.full((n, 2), 50.0),
            "sog": np.full((n, 2), 200.0),
        },
        ["Steady", "Boom"],
    )
    board = draft_board(proj, SCORING)
    assert rank(board, "safe").iloc[0]["player"] == "Steady"
    assert rank(board, "upside").iloc[0]["player"] == "Boom"


def test_rank_rejects_an_unknown_appetite(simple):
    with pytest.raises(ValueError, match="appetite must be one of"):
        rank(draft_board(simple, SCORING), "reckless")


def test_rescoring_needs_no_refit(simple):
    """The weights live outside the model, so a settings change is one pass
    over an existing array. This is what modelling points directly would cost."""
    goal_heavy = build_scoring("test", [("G", 12.0), ("A", 3.0), ("SOG", 0.5)])
    comparison = what_if(simple, SCORING, goal_heavy)
    assert comparison.loc["Sniper", "rank_after"] == 1
    assert comparison.loc["Sniper", "rank_before"] == 2
    assert comparison.loc["Sniper", "rank_change"] == 1
    assert isinstance(comparison, pd.DataFrame)


def test_head_to_head_from_stacked_draws_matches_the_pairwise_definition():
    """Batches are independent given the shared parameters, so their draws
    stack side by side and P(A > B) is the same comparison it always was.
    Chunking is an implementation detail and must not change the answer."""
    from hockey.model.staged import head_to_head_from_draws

    rng = np.random.default_rng(3)
    draws = rng.normal(400, 80, (500, 7))
    names = [f"p{i}" for i in range(7)]
    matrix = head_to_head_from_draws(draws, names, chunk=3)
    for i in range(7):
        for j in range(7):
            if i == j:
                assert np.isnan(matrix.iloc[i, j])
            else:
                assert matrix.iloc[i, j] == pytest.approx((draws[:, i] > draws[:, j]).mean())


def test_head_to_head_chunking_is_invisible():
    from hockey.model.staged import head_to_head_from_draws

    rng = np.random.default_rng(4)
    draws = rng.normal(0, 1, (200, 9))
    names = [f"p{i}" for i in range(9)]
    one_go = head_to_head_from_draws(draws, names, chunk=9)
    chunked = head_to_head_from_draws(draws, names, chunk=2)
    pd.testing.assert_frame_equal(one_go, chunked)


def test_the_category_table_reports_every_scoring_input(simple):
    from hockey.export import category_table

    table = category_table(simple).set_index("player")
    for stat in simple.totals:
        assert stat in table.columns
        assert f"{stat}_floor" in table.columns
        assert f"{stat}_ceiling" in table.columns
    for player in table.index:
        for stat in simple.totals:
            row = table.loc[player]
            assert row[f"{stat}_floor"] <= row[stat] <= row[f"{stat}_ceiling"]
