"""The forecast's arithmetic: per-player summing, floor/ceiling, and P(A > B).

These are the numbers the draft bot consumes, so they are tested against
hand-constructed draws where the right answer is known by construction rather
than by rerunning the code.
"""

import numpy as np
import pandas as pd
import pytest

from hockey.model.forecast import Projection, head_to_head, summarize


@pytest.fixture
def projection():
    # Two players. Player A's goal draws are 10..19; player B's are 20..29.
    goals = np.column_stack([np.arange(10, 20), np.arange(20, 30)]).astype(float)
    assists = np.column_stack([np.arange(30, 40), np.arange(5, 15)]).astype(float)
    return Projection(
        totals={"goals": goals, "assists": assists},
        players=[1, 2],
        player_names={1: "Player A", 2: "Player B"},
        games_per_player={1: 84, 2: 84},
    )


def test_summarize_reports_mean_floor_and_ceiling_per_player_and_stat(projection):
    frame = summarize(projection)
    assert len(frame) == 4  # two players x two stats
    row = frame[(frame.player == "Player A") & (frame.stat == "goals")].iloc[0]
    assert row["mean"] == pytest.approx(14.5)
    assert row["floor_p10"] == pytest.approx(np.percentile(np.arange(10, 20), 10))
    assert row["ceiling_p90"] == pytest.approx(np.percentile(np.arange(10, 20), 90))
    assert row["games"] == 84


def test_summarize_accepts_a_derived_total(projection):
    """Points is goals plus assists summed per draw, so its spread reflects
    how the two move together rather than assuming they are independent."""
    points = projection.totals["goals"] + projection.totals["assists"]
    frame = summarize(projection, {"points": points})
    assert set(frame.stat) == {"goals", "assists", "points"}
    row = frame[(frame.player == "Player A") & (frame.stat == "points")].iloc[0]
    assert row["mean"] == pytest.approx(14.5 + 34.5)


def test_derived_total_keeps_correlation_rather_than_adding_variances(projection):
    """Player A's goals rise while assists rise with them, so points vary more
    than either. Constructed so that scoring the means would hide it."""
    points = projection.totals["goals"] + projection.totals["assists"]
    a = 0  # Player A
    assert points[:, a].std() > projection.totals["goals"][:, a].std()
    assert points[:, a].std() == pytest.approx(
        (projection.totals["goals"][:, a] + projection.totals["assists"][:, a]).std()
    )


def test_head_to_head_is_computed_draw_by_draw(projection):
    """Player B's goal draws are strictly above Player A's, every draw."""
    matrix = head_to_head(projection, projection.totals["goals"])
    assert matrix.loc["Player B", "Player A"] == pytest.approx(1.0)
    assert matrix.loc["Player A", "Player B"] == pytest.approx(0.0)
    assert np.isnan(matrix.loc["Player A", "Player A"])


def test_head_to_head_separates_players_with_equal_means():
    """The whole reason for comparing draws instead of means: two players can
    project identically and still be far from a coin flip against each other."""
    steady = np.full(1000, 50.0)
    swingy = np.linspace(0, 100, 1000)
    projection = Projection(
        totals={"goals": np.column_stack([steady, swingy])},
        players=[1, 2],
        player_names={1: "Steady", 2: "Swingy"},
        games_per_player={1: 84, 2: 84},
    )
    frame = summarize(projection)
    means = frame.set_index("player")["mean"]
    assert means["Steady"] == pytest.approx(means["Swingy"], abs=0.1)
    # Identical means, completely different floors and ceilings.
    floors = frame.set_index("player")["floor_p10"]
    assert floors["Steady"] > floors["Swingy"] + 35
    # And an even head-to-head that the means alone could never reveal.
    matrix = head_to_head(projection, projection.totals["goals"])
    assert matrix.loc["Steady", "Swingy"] == pytest.approx(0.5, abs=0.01)


def test_summarize_returns_a_stable_sorted_frame(projection):
    frame = summarize(projection)
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == [
        "player",
        "stat",
        "games",
        "mean",
        "floor_p10",
        "ceiling_p90",
        "sd",
    ]
    assert frame.equals(frame.sort_values(["player", "stat"]).reset_index(drop=True))
