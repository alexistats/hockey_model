"""The hot and cold flags: last season against the two before it."""

import pandas as pd
import pytest

from hockey.export.form import MIN_GAMES, previous_season, season_swing
from hockey.scoring import build_scoring

# Goals and assists only, so every expected value can be worked out by hand.
SCORING = build_scoring("test", [("G", 5.0), ("A", 3.0)])
LAST = 20252026
# A league whose scoring never changed, so the swing tests read the raw math.
FLAT = pd.DataFrame(1.0, index=[20232024, 20242025, LAST], columns=["goals", "assists"])


def line(player, season, games, goals, assists):
    return {
        "player_id": player,
        "season": season,
        "games": games,
        "goals": goals,
        "assists": assists,
    }


def test_seasons_step_back_one_at_a_time():
    assert previous_season(20252026) == 20242025
    assert previous_season(previous_season(20252026)) == 20232024


def test_the_swing_is_last_season_against_the_average_of_the_two_before():
    totals = pd.DataFrame(
        [
            line(1, 20232024, 80, 40, 80),  # 200 + 240 = 440 points, 5.50 a game
            line(1, 20242025, 80, 40, 100),  # 200 + 300 = 500 points, 6.25 a game
            line(1, 20252026, 80, 56, 80),  # 280 + 240 = 520 points, 6.50 a game
        ]
    )
    out = season_swing(totals, SCORING, LAST, FLAT).set_index("player_id")
    assert out.loc[1, "before_per_game"] == pytest.approx(5.875, abs=0.005)
    assert out.loc[1, "per_game"] == pytest.approx(6.5)
    assert out.loc[1, "swing"] == pytest.approx(6.5 / 5.875 - 1, abs=0.0005)
    assert out.loc[1, "flag"] == ""  # +10.6%: under the flag


def test_hot_and_cold_at_twelve_percent_either_way():
    totals = pd.DataFrame(
        [
            # 5.0 a game in both earlier seasons: 400 points over 80 games.
            *(line(p, s, 80, 50, 50) for p in (1, 2, 3) for s in (20232024, 20242025)),
            line(1, LAST, 80, 58, 50),  # 440 / 80 = 5.5  -> +10%, not hot
            line(2, LAST, 80, 62, 50),  # 460 / 80 = 5.75 -> +15%, hot
            line(3, LAST, 80, 38, 50),  # 340 / 80 = 4.25 -> -15%, cold
        ]
    )
    out = season_swing(totals, SCORING, LAST, FLAT).set_index("player_id")
    assert out["flag"].to_dict() == {1: "", 2: "hot", 3: "cold"}
    assert out.loc[2, "swing"] == pytest.approx(0.15)
    assert out.loc[3, "games"] == 80 and out.loc[3, "before_games"] == 160


def test_a_thin_or_missing_season_means_no_comparison_at_all():
    totals = pd.DataFrame(
        [
            # Rookie: only last season.
            line(1, LAST, 82, 45, 70),
            # Two seasons, like a second-year player: nothing to average.
            line(2, 20242025, 70, 25, 38),
            line(2, LAST, 82, 45, 70),
            # Three seasons, but one of them a half season.
            line(3, 20232024, MIN_GAMES - 1, 20, 20),
            line(3, 20242025, 80, 30, 40),
            line(3, LAST, 80, 35, 45),
        ]
    )
    assert season_swing(totals, SCORING, LAST, FLAT).empty


def test_every_scored_category_has_to_be_there():
    totals = pd.DataFrame([line(1, s, 80, 30, 40) for s in (20232024, 20242025, LAST)])
    with pytest.raises(KeyError, match="shp"):
        season_swing(totals, build_scoring("test", [("G", 5.0), ("SHP", 1.5)]), LAST)


def test_earlier_seasons_are_read_in_last_seasons_scoring_environment():
    # The whole league scored 20% fewer goals a game last season. A player whose
    # goals fell by exactly as much has not had a cold season by his standards.
    league = [line(9, sn, 80, g, 0) for sn, g in ((20232024, 50), (20242025, 50), (LAST, 40))]
    totals = pd.DataFrame(
        [
            *league,
            line(1, 20232024, 80, 25, 0),
            line(1, 20242025, 80, 25, 0),
            line(1, LAST, 80, 20, 0),
        ]
    )
    out = season_swing(totals, SCORING, LAST).set_index("player_id")
    assert out.loc[1, "swing"] == pytest.approx(0.0, abs=0.001)
    assert out.loc[1, "flag"] == ""
    # Unadjusted it would have been a 20% drop, and cold.
    assert out.loc[1, "before_per_game"] == pytest.approx(out.loc[1, "per_game"], abs=0.01)
