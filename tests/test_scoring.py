"""Scoring is the one place where a quiet arithmetic error would corrupt every
downstream number without ever looking wrong, so the expected values here are
computed by hand in the test rather than by rerunning the code under test.

The fixture is the real league config from prompt.txt.
"""

import numpy as np
import pytest

from hockey.scoring import (
    ScoringConfigError,
    UnknownCategoryError,
    build_scoring,
    lookup,
    normalize_display,
    score_draws,
    score_statline,
)

# The league, exactly as stated: 14 teams, head-to-head points.
# Yahoo reports every category it knows and zeroes the unused ones, so the
# zero-modifier entries below are deliberate - they exercise the drop path.
LEAGUE_CATEGORIES = [
    ("G", 5.0),
    ("A", 3.0),
    ("+/-", 0.5),
    ("PPP", 0.5),
    ("SHP", 1.5),
    ("SOG", 0.5),
    ("HIT", 0.5),
    ("BLK", 0.5),
    ("GS", 1.0),
    ("W", 6.0),
    ("GA", -1.5),
    ("SV", 0.3),
    ("SHO", 4.0),
    # Not scored by this league.
    ("PIM", 0.0),
    ("FW", 0.0),
    ("SV%", 0.0),
]


@pytest.fixture
def scoring():
    return build_scoring("465.l.99999", LEAGUE_CATEGORIES)


def test_build_splits_skater_and_goalie_rules(scoring):
    assert set(scoring.skater_keys) == {
        "goals",
        "assists",
        "plus_minus",
        "ppp",
        "shp",
        "sog",
        "hits",
        "blocks",
    }
    assert set(scoring.goalie_keys) == {
        "games_started",
        "wins",
        "goals_against",
        "saves",
        "shutouts",
    }


def test_zero_modifier_categories_are_dropped(scoring):
    assert "pim" not in scoring.skater_keys
    assert "faceoffs_won" not in scoring.skater_keys


def test_skater_statline_matches_hand_computed_total(scoring):
    # McDavid-shaped night: 1G 2A, +2, 1 PPP, 0 SHP, 6 SOG, 1 hit, 1 block.
    line = {
        "goals": 1,
        "assists": 2,
        "plus_minus": 2,
        "ppp": 1,
        "shp": 0,
        "sog": 6,
        "hits": 1,
        "blocks": 1,
    }
    # 5 + 6 + 1.0 + 0.5 + 0 + 3.0 + 0.5 + 0.5
    assert score_statline(scoring, line, "P") == pytest.approx(16.5)


def test_shorthanded_point_is_worth_its_full_weight(scoring):
    """SHP at 1.5 is the category the app's schema never stored. A game with one
    is worth exactly 1.5 more than the same game without."""
    base = {
        "goals": 0,
        "assists": 0,
        "plus_minus": 0,
        "ppp": 0,
        "shp": 0,
        "sog": 2,
        "hits": 0,
        "blocks": 0,
    }
    with_shp = base | {"shp": 1}
    assert score_statline(scoring, with_shp, "P") - score_statline(scoring, base, "P") == (
        pytest.approx(1.5)
    )


def test_negative_plus_minus_reduces_the_total(scoring):
    line = {"goals": 0, "assists": 0, "plus_minus": -3, "sog": 0, "hits": 0, "blocks": 0}
    assert score_statline(scoring, line, "P") == pytest.approx(-1.5)


def test_goalie_statline_matches_hand_computed_total(scoring):
    # A 30-save shutout win: GS 1, W 1, GA 0, SV 30, SHO 1.
    line = {"games_started": 1, "wins": 1, "goals_against": 0, "saves": 30, "shutouts": 1}
    # 1 + 6 + 0 + 9.0 + 4
    assert score_statline(scoring, line, "G") == pytest.approx(20.0)


def test_goalie_blowout_loss_scores_negative(scoring):
    """Goals against at -1.5 has to be able to drive a start below zero, or the
    model's downside for goalies is wrong."""
    # Pulled after 7 goals on 20 shots: started, no win, 13 saves.
    line = {"games_started": 1, "wins": 0, "goals_against": 7, "saves": 13, "shutouts": 0}
    # 1 + 0 - 10.5 + 3.9 + 0
    assert score_statline(scoring, line, "G") == pytest.approx(-5.6)


def test_null_stats_count_as_zero(scoring):
    """ppp is NULL for a skater who dressed but never played. That skater did
    score zero power-play points, so NULL and 0 must agree."""
    with_null = {
        "goals": 1,
        "assists": 0,
        "ppp": None,
        "shp": None,
        "sog": 1,
        "hits": 0,
        "blocks": 0,
        "plus_minus": 0,
    }
    with_zero = with_null | {"ppp": 0, "shp": 0}
    assert score_statline(scoring, with_null, "P") == score_statline(scoring, with_zero, "P")


def test_goalie_rules_are_not_applied_to_skaters(scoring):
    """A skater line carrying a stray 'wins' key must not pick up 6 points."""
    line = {"goals": 1, "assists": 0, "plus_minus": 0, "sog": 0, "hits": 0, "blocks": 0, "wins": 1}
    assert score_statline(scoring, line, "P") == pytest.approx(5.0)


def test_unknown_category_is_rejected():
    with pytest.raises(UnknownCategoryError, match="ZZZ"):
        build_scoring("l", [("ZZZ", 1.0)])


def test_category_the_warehouse_cannot_serve_is_rejected():
    """Faceoff wins need play-by-play, which the default backfill skips. A
    league that scored them must fail loudly, not project everyone as if no one
    ever won a draw."""
    with pytest.raises(ScoringConfigError, match="no column"):
        build_scoring("l", [("G", 5.0), ("FW", 0.5)])


def test_rate_category_is_rejected():
    """Save percentage cannot be summed across starts."""
    with pytest.raises(ScoringConfigError, match="rate stats"):
        build_scoring("l", [("W", 6.0), ("SV%", 25.0)])


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("S", "SOG"), ("hits", "HIT"), ("BS", "BLK"), ("SO", "SHO"), (" G ", "G")],
)
def test_display_name_aliases(raw, expected):
    assert normalize_display(raw) == expected


def test_lookup_records_the_warehouse_column():
    assert lookup("HIT").source == "skater_game_logs.hits"
    assert lookup("SHP").source == "skater_game_logs.shp"
    assert lookup("FW").source is None


# --- posterior draws ---


def test_score_draws_matches_scoring_each_draw_individually(scoring):
    rng = np.random.default_rng(0)
    n = 500
    draws = {
        "goals": rng.poisson(0.5, n),
        "assists": rng.poisson(0.7, n),
        "plus_minus": rng.integers(-2, 3, n),
        "ppp": rng.poisson(0.3, n),
        "shp": rng.poisson(0.05, n),
        "sog": rng.poisson(3.0, n),
        "hits": rng.poisson(1.5, n),
        "blocks": rng.poisson(1.0, n),
    }
    vectorized = score_draws(scoring, draws, "P")
    one_at_a_time = np.array(
        [score_statline(scoring, {k: v[i] for k, v in draws.items()}, "P") for i in range(n)]
    )
    np.testing.assert_allclose(vectorized, one_at_a_time)


def test_score_draws_preserves_spread(scoring):
    """Scoring draw-by-draw must keep the variance that scoring the means
    would destroy. This is the whole reason the project is Bayesian."""
    rng = np.random.default_rng(1)
    n = 5000
    draws = {
        "goals": rng.poisson(0.5, n),
        "assists": rng.poisson(0.7, n),
        "plus_minus": np.zeros(n, dtype=int),
        "ppp": np.zeros(n, dtype=int),
        "shp": np.zeros(n, dtype=int),
        "sog": rng.poisson(3.0, n),
        "hits": np.zeros(n, dtype=int),
        "blocks": np.zeros(n, dtype=int),
    }
    scored = score_draws(scoring, draws, "P")
    assert scored.std() > 3.0
    mean_line = {k: v.mean() for k, v in draws.items()}
    assert score_statline(scoring, mean_line, "P") == pytest.approx(scored.mean(), rel=1e-9)


def test_score_draws_rejects_missing_category(scoring):
    draws = {"goals": np.zeros(3)}
    with pytest.raises(KeyError, match="missing scored categories"):
        score_draws(scoring, draws, "P")


def test_score_draws_rejects_misaligned_shapes(scoring):
    keys = scoring.skater_keys
    draws = {k: np.zeros(4) for k in keys}
    draws["goals"] = np.zeros(5)
    with pytest.raises(ValueError, match="mismatched shapes"):
        score_draws(scoring, draws, "P")
