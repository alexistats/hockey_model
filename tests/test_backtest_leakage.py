"""The held-out season must not reach the fit.

Every calibration number rests on this one property. If any of the test
season's games, or its games-played counts, get into the training panel, the
intervals will look beautifully calibrated and mean nothing - and the failure
is invisible in the output, because well-calibrated is exactly what leakage
produces.

These run against the real warehouse and skip when it is not up, since the
thing being tested is a query.
"""

import pandas as pd
import pytest
from sqlalchemy import text

from hockey.seasons import FITTING_SEASONS

TEST_SEASON = 20252026


@pytest.fixture(scope="module")
def session():
    from sqlalchemy.exc import OperationalError

    from hockey.db import SessionLocal

    try:
        with SessionLocal() as s:
            s.execute(text("SELECT 1"))
            yield s
    except OperationalError:
        pytest.skip("warehouse not reachable; start it with docker compose up -d")


@pytest.fixture(scope="module")
def prepared(session):
    from hockey.calibration.backtest import prepare_backtest

    return prepare_backtest(session, TEST_SEASON, pool_size=6)


def test_training_seasons_stop_before_the_held_out_one(prepared):
    data, _ = prepared
    assert TEST_SEASON not in data.maps.seasons
    assert data.maps.seasons == [s for s in FITTING_SEASONS if s < TEST_SEASON]


def test_no_game_from_the_held_out_season_is_in_the_panel(prepared):
    data, _ = prepared
    assert data.panel["season"].max() < TEST_SEASON


def test_no_games_played_count_from_the_held_out_season_is_in_the_fit(prepared):
    """Games played is the quantity the availability component predicts, so
    leaking it would make that part of the calibration meaningless."""
    data, _ = prepared
    assert data.availability["season"].max() < TEST_SEASON


def test_the_projection_step_is_the_held_out_season(prepared):
    """One step past the last trained season - so the walk drifts into the
    season being scored rather than freezing at the last one it saw."""
    data, _ = prepared
    assert data.projection_step == len(data.maps.seasons)
    assert data.n_steps == len(data.maps.seasons) + 1


def test_the_schedule_is_the_held_out_season(prepared, session):
    """The team is roster information a drafter has. The schedule has to be
    the real one, or the projection is over the wrong number of games."""
    data, _ = prepared
    game_ids = data.schedule["game_id"].unique().tolist()
    seasons = pd.DataFrame(
        session.execute(
            text("SELECT DISTINCT season FROM nhl_games WHERE nhl_game_id = ANY(:ids)"),
            {"ids": game_ids},
        ).mappings()
    )
    assert seasons["season"].tolist() == [TEST_SEASON]


def test_the_pool_is_chosen_without_hindsight(prepared):
    """A pool picked on test-season performance would flatter every number
    that follows it."""
    from hockey.calibration.backtest import POOL_SQL

    assert ":train_recent" in POOL_SQL
    assert str(TEST_SEASON) not in POOL_SQL


def test_actuals_come_only_from_the_held_out_season(prepared):
    data, actuals = prepared
    assert set(actuals["player_id"]) <= set(data.players)
    assert (actuals["games_played"] > 0).all()
    # A season cannot have produced more games than it holds.
    assert actuals["games_played"].max() <= 82
