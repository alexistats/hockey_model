"""The in-season update: the preseason posterior as the prior, this season's games as data.

The properties that make it worth having: the prior's spread is the rate's, not a
season's noise; no games leaves the projection where it was; games move it toward
the pace in proportion to how much evidence the prior is worth; and the rest of the
season never has more games in it than the schedule does.
"""

import numpy as np
import pytest

from hockey.model import in_season as ins


def _draws(rate=0.3, cv=0.2, seed=0, n=4000):
    rng = np.random.default_rng(seed)
    games = rng.integers(55, 83, n).astype(float)
    shape = 1 / cv**2
    true = rng.gamma(shape, rate / shape, n)
    prior = {c: rng.poisson(true * games).astype(float) for c in ins.COUNTS}
    prior[ins.SIGNED] = np.rint(rng.normal(0.02 * games, ins.SIGMA_PM * np.sqrt(games)))
    return prior, games, true


def test_the_prior_is_the_rates_spread_not_a_seasons_noise():
    prior, games, true = _draws(cv=0.2)
    m, v = ins.rate_prior(prior["goals"], games)
    assert m == pytest.approx(true.mean(), rel=0.02)
    # Uncorrected, the season's Poisson noise would roughly double this variance.
    assert np.sqrt(v) == pytest.approx(true.std(), rel=0.1)


def test_the_gamma_update_is_prior_plus_games():
    a, b = ins.gamma_update(0.3, 0.3**2 / 50, events=0, n=0)  # a prior worth 50/0.3 games
    assert a / b == pytest.approx(0.3)
    a2, b2 = ins.gamma_update(0.3, 0.3**2 / 50, events=12, n=20)
    assert (a2, b2) == pytest.approx((a + 12, b + 20))


def test_no_games_leaves_the_projection_where_it_was():
    prior, games, _ = _draws()
    got = ins.update_skater(prior, games, 82, ins.Seen(0, {}, 0, 82), np.random.default_rng(1))
    for c in ins.COUNTS:
        assert got["rates"][c] == pytest.approx(ins.rate_prior(prior[c], games)[0], rel=1e-9)


def test_games_move_the_rate_toward_the_pace_more_as_they_accumulate():
    prior, games, _ = _draws(rate=0.3, cv=0.25)
    rng = np.random.default_rng(2)
    pace = 0.6
    moved = []
    for n in (4, 20, 40, 80):
        seen = ins.Seen(n, {c: pace * n for c in ins.COUNTS}, n, 82 - n)
        moved.append(ins.update_skater(prior, games, 82, seen, rng)["rates"]["goals"])
    assert moved == sorted(moved)
    share = [(m - 0.3) / (pace - 0.3) for m in moved]
    assert 0 < share[0] < 0.1  # four games barely move him
    assert share[-1] < 1  # never all the way to the pace


def test_the_rest_of_the_season_fits_in_the_schedule():
    prior, games, _ = _draws()
    seen = ins.Seen(30, {c: 9.0 for c in ins.COUNTS}, 30, 54)
    got = ins.update_skater(prior, games, 84, seen, np.random.default_rng(3))
    assert got["draws"]["games"].max() <= 54


def test_games_missed_move_availability_only_when_asked():
    # Off by default: on 2025-26 held out, updating it from games missed made the
    # rest-of-season games worse, because absences come in runs.
    prior, games, _ = _draws()
    missed_half = ins.Seen(10, {c: 3.0 for c in ins.COUNTS}, 20, 62)
    rng = np.random.default_rng(5)
    kept = ins.update_skater(prior, games, 82, missed_half, rng)
    moved = ins.update_skater(prior, games, 82, missed_half, rng, availability=True)
    assert moved["avail"] < kept["avail"]
    assert moved["draws"]["games"].mean() < kept["draws"]["games"].mean()


def test_a_goalie_who_starts_more_than_expected_gets_a_bigger_share():
    rng = np.random.default_rng(4)
    points = rng.normal(300, 40, 4000)
    quiet = ins.update_goalie(
        points, 40, 84, starts=2, points=14.0, team_games=10, remaining=74, sigma_start=6.5, rng=rng
    )
    busy = ins.update_goalie(
        points, 40, 84, starts=9, points=63.0, team_games=10, remaining=74, sigma_start=6.5, rng=rng
    )
    assert busy["share"] > quiet["share"]
    assert busy["starts"].mean() > quiet["starts"].mean()
