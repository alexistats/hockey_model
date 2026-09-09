"""Forward-simulate a season from the multi-category posterior.

Each draw of the posterior gives a player a latent rate for every category and
a probability of dressing. For each remaining game, that draw decides whether
the player plays, and if so draws their stat line. Summing over the schedule
turns a posterior over rates into a posterior over season totals.

Doing the availability draw and the stat draws inside the same posterior draw
is what makes the floor honest. A player whose availability posterior is wide
gets seasons where they play 80 games and seasons where they play 45, and the
low ones drag the 5th percentile down - which is exactly the risk a draft pick
is trying to price, and exactly what an assumed 84 games hides.
"""

import logging
from dataclasses import dataclass

import numpy as np
import xarray as xr

from hockey.model.forecast import Projection
from hockey.model.multi import SIGNED_STAT, MultiData

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MultiProjection(Projection):
    """Adds the games-played distribution to the stat totals."""

    games_played: np.ndarray = None  # (draws, players)


def _stack(posterior: xr.Dataset, name: str) -> np.ndarray:
    return posterior[name].stack(sample=("chain", "draw")).transpose("sample", ...).to_numpy()


def project(
    idata,
    data: MultiData,
    seed: int = 20262027,
    assume_full_season: bool = False,
) -> MultiProjection:
    """Season totals per category, over the schedule each player actually has.

    `assume_full_season=True` ignores the availability component, which is only
    useful for showing what it is worth: it is the assumption the MVP made
    implicitly, and comparing the two is the cleanest way to see how much of a
    player's downside comes from missing games rather than from scoring less.
    """
    rng = np.random.default_rng(seed)
    posterior = idata.posterior
    stats = list(data.stats)
    index = data.player_index
    schedule = data.schedule

    game_player = schedule["player_id"].map(index).to_numpy()
    game_opponent = schedule["opponent_idx"].to_numpy()
    game_home = schedule["is_home"].to_numpy(dtype=float)
    n_players, n_games = len(data.players), len(schedule)

    indicator = np.zeros((n_games, n_players))
    indicator[np.arange(n_games), game_player] = 1.0

    # The last walk step is the projected season, so the drift into a season
    # nobody has played is carried rather than the latent state being frozen.
    mu_player = _stack(posterior, "mu_player")[:, :, :, -1]  # (draws, stat, player)
    opponent = _stack(posterior, "opponent")[:, :, :, -1]  # (draws, stat, team)
    b_home = _stack(posterior, "b_home")  # (draws, stat)
    n_draws = mu_player.shape[0]

    if assume_full_season:
        plays = np.ones((n_draws, n_games), dtype=bool)
    else:
        p_play = 1.0 / (1.0 + np.exp(-_stack(posterior, "logit_avail")[:, :, -1]))
        plays = rng.random((n_draws, n_games)) < p_play[:, game_player]

    totals: dict[str, np.ndarray] = {}
    for k, stat in enumerate(stats):
        log_rate = (
            mu_player[:, k, game_player]
            + opponent[:, k, game_opponent]
            + b_home[:, k, None] * game_home[None, :]
        )
        counts = rng.poisson(np.exp(log_rate)) * plays
        totals[stat] = counts @ indicator

    # Plus/minus rides the same availability draw so it is missing from the
    # same games as everything else. Normal rather than Poisson because it is
    # signed; the season total is rounded because the league scores whole
    # plus/minus, not a fractional one.
    pm_player = _stack(posterior, "pm_player")
    b_home_pm = _stack(posterior, "b_home_pm")
    sigma_pm = _stack(posterior, "sigma_pm")
    pm_mu = pm_player[:, game_player] + b_home_pm[:, None] * game_home[None, :]
    pm_draws = rng.normal(pm_mu, sigma_pm[:, None]) * plays
    totals[SIGNED_STAT] = np.rint(pm_draws @ indicator)

    games_played = plays.astype(float) @ indicator
    logger.info(
        "projected %d categories over %d draws for %d players (%s)",
        len(stats),
        n_draws,
        n_players,
        "full season assumed" if assume_full_season else "availability modelled",
    )
    return MultiProjection(
        totals=totals,
        players=data.players,
        player_names=data.player_names,
        games_per_player={
            int(pid): int(n) for pid, n in schedule.groupby("player_id").size().items()
        },
        games_played=games_played,
    )
