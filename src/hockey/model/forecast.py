"""Posterior predictive over the remaining schedule.

Inference and prediction are kept apart on purpose. The fit produces a
posterior over latent rates; this module forward-simulates the games that are
still to come from that posterior. Two things fall out of the separation:

- NUTS only ever has to move through continuous parameters. Projected game
  counts are discrete, and the original model got them sampled by Metropolis
  as a side effect of PyMC's default sampler assignment, which mixes badly.

- The daily re-run is cheap. Conditioning on the games played since the last
  fit means re-running this, not re-fitting: one pass of Poisson draws over a
  shortened schedule, seconds rather than an overnight batch.

Every projected count is drawn from the parameter draw that produced it, so a
draw where a player's latent rate is high is the same draw where their totals
are high. Categories stay correlated for the same reason, which is what makes
the fantasy-point posterior honest rather than a sum of independent averages.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from hockey.model.skater import ModelData

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Projection:
    """Projected rest-of-season totals: {stat: (draws, players)}."""

    totals: dict[str, np.ndarray]
    players: list[int]
    player_names: dict[int, str]
    games_per_player: dict[int, int]

    @property
    def n_draws(self) -> int:
        return next(iter(self.totals.values())).shape[0]

    def draws_for(self, player_id: int, stat: str) -> np.ndarray:
        return self.totals[stat][:, self.players.index(player_id)]


def _stack_draws(posterior: xr.Dataset, name: str) -> np.ndarray:
    """Chains and draws flattened into one axis, chain-major."""
    return posterior[name].stack(sample=("chain", "draw")).transpose("sample", ...).to_numpy()


def project(
    idata,
    data: ModelData,
    seed: int = 20262027,
    availability: np.ndarray | None = None,
) -> Projection:
    """Draw rest-of-season totals for every modelled player.

    `availability` is an optional per-player probability that the player
    dresses for any given remaining game. Left as None, every player is
    assumed to play every remaining game - which is the MVP's known
    limitation, not a modelling claim, and is why the availability component
    is the next piece of work rather than an optional extra.
    """
    rng = np.random.default_rng(seed)
    posterior = idata.posterior
    schedule = data.schedule
    player_index = data.player_index

    game_player = schedule["player_id"].map(player_index).to_numpy()
    game_opponent = schedule["opponent_idx"].to_numpy()
    game_home = schedule["is_home"].to_numpy(dtype=float)
    n_players = len(data.players)

    # Sums each player's own remaining games in one matrix multiply.
    indicator = np.zeros((len(schedule), n_players))
    indicator[np.arange(len(schedule)), game_player] = 1.0

    totals: dict[str, np.ndarray] = {}
    for stat in data.stats:
        # The last walk step is the projected season: the walk and the AR
        # process each take one more step past the last observed season, so
        # year-to-year drift is carried into the forecast instead of the
        # latent state being frozen at last season's value.
        mu_player = _stack_draws(posterior, f"{stat}_mu_player")[:, :, -1]
        opponent = _stack_draws(posterior, f"{stat}_opponent")[:, :, -1]
        b_home = _stack_draws(posterior, f"{stat}_b_home")

        log_rate = (
            mu_player[:, game_player]
            + opponent[:, game_opponent]
            + b_home[:, None] * game_home[None, :]
        )
        counts = rng.poisson(np.exp(log_rate))
        if availability is not None:
            played = rng.random(counts.shape) < availability[None, game_player]
            counts = counts * played
        totals[stat] = counts @ indicator

    games_per_player = {int(pid): int(n) for pid, n in schedule.groupby("player_id").size().items()}
    logger.info(
        "projected %d stats over %d draws for %d players",
        len(totals),
        next(iter(totals.values())).shape[0],
        n_players,
    )
    return Projection(
        totals=totals,
        players=data.players,
        player_names=data.player_names,
        games_per_player=games_per_player,
    )


def summarize(projection: Projection, extra_totals: dict[str, np.ndarray] | None = None):
    """Mean, floor, ceiling and spread per player and stat.

    Floor is the 5th percentile and ceiling the 95th, matching the definition
    the original analysis.py used so the numbers stay comparable.
    """
    rows = []
    everything = dict(projection.totals) | (extra_totals or {})
    for stat, values in everything.items():
        for i, player_id in enumerate(projection.players):
            draws = values[:, i]
            rows.append(
                {
                    "player": projection.player_names[player_id],
                    "stat": stat,
                    "games": projection.games_per_player.get(player_id, 0),
                    "mean": draws.mean(),
                    "floor_p5": np.percentile(draws, 5),
                    "ceiling_p95": np.percentile(draws, 95),
                    "sd": draws.std(),
                }
            )
    return pd.DataFrame(rows).sort_values(["player", "stat"]).reset_index(drop=True)


def head_to_head(projection: Projection, values: np.ndarray):
    """P(row finishes ahead of column), compared draw by draw.

    Means cannot answer this. Two players with the same projection can have
    very different odds of finishing ahead of each other, and that difference
    is the whole input to a risk-tunable draft strategy.
    """
    names = [projection.player_names[p] for p in projection.players]
    matrix = pd.DataFrame(index=names, columns=names, dtype=float)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            matrix.loc[a, b] = np.nan if i == j else float((values[:, i] > values[:, j]).mean())
    return matrix
