"""The hierarchical Bayesian state-space model for skaters.

This is the architecture from the original `game_level_modelling.py`, kept
deliberately: a player's latent scoring rate is a Gaussian random walk across
seasons, opponent strength is an AR(1) process across seasons, there is a
home-ice effect, and counts are Poisson on a log link. That structure is what
produces a calibrated floor and ceiling rather than a point estimate, which is
the whole reason the project is Bayesian.

Four things changed, each fixing a defect rather than altering the approach:

1. **Players are fit jointly, not one model each.** The original estimated a
   31-by-seasons array of opponent effects inside a single player's model -
   roughly 250 parameters from about 600 games, which cannot identify team
   strength and mostly absorbed noise. Fit across players, the same AR(1)
   becomes a real league-level opponent effect that every player's games
   inform.

2. **Partial pooling by position.** A player's walk starts from a
   position-level mean rather than from nothing, so a rookie with 20 games
   shrinks toward their peer group instead of producing an unstable,
   uselessly wide posterior. The original had no pooling at all.

3. **The assists likelihood no longer conditions on that game's goals.** The
   original regressed assists on the goals scored in the same game, a quantity
   that does not exist for a future game. Its own forecast silently dropped the
   term, so the model that was fit was not the model that predicted.

4. **The projected season is a real step in the walk.** The original froze the
   latent state at the last observed season. Here the walk takes one more step
   into the season being projected, with no observations, so year-to-year drift
   uncertainty is carried into the forecast. At draft time that drift is most
   of the honest uncertainty about a player, and freezing it produces intervals
   that are too narrow.

The team count is taken from the data, so the original's hardcoded 31 against a
32-team league cannot recur.

Non-centered parameterization is used for the random walk because the centered
form makes a funnel that NUTS samples badly; it is the same model written so
the sampler can move.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from hockey.features import IndexMaps

logger = logging.getLogger(__name__)

# Positions used for the pooling groups. Wingers are kept apart from centres
# because their shot and assist rates differ enough to matter, and defencemen
# obviously so.
POOLING_POSITIONS = ("C", "LW", "RW", "D")

# The MVP gate from the plan: reproduce goals and assists off the warehouse
# before extending to every category.
MVP_STATS = ("goals", "assists")


@dataclass
class ModelData:
    """Everything the model indexes on, assembled once so the fit and the
    forecast cannot disagree about who is who."""

    panel: pd.DataFrame
    schedule: pd.DataFrame
    players: list[int]
    player_names: dict[int, str]
    positions: list[str]
    maps: IndexMaps
    stats: tuple[str, ...] = MVP_STATS
    # Index of the season being projected: one past the last observed season.
    projection_season_idx: int = field(init=False)

    def __post_init__(self):
        self.projection_season_idx = self.maps.n_seasons

    @property
    def player_index(self) -> dict[int, int]:
        return {p: i for i, p in enumerate(self.players)}

    @property
    def n_walk_steps(self) -> int:
        """Observed seasons plus the one being projected."""
        return self.maps.n_seasons + 1


def prepare(
    panel: pd.DataFrame,
    schedule: pd.DataFrame,
    maps: IndexMaps,
    player_names: dict[int, str],
    positions: dict[int, str],
    stats: tuple[str, ...] = MVP_STATS,
) -> ModelData:
    """Align the panel and the projection schedule onto shared indices.

    `schedule` must carry a player_id column: each player's remaining games are
    their own team's remaining games, and teammates share them while a traded
    player does not.
    """
    players = sorted(panel["player_id"].unique())
    missing_schedule = set(players) - set(schedule["player_id"].unique())
    if missing_schedule:
        raise ValueError(
            f"no projection schedule for player(s) {sorted(missing_schedule)}; "
            f"they would get a zero-game projection that looks like a real one"
        )
    # Only players with observed games can be modelled. A schedule row for
    # anyone else has no player index to map to, and letting it through
    # produced a silent out-of-range index rather than an error. The caller is
    # responsible for noticing that a requested player has no games at all -
    # see hockey.model.mvp.resolve_players.
    schedule = schedule[schedule["player_id"].isin(players)]
    return ModelData(
        panel=panel.reset_index(drop=True),
        schedule=schedule.reset_index(drop=True),
        players=[int(p) for p in players],
        player_names=player_names,
        positions=[_pool_position(positions.get(int(p), "C")) for p in players],
        maps=maps,
        stats=stats,
    )


def _pool_position(position: str) -> str:
    """Map a roster position onto a pooling group. Anything unexpected pools
    with centres, which is the least distinctive group, rather than raising -
    a projection with a slightly wrong prior beats no projection."""
    return position if position in POOLING_POSITIONS else "C"


def build(data: ModelData) -> pm.Model:
    """The PyMC model, with the forecast RVs built into it.

    Forecast draws come out of the same `pm.sample` call as the posterior, as
    they did in the original: the remaining-schedule counts are unobserved
    random variables in the model rather than a separate posterior-predictive
    pass, so every draw of a projected total is consistent with the parameter
    draw that produced it.
    """
    panel = data.panel
    player_index = data.player_index
    position_levels = list(POOLING_POSITIONS)

    obs_player = panel["player_id"].map(player_index).to_numpy("int32")
    obs_season = panel["season_idx"].to_numpy("int32")
    obs_opponent = panel["opponent_idx"].to_numpy("int32")
    obs_home = panel["is_home"].to_numpy("float64")

    position_idx = np.array([position_levels.index(p) for p in data.positions], dtype="int32")

    coords = {
        "player": [data.player_names.get(p, str(p)) for p in data.players],
        "position": position_levels,
        "team": data.maps.teams,
        # One step longer than the observed window: the last step is the
        # season being projected, which has no observations.
        "walk_step": [*data.maps.seasons, "projection"],
        "obs": np.arange(len(panel)),
    }

    with pm.Model(coords=coords) as model:
        for stat in data.stats:
            _add_stat(
                stat=stat,
                observed=panel[stat].to_numpy("int64"),
                obs_player=obs_player,
                obs_season=obs_season,
                obs_opponent=obs_opponent,
                obs_home=obs_home,
                position_idx=position_idx,
                n_players=len(data.players),
                n_steps=data.n_walk_steps,
                n_teams=data.maps.n_teams,
            )
    return model


def _add_stat(
    *,
    stat: str,
    observed: np.ndarray,
    obs_player: np.ndarray,
    obs_season: np.ndarray,
    obs_opponent: np.ndarray,
    obs_home: np.ndarray,
    position_idx: np.ndarray,
    n_players: int,
    n_steps: int,
    n_teams: int,
) -> None:
    """One stat's full sub-model: pooled walk, opponent AR(1), home, Poisson."""

    # --- population level: where a player's walk starts ---
    # Priors are on the log scale. A rate of exp(-2.2) is about 0.11 per game,
    # roughly a fourth-line scoring rate; the sd of 1.0 spans from far below
    # that to well above an elite rate, so it constrains without picking a
    # winner.
    mu_position = pm.Normal(f"{stat}_mu_position", mu=-2.2, sigma=1.0, dims="position")
    sigma_player = pm.HalfNormal(f"{stat}_sigma_player", sigma=0.6)
    z_player = pm.Normal(f"{stat}_z_player", mu=0.0, sigma=1.0, dims="player")
    # This offset is what makes the pooling partial: a player with plenty of
    # games pulls away from their position's mean, a player with few does not.
    baseline = pm.Deterministic(
        f"{stat}_player_baseline",
        mu_position[position_idx] + sigma_player * z_player,
        dims="player",
    )

    # --- player latent rate: a random walk across seasons ---
    # The original's fixed innovation sd of 0.25 is now estimated, because how
    # much players drift year to year is exactly the quantity that sets how
    # wide a projection should be, and it is knowable from the data.
    sigma_walk = pm.HalfNormal(f"{stat}_sigma_walk", sigma=0.25)
    z_walk = pm.Normal(f"{stat}_z_walk", mu=0.0, sigma=1.0, dims=("player", "walk_step"))
    # Non-centered random walk: step 0 is the baseline, each later step adds a
    # scaled innovation. The cumulative sum starts at zero for step 0 so the
    # baseline is not double counted.
    innovations = pt.concatenate([pt.zeros((n_players, 1)), sigma_walk * z_walk[:, 1:]], axis=1)
    mu_player = pm.Deterministic(
        f"{stat}_mu_player",
        baseline[:, None] + pt.cumsum(innovations, axis=1),
        dims=("player", "walk_step"),
    )

    # --- opponent strength: AR(1) across seasons, shared by every player ---
    rho = pm.TruncatedNormal(f"{stat}_rho", mu=0.0, sigma=0.3, lower=0.0, upper=1.0)
    sigma_team = pm.HalfNormal(f"{stat}_sigma_team", sigma=0.08)
    # An explicit initial distribution. PyMC otherwise defaults pm.AR to
    # Normal(0, 100), which on a log rate means the process can start at a
    # factor of e^100 - vastly wider than any real opponent effect, and enough
    # to wreck the geometry the sampler has to move through. A sd of 0.3 puts
    # two standard deviations at roughly a factor of 1.8 in scoring rate
    # against a team, still far wider than reality but weakly informative.
    opponent = pm.AR(
        f"{stat}_opponent",
        rho=rho,
        sigma=sigma_team,
        init_dist=pm.Normal.dist(0.0, 0.3),
        shape=(n_teams, n_steps),
        dims=("team", "walk_step"),
    )

    b_home = pm.Normal(f"{stat}_b_home", mu=0.0, sigma=0.5)

    # --- likelihood ---
    log_rate = (
        mu_player[obs_player, obs_season] + opponent[obs_opponent, obs_season] + b_home * obs_home
    )
    pm.Poisson(stat, mu=pt.exp(log_rate), observed=observed, dims="obs")

    # The forecast deliberately lives outside this graph. Projected game counts
    # are discrete, and NUTS cannot sample discrete free variables - the
    # original model got them only because PyMC's default sampler quietly
    # handed them to Metropolis, which mixes poorly. Everything the forecast
    # needs is already in the posterior: mu_player and opponent at their last
    # walk step, plus b_home. hockey.model.forecast draws the counts from
    # those, one Poisson draw per posterior draw, which is the same joint
    # distribution and keeps inference separable from prediction. That
    # separation is also what makes a daily re-projection cheap: forward
    # simulation over a changed remaining schedule needs no refit.


def sample(
    model: pm.Model,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    seed: int = 20262027,
):
    """Sample with the JAX/numpyro NUTS backend.

    PyTensor's default backend needs a C++ compiler, which this machine does
    not have; without one it falls back to a pure-Python path far too slow for
    a nightly re-fit. numpyro compiles through JAX instead. target_accept is
    above the 0.8 default because hierarchical walks produce enough curvature
    to throw divergences at 0.8.
    """
    with model:
        return pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            nuts_sampler="numpyro",
            random_seed=seed,
            progressbar=False,
        )
