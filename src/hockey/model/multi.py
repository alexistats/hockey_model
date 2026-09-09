"""The full skater model: five categories, a shared form factor, availability.

Three things this adds to the MVP in `skater.py`, each answering something
measured in the data rather than assumed.

**Every scored category.** Goals, assists, shots, hits and blocks carry 99% of
what separates players under this league's settings, and get the full
treatment. Power-play and shorthanded points are counts too and ride the same
vectorised machinery for almost nothing. Plus/minus is the exception: it is
signed, so Poisson cannot describe it, and it carries about 1% of the
separating variance while averaging to roughly zero. It gets a partially
pooled per-player mean with a Normal likelihood, and no walk - deliberately
less machinery for something close to noise. Nothing is dropped, because a
category the scoring config expects and the projection omits is refused
outright rather than quietly scored as zero.

**A shared form factor.** A player's categories move together: year over year
the offensive ones correlate about 0.31, goals with shots at 0.43. Giving each
category an independent walk, as the MVP did, understates the spread of their
sum by 25% when measured against eight seasons of this warehouse - and their
sum is fantasy points, the only quantity that matters. So each player gets one
latent form walk that every category loads onto, plus a category-specific walk
for the rest. The loadings are estimated, which lets hits and blocks come out
near zero, as the data says they should: their correlation with goals is -0.01.

**Availability.** Games played is modelled as a binomial share of the season,
with its own partially-pooled rate and its own walk, so a player who lost most
of two seasons carries that into their floor. Without it, every player is
assumed to dress for all 84 games and risk becomes almost exactly proportional
to reward - which is what made the safe, balanced and upside draft rankings
come out identical on the MVP fit.

The state-space architecture is unchanged from the original model: latent rates
drift as random walks across seasons, opponent strength is AR(1), there is a
home effect, and counts are Poisson on a log link.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from hockey.features import IndexMaps

logger = logging.getLogger(__name__)

# Count categories, modelled as Poisson with the shared form factor. The first
# five carry 99% of the variance that separates players; ppp and shp are here
# because they are counts and the machinery is vectorised, so they cost little.
STATS: tuple[str, ...] = ("goals", "assists", "sog", "hits", "blocks", "ppp", "shp")

# Signed, so it cannot be Poisson, and worth about 1% of the separating
# variance. Modelled, but simply.
SIGNED_STAT = "plus_minus"

POOLING_POSITIONS = ("C", "LW", "RW", "D")

# Prior means for the log per-game rate of each category, roughly a league
# average skater. Wide enough (sd 1.0 below) to be weakly informative; their
# job is to keep the sampler out of absurd territory, not to pick an answer.
PRIOR_LOG_RATE = {
    "goals": -2.0,
    "assists": -1.7,
    "sog": 0.5,
    "hits": 0.0,
    "blocks": -0.2,
    "ppp": -2.4,
    "shp": -4.5,
}

# Logit of the prior mean availability: about 0.90 of the season.
PRIOR_LOGIT_AVAILABILITY = 2.2


@dataclass
class MultiData:
    panel: pd.DataFrame
    availability: pd.DataFrame
    schedule: pd.DataFrame
    players: list[int]
    player_names: dict[int, str]
    positions: list[str]
    maps: IndexMaps
    stats: tuple[str, ...] = STATS
    projection_step: int = field(init=False)

    def __post_init__(self):
        self.projection_step = self.maps.n_seasons

    @property
    def player_index(self) -> dict[int, int]:
        return {p: i for i, p in enumerate(self.players)}

    @property
    def n_steps(self) -> int:
        return self.maps.n_seasons + 1


def prepare(
    panel: pd.DataFrame,
    availability: pd.DataFrame,
    schedule: pd.DataFrame,
    maps: IndexMaps,
    player_names: dict[int, str],
    positions: dict[int, str],
    stats: tuple[str, ...] = STATS,
) -> MultiData:
    players = sorted(int(p) for p in panel["player_id"].unique())
    missing = set(players) - set(schedule["player_id"].unique())
    if missing:
        raise ValueError(
            f"no projection schedule for player(s) {sorted(missing)}; they would get "
            f"a zero-game projection that looks like a real one"
        )
    index = {p: i for i, p in enumerate(players)}
    availability = availability[availability["player_id"].isin(players)].copy()
    availability["player_idx"] = availability["player_id"].map(index)
    availability["season_idx"] = availability["season"].map(maps.season_index)
    return MultiData(
        panel=panel.reset_index(drop=True),
        availability=availability.reset_index(drop=True),
        schedule=schedule[schedule["player_id"].isin(players)].reset_index(drop=True),
        players=players,
        player_names=player_names,
        positions=[
            p if (p := positions.get(pid, "C")) in POOLING_POSITIONS else "C" for pid in players
        ],
        maps=maps,
        stats=stats,
    )


def build(data: MultiData) -> pm.Model:
    panel = data.panel
    index = data.player_index
    stats = list(data.stats)
    n_stats, n_players = len(stats), len(data.players)
    n_steps = data.n_steps
    positions = list(POOLING_POSITIONS)

    obs_player = panel["player_id"].map(index).to_numpy("int32")
    obs_season = panel["season_idx"].to_numpy("int32")
    obs_opponent = panel["opponent_idx"].to_numpy("int32")
    obs_home = panel["is_home"].to_numpy("float64")
    observed = panel[stats].to_numpy("int64")  # (n_obs, n_stats)
    position_idx = np.array([positions.index(p) for p in data.positions], dtype="int32")
    prior_means = np.array([PRIOR_LOG_RATE[s] for s in stats])

    avail = data.availability
    avail_player = avail["player_idx"].to_numpy("int32")
    avail_season = avail["season_idx"].to_numpy("int32")
    avail_played = avail["games_played"].to_numpy("int64")
    avail_total = avail["games_available"].to_numpy("int64")

    coords = {
        "stat": stats,
        "player": [data.player_names.get(p, str(p)) for p in data.players],
        "position": positions,
        "team": data.maps.teams,
        "walk_step": [*data.maps.seasons, "projection"],
        "obs": np.arange(len(panel)),
        "avail_obs": np.arange(len(avail)),
    }

    with pm.Model(coords=coords) as model:
        # --- where each player's rates start ---
        mu_position = pm.Normal(
            "mu_position",
            mu=prior_means[:, None],
            sigma=1.0,
            dims=("stat", "position"),
        )
        sigma_player = pm.HalfNormal("sigma_player", sigma=0.6, dims="stat")
        z_player = pm.Normal("z_player", 0.0, 1.0, dims=("stat", "player"))
        baseline = pm.Deterministic(
            "baseline",
            mu_position[:, position_idx] + sigma_player[:, None] * z_player,
            dims=("stat", "player"),
        )

        # --- the shared part of a player's year-to-year movement ---
        # A one-factor model, identified the standard way: the first category's
        # loading is FIXED at 1, so the factor is measured in goal-rate units
        # and every other loading is read relative to goals.
        #
        # The first attempt let the factor carry unit innovations and estimated
        # all seven loadings. That leaves the factor's scale free - shrink the
        # innovations, grow every loading, identical fit - and the chains duly
        # disagreed, at r-hat 1.83. Pinning a loading rather than the
        # innovation size is what removes the degeneracy; the sign comes along
        # with it, so the whole factor can no longer flip either.
        #
        # sigma_form is separable from the per-category walks only because the
        # factor is shared across seven of them: within any single category the
        # two are the same thing, but their common movement across categories
        # is not.
        sigma_form = pm.HalfNormal("sigma_form", sigma=0.1)
        z_form = pm.Normal("z_form", 0.0, 1.0, dims=("player", "walk_step"))
        form = pm.Deterministic(
            "form",
            pt.cumsum(
                pt.concatenate([pt.zeros((n_players, 1)), sigma_form * z_form[:, 1:]], axis=1),
                axis=1,
            ),
            dims=("player", "walk_step"),
        )
        loading_rest = pm.Normal("loading_rest", 0.0, 1.0, shape=n_stats - 1)
        loading = pm.Deterministic("loading", pt.concatenate([[1.0], loading_rest]), dims="stat")

        # --- what each category does on its own ---
        sigma_idio = pm.HalfNormal("sigma_idio", sigma=0.1, dims="stat")
        z_idio = pm.Normal("z_idio", 0.0, 1.0, dims=("stat", "player", "walk_step"))
        idio = pt.cumsum(
            pt.concatenate(
                [pt.zeros((n_stats, n_players, 1)), sigma_idio[:, None, None] * z_idio[:, :, 1:]],
                axis=2,
            ),
            axis=2,
        )

        mu_player = pm.Deterministic(
            "mu_player",
            baseline[:, :, None] + loading[:, None, None] * form[None, :, :] + idio,
            dims=("stat", "player", "walk_step"),
        )

        # --- opponent strength, shared across every player ---
        # The same AR(1) across seasons the original model used, written out as
        # its moving-average form rather than built with pm.AR. pm.AR runs a
        # scan whose rho and sigma cannot broadcast over a leading category
        # axis, so five categories would need five separate scans. Expanding
        # x[t] = rho*x[t-1] + sigma*e[t] into x = e @ M', where M[t,j] is
        # rho^(t-j) below the diagonal, is the identical process as one matrix
        # multiply - and over nine seasons that matrix is 9 by 9.
        rho = pm.TruncatedNormal("rho", mu=0.0, sigma=0.3, lower=0.0, upper=1.0, dims="stat")
        sigma_team = pm.HalfNormal("sigma_team", sigma=0.08, dims="stat")
        z_team = pm.Normal("z_team", 0.0, 1.0, dims=("stat", "team", "walk_step"))

        lag = np.arange(n_steps)[:, None] - np.arange(n_steps)[None, :]
        causal = (lag >= 0).astype("float64")  # a season cannot affect the past
        decay = rho[:, None, None] ** lag[None, :, :] * causal[None, :, :]
        innovations = sigma_team[:, None, None] * z_team
        opponent = pm.Deterministic(
            "opponent",
            (decay[:, None, :, :] * innovations[:, :, None, :]).sum(axis=-1),
            dims=("stat", "team", "walk_step"),
        )
        b_home = pm.Normal("b_home", 0.0, 0.5, dims="stat")

        log_rate = (
            mu_player[:, obs_player, obs_season]
            + opponent[:, obs_opponent, obs_season]
            + b_home[:, None] * obs_home[None, :]
        ).T  # (n_obs, n_stats)
        pm.Poisson("counts", mu=pt.exp(log_rate), observed=observed, dims=("obs", "stat"))

        # --- availability ---
        # Games played as a share of the season, on a logit scale, with the
        # same partial pooling and the same drift structure as the rates. A
        # player who lost most of two seasons keeps a lower floor because of
        # it, which is the whole point of modelling this at all.
        mu_avail = pm.Normal("mu_avail", mu=PRIOR_LOGIT_AVAILABILITY, sigma=1.0, dims="position")
        sigma_avail = pm.HalfNormal("sigma_avail", sigma=0.7)
        z_avail = pm.Normal("z_avail", 0.0, 1.0, dims="player")
        sigma_avail_walk = pm.HalfNormal("sigma_avail_walk", sigma=0.3)
        z_avail_walk = pm.Normal("z_avail_walk", 0.0, 1.0, dims=("player", "walk_step"))
        logit_avail = pm.Deterministic(
            "logit_avail",
            mu_avail[position_idx][:, None]
            + (sigma_avail * z_avail)[:, None]
            + pt.cumsum(
                pt.concatenate(
                    [pt.zeros((n_players, 1)), sigma_avail_walk * z_avail_walk[:, 1:]], axis=1
                ),
                axis=1,
            ),
            dims=("player", "walk_step"),
        )
        # --- plus/minus ---
        # Signed, so Normal rather than Poisson, and given a per-player mean
        # with partial pooling but no walk or factor loading. It is worth about
        # 1% of the variance that separates players and averages to roughly
        # zero across the league by construction, so more structure would be
        # machinery spent on noise.
        mu_pm = pm.Normal("mu_pm", 0.0, 0.2)
        sigma_pm_player = pm.HalfNormal("sigma_pm_player", 0.2)
        z_pm = pm.Normal("z_pm", 0.0, 1.0, dims="player")
        pm_player = pm.Deterministic("pm_player", mu_pm + sigma_pm_player * z_pm, dims="player")
        b_home_pm = pm.Normal("b_home_pm", 0.0, 0.2)
        sigma_pm = pm.HalfNormal("sigma_pm", 1.5)
        pm.Normal(
            "plus_minus",
            mu=pm_player[obs_player] + b_home_pm * obs_home,
            sigma=sigma_pm,
            observed=panel[SIGNED_STAT].to_numpy("float64"),
            dims="obs",
        )

        pm.Binomial(
            "games_played",
            n=avail_total,
            p=pm.math.sigmoid(logit_avail[avail_player, avail_season]),
            observed=avail_played,
            dims="avail_obs",
        )

    return model


def sample(model, draws=1000, tune=1500, chains=4, target_accept=0.95, seed=20262027):
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
