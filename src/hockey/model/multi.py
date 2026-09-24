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

**Availability.** Games played is a binomial share of the season, with a
partially-pooled durability level per player and AR(1) departures from it.
Without it every player is assumed to dress for all 84 games and risk comes out
almost exactly proportional to reward, which is what made the safe, balanced
and upside rankings identical on the MVP fit.

The reversion matters as much as the component. The first version used a random
walk here, matching the scoring rates, and that is the wrong shape: durability
has a level to return to and a walk has none, so the spread grew with the
length of the history instead of settling. Draisaitl came out expecting 61 of
84 games with a 5th-to-95th range of 0.19 to 0.99 of the season, against an
actual average of 73 of 82, and his floor fell to 117 fantasy points when his
worst full season on record is 454.

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

from hockey.features import IndexMaps, aggregate_panel
from hockey.features.aging import offsets_for

logger = logging.getLogger(__name__)

# Count categories, modelled as Poisson with the shared form factor. The first
# five carry 99% of the variance that separates players; ppp and shp are here
# because they are counts and the machinery is vectorised, so they cost little.
STATS: tuple[str, ...] = ("goals", "assists", "sog", "hits", "blocks", "ppp", "shp")

# Signed, so it cannot be Poisson, and worth about 1% of the separating
# variance. Modelled, but simply.
SIGNED_STAT = "plus_minus"


def _idio_categories(include_idio, stats) -> tuple[str, ...]:
    """Which categories get their own walk. True means all but the anchor,
    False means none, and a tuple names them explicitly."""
    if include_idio is True:
        return tuple(c for c in stats if c != stats[0])
    if not include_idio:
        return ()
    unknown = set(include_idio) - set(stats)
    if unknown:
        raise ValueError(f"unknown categories for idiosyncratic walks: {sorted(unknown)}")
    return tuple(c for c in stats if c in include_idio and c != stats[0])


def model_structure(include_opponent: bool, include_idio=False) -> tuple[str, ...]:
    """The shared terms a model carries, for fingerprinting a stage-one fit."""
    terms = ["position_means", "form_factor", "aging", "availability", "home"]
    if include_opponent:
        terms.append("opponent")
    # Normalised to the model's own order. Stage one is handed the categories as
    # the caller typed them and stage two as `build` orders them, so a fingerprint
    # of the raw tuple made ("hits", "assists") and ("assists", "hits") two
    # different models and refused a stage one it had just fitted.
    cats = _idio_categories(include_idio, STATS)
    if cats:
        terms.append("idio_walks:" + ",".join(cats))
    return tuple(terms)


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
    # (n_stats, n_players, n_steps): the population aging curve's expected log
    # offset for each player at each season, including the projected one.
    age_offsets: np.ndarray
    ages: np.ndarray
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
    birth_dates: dict[int, object],
    aging_curve: pd.DataFrame,
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
    # Age at each walk step, including the projected season, which is the one
    # after the last fitted season.
    end_years = [s % 10000 for s in maps.seasons]
    end_years.append(end_years[-1] + 1)
    missing_dob = [p for p in players if birth_dates.get(p) is None]
    if missing_dob:
        raise ValueError(
            f"no birth date for player(s) {missing_dob}; the aging curve cannot be "
            f"applied to them. Run: python -m hockey.ingest bios"
        )
    ages = np.array(
        [[year - birth_dates[p].year for year in end_years] for p in players], dtype=int
    )
    age_offsets = offsets_for(aging_curve, ages, list(stats))

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
        age_offsets=age_offsets,
        ages=ages,
        stats=stats,
    )


def build(
    data: MultiData,
    shared=None,
    include_opponent: bool = True,
    include_idio=False,
) -> pm.Model:
    panel = data.panel
    index = data.player_index
    stats = list(data.stats)
    n_stats, n_players = len(stats), len(data.players)
    n_steps = data.n_steps
    positions = list(POOLING_POSITIONS)

    # Stage two: the values every player shares are held at what stage one
    # measured, and only this batch's own latents are sampled. `fixed` returns
    # the frozen value when there is one and otherwise builds the prior, so
    # there is a single model definition rather than two that can drift apart.
    frozen = shared.values if shared is not None else {}
    structure = model_structure(include_opponent, _idio_categories(include_idio, stats))
    if shared is not None:
        shared.check_compatible(data.maps, tuple(stats), structure)

    def fixed(name, build_prior):
        return frozen[name] if name in frozen else build_prior()

    # Without the opponent in the linear predictor the likelihood collapses to
    # its sufficient statistics: a player-season-home cell carrying the summed
    # counts and how many games produced them. Arithmetically identical, and
    # 33 times fewer rows.
    if not include_opponent:
        panel = aggregate_panel(panel, stats, SIGNED_STAT)
        exposure = panel["n_games"].to_numpy("float64")
    else:
        exposure = np.ones(len(panel))

    obs_player = panel["player_id"].map(index).to_numpy("int32")
    obs_season = panel["season_idx"].to_numpy("int32")
    obs_opponent = (
        panel["opponent_idx"].to_numpy("int32")
        if include_opponent
        else np.zeros(len(panel), dtype="int32")
    )
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
        mu_position = fixed(
            "mu_position",
            lambda: pm.Normal(
                "mu_position", mu=prior_means[:, None], sigma=1.0, dims=("stat", "position")
            ),
        )
        sigma_player = fixed(
            "sigma_player", lambda: pm.HalfNormal("sigma_player", sigma=0.6, dims="stat")
        )
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
        # An empirical-Bayes prior, not a guess. Across 2,614 player-season
        # transitions in this warehouse, the year-over-year change in a
        # player's log goal rate has an observed sd of 0.498, of which 0.489 is
        # Poisson sampling noise from roughly 70 games; the latent drift left
        # over is 0.094. That is a small signal inside a large noise, which is
        # exactly why sigma_form is hard to pin down from 39 players and why
        # the chains found modes at 0.041 and 0.162 straddling it.
        #
        # Setting the prior from the whole population to fit a subset of it is
        # standard empirical Bayes, and honest here because the quantity is a
        # property of NHL scoring rather than of the players in any one fit.
        sigma_form = fixed(
            "sigma_form", lambda: pm.LogNormal("sigma_form", mu=np.log(0.094), sigma=0.25)
        )
        z_form = pm.Normal("z_form", 0.0, 1.0, dims=("player", "walk_step"))
        form = pm.Deterministic(
            "form",
            pt.cumsum(
                pt.concatenate([pt.zeros((n_players, 1)), sigma_form * z_form[:, 1:]], axis=1),
                axis=1,
            ),
            dims=("player", "walk_step"),
        )
        # Non-negative. Every category's measured year-over-year correlation
        # with goals is at or above zero - assists 0.18, shots 0.43, power-play
        # points 0.24, blocks 0.05, hits -0.01 - so nothing here genuinely
        # moves against a player's form. Allowing negative loadings bought no
        # fit and opened a mirror mode where the factor flips sign and every
        # loading follows it.
        if "loading" in frozen:
            loading = frozen["loading"]
        else:
            loading_rest = pm.HalfNormal("loading_rest", sigma=1.0, shape=n_stats - 1)
            loading = pm.Deterministic(
                "loading", pt.concatenate([[1.0], loading_rest]), dims="stat"
            )

        # --- what each category does on its own ---
        # The anchor category gets NO walk of its own. This is what actually
        # identifies the factor, and leaving it in was a real bug: fixing the
        # goals loading at 1 pins the sign and scale only if goals has no other
        # way to drift. Give goals an idiosyncratic walk as well and the factor
        # can shrink while that walk grows and the other loadings grow to
        # compensate, with no change to the fit. Four chains duly split into
        # two modes - sigma_form 0.145 with a +0.97 assists loading, against
        # sigma_form 0.053 with a -2.78 loading - whose products are the same
        # size and opposite sign. r-hat came back at 1.74.
        #
        # With the anchor's own walk removed, the drift in goals *is* the
        # factor, so the goals data pins both its scale and its direction, and
        # every other category keeps its own residual movement.
        # Off by default. These walks are 70% of a batch's latent variables -
        # 2,520 of 3,600 at 40 players - and their fitted spreads were 0.02 to
        # 0.1 on a log rate for every category but hits. That is a great deal
        # of posterior dimension spent on movement the shared factor already
        # carries, and dimension is what sets the sampler's step count: 127
        # leapfrog steps per draw at a step size of 0.03. Whether dropping them
        # costs accuracy is for the held-out backtest to say.
        idio_categories = _idio_categories(include_idio, stats)
        if idio_categories:
            # A mask, so a category without its own walk contributes exactly
            # zero rather than a tiny estimated one. Hits is the case that
            # earns a walk: its fitted spread was 0.26 while every other
            # category sat between 0.02 and 0.1, and dropping all of them
            # cut the 50% coverage on hits from 51% to 26% on the held-out
            # season.
            mask = np.array([1.0 if c in idio_categories else 0.0 for c in stats])
            if "sigma_idio" in frozen:
                sigma_idio = frozen["sigma_idio"]
            else:
                sigma_idio_rest = pm.HalfNormal("sigma_idio_rest", sigma=0.1, shape=n_stats - 1)
                sigma_idio = pm.Deterministic(
                    "sigma_idio", pt.concatenate([[0.0], sigma_idio_rest]) * mask, dims="stat"
                )
            z_idio = pm.Normal("z_idio", 0.0, 1.0, dims=("stat", "player", "walk_step"))
            idio = pt.cumsum(
                pt.concatenate(
                    [
                        pt.zeros((n_stats, n_players, 1)),
                        sigma_idio[:, None, None] * z_idio[:, :, 1:],
                    ],
                    axis=2,
                ),
                axis=2,
            )
        else:
            idio = 0.0

        # --- ageing ---
        # The population curve enters as the EXPECTED level at each age, not as
        # a subtraction from the answer. A player's own walk sits on top of it,
        # so someone with years of evidence that they are not declining drifts
        # above the curve, and someone with no contrary evidence follows it.
        # That is the same partial pooling used for levels, applied to drift.
        #
        # The curve itself is fixed data, measured within players across 856
        # careers - far more than any one fit contains, so estimating it here
        # would only add noise. age_scale is the one concession: a single
        # parameter that lets the fit say the curve is too strong or too weak,
        # centred on trusting it.
        #
        # There is no interaction with talent, because there is none in the
        # data: once talent is measured out of sample, its correlation with
        # rate of decline is 0.00. Elite players decline like everyone else,
        # and the impression otherwise is survivorship.
        age_scale = fixed("age_scale", lambda: pm.Normal("age_scale", mu=1.0, sigma=0.25))

        mu_player = pm.Deterministic(
            "mu_player",
            baseline[:, :, None]
            + age_scale * data.age_offsets
            + loading[:, None, None] * form[None, :, :]
            + idio,
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
        # Shared by the opponent process and the availability deviations: the
        # lag between two walk steps, and a mask so a season cannot affect the
        # past. Plain constants, needed whether or not the opponent process is
        # frozen.
        lag = np.arange(n_steps)[:, None] - np.arange(n_steps)[None, :]
        causal = (lag >= 0).astype("float64")

        if not include_opponent:
            # Facing the league's best defence rather than its worst moves a
            # projection by a couple of percent, against player-to-player
            # differences of several hundred. Dropping the term also lets the
            # likelihood collapse to its sufficient statistics, which is a
            # 32-fold reduction in rows. Whether that trade is free is a
            # question for the held-out backtest, not for taste.
            opponent = np.zeros((n_stats, data.maps.n_teams, n_steps))
        elif "opponent" in frozen:
            # Frozen whole: this drops rho, sigma_team and the z_team latents
            # together with the AR construction, which is 1,848 of the ~15,000
            # parameters a joint fit carries.
            opponent = frozen["opponent"]
        else:
            rho = pm.TruncatedNormal("rho", mu=0.0, sigma=0.3, lower=0.0, upper=1.0, dims="stat")
            sigma_team = pm.HalfNormal("sigma_team", sigma=0.08, dims="stat")
            z_team = pm.Normal("z_team", 0.0, 1.0, dims=("stat", "team", "walk_step"))

            decay = rho[:, None, None] ** lag[None, :, :] * causal[None, :, :]
            innovations = sigma_team[:, None, None] * z_team
            opponent = pm.Deterministic(
                "opponent",
                (decay[:, None, :, :] * innovations[:, :, None, :]).sum(axis=-1),
                dims=("stat", "team", "walk_step"),
            )

        b_home = fixed("b_home", lambda: pm.Normal("b_home", 0.0, 0.5, dims="stat"))

        log_rate = (
            mu_player[:, obs_player, obs_season]
            + opponent[:, obs_opponent, obs_season]
            + b_home[:, None] * obs_home[None, :]
        ).T  # (n_obs, n_stats)
        # exposure is 1 per game when rows are games, and the cell's game count
        # when they are aggregated: a sum of n Poissons is Poisson at n times
        # the rate.
        pm.Poisson(
            "counts",
            mu=exposure[:, None] * pt.exp(log_rate),
            observed=observed,
            dims=("obs", "stat"),
        )

        # --- availability ---
        # Each player has a durability level, and departures from it revert.
        # That reversion is the whole point, and getting it wrong the first way
        # produced nonsense: a random walk here, like the one on the scoring
        # rates, accumulates variance every season and never pulls back, so a
        # player whose games played bounced around - which is every player -
        # ended up with an unbounded projection. Draisaitl came out with a 5th
        # to 95th percentile of 0.19 to 0.99 of the season and a 61-game
        # expectation despite averaging 73 of 82, which dragged his floor to
        # 117 fantasy points against a career low of 454.
        #
        # A random walk is the right shape for a scoring rate, which genuinely
        # drifts and has no home to return to. It is the wrong shape for
        # durability, which does.
        mu_avail = fixed(
            "mu_avail",
            lambda: pm.Normal("mu_avail", mu=PRIOR_LOGIT_AVAILABILITY, sigma=1.0, dims="position"),
        )
        sigma_avail = fixed("sigma_avail", lambda: pm.HalfNormal("sigma_avail", sigma=0.7))
        z_avail = pm.Normal("z_avail", 0.0, 1.0, dims="player")
        durability = mu_avail[position_idx] + sigma_avail * z_avail

        # Season-to-season departures from that level: AR(1), so a bad year
        # carries into the next one and then fades, and the uncertainty at the
        # projection step settles at the stationary spread instead of growing
        # with the length of the history.
        phi_avail = fixed("phi_avail", lambda: pm.Beta("phi_avail", alpha=2.0, beta=2.0))
        sigma_avail_season = fixed(
            "sigma_avail_season", lambda: pm.HalfNormal("sigma_avail_season", sigma=0.5)
        )
        z_avail_season = pm.Normal("z_avail_season", 0.0, 1.0, dims=("player", "walk_step"))
        avail_decay = phi_avail**lag * causal
        avail_dev = (
            avail_decay[None, :, :] * (sigma_avail_season * z_avail_season)[:, None, :]
        ).sum(axis=-1)
        logit_avail = pm.Deterministic(
            "logit_avail", durability[:, None] + avail_dev, dims=("player", "walk_step")
        )
        # --- plus/minus ---
        # Signed, so Normal rather than Poisson, and given a per-player mean
        # with partial pooling but no walk or factor loading. It is worth about
        # 1% of the variance that separates players and averages to roughly
        # zero across the league by construction, so more structure would be
        # machinery spent on noise.
        mu_pm = fixed("mu_pm", lambda: pm.Normal("mu_pm", 0.0, 0.2))
        sigma_pm_player = fixed("sigma_pm_player", lambda: pm.HalfNormal("sigma_pm_player", 0.2))
        z_pm = pm.Normal("z_pm", 0.0, 1.0, dims="player")
        pm_player = pm.Deterministic("pm_player", mu_pm + sigma_pm_player * z_pm, dims="player")
        b_home_pm = fixed("b_home_pm", lambda: pm.Normal("b_home_pm", 0.0, 0.2))
        sigma_pm = fixed("sigma_pm", lambda: pm.HalfNormal("sigma_pm", 1.5))
        # A sum of n normals has n times the mean and sqrt(n) times the spread.
        pm.Normal(
            "plus_minus",
            mu=exposure * (pm_player[obs_player] + b_home_pm * obs_home),
            sigma=sigma_pm * np.sqrt(exposure),
            observed=panel[SIGNED_STAT].to_numpy("float64"),
            dims="obs",
        )

        # Beta-Binomial, not Binomial. A Binomial treats 82 games as 82
        # independent chances to be absent, giving a spread of under three
        # games; measured across 756 players with five or more seasons, games
        # played is 19 times more variable than that allows, because injuries
        # are contiguous. One knee costs fifteen consecutive games, not fifteen
        # unrelated ones, so the effective number of independent trials is
        # nearer four than eighty-two.
        #
        # Getting this wrong is not a small mis-fit. With the likelihood far
        # too tight, every season's variation had to be explained by the latent
        # durability instead, which drove its stationary spread to 2.02 on the
        # logit - a player at 89% availability ranging from 13% to 100% - and
        # made availability 53% of the fantasy-point variance while
        # over-covering at every interval. The held-out 90% interval caught
        # 97.4% of outcomes.
        #
        # kappa is the concentration: large recovers the Binomial, small means
        # clustered absences. The data implies roughly 3.5 before the AR(1)
        # takes its share, so the prior sits at a mean of 10 and lets the fit
        # settle it.
        kappa_avail = fixed("kappa_avail", lambda: pm.Gamma("kappa_avail", alpha=2.0, beta=0.2))
        p_avail = pm.math.sigmoid(logit_avail[avail_player, avail_season])
        pm.BetaBinomial(
            "games_played",
            n=avail_total,
            alpha=p_avail * kappa_avail,
            beta=(1.0 - p_avail) * kappa_avail,
            observed=avail_played,
            dims="avail_obs",
        )

    return model


def sample(model, draws=500, tune=800, chains=4, target_accept=0.9, seed=20262027):
    with model:
        return pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            nuts_sampler="numpyro",
            random_seed=seed,
            progressbar=False,
            # Our runners compute r-hat, ESS and divergences themselves with
            # az.summary. PyMC's built-in check calls arviz_stats.ess, which in
            # this environment can hit a circular import after a long sample and
            # crash the process - discarding the posterior it just spent an hour
            # on. A converged posterior is not the thing to lose to a warning.
            compute_convergence_checks=False,
        )
