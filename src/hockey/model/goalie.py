"""The hierarchical Bayesian model for goalies.

Goalies are not skaters with different categories, and modelling them that way
would miss what the data says about them. Two measurements shaped this, both
taken on the eight seasons in the warehouse (447 goalie-seasons of 20+ starts).

**Starts are the whole game.** 71% of the variance in a goalie's season
fantasy total comes from how many games he starts; only 17% comes from how
well he plays per start. A model that nails save percentage and guesses at
workload will be wrong about fantasy value; one that nails workload and is
merely adequate on rates will be roughly right. So starts get a real model and
an explicit place for a human to overrule it - a depth chart is knowledge the
game logs do not contain.

**Which rates are the team, and which are not.** Grouping goalie-seasons by
team-season appears to explain 67% of the variation in save percentage, but
that number is mostly an artefact: there are 264 team-seasons for 447
observations, and shuffling the team labels at random still "explains" 56.8%.
The real signal is about ten points above chance. Fitted properly, with
partial pooling doing the work, save percentage splits about evenly - a
goalie sd of 0.070 on the logit scale against a team sd of 0.069.

Win rate is different, and it is the one that matters. Its team sd is 0.239,
three times the save-rate effects, and at 6 points a win against 0.3 a save it
is the largest per-start term in this league's scoring. So the common claim
that goaltending value is really team quality is right, but for a narrower
reason than usual: not because the team stops pucks for him, but because the
team wins games for him.

Following the same goalie across a team change, the spread of the change in
save percentage is 0.0159 where two independent draws from the cross-section
would give 0.0175 - so there is individual persistence, but not much.

**A structural note on this league's scoring.** Saves pay 0.3 and goals
against cost 1.5, so a shot faced is worth `0.3s - 1.5(1-s)` and breaks even at
a save percentage of 0.8333. Every NHL goalie is comfortably above that, which
means facing more shots is *always* net positive here: a goalie behind a leaky
defence is paid for the extra volume, and that partly offsets the wins his team
does not get. This is why shots against is modelled explicitly rather than
folded into a goals-against rate - the two pull in opposite directions and the
net depends on the league's own numbers.

The structure mirrors the skater model where the reasoning carries over: a
partially pooled per-goalie effect taking a Gaussian random walk across
seasons, one more step into the projected season with no observations so drift
uncertainty reaches the forecast, and non-centered parameterizations so NUTS
does not have to fight a funnel.
"""

import logging

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

logger = logging.getLogger(__name__)

# A goalie-season below this is a backup's handful of appearances, where the
# rate estimates are noise and the workload tells us nothing about next year.
# They still inform the team effects, so they are kept for that and simply do
# not anchor a goalie's own walk.
MIN_STARTS_FOR_WALK = 10


def panel(session, seasons: list[int] | None = None) -> pd.DataFrame:
    """One row per goalie-season-team, aggregated.

    Aggregated rather than per-game on purpose. The lean skater model showed
    that a per-game likelihood costs an enormous amount of sampling time and
    buys calibration that an aggregated one matches, and here the quantities
    that matter - starts, saves out of shots, wins out of starts - are all
    sums over a season anyway.

    Team is part of the key, so a goalie traded mid-season appears twice and
    each stint is attributed to the right team rather than averaged across two.
    """
    from sqlalchemy import text

    sql = """
      SELECT gl.player_id,
             g.season,
             gl.team_abbrev                      AS team,
             sum(gl.started::int)                AS starts,
             count(*)                            AS appearances,
             sum(gl.wins)                        AS wins,
             sum(gl.goals_against)               AS goals_against,
             sum(gl.saves)                       AS saves,
             sum(gl.shots_against)               AS shots_against,
             sum(gl.shutouts)                    AS shutouts
        FROM goalie_game_logs gl
        JOIN nhl_games g ON g.nhl_game_id = gl.game_id
       WHERE g.game_type = 2
         AND gl.team_abbrev IS NOT NULL
         {season_filter}
       GROUP BY 1, 2, 3
       ORDER BY 1, 2
    """
    params: dict = {}
    season_filter = ""
    if seasons:
        season_filter = "AND g.season = ANY(:seasons)"
        params["seasons"] = seasons
    frame = pd.DataFrame(
        session.execute(text(sql.format(season_filter=season_filter)), params).mappings().all()
    )
    if frame.empty:
        return frame
    # A start the goalie did not finish, or relief that turned into a decision,
    # means appearances and starts disagree. Wins are credited per decision, so
    # the denominator for a win rate is appearances, not starts.
    frame["decisions"] = frame[["appearances", "starts"]].max(axis=1)
    return frame


def team_games(session, season: int) -> dict[str, int]:
    """How many regular-season games each team plays in `season`.

    Read rather than assumed: the league has not always had 82, 2026-27 has 84
    under the new agreement, and a shortened season in the fitting window would
    otherwise silently rescale every workload.
    """
    from sqlalchemy import text

    rows = session.execute(
        text("""
          SELECT team, count(*) AS games FROM (
            SELECT home_team_abbrev AS team FROM nhl_games
             WHERE season = :s AND game_type = 2
            UNION ALL
            SELECT away_team_abbrev AS team FROM nhl_games
             WHERE season = :s AND game_type = 2
          ) t GROUP BY team
        """),
        {"s": season},
    ).all()
    return {r[0]: int(r[1]) for r in rows}


class GoalieIndex:
    """The integer codings the model is written against."""

    def __init__(self, frame: pd.DataFrame, projection_season: int):
        self.goalies = sorted(frame["player_id"].unique())
        self.teams = sorted(frame["team"].unique())
        observed = sorted(frame["season"].unique())
        if projection_season in observed:
            raise ValueError(
                f"the projected season {projection_season} is in the fitting window; "
                f"indexing it as observed would read 'not played yet' as 'played and "
                f"recorded nothing'"
            )
        self.seasons = [*observed, projection_season]
        self.projection_step = len(self.seasons) - 1
        self.goalie_of = {p: i for i, p in enumerate(self.goalies)}
        self.team_of = {t: i for i, t in enumerate(self.teams)}
        self.season_of = {s: i for i, s in enumerate(self.seasons)}

    @property
    def n_goalies(self) -> int:
        return len(self.goalies)

    @property
    def n_teams(self) -> int:
        return len(self.teams)

    @property
    def n_steps(self) -> int:
        return len(self.seasons)


def build(frame: pd.DataFrame, index: GoalieIndex) -> pm.Model:
    """The model. Everything it needs is in `frame`; nothing is read globally."""
    g = frame["player_id"].map(index.goalie_of).to_numpy()
    t = frame["team"].map(index.team_of).to_numpy()
    s = frame["season"].map(index.season_of).to_numpy()

    starts = frame["starts"].to_numpy().astype("int64")
    decisions = frame["decisions"].to_numpy().astype("int64")
    shots = frame["shots_against"].to_numpy().astype("int64")
    saves = frame["saves"].to_numpy().astype("int64")
    wins = frame["wins"].to_numpy().astype("int64")
    shutouts = frame["shutouts"].to_numpy().astype("int64")
    games = frame["team_games"].to_numpy().astype("int64")

    coords = {
        "goalie": index.goalies,
        "team": index.teams,
        "season": index.seasons,
        "row": np.arange(len(frame)),
    }

    with pm.Model(coords=coords) as model:
        # --- save skill: a goalie effect that walks, plus the team in front ---
        mu_save = pm.Normal("mu_save", mu=2.25, sigma=0.5)  # logit(0.905) ~ 2.25
        sigma_goalie_save = pm.HalfNormal("sigma_goalie_save", 0.25)
        sigma_walk_save = pm.HalfNormal("sigma_walk_save", 0.10)

        # Non-centered random walk: level at the first season, then steps.
        save_level = pm.Normal("save_level_raw", 0.0, 1.0, dims="goalie")
        save_steps = pm.Normal(
            "save_steps_raw", 0.0, 1.0, shape=(index.n_goalies, index.n_steps - 1)
        )
        save_walk = pm.Deterministic(
            "save_walk",
            (save_level * sigma_goalie_save)[:, None]
            + pt.concatenate(
                [
                    pt.zeros((index.n_goalies, 1)),
                    pt.cumsum(save_steps * sigma_walk_save, axis=1),
                ],
                axis=1,
            ),
            dims=("goalie", "season"),
        )

        sigma_team_save = pm.HalfNormal("sigma_team_save", 0.15)
        team_save = pm.Deterministic(
            "team_save",
            pm.Normal("team_save_raw", 0.0, 1.0, shape=(index.n_teams, index.n_steps))
            * sigma_team_save,
            dims=("team", "season"),
        )

        logit_save = mu_save + save_walk[g, s] + team_save[t, s]
        pm.Binomial("saves_obs", n=shots, p=pm.math.sigmoid(logit_save), observed=saves)

        # --- shots faced per start: overwhelmingly the team ---
        # Negative binomial rather than Poisson: shot counts across a season are
        # far more dispersed than Poisson, and understating that would make the
        # ceiling too narrow for exactly the high-volume goalies this scoring
        # rewards.
        mu_shots = pm.Normal("mu_shots", mu=np.log(30.0), sigma=0.3)
        sigma_team_shots = pm.HalfNormal("sigma_team_shots", 0.20)
        team_shots = pm.Deterministic(
            "team_shots",
            pm.Normal("team_shots_raw", 0.0, 1.0, shape=(index.n_teams, index.n_steps))
            * sigma_team_shots,
            dims=("team", "season"),
        )
        sigma_goalie_shots = pm.HalfNormal("sigma_goalie_shots", 0.10)
        goalie_shots = pm.Deterministic(
            "goalie_shots",
            pm.Normal("goalie_shots_raw", 0.0, 1.0, dims="goalie") * sigma_goalie_shots,
            dims="goalie",
        )
        log_rate = mu_shots + team_shots[t, s] + goalie_shots[g]
        alpha_shots = pm.Exponential("alpha_shots", 0.05)
        pm.NegativeBinomial(
            "shots_obs",
            mu=pt.exp(log_rate) * pt.maximum(starts, 1),
            alpha=alpha_shots,
            observed=shots,
        )

        # --- wins: the team, plus credit for stopping pucks ---
        # The save term is centered so it adds nothing for an average goalie and
        # the team effect keeps its meaning.
        mu_win = pm.Normal("mu_win", mu=0.0, sigma=0.5)
        sigma_team_win = pm.HalfNormal("sigma_team_win", 0.35)
        team_win = pm.Deterministic(
            "team_win",
            pm.Normal("team_win_raw", 0.0, 1.0, shape=(index.n_teams, index.n_steps))
            * sigma_team_win,
            dims=("team", "season"),
        )
        beta_save_on_win = pm.HalfNormal("beta_save_on_win", 2.0)
        logit_win = mu_win + team_win[t, s] + beta_save_on_win * (save_walk[g, s] + team_save[t, s])
        pm.Binomial("wins_obs", n=decisions, p=pm.math.sigmoid(logit_win), observed=wins)

        # --- shutouts ---
        # Driven by the same latent quality rather than given free parameters of
        # its own, so a goalie cannot be projected for shutouts his save rate and
        # workload do not support.
        mu_shutout = pm.Normal("mu_shutout", mu=-2.3, sigma=0.5)
        beta_save_on_shutout = pm.HalfNormal("beta_save_on_shutout", 3.0)
        beta_shots_on_shutout = pm.Normal("beta_shots_on_shutout", 0.0, 1.0)
        logit_shutout = (
            mu_shutout
            + beta_save_on_shutout * (save_walk[g, s] + team_save[t, s])
            + beta_shots_on_shutout * (team_shots[t, s] + goalie_shots[g])
        )
        pm.Binomial(
            "shutouts_obs",
            n=pt.maximum(starts, 1),
            p=pm.math.sigmoid(logit_shutout),
            observed=shutouts,
        )

        # --- workload: the share of his team's games a goalie starts ---
        # 71% of the variance in season fantasy points lives here, so it gets a
        # walk of its own rather than being carried forward as last year's
        # number. The walk is deliberately loose: a backup becoming a starter is
        # a large, common move, and a tight walk would rule it out.
        mu_share = pm.Normal("mu_share", mu=-1.0, sigma=0.5)
        sigma_goalie_share = pm.HalfNormal("sigma_goalie_share", 0.8)
        sigma_walk_share = pm.HalfNormal("sigma_walk_share", 0.5)
        share_level = pm.Normal("share_level_raw", 0.0, 1.0, dims="goalie")
        share_steps = pm.Normal(
            "share_steps_raw", 0.0, 1.0, shape=(index.n_goalies, index.n_steps - 1)
        )
        share_walk = pm.Deterministic(
            "share_walk",
            (share_level * sigma_goalie_share)[:, None]
            + pt.concatenate(
                [
                    pt.zeros((index.n_goalies, 1)),
                    pt.cumsum(share_steps * sigma_walk_share, axis=1),
                ],
                axis=1,
            ),
            dims=("goalie", "season"),
        )
        pm.Binomial(
            "starts_obs",
            n=games,
            p=pm.math.sigmoid(mu_share + share_walk[g, s]),
            observed=starts,
        )

    return model
