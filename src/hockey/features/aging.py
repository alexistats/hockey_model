"""The population aging curve, measured within players.

Comparing 22-year-olds to 34-year-olds across the league confounds age with
survivorship: the 34-year-olds still on the ice are the ones who stayed good,
and the ones who fell off at 31 are simply absent. The only clean read is how a
player changes against himself, so this measures the year-over-year change in
each player's own log rate and attributes it to the age he moved through.

Two things it is deliberately NOT:

It is not a per-player forecast. Measured across 856 players, elite players
decline at the same rate as everyone else - the correlation between talent and
rate of decline is 0.00 once talent is measured out of sample. Ranking players
by last season's rate suggests the opposite, but that is regression to the mean,
since a high season is partly luck and luck regresses. The impression that
stars age gracefully is survivorship.

And it is not a subtraction applied to a projection. The model uses this as the
expected drift of a player's walk, which their own history can pull away from.
A player with seven flat seasons at 34 will be projected flatter than the curve;
a player with no contrary evidence will follow it.

Curves are per category because they genuinely differ: blocked shots barely age
at all, since blocking is a role rather than an athletic peak, while goals fall
about 12% a year past 34.
"""

import logging

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

CURVE_STATS = ("goals", "assists", "sog", "hits", "blocks", "ppp", "shp")

# The age a curve is measured relative to. Offsets are zero here by
# construction, so a player's baseline is their level at their peak.
REFERENCE_AGE = 26
MIN_AGE, MAX_AGE = 18, 44

# A season needs enough games for its rate to mean anything.
MIN_GAMES = 40

_SQL = """
SELECT s.player_id, g.season, p.birth_date,
       count(*) AS gp,
       sum(s.goals)::float goals, sum(s.assists)::float assists,
       sum(s.sog)::float sog, sum(s.hits)::float hits, sum(s.blocks)::float blocks,
       sum(coalesce(s.ppp, 0))::float ppp, sum(coalesce(s.shp, 0))::float shp
  FROM skater_game_logs s
  JOIN nhl_games g ON g.nhl_game_id = s.game_id
  JOIN players p ON p.nhl_id = s.player_id
 WHERE g.game_type = 2 AND p.birth_date IS NOT NULL
 GROUP BY 1, 2, 3
HAVING count(*) >= :min_games
"""


def season_age(season: pd.Series, birth_date: pd.Series) -> pd.Series:
    """Age at 1 February of the season, its rough midpoint."""
    end_year = pd.to_datetime(season % 10000, format="%Y").dt.year
    born = pd.to_datetime(birth_date)
    return end_year - born.dt.year


def measure(session: Session, smooth: int = 5) -> pd.DataFrame:
    """Cumulative log-rate offset by age, one column per category.

    Indexed by single year of age. The value at REFERENCE_AGE is zero, ages
    below it are the level a player was at on the way up, and ages above it the
    level on the way down.
    """
    df = pd.DataFrame(session.execute(text(_SQL), {"min_games": MIN_GAMES}).mappings().all())
    df["age"] = season_age(df["season"], df["birth_date"])
    for stat in CURVE_STATS:
        # +0.01 keeps a zero season finite without moving a real rate much.
        df[stat] = np.log(df[stat] / df["gp"] + 0.01)

    df = df.sort_values(["player_id", "season"])
    deltas = df.groupby("player_id")[list(CURVE_STATS)].diff()
    deltas["age_from"] = df.groupby("player_id")["age"].shift(1)
    deltas["step"] = df["age"] - deltas["age_from"]
    # Consecutive seasons only: a player who missed a year would otherwise have
    # two years of ageing charged to one transition.
    deltas = deltas[deltas["step"] == 1].dropna(subset=["age_from"])

    per_age = deltas.groupby("age_from")[list(CURVE_STATS)].mean()
    per_age = per_age.reindex(range(MIN_AGE, MAX_AGE + 1)).fillna(0.0)
    # Smooth, because a single year of age is a thin slice at the extremes and
    # the underlying curve has no reason to be jagged.
    per_age = per_age.rolling(smooth, center=True, min_periods=1).mean()

    curve = per_age.cumsum()
    curve = curve - curve.loc[REFERENCE_AGE]
    curve.index.name = "age"
    logger.info(
        "aging curve from %d transitions across %d players",
        len(deltas),
        df["player_id"].nunique(),
    )
    return curve


def offsets_for(curve: pd.DataFrame, ages: np.ndarray, stats: list[str]) -> np.ndarray:
    """Look up (stat, ...) offsets for an array of ages, clipped to the
    measured range so an unusually old or young player gets the endpoint rather
    than an extrapolation the data cannot support."""
    clipped = np.clip(ages, curve.index.min(), curve.index.max())
    table = curve[stats].to_numpy()
    rows = clipped - curve.index.min()
    return np.moveaxis(table[rows], -1, 0)
