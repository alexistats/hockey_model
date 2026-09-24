"""How a skater's last season compared with the two before it.

A warehouse fact written beside the posterior, like the calendar - not a model
output, and nothing here changes a projection. It is on the board because it
marks where the projection has measurably been wrong. Across three held-out
seasons (2023-24 to 2025-26), after a season 12% or more above a player's two
before, the model over-projected the next one by 0.48 fantasy points a game,
against 0.24 after a steady season - 0.39 for players under 30, 0.68 for 30 and
over. After a season 12% or more below, by 0.13: it expects part of a dip back,
and part came back.

The flag compares seasons in this league's currency, fantasy points a game,
scored from the league config rather than restated here, and in one scoring
environment: each earlier season is rescaled, category by category, to what the
league recorded last season. 2025-26 recorded 10% fewer hits and blocks a game
than 2023-24; a defenceman whose numbers fell with the league's did not have a
bad season by his own standards, and unadjusted, one in three players on the
board read as cold.
"""

import numpy as np
import pandas as pd
from sqlalchemy import text

from hockey.scoring import LeagueScoring, score_statline

# A season has to be close to full to count: a 20-game stretch is too noisy to
# call a player hot or cold, and it is not what the held-out measurement used.
MIN_GAMES = 40

# A swing this size or larger is flagged. 12% is about the upper quartile of
# year-over-year changes, not a cliff: the over-projection grows steadily with
# the jump, which is why the page shows its size rather than only the flag.
SWING = 0.12

# Every scored skater category is a column of the same name in the game logs.
_TOTALS_SQL = """
SELECT s.player_id, g.season, count(*) AS games,
       sum(s.goals) AS goals, sum(s.assists) AS assists, sum(s.plus_minus) AS plus_minus,
       sum(s.ppp) AS ppp, sum(s.shp) AS shp, sum(s.sog) AS sog,
       sum(s.hits) AS hits, sum(s.blocks) AS blocks
  FROM skater_game_logs s
  JOIN nhl_games g ON g.nhl_game_id = s.game_id
 WHERE g.game_type = 2
   AND s.team_abbrev IS NOT NULL
   AND g.season = ANY(:seasons)
 GROUP BY s.player_id, g.season
"""


def previous_season(season: int) -> int:
    """20252026 -> 20242025."""
    return season - 10001


def season_totals(session, seasons: list[int]) -> pd.DataFrame:
    """Regular-season games and category totals, one row per player-season."""
    return pd.DataFrame(session.execute(text(_TOTALS_SQL), {"seasons": seasons}).mappings())


def league_rates(totals: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Each category's rate a game league-wide, by season, over everyone who played."""
    rows = totals.copy()
    rows[keys] = rows[keys].astype(float)
    return rows.groupby("season")[keys].sum().div(rows.groupby("season")["games"].sum(), axis=0)


def season_swing(
    totals: pd.DataFrame, scoring: LeagueScoring, last: int, league: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Last season's fantasy points a game against the two seasons before it.

    One row per player with MIN_GAMES or more in each of the three seasons.
    Anyone short of that has no row: a rookie has nothing to compare against,
    and a player who missed half a season has a per-game rate too thin to call.
    `swing` is the relative change, so +0.23 is a season 23% above the two
    before, and `flag` is "hot", "cold" or "". `league` is the rate table
    to adjust by, `league_rates(totals)` unless given.
    """
    seasons = [previous_season(previous_season(last)), previous_season(last), last]
    keys = [rule.key for rule in scoring.skater_rules]
    missing = [k for k in keys if k not in totals.columns]
    if missing:
        raise KeyError(f"season totals are missing scored categories: {missing}")
    rows = totals[totals["season"].isin(seasons)].copy()
    rows[keys] = rows[keys].astype(float)
    # Each season scaled to the last one's league rates. Plus/minus is left
    # alone: it sums to about zero league-wide, so its ratio is noise.
    if league is None:
        league = league_rates(rows, keys)
    for key in keys:
        if key != "plus_minus" and last in league.index:
            scale = (league.loc[last, key] / league[key]).replace([np.inf, -np.inf], np.nan)
            rows[key] = rows[key] * rows["season"].map(scale.fillna(1.0))
    rows = rows[rows["games"] >= MIN_GAMES]
    # A NULL total (ppp for a player who never saw the power play) is a zero,
    # which is how score_statline reads a missing stat.
    rows["points"] = [
        score_statline(
            scoring,
            {k: None if pd.isna(v) else float(v) for k, v in zip(keys, values, strict=True)},
            "P",
        )
        for values in rows[keys].itertuples(index=False)
    ]
    # Reindexed so a season nobody qualified in is a column of gaps, not a
    # missing column.
    points = rows.pivot(index="player_id", columns="season", values="points")
    games = rows.pivot(index="player_id", columns="season", values="games")
    points, games = points.reindex(columns=seasons), games.reindex(columns=seasons)
    whole = points.notna().all(axis=1)
    points, games = points[whole], games[whole]
    per_game = points / games
    before = (per_game[seasons[0]] + per_game[seasons[1]]) / 2
    out = pd.DataFrame(
        {
            "player_id": points.index.astype(int),
            "season": last,
            "games": games[last].astype(int).to_numpy(),
            "per_game": per_game[last].round(2).to_numpy(),
            "before_games": (games[seasons[0]] + games[seasons[1]]).astype(int).to_numpy(),
            "before_per_game": before.round(2).to_numpy(),
            "swing": (per_game[last] / before - 1).round(3).to_numpy(),
        }
    )
    out["flag"] = np.select([out["swing"] >= SWING, out["swing"] <= -SWING], ["hot", "cold"], "")
    return out.reset_index(drop=True)
