"""Hold out a season, project it, and score what came back.

The floor and ceiling this project exists to produce are only worth quoting if
the intervals behind them are calibrated. A 90% interval that contains 60% of
outcomes is worse than no interval at all, because it invites confident
decisions that are wrong more often than the number implies - and one that
contains 99% is hedging, and says almost nothing.

So: fit on every season up to the held-out one, project the held-out season
over the schedule its teams actually played, and compare the projected
distribution against what happened. Nothing from the held-out season reaches
the fit.

One deliberate exception, and it is the same information a drafter has: the
player's team for the held-out season is taken from that season, because on
draft day you know which roster someone is on. Their games played is not - that
is the thing being predicted.
"""

import logging

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from hockey.calibration.scores import calibration_report, render_report
from hockey.features import availability_panel, build_index_maps, skater_panel, team_schedule
from hockey.model import multi, project_multi
from hockey.model.multi import SIGNED_STAT
from hockey.scoring import LeagueScoring, score_draws
from hockey.seasons import FITTING_SEASONS, season_label

logger = logging.getLogger(__name__)

# The pool, chosen only from seasons before the held-out one so the selection
# itself carries no hindsight. A player picked because they were good in the
# test season would flatter every number that follows.
POOL_SQL = """
SELECT s.player_id,
       p.first_name || ' ' || p.last_name AS name,
       p.position,
       sum(s.goals + s.assists)::float AS points
  FROM skater_game_logs s
  JOIN nhl_games g ON g.nhl_game_id = s.game_id
  JOIN players p ON p.nhl_id = s.player_id
 WHERE g.game_type = 2
   AND g.season = ANY(:train_recent)
   AND p.position <> 'G'
 GROUP BY s.player_id, p.first_name, p.last_name, p.position
HAVING count(*) >= :min_games
 ORDER BY points DESC
 LIMIT :limit
"""

# The team a player actually skated for in the held-out season, by games. This
# is roster information, known on draft day; their games played is not.
TEST_TEAM_SQL = """
SELECT player_id, team_abbrev, count(*) AS games
  FROM skater_game_logs s
  JOIN nhl_games g ON g.nhl_game_id = s.game_id
 WHERE g.game_type = 2 AND g.season = :season AND s.player_id = ANY(:players)
 GROUP BY player_id, team_abbrev
"""

ACTUALS_SQL = """
SELECT s.player_id,
       count(*) AS games_played,
       sum(s.goals) goals, sum(s.assists) assists, sum(s.sog) sog,
       sum(s.hits) hits, sum(s.blocks) blocks,
       sum(coalesce(s.ppp, 0)) ppp, sum(coalesce(s.shp, 0)) shp,
       sum(s.plus_minus) plus_minus
  FROM skater_game_logs s
  JOIN nhl_games g ON g.nhl_game_id = s.game_id
 WHERE g.game_type = 2 AND g.season = :season AND s.player_id = ANY(:players)
 GROUP BY s.player_id
"""


def choose_pool(session: Session, train_seasons: list[int], limit: int, min_games: int = 60):
    recent = train_seasons[-2:]
    return pd.DataFrame(
        session.execute(
            text(POOL_SQL),
            {"train_recent": recent, "min_games": min_games, "limit": limit},
        ).mappings()
    )


def test_season_teams(session: Session, season: int, player_ids: list[int]) -> dict[int, str]:
    rows = pd.DataFrame(
        session.execute(text(TEST_TEAM_SQL), {"season": season, "players": player_ids}).mappings()
    )
    if rows.empty:
        return {}
    top = rows.sort_values("games", ascending=False).drop_duplicates("player_id")
    return dict(zip(top["player_id"], top["team_abbrev"], strict=True))


def actual_totals(session: Session, season: int, player_ids: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        session.execute(text(ACTUALS_SQL), {"season": season, "players": player_ids}).mappings()
    )


def prepare_backtest(session: Session, test_season: int, pool_size: int):
    """Everything the fit needs, with the held-out season kept out of it."""
    train_seasons = [s for s in FITTING_SEASONS if s < test_season]
    if not train_seasons:
        raise ValueError(f"nothing to train on before {test_season}")
    logger.info(
        "train %s -> test %s",
        f"{season_label(train_seasons[0])}..{season_label(train_seasons[-1])}",
        season_label(test_season),
    )

    pool = choose_pool(session, train_seasons, pool_size)
    player_ids = [int(p) for p in pool["player_id"]]

    # Only players who actually appeared in the held-out season can be scored.
    teams = test_season_teams(session, test_season, player_ids)
    pool = pool[pool["player_id"].isin(teams)].reset_index(drop=True)
    player_ids = [int(p) for p in pool["player_id"]]
    logger.info("pool: %d skaters who also appear in the test season", len(pool))

    maps = build_index_maps(session, seasons=train_seasons)
    panel = skater_panel(session, maps, player_ids=player_ids, seasons=train_seasons)
    availability = availability_panel(session, player_ids=player_ids, seasons=train_seasons)

    frames = []
    for pid in player_ids:
        games = team_schedule(session, maps, teams[pid], test_season)
        games = games.copy()
        games["player_id"] = pid
        frames.append(games)
    schedule = pd.concat(frames, ignore_index=True)

    data = multi.prepare(
        panel=panel,
        availability=availability,
        schedule=schedule,
        maps=maps,
        player_names={int(r.player_id): r.name for r in pool.itertuples()},
        positions={int(r.player_id): r.position for r in pool.itertuples()},
    )
    actuals = actual_totals(session, test_season, player_ids)
    return data, actuals


def score_backtest(
    idata, data: multi.MultiData, actuals: pd.DataFrame, scoring: LeagueScoring, seed: int = 7
) -> dict:
    """Calibration of the fantasy-point projection, and of each category."""
    projection = project_multi.project(idata, data, seed=seed)
    order = [pid for pid in data.players]
    actuals = actuals.set_index("player_id").reindex(order)

    reports = {}
    categories = [*data.stats, SIGNED_STAT]
    for stat in categories:
        observed = actuals[stat].to_numpy(dtype=float)
        reports[stat] = calibration_report(projection.totals[stat], observed)

    points_draws = score_draws(scoring, projection.totals, "P")
    weights = {r.key: r.modifier for r in scoring.skater_rules}
    observed_points = sum(weights[c] * actuals[c].to_numpy(dtype=float) for c in weights)
    reports["FANTASY POINTS"] = calibration_report(points_draws, observed_points)

    games_report = calibration_report(
        projection.games_played, actuals["games_played"].to_numpy(dtype=float)
    )
    reports["games played"] = games_report

    return {
        "reports": reports,
        "projection": projection,
        "points_draws": points_draws,
        "observed_points": observed_points,
        "players": [data.player_names[p] for p in order],
    }


def render(result: dict, test_season: int) -> str:
    """The table that decides whether floor and ceiling can be quoted."""
    lines = [
        f"=== calibration on held-out {season_label(test_season)} ===",
        "",
        "  Coverage is the share of players whose actual total fell inside the",
        "  interval. Close to nominal is calibrated; far below means the model is",
        "  overconfident and its floor and ceiling are too narrow; far above means",
        "  it is hedging and the interval says little.",
        "",
        f"  {'category':16s} {'n':>4s} {'CRPS':>8s} {'50%':>7s} {'80%':>7s} {'90%':>7s} "
        f"{'PIT dev':>8s}",
        "  " + "-" * 62,
    ]
    for name, report in result["reports"].items():
        cover = {row["level"]: row["coverage"] for row in report["coverage"]}
        lines.append(
            f"  {name:16s} {report['n']:4d} {report['crps_mean']:8.1f} "
            f"{cover[0.5]:6.0%} {cover[0.8]:6.0%} {cover[0.9]:6.0%} "
            f"{report['pit_deviation']:8.2f}"
        )
    lines.append("")
    lines.append(render_report(result["reports"]["FANTASY POINTS"], "fantasy points in detail"))
    return "\n".join(lines)


def biggest_misses(result: dict, n: int = 10) -> pd.DataFrame:
    """Where the projection was furthest out, in standard deviations."""
    draws = result["points_draws"]
    observed = result["observed_points"]
    mean = draws.mean(axis=0)
    sd = draws.std(axis=0)
    frame = pd.DataFrame(
        {
            "player": result["players"],
            "projected": mean,
            "floor": np.percentile(draws, 5, axis=0),
            "ceiling": np.percentile(draws, 95, axis=0),
            "actual": observed,
            "error": observed - mean,
            "z": (observed - mean) / sd,
        }
    )
    frame["inside_90"] = (frame["actual"] >= frame["floor"]) & (frame["actual"] <= frame["ceiling"])
    return frame.reindex(frame["z"].abs().sort_values(ascending=False).index).head(n)
