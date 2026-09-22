"""The draft-day context that is not a posterior: a calendar and an injury list.

The board answers what a player is worth over a season. Two questions at the
table are not that, and neither comes out of the model:

  - would this player actually be in my lineup in the first two weeks, or do his
    games land on nights my starters already cover;
  - can he play at all right now.

Both are warehouse facts, so they are written into the board directory beside
the posterior rather than read live. That is deliberate: the draft-day server
must not need Postgres. A draft is two hours in which a container that will not
start is unrecoverable, and the schedule does not change while it runs.
"""

import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from hockey.db import SessionLocal
from hockey.seasons import PROJECTION_SEASON
from hockey.yahoo.settings import load_weeks_from_yaml

logger = logging.getLogger(__name__)


def write_draft_context(out: Path, season: int = PROJECTION_SEASON) -> bool:
    """Write schedule.csv, weeks.csv and injuries.csv into a board directory.

    Returns False when the warehouse is unreachable, having written nothing and
    said so. An export without a calendar is a board that cannot answer the
    schedule questions, which is worse than the alternative only if it pretends
    otherwise - so the files are absent rather than empty, and every consumer
    checks.
    """
    weeks = load_weeks_from_yaml()
    try:
        with SessionLocal() as session:
            games = pd.DataFrame(
                session.execute(
                    text(
                        """
                        select date, home_team_abbrev, away_team_abbrev
                        from nhl_games
                        where season = :season and game_type = 2
                        order by date
                        """
                    ),
                    {"season": season},
                ).all(),
                columns=["date", "home", "away"],
            )
            injuries = pd.DataFrame(
                session.execute(
                    text(
                        """
                        select player_id, status, injury_type, synced_at
                        from player_injuries
                        order by player_id
                        """
                    )
                ).all(),
                columns=["player_id", "status", "injury_type", "synced_at"],
            )
    except Exception as exc:  # the warehouse is down, or empty
        logger.warning(
            "no draft context written: the warehouse is unreachable (%s). The board "
            "still values players; it cannot answer schedule or injury questions.",
            str(exc).splitlines()[0][:100],
        )
        return False

    if games.empty:
        logger.warning(
            "no %s games in the warehouse, so no schedule was written. Run "
            "`python -m hockey.ingest schedule --season %s` first.",
            season,
            season,
        )
        return False

    # One row per team-game, which is the shape every consumer wants: a team's
    # dates. Home and away are the same question here.
    calendar = pd.concat(
        [
            games[["date", "home"]].rename(columns={"home": "team"}),
            games[["date", "away"]].rename(columns={"away": "team"}),
        ]
    ).sort_values(["team", "date"])
    calendar.to_csv(out / "schedule.csv", index=False)
    pd.DataFrame(weeks).to_csv(out / "weeks.csv", index=False)
    injuries.to_csv(out / "injuries.csv", index=False)

    logger.info(
        "draft context: %d team-games across %d teams, %d week(s), %d injured player(s)",
        len(calendar),
        calendar["team"].nunique(),
        len(weeks),
        len(injuries),
    )
    if not weeks:
        logger.warning(
            "no scoring weeks in the league config, so the opening-fortnight "
            "question cannot be asked. Add a `weeks:` block to config/league_*.yaml."
        )
    return True
