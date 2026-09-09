"""CLI: python -m hockey.ingest <target> [--season 20252026 | --seasons 20182019-20252026] [--all]

Unlike the app this was lifted from, this project backfills many seasons at
once, so every season-scoped target accepts a range. Seasons run in ascending
order deliberately: roster syncs overwrite players.team_abbrev, so finishing on
the newest season leaves each player pointing at their current team.
"""

import argparse
import logging

from hockey.db import SessionLocal
from hockey.ingest import sync
from hockey.ingest.client import EspnApiClient, NhlApiClient

# Fitting window agreed in docs/architecture.md: 2018-19 through 2025-26.
DEFAULT_SEASONS = [
    20182019,
    20192020,
    20202021,
    20212022,
    20222023,
    20232024,
    20242025,
    20252026,
]
CURRENT_SEASON = 20262027

logger = logging.getLogger(__name__)

SEASON_SCOPED = {"players", "schedule", "game-logs", "player-stats", "play-by-play"}


def season_range(start: int, end: int) -> list[int]:
    """Every NHL season id from start to end inclusive. Season ids are
    start_year * 10000 + end_year, so they cannot simply be incremented."""
    if start > end:
        raise ValueError(f"season range is backwards: {start} > {end}")
    seasons = []
    year = start // 10000
    while year * 10000 + year + 1 <= end:
        seasons.append(year * 10000 + year + 1)
        year += 1
    return seasons


def parse_seasons(arg: str) -> list[int]:
    """'20242025' -> one season; '20182019-20252026' -> the inclusive range."""
    if "-" in arg:
        start, end = arg.split("-", 1)
        return season_range(int(start), int(end))
    return [int(arg)]


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m hockey.ingest")
    parser.add_argument(
        "target",
        choices=[
            "teams",
            "players",
            "schedule",
            "game-logs",
            "player-stats",
            "play-by-play",
            "injuries",
            "backfill",
        ],
        help="'backfill' runs teams + players + schedule + game-logs + player-stats "
        "over every season in --seasons, in order.",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help=f"a single season id, e.g. {CURRENT_SEASON}",
    )
    parser.add_argument(
        "--seasons",
        type=str,
        default=None,
        help="a season or inclusive range, e.g. 20182019-20252026. "
        "Defaults to the full fitting window for `backfill`.",
    )
    parser.add_argument(
        "--all", dest="force", action="store_true", help="re-sync game logs already in the DB"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.seasons:
        seasons = parse_seasons(args.seasons)
    elif args.season:
        seasons = [args.season]
    elif args.target == "backfill":
        seasons = DEFAULT_SEASONS
    else:
        seasons = [CURRENT_SEASON]

    if args.target in SEASON_SCOPED or args.target == "backfill":
        logger.info("seasons: %s", ", ".join(str(s) for s in seasons))

    client = NhlApiClient()
    try:
        with SessionLocal() as session:
            if args.target == "teams":
                sync.sync_teams(session, client, None if args.seasons is None else seasons[-1])

            for season in seasons:
                # Which franchises existed this season. Driving roster and
                # schedule calls off today's team list 404s (Seattle before
                # 2021-22) and silently drops a franchise that has since gone
                # (Arizona from 2024-25). See sync.sync_teams.
                abbrevs: list[str] | None = None
                if args.target in ("players", "schedule", "backfill"):
                    abbrevs = sync.sync_teams(session, client, season)
                    logger.info("=== season %d: %d teams ===", season, len(abbrevs))

                if args.target in ("players", "backfill"):
                    logger.info("=== players %d ===", season)
                    sync.sync_players(session, client, season, abbrevs)
                if args.target in ("schedule", "backfill"):
                    logger.info("=== schedule %d ===", season)
                    sync.sync_schedule(session, client, season, abbrevs)
                if args.target in ("game-logs", "backfill"):
                    logger.info("=== game logs %d ===", season)
                    sync.sync_game_logs(session, client, season, force=args.force)
                if args.target in ("player-stats", "backfill"):
                    logger.info("=== player stats %d ===", season)
                    sync.sync_player_stats(session, client, season)
                # Play-by-play is deliberately not part of `backfill`. This
                # league scores no faceoff category, so the ~3.3M events it
                # would add buy nothing today. One command enables it if that
                # ever changes. See docs/architecture.md.
                if args.target == "play-by-play":
                    logger.info("=== play-by-play %d ===", season)
                    sync.sync_play_by_play(session, client, season, force=args.force)

            # Injuries come from ESPN (different host, own client) and are
            # current-state only, so they are not season-scoped.
            if args.target == "injuries":
                espn = EspnApiClient()
                try:
                    sync.sync_injuries(session, espn)
                finally:
                    espn.close()
    finally:
        client.close()


if __name__ == "__main__":
    main()
