"""Sync functions: NHL API -> Postgres. All upserts are idempotent.

Skater/goalie log rows are assembled from two sources (see
docs/architecture.md): boxscores carry hits/blocks/SOG/decision but not
PPP/SHP/SV%/shutouts; per-player game logs fill those in afterwards.
"""

import logging
import time
from datetime import UTC, date, datetime

from sqlalchemy import bindparam, delete, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from hockey.ingest import mappers, schemas
from hockey.ingest.client import EspnApiClient, NhlApiClient
from hockey.models import (
    Base,
    GoalieGameLog,
    NhlGame,
    NhlPlay,
    NhlTeam,
    Player,
    PlayerInjury,
    SkaterGameLog,
)

logger = logging.getLogger(__name__)

SYNCED_GAME_TYPES = (2, 3)  # regular season, playoffs
BOXSCORE_DELAY_SECONDS = 0.2
# The per-player pass has its own delay because this project runs it over 8
# seasons (~8k calls), not the single season the original was written for.
PLAYER_STATS_DELAY_SECONDS = 0.1


def upsert_rows(session: Session, model: type[Base], rows: list[dict]) -> None:
    """Insert rows, updating on conflict only the **non-PK columns actually
    provided** (the union of the rows' keys). A partial-column pass therefore
    never clobbers columns another pass fills: game-logs (no ppp/save_pctg/
    shutouts) leaves the player-stats columns intact, and re-syncing teams
    leaves the schedule-backfilled nhl_team_id intact. The two-source design
    (docs/architecture.md) depends on this."""
    if not rows:
        return
    table = model.__table__
    pk_cols = {c.name for c in table.primary_key.columns}
    provided = {key for row in rows for key in row}
    stmt = insert(table).values(rows)
    update_cols = {name: stmt.excluded[name] for name in provided if name not in pk_cols}
    if update_cols:
        session.execute(stmt.on_conflict_do_update(index_elements=list(pk_cols), set_=update_cols))
    else:  # rows carry only PK columns — nothing to update
        session.execute(stmt.on_conflict_do_nothing(index_elements=list(pk_cols)))


def insert_missing(session: Session, model: type[Base], rows: list[dict]) -> None:
    """Insert rows, leaving existing rows untouched (for player stubs)."""
    if not rows:
        return
    table = model.__table__
    pk_cols = [c.name for c in table.primary_key.columns]
    stmt = insert(table).values(rows).on_conflict_do_nothing(index_elements=pk_cols)
    session.execute(stmt)


def season_standings_date(season: int) -> str:
    """A calendar date guaranteed to fall inside a season, for the
    standings-by-date endpoint. February 1 of the end year is inside every
    season since the 2012-13 lockout, including 2020-21, which did not start
    until January 13, 2021."""
    end_year = season % 10000
    return f"{end_year}-02-01"


def sync_teams(session: Session, client: NhlApiClient, season: int | None = None) -> list[str]:
    """Upsert the teams that existed in `season` (or today's teams when season
    is None) and return their abbreviations.

    Franchises come and go inside the fitting window - Seattle arrived in
    2021-22, and Arizona became Utah in 2024-25 - so a backfill driven off the
    current team list 404s on roster and schedule calls for seasons where a
    team did not exist, and silently omits one that no longer does. The
    standings-by-date endpoint is the only source that reports the franchises
    of a past season, so it drives every season-scoped sync.

    Historical teams are upserted, never removed: nhl_games rows for 2018-19
    Arizona games carry a foreign key to nhl_teams.abbrev = 'ARI', so that row
    has to stay even though the franchise no longer exists.

    A season that has not started yet has no standings on any date inside it -
    the endpoint answers 200 with an empty list - so the upcoming season, which
    is the one being projected, falls back to today's teams. Those are the
    right teams for it.
    """
    payload = (
        client.standings_on(season_standings_date(season))
        if season is not None
        else client.standings_now()
    )
    raw = schemas.RawStandings.model_validate(payload)
    if season is not None and not raw.standings:
        logger.info(
            "no standings inside season %d (it has not been played yet); "
            "using the current team list",
            season,
        )
        raw = schemas.RawStandings.model_validate(client.standings_now())
    rows = [mappers.team_row(t) for t in raw.standings]
    upsert_rows(session, NhlTeam, rows)
    session.commit()
    abbrevs = sorted(row["abbrev"] for row in rows)
    logger.info("synced %d teams%s", len(rows), f" for season {season}" if season else "")
    return abbrevs


def _resolve_abbrevs(session: Session, abbrevs: list[str] | None) -> list[str]:
    if abbrevs is not None:
        return sorted(abbrevs)
    known = list(session.scalars(select(NhlTeam.abbrev).order_by(NhlTeam.abbrev)))
    if not known:
        raise RuntimeError("no teams in DB - run `sync teams` first")
    return known


def sync_players(
    session: Session, client: NhlApiClient, season: int, abbrevs: list[str] | None = None
) -> int:
    abbrevs = _resolve_abbrevs(session, abbrevs)
    total = 0
    for abbrev in abbrevs:
        raw = schemas.RawRoster.model_validate(client.team_roster(abbrev, season))
        players = raw.forwards + raw.defensemen + raw.goalies
        rows = [mappers.player_row(p, abbrev) for p in players]
        upsert_rows(session, Player, rows)
        total += len(rows)
    session.commit()
    logger.info("synced %d players across %d teams", total, len(abbrevs))
    return total


def sync_schedule(
    session: Session, client: NhlApiClient, season: int, abbrevs: list[str] | None = None
) -> int:
    abbrevs = _resolve_abbrevs(session, abbrevs)
    games: dict[int, dict] = {}
    team_ids: dict[str, int] = {}
    for abbrev in abbrevs:
        raw = schemas.RawClubSchedule.model_validate(client.club_schedule_season(abbrev, season))
        for game in raw.games:
            if game.game_type not in SYNCED_GAME_TYPES:
                continue
            games[game.id] = mappers.game_row(game)
            team_ids[game.home_team.abbrev] = game.home_team.id
            team_ids[game.away_team.abbrev] = game.away_team.id
    upsert_rows(session, NhlGame, list(games.values()))
    # Backfill numeric NHL team ids (the standings source doesn't carry them).
    for abbrev, nhl_team_id in team_ids.items():
        session.execute(
            update(NhlTeam).where(NhlTeam.abbrev == abbrev).values(nhl_team_id=nhl_team_id)
        )
    session.commit()
    logger.info("synced %d games for season %d", len(games), season)
    return len(games)


def apply_boxscore(session: Session, raw: schemas.RawBoxscore) -> None:
    """Upsert all log rows (and player stubs) from one parsed boxscore.

    The side of the boxscore a player is listed under is the only record of who
    they dressed for that night, so it is captured here rather than inferred
    later from players.team_abbrev - that column holds one current value and is
    wrong for every past season and every trade.
    """
    # (player, team_abbrev) pairs, keeping the side each player was listed on.
    skaters: list[tuple[schemas.RawBoxscoreSkater, str]] = []
    goalies: list[tuple[schemas.RawBoxscoreGoalie, str]] = []
    sides = (
        (raw.player_by_game_stats.home_team, raw.home_team.abbrev),
        (raw.player_by_game_stats.away_team, raw.away_team.abbrev),
    )
    for side, team_abbrev in sides:
        # The NHL boxscore occasionally lists a dressed backup goalie among the
        # forwards or defence with 0:00 played. Letting one into
        # skater_game_logs is not just a junk row: the player-stats pass then
        # asks the per-player endpoint for that goalie's *skater* log, gets a
        # goalie-shaped payload back with no powerPlayPoints, and the whole
        # backfill stops. Filter on the position the boxscore itself reports.
        skaters.extend((s, team_abbrev) for s in side.forwards + side.defense if s.position != "G")
        goalies.extend((g, team_abbrev) for g in side.goalies if mappers.toi_to_seconds(g.toi) > 0)

    everyone = [p for p, _ in skaters] + [p for p, _ in goalies]
    known_ids = set(
        session.scalars(
            select(Player.nhl_id).where(Player.nhl_id.in_([p.player_id for p in everyone]))
        )
    )
    stubs = [
        mappers.player_stub_row(p.player_id, p.name, p.position)
        for p in everyone
        if p.player_id not in known_ids
    ]
    insert_missing(session, Player, stubs)

    upsert_rows(session, SkaterGameLog, [mappers.skater_log_row(s, raw.id, t) for s, t in skaters])
    upsert_rows(session, GoalieGameLog, [mappers.goalie_log_row(g, raw.id, t) for g, t in goalies])


def sync_game_logs(session: Session, client: NhlApiClient, season: int, force: bool = False) -> int:
    stmt = (
        select(NhlGame.nhl_game_id)
        .where(
            NhlGame.season == season,
            NhlGame.game_type.in_(SYNCED_GAME_TYPES),
            NhlGame.game_state.in_(mappers.COMPLETED_GAME_STATES),
        )
        .order_by(NhlGame.nhl_game_id)
    )
    if not force:
        synced = select(SkaterGameLog.game_id).distinct()
        stmt = stmt.where(NhlGame.nhl_game_id.not_in(synced))
    game_ids = list(session.scalars(stmt))
    logger.info("syncing logs for %d games", len(game_ids))
    for i, game_id in enumerate(game_ids, start=1):
        raw = schemas.RawBoxscore.model_validate(client.boxscore(game_id))
        apply_boxscore(session, raw)
        session.commit()
        if i % 100 == 0:
            logger.info("game logs: %d/%d", i, len(game_ids))
        time.sleep(BOXSCORE_DELAY_SECONDS)
    return len(game_ids)


def sync_player_bios(session: Session, client: NhlApiClient, refresh: bool = False) -> int:
    """Fill players.birth_date from the per-player landing page.

    Only players who are missing one are fetched, so this is cheap to re-run
    after a roster sync brings in new names. A birth date never changes, which
    is why `refresh` exists but defaults off.
    """
    stmt = select(Player.nhl_id).order_by(Player.nhl_id)
    if not refresh:
        stmt = stmt.where(Player.birth_date.is_(None))
    player_ids = list(session.scalars(stmt))
    logger.info("fetching bios for %d players", len(player_ids))

    # UPDATE, not upsert_rows. These players already exist and only one column
    # is being filled, but Postgres validates the proposed INSERT row before it
    # gets to the conflict clause, so a row carrying just an id and a birth date
    # trips the NOT NULL on first_name. upsert_rows is for rows that would be
    # valid as an insert; this is a pure update.
    table = Player.__table__
    statement = (
        update(table).where(table.c.nhl_id == bindparam("pid")).values(birth_date=bindparam("born"))
    )

    def flush(batch):
        if batch:
            session.execute(statement, batch)
            session.commit()

    rows, missing = [], 0
    for i, player_id in enumerate(player_ids, start=1):
        raw = schemas.RawPlayerLanding.model_validate(client.player_landing(player_id))
        if raw.birth_date:
            rows.append({"pid": player_id, "born": date.fromisoformat(raw.birth_date)})
        else:
            missing += 1
        if len(rows) >= 200:
            flush(rows)
            rows = []
        if i % 200 == 0:
            logger.info("bios: %d/%d", i, len(player_ids))
        time.sleep(PLAYER_STATS_DELAY_SECONDS)
    flush(rows)
    logger.info(
        "bios: %d players updated, %d had no birth date", len(player_ids) - missing, missing
    )
    return len(player_ids) - missing


def sync_injuries(session: Session, client: EspnApiClient) -> dict[str, int]:
    """Refresh current player injuries from ESPN. ESPN carries no NHL player id,
    so athletes are matched to our players by normalized name, with the team as a
    tiebreak for duplicate names. Matched injuries are upserted; players no longer
    listed (recovered) have their rows deleted; unmatched ESPN entries (typically
    prospects not on any synced roster) are logged and skipped. Commits.
    Returns {'matched', 'missed'}."""
    raw = schemas.RawEspnInjuries.model_validate(client.injuries())
    now = datetime.now(UTC)

    name_index: dict[tuple[str, str], list[tuple[int, str | None]]] = {}
    for pid, first, last, team_abbrev in session.execute(
        select(Player.nhl_id, Player.first_name, Player.last_name, Player.team_abbrev)
    ):
        key = (mappers.normalize_name(first), mappers.normalize_name(last))
        name_index.setdefault(key, []).append((pid, team_abbrev))
    team_abbrev_by_name = dict(session.execute(select(NhlTeam.name, NhlTeam.abbrev)).all())

    rows: list[dict] = []
    seen: set[int] = set()
    missed = 0
    for team in raw.injuries:
        team_abbrev = team_abbrev_by_name.get(team.display_name)
        for inj in team.injuries:
            key = (
                mappers.normalize_name(inj.athlete.first_name),
                mappers.normalize_name(inj.athlete.last_name),
            )
            cands = name_index.get(key, [])
            if len(cands) == 1:
                pid = cands[0][0]
            elif len(cands) > 1 and team_abbrev is not None:
                pid = next((c[0] for c in cands if c[1] == team_abbrev), None)
            else:
                pid = None
            if pid is None or pid in seen:
                missed += 1
                logger.info(
                    "injury unmatched: %s %s (%s)",
                    inj.athlete.first_name,
                    inj.athlete.last_name,
                    team.display_name,
                )
                continue
            det = inj.details
            rows.append(
                {
                    "player_id": pid,
                    "status": inj.status,
                    "injury_type": det.type if det else None,
                    "detail": det.detail if det else None,
                    "return_date": det.return_date if det else None,
                    "comment": inj.short_comment,
                    "espn_updated": inj.date,
                    "synced_at": now,
                }
            )
            seen.add(pid)

    upsert_rows(session, PlayerInjury, rows)
    # Wholesale refresh: drop rows for players no longer listed as injured.
    if seen:
        session.execute(delete(PlayerInjury).where(PlayerInjury.player_id.not_in(seen)))
    else:
        session.execute(delete(PlayerInjury))
    session.commit()
    logger.info("injuries: %d matched, %d unmatched", len(rows), missed)
    return {"matched": len(rows), "missed": missed}


def sync_play_by_play(
    session: Session, client: NhlApiClient, season: int, force: bool = False
) -> int:
    """Ingest play-by-play events for a season's completed games (analytics
    dataset). Idempotent upsert keyed (game_id, event_id); mirrors game-logs.
    Without `force`, games that already have plays are skipped."""
    stmt = (
        select(NhlGame.nhl_game_id)
        .where(
            NhlGame.season == season,
            NhlGame.game_type.in_(SYNCED_GAME_TYPES),
            NhlGame.game_state.in_(mappers.COMPLETED_GAME_STATES),
        )
        .order_by(NhlGame.nhl_game_id)
    )
    if not force:
        synced = select(NhlPlay.game_id).distinct()
        stmt = stmt.where(NhlGame.nhl_game_id.not_in(synced))
    game_ids = list(session.scalars(stmt))
    logger.info("syncing play-by-play for %d games", len(game_ids))
    total_events = 0
    for i, game_id in enumerate(game_ids, start=1):
        raw = schemas.RawPlayByPlay.model_validate(client.play_by_play(game_id))
        rows = [mappers.play_row(p, game_id) for p in raw.plays]
        upsert_rows(session, NhlPlay, rows)
        session.commit()
        total_events += len(rows)
        if i % 100 == 0:
            logger.info("play-by-play: %d/%d games (%d events)", i, len(game_ids), total_events)
        time.sleep(BOXSCORE_DELAY_SECONDS)
    logger.info("play-by-play: %d games, %d events", len(game_ids), total_events)
    return len(game_ids)


def sync_player_stats(session: Session, client: NhlApiClient, season: int) -> int:
    """Fill stats only the per-player game-log endpoint provides: skater PPP and
    SHP, goalie SV% and shutouts. The boxscore carries none of these.

    Naturally idempotent - it overwrites the same values - so it is safe to
    re-run after an interrupted backfill.
    """
    # (player_id, game_type) pairs that actually have log rows for this season.
    # Goalie rows are excluded even though apply_boxscore should never create
    # one: the per-player endpoint answers by the player's real position, so a
    # single misfiled goalie returns a payload with no powerPlayPoints and
    # halts the run.
    skater_pairs = session.execute(
        select(SkaterGameLog.player_id, NhlGame.game_type)
        .join(NhlGame, SkaterGameLog.game_id == NhlGame.nhl_game_id)
        .where(NhlGame.season == season, SkaterGameLog.position != "G")
        .distinct()
    ).all()
    goalie_pairs = session.execute(
        select(GoalieGameLog.player_id, NhlGame.game_type)
        .join(NhlGame, GoalieGameLog.game_id == NhlGame.nhl_game_id)
        .where(NhlGame.season == season)
        .distinct()
    ).all()

    logger.info(
        "player-stats %d: %d skater and %d goalie player-seasons to fetch",
        season,
        len(skater_pairs),
        len(goalie_pairs),
    )
    calls = 0
    skater_params: list[dict] = []
    for player_id, game_type in skater_pairs:
        raw = schemas.RawSkaterGameLog.model_validate(
            client.player_game_log(player_id, season, game_type)
        )
        calls += 1
        time.sleep(PLAYER_STATS_DELAY_SECONDS)
        skater_params.extend(
            {
                "p_id": player_id,
                "g_id": e.game_id,
                "v_ppp": e.power_play_points,
                "v_shp": e.shorthanded_points,
            }
            for e in raw.game_log
        )
    if skater_params:
        # Core-table update: executemany against the ORM class would take the
        # "bulk update by primary key" path, which rejects WHERE bindparams.
        skater_table = SkaterGameLog.__table__
        session.execute(
            update(skater_table)
            .where(
                skater_table.c.player_id == bindparam("p_id"),
                skater_table.c.game_id == bindparam("g_id"),
            )
            .values(ppp=bindparam("v_ppp"), shp=bindparam("v_shp")),
            skater_params,
        )

    goalie_params: list[dict] = []
    for player_id, game_type in goalie_pairs:
        raw = schemas.RawGoalieGameLog.model_validate(
            client.player_game_log(player_id, season, game_type)
        )
        calls += 1
        time.sleep(PLAYER_STATS_DELAY_SECONDS)
        goalie_params.extend(
            {"p_id": player_id, "g_id": e.game_id, "v_pct": e.save_pctg, "v_so": e.shutouts}
            for e in raw.game_log
        )
    if goalie_params:
        goalie_table = GoalieGameLog.__table__
        session.execute(
            update(goalie_table)
            .where(
                goalie_table.c.player_id == bindparam("p_id"),
                goalie_table.c.game_id == bindparam("g_id"),
            )
            .values(save_pctg=bindparam("v_pct"), shutouts=bindparam("v_so")),
            goalie_params,
        )

    session.commit()
    logger.info(
        "player-stats: %d API calls, %d skater rows, %d goalie rows updated",
        calls,
        len(skater_params),
        len(goalie_params),
    )
    return calls
