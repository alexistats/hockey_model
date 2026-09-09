"""Pure functions turning raw NHL API shapes into DB row dicts."""

import unicodedata
from datetime import date

from hockey.ingest import schemas


def normalize_name(value: str) -> str:
    """Fold a name for cross-source matching (ESPN injuries and Yahoo players -> our players):
    strip accents, lowercase, drop periods/apostrophes, treat hyphens as spaces,
    collapse whitespace. 'P.K. Subban' / 'Tim Stützle' match reliably."""
    stripped = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    cleaned = stripped.lower().replace(".", "").replace("'", "").replace("-", " ")
    return " ".join(cleaned.split())


# The API uses L/R; our schema (and fantasy convention) uses LW/RW.
POSITION_MAP = {"C": "C", "L": "LW", "R": "RW", "D": "D", "G": "G"}

# Game states observed for completed games.
COMPLETED_GAME_STATES = {"OFF", "FINAL"}


def map_position(code: str) -> str:
    return POSITION_MAP[code]


def toi_to_seconds(toi: str) -> int:
    minutes, seconds = toi.split(":")
    return int(minutes) * 60 + int(seconds)


def team_row(team: schemas.RawStandingsTeam) -> dict:
    """One nhl_teams row.

    The conference key is omitted rather than set to None when the season had
    no conferences (2020-21). upsert_rows only updates the columns a row
    actually provides, so omitting it leaves whatever a neighbouring season
    already wrote intact instead of erasing it.
    """
    row = {
        "abbrev": team.team_abbrev.default,
        "name": team.team_name.default,
        "division": team.division_name,
    }
    if team.conference_name is not None:
        row["conference"] = team.conference_name
    return row


def player_row(player: schemas.RawRosterPlayer, team_abbrev: str) -> dict:
    return {
        "nhl_id": player.id,
        "first_name": player.first_name.default,
        "last_name": player.last_name.default,
        "position": map_position(player.position_code),
        "team_abbrev": team_abbrev,
        "status": "active",
        "shoots_catches": player.shoots_catches,
    }


def game_row(game: schemas.RawScheduleGame) -> dict:
    return {
        "nhl_game_id": game.id,
        "date": date.fromisoformat(game.game_date),
        "start_time_utc": game.start_time_utc,
        "season": game.season,
        "game_type": game.game_type,
        "home_team_abbrev": game.home_team.abbrev,
        "away_team_abbrev": game.away_team.abbrev,
        "game_state": game.game_state,
        "home_score": game.home_team.score,
        "away_score": game.away_team.score,
    }


def skater_log_row(skater: schemas.RawBoxscoreSkater, game_id: int, team_abbrev: str) -> dict:
    return {
        "player_id": skater.player_id,
        "game_id": game_id,
        "team_abbrev": team_abbrev,
        # Position actually skated this game (C/L/R/D -> C/LW/RW/D); drives
        # multi-position eligibility. Distinct from the roster's primary position.
        "position": map_position(skater.position),
        "goals": skater.goals,
        "assists": skater.assists,
        "points": skater.points,
        "plus_minus": skater.plus_minus,
        "pim": skater.pim,
        "sog": skater.sog,
        "hits": skater.hits,
        "blocks": skater.blocked_shots,
        "toi_seconds": toi_to_seconds(skater.toi),
    }


def goalie_log_row(goalie: schemas.RawBoxscoreGoalie, game_id: int, team_abbrev: str) -> dict:
    return {
        "player_id": goalie.player_id,
        "game_id": game_id,
        "team_abbrev": team_abbrev,
        "decision": goalie.decision,
        "wins": 1 if goalie.decision == "W" else 0,
        "goals_against": goalie.goals_against,
        "saves": goalie.saves,
        "shots_against": goalie.shots_against,
        "toi_seconds": toi_to_seconds(goalie.toi),
        "started": goalie.starter,
    }


def play_row(play: schemas.RawPlay, game_id: int) -> dict:
    """One nhl_plays row: common fields promoted to columns, the rest kept in the
    raw `details` JSONB (see the NhlPlay model)."""
    det = play.details or {}
    return {
        "game_id": game_id,
        "event_id": play.event_id,
        "sort_order": play.sort_order,
        "period_number": play.period_descriptor.number,
        "period_type": play.period_descriptor.period_type,
        "time_in_period": play.time_in_period,
        "type_code": play.type_code,
        "type_desc_key": play.type_desc_key,
        "situation_code": play.situation_code,
        "home_team_defending_side": play.home_team_defending_side,
        "x_coord": det.get("xCoord"),
        "y_coord": det.get("yCoord"),
        "zone_code": det.get("zoneCode"),
        "event_owner_team_id": det.get("eventOwnerTeamId"),
        "details": det,
    }


def player_stub_row(player_id: int, name: schemas.LocalizedName, position_code: str) -> dict:
    """Minimal player record for boxscore players missing from rosters
    (call-ups, mid-season trades). Boxscore names are abbreviated
    ("C. McDavid"), so split on the first space-dot boundary.
    """
    full = name.default
    first, _, last = full.partition(" ")
    return {
        "nhl_id": player_id,
        "first_name": first,
        "last_name": last or full,
        "position": map_position(position_code),
        "team_abbrev": None,
        "status": "inactive",
        "shoots_catches": None,
    }
