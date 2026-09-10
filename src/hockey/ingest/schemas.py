"""Pydantic models of raw NHL API responses.

Only the fields we consume are declared; required fields make API shape
changes fail loudly here instead of corrupting data downstream.
Field names mirror the API's camelCase via alias generation.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class RawModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class LocalizedName(RawModel):
    default: str


# --- standings/now (source of the team list) ---


class RawStandingsTeam(RawModel):
    team_abbrev: LocalizedName
    team_name: LocalizedName
    # Absent for 2020-21 only, and correctly so: that season was played in four
    # temporary divisions (Scotia North, MassMutual East, Discover Central,
    # Honda West) with no conference structure at all. Every other season in
    # the window carries it. Optional here rather than required because the
    # field is genuinely missing from the data, not from the payload.
    conference_name: str | None = None
    division_name: str


class RawStandings(RawModel):
    standings: list[RawStandingsTeam]


# --- roster/{team}/{season} ---


class RawRosterPlayer(RawModel):
    id: int
    first_name: LocalizedName
    last_name: LocalizedName
    position_code: str  # C / L / R / D / G
    shoots_catches: str | None = None


class RawRoster(RawModel):
    forwards: list[RawRosterPlayer]
    defensemen: list[RawRosterPlayer]
    goalies: list[RawRosterPlayer]


# --- club-schedule-season/{team}/{season} ---


class RawScheduleTeam(RawModel):
    id: int
    abbrev: str
    score: int | None = None


class RawScheduleGame(RawModel):
    id: int
    season: int
    game_type: int  # 1 preseason / 2 regular / 3 playoffs
    game_date: str  # YYYY-MM-DD
    # Explicit alias: the API spells it "startTimeUTC", not camelCase "startTimeUtc".
    start_time_utc: datetime = Field(alias="startTimeUTC")
    game_state: str
    home_team: RawScheduleTeam
    away_team: RawScheduleTeam


class RawClubSchedule(RawModel):
    games: list[RawScheduleGame]


# --- gamecenter/{gameId}/boxscore ---


class RawBoxscoreSkater(RawModel):
    player_id: int
    name: LocalizedName
    position: str
    goals: int
    assists: int
    points: int
    plus_minus: int
    pim: int
    hits: int
    blocked_shots: int
    sog: int
    toi: str  # "MM:SS"


class RawBoxscoreGoalie(RawModel):
    player_id: int
    name: LocalizedName
    position: str
    goals_against: int
    saves: int
    shots_against: int
    toi: str
    starter: bool = False
    decision: str | None = None  # W / L / O; absent if the goalie didn't play


class RawBoxscoreTeamPlayers(RawModel):
    forwards: list[RawBoxscoreSkater]
    defense: list[RawBoxscoreSkater]
    goalies: list[RawBoxscoreGoalie]


class RawBoxscorePlayerStats(RawModel):
    home_team: RawBoxscoreTeamPlayers
    away_team: RawBoxscoreTeamPlayers


class RawBoxscoreGameTeam(RawModel):
    id: int
    abbrev: str
    score: int | None = None


class RawBoxscore(RawModel):
    id: int
    season: int
    game_type: int
    game_state: str
    home_team: RawBoxscoreGameTeam
    away_team: RawBoxscoreGameTeam
    player_by_game_stats: RawBoxscorePlayerStats


# --- player/{id}/game-log/{season}/{gameType} ---


class RawSkaterGameLogEntry(RawModel):
    game_id: int
    power_play_points: int
    # The boxscore carries neither PPP nor SHP. This league scores SHP at 1.5,
    # so it is pulled in the same pass rather than left NULL.
    shorthanded_points: int


class RawGoalieGameLogEntry(RawModel):
    game_id: int
    save_pctg: float | None = None  # absent when shots_against == 0
    shutouts: int


# The per-player game-log endpoint omits `gameLog` entirely for a player with no
# games in the requested season/type (the payload is {..., "playerStatsSeasons": []}).
# That is a documented empty case, so default to no entries rather than failing. The
# entry models above keep their fields required, so a genuine shape change — a real
# game-log row missing a field — still breaks loudly (per docs/architecture.md).
class RawSkaterGameLog(RawModel):
    game_log: list[RawSkaterGameLogEntry] = Field(default_factory=list)


class RawGoalieGameLog(RawModel):
    game_log: list[RawGoalieGameLogEntry] = Field(default_factory=list)


# --- gamecenter/{gameId}/play-by-play ---


class RawPlayPeriod(RawModel):
    number: int
    period_type: str


class RawPlay(RawModel):
    event_id: int
    sort_order: int
    type_code: int
    type_desc_key: str
    time_in_period: str
    situation_code: str | None = None
    home_team_defending_side: str | None = None
    period_descriptor: RawPlayPeriod
    # `details` is a raw passthrough (heterogeneous per event type) stored as
    # JSONB; typed `dict` so pydantic keeps its camelCase keys untouched.
    details: dict = Field(default_factory=dict)


class RawPlayByPlay(RawModel):
    id: int
    plays: list[RawPlay]


# --- ESPN (unofficial) /injuries ---


class RawEspnAthlete(RawModel):
    first_name: str
    last_name: str


class RawEspnInjuryDetails(RawModel):
    type: str | None = None  # body part, e.g. "Knee"
    detail: str | None = None  # e.g. "Surgery"
    return_date: str | None = None  # a date or descriptive text


class RawEspnInjury(RawModel):
    status: str  # "Out", "Injured Reserve", "Day-To-Day", "Suspension"
    date: str | None = None  # ESPN's last-updated timestamp
    short_comment: str | None = None
    athlete: RawEspnAthlete
    details: RawEspnInjuryDetails | None = None


class RawEspnTeamInjuries(RawModel):
    display_name: str  # full team name, e.g. "Anaheim Ducks"
    injuries: list[RawEspnInjury]


class RawEspnInjuries(RawModel):
    injuries: list[RawEspnTeamInjuries]


# --- player/{id}/landing (bio) ---


class RawPlayerLanding(RawModel):
    player_id: int
    # Optional because the endpoint genuinely omits it for a few older players,
    # which is missing data rather than a change in the payload's shape.
    birth_date: str | None = None
