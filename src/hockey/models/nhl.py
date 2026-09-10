from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from hockey.models.base import Base

# nhl_games.game_type value for regular-season games (2 = regular season,
# 3 = playoffs; preseason 1 is intentionally not synced). The single home for
# the constant every schedule/scoring/projection filter uses.
REGULAR_SEASON = 2


class NhlTeam(Base):
    """Keyed by abbrev: the team-list source (standings) has no numeric id, and
    roster/schedule endpoints are abbrev-addressed. nhl_team_id is backfilled
    from schedule data. Known trade-off: abbrevs change on relocation (ARI->UTA).
    """

    __tablename__ = "nhl_teams"
    __table_args__ = {
        "comment": "NHL teams. Source: GET /v1/standings/now (api-web.nhle.com); "
        "numeric id backfilled from /v1/club-schedule-season."
    }

    abbrev: Mapped[str] = mapped_column(
        String(3),
        primary_key=True,
        comment="3-letter NHL team abbreviation (e.g. TOR). PK; changes on franchise "
        "relocation (e.g. ARI->UTA), which would need a data migration.",
    )
    name: Mapped[str] = mapped_column(
        String(64), comment="Full team name, English locale (e.g. 'Toronto Maple Leafs')."
    )
    conference: Mapped[str | None] = mapped_column(
        String(16),
        comment="Conference name: 'Eastern' or 'Western'. NULL for a season played "
        "without conferences - 2020-21 used four temporary divisions and had none. "
        "Reflects the most recently synced season, not any particular one.",
    )
    division: Mapped[str] = mapped_column(
        String(32),
        comment="Division name: Atlantic, Metropolitan, Central, or Pacific - or the "
        "temporary 2020-21 names (Scotia North, MassMutual East, Discover Central, "
        "Honda West), which is why this is wider than 16 characters. Reflects the "
        "most recently synced season.",
    )
    nhl_team_id: Mapped[int | None] = mapped_column(
        Integer,
        comment="NHL numeric team id (e.g. TOR=10). NULL until schedule sync backfills it; "
        "the standings endpoint does not provide it.",
    )


class Player(Base):
    __tablename__ = "players"
    __table_args__ = {
        "comment": "NHL players. Source: GET /v1/roster/{team}/{season}; minimal stub rows "
        "are created from boxscores for players absent from season rosters "
        "(call-ups, mid-season trades)."
    }

    nhl_id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, comment="NHL player id (e.g. McDavid=8478402). PK."
    )
    first_name: Mapped[str] = mapped_column(
        String(64),
        comment="First name. For boxscore stub rows this is an abbreviated initial "
        "(e.g. 'C.') until a roster sync fills the full name.",
    )
    last_name: Mapped[str] = mapped_column(String(64), comment="Last name, English locale.")
    position: Mapped[str] = mapped_column(
        String(2),
        comment="Position: C, LW, RW, D, or G. NHL API codes L/R are mapped to LW/RW "
        "at ingest (hockey/ingest/mappers.py).",
    )
    team_abbrev: Mapped[str | None] = mapped_column(
        ForeignKey("nhl_teams.abbrev"),
        comment="FK nhl_teams.abbrev. Team whose season roster listed the player; "
        "NULL for boxscore stub rows.",
    )
    status: Mapped[str] = mapped_column(
        String(16),
        default="active",
        comment="'active' = on a synced season roster; 'inactive' = seen only in boxscores. "
        "NOT injury status; that comes from ESPN into player_injuries.",
    )
    shoots_catches: Mapped[str | None] = mapped_column(
        String(1), comment="Shoots (skaters) or catches (goalies): L or R. NULL for stub rows."
    )
    birth_date: Mapped[date | None] = mapped_column(
        Date,
        comment="Date of birth, from GET /v1/player/{id}/landing. Feeds the aging curve: "
        "a player's age at a given season is what lets the model separate a decline "
        "that is age from a decline that is noise. NULL until the bios sync runs, and "
        "for anyone the endpoint has no record of.",
    )


class NhlGame(Base):
    __tablename__ = "nhl_games"
    __table_args__ = {
        "comment": "NHL games, regular season and playoffs only. "
        "Source: GET /v1/club-schedule-season/{team}/{season}."
    }

    nhl_game_id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        comment="NHL game id (e.g. 2025020001: season 2025, type 02, game 0001). PK.",
    )
    date: Mapped[date] = mapped_column(
        Date, index=True, comment="Game calendar date (Eastern Time) from the API gameDate."
    )
    start_time_utc: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        comment="Scheduled puck drop in UTC (API startTimeUTC). Used for lineup locking. "
        "NULL only for rows synced before this column existed; backfilled by "
        "re-running the schedule sync.",
    )
    season: Mapped[int] = mapped_column(
        Integer, index=True, comment="NHL season as start+end year int (e.g. 20252026)."
    )
    game_type: Mapped[int] = mapped_column(
        Integer,
        comment="NHL game type: 2 = regular season, 3 = playoffs. "
        "Preseason (1) is intentionally not synced.",
    )
    home_team_abbrev: Mapped[str] = mapped_column(
        ForeignKey("nhl_teams.abbrev"), comment="FK nhl_teams.abbrev, home side."
    )
    away_team_abbrev: Mapped[str] = mapped_column(
        ForeignKey("nhl_teams.abbrev"), comment="FK nhl_teams.abbrev, away side."
    )
    game_state: Mapped[str] = mapped_column(
        String(16),
        comment="NHL game state. OFF/FINAL = completed (only these get game logs); "
        "FUT = scheduled. Other live states may appear transiently.",
    )
    home_score: Mapped[int | None] = mapped_column(
        Integer, comment="Final home goals. NULL until the game is played."
    )
    away_score: Mapped[int | None] = mapped_column(
        Integer, comment="Final away goals. NULL until the game is played."
    )


class SkaterGameLog(Base):
    __tablename__ = "skater_game_logs"
    __table_args__ = (
        # The PK is (player_id, game_id), so game_id alone is unindexed. Both
        # the resumability check and every join from nhl_games lead with it.
        Index("ix_skater_game_logs_game", "game_id"),
        {
            "comment": "Per-skater per-game stats. Primary source: "
            "GET /v1/gamecenter/{gameId}/boxscore (one row per dressed skater); "
            "ppp and shp filled afterwards from GET /v1/player/{id}/game-log (the "
            "boxscore carries neither). See docs/architecture.md 'Game logs merge "
            "two NHL API sources'."
        },
    )

    player_id: Mapped[int] = mapped_column(
        ForeignKey("players.nhl_id"), primary_key=True, comment="FK players.nhl_id. Part of PK."
    )
    game_id: Mapped[int] = mapped_column(
        ForeignKey("nhl_games.nhl_game_id"),
        primary_key=True,
        comment="FK nhl_games.nhl_game_id. Part of PK.",
    )
    team_abbrev: Mapped[str | None] = mapped_column(
        ForeignKey("nhl_teams.abbrev"),
        comment="FK nhl_teams.abbrev. The team this player dressed for IN THIS GAME, "
        "taken from the side of the boxscore they were listed under. Distinct from "
        "players.team_abbrev, which is a single current value and is therefore wrong "
        "for past seasons and for anyone who was traded. The model needs this to know "
        "the opponent and whether the game was played at home. NULL only for rows "
        "synced before this column existed.",
    )
    position: Mapped[str | None] = mapped_column(
        String(2),
        comment="Position actually skated this game (C/LW/RW/D), from the boxscore — "
        "game-specific, not the roster's static primary_position (players.position). "
        "Drives multi-position eligibility. NULL for rows synced before this column "
        "existed; backfilled by re-running the game-log sync. NHL L/R mapped to LW/RW.",
    )
    goals: Mapped[int] = mapped_column(Integer, comment="Goals scored.")
    assists: Mapped[int] = mapped_column(Integer, comment="Assists (primary + secondary).")
    points: Mapped[int] = mapped_column(Integer, comment="Points (goals + assists).")
    plus_minus: Mapped[int] = mapped_column(
        Integer, comment="Plus/minus: even-strength and shorthanded goal differential while on ice."
    )
    pim: Mapped[int] = mapped_column(Integer, comment="Penalty minutes.")
    ppp: Mapped[int | None] = mapped_column(
        Integer,
        comment="Power-play points (PP goals + PP assists). NULL until the player-stats "
        "sync fills it from the per-player game-log endpoint; stays NULL for "
        "dressed-but-did-not-play rows (0:00 TOI), which that endpoint omits.",
    )
    shp: Mapped[int | None] = mapped_column(
        Integer,
        comment="Shorthanded points (SH goals + SH assists). Same source and NULL "
        "semantics as ppp: the boxscore does not carry it, so the player-stats "
        "sync fills it from the per-player game-log endpoint. Scored by the "
        "Yahoo league, so it is not optional here.",
    )
    sog: Mapped[int] = mapped_column(Integer, comment="Shots on goal.")
    hits: Mapped[int] = mapped_column(Integer, comment="Hits delivered.")
    blocks: Mapped[int] = mapped_column(Integer, comment="Shots blocked.")
    toi_seconds: Mapped[int] = mapped_column(
        Integer, comment="Time on ice in seconds (API 'MM:SS' parsed at ingest)."
    )


class GoalieGameLog(Base):
    __tablename__ = "goalie_game_logs"
    __table_args__ = (
        Index("ix_goalie_game_logs_game", "game_id"),
        {
            "comment": "Per-goalie per-game stats; only goalies with >0 seconds played get a row. "
            "Primary source: GET /v1/gamecenter/{gameId}/boxscore; save_pctg and "
            "shutouts filled afterwards from GET /v1/player/{id}/game-log."
        },
    )

    player_id: Mapped[int] = mapped_column(
        ForeignKey("players.nhl_id"), primary_key=True, comment="FK players.nhl_id. Part of PK."
    )
    game_id: Mapped[int] = mapped_column(
        ForeignKey("nhl_games.nhl_game_id"),
        primary_key=True,
        comment="FK nhl_games.nhl_game_id. Part of PK.",
    )
    team_abbrev: Mapped[str | None] = mapped_column(
        ForeignKey("nhl_teams.abbrev"),
        comment="FK nhl_teams.abbrev. The team this player dressed for IN THIS GAME, "
        "taken from the side of the boxscore they were listed under. Distinct from "
        "players.team_abbrev, which is a single current value and is therefore wrong "
        "for past seasons and for anyone who was traded. The model needs this to know "
        "the opponent and whether the game was played at home. NULL only for rows "
        "synced before this column existed.",
    )
    decision: Mapped[str | None] = mapped_column(
        String(1),
        comment="Game decision: W = win, L = regulation loss, O = overtime/shootout loss. "
        "NULL when the goalie played but was not charged with a decision.",
    )
    wins: Mapped[int] = mapped_column(
        Integer, comment="0 or 1; derived at ingest from decision = 'W'."
    )
    goals_against: Mapped[int] = mapped_column(Integer, comment="Goals allowed.")
    saves: Mapped[int] = mapped_column(Integer, comment="Saves made.")
    shots_against: Mapped[int] = mapped_column(Integer, comment="Shots faced (saves + goals).")
    save_pctg: Mapped[float | None] = mapped_column(
        Float,
        comment="Save percentage as a 0-1 fraction (e.g. 0.925). NULL until the player-stats "
        "sync fills it from the per-player game-log endpoint; the API may also omit it "
        "when no shots were faced.",
    )
    shutouts: Mapped[int | None] = mapped_column(
        Integer,
        comment="0 or 1, per NHL shutout rules (sole goalie, zero goals against, win). "
        "NULL until the player-stats sync fills it.",
    )
    toi_seconds: Mapped[int] = mapped_column(
        Integer, comment="Time on ice in seconds (API 'MM:SS' parsed at ingest)."
    )
    started: Mapped[bool] = mapped_column(
        Boolean, default=False, comment="True if this goalie started the game."
    )


class NhlPlay(Base):
    """Play-by-play events, one row per event. Source: GET
    /v1/gamecenter/{gameId}/play-by-play. Primarily an analytics dataset
    (shot maps, xG, zone play) — the fantasy app does not read it. Hybrid
    shape: the fields common to every event are promoted to columns for fast
    filtering; the heterogeneous per-event-type fields stay in the JSONB
    `details` (the raw API `details` object, camelCase keys preserved), since
    event types carry different fields and the NHL adds new ones.
    """

    __tablename__ = "nhl_plays"
    __table_args__ = (
        Index("ix_nhl_plays_type", "type_desc_key"),
        # GIN supports containment queries into details, which is how faceoff
        # wins would be derived: details @> '{"winningPlayerId": 8476392}'.
        Index("ix_nhl_plays_details", "details", postgresql_using="gin"),
        {
            "comment": "NHL play-by-play events (analytics dataset). Source: GET "
            "/v1/gamecenter/{gameId}/play-by-play. Common fields promoted to columns, "
            "per-event fields in JSONB details."
        },
    )

    game_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("nhl_games.nhl_game_id"),
        primary_key=True,
        comment="FK nhl_games.nhl_game_id. Part of PK.",
    )
    event_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, comment="NHL eventId, unique within a game. Part of PK."
    )
    sort_order: Mapped[int] = mapped_column(
        Integer, comment="API sortOrder; the canonical chronological order of events in the game."
    )
    period_number: Mapped[int] = mapped_column(
        Integer, comment="Period number (1-3 regulation, 4+ OT, shootout per periodType)."
    )
    period_type: Mapped[str] = mapped_column(
        String(8), comment="Period type: REG, OT, or SO (periodDescriptor.periodType)."
    )
    time_in_period: Mapped[str] = mapped_column(
        String(5), comment="Elapsed time in the period, 'MM:SS' (as the API reports it)."
    )
    type_code: Mapped[int] = mapped_column(Integer, comment="NHL numeric event type code.")
    type_desc_key: Mapped[str] = mapped_column(
        String(32),
        comment="Event type key: faceoff, shot-on-goal, missed-shot, blocked-shot, goal, "
        "hit, giveaway, takeaway, penalty, stoppage, period-start/end, game-end, etc.",
    )
    situation_code: Mapped[str | None] = mapped_column(
        String(4),
        comment="Strength/situation code, e.g. '1551' = 5v5 (away goalie, away skaters, "
        "home skaters, home goalie). NULL when the API omits it.",
    )
    home_team_defending_side: Mapped[str | None] = mapped_column(
        String(5),
        comment="'left' or 'right' — the side the home team defends this event, for "
        "normalizing x/y coordinates. NULL on events without it.",
    )
    x_coord: Mapped[int | None] = mapped_column(
        Integer,
        comment="Event x coordinate on the rink (approx -100..100; details.xCoord). NULL for "
        "events without a location (stoppages, period boundaries).",
    )
    y_coord: Mapped[int | None] = mapped_column(
        Integer, comment="Event y coordinate (approx -42..42; details.yCoord). NULL when absent."
    )
    zone_code: Mapped[str | None] = mapped_column(
        String(1),
        comment="Zone of the event relative to the event-owner team: O(ffensive), "
        "D(efensive), N(eutral). NULL when absent.",
    )
    event_owner_team_id: Mapped[int | None] = mapped_column(
        Integer,
        comment="NHL numeric team id that 'owns' the event (shooter's/hitter's team, etc.; "
        "details.eventOwnerTeamId). NULL when absent.",
    )
    details: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        comment="Raw API `details` object for the event (camelCase keys preserved): the "
        "per-event-type fields not promoted above — role player ids "
        "(shootingPlayerId, scoringPlayerId, assist{1,2}PlayerId, winning/losingPlayerId, "
        "hitting/hitteePlayerId, committedBy/drawnByPlayerId), goalieInNetId, shotType, "
        "running scores/SOG, clip urls. Empty object for events with no details.",
    )


class PlayerInjury(Base):
    """Current injury/absence for a player, from ESPN's unofficial injuries feed
    (no official NHL injury source exists — see docs/architecture.md). One row
    per currently-injured player; each sync refreshes the whole set, so a
    recovered player's row is deleted. Players are matched from ESPN by
    normalized name (+ team tiebreak), since ESPN carries no NHL player id.
    """

    __tablename__ = "player_injuries"
    __table_args__ = {
        "comment": "Current player injuries from ESPN's unofficial feed (no official NHL "
        "source). One row per injured player; refreshed wholesale each sync (recovered "
        "players are removed). Matched to players by normalized name + team."
    }

    player_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("players.nhl_id"),
        primary_key=True,
        comment="FK players.nhl_id. PK — a player has at most one current injury.",
    )
    status: Mapped[str] = mapped_column(
        String(32),
        comment="ESPN status: 'Out', 'Injured Reserve', 'Day-To-Day', 'Suspension', etc.",
    )
    injury_type: Mapped[str | None] = mapped_column(
        String(64),
        comment="Body part / injury type (ESPN details.type, e.g. 'Knee'). NULL if absent.",
    )
    detail: Mapped[str | None] = mapped_column(
        String(128), comment="Extra detail (ESPN details.detail, e.g. 'Surgery'). NULL if absent."
    )
    return_date: Mapped[str | None] = mapped_column(
        String(32),
        comment="Estimated return (ESPN details.returnDate), stored as given (a date or "
        "descriptive text). NULL if absent.",
    )
    comment: Mapped[str | None] = mapped_column(
        String(255), comment="Short human note (ESPN shortComment). NULL if absent."
    )
    espn_updated: Mapped[str | None] = mapped_column(
        String(32),
        comment="ESPN's last-updated timestamp for the injury (raw string). NULL if absent.",
    )
    synced_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        comment="When this row was last refreshed from ESPN, UTC (set each sync).",
    )
