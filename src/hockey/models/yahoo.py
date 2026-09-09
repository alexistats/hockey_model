"""Yahoo league configuration and the Yahoo <-> NHL player crosswalk.

The model is driven by whatever these tables say, never by hardcoded
categories. Source: the official Yahoo Fantasy Sports API (OAuth2), read by
hockey/yahoo/. A YAML snapshot of each sync is written alongside so a model run
is reproducible without a network call.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, Boolean, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from hockey.models.base import Base


class LeagueSettings(Base):
    __tablename__ = "league_settings"
    __table_args__ = {
        "comment": "One row per Yahoo league season. Source: GET "
        "/fantasy/v2/league/{league_key}/settings (fantasysports.yahooapis.com)."
    }

    league_key: Mapped[str] = mapped_column(
        String(32),
        primary_key=True,
        comment="Yahoo league key, '{game_key}.l.{league_id}' (e.g. '465.l.12345'). PK. "
        "The game key changes every season, so this is season-unique.",
    )
    league_id: Mapped[str] = mapped_column(
        String(16), comment="Numeric league id from the league URL, without the game key."
    )
    season: Mapped[int] = mapped_column(
        Integer, comment="NHL season start year as Yahoo reports it (e.g. 2026 for 2026-27)."
    )
    name: Mapped[str] = mapped_column(String(128), comment="League display name.")
    num_teams: Mapped[int] = mapped_column(Integer, comment="Number of fantasy teams.")
    scoring_type: Mapped[str] = mapped_column(
        String(16),
        comment="Yahoo scoring type: 'headpoint' = head-to-head points, 'head' = "
        "head-to-head categories, 'point' = season points, 'roto'. This project "
        "assumes a points-based type; a category league would need different "
        "value logic downstream.",
    )
    raw: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        comment="Full unmodified settings payload from Yahoo, so a later question about "
        "a setting we did not normalize can be answered without a re-sync.",
    )
    synced_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), comment="When this row was last read from Yahoo, UTC."
    )


class LeagueStatCategory(Base):
    """One scoring category and its point modifier. This table, not any constant
    in the code, is what the scoring functions read."""

    __tablename__ = "league_stat_categories"
    __table_args__ = {
        "comment": "Scoring categories and point modifiers for a league, from Yahoo's "
        "settings stat_categories + stat_modifiers. Drives hockey/scoring/."
    }

    league_key: Mapped[str] = mapped_column(
        ForeignKey("league_settings.league_key"), primary_key=True, comment="FK league_settings."
    )
    stat_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, comment="Yahoo's numeric stat id (e.g. 1 = Goals). Part of PK."
    )
    name: Mapped[str] = mapped_column(
        String(64), comment="Yahoo's full stat name, e.g. 'Power Play Points'."
    )
    display_name: Mapped[str] = mapped_column(
        String(16), comment="Yahoo's abbreviation, e.g. 'PPP', 'SHP', 'HIT', 'BLK'."
    )
    position_type: Mapped[str] = mapped_column(
        String(1), comment="'P' = skater stat, 'G' = goalie stat."
    )
    modifier: Mapped[float] = mapped_column(
        Float,
        comment="Fantasy points per unit of this stat. Can be negative (goals against "
        "is -1.5 in this league).",
    )
    source_column: Mapped[str | None] = mapped_column(
        String(32),
        comment="Warehouse column this category reads, as 'table.column' "
        "(e.g. 'skater_game_logs.hits'). NULL means the warehouse cannot serve "
        "this category; hockey/scoring/ raises at config-load time rather than "
        "silently scoring it as zero.",
    )


class LeagueRosterPosition(Base):
    __tablename__ = "league_roster_positions"
    __table_args__ = {
        "comment": "Roster slots and counts for a league (C x2, D x4, BN x5, ...), from "
        "Yahoo settings roster_positions. Used by the draft export to size the pool."
    }

    league_key: Mapped[str] = mapped_column(
        ForeignKey("league_settings.league_key"), primary_key=True, comment="FK league_settings."
    )
    position: Mapped[str] = mapped_column(
        String(8),
        primary_key=True,
        comment="Slot code: C, LW, RW, D, G, BN, IR+, and flex codes like Util. Part of PK.",
    )
    count: Mapped[int] = mapped_column(Integer, comment="How many of this slot each roster has.")
    is_starting: Mapped[bool] = mapped_column(
        Boolean,
        comment="True for slots that score (C/LW/RW/D/G); False for BN and IR+.",
    )


class PlayerCrosswalk(Base):
    """Yahoo player id -> NHL player id. Yahoo ids are their own id space and
    share nothing with the NHL's, so the mapping is built from normalized names
    with a team tiebreak - the same method hockey/ingest/mappers.normalize_name
    uses for ESPN. An ambiguous or unmatched player is stored with nhl_id NULL
    and a match_method saying why, never guessed into a wrong id.
    """

    __tablename__ = "player_crosswalk"
    __table_args__ = {
        "comment": "Yahoo <-> NHL player id mapping, built by normalized-name matching. "
        "Rows with nhl_id NULL are deliberate misses kept for review; they are never "
        "guessed. See hockey/yahoo/crosswalk.py."
    }

    yahoo_player_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, comment="Yahoo's numeric player id. PK."
    )
    nhl_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("players.nhl_id"),
        comment="FK players.nhl_id. NULL when the name matched nothing or matched "
        "several players that the team tiebreak could not separate.",
    )
    yahoo_name: Mapped[str] = mapped_column(
        String(128), comment="Full name exactly as Yahoo spells it, for review of misses."
    )
    yahoo_team: Mapped[str | None] = mapped_column(
        String(8), comment="Yahoo's team abbreviation, used as the tiebreak. NULL if absent."
    )
    yahoo_positions: Mapped[str] = mapped_column(
        String(32),
        comment="Comma-separated Yahoo eligible positions (e.g. 'C,LW'). Yahoo is the "
        "authority on eligibility for this league, not the NHL roster position.",
    )
    match_method: Mapped[str] = mapped_column(
        String(24),
        comment="How the row was decided: 'exact' (one name match), 'team_tiebreak' "
        "(several matches, one on the right team), 'ambiguous' (several, none "
        "separable) or 'unmatched' (no name match).",
    )
    synced_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), comment="When this row was last refreshed from Yahoo, UTC."
    )
