"""Read the league's real scoring config from Yahoo into the warehouse.

Yahoo lists every category it knows about and gives the unused ones a modifier
of zero, so the stored rows are the full catalogue and `hockey.scoring` decides
what actually counts. Storing the zeros too means a mid-season settings change
shows up as a diff rather than as a mystery.

A YAML snapshot is written next to every sync so a model run can be reproduced
later without a network call, and so a settings change between two runs is
visible in a diff.
"""

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import delete
from sqlalchemy.orm import Session

from hockey.ingest.sync import upsert_rows
from hockey.models import LeagueRosterPosition, LeagueSettings, LeagueStatCategory
from hockey.scoring import build_scoring, lookup
from hockey.yahoo.client import YahooFantasyClient, merge_fragments, unwrap

logger = logging.getLogger(__name__)

# Roster slots that actually score. Everything else (BN, IR, IR+, NA) holds a
# player without producing points.
STARTING_SLOTS = {"C", "LW", "RW", "D", "G", "F", "Util", "W"}

SNAPSHOT_DIR = Path("artifacts/league_config")


def build_league_key(game_key: str, league_id: str) -> str:
    return f"{game_key}.l.{league_id}"


def _stat_rows(league_key: str, league: dict) -> list[dict]:
    """One row per category, pairing Yahoo's stat_categories with the
    stat_modifiers that give each one its point value."""
    settings = merge_fragments(league["settings"])
    categories = unwrap(merge_fragments(settings["stat_categories"])["stats"], "stat")
    modifier_entries = unwrap(merge_fragments(settings["stat_modifiers"])["stats"], "stat")
    modifiers = {int(m["stat_id"]): float(m["value"]) for m in modifier_entries}

    rows = []
    for stat in categories:
        stat_id = int(stat["stat_id"])
        display = str(stat["display_name"])
        # A category Yahoo lists but this project has never seen is only a
        # problem if the league scores it; lookup raises for those in
        # build_scoring. Here it just means source_column stays NULL.
        try:
            source = lookup(display).source
        except KeyError:
            source = None
        rows.append(
            {
                "league_key": league_key,
                "stat_id": stat_id,
                "name": str(stat["name"]),
                "display_name": display,
                "position_type": str(stat.get("position_type") or "P"),
                "modifier": modifiers.get(stat_id, 0.0),
                "source_column": source,
            }
        )
    return rows


def _roster_rows(league_key: str, league: dict) -> list[dict]:
    settings = merge_fragments(league["settings"])
    positions = unwrap(settings["roster_positions"], "roster_position")
    return [
        {
            "league_key": league_key,
            "position": str(p["position"]),
            "count": int(p["count"]),
            "is_starting": str(p["position"]) in STARTING_SLOTS,
        }
        for p in positions
    ]


def sync_league_settings(
    session: Session, client: YahooFantasyClient, league_id: str, game_key: str | None = None
) -> str:
    """Read the league's settings from Yahoo, store them, and verify that the
    scoring config this project can actually compute.

    Returns the league key. Raises before writing anything if the league scores
    a category the warehouse cannot serve, so a broken config is caught here
    rather than showing up as a quietly wrong projection.
    """
    game_key = game_key or client.current_game_key()
    league_key = build_league_key(game_key, league_id)
    league = client.league_settings(league_key)

    stat_rows = _stat_rows(league_key, league)
    roster_rows = _roster_rows(league_key, league)

    # Fail before writing: build_scoring raises on an unknown category, one the
    # warehouse cannot serve, or a rate stat that cannot be scored per game.
    scoring = build_scoring(league_key, [(r["display_name"], r["modifier"]) for r in stat_rows])

    now = datetime.now(UTC)
    upsert_rows(
        session,
        LeagueSettings,
        [
            {
                "league_key": league_key,
                "league_id": league_id,
                "season": int(league["season"]),
                "name": str(league["name"]),
                "num_teams": int(league["num_teams"]),
                "scoring_type": str(league["scoring_type"]),
                "raw": league,
                "synced_at": now,
            }
        ],
    )
    # Categories and roster slots are replaced wholesale: a category dropped
    # from the league settings must disappear here too, or the model keeps
    # scoring it.
    session.execute(delete(LeagueStatCategory).where(LeagueStatCategory.league_key == league_key))
    session.execute(
        delete(LeagueRosterPosition).where(LeagueRosterPosition.league_key == league_key)
    )
    upsert_rows(session, LeagueStatCategory, stat_rows)
    upsert_rows(session, LeagueRosterPosition, roster_rows)
    session.commit()

    write_snapshot(league_key, league, stat_rows, roster_rows)

    scored = [r for r in stat_rows if r["modifier"] != 0]
    logger.info(
        "league %s (%s): %d scored categories (%d skater, %d goalie), %d roster slots",
        league_key,
        league["name"],
        len(scored),
        len(scoring.skater_rules),
        len(scoring.goalie_rules),
        sum(r["count"] for r in roster_rows),
    )
    return league_key


def write_snapshot(
    league_key: str, league: dict[str, Any], stat_rows: list[dict], roster_rows: list[dict]
) -> Path:
    """Write a human-readable, diffable record of the config used for a run."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / f"{league_key}.yaml"
    payload = {
        "league_key": league_key,
        "name": league.get("name"),
        "season": league.get("season"),
        "num_teams": league.get("num_teams"),
        "scoring_type": league.get("scoring_type"),
        "synced_at": datetime.now(UTC).isoformat(),
        "scoring": [
            {
                "display_name": r["display_name"],
                "name": r["name"],
                "position_type": r["position_type"],
                "modifier": r["modifier"],
                "source_column": r["source_column"],
            }
            for r in stat_rows
            if r["modifier"] != 0
        ],
        "roster_positions": [
            {"position": r["position"], "count": r["count"], "starting": r["is_starting"]}
            for r in roster_rows
        ],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    logger.info("wrote config snapshot %s", path)
    return path


DEFAULT_CONFIG = Path("config/league_2270.yaml")


def load_scoring_from_yaml(path: Path | str = DEFAULT_CONFIG):
    """Scoring rules from a checked-in config file rather than from Yahoo.

    The live sync is authoritative and should be preferred; this exists so the
    model is never blocked on Yahoo being reachable, and so a run is
    reproducible offline. It goes through the same build_scoring(), so a
    category the warehouse cannot serve fails here exactly as it would there -
    the fallback cannot be a quieter path than the real one.
    """
    path = Path(path)
    if not path.exists():
        raise LookupError(f"no league config at {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return build_scoring(
        payload["league_key"],
        [(row["display_name"], float(row["modifier"])) for row in payload["scoring"]],
    )


def load_roster_from_yaml(path: Path | str = DEFAULT_CONFIG) -> list[dict]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return payload["roster_positions"]


def load_scoring(session: Session, league_key: str):
    """Rebuild the scoring rules from what is stored, with no network call.

    This is what the model and the export read. It goes through the same
    `build_scoring` as the sync, so a config that passed at sync time cannot
    fail differently at model time.
    """
    rows = session.execute(
        LeagueStatCategory.__table__.select().where(
            LeagueStatCategory.__table__.c.league_key == league_key
        )
    ).all()
    if not rows:
        raise LookupError(
            f"no stored config for league {league_key}. Run "
            f"`python -m hockey.yahoo settings` first."
        )
    return build_scoring(league_key, [(r.display_name, r.modifier) for r in rows])
