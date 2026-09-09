"""Yahoo player id -> NHL player id.

Yahoo's ids are their own id space with nothing in common with the NHL's, so
the only bridge is the player's name. This reuses the method the app repo
proved on ESPN (99 of 105 matched on a real run): fold the name, index our
players by the folded pair once, accept a single match, break a tie on team,
and record anything still ambiguous as a miss rather than guessing.

Guessing is the failure mode that matters. A wrong id does not look wrong - it
produces a complete, plausible projection for the wrong player, and nothing
downstream can detect it. A recorded miss is visible and fixable.

Yahoo is also the authority on position eligibility for this league, so the
eligible positions come across on the same pass. The NHL roster position is not
used for that.
"""

import logging
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from hockey.ingest.mappers import normalize_name
from hockey.ingest.sync import upsert_rows
from hockey.models import NhlTeam, Player, PlayerCrosswalk
from hockey.yahoo.client import YahooFantasyClient

logger = logging.getLogger(__name__)

PAGE_SIZE = 25  # Yahoo's hard cap per players request.

# Yahoo abbreviates a handful of teams differently from the NHL. Anything not
# listed matches on the abbreviation as-is.
YAHOO_TEAM_FIXUPS = {
    "LA": "LAK",
    "SJ": "SJS",
    "TB": "TBL",
    "NJ": "NJD",
    "MON": "MTL",
    "WAS": "WSH",
    "CLS": "CBJ",
    "ANH": "ANA",
    "VGK": "VGK",
    "UTA": "UTA",
}


def normalize_team(abbrev: str | None) -> str | None:
    if not abbrev:
        return None
    cleaned = abbrev.strip().upper()
    return YAHOO_TEAM_FIXUPS.get(cleaned, cleaned)


def _name_index(session: Session) -> dict[tuple[str, str], list[tuple[int, str | None]]]:
    """(folded first, folded last) -> [(nhl_id, team_abbrev)], built once.

    Built for every player, not just current ones, because Yahoo's pool
    includes players who changed teams and prospects who appear in our data
    only as boxscore stubs.
    """
    index: dict[tuple[str, str], list[tuple[int, str | None]]] = {}
    for pid, first, last, team in session.execute(
        select(Player.nhl_id, Player.first_name, Player.last_name, Player.team_abbrev)
    ):
        index.setdefault((normalize_name(first), normalize_name(last)), []).append((pid, team))
    return index


def _split_name(full: str) -> tuple[str, str]:
    """Yahoo gives a full name; our index is keyed on first and last. Splitting
    on the first space keeps multi-word surnames intact, which is what the NHL
    side stores too."""
    first, _, last = full.strip().partition(" ")
    return normalize_name(first), normalize_name(last or full)


def _match(
    index: dict[tuple[str, str], list[tuple[int, str | None]]],
    full_name: str,
    team: str | None,
) -> tuple[int | None, str]:
    """(nhl_id, method). nhl_id is None for a deliberate miss."""
    candidates = index.get(_split_name(full_name), [])
    if len(candidates) == 1:
        return candidates[0][0], "exact"
    if len(candidates) > 1 and team is not None:
        on_team = [c[0] for c in candidates if c[1] == team]
        if len(on_team) == 1:
            return on_team[0], "team_tiebreak"
        return None, "ambiguous"
    if len(candidates) > 1:
        return None, "ambiguous"
    return None, "unmatched"


def sync_crosswalk(
    session: Session, client: YahooFantasyClient, league_key: str, max_players: int = 1200
) -> dict[str, int]:
    """Walk the league's player pool and map every player to an NHL id.

    Returns counts by match method. The caller is expected to look at the
    misses before trusting the export - an unmatched player is simply absent
    from every projection.
    """
    index = _name_index(session)
    known_teams = set(session.scalars(select(NhlTeam.abbrev)))
    now = datetime.now(UTC)

    rows: list[dict] = []
    counts = {"exact": 0, "team_tiebreak": 0, "ambiguous": 0, "unmatched": 0}
    start = 0
    while start < max_players:
        page = client.league_players(league_key, start, PAGE_SIZE)
        if not page:
            break
        for entry in page:
            player = _merge_player_fragments(entry)
            if player is None:
                continue
            full_name = player["name"]
            team = normalize_team(player.get("editorial_team_abbr"))
            if team is not None and team not in known_teams:
                # Keep the raw value for review but do not tiebreak on a team
                # we do not recognize.
                team_for_match = None
            else:
                team_for_match = team
            nhl_id, method = _match(index, full_name, team_for_match)
            counts[method] += 1
            if nhl_id is None:
                logger.info("crosswalk %s: %s (%s)", method, full_name, team or "no team")
            rows.append(
                {
                    "yahoo_player_id": int(player["player_id"]),
                    "nhl_id": nhl_id,
                    "yahoo_name": full_name,
                    "yahoo_team": team,
                    "yahoo_positions": ",".join(player["positions"]),
                    "match_method": method,
                    "synced_at": now,
                }
            )
        start += PAGE_SIZE

    upsert_rows(session, PlayerCrosswalk, rows)
    session.commit()
    matched = counts["exact"] + counts["team_tiebreak"]
    logger.info(
        "crosswalk: %d players, %d matched (%d exact, %d by team), %d ambiguous, %d unmatched",
        len(rows),
        matched,
        counts["exact"],
        counts["team_tiebreak"],
        counts["ambiguous"],
        counts["unmatched"],
    )
    return counts


def _eligible_positions(node) -> list[str]:
    """Yahoo's eligible_positions, which arrives either as a list of
    {'position': 'C'} wrappers or as bare strings depending on the endpoint.

    Tolerant of both rather than strict, because eligibility drives which slot
    a player can fill and losing it would quietly shrink the draft pool. An
    empty result is reported by the caller as 'UNK', which is visible.
    """
    if node is None:
        return []
    entries = node if isinstance(node, list) else [node]
    positions = []
    for entry in entries:
        if isinstance(entry, dict):
            value = entry.get("position")
            if value:
                positions.append(str(value))
        elif isinstance(entry, str) and entry:
            positions.append(entry)
    return positions


def _merge_player_fragments(entry) -> dict | None:
    """Pull the fields we need out of Yahoo's split player record.

    A Yahoo player arrives as a list of small dicts, one of which holds the
    name as a nested dict and another the eligible positions as a wrapped
    collection. Anything without a player_id is padding.
    """
    fragments = entry if isinstance(entry, list) else [entry]
    merged: dict = {}
    for fragment in fragments:
        if isinstance(fragment, dict):
            merged |= fragment
    if "player_id" not in merged:
        return None
    name = merged.get("name")
    full = name.get("full") if isinstance(name, dict) else name
    if not full:
        return None
    positions = _eligible_positions(merged.get("eligible_positions"))
    return {
        "player_id": merged["player_id"],
        "name": str(full),
        "editorial_team_abbr": merged.get("editorial_team_abbr"),
        "positions": positions or ["UNK"],
    }


def unmatched_report(session: Session) -> list[tuple[str, str | None, str]]:
    """Every player the crosswalk could not place, for review before the draft."""
    return [
        (r.yahoo_name, r.yahoo_team, r.match_method)
        for r in session.execute(
            select(
                PlayerCrosswalk.yahoo_name,
                PlayerCrosswalk.yahoo_team,
                PlayerCrosswalk.match_method,
            )
            .where(PlayerCrosswalk.nhl_id.is_(None))
            .order_by(PlayerCrosswalk.yahoo_name)
        )
    ]
