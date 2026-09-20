"""Yahoo position eligibility, read from a copy-paste instead of the API.

Yahoo closed self-serve Fantasy API access on 22 July 2026 and moved it behind
a manual approval programme, so `crosswalk.py` cannot run and the league's
position eligibility cannot be fetched. Eligibility is not a nicety: this
league starts two centres and four defencemen per team, so which pool a player
belongs to sets what the pick is worth, and a player eligible at two positions
is genuinely worth more than one who is not.

The player list pastes cleanly out of the Yahoo web UI, which is the same data
the API would have returned. This parses that paste and resolves each name to
an NHL id using the matcher `crosswalk.py` already uses, so there is one
matching method in the project rather than two that can disagree.

It does not guess. A name that matches nothing, or matches several players that
neither team nor position can separate, is reported as a miss and left out. A
wrong id here would quietly price the wrong player, and nothing downstream
could detect it.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy.orm import Session

from hockey.models import Player
from hockey.yahoo.crosswalk import _match, _name_index, _split_name, normalize_team

logger = logging.getLogger(__name__)

# The paste puts each player's team and eligibility on one line, "COL - C" or
# "TOR - C,LW", with the bare name two lines above it. Everything else in the
# block is stats, which we already have and better.
_LINE = re.compile(r"^([A-Z]{2,3}) - ([A-Z,]+)$")

VALID_POSITIONS = frozenset({"C", "LW", "RW", "D", "G"})

# Yahoo spells a few first names differently from the NHL. These are listed one
# by one and deliberately not inferred: a rule that turns Thomas into Tommy
# would also turn some other Thomas into the wrong player. Each entry here has
# been checked against the players table by surname and team.
NAME_ALIASES = {
    "Thomas Novak": "Tommy Novak",  # NHL 8478438, C, PIT - the only Novak on file
    "Nicholas Robertson": "Nick Robertson",  # NHL 8481582, LW, PIT - only Robertson on PIT
}


@dataclass(frozen=True)
class Pasted:
    name: str
    team: str
    positions: tuple[str, ...]


@dataclass(frozen=True)
class Resolved:
    nhl_id: int
    name: str
    team: str
    positions: tuple[str, ...]
    method: str


def parse(text: str) -> list[Pasted]:
    """Every player in a Yahoo player-list paste, in the order they appear."""
    lines = text.splitlines()
    out = []
    for i, line in enumerate(lines):
        match = _LINE.match(line.strip())
        if not match or i < 2:
            continue
        team, raw = match.groups()
        positions = tuple(p for p in raw.split(",") if p)
        unknown = set(positions) - VALID_POSITIONS
        if unknown:
            # Better to stop than to carry a position the roster config has no
            # slot for, which would later fail as a missing replacement level
            # far away from the line that caused it.
            raise ValueError(f"line {i + 1}: unrecognized position(s) {sorted(unknown)} in {raw!r}")
        # The line directly above is the name with note text appended; the one
        # above that is the bare name. Requiring the first to start with the
        # second is what confirms the block is shaped as expected.
        bare, noted = lines[i - 2].strip(), lines[i - 1].strip()
        if not bare or not noted.startswith(bare):
            raise ValueError(f"line {i + 1}: cannot read a player name above {line.strip()!r}")
        out.append(Pasted(name=bare, team=team, positions=positions))
    return out


def resolve(
    session: Session, pasted: list[Pasted]
) -> tuple[list[Resolved], list[tuple[Pasted, str]]]:
    """(resolved, misses). A miss carries the reason it was not resolved."""
    index = _name_index(session)
    position_of = dict(session.execute(sa.select(Player.nhl_id, Player.position)).all())

    resolved: list[Resolved] = []
    misses: list[tuple[Pasted, str]] = []
    seen: dict[int, str] = {}

    for entry in pasted:
        name = NAME_ALIASES.get(entry.name, entry.name)
        team = normalize_team(entry.team)
        nhl_id, method = _match(index, name, team)

        if nhl_id is None and method == "ambiguous":
            # Two players, same name, same team - Vancouver had two Elias
            # Petterssons, a centre and a defenceman. Yahoo's own eligibility
            # separates them, and that is data rather than a guess, so it is a
            # legitimate third tiebreak after name and team.
            candidates = [c[0] for c in index.get(_split_name(name), []) if c[1] == team]
            fits = [c for c in candidates if position_of.get(c) in entry.positions]
            if len(fits) == 1:
                nhl_id, method = fits[0], "position_tiebreak"

        if nhl_id is None:
            misses.append((entry, method))
            continue
        if nhl_id in seen:
            # Two paste rows claiming one player means the matcher is wrong
            # about at least one of them, so neither is trustworthy.
            misses.append((entry, f"duplicate of {seen[nhl_id]}"))
            continue
        seen[nhl_id] = entry.name
        resolved.append(Resolved(nhl_id, entry.name, entry.team, entry.positions, method))

    return resolved, misses


def write_csv(resolved: list[Resolved], path: Path) -> None:
    """The resolved eligibility, as a config file the export layer reads.

    A file rather than a table because it is hand-sourced and wants to be
    reviewable in a diff, and because the export layer must keep working on a
    machine with no database.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = ["nhl_id,name,team,positions,match_method"]
    rows += [
        f'{r.nhl_id},"{r.name}",{r.team},{"|".join(r.positions)},{r.method}'
        for r in sorted(resolved, key=lambda r: r.name)
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def load_csv(path: Path) -> dict[int, tuple[str, ...]]:
    """nhl_id -> eligible positions. Empty dict when the file is absent, so the
    board still builds on NHL primary positions and says so."""
    if not path.exists():
        return {}
    out: dict[int, tuple[str, ...]] = {}
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        nhl_id, _, rest = line.partition(",")
        positions = rest.rsplit(",", 2)[-2]
        out[int(nhl_id)] = tuple(positions.split("|"))
    return out
