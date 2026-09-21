"""Yahoo's draft board shows names. This turns them into player ids, or fails.

The most dangerous moment in the whole system. Every other error announces
itself - a browser times out, an endpoint 500s, a pick is missed - but a name
matched to the wrong player produces a complete, well-formed, entirely
plausible decision about someone else, and nothing downstream can detect it.
The board will not look wrong. The draft will not look wrong. The team will
simply be worse than it should be, for reasons that never surface.

So this does not guess. A name that matches nothing, or matches more than one
player that the position cannot separate, comes back as a miss with a reason.
The caller's job is to surface it, not to pick the closest thing.

The matcher is the one `hockey.yahoo.crosswalk` already uses and that resolved
450 of 450 players when the eligibility file was built: fold the name, index on
the folded pair, accept a single match, break a tie on position.
"""

import logging
import re
import unicodedata
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)

# Yahoo decorates names on the draft board with status and position text -
# "Connor McDavid EDM - C", "Jack Hughes NJ - C, Q". Strip that before matching
# rather than asking the matcher to cope with it.
_TRAILING = re.compile(r"\s*[-–—]\s*[A-Z,\s]+$")
_TEAM_SUFFIX = re.compile(r"\s+[A-Z]{2,3}$")
_NOISE = re.compile(r"\s*\((?:IR|IR\+|DTD|O|NA|Q|SUSP)\)\s*", re.IGNORECASE)


def fold(name: str) -> str:
    """Casefold and strip accents, so Stutzle matches Stützle."""
    decomposed = unicodedata.normalize("NFKD", name)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^a-z ]", "", stripped.casefold()).strip()


def clean(raw: str) -> str:
    """A bare player name out of whatever the draft board rendered."""
    text = _NOISE.sub(" ", raw).strip()
    text = _TRAILING.sub("", text).strip()
    text = _TEAM_SUFFIX.sub("", text).strip()
    return re.sub(r"\s+", " ", text)


@dataclass(frozen=True)
class Resolution:
    raw: str
    player_id: int | None
    method: str
    candidates: tuple[int, ...] = ()

    @property
    def ok(self) -> bool:
        return self.player_id is not None


class Resolver:
    """Names to player ids, against one board."""

    def __init__(self, players: pd.DataFrame):
        self._by_name: dict[str, list[tuple[int, str]]] = {}
        for row in players.itertuples():
            key = fold(str(row.player))
            self._by_name.setdefault(key, []).append((int(row.player_id), str(row.slot)))
        self._ids = {int(p) for p in players["player_id"]}
        duplicates = {k: v for k, v in self._by_name.items() if len(v) > 1}
        if duplicates:
            # Worth saying out loud at startup rather than at pick time. The
            # board has had two Elias Petterssons in it before.
            logger.warning(
                "%d name(s) on this board are not unique and will need a position to resolve: %s",
                len(duplicates),
                ", ".join(sorted(duplicates)[:5]),
            )

    def resolve(self, raw: str, position: str | None = None) -> Resolution:
        name = clean(raw)
        if not name:
            return Resolution(raw, None, "empty")
        candidates = self._by_name.get(fold(name), [])
        if len(candidates) == 1:
            return Resolution(raw, candidates[0][0], "exact")
        if not candidates:
            return Resolution(raw, None, "unmatched")
        if position:
            fits = [pid for pid, slot in candidates if slot == position]
            if len(fits) == 1:
                return Resolution(raw, fits[0], "position_tiebreak")
        return Resolution(raw, None, "ambiguous", tuple(pid for pid, _ in candidates))

    def known(self, player_id: int) -> bool:
        return int(player_id) in self._ids
