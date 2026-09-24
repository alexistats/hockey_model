"""What a player would actually start, given the roster I already have.

A late pick is not worth his team's games, it is worth the games he would be in
my lineup for. Yahoo sets a daily lineup, so a fourth centre whose team plays
the same nights as my first three adds almost nothing: those nights my two
centre slots are already spoken for by better players, and he sits. The same
player on a team that plays Tuesdays and Thursdays, when my centres are idle,
starts most nights.

So this counts starts, not games, and it counts them against the roster as it
stands - which is why it lives here rather than in the bot. Eligibility overlaps
and slots have capacity, so the only honest way to ask "would he be in the
lineup on the 14th" is to set that night's lineup the way a manager does: as
many slots filled as possible, and the best players in them (`lineup`).

That is not quite the league's roster rule, `fill_slots` - best first, into the
open slot with the most room - which `_fill_night` below copies without the
DataFrame. For one night that greedy can leave a slot empty that a manager fills
by moving a dual-eligible player: a C/LW takes left wing because it has more
room, and the pure centre behind him sits with centre open. A manager never
does that, so nights are set with `lineup`. `_fill_night` stays for the season
roster, where the league's rule is the right one.

And a start is not always a game gained. A centre better than both of mine
starts every night he plays, but on a night all three play he only bumps one of
mine and my lineup is no bigger. `added_games` counts the nights it is.

Nothing here touches a posterior. It is a calendar question asked of a roster.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Starts:
    """One candidate's schedule, read against my roster."""

    games: int
    starts: int
    blocked: int
    dates: list[str]

    def as_dict(self, with_dates: bool = True) -> dict:
        out = {
            "games": self.games,
            "starts": self.starts,
            "blocked": self.blocked,
            "start_share": round(self.starts / self.games, 3) if self.games else None,
        }
        # Over a fortnight the dates are the evidence and fit in a log line. Over
        # a season they are 150 strings nobody reads, so they are left out.
        if with_dates:
            out["start_dates"] = self.dates
        return out


def _fill_night(
    players: list[tuple[int, float, tuple[str, ...]]], shape: dict[str, int]
) -> dict[int, str]:
    """One night's lineup: best first, into the open slot with the most room.

    The same rule as `fill_slots`, without the DataFrame. A season window asks
    this question about 180 nights per candidate, and building a frame per night
    made that slow enough to notice at the table. `test_the_fast_lineup_fill_
    matches_fill_slots` holds the two together, because a second implementation
    that quietly disagrees is exactly what this project keeps having to undo.
    """
    openings = {p: int(n) for p, n in shape.items()}
    assigned: dict[int, str] = {}
    for pid, _, eligible in sorted(players, key=lambda r: -r[1]):
        open_to = [p for p in eligible if openings.get(p, 0) > 0]
        if not open_to:
            continue
        pick = max(open_to, key=lambda p: openings[p])
        openings[pick] -= 1
        assigned[pid] = pick
    return assigned


def lineup(players: list[tuple[int, float, tuple[str, ...]]], shape: dict[str, int]) -> dict:
    """One night's lineup as a manager sets it: player id -> the slot he fills.

    As many slots filled as possible, and the best players in them. Players are
    taken best first and each starts if the lineup can take him - into an open
    slot, or into one freed by moving an earlier starter to another slot he is
    eligible for - without benching anyone already in. That greedy is exact
    here: the sets of players who can all start together form a matroid, so
    taking the best who still fit gives the best lineup, and the most players.
    """
    units = [slot for slot, n in shape.items() for _ in range(int(n))]
    holder = [-1] * len(units)

    def place(i: int, seen: list[bool]) -> bool:
        for u, slot in enumerate(units):
            if seen[u] or slot not in players[i][2]:
                continue
            seen[u] = True
            if holder[u] < 0 or place(holder[u], seen):
                holder[u] = i
                return True
        return False

    for i in sorted(range(len(players)), key=lambda k: -players[k][1]):
        place(i, [False] * len(units))
    return {players[holder[u]][0]: units[u] for u in range(len(units)) if holder[u] >= 0}


def _entry(row, column: str) -> tuple[int, float, tuple[str, ...], str]:
    return (
        int(row["player_id"]),
        float(row[column]),
        tuple(row["eligible"]) if row.get("eligible") is not None else (str(row["position"]),),
        str(row["team"]),
    )


def marginal_starts(
    candidate: pd.Series,
    mine: pd.DataFrame,
    calendar: dict[str, list[date]],
    shape: dict[str, int],
    window: tuple[date, date],
    column: str = "mean",
) -> Starts:
    """How many of `candidate`'s games in `window` he would actually start.

    `calendar` maps a team abbreviation to the dates it plays. Only the starting
    slots count: a player who lands on the bench on a given night scores nothing
    that night, which is the whole point of asking.
    """
    first, last = window
    team = str(candidate.get("team", ""))
    plays = {t: set(days) for t, days in calendar.items()}
    games = sorted(d for d in plays.get(team, set()) if first <= d <= last)
    if not games:
        # No games is a real answer - a team on a bye, or a schedule that does
        # not cover this window - and it is not the same as "not checked".
        return Starts(games=0, starts=0, blocked=0, dates=[])

    # The candidate has to win a slot against the players I already hold, so he
    # goes into the pool and each night's lineup is set with him in it.
    pid = int(candidate["player_id"])
    pool = [_entry(r, column) for _, r in mine.iterrows()]
    pool.append(_entry(candidate, column))

    started: list[str] = []
    for day in games:
        tonight = [(p, m, e) for p, m, e, t in pool if day in plays.get(t, ())]
        if lineup(tonight, shape).get(pid):
            started.append(day.isoformat())
    return Starts(
        games=len(games),
        starts=len(started),
        blocked=len(games) - len(started),
        dates=started,
    )


@dataclass
class Added:
    """The games one more player would add to my lineup."""

    games: int
    added: int
    dates: list[str]


def added_games(
    candidate: pd.Series,
    mine: pd.DataFrame,
    calendar: dict[str, list[date]],
    shape: dict[str, int],
    window: tuple[date, date],
    column: str = "mean",
) -> Added:
    """How many of his games would put one more game in my lineup.

    The question a schedule pick asks, and not the same as his starts: on a
    night my lineup is already full at every slot he could take, he adds nothing
    however good he is - a better player there bumps one of mine and the lineup
    is no bigger. His quality is the board's other columns. This counts the
    nights my lineup, as it stands, has room for him, directly or by moving one
    of my players who is eligible somewhere else. A player already on my roster
    is counted against the rest of it.
    """
    first, last = window
    team = str(candidate.get("team", ""))
    plays = {t: set(days) for t, days in calendar.items()}
    games = sorted(d for d in plays.get(team, set()) if first <= d <= last)
    pid = int(candidate["player_id"])
    pool = [_entry(r, column) for _, r in mine.iterrows() if int(r["player_id"]) != pid]
    who = _entry(candidate, column)[:3]
    added: list[str] = []
    for day in games:
        tonight = [(p, m, e) for p, m, e, t in pool if day in plays.get(t, ())]
        if len(lineup([*tonight, who], shape)) > len(lineup(tonight, shape)):
            added.append(day.isoformat())
    return Added(games=len(games), added=len(added), dates=added)


def week_window(weeks: list[dict], first: int, count: int) -> tuple[date, date] | None:
    """The calendar span of `count` fantasy weeks starting at week `first`.

    Fantasy weeks are the league's, not the NHL's: Yahoo's scoring week runs
    Monday to Sunday and the first one is short whenever the season opens
    midweek. Counting seven days from opening night would quietly measure the
    wrong fortnight, so the boundaries come from the league settings.
    """
    wanted = [w for w in weeks if first <= int(w["week"]) < first + count]
    if len(wanted) < count:
        logger.warning(
            "asked for %d week(s) from week %d but the league defines %d; "
            "the schedule window is short",
            count,
            first,
            len(wanted),
        )
    if not wanted:
        return None
    return (
        min(date.fromisoformat(str(w["start"])) for w in wanted),
        max(date.fromisoformat(str(w["end"])) for w in wanted),
    )
