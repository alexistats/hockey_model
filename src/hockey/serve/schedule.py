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
lineup on the 14th" is to fill that night's lineup the same way the league fills
a roster: best first, into an open slot he is eligible for. That is `fill_slots`,
the primitive the rest of the project already uses, and `_fill_night` below is
the same rule without the DataFrame - pinned to it by a test, because a season
window asks the question 180 times per candidate.

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
    # goes into the pool and each night is filled best first.
    pid = int(candidate["player_id"])
    pool: list[tuple[int, float, tuple[str, ...], str]] = [
        (
            int(r["player_id"]),
            float(r[column]),
            tuple(r["eligible"]) if r.get("eligible") is not None else (str(r["position"]),),
            str(r["team"]),
        )
        for _, r in mine.iterrows()
    ]
    pool.append(
        (
            pid,
            float(candidate[column]),
            tuple(candidate["eligible"])
            if candidate.get("eligible") is not None
            else (str(candidate["position"]),),
            team,
        )
    )

    started: list[str] = []
    for day in games:
        tonight = [(p, m, e) for p, m, e, t in pool if day in plays.get(t, ())]
        if _fill_night(tonight, shape).get(pid):
            started.append(day.isoformat())
    return Starts(
        games=len(games),
        starts=len(started),
        blocked=len(games) - len(started),
        dates=started,
    )


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
