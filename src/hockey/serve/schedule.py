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
from itertools import product

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


# A goalie's chance to start the second night of a back-to-back, against his
# share of his team's starts over the season: (share, chance) knots, straight
# lines between them and flat past the last. Measured by
# scripts/measure_back_to_backs.py on 643 goalie-seasons spent on one team,
# 2018-19 to 2025-26 without 2020-21: a starter with 71% of his team's starts
# takes 37% of its second nights, a backup at 28% takes 45%, a third goalie at 6%
# takes 13%. Fitted on the seasons before 2025-26 and scored on it, the curve cut
# the error on second nights by a seventh (Brier 0.235 to 0.201) and took the
# bias out of a starter's week with a back-to-back: a flat share gave him 0.14
# starts too many, this 0.04 too few. The highest band, four goalie-seasons over
# 80%, is too thin to use.
BACK_TO_BACK_SECOND = (
    (0.0, 0.0),
    (0.063, 0.132),
    (0.275, 0.447),
    (0.42, 0.457),
    (0.576, 0.391),
    (0.712, 0.366),
)


def night_kinds(days, date_of=None) -> dict:
    """Each of a team's nights: "second" when it played the day before, "first"
    when it plays the day after, and "rest" otherwise. `date_of` turns a
    calendar day into its date when days are indices; by default they are
    dates."""
    to_date = date_of or (lambda d: d)
    ordinal = {d: date.fromisoformat(str(to_date(d))[:10]).toordinal() for d in days}
    ordered = sorted(days, key=lambda d: ordinal[d])
    kinds = {}
    for j, d in enumerate(ordered):
        if j and ordinal[d] - ordinal[ordered[j - 1]] == 1:
            kinds[d] = "second"
        elif j + 1 < len(ordered) and ordinal[ordered[j + 1]] - ordinal[d] == 1:
            kinds[d] = "first"
        else:
            kinds[d] = "rest"
    return kinds


def second_night_chance(share: float) -> float:
    knots = BACK_TO_BACK_SECOND
    for (x0, y0), (x1, y1) in zip(knots, knots[1:], strict=False):
        if share <= x1:
            return y0 + (y1 - y0) * (max(share, x0) - x0) / (x1 - x0)
    return knots[-1][1]


def start_chances(share: float, days, date_of=None) -> dict:
    """A goalie's chance to start each of his team's nights: his season share,
    bent by back-to-backs and held to the same season total. The second night of
    one goes by `BACK_TO_BACK_SECOND`, the first goes as the season does, and the
    other nights make up the difference - a starter gains a little on ordinary
    nights what he gives up on second ones, which is also what was measured."""
    kinds = night_kinds(days, date_of)
    n = {k: sum(1 for v in kinds.values() if v == k) for k in ("rest", "first", "second")}
    second = second_night_chance(share)
    rest = share + n["second"] * (share - second) / n["rest"] if n["rest"] else share
    if rest > 1:
        rest, second = 1.0, share + n["rest"] * (share - 1) / n["second"]
    elif rest < 0:
        rest, second = 0.0, share + n["rest"] * share / n["second"]
    chance = {"rest": rest, "first": share, "second": second}
    return {d: chance[k] for d, k in kinds.items()}


def night_points(tonight: list[dict], shape: dict[str, int], day=None) -> float:
    """One night's expected lineup points, for the players on the ice tonight.

    Each entry is {"id", "rate", "elig"} and, for a goalie, "share" and "team",
    with "chance" - his chance on each night, `start_chances` - when back-to-backs
    are counted; `day` picks tonight's. A skater plays every game his team does,
    and the skater slots are set as a manager sets them (`lineup`) on per-game
    rates. A goalie starts only some of his team's games, so the goalie slots are
    an expectation over which of mine start tonight: every combination of
    starters, weighted by its chance, the slots taking the best of those who
    start. A team starts one goalie a night, so two of mine from one team never
    both start: each team is one draw, none of them or exactly one. A night
    rarely has more than three of my goalies on it, so enumerating is exact and
    cheap.
    """
    skaters = [p for p in tonight if p.get("share") is None]
    goalies = [p for p in tonight if p.get("share") is not None]
    skater_shape = {s: n for s, n in shape.items() if s != "G"}
    chosen = lineup([(p["id"], p["rate"], tuple(p["elig"])) for p in skaters], skater_shape)
    rate = {p["id"]: p["rate"] for p in skaters}
    points = sum(rate[i] for i in chosen)

    slots = int(shape.get("G", 0))
    if slots and goalies:
        teams: dict = {}
        for g in goalies:
            chance = g.get("chance")
            p = chance.get(day, g["share"]) if chance and day is not None else g["share"]
            teams.setdefault(g.get("team", f"#{g['id']}"), []).append((p, g["rate"]))
        draws = []
        for group in teams.values():
            total = sum(p for p, _ in group)
            k = 1.0 / total if total > 1 else 1.0
            draws.append([(max(0.0, 1.0 - total * k), None)] + [(p * k, r) for p, r in group])
        for combo in product(*draws):
            chance = 1.0
            for p, _ in combo:
                chance *= p
            if chance:
                up = sorted((r for _, r in combo if r is not None), reverse=True)
                points += chance * sum(up[:slots])
    return points


def window_points(
    entries: list[dict], calendar: dict[str, list], shape: dict[str, int], first, last
) -> dict:
    """Expected lineup points each night from `first` to `last`, inclusive.

    Entries are {"id", "team", "rate", "elig"} plus "share" for a goalie, his
    "chance" on each night when back-to-backs are counted, and an optional
    "from", the first night he can play - a return date from injury.
    A roster's week is the sum of its nights, and what a move is worth is the
    difference between two rosters' sums over the same nights: that is the
    question a schedule tool that shows only games cannot answer, because a
    player's games are not his starts and his starts are not all points gained.
    """
    plays = {t: set(days) for t, days in calendar.items()}
    on: dict = {}
    for e in entries:
        for d in plays.get(e["team"], ()):
            if first <= d <= last and (e.get("from") is None or d >= e["from"]):
                on.setdefault(d, []).append(e)
    return {d: night_points(tonight, shape, d) for d, tonight in sorted(on.items())}
