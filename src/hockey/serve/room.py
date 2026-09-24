"""What the rest of the room takes before my next turn.

    python -m hockey.serve.room --mocks ../draft_bot/artifacts/mocks

`cost_of_waiting` asks one question: if I pass on a position now, what will the
best player there be worth when my turn comes back? The answer depends on
nothing but what the other managers take in between, and the first version
answered it with the one assumption the room was measured not to keep - that
they draft straight down *our* value board.

Measured on eight saved Yahoo mock rooms, 1,643 picks by other managers:

- From round 4 on, the median room pick was the 20th to 35th best player left on
  our board. Within one position the room took our best player there only 6-16%
  of the time; the median was our fifth to eighth.
- So a straight run down our board had our best forwards and goalies gone when
  the room left them standing. Waiting on C, LW, RW or G was priced 25-50 points
  too high on average, and on one side only. Defence came out roughly right,
  and its flat 0.0 was the same assumption seen from below: no defenceman sat in
  the top dozen of our board, so the straight run could never take one.
- The room drafts from its own order - Yahoo's ranking, which its autopick
  follows - and toward its open starting slots: 84-100% of room picks through
  round 12 filled one. The average pick number of a player over the saved rooms
  predicts the next room pick far better than our value does (held out one room
  at a time, 3.15 against 4.28 nats per pick), and once it is known, our value
  adds nothing.

So the room is modelled the way it measured. Each of the next picks is drawn
from the players left, more likely the earlier the room usually takes him and
more likely again if he fills an open starting slot for the team picking - and
sometimes it is somebody not on our board at all, which about one pick in seven
was.
Simulated a few hundred times, the best player left at each position is
averaged. `best_after` is that expectation and `cost` the expected loss, both
in our value: what the room takes is its choice, and what that costs me is ours
to measure.

The room's order comes from saved mock drafts, written to `config/room_<season>
.json` by the command above. Those were Yahoo's standard mock rooms, not this
league: the real room may draft differently, and a model fitted on mocks is a
measurement of mocks. It is still a far better guess at thirteen other people
than a board none of them can see.

Nothing here touches a posterior. The room's order is a fact about other people.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# How many simulated rooms each answer averages over. At 500 the Monte Carlo
# error on an expected best-after is under a point, which is well inside what
# the model itself gets wrong.
SIMS = 500

# Rooms need to be at least this far along to say anything about the middle of
# a draft, which is where the positional question is hardest.
MIN_ROUNDS = 8

# Opening picks that identify a room. The same room is often logged more than
# once - the bot restarted mid-draft - and those logs share their first picks.
ROOM_KEY = 30


def _eligible(row) -> tuple[str, ...]:
    got = getattr(row, "eligible", None)
    return tuple(got) if isinstance(got, tuple | list) and got else (str(row.slot),)


def open_slots(players: list[tuple[int, float, tuple[str, ...]]], shape: dict[str, int]) -> dict:
    """A team's open starting slots, filled the way the league fills them.

    `players` is (player_id, mean, eligible) for the board players the team
    holds. The fill is `schedule._fill_night`, which is `fill_slots` without the
    DataFrame and is held to it by a test.
    """
    from hockey.serve.schedule import _fill_night

    assigned = _fill_night(players, shape)
    filled: dict[str, int] = {}
    for slot in assigned.values():
        filled[slot] = filled.get(slot, 0) + 1
    return {p: max(0, int(n) - filled.get(p, 0)) for p, n in shape.items()}


def snake_seat(overall: int, n_teams: int, snake: bool = True) -> int:
    """Which seat makes overall pick `overall`."""
    rnd = (overall - 1) // n_teams + 1
    within = (overall - 1) % n_teams + 1
    return n_teams + 1 - within if (snake and rnd % 2 == 0) else within


@dataclass
class RoomModel:
    """How the rest of a draft room picks, fitted on saved drafts.

    Utility of a player to the team picking:

        order_weight * -ln(his average pick) + need_weight * fills_an_open_slot

    and `off_board` is the utility of a pick outside our board altogether. Each
    pick is the best utility after independent Gumbel noise, which is exactly a
    multinomial logit over everything left.

    On the log of the average pick, not the pick itself, because the room is
    nearly certain at the top and loose further down: the 10th and 20th players
    by average pick are not the same bet as the 100th and 110th. Held out one
    room at a time the log scored 3.150 nats per pick against 3.302 for the
    linear version, and it stopped the linear one's habit of calling a
    coin-flip where the best player was in fact taken nine times in ten.
    """

    order_weight: float
    need_weight: float
    off_board: float
    adp: dict[int, float]
    # The average pick given to a player the saved rooms never took, a round
    # past the end of the longest of them.
    undrafted: float
    fitted: str = ""
    rooms: int = 0
    picks: int = 0
    heldout: dict = field(default_factory=dict)
    source: str = ""
    players: list[dict] = field(default_factory=list)
    # Fitted without the need term, for when the teams' rosters cannot be read.
    # Running the coefficients above with the need term switched off is not the
    # same room: the off-board constant was fitted against players who mostly
    # carried the need bonus, and without it the room drafts off our board about
    # a fifth more often than the saved rooms did.
    blind_order_weight: float | None = None
    blind_off_board: float | None = None

    def order_of(self, player_ids) -> np.ndarray:
        return np.array([self.adp.get(int(p), self.undrafted) for p in player_ids], dtype=float)

    def describe(self) -> dict:
        return {
            "source": self.source,
            "fitted": self.fitted,
            "rooms": self.rooms,
            "picks": self.picks,
            "order_weight": round(self.order_weight, 4),
            "need_weight": round(self.need_weight, 4),
            "off_board": round(self.off_board, 4),
            "blind_order_weight": None
            if self.blind_order_weight is None
            else round(self.blind_order_weight, 4),
            "blind_off_board": None
            if self.blind_off_board is None
            else round(self.blind_off_board, 4),
            "heldout": self.heldout,
        }

    def save(self, path: Path) -> None:
        body = {
            "what": (
                "How the rest of a Yahoo draft room picks, fitted on saved mock drafts. "
                "Read by hockey.serve for cost_of_waiting. Rebuild with "
                "python -m hockey.serve.room --mocks <dir>."
            ),
            **self.describe(),
            "undrafted": self.undrafted,
            "players": self.players
            or [{"player_id": p, "adp": round(a, 2)} for p, a in sorted(self.adp.items())],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(body, indent=1, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> RoomModel:
        body = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            order_weight=float(body["order_weight"]),
            need_weight=float(body["need_weight"]),
            off_board=float(body["off_board"]),
            adp={int(r["player_id"]): float(r["adp"]) for r in body["players"]},
            undrafted=float(body["undrafted"]),
            fitted=str(body.get("fitted", "")),
            rooms=int(body.get("rooms", 0)),
            picks=int(body.get("picks", 0)),
            heldout=dict(body.get("heldout") or {}),
            source=str(body.get("source", "")),
            players=list(body["players"]),
            blind_order_weight=body.get("blind_order_weight"),
            blind_off_board=body.get("blind_off_board"),
        )


# ---- the room, simulated ---------------------------------------------------


def expected_after(
    pool: pd.DataFrame,
    positions: list[str],
    seats: list[int],
    room: RoomModel,
    openings: dict[int, dict[str, int]] | None = None,
    sims: int = SIMS,
    seed: int = 0,
) -> dict[str, dict | None]:
    """Best available at each position now, and expected at my next turn.

    `pool` is the players left, with `player_id`, `vorp` and `eligible`.
    `seats` is who picks before my next turn, in order, and `openings` each of
    those seats' open starting slots; without it the room still drafts in its
    own order, just blind to what each team is missing.

    Every simulated room is scored in this turn's `vorp`, so `cost` is a
    difference in points at one position and the replacement level cancels.
    """
    if pool.empty:
        return dict.fromkeys(positions)
    ids = pool["player_id"].astype(int).to_numpy()
    value = pool["vorp"].to_numpy(dtype=float)
    elig = np.array([[p in _eligible(r) for p in positions] for r in pool.itertuples()])
    n, m = len(ids), int(sims)
    rows = np.arange(m)
    rng = np.random.default_rng(seed)
    blind = openings is None and room.blind_order_weight is not None
    weight = room.blind_order_weight if blind else room.order_weight
    off_board = room.blind_off_board if blind else room.off_board
    base = -weight * np.log(room.order_of(ids))

    avail = np.ones((m, n), dtype=bool)
    left = None
    if openings is not None:
        left = {
            s: np.tile([int(openings.get(s, {}).get(p, 0)) for p in positions], (m, 1))
            for s in set(seats)
        }
    for seat in seats:
        u = np.tile(base, (m, 1))
        if left is not None:
            fills = (elig[None, :, :] & (left[seat][:, None, :] > 0)).any(axis=2)
            u += room.need_weight * fills
        u[~avail] = -np.inf
        u += rng.gumbel(size=(m, n))
        choice = u.argmax(axis=1)
        took = u[rows, choice] > off_board + rng.gumbel(size=m)
        avail[rows[took], choice[took]] = False
        if left is not None:
            # Into the open slot he is eligible for with the most room, the
            # same rule the league's own fill uses.
            slots = left[seat]
            fit = elig[choice] & (slots > 0)
            where = np.where(fit, slots, -1).argmax(axis=1)
            filled = took & fit.any(axis=1)
            slots[rows[filled], where[filled]] -= 1

    out: dict[str, dict | None] = {}
    for j, position in enumerate(positions):
        at = np.flatnonzero(elig[:, j])
        if not len(at):
            out[position] = None
            continue
        ranked = at[np.argsort(-value[at], kind="stable")]
        best = ranked[0]
        now = float(value[best])
        survivors = np.where(avail[:, ranked], value[ranked][None, :], -np.inf).max(axis=1)
        some = np.isfinite(survivors)
        # A position emptied in most of the simulated rooms has no "best left"
        # to speak of, and averaging the few where someone survived would
        # report the rare case as the typical one.
        then = float(survivors[some].mean()) if some.mean() >= 0.5 else None
        out[position] = {
            "best_now": round(now, 1),
            "best_after": None if then is None else round(then, 1),
            "cost": None if then is None else round(max(0.0, now - then), 1),
            "p_best_survives": round(float(avail[:, best].mean()), 3),
        }
    return out


# ---- saved rooms -----------------------------------------------------------


@dataclass
class Room:
    """One saved draft: every pick in order, and which seat was the bot's."""

    name: str
    picks: list[int | None]  # board player_id, or None for a player not on our board
    bot_seat: int | None


def rooms_from_logs(
    directory: Path, resolver, n_teams: int = 14
) -> tuple[list[Room], list[tuple[str, str]]]:
    """Saved drafts, read from draft_bot's record of its calls to this API.

    Each log is `api.jsonl`, one line per call: the path, the payload sent and
    the response. The longest `/draft/observed` payload is the room's pick list,
    in draft order, because that is the order the bot reads Yahoo's Picks tab
    in - and that order is checked rather than trusted: every pick the bot made
    must land on one seat of the snake. A log that fails is skipped and said so.

    Names are resolved the one way this server resolves names, and a name that
    does not resolve is a pick of somebody not on our board, never a guess.
    Simulated rooms are skipped: they are draft_bot's opponent models, and a
    room model fitted to them would be fitted to a guess.
    """
    rooms: dict[tuple, Room] = {}
    longest: dict[tuple, int] = {}
    skipped: list[tuple[str, str]] = []
    for log in sorted(directory.glob("*/api.jsonl")):
        name = log.parent.name
        if "-sim-" in name:
            skipped.append((name, "simulated room"))
            continue
        names: list[str] = []
        bot: list[int] = []
        for line in log.read_text(encoding="utf-8").splitlines():
            call = json.loads(line)
            if call.get("path") == "/draft/observed":
                got = (call.get("payload") or {}).get("names") or []
                if len(got) > len(names):
                    names = list(got)
            elif call.get("path") == "/draft/mine" and isinstance(call.get("response"), dict):
                pid = call["response"].get("player_id")
                if pid is not None and int(pid) not in bot:
                    bot.append(int(pid))
        if len(names) < n_teams * MIN_ROUNDS:
            skipped.append((name, f"only {len(names)} picks"))
            continue
        picks: list[int | None] = []
        for raw in names:
            got = resolver.resolve(raw)
            picks.append(int(got.player_id) if got.ok else None)
        seats = {snake_seat(picks.index(p) + 1, n_teams) for p in bot if p in picks}
        if len(seats) > 1:
            skipped.append((name, f"the bot's own picks land on seats {sorted(seats)}"))
            continue
        key = tuple(names[:ROOM_KEY])
        if len(names) > longest.get(key, 0):
            if key in rooms:
                skipped.append((rooms[key].name, f"same room as {name}, shorter"))
            rooms[key] = Room(name, picks, next(iter(seats)) if seats else None)
            longest[key] = len(names)
        else:
            skipped.append((name, f"same room as {rooms[key].name}, shorter"))
    return list(rooms.values()), skipped


def average_draft_position(rooms: list[Room], n_teams: int) -> tuple[dict, float, list[dict]]:
    """Each player's mean pick over the rooms that exposed him to the room.

    A player the bot took was never on offer to the rest of that room, so that
    room says nothing about when the room would have taken him: it is left out
    of his average, not counted as undrafted. A player some rooms never took at
    all is counted at `undrafted` there - a round past the end of the longest
    room - which is a floor on the truth and the best a censored draft can say.
    """
    undrafted = float(max(len(r.picks) for r in rooms) + n_teams)
    ids = sorted({p for r in rooms for p in r.picks if p is not None})
    total = dict.fromkeys(ids, 0.0)
    seen = dict.fromkeys(ids, 0)
    taken = dict.fromkeys(ids, 0)
    for room in rooms:
        where = {p: k for k, p in enumerate(room.picks, start=1) if p is not None}
        for pid in ids:
            k = where.get(pid)
            if k is not None and snake_seat(k, n_teams) == room.bot_seat:
                continue
            seen[pid] += 1
            if k is None:
                total[pid] += undrafted
            else:
                total[pid] += k
                taken[pid] += 1
    adp = {p: total[p] / seen[p] for p in ids if seen[p]}
    table = [
        {"player_id": p, "adp": round(adp[p], 2), "rooms": seen[p], "taken": taken[p]}
        for p in sorted(adp, key=adp.get)
    ]
    return adp, undrafted, table


def _situations(
    rooms: list[Room],
    adp: dict[int, float],
    undrafted: float,
    players: pd.DataFrame,
    shape: dict[str, int],
    n_teams: int,
    need: bool = True,
) -> list[tuple[np.ndarray, int]]:
    """Every pick a room manager made: features of the alternatives, and which.

    Rows are the players left on our board plus one last row for "somebody not
    on our board". Columns are the order term, the need term and the off-board
    intercept, so the three coefficients are fitted together.
    """
    ids = players["player_id"].astype(int).to_numpy()
    index = {p: i for i, p in enumerate(ids)}
    order = -np.log([adp.get(int(p), undrafted) for p in ids])
    positions = list(shape)
    elig = np.array([[p in _eligible(r) for p in positions] for r in players.itertuples()])
    mean = players["mean"].to_numpy(dtype=float)
    out = []
    for room in rooms:
        avail = np.ones(len(ids), dtype=bool)
        held: dict[int, list[int]] = {}
        for k, pid in enumerate(room.picks, start=1):
            seat = snake_seat(k, n_teams)
            if seat != room.bot_seat and (pid is None or (pid in index and avail[index[pid]])):
                if need:
                    have = [
                        (ids[i], mean[i], tuple(np.array(positions)[elig[i]]))
                        for i in held.get(seat, [])
                    ]
                    gaps = open_slots(have, shape)
                    fills = (elig & (np.array([gaps[p] for p in positions]) > 0)).any(axis=1)
                else:
                    fills = np.zeros(len(ids), dtype=bool)
                live = np.flatnonzero(avail)
                x = np.zeros((len(live) + 1, 3))
                x[:-1, 0] = order[live]
                x[:-1, 1] = fills[live]
                x[-1, 2] = 1.0
                chosen = len(live) if pid is None else int(np.searchsorted(live, index[pid]))
                out.append((x, chosen))
            if pid is not None and pid in index and avail[index[pid]]:
                avail[index[pid]] = False
                held.setdefault(seat, []).append(index[pid])
    return out


def _newton(
    situations: list[tuple[np.ndarray, int]], need: bool = True
) -> tuple[np.ndarray, float]:
    """Maximum likelihood for the conditional logit, by damped Newton.

    Three parameters and a concave log-likelihood, so it needs nothing beyond
    numpy. Damped because the off-board intercept sits near -9: a full first
    step from zero overshoots it, the off-board probability underflows, and
    the curvature along it vanishes. Halving the step until the likelihood
    improves keeps every iterate where the curvature is real.
    """
    keep = [0, 1, 2] if need else [0, 2]

    def evaluate(theta):
        grad = np.zeros(len(keep))
        hess = np.zeros((len(keep), len(keep)))
        loglik = 0.0
        for x, chosen in situations:
            x = x[:, keep]
            u = x @ theta
            top = u.max()
            w = np.exp(u - top)
            p = w / w.sum()
            mean = p @ x
            grad += x[chosen] - mean
            hess -= (x.T * p) @ x - np.outer(mean, mean)
            loglik += u[chosen] - top - math.log(w.sum())
        return loglik, grad, hess

    theta = np.zeros(len(keep))
    loglik, grad, hess = evaluate(theta)
    for _ in range(100):
        # Ascent direction; the ridge only matters if a feature is constant.
        step = -np.linalg.solve(hess - 1e-9 * np.eye(len(keep)), grad)
        scale = 1.0
        while True:
            trial = theta + scale * step
            got = evaluate(trial)
            if got[0] >= loglik or scale < 1e-8:
                break
            scale /= 2
        theta, (loglik, grad, hess) = trial, got
        if np.abs(scale * step).max() < 1e-8:
            break
    full = np.zeros(3)
    full[keep] = theta
    return full, loglik


def _loglik(theta: np.ndarray, situations: list[tuple[np.ndarray, int]]) -> float:
    total = 0.0
    for x, chosen in situations:
        u = x @ theta
        top = u.max()
        total += u[chosen] - top - math.log(np.exp(u - top).sum())
    return total


def fit(
    rooms: list[Room],
    players: pd.DataFrame,
    shape: dict[str, int],
    n_teams: int = 14,
    evidence: bool = True,
) -> RoomModel:
    """The room model, with its held-out evidence.

    The evidence is leave-one-room-out: the average pick and the coefficients
    are fitted without a room, then scored on it, room by room. That is the only
    honest number when there are ten rooms, and it is what the file carries.
    `evidence=False` skips it, for a refit that is itself the held-out half of
    somebody else's test.
    """
    adp, undrafted, table = average_draft_position(rooms, n_teams)
    situations = _situations(rooms, adp, undrafted, players, shape, n_teams)
    theta, _ = _newton(situations)
    blind, _ = _newton(
        _situations(rooms, adp, undrafted, players, shape, n_teams, need=False), need=False
    )

    heldout = {"nats_per_pick": {}, "picks": 0}
    scores: dict[str, float] = {"order and need": 0.0, "order only": 0.0, "uniform": 0.0}
    count = 0
    if evidence and len(rooms) > 1:
        for held in rooms:
            train = [r for r in rooms if r is not held]
            a, u, _ = average_draft_position(train, n_teams)
            test = _situations([held], a, u, players, shape, n_teams)
            with_need, _ = _newton(_situations(train, a, u, players, shape, n_teams))
            no_need, _ = _newton(
                _situations(train, a, u, players, shape, n_teams, need=False), need=False
            )
            scores["order and need"] -= _loglik(with_need, test)
            test_blind = _situations([held], a, u, players, shape, n_teams, need=False)
            scores["order only"] -= _loglik(no_need, test_blind)
            scores["uniform"] += sum(math.log(len(x)) for x, _ in test)
            count += len(test)
        heldout = {
            "nats_per_pick": {k: round(float(v) / count, 4) for k, v in scores.items()},
            "picks": count,
            "how": "leave one room out: average pick and coefficients refitted without it",
        }

    names = players.set_index(players["player_id"].astype(int))["player"].to_dict()
    for row in table:
        row["player"] = str(names.get(row["player_id"], ""))
    by_room = sum(
        1 for r in rooms for k, p in enumerate(r.picks, 1) if snake_seat(k, n_teams) != r.bot_seat
    )
    off = sum(
        1
        for r in rooms
        for k, p in enumerate(r.picks, 1)
        if p is None and snake_seat(k, n_teams) != r.bot_seat
    )
    return RoomModel(
        order_weight=float(theta[0]),
        need_weight=float(theta[1]),
        off_board=float(theta[2]),
        blind_order_weight=float(blind[0]),
        blind_off_board=float(blind[2]),
        adp=adp,
        undrafted=undrafted,
        fitted=date.today().isoformat(),
        rooms=len(rooms),
        picks=by_room,
        heldout=heldout,
        source=(
            f"{len(rooms)} saved Yahoo mock rooms, {by_room} picks by other managers "
            f"({off} of players not on our board)"
        ),
        players=table,
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m hockey.serve.room")
    parser.add_argument("--mocks", required=True, help="draft_bot's artifacts/mocks directory")
    parser.add_argument("--board", default="artifacts/board_v3")
    parser.add_argument("--goalies", default="artifacts/goalies_v2")
    parser.add_argument("--teams", type=int, default=14)
    parser.add_argument("--out", default=None, help="default config/room_<season>.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    from hockey.seasons import PROJECTION_SEASON
    from hockey.serve import board as board_mod
    from hockey.serve.identity import Resolver

    data = board_mod.load(Path(args.board), Path(args.goalies) if args.goalies else None)
    rooms, skipped = rooms_from_logs(Path(args.mocks), Resolver(data.players), args.teams)
    for name, why in skipped:
        print(f"skipped {name}: {why}")
    if not rooms:
        raise SystemExit(f"no usable rooms under {args.mocks}")
    model = fit(rooms, data.players, data.roster_shape, args.teams)
    out = Path(args.out or f"config/room_{PROJECTION_SEASON // 10000}.json")
    model.save(out)
    print(f"{model.source}")
    print(
        f"order_weight {model.order_weight:.3f} per unit of -ln(average pick), need_weight "
        f"{model.need_weight:.3f}, off_board {model.off_board:.3f}"
    )
    print(f"held out: {model.heldout}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
