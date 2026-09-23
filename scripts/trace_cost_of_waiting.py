"""Replay a saved mock through the draft API and hold cost_of_waiting to it.

    python scripts/trace_cost_of_waiting.py ../draft_bot/artifacts/mocks/20260923-131215-live-slot8
    python scripts/trace_cost_of_waiting.py <mock dir> --holdout
    python scripts/trace_cost_of_waiting.py --all ../draft_bot/artifacts/mocks

Every call the bot logged is re-sent to the API in order, and at each of my
turns, for each position:

  best_now   the best player left, in this turn's value
  old        best_after under the old rule, the next picks straight off the top
             of our board
  new        best_after as the server now has it: an expectation over simulated
             rooms
  actual     the best player there at my next turn had I passed on the
             position - this turn's value, my own pick put back

Two yardsticks, because they answer different questions:

  brief   best_after against the next turn's `best_now` as the server reported
          it. This is the one draft_bot measured. It is also charged with two
          things no prediction of the room is asked about: my own pick (taking
          the best defenceman myself drops the best defenceman) and the
          replacement level sinking as the pool drains, which lifts every value
          a little from one turn to the next.
  fair    against `actual`, and only at positions my own pick could not fill -
          when I take the best D, the room never had the chance to, and putting
          him back would score the model against a room that was never offered
          him.

--holdout refits the room model without this mock's own room first, so `new`
is out of sample. --all does that for every saved room with a bot seat and
pools the turns.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from hockey.serve import board as board_mod
from hockey.serve.app import create_app
from hockey.serve.identity import Resolver
from hockey.serve.room import fit, rooms_from_logs

BOARD = Path("artifacts/board_v3")
GOALIES = Path("artifacts/goalies_v2")


def eligible(row) -> tuple[str, ...]:
    e = row.eligible
    return tuple(e) if isinstance(e, tuple | list) else (str(row.slot),)


def straight_down_the_board(valued: pd.DataFrame, positions: list[str], picks: int) -> dict:
    """The old rule, kept here only as the 'before' column."""
    pool = valued.sort_values("vorp", ascending=False)
    taken = set(pool.head(picks)["player_id"].astype(int))
    out = {}
    for position in positions:
        there = pool[[position in eligible(r) for r in pool.itertuples()]]
        later = there[~there["player_id"].astype(int).isin(taken)]
        out[position] = float(later.iloc[0]["vorp"]) if len(later) else None
    return out


def best_among(valued: pd.DataFrame, ids: set[int], position: str) -> float | None:
    rows = valued[valued["player_id"].astype(int).isin(ids)]
    rows = rows[[position in eligible(r) for r in rows.itertuples()]]
    return float(rows["vorp"].max()) if len(rows) else None


def replay(mock: Path, room=None) -> list[dict]:
    """My turns in a saved mock, with what the API said and what happened."""
    calls = [json.loads(line) for line in open(mock / "api.jsonl", encoding="utf-8")]
    first = next(c for c in calls if c["path"] == "/draft/state")["response"]
    if room is not None:
        board_mod._room = lambda players: room
    app = create_app(BOARD, GOALIES, int(first["n_teams"]), int(first["slot"]))
    client = TestClient(app)
    client.post("/draft/settings", json={"rules": first.get("rules") or {}})
    data, state = app.state.board, app.state.draft

    turns: dict[int, dict] = {}
    for call in calls:
        path, payload = call["path"], call["payload"]
        if path in ("/draft/observed", "/draft/mine", "/draft/settings"):
            client.post(path, json=payload)
        elif path == "/draft/reset":
            client.post(path)
        elif path.startswith("/recommend"):
            got = client.get(path).json()
            if not got.get("on_the_clock"):
                continue
            valued, _ = board_mod.revalue(data, state.drafted)
            positions = list(got["cost_of_waiting"])
            turns[got["overall_pick"]] = {
                "round": got["round"],
                "pick": got["overall_pick"],
                "picks": got["picks_until_my_turn"],
                "needs": got["needs"],
                "new": got["cost_of_waiting"],
                "logged": (call.get("response") or {}).get("cost_of_waiting"),
                "old": straight_down_the_board(valued, positions, got["picks_until_my_turn"]),
                "valued": valued,
                "available": set(valued["player_id"].astype(int)),
                "mine": list(state.mine),
                "needs_read": (got.get("room") or {}).get("needs"),
            }
    ordered = [turns[k] for k in sorted(turns)]
    rows = []
    for now, nxt in zip(ordered, ordered[1:], strict=False):
        # The next turn has to be the one the prediction was for.
        if nxt["pick"] != now["pick"] + now["picks"] + 1:
            continue
        mine_now = [p for p in nxt["mine"] if p not in now["mine"]]
        own = mine_now[0] if mine_now else None
        own_row = now["valued"][now["valued"]["player_id"].astype(int) == own]
        own_elig = eligible(next(own_row.itertuples())) if len(own_row) else ()
        back = nxt["available"] | ({own} if own is not None else set())
        for position, c in now["new"].items():
            if not c or c.get("best_after") is None or now["old"].get(position) is None:
                continue
            actual = best_among(now["valued"], back, position)
            reported = (nxt["new"].get(position) or {}).get("best_now")
            there = now["valued"][[position in eligible(r) for r in now["valued"].itertuples()]]
            best_id = int(there.loc[there["vorp"].idxmax(), "player_id"])
            rows.append(
                {
                    "round": now["round"],
                    "pick": now["pick"],
                    "pos": position,
                    "need": now["needs"].get(position, 0),
                    "best_now": c["best_now"],
                    "old": round(now["old"][position], 1),
                    "new": c["best_after"],
                    "p_survive": c.get("p_best_survives"),
                    "actual": None if actual is None else round(actual, 1),
                    "next_now": reported,
                    "own_here": position in own_elig,
                    "survived": best_id in back,
                    "logged_old": ((now["logged"] or {}).get(position) or {}).get("best_after"),
                }
            )
    return rows


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """Bias, spread and sidedness of both predictions under both yardsticks."""
    out = []
    fair = df[~df["own_here"] & df["actual"].notna()]
    brief = df[df["next_now"].notna()]
    for label, sub, target in (("brief", brief, "next_now"), ("fair", fair, "actual")):
        for model in ("old", "new"):
            for position, g in sub.groupby("pos"):
                err = g[model] - g[target]
                out.append(
                    {
                        "yardstick": label,
                        "model": model,
                        "pos": position,
                        "n": len(err),
                        "bias": round(err.mean(), 1),
                        "mae": round(err.abs().mean(), 1),
                        "max_abs": round(err.abs().max(), 1),
                        "over": round((err < -1).mean(), 2),
                        "under": round((err > 1).mean(), 2),
                    }
                )
    return pd.DataFrame(out)


def held_out_room(mock: Path, mocks: Path):
    """The room model refitted without this mock's own room."""
    data = board_mod.load(BOARD, GOALIES)
    resolver = Resolver(data.players)
    rooms, _ = rooms_from_logs(mocks, resolver)
    own, _ = rooms_from_logs(mock.parent, resolver)
    mine = next((r for r in own if r.name == mock.name), None)
    keep = [
        r
        for r in rooms
        if r.name != mock.name and (mine is None or r.picks[:30] != mine.picks[:30])
    ]
    print(f"room model refitted on {len(keep)} of {len(rooms)} rooms, without {mock.name}")
    return fit(keep, data.players, data.roster_shape, evidence=False)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python scripts/trace_cost_of_waiting.py")
    parser.add_argument("mock", nargs="?", help="one mock directory, holding api.jsonl")
    parser.add_argument("--holdout", action="store_true", help="refit the room without it")
    parser.add_argument("--all", metavar="MOCKS", help="every saved room in this directory")
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 400)

    if args.all:
        mocks = Path(args.all)
        data = board_mod.load(BOARD, GOALIES)
        rooms, _ = rooms_from_logs(mocks, Resolver(data.players))
        frames = []
        for room in rooms:
            if room.bot_seat is None:
                continue
            mock = mocks / room.name
            model = held_out_room(mock, mocks)
            rows = pd.DataFrame(replay(mock, model))
            if rows.empty:
                # Early logs asked /recommend off the clock or skipped turns, so
                # no consecutive pair of my turns can be checked.
                print(f"{room.name}: no consecutive turns to check")
                continue
            rows["room"] = room.name
            frames.append(rows)
            print(f"{room.name}: {rows['pick'].nunique()} turns")
        df = pd.concat(frames, ignore_index=True)
    else:
        if not args.mock:
            sys.exit("give a mock directory, or --all <mocks dir>")
        mock = Path(args.mock)
        model = held_out_room(mock, mock.parent) if args.holdout else None
        df = pd.DataFrame(replay(mock, model))
        same = df.dropna(subset=["logged_old"])
        if len(same):
            agree = (same["logged_old"] - same["old"]).abs().max()
            print(f"old rule re-run against the log: largest difference {agree:.2f}")
        for position, g in df.groupby("pos", sort=False):
            print(f"\n--- {position}")
            print(
                g[
                    [
                        "round",
                        "pick",
                        "need",
                        "best_now",
                        "old",
                        "new",
                        "p_survive",
                        "actual",
                        "next_now",
                        "own_here",
                    ]
                ].to_string(index=False)
            )
    print()
    print(summary(df).to_string(index=False))
    fair = df[~df["own_here"]]
    bins = pd.cut(fair["p_survive"], [-0.001, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0])
    calib = fair.groupby(bins, observed=True).agg(
        n=("survived", "size"), said=("p_survive", "mean"), happened=("survived", "mean")
    )
    print()
    print("p_best_survives against what happened (fair turns):")
    print(calib.round(3).to_string())


if __name__ == "__main__":
    main()
