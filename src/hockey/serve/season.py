"""The league as it stands in season: whose players are whose.

    python -m hockey.serve.season seed config/draft_2026.csv --me "Alexis's Amazing Team" \
        --add "Jake DeBrusk" --drop "Braden Schneider" --injured "Seth Jarvis=2026-11-15"

The draft state ends with the draft. After it the page asks a different
question - who is on my roster, who is on someone else's, and who is hurt - and
the answer has to outlive a server restart and, once Yahoo answers again, come
from Yahoo instead of from me. So it is a file, owned here, that the page reads
and writes and a Yahoo sync can later write too.

Ids only. A name that does not resolve is kept as a name, in `unresolved`, where
the page shows it - never guessed into an id. A wrong id would put a plausible
projection for the wrong player on my roster, and nothing downstream could tell.

Nothing here changes a posterior.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_PATH = Path("artifacts/season/league.json")


@dataclass
class League:
    """Who holds whom. `out` maps a player id to his return date, or None when
    he is out with no date - either way he is left out of lineups until then."""

    mine: list[int] = field(default_factory=list)
    taken: list[int] = field(default_factory=list)
    out: dict[int, str | None] = field(default_factory=dict)
    unresolved: list[str] = field(default_factory=list)
    me: str | None = None
    updated_at: str | None = None

    def as_dict(self) -> dict:
        d = asdict(self)
        d["out"] = {str(k): v for k, v in self.out.items()}
        return d


def load(path: Path) -> League:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return League(
        mine=[int(i) for i in raw.get("mine", [])],
        taken=[int(i) for i in raw.get("taken", [])],
        out={int(k): v for k, v in raw.get("out", {}).items()},
        unresolved=list(raw.get("unresolved", [])),
        me=raw.get("me"),
        updated_at=raw.get("updated_at"),
    )


def save(league: League, path: Path) -> League:
    """Write the league, stamped. My players are never also someone else's."""
    mine = list(dict.fromkeys(league.mine))
    taken = [i for i in dict.fromkeys(league.taken) if i not in set(mine)]
    for d in league.out.values():
        if d is not None:
            date.fromisoformat(d)  # a malformed date is an error here, not a silent "out forever"
    stamped = League(
        mine,
        taken,
        dict(league.out),
        list(league.unresolved),
        league.me,
        datetime.now(UTC).isoformat(),
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stamped.as_dict(), indent=1), encoding="utf-8")
    return stamped


def seed_from_draft(rows: list[dict], me: str, resolve) -> League:
    """A league from a draft's results: my picks, everyone else's, and the names
    that did not resolve. `rows` carry "team" (the fantasy team), "player" and
    "positions" (Yahoo's, like "C,LW"); `resolve(name, position)` is the
    server's Resolver, which answers with an id or a miss and never a guess."""
    league = League(me=me)
    for r in rows:
        first = str(r.get("positions", "")).split(",")[0].strip() or None
        hit = resolve(r["player"], first)
        if not hit.ok:
            league.unresolved.append(f"{r['player']} ({r['team']}): {hit.method}")
            continue
        (league.mine if r["team"] == me else league.taken).append(int(hit.player_id))
    return league


def resolve_unresolved(league: League, resolve) -> League:
    """Place the names that now resolve - after players were added to the
    board or the sheet of players outside the model - on the team they were
    recorded against; the rest stay names."""
    still = []
    for entry in league.unresolved:
        name, _, rest = entry.partition(" (")
        team = rest.split(")")[0].split(",")[0]
        hit = resolve(name, None)
        if not hit.ok:
            still.append(entry)
            continue
        pid = int(hit.player_id)
        (league.mine if team == league.me else league.taken).append(pid)
    league.unresolved = still
    return league


def _resolve_or_fail(resolve, name: str) -> int:
    hit = resolve(name, None)
    if not hit.ok:
        raise SystemExit(f"{name!r} did not resolve ({hit.method}); give the player id instead")
    return int(hit.player_id)


def main() -> None:
    parser = argparse.ArgumentParser(prog="hockey.serve.season")
    sub = parser.add_subparsers(dest="command", required=True)
    seed = sub.add_parser("seed", help="start the season's league file from the draft results")
    seed.add_argument("draft", help="CSV with team, player, positions per pick")
    seed.add_argument("--me", required=True, help="my fantasy team's name, as in the CSV")
    seed.add_argument("--board", default="artifacts/board_v3")
    seed.add_argument("--goalies", default="artifacts/goalies_v2")
    seed.add_argument("--out", dest="path", default=str(DEFAULT_PATH))
    seed.add_argument("--add", action="append", default=[], help="a player I picked up since")
    seed.add_argument("--drop", action="append", default=[], help="a player I dropped since")
    seed.add_argument(
        "--taken", action="append", default=[], help="a player someone else picked up"
    )
    seed.add_argument(
        "--injured",
        action="append",
        default=[],
        metavar="NAME[=YYYY-MM-DD]",
        help="one of mine who is out, with his expected return date if there is one",
    )
    fix = sub.add_parser("resolve", help="place names that now resolve, keeping everything else")
    fix.add_argument("--board", default="artifacts/board_v3")
    fix.add_argument("--goalies", default="artifacts/goalies_v2")
    fix.add_argument("--out", dest="path", default=str(DEFAULT_PATH))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import pandas as pd

    from hockey import extras as extras_mod
    from hockey.serve import board as board_mod
    from hockey.serve.identity import Resolver

    data = board_mod.load(Path(args.board), Path(args.goalies) if args.goalies else None)
    known = data.players[["player", "player_id", "slot"]]
    sheet = extras_mod.load()
    if sheet is not None:
        outside = sheet[~sheet["nhl_id"].isin(known["player_id"])]
        known = pd.concat(
            [
                known,
                pd.DataFrame(
                    {
                        "player": outside["player"],
                        "player_id": outside["nhl_id"],
                        "slot": outside["eligible"].map(lambda e: e[0]),
                    }
                ),
            ],
            ignore_index=True,
        )
    resolve = Resolver(known).resolve
    if args.command == "resolve":
        before = load(Path(args.path))
        n = len(before.unresolved)
        saved = save(resolve_unresolved(before, resolve), Path(args.path))
        logger.info(
            "placed %d of %d names; %d still unresolved: %s",
            n - len(saved.unresolved),
            n,
            len(saved.unresolved),
            ", ".join(u.split(" (")[0] for u in saved.unresolved) or "none",
        )
        return
    with open(args.draft, encoding="utf-8") as f:
        league = seed_from_draft(list(csv.DictReader(f)), args.me, resolve)
    for name in args.add:
        # A pickup the model has never seen (a rookie, a third goalie) is still
        # mine; it is recorded by name, like a draft pick that did not resolve.
        hit = resolve(name, None)
        if hit.ok:
            league.mine.append(int(hit.player_id))
        else:
            league.unresolved.append(f"{name} ({args.me}, added): {hit.method}")
    for name in args.drop:
        pid = _resolve_or_fail(resolve, name)
        league.mine = [i for i in league.mine if i != pid]
    for name in args.taken:
        league.taken.append(_resolve_or_fail(resolve, name))
    for spec in args.injured:
        name, _, back = spec.partition("=")
        league.out[_resolve_or_fail(resolve, name)] = back or None
    saved = save(league, Path(args.path))
    logger.info(
        "wrote %s: %d mine, %d taken, %d out, %d unresolved",
        args.path,
        len(saved.mine),
        len(saved.taken),
        len(saved.out),
        len(saved.unresolved),
    )
    for miss in saved.unresolved:
        logger.info("  not on the board: %s", miss)


if __name__ == "__main__":
    main()
