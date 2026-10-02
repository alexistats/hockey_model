"""In-season helpers for the hockey-gm skill.

Everything here reads what the draft page reads - the built page's payload, the
league file, the season export - and prices a move the way the Season tab does
(`hockey.serve.schedule`: each night's lineup set as a manager sets it, goalie
starts bent by back-to-backs). Run from the project root with the venv's Python:

    gm.py stats NAME [NAME ...]              last two seasons and this one, in league scoring
    gm.py fits NAME [--drop NAME] [--weeks 3]     lineup games he would fill, week by week
    gm.py swap --drop A --add B [--drop C --add D]   a move priced this week, next, season
    gm.py scan [--keep NAME ...] [--extras]  every free agent against my roster
    gm.py resolve NAME [--team ABBR]         a name to an NHL id, or a miss - never a guess
    gm.py record LOG.txt [--apply]           a pasted Yahoo transaction log into the league file

Numbers are the model's rates, updated by this season's games once there are
some. A player on the extras sheet is valued at replacement level - a
placeholder - so his model number means nothing; read his stats and his role.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PAGE = ROOT / "artifacts/ui/draft_board.html"
LEAGUE = ROOT / "artifacts/season/league.json"
STATS = ROOT / "artifacts/season/stats.json"
ROS = ROOT / "artifacts/season/ros.json"
ROSTERS = ROOT / "artifacts/season/rosters.json"
EXTRAS = ROOT / "config/extra_players_2026.csv"
BUILD = [
    sys.executable,
    "scripts/build_draft_ui.py",
    "artifacts/board_v3",
    "--goalies",
    "artifacts/goalies_v2",
]
YAHOO_TO_NHL = {"SJ": "SJS", "NJ": "NJD", "TB": "TBL", "LA": "LAK"}


def fold(name: str) -> str:
    from hockey.serve.identity import fold as f

    return f(name)


class Ctx:
    """The page, the league and the season export, loaded once."""

    def __init__(self):
        from hockey.serve import season

        html = PAGE.read_text(encoding="utf-8")
        found = re.search(r'<script[^>]*id="payload"[^>]*>(.*?)</script>', html, re.S)
        if not found:
            raise SystemExit(f"no payload in {PAGE}; rebuild it: {' '.join(BUILD[1:])}")
        self.data = json.loads(found.group(1))
        self.days = self.data["calendar"]["days"]
        self.cal = self.data["calendar"]["teams"]
        self.shape = self.data["rosterShape"]
        self.games = self.data["gamesInSeason"]
        self.weeks = self.data["weeks"]
        self.players = self.data["players"]
        self.by_id = {p["id"]: p for p in self.players}
        self.league = season.load(LEAGUE)
        self.stats = (
            json.loads(STATS.read_text(encoding="utf-8")).get("players", {})
            if STATS.exists()
            else {}
        )
        self.ros = (
            json.loads(ROS.read_text(encoding="utf-8")).get("players", {}) if ROS.exists() else {}
        )
        today = date.today().isoformat()
        self.first = next((i for i, d in enumerate(self.days) if d >= today), len(self.days) - 1)
        self.end = len(self.days) - 1

    # --- players -----------------------------------------------------------------
    def find(self, name: str) -> dict:
        hits = [p for p in self.players if fold(p["n"]) == fold(name)]
        if len(hits) != 1:
            hint = "" if hits else " - resolve it (gm.py resolve) and add him to the extras sheet"
            raise SystemExit(f"{name!r}: {len(hits)} players of that name on the page{hint}")
        return hits[0]

    def team(self, p: dict) -> str:
        return (self.stats.get(str(p["id"])) or {}).get("team") or p["t"]

    @staticmethod
    def is_goalie(p: dict) -> bool:
        return p.get("gcat") is not None

    def elig(self, p: dict) -> list[str]:
        return ["G"] if self.is_goalie(p) else (p["e"] or [p["p"]])

    def entry(self, p: dict) -> dict:
        """A player as the Season tab's lineup code takes him (seasonEntry)."""
        from hockey.serve.schedule import start_chances

        team = self.team(p)
        r = self.ros.get(str(p["id"]))
        updated = r is not None and (r.get("seen") or 0) > 0
        e = {
            "id": p["id"],
            "team": team,
            "rate": r["rate"] if updated else p["m"] / max(p["g"], 1e-6),
        }
        if p["id"] in self.league.out:
            back = self.league.out[p["id"]]
            k = (
                None
                if back is None
                else next((i for i, d in enumerate(self.days) if d >= back), None)
            )
            e["from"] = 10**9 if k is None else k
        if self.is_goalie(p):
            share = (
                r["share"]
                if updated and r.get("share") is not None
                else min(1.0, p["g"] / self.games)
            )
            e.update(
                elig=["G"],
                share=share,
                chance=start_chances(share, self.cal.get(team, []), lambda d: self.days[d]),
            )
        else:
            e["elig"] = self.elig(p)
        return e

    def mine(self) -> list[dict]:
        missing = [i for i in self.league.mine if i not in self.by_id]
        if missing:
            print(
                f"warning: {len(missing)} of my players are not on the page: {missing}",
                file=sys.stderr,
            )
        return [self.by_id[i] for i in self.league.mine if i in self.by_id]

    # --- windows ------------------------------------------------------------------
    def week(self, k: int) -> tuple[int, int, dict]:
        """Days of the fantasy week `k` after the current one (0 = this week, from today)."""
        now = self.days[self.first]
        w0 = next((j for j, w in enumerate(self.weeks) if w["end"] >= now), len(self.weeks) - 1)
        w = self.weeks[min(w0 + k, len(self.weeks) - 1)]
        inside = [
            i for i, d in enumerate(self.days) if w["start"] <= d <= w["end"] and i >= self.first
        ]
        return (inside[0], inside[-1], w) if inside else (self.first, self.first - 1, w)

    def total(self, entries: list[dict], first: int, last: int) -> float:
        from hockey.serve.schedule import window_points

        return (
            sum(window_points(entries, self.cal, self.shape, first, last).values())
            if last >= first
            else 0.0
        )

    def fits(self, cand: dict, roster: list[dict], first: int, last: int) -> list[str]:
        """His team's nights in the window on which my lineup has a free slot he fits."""
        from hockey.serve.schedule import lineup

        shape = {k: v for k, v in self.shape.items() if k != "G"}
        entries = {p["id"]: self.entry(p) for p in roster}

        def plays(p, d):
            e = entries[p["id"]]
            return d in set(self.cal.get(e["team"], [])) and not (
                e.get("from") is not None and d < e["from"]
            )

        out = []
        for d in sorted(x for x in self.cal.get(self.team(cand), []) if first <= x <= last):
            tonight = [
                (p["id"], 0.0, tuple(self.elig(p)))
                for p in roster
                if not self.is_goalie(p) and plays(p, d)
            ]
            if len(lineup([*tonight, (-1, -1.0, tuple(self.elig(cand)))], shape)) > len(
                lineup(tonight, shape)
            ):
                out.append(self.days[d][5:])
        return out


# --- identity ---------------------------------------------------------------------
def resolve(ctx: Ctx | None, name: str, team: str | None = None):
    """(nhl id, full name, where) or a SystemExit naming the miss. The page first
    (board or extras), then the warehouse; a team, when given, must agree with
    the player's current NHL roster."""
    from sqlalchemy import text

    from hockey.db import SessionLocal

    team = YAHOO_TO_NHL.get(team, team) if team else None
    rosters = (
        {int(k): v for k, v in json.loads(ROSTERS.read_text(encoding="utf-8"))["teams"].items()}
        if ROSTERS.exists()
        else {}
    )
    if ctx is not None:
        on_page = [p for p in ctx.players if fold(p["n"]) == fold(name)]
        if len(on_page) == 1:
            p = on_page[0]
            now = rosters.get(p["id"])
            if team and now and now != team:
                raise SystemExit(
                    f"{name}: the page has id {p['id']}, whose NHL roster is {now}, not {team}"
                )
            return p["id"], p["n"], "extras" if p.get("x") else "board"
        if len(on_page) > 1:
            raise SystemExit(f"{name}: {len(on_page)} players of that name on the page")
    with SessionLocal() as s:
        rows = s.execute(text("SELECT nhl_id, first_name || ' ' || last_name FROM players")).all()
    hits = [(int(i), n) for i, n in rows if fold(n) == fold(name)]
    if len(hits) > 1 and team:
        hits = [h for h in hits if rosters.get(h[0]) == team] or hits
    if len(hits) != 1:
        raise SystemExit(f"{name}: {len(hits)} warehouse matches - not recorded; check the name")
    pid, full = hits[0]
    now = rosters.get(pid)
    if team and now and now != team:
        raise SystemExit(
            f"{name}: warehouse id {pid} is on {now}'s roster, not {team} - not recorded"
        )
    return pid, full, "warehouse"


# --- commands ---------------------------------------------------------------------
def label(season: int) -> str:
    """20252026 -> 25-26."""
    return f"{str(season)[2:4]}-{str(season)[6:]}"


def cmd_stats(args):
    from sqlalchemy import text

    from hockey.db import SessionLocal
    from hockey.seasons import PROJECTION_SEASON
    from hockey.yahoo.settings import load_scoring_from_yaml

    rules = load_scoring_from_yaml()
    sk = {r.key: r.modifier for r in rules.skater_rules}
    gk = {r.key: r.modifier for r in rules.goalie_rules}
    seasons = [PROJECTION_SEASON - 20002, PROJECTION_SEASON - 10001, PROJECTION_SEASON]
    ctx = Ctx()
    with SessionLocal() as s:
        for name in args.names:
            pid, full, where = resolve(ctx, name)
            lines = []
            for season in seasons:
                r = s.execute(
                    text("""
                    SELECT count(*), sum(goals), sum(assists), sum(plus_minus), sum(ppp),
                           sum(shp), sum(sog),
                           sum(hits), sum(blocks), avg(toi_seconds) / 60.0
                      FROM skater_game_logs l JOIN nhl_games g ON g.nhl_game_id = l.game_id
                     WHERE g.game_type = 2 AND g.season = :s AND l.player_id = :p"""),
                    {"s": season, "p": pid},
                ).one()
                if r[0]:
                    gp, g, a, pm, ppp, shp, sog, hits, blk, toi = [x or 0 for x in r]
                    fp = (
                        sk["goals"] * g
                        + sk["assists"] * a
                        + sk["plus_minus"] * pm
                        + sk["ppp"] * ppp
                        + sk["shp"] * shp
                        + sk["sog"] * sog
                        + sk["hits"] * hits
                        + sk["blocks"] * blk
                    )
                    lines.append(
                        f"{label(season)}: {gp} gp, {g}g {a}a, {ppp} ppp, {sog} sog, "
                        f"{hits} hits, {blk} blk, {int(pm):+d}, {float(toi):.1f} min "
                        f"-> {fp / gp:.2f}/g"
                    )
                    continue
                r = s.execute(
                    text("""
                    SELECT sum(started::int), sum(wins), sum(goals_against), sum(saves),
                           sum(coalesce(shutouts, 0))
                      FROM goalie_game_logs l JOIN nhl_games g ON g.nhl_game_id = l.game_id
                     WHERE g.game_type = 2 AND g.season = :s AND l.player_id = :p"""),
                    {"s": season, "p": pid},
                ).one()
                if r[0]:
                    gs, w, ga, sv, so = [x or 0 for x in r]
                    fp = (
                        gk["games_started"] * gs
                        + gk["wins"] * w
                        + gk["goals_against"] * ga
                        + gk["saves"] * sv
                        + gk["shutouts"] * so
                    )
                    lines.append(
                        f"{label(season)}: {gs} starts, {w}W, "
                        f"sv .{int(1000 * sv / max(sv + ga, 1)):03d}, "
                        f"{so} SO -> {fp / max(gs, 1):.2f}/start"
                    )
            p = ctx.by_id.get(pid)
            status = (
                (
                    "mine"
                    if pid in ctx.league.mine
                    else "taken"
                    if pid in ctx.league.taken
                    else "free"
                )
                if p
                else "not on the page"
            )
            model = (
                ""
                if p is None
                else (
                    " | page: replacement-level placeholder"
                    if p.get("x")
                    else f" | model {p['m'] / max(p['g'], 1e-6):.2f}"
                    f"/{'start' if Ctx.is_goalie(p) else 'g'}"
                )
            )
            print(f"{full} ({ctx.team(p) if p else '?'}, {status}{model})")
            for line in lines or ["  no NHL games in these seasons"]:
                print(f"  {line}")


def cmd_fits(args):
    ctx = Ctx()
    cand = ctx.find(args.name)
    drop = {ctx.find(n)["id"] for n in (args.drop or [])}
    roster = [p for p in ctx.mine() if p["id"] not in drop and p["id"] != cand["id"]]
    if Ctx.is_goalie(cand):
        e = ctx.entry(cand)
        for k in range(args.weeks):
            first, last, w = ctx.week(k)
            nights = [d for d in ctx.cal.get(e["team"], []) if first <= d <= last]
            starts = sum(e["chance"].get(d, e["share"]) for d in nights)
            print(
                f"week {w['week']} ({w['start']}..{w['end']}): {len(nights)} games, "
                f"{starts:.1f} expected starts"
            )
        return
    for k in range(args.weeks):
        first, last, w = ctx.week(k)
        games = [d for d in ctx.cal.get(ctx.team(cand), []) if first <= d <= last]
        fit = ctx.fits(cand, roster, first, last)
        print(
            f"week {w['week']} ({w['start']}..{w['end']}): {len(fit)} of {len(games)} "
            f"({', '.join(fit)})"
        )
    season = ctx.fits(cand, roster, ctx.first, ctx.end)
    total = [d for d in ctx.cal.get(ctx.team(cand), []) if d >= ctx.first]
    print(f"rest of season: {len(season)} of {len(total)}")


def cmd_swap(args):
    ctx = Ctx()
    if len(args.drop or []) != len(args.add or []):
        raise SystemExit("give one --add for each --drop")
    drops = [ctx.find(n) for n in args.drop]
    adds = [ctx.find(n) for n in args.add]
    before = [ctx.entry(p) for p in ctx.mine()]
    after = [e for e in before if e["id"] not in {p["id"] for p in drops}] + [
        ctx.entry(p) for p in adds
    ]
    spans = [ctx.week(k) for k in range(3)] + [(ctx.first, ctx.end, {"week": "rest of season"})]
    parts = []
    for first, last, w in spans:
        delta = ctx.total(after, first, last) - ctx.total(before, first, last)
        parts.append(
            f"{'rest of season' if w['week'] == 'rest of season' else 'week ' + str(w['week'])} "
            f"{delta:+.1f}"
        )
    flags = [p["n"] for p in adds if p.get("x")]
    print(
        f"drop {', '.join(p['n'] for p in drops)} / add {', '.join(p['n'] for p in adds)}: "
        + ", ".join(parts)
    )
    if flags:
        print(
            f"note: {', '.join(flags)} valued as a replacement-level placeholder; "
            "judge him on stats and role"
        )


def cmd_scan(args):
    ctx = Ctx()
    keep = {ctx.find(n)["id"] for n in (args.keep or [])}
    mine = ctx.mine()
    roster = [ctx.entry(p) for p in mine]
    droppable = [e for e in roster if e["id"] not in keep and e["id"] not in ctx.league.out]
    held = set(ctx.league.mine) | set(ctx.league.taken)
    pool = [
        ctx.entry(p) for p in ctx.players if p["id"] not in held and (args.extras or not p.get("x"))
    ]
    w1, w2 = ctx.week(0), ctx.week(1)
    base = {k: ctx.total(roster, s[0], s[1]) for k, s in (("w1", w1), ("w2", w2))}
    rows = []
    for d in droppable:
        rest = [e for e in roster if e["id"] != d["id"]]
        for f in pool:
            after = rest + [f]
            rows.append(
                (
                    d["id"],
                    f["id"],
                    ctx.total(after, w1[0], w1[1]) - base["w1"],
                    ctx.total(after, w2[0], w2[1]) - base["w2"],
                )
            )
    season_base = ctx.total(roster, ctx.first, ctx.end)
    season = {}
    for d, f, _, _ in sorted(rows, key=lambda r: -(r[2] + r[3]))[: args.examine]:
        rest = [e for e in roster if e["id"] != d]
        season[(d, f)] = (
            ctx.total(rest + [next(x for x in pool if x["id"] == f)], ctx.first, ctx.end)
            - season_base
        )
    name = lambda i: ctx.by_id[i]["n"]  # noqa: E731

    def show(title, key, need_season=True):
        print(f"\n{title}")
        n = 0
        for d, f, a, b in sorted(rows, key=key):
            s = season.get((d, f))
            if need_season and (s is None or s < 0):
                continue
            print(
                f"  drop {name(d):20s} add {name(f):20s} {'/'.join(ctx.elig(ctx.by_id[f])):7s} "
                f"{ctx.team(ctx.by_id[f]):4s} this wk {a:+5.1f}  next wk {b:+5.1f}"
                + (f"  season {s:+5.0f}" if s is not None else "")
            )
            n += 1
            if n == args.top:
                break

    show(
        f"Best this week ({w1[2]['start']}..{w1[2]['end']}), not costing the season",
        lambda r: -r[2],
    )
    show(
        f"Best next week ({w2[2]['start']}..{w2[2]['end']}), not costing the season",
        lambda r: -r[3],
    )
    print("\nBest over the season (among swaps that help in the next two weeks)")
    for (d, f), s in sorted(season.items(), key=lambda kv: -kv[1])[: args.top]:
        print(f"  drop {name(d):20s} add {name(f):20s} season {s:+5.0f}")


# --- the transaction log ----------------------------------------------------------
PLAYER = re.compile(
    r"^(?P<name>.+?) (?P<team>[A-Z]{2,3}) - (?P<pos>(?:C|LW|RW|D|G)(?:,(?:C|LW|RW|D|G))*)"
    r"(?: (?P<status>[A-Z+-]{1,6}))?$"  # IR, IR+, DTD, O, NA, IR-LT
)
ACTIONS = {"Free Agent": "add", "Waiver": "add", "To Waivers": "drop", "To Free Agent": "drop"}
STAMP = re.compile(
    r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{1,2}, \d{1,2}:\d{2} (am|pm)$"
)


def parse_log(text: str):
    """Yahoo's transaction log as pasted from the web page: blocks of moves, each
    closed by the manager's team and a time. Returns (blocks oldest first, problems)."""
    blocks, moves, manager, pending, problems = [], [], None, None, []
    for raw in text.splitlines():
        line = raw.strip()
        if (
            not line
            or line == "logo"
            or line.startswith("Previous 25")
            or line.startswith("Next 25")
        ):
            continue
        m = PLAYER.match(line)
        if m:
            if pending is not None:
                problems.append(f"{pending['name']}: no add or drop after him")
            pending = m.groupdict()
            continue
        if line in ACTIONS:
            if pending is None:
                problems.append(f"'{line}' with no player before it")
            else:
                moves.append({**pending, "kind": ACTIONS[line]})
                pending = None
            continue
        if STAMP.match(line):
            if pending is not None:
                problems.append(f"{pending['name']}: no add or drop after him")
                pending = None
            blocks.append({"moves": moves, "manager": manager, "when": line})
            moves, manager = [], None
            continue
        if pending is not None:
            problems.append(
                f"{pending['name']}: unrecognised move '{line}' (a trade?) - not recorded"
            )
            pending = None
            continue
        manager = line
    return list(reversed(blocks)), problems


def cmd_record(args):
    from hockey.serve import season

    ctx = Ctx()
    blocks, problems = parse_log(Path(args.log).read_text(encoding="utf-8"))
    me = (ctx.league.me or "").replace("’", "'")
    names, state = {}, {}
    for b in blocks:
        mine = (b["manager"] or "").replace("’", "'") == me
        for mv in b["moves"]:
            names.setdefault(mv["name"], mv)
            state[mv["name"]] = ("mine" if mine else "taken") if mv["kind"] == "add" else "free"
    resolved = {}
    for n, mv in names.items():
        try:
            resolved[n] = resolve(ctx, n, mv["team"])
        except SystemExit as e:
            problems.append(str(e))
    print(f"{len(blocks)} transactions, {len(names)} players, {len(resolved)} resolved")
    changes = []
    for n in sorted(resolved, key=lambda x: (state[x], x)):
        pid, full, where = resolved[n]
        had = "mine" if pid in ctx.league.mine else "taken" if pid in ctx.league.taken else "free"
        mark = "" if had == state[n] else "  <- changes"
        if mark:
            changes.append(n)
        print(f"  {full:24s} {pid:>8d}  {where:9s} now {state[n]:5s} file {had:5s}{mark}")
    if problems:
        print("\nnot recorded:", *problems, sep="\n  ")
    if not args.apply:
        print(
            "\ndry run; add --apply to write the extras sheet, rebuild the page "
            "and update the league file"
        )
        return
    new = [n for n in resolved if resolved[n][2] == "warehouse"]
    if new:
        rows = list(csv.DictReader(EXTRAS.open(encoding="utf-8", newline="")))
        have = {int(r["nhl_id"]) for r in rows}
        for n in new:
            pid, full, _ = resolved[n]
            if pid in have:
                continue
            mv = names[n]
            rows.append(
                {
                    "nhl_id": str(pid),
                    "player": full,
                    "team": YAHOO_TO_NHL.get(mv["team"], mv["team"]),
                    "positions": mv["pos"].replace(",", "|"),
                    "games": "",
                    "note": f"From the league's transaction log, {date.today().isoformat()}; "
                    "positions as Yahoo lists them",
                }
            )
        with EXTRAS.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["nhl_id", "player", "team", "positions", "games", "note"],
                lineterminator="\n",
            )
            w.writeheader()
            w.writerows(rows)
        print(f"extras sheet: added {', '.join(resolved[n][1] for n in new)}; rebuilding the page")
        subprocess.run(BUILD, cwd=ROOT, check=True)
        ctx = Ctx()
    lg = season.load(LEAGUE)
    ids = {resolved[n][0]: state[n] for n in resolved}
    missing = [i for i in ids if i not in ctx.by_id]
    if missing:
        raise SystemExit(f"ids the page does not know: {missing}; not written")
    lg.mine = [i for i in lg.mine if ids.get(i, "mine") == "mine"] + [
        i for i, s in ids.items() if s == "mine" and i not in lg.mine
    ]
    lg.taken = [i for i in lg.taken if ids.get(i, "taken") == "taken"] + [
        i for i, s in ids.items() if s == "taken" and i not in lg.taken
    ]
    saved = season.save(lg, LEAGUE)
    print(f"league file: {len(saved.mine)} mine, {len(saved.taken)} taken, {len(changes)} changed")


def cmd_resolve(args):
    pid, full, where = resolve(Ctx(), args.name, args.team)
    print(f"{full}: {pid} ({where})")


def main():
    # Names like Stützle, whatever launched us (a scheduled task, Git Bash, a pipe).
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(prog="gm.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("stats")
    p.add_argument("names", nargs="+")
    p.set_defaults(fn=cmd_stats)  # noqa: E702
    p = sub.add_parser("fits")
    p.add_argument("name")
    p.add_argument("--drop", action="append")  # noqa: E702
    p.add_argument("--weeks", type=int, default=3)
    p.set_defaults(fn=cmd_fits)  # noqa: E702
    p = sub.add_parser("swap")
    p.add_argument("--drop", action="append")
    p.add_argument("--add", action="append")  # noqa: E702
    p.set_defaults(fn=cmd_swap)
    p = sub.add_parser("scan")
    p.add_argument("--keep", action="append")
    p.add_argument("--extras", action="store_true")  # noqa: E702
    p.add_argument("--top", type=int, default=8)
    p.add_argument("--examine", type=int, default=250)
    p.set_defaults(fn=cmd_scan)  # noqa: E702
    p = sub.add_parser("resolve")
    p.add_argument("name")
    p.add_argument("--team")
    p.set_defaults(fn=cmd_resolve)  # noqa: E702
    p = sub.add_parser("record")
    p.add_argument("log")
    p.add_argument("--apply", action="store_true")
    p.set_defaults(fn=cmd_record)  # noqa: E702
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
