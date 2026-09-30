"""Catch up on gamedaytweets.com since the last read, and flag what touches my team.

    news.py              new tweets since the bookmark, flagged ones first; moves the bookmark
    news.py --dry-run    the same, bookmark left where it was
    news.py --all        print the unflagged ones too
    news.py --pages 10   how far back to page on a first run or after a long gap

The site is server-rendered: each item is an embedded tweet whose status id only
ever grows, so the bookmark is the highest id read (artifacts/season/news_state.json)
and a catch-up reads pages until it meets it. Every new tweet is archived to
artifacts/season/news/YYYY-MM-DD.jsonl. robots.txt allows the pages read here.

A tweet is flagged when it names one of my players or someone on the watch list
(artifacts/season/watchlist.json), or when it is about one of my players' teams
and reads like lines, power play, a goalie start, an injury or a transaction.
Flags are a filter, not a verdict: read them, and say what changes a plan.
"""

from __future__ import annotations

import argparse
import html as H
import json
import re
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[3]
PAGE = ROOT / "artifacts/ui/draft_board.html"
LEAGUE = ROOT / "artifacts/season/league.json"
STATS = ROOT / "artifacts/season/stats.json"
STATE = ROOT / "artifacts/season/news_state.json"
WATCH = ROOT / "artifacts/season/watchlist.json"
ARCHIVE = ROOT / "artifacts/season/news"
URL = "https://www.gamedaytweets.com/?page={}"

TEAMS = {
    "ANA": ["Ducks", "FlyTogether"],
    "BOS": ["Bruins", "NHLBruins"],
    "BUF": ["Sabres", "LetsGoBuffalo", "SabreHood"],
    "CAR": ["Hurricanes", "Canes", "LetsGoCanes"],
    "CBJ": ["Blue Jackets", "CBJ"],
    "CGY": ["Flames"],
    "CHI": ["Blackhawks", "Hawks"],
    "COL": ["Avalanche", "Avs", "GoAvsGo"],
    "DAL": ["Stars", "TexasHockey"],
    "DET": ["Red Wings", "LGRW"],
    "EDM": ["Oilers", "LetsGoOilers"],
    "FLA": ["Panthers", "TimeToHunt"],
    "LAK": ["Kings", "GoKingsGo"],
    "MIN": ["Wild", "mnwild"],
    "MTL": ["Canadiens", "Habs", "GoHabsGo"],
    "NJD": ["Devils", "NJDevils"],
    "NSH": ["Predators", "Preds", "Smashville"],
    "NYI": ["Islanders", "Isles"],
    "NYR": ["Rangers", "NYR"],
    "OTT": ["Senators", "Sens", "GoSensGo"],
    "PHI": ["Flyers", "LetsGoFlyers"],
    "PIT": ["Penguins", "Pens", "LetsGoPens"],
    "SJS": ["Sharks", "SJSharks"],
    "SEA": ["Kraken", "SeaKraken"],
    "STL": ["Blues", "stlblues"],
    "TBL": ["Lightning", "Bolts", "GoBolts"],
    "TOR": ["Maple Leafs", "Leafs", "LeafsForever"],
    "UTA": ["Mammoth", "Utah"],
    "VAN": ["Canucks"],
    "VGK": ["Golden Knights", "VegasBorn"],
    "WPG": ["Jets", "GoJetsGo"],
    "WSH": ["Capitals", "Caps", "ALLCAPS"],
}
KINDS = {
    "injury": r"injur|\bIR\b|LTIR|day-to-day|\bDTD\b|week-to-week|lower-body|upper-body|"
    r"surgery|won't play|will not play|questionable|doubtful|ruled out|"
    r"\b(is|are|will be|remains?) out\b|out (for|with|tonight|indefinitely)|"
    r"scratch|illness|non-contact|return(s|ed|ing)? to|activated",
    "goalie": r"\bstart(s|ing|er)?\b|in net|in goal|the crease|\bG\b.*\bstart|backup",
    "power play": r"\bPP ?[12]\b|power ?play|PP units|PP groups|man advantage",
    "lines": r"\blines?\b|line rushes|rushes|top six|promot|moved up|bumped up|demot|"
    r"practice groups|skating with",
    "transaction": r"waiv|recall|assign|loaned|\bsign(ed|s)?\b|trade|claimed|called up|sent down",
}


def fold(name: str) -> str:
    from hockey.serve.identity import fold as f

    return f(name)


def tweets(page_html: str) -> list[dict]:
    out = []
    for block in re.findall(r"<blockquote.*?</blockquote>", page_html, re.S):
        ids = [int(i) for i in re.findall(r"/status/(\d+)", block)]
        if not ids:
            continue
        text = re.sub(r"\s+", " ", H.unescape(re.sub(r"<[^>]+>", " ", block))).strip()
        out.append({"id": max(ids), "text": text, "url": f"https://x.com/i/status/{max(ids)}"})
    return out


def watch_index():
    """Names to flag: my roster and the watch list, with each player's team."""
    data = json.loads(
        re.search(
            r'<script[^>]*id="payload"[^>]*>(.*?)</script>', PAGE.read_text(encoding="utf-8"), re.S
        ).group(1)
    )
    by_id = {p["id"]: p for p in data["players"]}
    stats = (
        json.loads(STATS.read_text(encoding="utf-8")).get("players", {}) if STATS.exists() else {}
    )
    league = json.loads(LEAGUE.read_text(encoding="utf-8"))
    watch = (
        json.loads(WATCH.read_text(encoding="utf-8")).get("players", []) if WATCH.exists() else []
    )
    last_counts = {}
    for p in data["players"]:
        last = fold(p["n"]).split(" ")[-1]
        last_counts[last] = last_counts.get(last, 0) + 1
    people = []
    for i in league["mine"]:
        if i in by_id:
            p = by_id[i]
            people.append((p["n"], (stats.get(str(i)) or {}).get("team") or p["t"], "mine"))
    names = {fold(p["n"]): p for p in data["players"]}
    for n in watch:
        p = names.get(fold(n))
        people.append(
            (n, ((stats.get(str(p["id"])) or {}).get("team") or p["t"]) if p else None, "watch")
        )
    return people, last_counts, {t for _, t, why in people if why == "mine" and t}


def flag(t: dict, people, last_counts, my_teams) -> tuple[list[str], list[str]]:
    text = t["text"]
    folded = fold(text)
    hits = []
    for name, team, why in people:
        full = fold(name)
        last = full.split(" ")[-1]
        team_named = team and any(
            re.search(rf"\b{re.escape(w)}\b", text, re.I) for w in TEAMS.get(team, [])
        )
        if re.search(rf"\b{re.escape(full)}\b", folded) or (
            re.search(rf"\b{re.escape(last)}\b", folded)
            and (last_counts.get(last, 0) <= 1 or team_named)
        ):
            hits.append(f"{name}{' (watch)' if why == 'watch' else ''}")
    kinds = [k for k, rx in KINDS.items() if re.search(rx, text, re.I)]
    teams = [
        t
        for t in sorted(my_teams)
        if any(re.search(rf"#?\b{re.escape(w)}\b", text, re.I) for w in TEAMS.get(t, []))
    ]
    if not hits and teams and kinds:
        hits = [f"team: {', '.join(teams)}"]
    return hits, kinds


def main():
    # Names like Stützle, whatever launched us (a scheduled task, Git Bash, a pipe).
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(prog="news.py")
    ap.add_argument("--pages", type=int, default=6, help="most pages to read back")
    ap.add_argument("--dry-run", action="store_true", help="leave the bookmark where it is")
    ap.add_argument("--all", action="store_true", help="print unflagged tweets too")
    args = ap.parse_args()

    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    last_id = int(state.get("last_id", 0))
    new, reached = [], False
    with httpx.Client(
        headers={"User-Agent": "Mozilla/5.0 (hockey_models personal use)"},
        timeout=30,
        follow_redirects=True,
    ) as client:
        for page in range(1, args.pages + 1):
            r = client.get(URL.format(page))
            r.raise_for_status()
            items = tweets(r.text)
            if not items:
                break
            fresh = [t for t in items if t["id"] > last_id]
            new.extend(fresh)
            if len(fresh) < len(items):  # this page reaches back to what was read last time
                reached = True
                break
    seen, unique = set(), []
    for t in sorted(new, key=lambda t: t["id"]):
        if t["id"] not in seen:
            seen.add(t["id"])
            unique.append(t)
    if last_id and not reached and unique:
        print(
            f"note: read {args.pages} pages without reaching the last tweet read; "
            "older ones may be missed "
            f"(rerun with --pages {args.pages * 2})"
        )
    people, last_counts, my_teams = watch_index()
    flagged, rest = [], []
    for t in unique:
        hits, kinds = flag(t, people, last_counts, my_teams)
        (flagged if hits else rest).append((t, hits, kinds))
    print(
        f"{len(unique)} new tweets since {state.get('read_at', 'the start')}; "
        f"{len(flagged)} flagged"
    )
    for t, hits, kinds in flagged:
        print(
            f"\n[{', '.join(kinds) or 'news'}] {' / '.join(hits)}"
            f"\n  {t['text'][:400]}\n  {t['url']}"
        )
    if args.all:
        print("\n--- unflagged")
        for t, _, kinds in rest:
            print(f"- [{', '.join(kinds) or 'news'}] {t['text'][:200]}")
    if unique:
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        with (ARCHIVE / f"{date.today().isoformat()}.jsonl").open("a", encoding="utf-8") as f:
            for t, hits, kinds in flagged + rest:
                f.write(json.dumps({**t, "flags": hits, "kinds": kinds}) + "\n")
    if not args.dry_run and unique:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(
            json.dumps(
                {
                    "last_id": max(t["id"] for t in unique),
                    "read_at": datetime.now(UTC).isoformat(timespec="seconds"),
                }
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    sys.exit(main())
