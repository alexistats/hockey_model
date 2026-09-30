---
name: hockey-gm
description: In-season manager for the user's Yahoo fantasy hockey team (league 2270, this repo). Add/drop, stream and trade analysis in the league's scoring; recording roster moves and pasted Yahoo transaction logs in the league file; rebuilding the draft page and running its server; the morning and evening gamedaytweets.com news catch-up; and the decision log. Use for any question about adds, drops, streams, trades, lineups, goalie starts, injuries or news for this team, or when the user pastes a Yahoo transaction log.
---

# hockey-gm: running the team in season

The draft page's Season tab (served at localhost:8899/ui) is the tool; this skill
is how to use it and the data around it day to day. Everything league-specific
that is private (the league file, plans, decision log, news archive, watch list)
lives under `artifacts/`, which git ignores. **This repo is public: keep other
managers' names out of anything committed** (this folder, `config/`, docs).

## The league

- Yahoo league 2270, 14 teams, one-year league (no keepers). My team: `me` in the
  league file. **3 adds a week.**
- Lineup: C 2, LW 2, RW 2, D 4, G 2, set daily; 5 bench; IR slots.
- Scoring (from `config/league_2270.yaml`, never hard-code): skaters G 5, A 3, +/-
  0.5, PPP 0.5, SHP 1.5, SOG 0.5, **HIT 0.5, BLK 0.5**; goalies GS 1, W 6, GA −1.5,
  SV 0.3, SHO 4. Hits and blocks make grinders and shot-blocking defensemen
  valuable here; a points-only player with no hits is worth less than his points.

## Files

| What | Where |
|---|---|
| My roster, everyone else's, who's out (ids only) | `artifacts/season/league.json` |
| Players outside the model, at replacement level | `config/extra_players_2026.csv` |
| The built page (payload: players, calendar, weeks) | `artifacts/ui/draft_board.html` |
| Season so far / rest-of-season rates | `artifacts/season/stats.json`, `ros.json` |
| Current NHL rosters (from the refresh) | `artifacts/season/rosters.json` |
| Weekly plans and the decision log | `artifacts/season/plans/` (`decisions.md`) |
| News bookmark, archive, watch list | `artifacts/season/news_state.json`, `news/`, `watchlist.json` |

## How the user wants to work

- **Evidence, then a clear recommendation.** Numbers from the tools below plus
  news with source links. Say what changes a plan; don't recite everything.
- **Short lists, narrow tables.** Wide tables get cut off in the user's terminal.
- **Keep runs short.** The user interrupts long computations; the helpers here run
  in seconds. Warn before anything that takes minutes.
- **One dedicated stream spot**; the rest of the roster are holds. Never suggest
  dropping a core player for a one-week stream. Fading a week for real upside is
  fine when the numbers are close.
- **Never guess an identity.** Names become NHL ids by exact match, with the team
  checked against current rosters; a miss is reported, not filled in.
- **Log decisions.** Every move or plan change gets an entry in
  `artifacts/season/plans/decisions.md` (below).

## Tools (`.claude/skills/hockey-gm/`)

Run from the project root with `.venv/Scripts/python.exe` (and
`PYTHONIOENCODING=utf-8` in Git Bash).

- `gm.py stats NAME...`: last two seasons and this one in league scoring (per game,
  per start for goalies), status (mine/taken/free) and the model's rate. The
  quickest honest read of a player.
- `gm.py fits NAME [--drop NAME] [--weeks 3]`: his team's games each week and how
  many my lineup has a free slot for (a bench player's value is the nights he
  fills). For a goalie: expected starts per week, back-to-backs counted.
- `gm.py swap --drop A --add B [...]`: the change in my lineup's expected points
  this week, next week, the week after, and the rest of the season (the page's
  night-by-night math). Trades: repeat `--drop/--add` for each player.
- `gm.py scan [--keep NAME ...]`: every free agent against my roster: best this
  week, next week and over the season, without costing the season.
- `gm.py resolve NAME [--team ABBR]`: a name to an NHL id, or the reason it isn't one.
- `gm.py record LOG.txt [--apply]`: a pasted Yahoo transaction log (save the paste
  to a file first). Dry run by default. `--apply` adds untracked players to the
  extras sheet with Yahoo's positions, rebuilds the page, and writes the league file.
- `news.py [--dry-run] [--all] [--pages N]`: the gamedaytweets.com catch-up (below).

Caveats that have bitten before:
- **Extras are placeholders.** Players on the extras sheet are valued at
  replacement level; their model numbers mean nothing. Judge them on `stats` and
  their role in the news. `scan` leaves them out unless `--extras`.
- **Goalie shares can be stale.** They come from the goalie model's last re-read.
  If the news says a goalie's role changed (e.g. a clear starter named), record it
  in `config/goalie_priors.yaml` and re-read the posterior (below) before trusting
  a goalie suggestion.
- **The free-agent pool is only as good as the league file.** Ask for the latest
  transaction log if it's been a few days.

## Routines

### Morning and evening: news catch-up

1. `news.py`: new tweets since the bookmark, flagged when they name my players or
   the watch list, or concern my players' teams (lines, power play, goalie starts,
   injuries, transactions). It moves the bookmark and archives everything.
2. Read the flagged items and report **only what changes something**: a line
   promotion or demotion, power-play unit changes, confirmed goalie starts for my
   goalies (and streaming options), injuries and returns, call-ups and waivers, and
   anything touching a planned move or trigger in the current week's plan.
3. For team-level checks go straight to the source pages:
   `https://www.gamedaytweets.com/lines?team=PIT`, `/goalies` (today's confirmed
   and guessed starters), `/news`.
4. Update `watchlist.json` when a plan adds or drops a name, and log anything
   decided.

### A move the user made or wants to make

1. Price it: `stats`, `fits`, `swap` (and `scan` for alternatives). Check the
   player's current role in the news (lines, power play, goalie depth chart).
2. Recommend, with the numbers and the triggers that would change it.
3. Once made in Yahoo, record it in the league file:
   - one or two moves: through the running server (`GET`/`PUT
     http://127.0.0.1:8899/season/league`, the same validation the page uses),
     or, with the server down, `hockey.serve.season.load/save` on the file.
     Return dates go in `out` as `{id: "YYYY-MM-DD"}` (the page's Out button can
     only mark "out indefinitely").
   - a pasted transaction log: `gm.py record LOG.txt` then `--apply`.
4. Tell the user to reload the page. Log the decision.

### The page and the server

- Rebuild the page after the extras sheet or the boards change:
  `python scripts/build_draft_ui.py artifacts/board_v3 --goalies artifacts/goalies_v2`
  (the server reads the file on every request, so no restart for a rebuild).
- The server:
  `python -m hockey.serve --board artifacts/board_v3 --goalies artifacts/goalies_v2 --slot 4`
  on port 8899. Check with `curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8899/ui`.
  Restart it after the extras sheet changes: its own league saves validate ids
  against the sheet it loaded at startup.
- Started from Claude Code as a background task, the server gets killed when the
  machine is low on memory and the session is idle. Suggest the user run it in
  their own terminal; don't restart it unasked.
- Season stats: the page's **Refresh stats** button (NHL rosters, final games,
  injuries, then the season export and the in-season update). By hand:
  `python -m hockey.ingest refresh`, `python -m hockey.export.season_to_date`,
  `python -m hockey.model.in_season --board artifacts/board_v3 --goalies artifacts/goalies_v2`.
- A goalie's role changed (trade, named starter, demotion): edit
  `config/goalie_priors.yaml`, then
  `python -m hockey.model.run_goalies --posterior artifacts/goalies_v2/posterior.nc --out artifacts/goalies_v2`
  (about 2 minutes, no refit), `python -m hockey.export artifacts/board_v3 --goalies artifacts/goalies_v2`,
  rebuild the page, rerun `in_season`.

## The decision log

`artifacts/season/plans/decisions.md`, newest entry at the bottom:

```
## YYYY-MM-DD: short title
- **Decision:** what was done or decided (or declined).
- **Why:** the numbers (fits, swap deltas, per-game in league scoring) and the news, with links.
- **Triggers:** what would change it, and when to look again.
```

Weekly plans (`plans/YYYY-MM-DD-weekN-plan.md`) hold the week's moves, the
planned adds and their triggers, and what to check mid-week. Start each week by
reading the latest plan and the last few log entries; update the plan when a
move is made.
