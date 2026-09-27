"""The season so far, read against the model.

    python -m hockey.export.season_to_date --board artifacts/board_v3 --goalies artifacts/goalies_v2

Season-to-date totals in this league's scoring, each player's current team and
injury, and how surprising his pace is given the model's projection and the
number of games behind it - written to one JSON file the Season tab reads.

"Surprising" is the point. Four games at double a player's projected rate is
noise: fantasy points swing by about as much as they average from one game to
the next, so it takes a run of games before a pace says anything the
projection did not. The predictive spread of an n-game total is the model's
uncertainty about the rate, which scales with n, plus game-to-game noise, which
scales with the square root of n; a player is flagged only when his total sits
in the outer tenth of that spread. Early on almost no one is, by design.

Nothing here changes a posterior. Whether the season so far should move the
projection is the model's question, answered by refitting it with these games
as data, not by this file.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from hockey import extras as extras_mod
from hockey.scoring import score_draws
from hockey.seasons import PROJECTION_SEASON

logger = logging.getLogger(__name__)

DEFAULT_OUT = Path("artifacts/season/stats.json")
DEFAULT_ROSTERS = Path("artifacts/season/rosters.json")
# Outer tenth either side: above the 90th percentile of the predictive spread is
# running hot against the model, below the 10th cold.
FLAG = 0.10
# Enough games in a season for its game-to-game spread to be a player's own.
MIN_SD_GAMES = {"P": 40, "G": 20}

SKATER_KEYS = ["goals", "assists", "plus_minus", "ppp", "shp", "sog", "hits", "blocks"]
GOALIE_KEYS = ["games_started", "wins", "goals_against", "saves", "shutouts"]

_SKATER_GAMES = """
SELECT s.player_id, g.season, g.date, s.team_abbrev, s.goals, s.assists, s.plus_minus,
       s.ppp, s.shp, s.sog, s.hits, s.blocks
  FROM skater_game_logs s
  JOIN nhl_games g ON g.nhl_game_id = s.game_id
 WHERE g.game_type = 2 AND g.season = ANY(:seasons) AND g.date <= :as_of
"""
_GOALIE_GAMES = """
SELECT gl.player_id, g.season, g.date, gl.team_abbrev, gl.started::int AS games_started,
       gl.wins, gl.goals_against, gl.saves, gl.shutouts
  FROM goalie_game_logs gl
  JOIN nhl_games g ON g.nhl_game_id = gl.game_id
 WHERE g.game_type = 2 AND g.season = ANY(:seasons) AND g.date <= :as_of
"""


def surprise(actual: float, n: float, rate: float, rate_var: float, game_sd: float):
    """(expected total, percentile of `actual`) for `n` games at a projected
    per-game `rate`: the n-game total's predictive spread is the rate's
    uncertainty, n^2 * rate_var, plus game-to-game noise, n * game_sd^2."""
    if n <= 0:
        return 0.0, None
    expected = n * rate
    spread = math.sqrt(n * n * max(rate_var, 0.0) + n * game_sd * game_sd)
    if spread <= 0:
        return expected, None
    z = (actual - expected) / spread
    return expected, 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _score(frame: pd.DataFrame, keys: list[str], scoring, position_type: str) -> np.ndarray:
    return score_draws(
        scoring, {k: frame[k].fillna(0).to_numpy(float) for k in keys}, position_type
    )


def game_spread(session, scoring, seasons: list[int], as_of: str) -> dict[str, dict[int, float]]:
    """Each player's game-to-game standard deviation in fantasy points over the
    seasons given, where he has enough games for it to be his own, and the
    median of those for everyone else by position type."""
    out = {}
    for sql, keys, kind in ((_SKATER_GAMES, SKATER_KEYS, "P"), (_GOALIE_GAMES, GOALIE_KEYS, "G")):
        games = pd.DataFrame(
            session.execute(text(sql), {"seasons": seasons, "as_of": as_of}).mappings()
        )
        if kind == "G" and not games.empty:
            games = games[games["games_started"] == 1]  # a start is the unit a goalie is scored by
        if games.empty:
            out[kind] = {}
            continue
        games["fp"] = _score(games, keys, scoring, kind)
        by = games.groupby("player_id")["fp"].agg(["std", "size"])
        own = by[by["size"] >= MIN_SD_GAMES[kind]]["std"]
        out[kind] = {int(k): float(v) for k, v in own.items()}
        out[kind][-1] = float(own.median()) if len(own) else float(games["fp"].std())
    return out


def _rate_spread(
    board_dir: Path, goalies_dir: Path | None, board: pd.DataFrame
) -> dict[int, tuple[float, float]]:
    """Per player: (mean, variance) of his per-game rate across the posterior
    draws - per start for a goalie. These are season-total draws, so their
    per-game rate still carries a season's worth of game noise; the caller
    takes that out."""
    out = {}
    for path in sorted(board_dir.glob("draws_batch_*.npz")):
        with np.load(path) as z:
            ids, draws, games = z["player_ids"], z["draws"], z["games_played"]
            rate = np.where(games > 0, draws / np.maximum(games, 1), np.nan)
            for j, pid in enumerate(ids):
                r = rate[:, j][np.isfinite(rate[:, j])]
                if len(r):
                    out[int(pid)] = (float(r.mean()), float(r.var()))
    starts = dict(zip(board["player_id"].astype(int), board["exp_starts"], strict=True))
    if goalies_dir is not None and (goalies_dir / "goalie_draws.npz").exists():
        with np.load(goalies_dir / "goalie_draws.npz") as z:
            for j, pid in enumerate(z["player_ids"]):
                s = starts.get(int(pid))
                if s and s > 0:
                    r = z["draws"][:, j] / s
                    out[int(pid)] = (float(r.mean()), float(r.var()))
    return out


def current_teams(rosters: Path | None, session) -> dict[int, str]:
    """Who plays where now: the refresh's roster record, where there is one. Only
    that says who is on a current roster; `players.team_abbrev` keeps a retired
    or unsigned player on the last team he had, so it is the fallback, not the
    source."""
    if rosters is not None and rosters.exists():
        doc = json.loads(rosters.read_text(encoding="utf-8"))
        return {int(k): v for k, v in doc.get("teams", {}).items()}
    logger.warning("no %s: current teams from players.team_abbrev, which can be stale", rosters)
    return dict(
        session.execute(
            text("SELECT nhl_id, team_abbrev FROM players WHERE team_abbrev IS NOT NULL")
        ).all()
    )


def season_to_date(
    session,
    board_dir: Path,
    goalies_dir: Path | None,
    scoring,
    season: int,
    as_of: date,
    rosters: Path | None = DEFAULT_ROSTERS,
) -> dict:
    board = pd.read_csv(board_dir / "value_board.csv")
    if "exp_starts" not in board:
        board["exp_starts"] = np.nan
    goalie_ids = set(board.loc[board["exp_starts"].notna(), "player_id"].astype(int))
    rates = _rate_spread(board_dir, goalies_dir, board)
    spread = game_spread(session, scoring, [season - 10001, season - 20002], as_of.isoformat())

    params = {"seasons": [season], "as_of": as_of.isoformat()}
    skaters = pd.DataFrame(session.execute(text(_SKATER_GAMES), params).mappings())
    goalies = pd.DataFrame(session.execute(text(_GOALIE_GAMES), params).mappings())
    teams = current_teams(rosters, session)
    injured = {
        int(r.player_id): {"status": r.status, "type": r.injury_type, "back": r.return_date}
        for r in session.execute(
            text("SELECT player_id, status, injury_type, return_date FROM player_injuries")
        )
    }

    def totals(frame: pd.DataFrame, keys: list[str], kind: str) -> dict[int, dict]:
        if frame.empty:
            return {}
        frame = frame.sort_values("date")
        frame["fp"] = _score(frame, keys, scoring, kind)
        out = {}
        for pid, g in frame.groupby("player_id"):
            # A goalie is projected per start, so starts are his games; the saves
            # he made in relief still scored, so his points count them.
            n = int(g["games_started"].sum()) if kind == "G" else int(len(g))
            if n == 0:
                continue
            out[int(pid)] = {
                "gp": n,
                "pts": round(float(g["fp"].sum()), 1),
                "cats": {k: round(float(g[k].fillna(0).sum()), 3) for k in keys},
                "last_team": str(g["team_abbrev"].iloc[-1]),
            }
        return out

    played = {**totals(skaters, SKATER_KEYS, "P"), **totals(goalies, GOALIE_KEYS, "G")}
    players = {}
    for row in board.itertuples():
        pid = int(row.player_id)
        kind = "G" if pid in goalie_ids else "P"
        team = teams.get(pid) or played.get(pid, {}).get("last_team") or row.team
        entry = {
            "team": str(team),
            "gp": 0,
            "pts": 0.0,
            "pg": None,
            "exp": None,
            "pct": None,
            "flag": "",
        }
        got = played.get(pid)
        if got:
            entry.update(
                gp=got["gp"], pts=got["pts"], pg=round(got["pts"] / got["gp"], 2), cats=got["cats"]
            )
            rate, rate_var = rates.get(pid, (float("nan"), 0.0))
            n_season = float(row.exp_starts if kind == "G" else row.exp_games) or 1.0
            sd = spread[kind].get(pid, spread[kind].get(-1, 0.0))
            # The draws' per-game spread includes a season of game noise; what is
            # left once that is taken out is uncertainty about the rate itself.
            true_var = max(rate_var - sd * sd / n_season, 0.0)
            if math.isfinite(rate):
                expected, pct = surprise(got["pts"], got["gp"], rate, true_var, sd)
                entry.update(exp=round(expected, 1), pct=None if pct is None else round(pct, 3))
                if pct is not None:
                    entry["flag"] = "above" if pct >= 1 - FLAG else "below" if pct <= FLAG else ""
        players[str(pid)] = entry
    # Players outside the model: their season so far and where they play, but
    # never a flag - against a replacement-level placeholder, "above the model"
    # would say nothing about the player.
    sheet = extras_mod.load()
    if sheet is not None:
        for r in sheet.itertuples():
            pid = int(r.nhl_id)
            if str(pid) in players:
                continue
            entry = {
                "team": str(teams.get(pid) or r.team),
                "gp": 0,
                "pts": 0.0,
                "pg": None,
                "exp": None,
                "pct": None,
                "flag": "",
            }
            got = played.get(pid)
            if got:
                entry.update(
                    gp=got["gp"],
                    pts=got["pts"],
                    pg=round(got["pts"] / got["gp"], 2),
                    cats=got["cats"],
                )
            players[str(pid)] = entry
    return {
        "season": season,
        "as_of": as_of.isoformat(),
        "refreshed_at": datetime.now(UTC).isoformat(),
        "games": int(skaters["player_id"].count()) if not skaters.empty else 0,
        "players": players,
        "injuries": {str(k): v for k, v in injured.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="hockey.export.season_to_date")
    parser.add_argument("--board", default="artifacts/board_v3")
    parser.add_argument("--goalies", default="artifacts/goalies_v2")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--season", type=int, default=PROJECTION_SEASON)
    parser.add_argument("--as-of", default=None, help="YYYY-MM-DD; today by default")
    parser.add_argument("--rosters", default=str(DEFAULT_ROSTERS))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from hockey.db import SessionLocal
    from hockey.yahoo.settings import load_scoring_from_yaml

    as_of = date.fromisoformat(args.as_of) if args.as_of else date.today()
    with SessionLocal() as session:
        doc = season_to_date(
            session,
            Path(args.board),
            Path(args.goalies) if args.goalies else None,
            load_scoring_from_yaml(),
            args.season,
            as_of,
            Path(args.rosters),
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc), encoding="utf-8")
    flagged = [p for p in doc["players"].values() if p["flag"]]
    logger.info(
        "wrote %s: season %d to %s, %d player-games, %d players flagged",
        out,
        doc["season"],
        doc["as_of"],
        doc["games"],
        len(flagged),
    )


if __name__ == "__main__":
    main()
