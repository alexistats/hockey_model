"""The season so far, folded into the projection.

    python -m hockey.model.in_season --board artifacts/board_v3 --goalies artifacts/goalies_v2

A Bayesian update of the preseason posterior with this season's games, category by
category, and a simulation of the rest of the season from the result. The preseason
fit is the prior; the games played are the data; nothing is refitted.

The update is conjugate. For each player and counting category the posterior draws
give his per-game rate - a draw's season total over its games, less the Poisson noise
one season adds - and a gamma matched to that updates exactly with a Poisson count:
gamma(a, b) after Y events in n games is gamma(a + Y, b + n). `b` is what the
preseason posterior is worth in games, so four games barely move a player and a
season's worth moves him a long way. Plus/minus updates through a normal, and a
goalie's points a start through a normal with his start share a beta.

Checked on 2025-26 held out (fitted through 2024-25, 290 skaters, cut when the median
team had played 10, 20 and 40 games; see docs/architecture.md): the
updated rate beat both the untouched preseason rate and the pace so far at every cut
(mean absolute error 0.629 against 0.658 and 1.050 points a game after 10 games,
0.675 against 0.753 and 0.778 after 40), and the rest-of-season totals improved on
every measure - error, CRPS - with interval coverage unchanged. Two things did not
survive the check. Updating a player's chance of dressing from the games he has
missed made the rest-of-season games worse and collapsed the intervals to half their
coverage: absences come in runs, not the independent games a beta-binomial update
assumes, so ten healthy games say little about the next injury. It is off by default,
and a known absence is entered as one. Discounting the games (half evidence) made the
rate worse at every cut: the constant-rate update is the right weight here.

The rest of the season is simulated from the updated rates, draw for draw: each draw
keeps its rank in every category, so a draw that was a strong shooter preseason is a
strong shooter after the update, and the correlation between categories the fit
learned survives it. The opponent and home effects the fit carries are left out of
the update: over a run of games they average out, and the error is small against the
prior's own spread.

This changes a posterior, which is why it lives in the model layer and not the
export. A full refit with this season's games as data is the check on it, not a
replacement for it.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st
from sqlalchemy import text

from hockey.scoring import score_draws
from hockey.seasons import PROJECTION_SEASON, SEASON_LENGTH

logger = logging.getLogger(__name__)

COUNTS = ("goals", "assists", "ppp", "shp", "sog", "hits", "blocks")
SIGNED = "plus_minus"
DEFAULT_OUT = Path("artifacts/season/ros.json")
# A rate's prior spread never goes below this share of its mean: a posterior that
# certain would ignore any season, and a moment-matched gamma needs a spread.
MIN_CV = 0.05
# Game-to-game plus/minus spread, when the warehouse is not asked for its own.
SIGMA_PM = 1.05
# How firmly a goalie's preseason start share is held, in games of evidence.
SHARE_GAMES = 20.0


def rate_prior(totals: np.ndarray, games: np.ndarray) -> tuple[float, float]:
    """Mean and variance of a latent per-game rate from season-total draws.

    A draw's total over its games carries the Poisson noise of one season on top of
    the uncertainty about the rate; with a mean rate m over g games that noise is
    m / g, which is taken out."""
    ok = games > 0
    if not ok.any():
        return 0.0, 0.0
    r = totals[ok] / games[ok]
    m = float(r.mean())
    v = float(r.var()) - m * float(np.mean(1.0 / games[ok]))
    return m, max(v, (MIN_CV * m) ** 2)


def gamma_update(m: float, v: float, events: float, n: float) -> tuple[float, float]:
    """(shape, rate) of the gamma for a per-game rate after `events` in `n` games."""
    if m <= 0 or v <= 0:
        return events + 1e-3, n + 1e-3 / max(m, 1e-6)
    b = m / v
    return m * b + events, b + n


def beta_update(p: np.ndarray, hits: float, misses: float) -> tuple[float, float]:
    """A beta matched to draws of a probability, after `hits` and `misses`."""
    p = np.clip(p, 1e-4, 1 - 1e-4)
    m, v = float(p.mean()), float(p.var())
    v = min(max(v, 1e-5), m * (1 - m) * 0.99)
    k = m * (1 - m) / v - 1
    return m * k + hits, (1 - m) * k + misses


def _ranks(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Each draw's rank as a probability in (0, 1), ties broken at random."""
    order = np.lexsort((rng.random(len(x)), x))
    u = np.empty(len(x))
    u[order] = (np.arange(len(x)) + 0.5) / len(x)
    return u


@dataclass
class Seen:
    """A player's season so far: games, category sums, and his team's games."""

    games: int
    sums: dict[str, float]
    team_games: int
    remaining: int


def update_skater(
    prior: dict[str, np.ndarray],
    games: np.ndarray,
    season_games: int,
    seen: Seen,
    rng: np.random.Generator,
    sigma_pm: float = SIGMA_PM,
    availability: bool = False,
    evidence: float = 1.0,
) -> dict:
    """Rest-of-season draws for one skater: per category, games, and his updated
    per-game rates (the category means a game he plays).

    `availability` updates his chance of dressing from the games he has missed so
    far; `evidence` is what each game played counts for, 1 being the full weight a
    constant rate would give it."""
    n_draws = len(games)
    missed = max(seen.team_games - seen.games, 0)
    avail_a, avail_b = beta_update(
        games / season_games, seen.games if availability else 0, missed if availability else 0
    )
    p = rng.beta(avail_a, avail_b, n_draws)
    ros_games = rng.binomial(max(seen.remaining, 0), p)
    out, rates = {"games": ros_games.astype(float)}, {}
    safe = np.where(games > 0, games, np.nan)
    for c in COUNTS:
        m, v = rate_prior(prior[c], games)
        a, b = gamma_update(m, v, evidence * seen.sums.get(c, 0.0), evidence * seen.games)
        u = _ranks(np.nan_to_num(prior[c] / safe, nan=m), rng)
        rate = st.gamma.ppf(u, a, scale=1.0 / b)
        rates[c] = a / b
        out[c] = rng.poisson(rate * ros_games).astype(float)
    # Plus/minus: a normal rate with game noise, updated the same way.
    r = np.nan_to_num(prior[SIGNED] / safe, nan=0.0)
    m = float(r.mean())
    v = max(float(r.var()) - sigma_pm**2 * float(np.nanmean(1.0 / safe)), 1e-4)
    post_v = 1.0 / (1.0 / v + evidence * seen.games / sigma_pm**2)
    post_m = post_v * (m / v + evidence * seen.sums.get(SIGNED, 0.0) / sigma_pm**2)
    rate_pm = st.norm.ppf(_ranks(r, rng), post_m, np.sqrt(post_v))
    rates[SIGNED] = post_m
    out[SIGNED] = np.rint(
        rng.normal(rate_pm * ros_games, sigma_pm * np.sqrt(np.maximum(ros_games, 0)))
    )
    return {"draws": out, "rates": rates, "avail": avail_a / (avail_a + avail_b)}


def update_goalie(
    season_points: np.ndarray,
    exp_starts: float,
    season_games: int,
    starts: int,
    points: float,
    team_games: int,
    remaining: int,
    sigma_start: float,
    rng: np.random.Generator,
    rate_update: bool = False,
) -> dict | None:
    """Rest-of-season draws for one goalie: points a start as a normal, and his share
    of the team's starts as a beta, updated by his starts so far.

    The share updates by default: who starts is a role, and a month of it is
    evidence. Points a start do not, unless asked: there is no goalie posterior
    fitted through a held-out season to check that update on, and a goalie's early
    save percentage is noisy enough that an unchecked update could move him a long
    way on a bad fortnight.

    None for a goalie with no preseason starts - squeezed out of a full net, like
    Vancouver's fourth in September 2026. His draws hold no starts, so there is no
    rate a start to update; the next re-read of the goalie posterior, which follows
    the rosters, is what brings him back."""
    if exp_starts < 0.5:
        return None
    n_draws = len(season_points)
    r = season_points / max(exp_starts, 1e-6)
    m = float(r.mean())
    v = max(float(r.var()) - sigma_start**2 / max(exp_starts, 1.0), (MIN_CV * m) ** 2)
    seen = starts if rate_update else 0
    post_v = 1.0 / (1.0 / v + seen / sigma_start**2)
    post_m = post_v * (m / v + (points if rate_update else 0.0) / sigma_start**2)
    share = min(max(exp_starts / season_games, 1e-3), 0.95)
    a = share * SHARE_GAMES + starts
    b = (1 - share) * SHARE_GAMES + max(team_games - starts, 0)
    ros_starts = rng.binomial(max(remaining, 0), rng.beta(a, b, n_draws))
    rate = st.norm.ppf(_ranks(r, rng), post_m, np.sqrt(post_v))
    ros_points = rng.normal(rate * ros_starts, sigma_start * np.sqrt(np.maximum(ros_starts, 0)))
    return {
        "points": ros_points,
        "starts": ros_starts.astype(float),
        "rate": post_m,
        "share": a / (a + b),
    }


# ---------------------------------------------------------------------------------------
# From the warehouse and the board to rest-of-season projections

_SEEN_SKATERS = """
SELECT s.player_id, count(*) AS games, sum(s.goals) AS goals, sum(s.assists) AS assists,
       sum(s.ppp) AS ppp, sum(s.shp) AS shp, sum(s.sog) AS sog, sum(s.hits) AS hits,
       sum(s.blocks) AS blocks, sum(s.plus_minus) AS plus_minus
  FROM skater_game_logs s JOIN nhl_games g ON g.nhl_game_id = s.game_id
 WHERE g.game_type = 2 AND g.season = :season AND g.date <= :as_of
 GROUP BY s.player_id
"""
_SEEN_GOALIES = """
SELECT gl.player_id, sum(gl.started::int) AS starts, sum(gl.wins) AS wins,
       sum(gl.goals_against) AS goals_against, sum(gl.saves) AS saves,
       sum(coalesce(gl.shutouts, 0)) AS shutouts, sum(gl.started::int) AS games_started
  FROM goalie_game_logs gl JOIN nhl_games g ON g.nhl_game_id = gl.game_id
 WHERE g.game_type = 2 AND g.season = :season AND g.date <= :as_of
 GROUP BY gl.player_id
"""
_TEAM_GAMES = """
SELECT t.team, sum((g.date <= :as_of)::int) AS played, sum((g.date > :as_of)::int) AS remaining
  FROM nhl_games g
  CROSS JOIN LATERAL (VALUES (g.home_team_abbrev), (g.away_team_abbrev)) AS t(team)
 WHERE g.game_type = 2 AND g.season = :season
 GROUP BY t.team
"""


def team_games(session, season: int, as_of: date) -> dict[str, tuple[int, int]]:
    rows = session.execute(text(_TEAM_GAMES), {"season": season, "as_of": as_of.isoformat()})
    return {r.team: (int(r.played), int(r.remaining)) for r in rows}


def run(
    session,
    board_dir: Path,
    goalies_dir: Path | None,
    scoring,
    season: int,
    as_of: date,
    teams: dict[int, str],
    seed: int = 11,
) -> dict:
    rng = np.random.default_rng(seed)
    board = pd.read_csv(board_dir / "value_board.csv").set_index("player_id")
    schedule = team_games(session, season, as_of)
    params = {"season": season, "as_of": as_of.isoformat()}
    seen_sk = pd.DataFrame(session.execute(text(_SEEN_SKATERS), params).mappings())
    seen_sk = seen_sk.set_index("player_id") if not seen_sk.empty else seen_sk
    seen_g = pd.DataFrame(session.execute(text(_SEEN_GOALIES), params).mappings())
    seen_g = seen_g.set_index("player_id") if not seen_g.empty else seen_g
    length = SEASON_LENGTH[season]
    weights = {r.key: r.modifier for r in scoring.skater_rules}
    players = {}

    def team_of(pid: int) -> str:
        return teams.get(pid) or str(board.at[pid, "team"])

    for path in sorted(board_dir.glob("draws_batch_*.npz")):
        with np.load(path) as z:
            for j, pid in enumerate(z["player_ids"]):
                pid = int(pid)
                if pid not in board.index:
                    continue
                played, remaining = schedule.get(team_of(pid), (0, length))
                row = seen_sk.loc[pid] if pid in getattr(seen_sk, "index", ()) else None
                seen = Seen(
                    games=int(row["games"]) if row is not None else 0,
                    sums={c: float(row[c] or 0) for c in (*COUNTS, SIGNED)}
                    if row is not None
                    else {},
                    team_games=played,
                    remaining=remaining,
                )
                prior = {c: z[f"stat_{c}"][:, j].astype(float) for c in (*COUNTS, SIGNED)}
                got = update_skater(prior, z["games_played"][:, j].astype(float), length, seen, rng)
                pts = score_draws(scoring, got["draws"], "P")
                pre = sum(
                    weights[c] * rate_prior(prior[c], z["games_played"][:, j])[0]
                    for c in (*COUNTS,)
                ) + weights.get(SIGNED, 0) * float(
                    np.nanmean(
                        prior[SIGNED]
                        / np.where(z["games_played"][:, j] > 0, z["games_played"][:, j], np.nan)
                    )
                )
                rate = (
                    sum(weights[c] * got["rates"][c] for c in COUNTS)
                    + weights.get(SIGNED, 0) * got["rates"][SIGNED]
                )
                players[str(pid)] = {
                    "rate": round(float(rate), 3),
                    "pre_rate": round(float(pre), 3),
                    "games": round(float(got["draws"]["games"].mean()), 1),
                    "ros": round(float(pts.mean()), 1),
                    "p10": round(float(np.percentile(pts, 10)), 1),
                    "p90": round(float(np.percentile(pts, 90)), 1),
                    "seen": seen.games,
                }

    if goalies_dir is not None and (goalies_dir / "goalie_draws.npz").exists():
        from hockey.export.season_to_date import game_spread

        spread = game_spread(session, scoring, [season - 10001, season - 20002], as_of.isoformat())
        with np.load(goalies_dir / "goalie_draws.npz") as z:
            for j, pid in enumerate(z["player_ids"]):
                pid = int(pid)
                if pid not in board.index or not np.isfinite(board.at[pid, "exp_starts"]):
                    continue
                played, remaining = schedule.get(team_of(pid), (0, length))
                row = seen_g.loc[pid] if pid in getattr(seen_g, "index", ()) else None
                starts = int(row["starts"]) if row is not None else 0
                points = (
                    float(
                        score_draws(
                            scoring,
                            {
                                r.key: np.array([float(row[r.key] or 0)])
                                for r in scoring.goalie_rules
                            },
                            "G",
                        )[0]
                    )
                    if row is not None
                    else 0.0
                )
                exp = float(board.at[pid, "exp_starts"])
                sd = spread["G"].get(pid, spread["G"].get(-1, 6.5))
                got = update_goalie(
                    z["draws"][:, j].astype(float),
                    exp,
                    length,
                    starts,
                    points,
                    played,
                    remaining,
                    sd,
                    rng,
                )
                if got is None:
                    logger.info("goalie %d has no preseason starts; not updated", pid)
                    continue
                players[str(pid)] = {
                    "rate": round(float(got["rate"]), 3),
                    "pre_rate": round(float(board.at[pid, "mean"]) / exp, 3),
                    "games": round(float(got["starts"].mean()), 1),
                    "share": round(float(got["share"]), 3),
                    "ros": round(float(got["points"].mean()), 1),
                    "p10": round(float(np.percentile(got["points"], 10)), 1),
                    "p90": round(float(np.percentile(got["points"], 90)), 1),
                    "seen": starts,
                }
    return {
        "season": season,
        "as_of": as_of.isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
        "players": players,
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="hockey.model.in_season")
    parser.add_argument("--board", default="artifacts/board_v3")
    parser.add_argument("--goalies", default="artifacts/goalies_v2")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--season", type=int, default=PROJECTION_SEASON)
    parser.add_argument("--as-of", default=None, help="YYYY-MM-DD; today by default")
    parser.add_argument("--rosters", default="artifacts/season/rosters.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from hockey.db import SessionLocal
    from hockey.export.season_to_date import current_teams
    from hockey.yahoo.settings import load_scoring_from_yaml

    as_of = date.fromisoformat(args.as_of) if args.as_of else date.today()
    with SessionLocal() as session:
        doc = run(
            session,
            Path(args.board),
            Path(args.goalies) if args.goalies else None,
            load_scoring_from_yaml(),
            args.season,
            as_of,
            current_teams(Path(args.rosters), session),
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc), encoding="utf-8")
    moved = sum(1 for p in doc["players"].values() if abs(p["rate"] - p["pre_rate"]) > 0.25)
    logger.info(
        "wrote %s: %d players through %s, %d with a rate moved by more than 0.25 a game",
        out,
        len(doc["players"]),
        doc["as_of"],
        moved,
    )


if __name__ == "__main__":
    main()
