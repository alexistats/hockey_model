"""Read the goalie posterior as a season projection, with room for a human.

The model knows what the game logs know. It does not know that a team signed a
starter in July, that a prospect has been handed the net, or that a tandem is
about to become a workhorse. Those are depth-chart facts, they live nowhere in
eight seasons of box scores, and they move the number that matters most:
starts are 71% of the variance in a goalie's fantasy season.

So `config/goalie_priors.yaml` can override the workload for named goalies, and
nudge a team's shot volume or win rate. Overrides are applied as distributions,
not as point estimates - saying "about 55 starts, give or take 6" keeps the
floor and ceiling honest, where pinning it at 55 would quietly claim certainty
nobody has. Every override that fires is logged, and an override naming a
goalie who is not in the pool is an error rather than a silent no-op.

Two depth-chart facts do not need a human. Who is on which roster now is in the
refresh's record, so a goalie traded in the summer is projected on his new team.
And a team plays a fixed number of games, so however the walks add up, its
goalies' starts cannot exceed them - see `fill_the_net`.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

PRIORS_PATH = Path("config/goalie_priors.yaml")
# Written by `python -m hockey.ingest refresh`.
ROSTERS_PATH = Path("artifacts/season/rosters.json")


def load_priors(path: Path = PRIORS_PATH) -> dict:
    if not path.exists():
        return {"goalies": [], "teams": []}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {"goalies": loaded.get("goalies") or [], "teams": loaded.get("teams") or []}


def load_rosters(path: Path, season: int) -> dict[int, str] | None:
    """{player id: team} for everyone on a current NHL roster, as the refresh
    recorded it, or None when there is no record. The one source that knows a
    summer trade before the player has dressed for his new team."""
    if not path.exists():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    if int(doc.get("season", 0)) != season:
        raise SystemExit(
            f"{path} holds {doc.get('season')} rosters, not {season}'s; run "
            f"python -m hockey.ingest refresh first"
        )
    logger.info("current rosters from %s, synced %s", path, doc.get("synced_at"))
    return {int(k): v for k, v in doc["teams"].items()}


def fill_the_net(
    expected: dict[int, float], fixed: set[int], depth: dict[str, list[int]], games: int
) -> dict[int, float]:
    """A scale on each goalie's share so that no team starts goalies in more games
    than it plays. `expected` is each goalie's projected starts on his own,
    `fixed` the ones the priors file sets, `depth` each team's goalies.

    The workload walk is per goalie: it knows how much of his team's net each one
    has had, not who else is in it now. A goalie traded behind a starter keeps
    the workload he had, and the pair add up to more starts than there are games
    - Philadelphia's came to 89 of 84 in September 2026, Vancouver's four to 108.

    So each team's net fills in order: first the goalies the priors file sets,
    which is a human's depth chart, then the model's by their own projected
    starts, each taking what he projects or what is left. Whoever the model has
    playing most keeps his workload, and the one squeezed is the one it has
    playing least, which on a September roster is usually a goalie bound for the
    minors. Shrinking all of them alike would have taken a fifth of Vancouver's
    starter's games to make room for two who will not both be there. A team
    under its games is left alone: the rest go to call-ups the pool has never
    seen.
    """
    scale: dict[int, float] = {}
    for team, members in depth.items():
        by_hand = sum(expected[m] for m in members if m in fixed)
        if by_hand > games:
            raise SystemExit(
                f"goalie_priors.yaml gives {team}'s goalies {by_hand:.0f} starts in a "
                f"{games}-game season"
            )
        room = games - by_hand
        for m in sorted((m for m in members if m not in fixed), key=lambda m: (-expected[m], m)):
            scale[m] = 1.0 if expected[m] <= room else max(room, 0.0) / expected[m]
            room -= expected[m] * scale[m]
    return scale


def on_the_board(
    board: pd.DataFrame,
    latest: int,
    min_starts: int,
    rostered: set[int],
    include_retired: bool = False,
) -> pd.DataFrame:
    """The projected goalies worth reading. The forecast covers every goalie the
    model knows; these cutoffs are only about what is worth reading, and a goalie
    left out is not a goalie judged bad.

    Recency is not optional. The workload walk takes a step per season whether or
    not anyone played, so a goalie last seen in 2018-19 still arrives at the
    projected season with a plausible-looking number attached - the first run of
    this put Roberto Luongo, Henrik Lundqvist and Corey Crawford on the board, all
    long retired, with Ben Bishop projected 404 points at 47 starts. Nothing
    downstream could have caught that, because the projection is complete and well
    formed; it is simply about someone who will not play.

    A goalie on a current roster is kept whatever his last season said: he has not
    retired, and a backup left out is a gap on the page - Levi sat there at a
    placeholder 30 starts while his own projection, already sharing Edmonton's net
    in the forecast, went unshown.
    """
    on = board["player_id"].isin(rostered)
    keep = (board["last_season_starts"] >= min_starts) | on
    if not include_retired:
        recent = board["last_season"] == latest
        stale = int((~recent & ~on).sum())
        if stale:
            logger.info(
                "left out %d goalie(s) who did not appear in %d, the most recent season "
                "in the data, and are on no current roster",
                stale,
                latest,
            )
        keep &= recent | on
    return board[keep].reset_index(drop=True)


def project(
    idata,
    frame: pd.DataFrame,
    index,
    scoring: dict[str, float],
    season_games: int,
    priors: dict | None = None,
    rng: np.random.Generator | None = None,
    current: dict[int, str] | None = None,
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """(summary per goalie, fantasy-point draws per goalie).

    Every quantity is drawn, not averaged: starts from the workload walk, shots
    from the team's rate given those starts, saves from those shots, and so on
    down. Summarising at each step instead would collapse exactly the spread the
    project exists to report.

    `current` is who is on which roster now (`load_rosters`). Without it every
    goalie is projected on the team he finished last season on.
    """
    rng = rng or np.random.default_rng(20262027)
    priors = priors or {"goalies": [], "teams": []}
    post = idata.posterior
    step = index.projection_step

    def flat(name):
        return post[name].values.reshape((-1,) + post[name].values.shape[2:])

    n_draws = flat("mu_save").shape[0]

    mu_save, mu_shots, mu_win, mu_share = (
        flat(n) for n in ("mu_save", "mu_shots", "mu_win", "mu_share")
    )
    mu_shutout = flat("mu_shutout")
    beta_win = flat("beta_save_on_win")
    beta_so_save = flat("beta_save_on_shutout")
    beta_so_shots = flat("beta_shots_on_shutout")
    alpha_shots = flat("alpha_shots")
    save_walk, share_walk = flat("save_walk"), flat("share_walk")
    team_save, team_shots, team_win = flat("team_save"), flat("team_shots"), flat("team_win")
    goalie_shots = flat("goalie_shots")

    # The team he finished on is the team of his last game. Sorting stints by
    # season alone left a goalie traded mid-season on whichever team sorted
    # last - Jarry's 2025-26 went Pittsburgh then Edmonton, and the alphabet had
    # him in Pittsburgh with 13 starts, under the board's cutoff. His season's
    # starts are the season's, across both teams.
    order = ["season", "last_game"] if "last_game" in frame else ["season"]
    recent = frame.sort_values(order).groupby("player_id").tail(1).set_index("player_id")
    season_starts = frame.groupby(["player_id", "season"])["starts"].sum()
    finished = recent["team"].to_dict()

    # A summer trade is in no game log, but it is in the roster record: a goalie
    # on a current roster is projected on that team - its shots, its defence,
    # its wins. One on none keeps the team he finished on. That is not always a
    # goalie who has gone: injured players drop off the roster listing, as
    # Gustavsson and Merzlikins had in September 2026.
    team_of = dict(finished)
    for pid in index.goalies:
        now = None if current is None else current.get(int(pid))
        if now is None or now == finished.get(pid):
            continue
        if now not in index.team_of:
            logger.warning(
                "%s is on %s's roster, a team the model has no seasons for; projected on %s",
                pid,
                now,
                finished.get(pid),
            )
            continue
        team_of[pid] = now

    team_adjust = {t["team"]: t for t in priors["teams"]}
    by_id = {int(g["nhl_id"]): g for g in priors["goalies"] if g.get("nhl_id") is not None}
    unused = set(by_id)

    # --- workload: each goalie's share of his team's games, on his own ---
    shares: dict[int, np.ndarray] = {}
    fixed: set[int] = set()
    for pid in index.goalies:
        team = team_of.get(pid)
        if team is None or team not in index.team_of:
            continue
        override = by_id.get(int(pid))
        if override is not None and override.get("starts") is not None:
            unused.discard(int(pid))
            mean = float(override["starts"])
            sd = float(override.get("starts_sd", 6.0))
            starts = np.clip(np.rint(rng.normal(mean, sd, size=n_draws)), 0, season_games)
            shares[pid] = starts / season_games
            fixed.add(pid)
            logger.info(
                "workload override: %s starts ~ N(%.0f, %.0f)",
                override.get("name", pid),
                mean,
                sd,
            )
        else:
            gi = index.goalie_of[pid]
            shares[pid] = 1.0 / (1.0 + np.exp(-(mu_share + share_walk[:, gi, step])))

    # --- then shared out: a team's net holds only as many starts as it has games ---
    # Its depth chart is its current roster when there is a record of one, and
    # otherwise whoever played for it last season. A goalie on neither - retired,
    # released, in the minors - is on no team's chart and squeezes nobody.
    latest = frame["season"].max()
    depth: dict[str, list[int]] = {}
    for pid in shares:
        if current is not None:
            on = current.get(int(pid))
        else:
            on = finished[pid] if recent.loc[pid, "season"] == latest else None
        if on is not None and on == team_of[pid]:
            depth.setdefault(on, []).append(pid)
    own = {pid: float(share.mean()) * season_games for pid, share in shares.items()}
    scale = fill_the_net(own, fixed, depth, season_games)

    rows, draws_by_id = [], {}
    for pid, share in shares.items():
        gi = index.goalie_of[pid]
        team = team_of[pid]
        ti = index.team_of[team]
        if pid in fixed:
            starts = np.rint(share * season_games).astype(int)
        else:
            p_share = share * scale.get(pid, 1.0)
            starts = rng.binomial(season_games, np.clip(p_share, 1e-6, 1 - 1e-6))

        # --- shots faced, given those starts ---
        shots_adj = float(team_adjust.get(team, {}).get("shots_adjust", 0.0))
        log_rate = mu_shots + team_shots[:, ti, step] + goalie_shots[:, gi] + shots_adj
        expected_shots = np.exp(log_rate) * np.maximum(starts, 0)
        shots = np.where(
            expected_shots > 0,
            rng.negative_binomial(
                alpha_shots,
                alpha_shots / np.maximum(alpha_shots + expected_shots, 1e-9),
            ),
            0,
        )

        # --- saves, and therefore goals against ---
        quality = save_walk[:, gi, step] + team_save[:, ti, step]
        p_save = 1.0 / (1.0 + np.exp(-(mu_save + quality)))
        saves = rng.binomial(shots, np.clip(p_save, 1e-6, 1 - 1e-6))
        goals_against = shots - saves

        # --- wins ---
        win_adj = float(team_adjust.get(team, {}).get("win_adjust", 0.0))
        p_win = 1.0 / (
            1.0 + np.exp(-(mu_win + team_win[:, ti, step] + beta_win * quality + win_adj))
        )
        wins = rng.binomial(starts, np.clip(p_win, 1e-6, 1 - 1e-6))

        # --- shutouts ---
        p_so = 1.0 / (
            1.0
            + np.exp(
                -(
                    mu_shutout
                    + beta_so_save * quality
                    + beta_so_shots * (team_shots[:, ti, step] + goalie_shots[:, gi])
                )
            )
        )
        shutouts = rng.binomial(starts, np.clip(p_so, 1e-6, 1 - 1e-6))

        points = (
            starts * scoring.get("games_started", 0.0)
            + wins * scoring.get("wins", 0.0)
            + goals_against * scoring.get("goals_against", 0.0)
            + saves * scoring.get("saves", 0.0)
            + shutouts * scoring.get("shutouts", 0.0)
        )
        draws_by_id[int(pid)] = points.astype("float32")
        last_season = recent.loc[pid, "season"]
        rows.append(
            {
                "player_id": int(pid),
                "team": team,
                "last_team": finished[pid],
                "mean": float(points.mean()),
                "floor": float(np.quantile(points, 0.10)),
                "p20": float(np.quantile(points, 0.20)),
                "p80": float(np.quantile(points, 0.80)),
                "ceiling": float(np.quantile(points, 0.90)),
                "sd": float(points.std()),
                "exp_starts": float(starts.mean()),
                "exp_wins": float(wins.mean()),
                "exp_saves": float(saves.mean()),
                "exp_ga": float(goals_against.mean()),
                "exp_shutouts": float(shutouts.mean()),
                "save_pct": float(saves.sum() / max(shots.sum(), 1)),
                # What his own workload said before his team's net was shared out.
                "own_starts": own[pid],
                "last_season": int(last_season),
                "last_season_starts": int(season_starts[(pid, last_season)]),
                "override": pid in fixed,
            }
        )

    if unused:
        # A priors file that silently stops applying is worse than no priors
        # file, because it looks like it is still working.
        names = [by_id[p].get("name", p) for p in sorted(unused)]
        raise SystemExit(
            f"goalie_priors.yaml names {len(names)} goalie(s) not in the projection pool: "
            f"{names}. Fix the nhl_id, or remove the entry - an override that does "
            f"nothing is worse than none, because it still looks applied."
        )

    board = pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)
    return board, draws_by_id
