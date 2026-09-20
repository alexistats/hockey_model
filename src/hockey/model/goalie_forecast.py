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
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

PRIORS_PATH = Path("config/goalie_priors.yaml")


def load_priors(path: Path = PRIORS_PATH) -> dict:
    if not path.exists():
        return {"goalies": [], "teams": []}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {"goalies": loaded.get("goalies") or [], "teams": loaded.get("teams") or []}


def project(
    idata,
    frame: pd.DataFrame,
    index,
    scoring: dict[str, float],
    season_games: int,
    priors: dict | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """(summary per goalie, fantasy-point draws per goalie).

    Every quantity is drawn, not averaged: starts from the workload walk, shots
    from the team's rate given those starts, saves from those shots, and so on
    down. Summarising at each step instead would collapse exactly the spread the
    project exists to report.
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

    # Each goalie is projected on the team he finished on. A summer trade is
    # exactly the thing the priors file is for; there is nothing in the game
    # logs that could know it.
    last = frame.sort_values("season").groupby("player_id").tail(1)
    team_of = dict(zip(last["player_id"], last["team"], strict=True))
    recent = last.set_index("player_id")

    team_adjust = {t["team"]: t for t in priors["teams"]}
    by_id = {int(g["nhl_id"]): g for g in priors["goalies"] if g.get("nhl_id") is not None}
    unused = set(by_id)

    rows, draws_by_id = [], {}
    for pid in index.goalies:
        gi = index.goalie_of[pid]
        team = team_of.get(pid)
        if team is None or team not in index.team_of:
            continue
        ti = index.team_of[team]

        # --- workload ---
        override = by_id.get(int(pid))
        if override is not None and override.get("starts") is not None:
            unused.discard(int(pid))
            mean = float(override["starts"])
            sd = float(override.get("starts_sd", 6.0))
            starts = rng.normal(mean, sd, size=n_draws)
            starts = np.clip(np.rint(starts), 0, season_games).astype(int)
            logger.info(
                "workload override: %s starts ~ N(%.0f, %.0f)",
                override.get("name", pid),
                mean,
                sd,
            )
        else:
            p_share = 1.0 / (1.0 + np.exp(-(mu_share + share_walk[:, gi, step])))
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
        rows.append(
            {
                "player_id": int(pid),
                "team": team,
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
                "last_season": int(recent.loc[pid, "season"]),
                "last_season_starts": int(recent.loc[pid, "starts"]),
                "override": override is not None and override.get("starts") is not None,
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
