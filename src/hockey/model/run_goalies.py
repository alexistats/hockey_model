"""Fit the goalie model and write a goalie board.

    python -m hockey.model.run_goalies --out artifacts/goalies_v1

About ten minutes at the defaults. Unlike the skater board there is no staging:
there are only ~210 goalies across eight seasons and the aggregated likelihood
is 843 rows, so the whole thing fits jointly in one pass.
"""

import argparse
import logging
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from hockey.db import SessionLocal
from hockey.keepawake import keep_awake
from hockey.model import goalie, goalie_forecast, multi
from hockey.seasons import PROJECTION_SEASON, SEASON_LENGTH
from hockey.yahoo.settings import load_scoring_from_yaml

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m hockey.model.run_goalies")
    parser.add_argument("--out", default="artifacts/goalies_v1")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1500)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--min-starts", type=int, default=15, help="board cutoff, not fit cutoff")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with SessionLocal() as session:
        frame = goalie.panel(session)
        if frame.empty:
            raise SystemExit("no goalie game logs; run the ingest first")
        games = {s: goalie.team_games(session, s) for s in sorted(frame["season"].unique())}
    frame["team_games"] = [games[r.season].get(r.team, 82) for r in frame.itertuples()]

    index = goalie.GoalieIndex(frame, PROJECTION_SEASON)
    logger.info(
        "%d goalie-seasons, %d goalies, %d teams, projecting %d",
        len(frame),
        index.n_goalies,
        index.n_teams,
        PROJECTION_SEASON,
    )

    with keep_awake("goalie fit"):
        model = goalie.build(frame, index)
        idata = multi.sample(
            model, draws=args.draws, tune=args.tune, chains=args.chains, seed=PROJECTION_SEASON
        )

    # Save the posterior before computing anything from it. Diagnostics are
    # cheap to redo and a fit is not; the first version of this file worked out
    # r-hat first and lost a completed sample to an AttributeError in arviz.
    idata.to_netcdf(str(out / "posterior.nc"))
    logger.info("posterior saved to %s", out / "posterior.nc")

    # Convergence travels with the numbers, not in a separate conversation.
    watched = [
        "mu_save",
        "mu_shots",
        "mu_win",
        "mu_share",
        "mu_shutout",
        "sigma_goalie_save",
        "sigma_team_save",
        "sigma_team_win",
        "sigma_goalie_share",
        "sigma_walk_share",
        "alpha_shots",
        "beta_save_on_win",
    ]
    summary = az.summary(idata, var_names=watched)
    summary.to_csv(out / "parameters.csv")
    diagnostics = pd.DataFrame(
        [
            {
                "goalies": index.n_goalies,
                "rows": len(frame),
                "draws": args.draws,
                "tune": args.tune,
                "chains": args.chains,
                "worst_rhat": _worst(az.rhat(idata), "max"),
                "lowest_ess": _worst(az.ess(idata), "min"),
                "divergences": int(idata.sample_stats["diverging"].sum()),
            }
        ]
    )
    diagnostics.to_csv(out / "diagnostics.csv", index=False)

    scoring = {r.key: r.modifier for r in load_scoring_from_yaml().goalie_rules}
    priors = goalie_forecast.load_priors()
    board, draws = goalie_forecast.project(
        idata,
        frame,
        index,
        scoring,
        season_games=SEASON_LENGTH[PROJECTION_SEASON],
        priors=priors,
    )

    # The board is every goalie the model knows; the cutoff is only about what
    # is worth reading. A goalie left out is not a goalie judged bad.
    board = board[board["last_season_starts"] >= args.min_starts].reset_index(drop=True)
    board.insert(0, "player", _names(board["player_id"]))
    board.to_csv(out / "goalie_board.csv", index=False)
    np.savez_compressed(
        out / "goalie_draws.npz",
        player_ids=np.array(list(draws), dtype="int64"),
        draws=np.column_stack([draws[p] for p in draws]),
    )
    worst = diagnostics["worst_rhat"].iloc[0]
    print(f"\n=== goalie board, {PROJECTION_SEASON} ===")
    columns = ["player", "team", "mean", "floor", "ceiling", "exp_starts", "exp_wins", "save_pct"]
    print(board.head(20)[columns].round(3).to_string(index=False))
    print(
        f"\nworst r-hat {worst:.4f}, "
        f"{int(diagnostics['divergences'].iloc[0])} divergences, "
        f"lowest ESS {diagnostics['lowest_ess'].iloc[0]:.0f}"
    )
    if worst > 1.01:
        print("r-hat above 1.01: treat these as provisional and rerun with more tuning.")
    print(f"wrote {out}/goalie_board.csv, goalie_draws.npz, parameters.csv, diagnostics.csv")


def _worst(diagnostic, how: str) -> float:
    """The worst r-hat or lowest ESS across every variable.

    Written per variable rather than with `.to_array()` for two reasons, both
    met on this model: arviz 1.3 hands back a DataTree, which has no
    `to_array`, and converting to a Dataset and stacking it broadcasts every
    variable to a common shape - with a 210-goalie walk in the posterior that
    allocation is larger than numpy will make.
    """
    dataset = diagnostic.to_dataset() if hasattr(diagnostic, "to_dataset") else diagnostic
    values = [
        float(dataset[name].max() if how == "max" else dataset[name].min())
        for name in dataset.data_vars
    ]
    return max(values) if how == "max" else min(values)


def _names(ids: pd.Series) -> list[str]:
    from sqlalchemy import text

    with SessionLocal() as session:
        rows = session.execute(
            text(
                "SELECT nhl_id, first_name || ' ' || last_name FROM players WHERE nhl_id = ANY(:i)"
            ),
            {"i": [int(i) for i in ids]},
        ).all()
    by_id = {int(r[0]): r[1] for r in rows}
    return [by_id.get(int(i), str(i)) for i in ids]


if __name__ == "__main__":
    main()
