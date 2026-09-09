"""The MVP gate: goals and assists, off the warehouse, for a handful of players.

This exists to prove the whole path works end to end - Postgres to panel to
posterior to a floor/ceiling summary and a head-to-head probability - before
any of it is extended to every category, every player, availability or
calibration. It is the checkpoint the plan puts before the rest of the model
work.

Known limitation, stated rather than hidden: there is no availability component
yet, so these totals assume every player dresses for every game. They are
therefore optimistic, and the ceiling more so than the floor. Availability is
the next piece of work.

Run:
  python -m hockey.model.mvp --players "Connor McDavid" "Cale Makar" "Jack Hughes"
"""

import argparse
import logging
from pathlib import Path

import arviz as az
import pandas as pd

from hockey.db import SessionLocal
from hockey.features import build_index_maps, find_player, skater_panel, team_schedule
from hockey.model import forecast, skater
from hockey.seasons import PROJECTION_SEASON, SEASON_LENGTH, season_label

logger = logging.getLogger(__name__)

ARTIFACTS = Path("artifacts/mvp")

DEFAULT_PLAYERS = ["Connor McDavid", "Cale Makar", "Nathan MacKinnon"]


def resolve_players(session, names: list[str]) -> pd.DataFrame:
    """Name -> (player_id, position, current team), taking the player with the
    most games when a name is ambiguous, and saying so."""
    rows = []
    for name in names:
        hits = find_player(session, name)
        if hits.empty:
            raise SystemExit(f"no player matching {name!r}")
        if len(hits) > 1:
            logger.warning(
                "%r matched %d players; taking %s (%d games)",
                name,
                len(hits),
                hits.iloc[0]["name"],
                hits.iloc[0]["games"],
            )
        rows.append(hits.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def build_projection_schedule(session, maps, players: pd.DataFrame) -> pd.DataFrame:
    """Each player's remaining games: their current team's schedule for the
    projected season, tagged with the player id."""
    frames = []
    for row in players.itertuples():
        if row.current_team is None:
            raise SystemExit(
                f"{row.name} has no current team in the warehouse, so there is no "
                f"schedule to project over. Run: python -m hockey.ingest players "
                f"--season {PROJECTION_SEASON}"
            )
        games = team_schedule(session, maps, row.current_team, PROJECTION_SEASON)
        if games.empty:
            raise SystemExit(
                f"no {PROJECTION_SEASON} schedule for {row.current_team}. Run: "
                f"python -m hockey.ingest schedule --season {PROJECTION_SEASON}"
            )
        games = games.copy()
        games["player_id"] = row.player_id
        frames.append(games)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m hockey.model.mvp")
    parser.add_argument("--players", nargs="+", default=DEFAULT_PLAYERS)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    with SessionLocal() as session:
        players = resolve_players(session, args.players)
        player_ids = [int(p) for p in players["player_id"]]
        logger.info(
            "players: %s",
            ", ".join(f"{r.name} ({r.position}, {r.current_team})" for r in players.itertuples()),
        )

        maps = build_index_maps(session)
        logger.info("window: %d seasons, %d teams", maps.n_seasons, maps.n_teams)

        panel = skater_panel(session, maps, player_ids=player_ids)
        if panel.empty:
            raise SystemExit("no game logs for these players; is the backfill still running?")
        # A requested player with no games would otherwise drop out of the fit
        # while still appearing in the schedule, which is how a projection for
        # the wrong set of players gets produced quietly.
        no_games = set(player_ids) - set(panel["player_id"].unique())
        if no_games:
            names = players.set_index("player_id").loc[sorted(no_games), "name"].tolist()
            raise SystemExit(
                f"no game logs in the fitting window for: {', '.join(names)}. Either the "
                f"backfill has not reached their seasons yet, or they did not play in it. "
                f"Check with: python -m hockey.ingest.coverage"
            )
        by_id = players.set_index("player_id")["name"]
        logger.info(
            "panel: %d player-games (%s)",
            len(panel),
            ", ".join(f"{by_id[pid]}: {n}" for pid, n in panel.groupby("player_id").size().items()),
        )

        schedule = build_projection_schedule(session, maps, players)
        logger.info("projection: %d games over %d players", len(schedule), len(players))

        data = skater.prepare(
            panel=panel,
            schedule=schedule,
            maps=maps,
            player_names={int(r.player_id): r.name for r in players.itertuples()},
            positions={int(r.player_id): r.position for r in players.itertuples()},
        )

    model = skater.build(data)
    logger.info("sampling (%d draws, %d tune, %d chains)", args.draws, args.tune, args.chains)
    idata = skater.sample(model, draws=args.draws, tune=args.tune, chains=args.chains)

    # Convergence first. A summary table from chains that did not converge is a
    # set of numbers, not evidence.
    latent = [f"{s}_mu_player" for s in data.stats] + [f"{s}_b_home" for s in data.stats]
    diagnostics = az.summary(idata, var_names=latent, round_to=3)
    worst_rhat = float(diagnostics["r_hat"].max())
    lowest_ess = float(diagnostics["ess_bulk"].min())
    divergences = int(idata.sample_stats["diverging"].sum())

    print("\n=== convergence ===")
    print(f"  worst r_hat       {worst_rhat:.4f}  (want < 1.01)")
    print(f"  lowest bulk ESS   {lowest_ess:.0f}      (want > 400)")
    print(f"  divergences       {divergences}       (want 0)")
    if worst_rhat > 1.01 or divergences:
        print("  NOT CONVERGED - treat the numbers below as provisional.")

    projection = forecast.project(idata, data)
    # Points is summed per draw, not from the means, so its spread carries the
    # correlation between goals and assists.
    points = projection.totals["goals"] + projection.totals["assists"]
    summary = forecast.summarize(projection, {"points": points})

    # 84, not 82: the regular season expands in 2026-27.
    games = SEASON_LENGTH[PROJECTION_SEASON]
    print(
        f"\n=== projected {season_label(PROJECTION_SEASON)} totals (assumes all {games} games) ==="
    )
    print(summary.round(2).to_string(index=False))

    print("\n=== P(row outscores column, points) ===")
    print(forecast.head_to_head(projection, points).round(3).to_string())

    summary.to_csv(ARTIFACTS / "mvp_summary.csv", index=False)
    idata.to_netcdf(str(ARTIFACTS / "mvp_trace.nc"))
    print(f"\nwrote {ARTIFACTS / 'mvp_summary.csv'} and {ARTIFACTS / 'mvp_trace.nc'}")


if __name__ == "__main__":
    main()
