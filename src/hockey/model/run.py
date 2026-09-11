"""Fit the full model and produce a draft board.

  python -m hockey.model.run --top 40
  python -m hockey.model.run --players "Connor McDavid" "Cale Makar"

Picks the player pool by recent games played unless names are given, fits the
five-category model with availability, and writes a board with floor, ceiling
and head-to-head probabilities in fantasy points under the league config.
"""

import argparse
import logging
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from sqlalchemy import text

from hockey.db import SessionLocal
from hockey.export import beats, draft_board, rank
from hockey.features import (
    aging,
    availability_panel,
    build_index_maps,
    find_player,
    skater_panel,
    team_schedule,
)
from hockey.keepawake import keep_awake
from hockey.model import multi, project_multi
from hockey.model.birthdates import player_birth_dates
from hockey.seasons import PROJECTION_SEASON, SEASON_LENGTH, season_label
from hockey.yahoo.settings import load_scoring_from_yaml

logger = logging.getLogger(__name__)

ARTIFACTS = Path("artifacts/board")

# The pool: skaters with enough recent NHL time to be draftable, ranked by what
# they were actually worth in this league rather than by goals and assists.
# Ranking on scoring alone was wrong twice over. It missed the defencemen and
# checkers whose value is hits and blocks - half a point each here, which adds
# up to more than a 20-goal season - and it left the pool far too thin at
# defence, where 14 teams draft 84 and a ranking by points supplied 64. A
# replacement level measured off a pool that runs out before the league does is
# not a replacement level.
#
# The column expressions are fixed here and the weights are bound parameters, so
# the league config drives the arithmetic without any of it reaching SQL as text.
STAT_COLUMNS = {
    "goals": "s.goals",
    "assists": "s.assists",
    "plus_minus": "s.plus_minus",
    "ppp": "coalesce(s.ppp, 0)",
    "shp": "coalesce(s.shp, 0)",
    "sog": "s.sog",
    "hits": "s.hits",
    "blocks": "s.blocks",
}

POOL_SQL = """
SELECT s.player_id,
       p.first_name || ' ' || p.last_name AS name,
       p.position,
       p.team_abbrev AS current_team,
       count(*) AS recent_games,
       sum({points})::float AS recent_points
  FROM skater_game_logs s
  JOIN nhl_games g ON g.nhl_game_id = s.game_id
  JOIN players p ON p.nhl_id = s.player_id
 WHERE g.game_type = 2
   AND g.season = ANY(:recent)
   AND p.team_abbrev IS NOT NULL
   AND p.position <> 'G'
 GROUP BY s.player_id, p.first_name, p.last_name, p.position, p.team_abbrev
HAVING count(*) >= :min_games
 ORDER BY recent_points DESC
"""


def _points_expression(scoring) -> tuple[str, dict]:
    terms, params = [], {}
    for rule in scoring.skater_rules:
        column = STAT_COLUMNS.get(rule.key)
        if column is None:
            raise KeyError(
                f"category {rule.key!r} has no game-log column, so the pool cannot be "
                f"ranked on what this league actually scores"
            )
        name = f"w_{rule.key}"
        terms.append(f":{name} * {column}")
        params[name] = float(rule.modifier)
    return " + ".join(terms), params


def choose_pool(
    session,
    limit: int,
    recent=(20242025, 20252026),
    min_games=60,
    scoring=None,
    quotas: dict[str, int] | None = None,
) -> pd.DataFrame:
    """The draftable pool, ranked by recent fantasy points in this league.

    `quotas` takes the top N at each position instead of the top N overall.
    Without it a single ranked list decides the mix, and the mix it produces is
    not the one the roster needs: centres are the deepest scoring position, so
    they crowd out the defencemen the league is obliged to draft.
    """
    if scoring is None:
        scoring = load_scoring_from_yaml()
    expression, weights = _points_expression(scoring)
    rows = session.execute(
        text(POOL_SQL.format(points=expression)),
        {"recent": list(recent), "min_games": min_games, **weights},
    ).mappings()
    pool = pd.DataFrame(rows)

    if quotas:
        kept = [
            group.head(quotas.get(position, 0))
            for position, group in pool.groupby("position", sort=False)
        ]
        pool = pd.concat(kept).sort_values("recent_points", ascending=False)
        short = {
            position: (quotas[position], int((pool["position"] == position).sum()))
            for position in quotas
            if int((pool["position"] == position).sum()) < quotas[position]
        }
        if short:
            logger.warning("pool is short of its quota at %s (wanted, got)", short)
        return pool.reset_index(drop=True)
    return pool.head(limit).reset_index(drop=True)


def named_pool(session, names: list[str]) -> pd.DataFrame:
    out = []
    for name in names:
        hits = find_player(session, name)
        if hits.empty:
            raise SystemExit(f"no player matching {name!r}")
        out.append(hits.iloc[0])
    return pd.DataFrame(out).reset_index(drop=True)


def build_schedule(session, maps, players: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for row in players.itertuples():
        games = team_schedule(session, maps, row.current_team, PROJECTION_SEASON)
        if games.empty:
            raise SystemExit(f"no {PROJECTION_SEASON} schedule for {row.current_team}")
        games = games.copy()
        games["player_id"] = row.player_id
        frames.append(games)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m hockey.model.run")
    parser.add_argument("--players", nargs="+", default=None)
    parser.add_argument("--top", type=int, default=40, help="pool size when no names are given")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1500)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument(
        "--out",
        default=None,
        help="artifact directory; defaults to artifacts/board. Give each run its "
        "own so a later one does not overwrite an earlier one's board.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with keep_awake("draft board fit"):
        _main(args)


def _main(args) -> None:
    out_dir = Path(args.out) if args.out else ARTIFACTS
    out_dir.mkdir(parents=True, exist_ok=True)

    with SessionLocal() as session:
        players = (
            named_pool(session, args.players) if args.players else choose_pool(session, args.top)
        )
        player_ids = [int(p) for p in players["player_id"]]
        logger.info("pool: %d skaters", len(players))

        maps = build_index_maps(session)
        panel = skater_panel(session, maps, player_ids=player_ids)
        availability = availability_panel(session, player_ids=player_ids)
        schedule = build_schedule(session, maps, players)
        logger.info(
            "panel: %d player-games, %d player-seasons of availability, %d scheduled games",
            len(panel),
            len(availability),
            len(schedule),
        )

        curve = aging.measure(session)
        births = player_birth_dates(session, player_ids)
        data = multi.prepare(
            panel=panel,
            availability=availability,
            schedule=schedule,
            maps=maps,
            player_names={int(r.player_id): r.name for r in players.itertuples()},
            positions={int(r.player_id): r.position for r in players.itertuples()},
            birth_dates=births,
            aging_curve=curve,
        )

    model = multi.build(data)
    logger.info("sampling %d draws x %d chains", args.draws, args.chains)
    idata = multi.sample(model, draws=args.draws, tune=args.tune, chains=args.chains)

    divergences = int(idata.sample_stats["diverging"].sum())
    diag = az.summary(idata, var_names=["mu_player", "b_home", "loading"], round_to=4)
    print("\n=== convergence ===")
    print(f"  worst r_hat       {diag['r_hat'].max():.4f}  (want < 1.01)")
    print(f"  lowest bulk ESS   {diag['ess_bulk'].min():.0f}      (want > 400)")
    print(f"  divergences       {divergences}       (want 0)")
    if diag["r_hat"].max() > 1.01 or divergences:
        print("  NOT CONVERGED - treat what follows as provisional.")

    print("\n=== how much each category follows a player's shared form ===")
    load = az.summary(idata, var_names=["loading"], round_to=3)
    load.index = [f"loading[{s}]" for s in data.stats]
    print(load[["mean", "sd"]].to_string())

    scoring = load_scoring_from_yaml()
    projection = project_multi.project(idata, data)
    board = draft_board(projection, scoring)
    # draft_board sorts by mean and resets the index, so a list built in
    # projection-player order lands against the wrong rows. Join on the id.
    expected = pd.Series(
        projection.games_played.mean(axis=0), index=projection.players, name="exp_games"
    )
    board["exp_games"] = board["player_id"].map(expected)

    games = SEASON_LENGTH[PROJECTION_SEASON]
    print(f"\n=== {season_label(PROJECTION_SEASON)} draft board, fantasy points ===")
    print(f"    ({len(data.stats)} categories; schedule is {games} games)")
    print(board.drop(columns="player_id").round(1).to_string(index=False))

    # What availability is worth: the same posterior with and without it.
    full = project_multi.project(idata, data, assume_full_season=True)
    naive = draft_board(full, scoring).set_index("player")
    real = board.set_index("player")
    delta = pd.DataFrame(
        {
            "floor_if_always_healthy": naive["floor"],
            "floor_with_availability": real["floor"],
            "floor_cost": naive["floor"] - real["floor"],
            "exp_games": real["exp_games"],
        }
    ).sort_values("floor_cost", ascending=False)
    print("\n=== what the availability component is worth (biggest floor hits) ===")
    print(delta.head(12).round(1).to_string())

    print("\n=== safe vs upside, top 10 ===")
    safe = rank(board, "safe").head(10)[["rank", "player", "floor", "mean", "ceiling"]]
    upside = rank(board, "upside").head(10)[["rank", "player", "floor", "mean", "ceiling"]]
    print("\n  safe (rank on floor):")
    print("   " + safe.round(0).to_string(index=False).replace("\n", "\n   "))
    print("\n  upside (rank on ceiling):")
    print("   " + upside.round(0).to_string(index=False).replace("\n", "\n   "))

    # A set difference over two top-tens says nothing when the pool is small
    # and both lists hold everyone. What matters is how far a player moves.
    safe_rank = rank(board, "safe").set_index("player")["rank"]
    upside_rank = rank(board, "upside").set_index("player")["rank"]
    movement = (
        pd.DataFrame({"safe_rank": safe_rank, "upside_rank": upside_rank})
        .assign(shift=lambda d: d["upside_rank"] - d["safe_rank"])
        .sort_values("shift")
    )
    print("\n=== who moves between the two rankings ===")
    print("    (negative shift = better as an upside pick than as a safe one)")
    print(pd.concat([movement.head(4), movement.tail(4)]).drop_duplicates().to_string())
    print(
        f"\n  mean absolute rank shift: {movement['shift'].abs().mean():.1f} places "
        f"across {len(movement)} players"
    )

    board.to_csv(out_dir / "draft_board.csv", index=False)
    np.save(
        out_dir / "fantasy_points_draws.npy",
        np.asarray([projection.totals[s] for s in data.stats]),
    )
    beats(projection, scoring).to_csv(out_dir / "head_to_head.csv")
    idata.to_netcdf(str(out_dir / "trace.nc"))
    print(f"\nwrote {ARTIFACTS}/draft_board.csv, head_to_head.csv, trace.nc")


if __name__ == "__main__":
    main()
