"""Build a full-pool draft board in two stages.

    python -m hockey.model.run_staged --pool 400 --stage-one 120 --batch 40

Stage one fits the shared parameters jointly on a sample. Stage two fits the
whole pool in batches with those held fixed, so memory stays flat as the pool
grows. Each batch writes its board as it finishes, so an interrupted run leaves
usable output rather than nothing.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from hockey.db import SessionLocal
from hockey.export import rank
from hockey.features import (
    aging,
    availability_panel,
    build_index_maps,
    skater_panel,
)
from hockey.keepawake import keep_awake
from hockey.model import multi, shared, staged
from hockey.model.birthdates import player_birth_dates
from hockey.model.run import build_schedule, choose_pool
from hockey.progress import Progress
from hockey.seasons import PROJECTION_SEASON, season_label
from hockey.yahoo.settings import load_scoring_from_yaml

logger = logging.getLogger(__name__)


def prepare_for(session, players: pd.DataFrame, maps, curve) -> multi.MultiData:
    ids = [int(p) for p in players["player_id"]]
    return multi.prepare(
        panel=skater_panel(session, maps, player_ids=ids),
        availability=availability_panel(session, player_ids=ids),
        schedule=build_schedule(session, maps, players),
        maps=maps,
        player_names={int(r.player_id): r.name for r in players.itertuples()},
        positions={int(r.player_id): r.position for r in players.itertuples()},
        birth_dates=player_birth_dates(session, ids),
        aging_curve=curve,
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m hockey.model.run_staged")
    parser.add_argument("--pool", type=int, default=400)
    parser.add_argument("--stage-one", type=int, default=120)
    parser.add_argument("--batch", type=int, default=40)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--tune", type=int, default=800)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--out", default="artifacts/board_full")
    parser.add_argument(
        "--opponent",
        action="store_true",
        help="keep the per-opponent strength term. Off by default: it moves a "
        "projection by about 2%% and blocks collapsing the likelihood 33-fold.",
    )
    parser.add_argument(
        "--idio-walks",
        nargs="*",
        default=["hits"],
        help="categories that keep their own walk (default: hits, the one whose walk "
        "carries real spread). Pass with no names for none.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with keep_awake("full-pool draft board"):
        _main(args)


def _main(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scoring = load_scoring_from_yaml()

    with SessionLocal() as session:
        maps = build_index_maps(session)
        curve = aging.measure(session)
        pool = choose_pool(session, args.pool)
        logger.info("pool: %d skaters", len(pool))

        # --- stage one ---
        shared_path = out / "shared.npz"
        if shared_path.exists():
            params = shared.load(shared_path)
            logger.info("reusing shared parameters from %d players", params.n_source_players)
        else:
            head = pool.head(args.stage_one)
            stage_one_data = prepare_for(session, head, maps, curve)
            idata, params = staged.fit_stage_one(
                stage_one_data,
                args.draws,
                args.tune,
                args.chains,
                include_opponent=args.opponent,
                include_idio=tuple(args.idio_walks),
            )
            shared.save(params, shared_path)
            print("\n=== stage one ===")
            print(f"  {staged.diagnose(idata, 'stage one')}")
            print(f"  memory still free {staged.available_memory_gb():.1f} GB")
            del idata

        # --- stage two ---
        boards, diagnostics, draw_blocks = [], [], []
        batches = staged.batch_players(pool, args.batch)
        progress = Progress(total=len(batches), label="batch", path=out / "progress.txt")
        for i, batch in enumerate(batches, start=1):
            label = f"batch {i} ({len(batch)} players)"
            logger.info("stage two: %s", label)
            data = prepare_for(session, batch, maps, curve)
            idata = staged.fit_batch(
                data,
                params,
                args.draws,
                args.tune,
                args.chains,
                include_opponent=args.opponent,
                include_idio=tuple(args.idio_walks),
            )
            board, draws = staged.projected_board(idata, data, scoring, shared=params)
            boards.append(board)
            draw_blocks.append((list(data.players), draws))
            row = staged.diagnose(idata, label)
            row["free_gb_after"] = round(staged.available_memory_gb(), 1)
            diagnostics.append(row)
            progress.advance(f"r_hat {row['worst_rhat']:.4f}, {row['free_gb_after']} GB free")
            # Write after every batch, so an interrupted run still leaves a board.
            staged.stack_boards(boards).to_csv(out / "draft_board.csv", index=False)
            pd.DataFrame(diagnostics).to_csv(out / "diagnostics.csv", index=False)
            np.savez_compressed(
                out / f"draws_batch_{i:02d}.npz",
                player_ids=np.array(data.players),
                draws=draws.astype("float32"),
            )
            del idata

    progress.finish()
    full = staged.stack_boards(boards)
    full.to_csv(out / "draft_board.csv", index=False)
    diag = pd.DataFrame(diagnostics)

    # Head-to-head across every batch. Batches are independent given the shared
    # parameters, so their draws stack side by side.
    all_ids = [pid for ids, _ in draw_blocks for pid in ids]
    stacked = np.concatenate([d for _, d in draw_blocks], axis=1)
    name_of = dict(zip(full["player_id"], full["player"], strict=True))
    h2h = staged.head_to_head_from_draws(stacked, [name_of[p] for p in all_ids])
    h2h.to_csv(out / "head_to_head.csv")

    print(f"\n=== {season_label(PROJECTION_SEASON)} draft board, {len(full)} skaters ===")
    print(full.drop(columns="player_id").head(40).round(1).to_string(index=False))

    print("\n=== per-batch convergence ===")
    print(diag.to_string(index=False))
    bad = diag[(diag["worst_rhat"] > 1.01) | (diag["divergences"] > 0)]
    if len(bad):
        print(f"\n  {len(bad)} of {len(diag)} batches did not converge cleanly.")

    print("\n=== safe vs upside, top 15 ===")
    for appetite in ("safe", "upside"):
        top = rank(full, appetite).head(15)[["rank", "player", "age", "floor", "mean", "ceiling"]]
        print(f"\n  {appetite}:")
        print("   " + top.round(0).to_string(index=False).replace("\n", "\n   "))

    print(f"\nwrote {out}/draft_board.csv and diagnostics.csv")


if __name__ == "__main__":
    main()
