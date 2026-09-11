"""Turn a finished board directory into the files a draft is run from.

    python -m hockey.export artifacts/board_v2

Reads what a fitting run left behind - the board, the per-category table and
the saved draws - and adds everything that depends on the league's shape rather
than on the posterior: replacement level per position, value over replacement,
tier breaks, and the positional drop-off curve.

Kept separate from the fit on purpose. These numbers change when the roster
changes, when a keeper is declared, or when a position dries up mid-draft, and
none of that should cost a refit.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from hockey.export import (
    add_value_over_replacement,
    replacement_levels,
    replacement_slots,
    scarcity,
    tiers,
)
from hockey.yahoo.settings import load_roster_from_yaml

logger = logging.getLogger(__name__)

N_TEAMS = 14


def load_draws(out: Path) -> tuple[list[int], np.ndarray]:
    """Fantasy-point draws for every player, and the ids they belong to.

    Batches are stacked side by side. They are conditionally independent given
    the shared parameters, which is exactly the assumption the staged fit makes,
    so a column from one batch and a column from another may be compared draw
    for draw.
    """
    ids: list[int] = []
    blocks = []
    for path in sorted(out.glob("draws_batch_*.npz")):
        with np.load(path) as f:
            ids.extend(int(p) for p in f["player_ids"])
            blocks.append(f["draws"].astype("float32"))
    if not blocks:
        raise SystemExit(f"no draws_batch_*.npz in {out}; rerun the board")
    return ids, np.concatenate(blocks, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m hockey.export")
    parser.add_argument("directory", help="a board directory, e.g. artifacts/board_v2")
    parser.add_argument("--teams", type=int, default=N_TEAMS)
    parser.add_argument(
        "--no-bench",
        action="store_true",
        help="set replacement at the starting slots only, ignoring bench depth",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    out = Path(args.directory)
    board = pd.read_csv(out / "draft_board.csv")
    if "position" not in board:
        raise SystemExit(
            f"{out}/draft_board.csv has no position column, so nothing here can be "
            f"measured against a roster slot. It predates the positional export; "
            f"rerun hockey.model.run_staged."
        )

    roster = load_roster_from_yaml()
    slots = replacement_slots(roster, args.teams, bench_to_skaters=not args.no_bench)
    # The pool is skaters only until the goalie model exists, so a goalie slot
    # would set a replacement level against an empty pool.
    slots = {p: n for p, n in slots.items() if p in set(board["position"])}
    levels = replacement_levels(board, slots)
    valued = add_value_over_replacement(board, levels)

    ids, draws = load_draws(out)
    valued["tier"] = valued["player_id"].map(tiers(valued, draws, ids))
    curve = scarcity(valued, levels)

    valued.to_csv(out / "value_board.csv", index=False)
    levels.to_csv(out / "replacement_levels.csv", index=False)
    curve.to_csv(out / "scarcity.csv", index=False)

    print(f"\n=== replacement level, {args.teams}-team league ===")
    print(levels.round(1).to_string(index=False))
    print("\n=== top 25 by value over replacement ===")
    columns = ["player", "position", "team", "age", "mean", "vorp", "pos_rank", "tier"]
    print(valued.head(25)[columns].round(1).to_string(index=False))
    print(f"\nwrote {out}/value_board.csv, replacement_levels.csv and scarcity.csv")


if __name__ == "__main__":
    main()
