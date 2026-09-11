"""Recompute the board's summary columns from the saved per-batch draws.

Any quantile or probability is a pass over the saved draws, not a refit, so a
change to what the board reports (a new percentile, a new threshold) is
applied here to an existing run. Usage:

    python scripts/rebuild_board.py artifacts/board_lean
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from hockey.export.draft import CEILING_PCT, FLOOR_PCT, HIGH_PCT, LOW_PCT


def main(out: Path) -> None:
    board = pd.read_csv(out / "draft_board.csv").set_index("player_id")
    for path in sorted(out.glob("draws_batch_*.npz")):
        with np.load(path) as f:
            ids, draws = f["player_ids"], f["draws"].astype("float64")
        q = np.percentile(draws, [FLOOR_PCT, LOW_PCT, HIGH_PCT, CEILING_PCT], axis=0)
        board.loc[ids, "mean"] = draws.mean(axis=0)
        board.loc[ids, "floor"] = q[0]
        board.loc[ids, "p20"] = q[1]
        board.loc[ids, "p80"] = q[2]
        board.loc[ids, "ceiling"] = q[3]
        board.loc[ids, "spread"] = q[3] - q[0]
        board.loc[ids, "sd"] = draws.std(axis=0)
    order = ["player", "games", "mean", "floor", "p20", "p80", "ceiling", "spread", "sd"]
    order += [c for c in board.columns if c not in order]
    board = board[order].sort_values("mean", ascending=False).reset_index()
    board.to_csv(out / "draft_board.csv", index=False)
    print(board.head(20).round(1).to_string(index=False))


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/board_lean"))
