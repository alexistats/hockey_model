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
from hockey.export.context import write_draft_context
from hockey.seasons import PROJECTION_SEASON
from hockey.yahoo.eligibility import load_csv as load_eligibility
from hockey.yahoo.settings import load_roster_from_yaml

logger = logging.getLogger(__name__)

N_TEAMS = 14

# What a goalie scores in, as written by hockey.model.goalie_forecast.
GOALIE_COLUMNS = ("exp_starts", "exp_wins", "exp_saves", "exp_ga", "exp_shutouts", "save_pct")


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


def merge_goalies(
    board: pd.DataFrame, directory: Path
) -> tuple[pd.DataFrame, list[int], np.ndarray]:
    """Fold a goalie board into the skater board so one ranking covers both.

    Goalies are fitted by a separate model - they share no parameters with
    skaters and nothing about a goalie's season informs a winger's - so the two
    posteriors are independent. That independence is what makes stacking them
    legitimate: a goalie draw and a skater draw are not the same posterior
    sample, but comparing independent draws is exactly how P(A > B) is defined
    for two unrelated quantities.

    What they do share is the league. Replacement level, value over replacement
    and tiers are all properties of the roster, so once both are on one board
    the existing functions price a goalie against a goalie slot without knowing
    anything about how he was fitted.
    """
    goalies = pd.read_csv(directory / "goalie_board.csv")
    with np.load(directory / "goalie_draws.npz") as f:
        goalie_ids = [int(x) for x in f["player_ids"]]
        goalie_draws = f["draws"].astype("float32")

    keep = set(goalies["player_id"].astype(int))
    order = [i for i, pid in enumerate(goalie_ids) if pid in keep]
    goalie_ids = [goalie_ids[i] for i in order]
    goalie_draws = goalie_draws[:, order]

    rows = goalies.rename(columns={"player": "player"}).copy()
    rows["position"] = "G"
    rows["eligible"] = [("G",)] * len(rows)
    # The skater board carries columns a goalie has no analogue for. Filling
    # them with the goalie's own equivalents where one exists, and leaving the
    # rest absent, keeps a downstream reader from quietly reading a zero as a
    # measurement.
    rows["games"] = rows["exp_starts"].round().astype(int)
    rows["exp_games"] = rows["exp_starts"]
    rows["spread"] = rows["ceiling"] - rows["floor"]
    for column in board.columns:
        if column not in rows:
            rows[column] = np.nan
    # Carry the goalie-only columns through rather than projecting onto the
    # skater board's shape. The draft page needs them to show a goalie's own
    # categories, and a skater's NaN in "exp_wins" is honestly an absence.
    keep = [*board.columns, *(c for c in GOALIE_COLUMNS if c in rows)]
    for column in keep:
        if column not in board:
            board[column] = np.nan
    merged = pd.concat([board[keep], rows[keep]], ignore_index=True)
    return merged, goalie_ids, goalie_draws


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m hockey.export")
    parser.add_argument("directory", help="a board directory, e.g. artifacts/board_v2")
    parser.add_argument("--teams", type=int, default=N_TEAMS)
    parser.add_argument(
        "--goalies",
        default=None,
        help="a goalie board directory, e.g. artifacts/goalies_v1. Without it the "
        "board is skaters only and the goalie slots are left unpriced.",
    )
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

    # Yahoo eligibility when we have it, NHL primary position otherwise. The
    # difference is not cosmetic: a player Yahoo lists at two positions is
    # drafted out of whichever pool is shorter, and one Yahoo lists somewhere
    # other than their NHL position is being priced against the wrong pool
    # entirely.
    eligibility = load_eligibility(Path(f"config/eligibility_{PROJECTION_SEASON // 10000}.csv"))
    if eligibility:
        covered = board["player_id"].isin(eligibility)
        board["eligible"] = [
            eligibility.get(int(p), (pos,))
            for p, pos in zip(board["player_id"], board["position"], strict=True)
        ]
        logger.info(
            "Yahoo eligibility for %d of %d players; the other %d fall back to their "
            "NHL primary position",
            int(covered.sum()),
            len(board),
            int((~covered).sum()),
        )
    else:
        logger.warning(
            "no eligibility file, so positions are NHL primary positions and every "
            "dual-eligible player is undervalued. Build one with: "
            "python -m hockey.yahoo eligibility <paste file>"
        )

    goalie_ids: list[int] = []
    goalie_draws = None
    if args.goalies:
        board, goalie_ids, goalie_draws = merge_goalies(board, Path(args.goalies))
        logger.info("merged %d goalies from %s", len(goalie_ids), args.goalies)
    else:
        logger.warning(
            "no --goalies directory, so the board is skaters only and the two goalie "
            "slots per team are left unpriced. Fit one with hockey.model.run_goalies."
        )

    roster = load_roster_from_yaml()
    slots = replacement_slots(roster, args.teams, bench_to_skaters=not args.no_bench)
    # A slot with no pool behind it would set replacement against nothing, so
    # positions absent from the board are dropped rather than priced at zero.
    known = set(board["position"])
    if "eligible" in board:
        known |= {p for e in board["eligible"] for p in e}
    slots = {p: n for p, n in slots.items() if p in known}
    levels = replacement_levels(board, slots)
    valued = add_value_over_replacement(board, levels)

    ids, draws = load_draws(out)
    if goalie_draws is not None:
        if goalie_draws.shape[0] != draws.shape[0]:
            # Tiers compare columns draw for draw, so two posteriors of
            # different length cannot be stacked without silently truncating
            # one of them.
            raise SystemExit(
                f"the skater posterior has {draws.shape[0]} draws and the goalie "
                f"posterior {goalie_draws.shape[0]}; refit one of them so the two "
                f"match, rather than comparing a column against a shorter one"
            )
        ids = [*ids, *goalie_ids]
        draws = np.concatenate([draws, goalie_draws], axis=1)
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

    # The calendar and the injury list, so the draft-day server never needs a
    # database. Absent rather than empty when the warehouse is down: every
    # consumer checks, and a board that cannot answer a schedule question should
    # say so rather than answer it with zeros.
    if write_draft_context(out):
        print(f"wrote {out}/schedule.csv, weeks.csv, injuries.csv and form.csv")


if __name__ == "__main__":
    main()
