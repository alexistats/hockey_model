"""Fit a large pool in two stages.

Stage one fits the full joint model on a sample big enough to pin down the
values every player shares - about 3,800 of them. Stage two holds those and
fits the rest of the pool in batches, each of which carries only its own
players' latents, roughly 161 values each.

Memory then stops scaling with the size of the pool. A 400-player board costs
what a 40-player batch costs, ten times over, instead of the 19 GB that killed
a single 150-player joint fit on a 32 GB machine.

What this gives up: stage two conditions on stage one's answer rather than
re-estimating it, so uncertainty in the shared values is treated as resolved.
That is a real approximation, and `validate_batching` measures it by fitting
the same players both ways and comparing the intervals rather than assuming
they agree.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from hockey.model import multi, project_multi, shared

logger = logging.getLogger(__name__)


@dataclass
class StagedResult:
    boards: pd.DataFrame
    shared: shared.SharedParameters
    batch_diagnostics: pd.DataFrame


def fit_stage_one(
    data: multi.MultiData,
    draws: int,
    tune: int,
    chains: int,
    seed: int = 1,
    include_opponent: bool = True,
    include_idio: bool = False,
):
    """The joint fit whose only job is the shared parameters."""
    logger.info("stage one: %d players, joint fit", len(data.players))
    model = multi.build(data, include_opponent=include_opponent, include_idio=include_idio)
    idata = multi.sample(model, draws=draws, tune=tune, chains=chains, seed=seed)
    params = shared.extract(
        idata,
        data.maps,
        tuple(data.stats),
        len(data.players),
        structure=multi.model_structure(include_opponent, include_idio),
    )
    return idata, params


def fit_batch(
    data: multi.MultiData,
    params: shared.SharedParameters,
    draws: int,
    tune: int,
    chains: int,
    seed: int = 1,
    include_opponent: bool = True,
    include_idio: bool = False,
):
    """One batch of players, with the shared values held fixed."""
    model = multi.build(
        data, shared=params, include_opponent=include_opponent, include_idio=include_idio
    )
    return multi.sample(model, draws=draws, tune=tune, chains=chains, seed=seed)


def batch_players(players: pd.DataFrame, size: int) -> list[pd.DataFrame]:
    """Split a pool into batches. Order is preserved so a batch holds players
    of similar calibre, which keeps each fit's geometry uniform."""
    return [players.iloc[i : i + size].copy() for i in range(0, len(players), size)]


def diagnose(idata, label: str) -> dict:
    import arviz as az

    summary = az.summary(idata, var_names=["mu_player"], round_to=4)
    return {
        "batch": label,
        "worst_rhat": float(summary["r_hat"].max()),
        "lowest_ess": float(summary["ess_bulk"].min()),
        "divergences": int(idata.sample_stats["diverging"].sum()),
    }


def compare_intervals(joint_board: pd.DataFrame, staged_board: pd.DataFrame) -> pd.DataFrame:
    """How far the two-stage shortcut moves a board against a joint fit.

    The mean should barely move; the interval is where a shortcut that treats
    shared parameters as known would show up, by being too narrow.
    """
    a = joint_board.set_index("player")[["mean", "floor", "ceiling"]]
    b = staged_board.set_index("player")[["mean", "floor", "ceiling"]]
    both = a.join(b, lsuffix="_joint", rsuffix="_staged", how="inner")
    both["mean_diff"] = both["mean_staged"] - both["mean_joint"]
    both["width_joint"] = both["ceiling_joint"] - both["floor_joint"]
    both["width_staged"] = both["ceiling_staged"] - both["floor_staged"]
    both["width_ratio"] = both["width_staged"] / both["width_joint"]
    return both


def summarise_comparison(comparison: pd.DataFrame) -> str:
    mean_shift = comparison["mean_diff"].abs().mean()
    ratio = comparison["width_ratio"].mean()
    lines = [
        f"  players compared            {len(comparison)}",
        f"  mean absolute shift in mean {mean_shift:.1f} fantasy points",
        f"  mean 90% interval width     joint {comparison['width_joint'].mean():.0f}, "
        f"staged {comparison['width_staged'].mean():.0f}",
        f"  width ratio                 {ratio:.3f}   (1.0 = the shortcut costs nothing)",
    ]
    if ratio < 0.95:
        lines.append(
            f"  => the staged intervals are {100 * (1 - ratio):.0f}% narrower. That is the "
            f"cost of treating the shared parameters as known, and it makes floors "
            f"look better than they are."
        )
    else:
        lines.append("  => the shortcut does not materially narrow the intervals.")
    return "\n".join(lines)


def projected_board(idata, data: multi.MultiData, scoring, seed: int = 7, shared=None):
    """The board, the fantasy-point draws behind it, and the projection itself.

    The draws are returned rather than discarded because head-to-head needs
    full distributions, not summaries. A first version of this kept only the
    board, and P(A beats B) then could not be produced without refitting every
    batch - the traces were already gone. The projection carries the
    per-category draws for the same reason.
    """
    from hockey.export import draft_board, fantasy_points

    projection = project_multi.project(idata, data, seed=seed, shared=shared)
    board = draft_board(projection, scoring)
    expected = pd.Series(
        projection.games_played.mean(axis=0), index=projection.players, name="exp_games"
    )
    board["exp_games"] = board["player_id"].map(expected)
    board["age"] = board["player_id"].map(dict(zip(data.players, data.ages[:, -1], strict=True)))
    # Columns follow projection.players order, which is what the caller needs
    # to line draws up with player ids across batches.
    return board, fantasy_points(projection, scoring), projection


def head_to_head_from_draws(draws: np.ndarray, names: list[str], chunk: int = 200) -> pd.DataFrame:
    """P(row finishes ahead of column) for every pair, from stacked draws.

    Chunked over rows because 400 players against 3,200 draws is a 400x400x3200
    comparison if done in one go, which is 4 GB of booleans.
    """
    n = draws.shape[1]
    matrix = np.empty((n, n), dtype=float)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = draws[:, start:stop, None] > draws[:, None, :]
        matrix[start:stop] = block.mean(axis=0)
    np.fill_diagonal(matrix, np.nan)
    return pd.DataFrame(matrix, index=names, columns=names)


def available_memory_gb() -> float:
    """System memory still free. This is what decides whether the next batch
    survives, and a 150-player joint fit exhausting 32 GB is why it is logged
    rather than assumed."""
    import ctypes

    class Status(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = Status()
    status.dwLength = ctypes.sizeof(Status)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return status.ullAvailPhys / 2**30


def stack_boards(boards: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(boards, ignore_index=True)
    return combined.sort_values("mean", ascending=False).reset_index(drop=True)


def draws_to_disk(path, arrays: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **arrays)
