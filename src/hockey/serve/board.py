"""The board, held in memory and re-read against a live draft.

The posterior is fixed the moment the fit finishes. Everything the bot asks for
during a draft - who is left, what they are worth now, which position is
thinning - is the same posterior re-read against a different league state, and
the export layer already knows how to do that. This holds one copy in memory so
the answer costs a recompute rather than a reload.

Measured on the real board, 361 players with 90 drafted: a full recompute
(capacity-aware slot assignment, replacement level, value over replacement) is
about 32 ms, and one P(A > B) over 4000 draws is 22 microseconds. Against a
draft clock measured in tens of seconds, nothing here needs caching or
precomputing, and anything that looks like an optimisation is buying time that
was never scarce. The scarce thing is the browser.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from hockey.export import add_value_over_replacement, replacement_levels, replacement_slots
from hockey.seasons import PROJECTION_SEASON
from hockey.yahoo.eligibility import load_csv as load_eligibility
from hockey.yahoo.settings import load_roster_from_yaml

logger = logging.getLogger(__name__)


@dataclass
class Board:
    """Everything the fit produced, plus the league shape it is priced against."""

    players: pd.DataFrame  # one row per player, static
    draws: np.ndarray  # (n_draws, n_players), column order matches `order`
    order: list[int]  # player_id per draws column
    slots: dict[str, float]  # league-wide slots per position
    roster_shape: dict[str, int]  # one team's starting slots
    bench: int
    n_teams: int
    fitted: dict[str, str] = field(default_factory=dict)

    @property
    def column_of(self) -> dict[int, int]:
        return {pid: i for i, pid in enumerate(self.order)}

    def draws_for(self, player_id: int) -> np.ndarray | None:
        i = self.column_of.get(int(player_id))
        return None if i is None else self.draws[:, i]


def load(directory: Path, goalies: Path | None = None, n_teams: int = 14) -> Board:
    """Read a finished board directory into memory.

    Goalies are optional and, when present, come from their own fit. The two
    posteriors are independent - no shared parameters, nothing about a goalie's
    season informs a winger's - which is exactly what makes stacking their draws
    legitimate, because independent draws are what P(A > B) is defined over.
    """
    players = pd.read_csv(directory / "value_board.csv")
    ids: list[int] = []
    blocks: list[np.ndarray] = []
    for path in sorted(directory.glob("draws_batch_*.npz")):
        with np.load(path) as f:
            ids.extend(int(p) for p in f["player_ids"])
            blocks.append(f["draws"].astype("float32"))
    if not blocks:
        raise SystemExit(f"no draws_batch_*.npz in {directory}")
    draws = np.concatenate(blocks, axis=1)

    if goalies is not None:
        with np.load(goalies / "goalie_draws.npz") as f:
            goalie_ids = [int(p) for p in f["player_ids"]]
            goalie_draws = f["draws"].astype("float32")
        if goalie_draws.shape[0] != draws.shape[0]:
            raise SystemExit(
                f"the skater posterior has {draws.shape[0]} draws and the goalie "
                f"posterior {goalie_draws.shape[0]}; they cannot be compared column "
                f"for column until one is refitted to match"
            )
        on_board = set(players["player_id"].astype(int))
        keep = [i for i, pid in enumerate(goalie_ids) if pid in on_board]
        ids.extend(goalie_ids[i] for i in keep)
        draws = np.concatenate([draws, goalie_draws[:, keep]], axis=1)

    eligibility = load_eligibility(Path(f"config/eligibility_{PROJECTION_SEASON // 10000}.csv"))
    players["eligible"] = [
        eligibility.get(int(p), (s,))
        for p, s in zip(players["player_id"], players["slot"], strict=True)
    ]

    roster = load_roster_from_yaml()
    slots = replacement_slots(roster, n_teams)
    known = {p for e in players["eligible"] for p in e}
    slots = {p: n for p, n in slots.items() if p in known}
    shape = {r["position"]: int(r["count"]) for r in roster if r.get("starting")}
    bench = sum(int(r["count"]) for r in roster if r["position"] == "BN")

    missing = set(players["player_id"].astype(int)) - set(ids)
    if missing:
        raise SystemExit(
            f"{len(missing)} player(s) on the board have no draws, first "
            f"{sorted(missing)[:3]}. Pass the goalie directory if the board has goalies."
        )

    diagnostics = directory / "diagnostics.csv"
    fitted = {}
    if diagnostics.exists():
        worst = pd.read_csv(diagnostics)["worst_rhat"].max()
        fitted["skater_worst_rhat"] = f"{float(worst):.4f}"
    if goalies is not None and (goalies / "diagnostics.csv").exists():
        worst = pd.read_csv(goalies / "diagnostics.csv")["worst_rhat"].max()
        fitted["goalie_worst_rhat"] = f"{float(worst):.4f}"

    logger.info(
        "board loaded: %d players (%d goalies), %d draws each",
        len(players),
        int((players["slot"] == "G").sum()),
        draws.shape[0],
    )
    return Board(
        players=players,
        draws=draws,
        order=ids,
        slots=slots,
        roster_shape=shape,
        bench=bench,
        n_teams=n_teams,
        fitted=fitted,
    )


def revalue(board: Board, drafted: set[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(available players with live value, replacement levels) given who is gone.

    This is the whole point of the service. A value computed before the draft is
    stale the moment a pick happens: as a position empties, every remaining
    player at it is worth more, and the board's static `vorp` column cannot say
    so. Nothing here touches the posterior.
    """
    live = board.players[~board.players["player_id"].astype(int).isin(drafted)]
    if live.empty:
        empty = pd.DataFrame(columns=board.players.columns)
        return empty, pd.DataFrame(
            columns=["position", "replacement", "extrapolated", "free_below"]
        )
    levels = replacement_levels(live, board.slots)
    return add_value_over_replacement(live, levels), levels
