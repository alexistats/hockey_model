"""Players the model has no projection for, placed at replacement level.

    config/extra_players_2026.csv: nhl_id, player, team, positions, games, note

The model needs a player's NHL games to say anything about him. A rookie, a
prospect with a handful of games, or a goalie back from the minors has too few,
so he is not on the board - and then not on my roster or among the free agents
either, which hides a real player behind a gap. This sheet puts the notable
ones back, each at replacement level for his first position: the value of the
player a league this size leaves on waivers, as a rate a game, spread over
`games` (72 for a skater, 30 starts for a goalie when blank). It is an
assumption, not a projection, and everything that shows these players marks
them so.

The ids are NHL ids, checked against the warehouse when the sheet was written.
Nothing here matches a name.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hockey.seasons import PROJECTION_SEASON

DEFAULT_PATH = Path(f"config/extra_players_{PROJECTION_SEASON // 10000}.csv")
# The games a replacement-level season is measured over, to turn it into a rate:
# the board's replacement level is a season total.
SEASON_GAMES = {"skater": 72.0, "goalie": 45.0}
DEFAULT_GAMES = {"skater": 72.0, "goalie": 30.0}
FLOOR_PCT, LOW_PCT, HIGH_PCT, CEILING_PCT = 10, 20, 80, 90


def load(path: Path = DEFAULT_PATH) -> pd.DataFrame | None:
    """The sheet, with `eligible` as a tuple of positions; None when absent."""
    if not Path(path).exists():
        return None
    sheet = pd.read_csv(path)
    sheet["nhl_id"] = sheet["nhl_id"].astype(int)
    sheet["eligible"] = sheet["positions"].map(lambda s: tuple(str(s).split("|")))
    return sheet


def board_rows(
    sheet: pd.DataFrame,
    replacement: dict[str, float],
    board: pd.DataFrame,
    n_draws: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Rows shaped like value_board.csv for the sheet's players the board does
    not already have, and a column of draws for each.

    The draws are a normal spread around the replacement season, as wide as
    the board's own players near replacement at that position: without them
    the page could not place these players in a percentile sort or a range,
    and with a spread of zero it would present an assumption as a certainty.
    """
    on_board = set(board["player_id"].astype(int))
    near = board[board["vorp"].abs() < 50]
    rows, columns = [], []
    for r in sheet.itertuples():
        if int(r.nhl_id) in on_board:
            continue  # the model's projection wins wherever there is one
        slot = r.eligible[0]
        if slot not in replacement:
            raise ValueError(f"{r.player}: no replacement level for position {slot!r}")
        kind = "goalie" if slot == "G" else "skater"
        games = float(r.games) if pd.notna(r.games) and str(r.games) != "" else DEFAULT_GAMES[kind]
        rate = replacement[slot] / SEASON_GAMES[kind]
        mean = rate * games
        typical = near.loc[near["slot"] == slot, "sd"]
        sd = (float(typical.median()) if len(typical) else 0.25 * replacement[slot]) * (
            games / SEASON_GAMES[kind]
        )
        draws = np.clip(rng.normal(mean, sd, n_draws), 0.0, None)
        floor, low, high, ceiling = np.percentile(
            draws, [FLOOR_PCT, LOW_PCT, HIGH_PCT, CEILING_PCT]
        )
        rows.append(
            {
                "player": r.player,
                "player_id": int(r.nhl_id),
                "mean": mean,
                "floor": floor,
                "p20": low,
                "p80": high,
                "ceiling": ceiling,
                "sd": sd,
                "exp_games": games,
                "exp_starts": games if kind == "goalie" else np.nan,
                "age": np.nan,
                "position": slot,
                "slot": slot,
                "team": r.team,
                "eligible": r.eligible,
                "replacement": replacement[slot],
                "vorp": mean - replacement[slot],
                "extra": True,
            }
        )
        columns.append(draws)
    draws = np.column_stack(columns) if columns else np.zeros((n_draws, 0))
    return pd.DataFrame(rows), draws
