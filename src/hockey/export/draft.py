"""From stat posteriors to a draft board.

This is the whole answer to "how do the categories become fantasy points": for
every posterior draw, apply the league's weights to that draw's stat line and
add them up. One draw of a stat line becomes one draw of a fantasy-point total,
so a set of stat posteriors becomes a fantasy-point posterior with no extra
modelling and no extra assumptions.

Two properties fall out of doing it draw by draw rather than on the means.

The correlation survives. A draw where a player scored more also had more
shots, and summing within the draw keeps that; scoring the averages instead
would collapse the joint distribution into a point and understate the spread.

And the weights are never baked into the model. Re-scoring an existing
posterior under different settings is arithmetic over an array, not a refit -
which is what makes `what_if` below possible, and what would be lost by
modelling fantasy points directly.
"""

import logging

import numpy as np
import pandas as pd

from hockey.model.forecast import Projection
from hockey.scoring import LeagueScoring, score_draws

logger = logging.getLogger(__name__)


def fantasy_points(
    projection: Projection, scoring: LeagueScoring, position_type: str = "P"
) -> np.ndarray:
    """Fantasy-point draws, shaped (draws, players).

    Raises if the projection is missing a category the league scores, rather
    than scoring it as zero. A silently absent category understates every
    player by a similar amount, so the ranking still looks reasonable and is
    wrong.
    """
    return score_draws(scoring, projection.totals, position_type)


def draft_board(
    projection: Projection, scoring: LeagueScoring, position_type: str = "P"
) -> pd.DataFrame:
    """One row per player: the numbers a draft pick is actually made from.

    `floor` and `ceiling` are the 5th and 95th percentiles. `spread` is the gap
    between them, which is the risk in the pick expressed in the same units as
    the reward - a player whose ceiling is 40 points above their floor is a
    different proposition from one whose range is 15, even at the same mean.
    """
    points = fantasy_points(projection, scoring, position_type)
    rows = []
    for i, player_id in enumerate(projection.players):
        draws = points[:, i]
        floor, ceiling = np.percentile(draws, [5, 95])
        rows.append(
            {
                "player": projection.player_names[player_id],
                "player_id": player_id,
                "games": projection.games_per_player.get(player_id, 0),
                "mean": draws.mean(),
                "floor": floor,
                "ceiling": ceiling,
                "spread": ceiling - floor,
                "sd": draws.std(),
            }
        )
    board = pd.DataFrame(rows).sort_values("mean", ascending=False)
    return board.reset_index(drop=True)


def rank(board: pd.DataFrame, appetite: str = "balanced") -> pd.DataFrame:
    """Order the board by risk appetite.

    The same posterior supports three different draft strategies, which is the
    point of carrying a distribution rather than a projection:

    - `safe` ranks on the floor. Early picks, where a bust costs more than a
      breakout gains.
    - `balanced` ranks on the mean.
    - `upside` ranks on the ceiling. Late picks, where the downside is a
      waiver-wire replacement and only the top end matters.
    """
    column = {"safe": "floor", "balanced": "mean", "upside": "ceiling"}
    if appetite not in column:
        raise ValueError(f"appetite must be one of {sorted(column)}, got {appetite!r}")
    ordered = board.sort_values(column[appetite], ascending=False).reset_index(drop=True)
    ordered.insert(0, "rank", np.arange(1, len(ordered) + 1))
    return ordered


def beats(projection: Projection, scoring: LeagueScoring, position_type: str = "P"):
    """P(row finishes ahead of column) in fantasy points, compared draw by draw.

    This is the number a pick actually turns on. Two players can project within
    a point of each other and still be far from a coin flip, and no pair of
    point projections can tell you which.
    """
    points = fantasy_points(projection, scoring, position_type)
    names = [projection.player_names[p] for p in projection.players]
    matrix = pd.DataFrame(index=names, columns=names, dtype=float)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            matrix.loc[a, b] = np.nan if i == j else float((points[:, i] > points[:, j]).mean())
    return matrix


def what_if(
    projection: Projection,
    baseline: LeagueScoring,
    changed: LeagueScoring,
    position_type: str = "P",
) -> pd.DataFrame:
    """How the board moves under different scoring settings, without refitting.

    Because the weights are applied after the model rather than inside it, a
    settings change costs one pass over an existing array. Useful for a
    mid-season rule change, and for seeing which players a league's quirks are
    actually rewarding.
    """
    before = draft_board(projection, baseline, position_type).set_index("player")
    after = draft_board(projection, changed, position_type).set_index("player")
    comparison = pd.DataFrame(
        {
            "mean_before": before["mean"],
            "mean_after": after["mean"],
            "rank_before": before["mean"].rank(ascending=False).astype(int),
            "rank_after": after["mean"].rank(ascending=False).astype(int),
        }
    )
    comparison["rank_change"] = comparison["rank_before"] - comparison["rank_after"]
    return comparison.sort_values("mean_after", ascending=False)
