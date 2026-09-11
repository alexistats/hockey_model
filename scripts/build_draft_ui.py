"""Pack a board directory into the single-file draft interface.

    python scripts/build_draft_ui.py artifacts/board_v2

The page gets the full posterior, not a summary: a subsample of every player's
fantasy-point draws travels with it as one packed array. That is what lets it
answer head-to-head questions and redraw tier breaks in the browser as players
come off the board, without a round trip and without this script having to
guess in advance which comparisons matter.
"""

import argparse
import base64
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from hockey.export import replacement_slots
from hockey.seasons import PROJECTION_SEASON, SEASON_LENGTH, season_label
from hockey.yahoo.settings import load_roster_from_yaml, load_scoring_from_yaml

TEMPLATE = Path("ui/draft_room.html")
N_TEAMS = 14

# Enough draws for a head-to-head probability to be stable to about a point,
# and small enough that the packed array stays under a megabyte.
UI_DRAWS = 1000

LABELS = {
    "goals": "Goals",
    "assists": "Assists",
    "ppp": "Power-play points",
    "shp": "Shorthanded points",
    "sog": "Shots on goal",
    "hits": "Hits",
    "blocks": "Blocked shots",
    "plus_minus": "Plus/minus",
}
ORDER = ("goals", "assists", "ppp", "shp", "sog", "hits", "blocks", "plus_minus")


def scoring_weights() -> dict[str, float]:
    """Fantasy points per unit of each category, read from the league config
    rather than restated here - a weight typed twice is a weight that drifts."""
    scoring = load_scoring_from_yaml()
    return {rule.key: rule.modifier for rule in scoring.skater_rules}


def load_draws(out: Path, ids: list[int], rng) -> np.ndarray:
    """Fantasy-point draws for `ids`, subsampled, in that order."""
    columns: dict[int, np.ndarray] = {}
    for path in sorted(out.glob("draws_batch_*.npz")):
        with np.load(path) as f:
            for k, pid in enumerate(f["player_ids"]):
                columns[int(pid)] = f["draws"][:, k]
    missing = [p for p in ids if p not in columns]
    if missing:
        raise SystemExit(f"no saved draws for {len(missing)} player(s), first {missing[:3]}")
    total = len(next(iter(columns.values())))
    take = rng.choice(total, size=min(UI_DRAWS, total), replace=False)
    take.sort()
    return np.column_stack([columns[p][take] for p in ids])


def main() -> None:
    parser = argparse.ArgumentParser(prog="build_draft_ui")
    parser.add_argument("directory")
    parser.add_argument("--out", default="artifacts/ui/draft_board.html")
    args = parser.parse_args()

    out = Path(args.directory)
    board = pd.read_csv(out / "value_board.csv")
    cats = pd.read_csv(out / "category_projections.csv").set_index("player_id")
    weights = scoring_weights()
    rng = np.random.default_rng(11)

    ids = [int(p) for p in board["player_id"]]
    draws = load_draws(out, ids, rng)
    # int16 holds every plausible fantasy total and halves what the page carries.
    packed = np.rint(draws).astype("<i2")
    if not (-32768 < packed.min() and packed.max() < 32767):
        raise SystemExit("fantasy totals overflow int16; widen the packing")
    # Column-major: each player's draws contiguous, which is what the page slices.
    blob = base64.b64encode(packed.T.tobytes()).decode("ascii")

    players = []
    for n, row in enumerate(board.itertuples()):
        pid = int(row.player_id)
        c = cats.loc[pid]
        players.append(
            {
                "i": n,
                "id": pid,
                "n": row.player,
                "p": row.position,
                "t": row.team,
                "a": int(row.age),
                "m": round(float(row.mean), 1),
                "f": round(float(row.floor), 1),
                "l": round(float(row.p20), 1),
                "h": round(float(row.p80), 1),
                "c": round(float(row.ceiling), 1),
                "g": round(float(row.exp_games), 1),
                "cat": {
                    k: [
                        round(float(c[k]), 2),
                        round(float(c[f"{k}_floor"]), 1),
                        round(float(c[f"{k}_ceiling"]), 1),
                    ]
                    for k in ORDER
                },
            }
        )

    roster = load_roster_from_yaml()
    slots = replacement_slots(roster, N_TEAMS)
    positions = sorted(set(board["position"]), key=lambda p: ("C", "LW", "RW", "D").index(p))
    diagnostics = pd.read_csv(out / "diagnostics.csv")
    worst = float(diagnostics["worst_rhat"].max())

    payload = {
        "kicker": (
            f"{season_label(PROJECTION_SEASON)} · {N_TEAMS}-team head-to-head points · "
            f"{len(players)} skaters"
        ),
        "gamesInSeason": SEASON_LENGTH[PROJECTION_SEASON],
        "positions": positions,
        "slots": {p: slots[p] for p in positions},
        "cats": list(ORDER),
        "catLabels": LABELS,
        "weights": weights,
        "nDraws": packed.shape[0],
        "draws": blob,
        "players": players,
        "footnote": (
            f"<b>How to read this.</b> Every number is a posterior from a hierarchical "
            f"Bayesian model fitted on eight seasons of game logs, not a point projection. "
            f"The bar spans the 10th to 90th percentile of a player's season; the solid part "
            f"is the middle three-fifths and the tick is the mean. <b>Value</b> is points "
            f"above the replacement player at that position, and it moves as the board "
            f"empties, because a pick is worth the gap to whoever else would fill the slot. "
            f"<b>Tiers</b> break where the next player's chance of outscoring the one who "
            f"opened the tier falls below 40 percent, so they follow the distributions "
            f"rather than the point gaps. Positions are NHL primary positions, not Yahoo "
            f"eligibility, so dual-eligible players are undervalued here. Goalies are not "
            f"modelled yet. Fitted {date.today():%d %B %Y}, worst batch r-hat {worst:.4f}."
        ),
    }

    html = TEMPLATE.read_text(encoding="utf-8")
    page = html.replace("__DATA__", json.dumps(payload, separators=(",", ":")))
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(page, encoding="utf-8")
    print(
        f"wrote {destination} ({len(page) / 1e6:.2f} MB) "
        f"with {len(players)} players and {packed.shape[0]} draws each"
    )


if __name__ == "__main__":
    main()
