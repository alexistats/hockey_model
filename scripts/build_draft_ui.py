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
from hockey.yahoo.eligibility import load_csv as load_eligibility
from hockey.yahoo.settings import load_roster_from_yaml, load_scoring_from_yaml

TEMPLATE = Path("ui/draft_room.html")
N_TEAMS = 14

# Where I pick, and whether the order reverses each round. From these the page
# works out how many picks separate my turns, which is not one number: in a
# snake it alternates. At slot 8 of 14 the gaps run 13, 15, 13, 15 - so a fixed
# "two rounds" is wrong in both directions, and wrong by more the nearer the
# slot is to either end.
DRAFT_SLOT = 8
SNAKE = True

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


def load_draws(out: Path, ids: list[int], rng, arrays: tuple[str, ...] = ("draws",)):
    """Posterior draws for `ids`, subsampled, in that order - one array each.

    Every array is subsampled on the *same* draw indices. Within a batch those
    indices are the same posterior sample, so a player's goals, shots and games
    played all come from one coherent season rather than three unrelated ones.
    Comparing a pair on several categories at once is only honest if they do.
    """
    columns: dict[str, dict[int, np.ndarray]] = {name: {} for name in arrays}
    for path in sorted(out.glob("draws_batch_*.npz")):
        with np.load(path) as f:
            for name in arrays:
                if name not in f:
                    raise SystemExit(
                        f"{path.name} has no '{name}'; this board predates per-category "
                        f"draws and the comparison panel cannot be built from it"
                    )
            for k, pid in enumerate(f["player_ids"]):
                for name in arrays:
                    columns[name][int(pid)] = f[name][:, k]
    missing = [p for p in ids if p not in columns["draws"]]
    if missing:
        raise SystemExit(f"no saved draws for {len(missing)} player(s), first {missing[:3]}")
    total = len(next(iter(columns["draws"].values())))
    take = rng.choice(total, size=min(UI_DRAWS, total), replace=False)
    take.sort()
    return tuple(np.column_stack([columns[name][p][take] for p in ids]) for name in arrays)


def pack(values: np.ndarray, what: str) -> str:
    """Round to int16 and base64 the column-major bytes, as the page expects."""
    packed = np.rint(values).astype("<i2")
    if not (-32768 < packed.min() and packed.max() < 32767):
        raise SystemExit(f"{what} overflows int16; widen the packing")
    return base64.b64encode(packed.T.tobytes()).decode("ascii")


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

    # Eligibility drives the pools, so the page has to carry it: a live
    # recompute that fell back to NHL primary positions would disagree with the
    # value_board.csv it was built from, and the disagreement would be silent.
    eligibility = load_eligibility(Path(f"config/eligibility_{PROJECTION_SEASON // 10000}.csv"))
    if not eligibility:
        print("warning: no eligibility file; the page will use NHL primary positions")

    ids = [int(p) for p in board["player_id"]]
    # The categories and games played travel too, so the comparison panel can
    # answer "is his shot volume actually higher" from the posterior rather than
    # from two point estimates that cannot say how often it is true.
    wanted = ("draws", "games_played", *(f"stat_{k}" for k in ORDER))
    loaded = load_draws(out, ids, rng, wanted)
    by_name = dict(zip(wanted, loaded, strict=True))

    draws = by_name["draws"]
    # int16 holds every plausible fantasy total and halves what the page carries.
    # Column-major: each player's draws contiguous, which is what the page slices.
    blob = pack(draws, "fantasy totals")
    games_blob = pack(by_name["games_played"], "games played")
    # One array per category, concatenated in ORDER, so the page finds a
    # player's category draws at ((category * nPlayers) + player) * nDraws.
    cat_blob = pack(np.concatenate([by_name[f"stat_{k}"] for k in ORDER], axis=1), "categories")

    players = []
    for n, row in enumerate(board.itertuples()):
        pid = int(row.player_id)
        c = cats.loc[pid]
        players.append(
            {
                "i": n,
                "id": pid,
                "n": row.player,
                # The slot this player is valued at, and everything they are
                # eligible for. The page needs both: one to sort and filter by,
                # the other to refill the pools as the board empties.
                "p": getattr(row, "slot", row.position),
                "e": list(eligibility.get(pid, (getattr(row, "slot", row.position),))),
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
    # One team's shape, not the league's. `slots` answers where replacement
    # level sits; this answers what *I* still have to fill, which is a different
    # question and the one the board-shape panel is built on.
    shape = {r["position"]: int(r["count"]) for r in roster if r.get("starting")}
    bench = sum(int(r["count"]) for r in roster if r["position"] == "BN")
    slot_column = "slot" if "slot" in board else "position"
    positions = sorted(set(board[slot_column]), key=lambda p: ("C", "LW", "RW", "D").index(p))
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
        "rosterShape": shape,
        "bench": bench,
        "nTeams": N_TEAMS,
        "draftSlot": DRAFT_SLOT,
        "snake": SNAKE,
        "cats": list(ORDER),
        "catLabels": LABELS,
        "weights": weights,
        "nDraws": draws.shape[0],
        "draws": blob,
        "catDraws": cat_blob,
        "gamesDraws": games_blob,
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
            f"rather than the point gaps. <b>Positions are Yahoo eligibility</b>, "
            f"transcribed from the league's player list, so a player listed at two "
            f"counts at both. Each is valued at the scarcest slot they can fill, and "
            f"replacement level is set by filling the league's slots best-first and "
            f"reading off whoever is left - the pools overlap, so counting each "
            f"separately would make every position look deeper than it is. "
            f"<b>What is left</b> counts the players still "
            f"available in each band, each in one slot so the totals are real, while "
            f"the cards above count everyone eligible and so overlap. "
            f"Tiers are struck within a position, so a tier 4 "
            f"defenceman and a tier 4 centre are not the same player; value bands are the "
            f"comparable read, because replacement level is already per position. "
            f"<b>Value if I wait</b> assumes the next picks come off the top of the value "
            f"board - the room will not do exactly that, so read it as the direction and "
            f"rough size of the cost, not a forecast. Goalies are not "
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
        f"with {len(players)} players and {draws.shape[0]} draws each"
    )


if __name__ == "__main__":
    main()
