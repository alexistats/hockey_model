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
from hockey.serve.recommend import DEFAULT_RULES, NO_CLEAR_CALL, URGENT
from hockey.serve.room import SIMS, RoomModel
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

# Goalies score in a different currency. They get their own row set rather than
# zeros in the skaters' categories, because a zero in "hits" reads as a
# measurement and is really an absence.
GOALIE_LABELS = {
    "games_started": "Starts",
    "wins": "Wins",
    "saves": "Saves",
    "goals_against": "Goals against",
    "shutouts": "Shutouts",
}
GOALIE_ORDER = ("games_started", "wins", "saves", "goals_against", "shutouts")
GOALIE_COLUMNS = {
    "games_started": "exp_starts",
    "wins": "exp_wins",
    "saves": "exp_saves",
    "goals_against": "exp_ga",
    "shutouts": "exp_shutouts",
}


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
    parser.add_argument(
        "--goalies", default=None, help="a goalie board directory, for the goalie draws"
    )
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

    # The room model the API prices waiting with, so the page prices it the same
    # way. Without one the page says it cannot, rather than falling back to a room
    # that drafts down this board - the assumption that was measured wrong.
    room_file = Path(f"config/room_{PROJECTION_SEASON // 10000}.json")
    room = RoomModel.load(room_file) if room_file.exists() else None
    if room is None:
        print(f"warning: no {room_file}; the page will show no cost of waiting")

    # The season's schedule, so the page can say how many of a player's games he
    # would actually start for me. Days are sent once and teams as indices into
    # them: 2,688 team-games in a few kilobytes.
    calendar = None
    off_night_max = None
    schedule_file = out / "schedule.csv"
    if schedule_file.exists():
        games = pd.read_csv(schedule_file)
        days = sorted(games["date"].unique())
        at = {d: k for k, d in enumerate(days)}
        calendar = {
            "days": days,
            "teams": {
                str(team): sorted(at[d] for d in group["date"])
                for team, group in games.groupby("team")
            },
        }
        # An off night has fewer than half the league's teams playing: with 32
        # teams, fewer than 16, which is 7 games or fewer. The page can move it.
        off_night_max = (len(calendar["teams"]) // 2 - 1) // 2
    else:
        print(f"warning: no {schedule_file}; the page will show no schedule fit")

    # Goalies reach this file already merged into value_board.csv by the export,
    # so they are ranked and tiered; what they still need is their draws. They
    # come from a separate fit and a separate file, and because the board is
    # sorted by value they are interleaved among the skaters rather than sitting
    # at the end.
    goalie_draws_by_id: dict[int, np.ndarray] = {}
    if args.goalies:
        with np.load(Path(args.goalies) / "goalie_draws.npz") as f:
            for k, gid in enumerate(f["player_ids"]):
                goalie_draws_by_id[int(gid)] = f["draws"][:, k]

    is_goalie = [str(slot) == "G" for slot in board["slot"]]
    skater_rows = [n for n, g in enumerate(is_goalie) if not g]
    skater_ids = [int(board["player_id"].iloc[n]) for n in skater_rows]
    missing = [
        int(board["player_id"].iloc[n])
        for n, g in enumerate(is_goalie)
        if g and int(board["player_id"].iloc[n]) not in goalie_draws_by_id
    ]
    if missing:
        raise SystemExit(
            f"{len(missing)} goalie(s) on the board have no draws, first {missing[:3]}. "
            f"Pass --goalies pointing at the directory the goalie board came from."
        )

    # The categories and games played travel too, so the comparison panel can
    # answer "is his shot volume actually higher" from the posterior rather than
    # from two point estimates that cannot say how often it is true.
    wanted = ("draws", "games_played", *(f"stat_{k}" for k in ORDER))
    loaded = load_draws(out, skater_ids, rng, wanted)
    by_name = dict(zip(wanted, loaded, strict=True))
    n_draws = by_name["draws"].shape[0]

    # Totals cover everyone, in board order, so the page can slice by row index.
    totals = np.zeros((n_draws, len(board)), dtype="float32")
    totals[:, skater_rows] = by_name["draws"]
    for n, g in enumerate(is_goalie):
        if not g:
            continue
        column = goalie_draws_by_id[int(board["player_id"].iloc[n])]
        if len(column) < n_draws:
            raise SystemExit(
                f"the goalie posterior has {len(column)} draws and the skater "
                f"posterior {n_draws}; refit so the two match rather than padding one"
            )
        totals[:, n] = column[rng.choice(len(column), size=n_draws, replace=False)]

    # int16 holds every plausible fantasy total and halves what the page carries.
    # Column-major: each player's draws contiguous, which is what the page slices.
    blob = pack(totals, "fantasy totals")
    games_blob = pack(by_name["games_played"], "games played")
    # One array per category, concatenated in ORDER, so the page finds a
    # player's category draws at ((category * nSkaters) + k) * nDraws. `catIndex`
    # maps a board row to its k, and -1 for a goalie: the skater arrays stay
    # skater-sized rather than being padded with zeros a reader could mistake for
    # measurements, and the board is sorted by value so goalies are interleaved.
    cat_blob = pack(np.concatenate([by_name[f"stat_{k}"] for k in ORDER], axis=1), "categories")
    cat_index = [-1] * len(board)
    for k, n in enumerate(skater_rows):
        cat_index[n] = k

    players = []
    for n, row in enumerate(board.itertuples()):
        pid = int(row.player_id)
        goalie = is_goalie[n]
        c = None if goalie else cats.loc[pid]
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
                # Goalies carry no age: the aging curve is measured on skaters
                # and was never fitted for them, so there is nothing to show.
                "a": None if pd.isna(row.age) else int(row.age),
                "m": round(float(row.mean), 1),
                "f": round(float(row.floor), 1),
                "l": round(float(row.p20), 1),
                "h": round(float(row.p80), 1),
                "c": round(float(row.ceiling), 1),
                "g": round(float(row.exp_games), 1),
                # Where the room usually takes him (average pick over saved mock
                # drafts), or null for a player it never has.
                "o": None if room is None else room.adp.get(pid),
                "cat": (
                    None
                    if goalie
                    else {
                        k: [
                            round(float(c[k]), 2),
                            round(float(c[f"{k}_floor"]), 1),
                            round(float(c[f"{k}_ceiling"]), 1),
                        ]
                        for k in ORDER
                    }
                ),
                "gcat": (
                    {k: round(float(getattr(row, GOALIE_COLUMNS[k])), 1) for k in GOALIE_ORDER}
                    if goalie
                    else None
                ),
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
    # Goalies last, because that is the order a roster is read in and the order
    # the position cards should sit in.
    order = ("C", "LW", "RW", "D", "G")
    positions = sorted(set(board[slot_column]), key=order.index)
    diagnostics = pd.read_csv(out / "diagnostics.csv")
    worst = float(diagnostics["worst_rhat"].max())

    payload = {
        "kicker": (
            f"{season_label(PROJECTION_SEASON)} · {N_TEAMS}-team head-to-head points · "
            f"{len(players) - sum(is_goalie)} skaters"
            + (f" · {sum(is_goalie)} goalies" if any(is_goalie) else "")
        ),
        "gamesInSeason": SEASON_LENGTH[PROJECTION_SEASON],
        "positions": positions,
        "slots": {p: slots[p] for p in positions},
        "rosterShape": shape,
        "bench": bench,
        "nTeams": N_TEAMS,
        "draftSlot": DRAFT_SLOT,
        "snake": SNAKE,
        # The draft API's own rules and thresholds, so the page and the bot
        # make the same call from the same numbers rather than two copies.
        "rules": DEFAULT_RULES,
        "noClearCall": NO_CLEAR_CALL,
        "urgent": URGENT,
        "calendar": calendar,
        "offNightMax": off_night_max,
        "room": None
        if room is None
        else {
            "orderWeight": room.order_weight,
            "needWeight": room.need_weight,
            "offBoard": room.off_board,
            "blindOrderWeight": room.blind_order_weight,
            "blindOffBoard": room.blind_off_board,
            "undrafted": room.undrafted,
            "sims": SIMS,
            "source": room.source,
            "fitted": room.fitted,
        },
        "cats": list(ORDER),
        "catLabels": LABELS,
        "weights": weights,
        "nDraws": n_draws,
        "draws": blob,
        "catDraws": cat_blob,
        "catIndex": cat_index,
        "gamesDraws": games_blob,
        "goalieCats": list(GOALIE_ORDER),
        "goalieCatLabels": GOALIE_LABELS,
        "goalieWeights": {r.key: r.modifier for r in load_scoring_from_yaml().goalie_rules},
        "players": players,
        "footnote": (
            "<b>How to read this.</b> Every number is a posterior from a hierarchical "
            "Bayesian model fitted on eight seasons of game logs, not a point projection. "
            "The bar spans the 10th to 90th percentile of a player's season; the solid part "
            "is the middle three-fifths and the tick is the mean. <b>Value</b> is points "
            "above the replacement player at that position, and it moves as the board "
            "empties, because a pick is worth the gap to whoever else would fill the slot. "
            "<b>Tiers</b> break where the next player's chance of outscoring the one who "
            "opened the tier falls below 40 percent, so they follow the distributions "
            "rather than the point gaps. <b>Positions are Yahoo eligibility</b>, "
            "transcribed from the league's player list, so a player listed at two "
            "counts at both. Each is valued at the scarcest slot they can fill, and "
            "replacement level is set by filling the league's slots best-first and "
            "reading off whoever is left - the pools overlap, so counting each "
            "separately would make every position look deeper than it is. "
            "<b>What is left</b> counts the players still "
            "available in each band, each in one slot so the totals are real, while "
            "the cards above count everyone eligible and so overlap. "
            "Tiers are struck within a position, so a tier 4 "
            "defenceman and a tier 4 centre are not the same player; value bands are the "
            "comparable read, because replacement level is already per position. "
            "<b>Score</b> is the order the draft bot picks in: a blend of the 20th and "
            "80th percentiles, each above replacement, that leans on the cautious end "
            "in the first round and on the upside by the last. Early misses cannot be "
            "replaced and late ones cost a waiver claim. "
            + (
                "<b>Value if I wait</b> is the expected drop in the best player left at a "
                "position by my next turn, over simulated rooms that draft in the room's own "
                f"order toward each team's open slots - {room.source}, fitted {room.fitted}. "
                "It is the draft API's model, and a mock room is not this league. Hover it for "
                "the chance the best player there now is still there. "
                if room is not None
                else "<b>Value if I wait</b> is not shown: this page was built without a room "
                "model, and a room that drafts down this board was measured wrong. "
            )
            + (
                "<b>Fits</b> counts the games a player would add to my lineup this season: "
                "the nights his team plays and my lineup, as it stands, has room for him - an "
                "open slot he fits, or one freed by moving a player of mine who is eligible "
                "elsewhere. A better player on a full night only bumps one of mine, so that "
                "night adds nothing; his quality is the other columns. With two centres on my "
                "roster, a third one fits only their nights off. It moves as my roster fills, "
                "which is when it matters: early on everyone fits every game. For a goalie it "
                "counts his team's games, and he starts only some of them. "
                f"<b>Off nights</b> are his team's games on nights with {off_night_max} games "
                "or fewer, when fewer than "
                "half the league's teams play and my lineup is likelier to have a slot open; "
                "the box above the board moves the cutoff. <b>My roster</b> puts my players "
                "in their slots, with the nights each one starts, and counts for each position "
                "the nights its slots are filled. Every night's lineup is set the way a manager "
                "sets it: as many slots filled as possible, the best players in them. "
                if calendar is not None
                else "This board was built without a schedule, so it cannot say how a player "
                "fits my lineup. "
            )
            + (
                "<b>Goalies</b> come from a separate model and a separate fit, so their "
                "draws are independent of the skaters' - which is exactly what makes "
                "comparing the two legitimate. They are valued against the goalie slots "
                "the same way everyone else is valued against theirs, but they carry no "
                "age, because the aging curve is measured on skaters, and no per-category "
                "probabilities, because only their point totals were saved. "
                if any(is_goalie)
                else "Goalies are not modelled on this board. "
            )
            + f"Fitted {date.today():%d %B %Y}, worst batch r-hat {worst:.4f}."
        ),
    }

    html = TEMPLATE.read_text(encoding="utf-8")
    page = html.replace("__DATA__", json.dumps(payload, separators=(",", ":")))
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(page, encoding="utf-8")
    print(
        f"wrote {destination} ({len(page) / 1e6:.2f} MB) "
        f"with {len(players)} players and {n_draws} draws each"
    )


if __name__ == "__main__":
    main()
