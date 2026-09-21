"""The draft-day API: one authority on what is left and what it is worth.

    python -m hockey.serve --board artifacts/board_v3 --goalies artifacts/goalies_v2

The bot is the hands and this is the brain, so the valuation lives here and is
implemented once. That is not tidiness. The capacity-aware slot assignment,
frozen tiers, eligibility pools and the goalie merge are subtle enough that a
second implementation drifts, and this project has already watched that happen
between its own page and its own Python.

It is also the one place that knows the live draft, which means the human board
and the bot can read the same state instead of each believing a different
thing about who is still available.

Nothing here changes a posterior. Every number is that posterior re-read
against the league as it currently stands.
"""

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from hockey.serve import board as board_mod
from hockey.serve import recommend as recommend_mod
from hockey.serve.identity import Resolver
from hockey.serve.state import DraftState

logger = logging.getLogger(__name__)


class ObservedBoard(BaseModel):
    """A complete statement of who is off the board, as the bot can see it.

    Complete is the important word. This is not "here is a new pick" - it is
    "here is everything currently drafted", and the server diffs it. That is
    what makes a missed poll, a double poll or a reconnect self-healing rather
    than permanently corrupting.
    """

    names: list[str] = Field(default_factory=list)
    player_ids: list[int] = Field(default_factory=list)
    mine: list[str] = Field(default_factory=list)
    mine_ids: list[int] = Field(default_factory=list)


class MyPick(BaseModel):
    name: str | None = None
    player_id: int | None = None
    position: str | None = None


class Settings(BaseModel):
    slot: int | None = None
    n_teams: int | None = None
    snake: bool | None = None
    rules: dict | None = None


def create_app(
    directory: Path, goalies: Path | None = None, n_teams: int = 14, slot: int = 8
) -> FastAPI:
    data = board_mod.load(directory, goalies, n_teams=n_teams)
    resolver = Resolver(data.players)
    state = DraftState(slot=slot, n_teams=n_teams)
    rules: dict = dict(recommend_mod.DEFAULT_RULES)

    app = FastAPI(title="hockey_models draft API", version="1.0")

    def _resolve_many(names: list[str], ids: list[int]) -> tuple[list[int], list[dict]]:
        resolved: list[int] = []
        misses: list[dict] = []
        for pid in ids:
            if resolver.known(pid):
                resolved.append(int(pid))
            else:
                misses.append({"raw": str(pid), "reason": "unknown player_id"})
        for raw in names:
            got = resolver.resolve(raw)
            if got.ok:
                resolved.append(int(got.player_id))
            else:
                misses.append(
                    {"raw": raw, "reason": got.method, "candidates": list(got.candidates)}
                )
        return resolved, misses

    def _valued():
        return board_mod.revalue(data, state.drafted)

    @app.get("/health")
    def health():
        return {
            "players": len(data.players),
            "goalies": int((data.players["slot"] == "G").sum()),
            "draws": int(data.draws.shape[0]),
            "board": str(directory),
            "goalie_board": None if goalies is None else str(goalies),
            "convergence": data.fitted,
            "slots": data.slots,
        }

    @app.get("/draft/state")
    def draft_state():
        return {
            "slot": state.slot,
            "n_teams": state.n_teams,
            "snake": state.snake,
            "round": state.current_round,
            "overall_pick": state.current_pick,
            "on_the_clock": state.on_the_clock,
            "picks_until_my_turn": state.picks_until_my_turn(),
            "my_next_picks": [p for p in state.my_pick_numbers() if p >= state.current_pick][:4],
            "drafted": len(state.picks),
            "mine": state.mine,
            "rules": rules,
        }

    @app.post("/draft/settings")
    def settings(payload: Settings):
        if payload.slot is not None:
            if not 1 <= payload.slot <= state.n_teams:
                raise HTTPException(422, f"slot must be 1..{state.n_teams}")
            state.slot = payload.slot
        if payload.n_teams is not None:
            state.n_teams = payload.n_teams
        if payload.snake is not None:
            state.snake = payload.snake
        if payload.rules is not None:
            rules.update(payload.rules)
        return draft_state()

    @app.post("/draft/observed")
    def observed(payload: ObservedBoard):
        """Reconcile against a complete observation. The primary write."""
        ids, misses = _resolve_many(payload.names, payload.player_ids)
        mine_ids, mine_misses = _resolve_many(payload.mine, payload.mine_ids)
        change = state.reconcile(ids, mine_ids)
        if misses:
            # Never guessed, never silently dropped. An unresolved name means
            # the board holds someone this server cannot identify, and acting
            # on a board you cannot fully read is how the wrong player gets
            # taken.
            logger.warning("%d observed name(s) did not resolve: %s", len(misses), misses[:3])
        return {
            **change,
            "unresolved": misses + mine_misses,
            "round": state.current_round,
            "overall_pick": state.current_pick,
            "on_the_clock": state.on_the_clock,
        }

    @app.post("/draft/mine")
    def mine(payload: MyPick):
        if payload.player_id is not None:
            if not resolver.known(payload.player_id):
                raise HTTPException(404, f"unknown player_id {payload.player_id}")
            pid = int(payload.player_id)
        elif payload.name:
            got = resolver.resolve(payload.name, payload.position)
            if not got.ok:
                raise HTTPException(
                    422,
                    {
                        "error": "could not identify that player, and will not guess",
                        "raw": payload.name,
                        "reason": got.method,
                        "candidates": list(got.candidates),
                    },
                )
            pid = int(got.player_id)
        else:
            raise HTTPException(422, "give a name or a player_id")
        fresh = state.take(pid, by_me=True)
        row = data.players[data.players["player_id"].astype(int) == pid].iloc[0]
        return {
            "player_id": pid,
            "player": str(row.player),
            "slot": str(row.slot),
            "newly_recorded": fresh,
            "my_roster_size": len(state.mine),
        }

    @app.post("/draft/reset")
    def reset():
        state.reset()
        return draft_state()

    @app.get("/board")
    def get_board(limit: int = 50, position: str | None = None):
        valued, levels = _valued()
        if position:
            valued = valued[
                [
                    position in (r.eligible if isinstance(r.eligible, tuple | list) else (r.slot,))
                    for r in valued.itertuples()
                ]
            ]
        top = valued.sort_values("vorp", ascending=False).head(limit)
        columns = [
            "player_id",
            "player",
            "slot",
            "team",
            "mean",
            "floor",
            "p20",
            "p80",
            "ceiling",
            "sd",
            "vorp",
            "pos_rank",
            "tier",
            "exp_games",
        ]
        rows = top[[c for c in columns if c in top]].to_dict("records")
        for row, elig in zip(rows, top["eligible"], strict=True):
            row["eligible"] = list(elig)
        return {
            "available": int(len(valued)),
            "drafted": len(state.picks),
            "replacement": levels.to_dict("records"),
            "players": rows,
        }

    @app.get("/recommend")
    def recommend(limit: int = 4):
        valued, levels = _valued()
        if valued.empty:
            raise HTTPException(409, "no players left on the board")
        return recommend_mod.shortlist(
            data, state, valued, levels, data.draws_for, limit=limit, rules=rules
        )

    @app.get("/team/me")
    def team_me():
        valued, levels = _valued()
        mine = data.players[data.players["player_id"].astype(int).isin(state.mine)]
        positions = [p for p in data.roster_shape if p in data.slots]
        # Assigned under the real slots, not counted off a column: four
        # RW-eligible players are not four right wings if two of them are also
        # centres and the centre slots are open.
        roster = recommend_mod.assign_roster(mine, data.roster_shape, data.bench)
        waiting = recommend_mod.cost_of_waiting(valued, positions, state.picks_until_my_turn())
        return {
            "roster": mine[
                ["player_id", "player", "slot", "team", "mean", "floor", "ceiling"]
            ].to_dict("records"),
            "lineup": roster["lineup"],
            "filled": roster["filled"],
            "needs": roster["needs"],
            "bench": roster["bench"],
            "bench_slots_left": roster["bench_slots_left"],
            "projected_points": round(float(mine["mean"].sum()), 1) if len(mine) else 0.0,
            "cost_of_waiting": waiting,
            "replacement": levels.to_dict("records"),
            "note": (
                "projected_points is a plain sum of every drafted player's mean and is "
                "NOT a lineup score - it ignores roster slots and counts bench players "
                "in full. Do not sum vorp across a roster at all: each player's vorp "
                "independently claims his best eligible slot, so flexibility is "
                "double-counted. Assign to slots first, then sum the starters."
            ),
        }

    @app.get("/compare")
    def compare(ids: str):
        """P(A > B) between any players, over the draws."""
        try:
            wanted = [int(x) for x in ids.split(",") if x.strip()]
        except ValueError:
            raise HTTPException(422, "ids must be comma-separated integers") from None
        if len(wanted) < 2:
            raise HTTPException(422, "give at least two ids")
        unknown = [p for p in wanted if not resolver.known(p)]
        if unknown:
            raise HTTPException(404, f"unknown player_id(s) {unknown}")
        out = {}
        for a in wanted:
            x = data.draws_for(a)
            out[str(a)] = {
                str(b): round(float((x > y).mean() + (x == y).mean() / 2), 4)
                for b in wanted
                if b != a and (y := data.draws_for(b)) is not None
            }
        names = data.players.set_index("player_id")["player"].to_dict()
        return {"names": {str(p): names.get(p, str(p)) for p in wanted}, "p_beats": out}

    return app
