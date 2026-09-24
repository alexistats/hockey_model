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
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
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

    `names` is expected in draft order, as Yahoo's pick list shows it, names
    that will not resolve included in their places. The order is how the
    server knows which team made each pick - which is what `cost_of_waiting`
    needs, since a team missing a defenceman drafts one - and it is checked
    against my own picks before it is believed. An unordered list still works;
    it just cannot tell the teams apart.
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
    directory: Path,
    goalies: Path | None = None,
    n_teams: int = 14,
    slot: int = 8,
    ui: Path | None = None,
) -> FastAPI:
    data = board_mod.load(directory, goalies, n_teams=n_teams)
    resolver = Resolver(data.players)
    state = DraftState(slot=slot, n_teams=n_teams)
    rules: dict = dict(recommend_mod.DEFAULT_RULES)
    names = dict(zip(data.players["player_id"].astype(int), data.players["player"], strict=True))
    # What the last observation could not read. The bot gets it in the reply to
    # its own post; the draft page, following along, gets it from
    # /draft/state, so a name that did not resolve reaches the person who can
    # mark it by hand instead of stopping at a log line.
    last_seen: dict = {"unresolved": [], "at": None}

    app = FastAPI(title="hockey_models draft API", version="1.0")
    # Read-only handles for tools that replay a saved draft through the API and
    # need to see what it saw. Nothing reads these to serve a request.
    app.state.board = data
    app.state.draft = state

    def _resolve_many(
        names: list[str], ids: list[int]
    ) -> tuple[list[int], list[dict], dict[int, int]]:
        """Resolved ids, the misses, and each named player's place in `names`.

        `names` is read in draft order - the order Yahoo's pick list shows - so
        a place is a pick number, and an unresolved name still holds its place.
        Bare ids carry no place.
        """
        resolved: list[int] = []
        misses: list[dict] = []
        order: dict[int, int] = {}
        for pid in ids:
            if resolver.known(pid):
                resolved.append(int(pid))
            else:
                misses.append({"raw": str(pid), "reason": "unknown player_id"})
        for place, raw in enumerate(names, start=1):
            got = resolver.resolve(raw)
            if got.ok:
                resolved.append(int(got.player_id))
                order.setdefault(int(got.player_id), place)
            else:
                misses.append(
                    {"raw": raw, "reason": got.method, "candidates": list(got.candidates)}
                )
        return resolved, misses, order

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
            "room": data.room.describe() if data.room is not None else None,
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
            # Identified against what the board actually shows. A gap means
            # picks the server cannot name - usually players outside the
            # modelled pool - which are counted toward the clock but are not in
            # `drafted` and never will be.
            "picks_on_the_board": state.total_picks,
            "unidentified": state.unidentified,
            "mine": state.mine,
            # Everyone drafted, in draft order, for the page to mark. A pick the
            # observation numbered sits at its number; one it did not (a pick
            # recorded through /draft/mine, or an unordered observation)
            # follows, in the order this server saw it.
            "picks": [
                {
                    "player_id": p.player_id,
                    "player": names.get(p.player_id),
                    "by_me": p.by_me,
                    "number": p.number,
                }
                for p in sorted(
                    state.picks,
                    key=lambda p: (p.number is None, p.number or 0, p.overall or 0),
                )
            ],
            "unresolved": last_seen["unresolved"],
            "observed_at": last_seen["at"],
            # Whether the observed order matched my own picks' seats, which is
            # what lets the room model read each team's open slots.
            "order_check": dict(zip(("status", "detail"), state.order_check(), strict=True)),
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
            if payload.rules.get("rank_by", "risk") not in ("risk", "vorp"):
                raise HTTPException(422, "rules.rank_by must be 'risk' or 'vorp'")
            for k in ("risk_start", "risk_end"):
                if k in payload.rules and not 0 <= float(payload.rules[k]) <= 1:
                    raise HTTPException(422, f"rules.{k} must be between 0 and 1")
            rules.update(payload.rules)
        return draft_state()

    @app.post("/draft/observed")
    def observed(payload: ObservedBoard):
        """Reconcile against a complete observation. The primary write."""
        ids, misses, order = _resolve_many(payload.names, payload.player_ids)
        mine_ids, mine_misses, _ = _resolve_many(payload.mine, payload.mine_ids)
        # A pick the server cannot identify is still a pick. Counting the
        # distinct ones keeps the clock honest without ever attaching a name to
        # a player - `mine` usually repeats names from `names`, so they are
        # folded together rather than counted twice.
        unidentified = len({str(m["raw"]).strip().casefold() for m in misses + mine_misses})
        change = state.reconcile(
            ids, mine_ids, unidentified=unidentified, order=order if payload.names else None
        )
        # Distinct, in the order the board shows them: `mine` usually repeats
        # a name from `names`.
        last_seen["unresolved"] = list(dict.fromkeys(str(m["raw"]) for m in misses + mine_misses))
        last_seen["at"] = datetime.now(UTC).isoformat(timespec="seconds")
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
        last_seen.update(unresolved=[], at=None)
        return draft_state()

    # The draft page, from the same origin as the API. Opened from disk it
    # cannot call this server - the browser refuses a file page's requests to
    # localhost - so it is served here, and follows the draft by polling
    # /draft/state.
    @app.get("/ui")
    def page():
        if ui is None or not ui.exists():
            raise HTTPException(
                404,
                f"no draft page at {ui}: build it with python scripts/build_draft_ui.py "
                "<board> --goalies <goalies>, then restart with --ui <that file>",
            )
        return FileResponse(ui, media_type="text/html")

    @app.get("/", include_in_schema=False)
    def home():
        return RedirectResponse("/ui")

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
    def recommend(limit: int = 6):
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
        # Assigned under the real slots, not counted off a column: four
        # RW-eligible players are not four right wings if two of them are also
        # centres and the centre slots are open.
        roster = recommend_mod.assign_roster(mine, data.roster_shape, data.bench)
        waiting, room = recommend_mod.cost_of_waiting(data, state, valued)
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
            "room": room,
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
