"""Who has been drafted, and by whom.

Deliberately written around **reconciliation rather than deltas**. The obvious
design is an endpoint per pick - "player X just went" - and it is wrong for a
draft. It needs every message to arrive exactly once: miss one because a poll
landed mid-render, or send one twice because two polls saw the same pick, and
the server's idea of the board is permanently wrong with nothing to correct it.
For the rest of the draft every value it returns is subtly off, and nobody
finds out.

So the primary write takes the **whole observed board** and diffs it. A missed
poll, a duplicated poll, a reconnection, a pick that landed while the bot was
busy clicking - all of them heal on the next observation, because the next
observation is a complete statement of what is true rather than an increment
on a history the server has to have followed perfectly.

Order is kept where it can be inferred, because a run on a position is only
visible in sequence. The values depend on the set. The one thing that leans on
the order is `cost_of_waiting`, which needs to know which team made each pick,
because a team missing a defenceman drafts one; and that order is checked
against my own picks before it is believed.
"""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


@dataclass
class Pick:
    player_id: int
    by_me: bool
    overall: int | None = None
    at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    # The draft's own pick number: the player's place in an observation sent
    # in draft order, where unidentified picks still hold their places. None
    # until an ordered observation has included him. `overall` is the order
    # this server saw picks in, which is not the same thing once a pick it
    # cannot name has gone by.
    number: int | None = None


@dataclass
class DraftState:
    """The live draft. `slot` and `n_teams` set where my turns fall."""

    slot: int = 8
    n_teams: int = 14
    snake: bool = True
    picks: list[Pick] = field(default_factory=list)
    # Picks the server can see happened but cannot attach to a player: a name it
    # cannot read, or - far more often - a real player who is not in the
    # modelled pool at all. The board holds 295 skaters and 66 goalies, and a
    # live draft will take people outside it.
    #
    # They are counted, never identified. Dropping them was a quiet bug: the
    # clock is derived from the pick count, so every off-board pick left the
    # server believing the draft was a pick behind. That moves the round number,
    # which gates the goalie rules and the risk ramp, and it moves
    # picks_until_my_turn, which is what the cost of waiting is measured over.
    unidentified: int = 0

    @property
    def drafted(self) -> set[int]:
        return {p.player_id for p in self.picks}

    @property
    def total_picks(self) -> int:
        """Every pick the board shows, identified or not. The clock runs on this."""
        return len(self.picks) + self.unidentified

    @property
    def mine(self) -> list[int]:
        return [p.player_id for p in self.picks if p.by_me]

    def my_pick_numbers(self, rounds: int = 20) -> list[int]:
        """Overall pick numbers that belong to me."""
        out = []
        for r in range(1, rounds + 1):
            within = self.n_teams + 1 - self.slot if (self.snake and r % 2 == 0) else self.slot
            out.append((r - 1) * self.n_teams + within)
        return out

    @property
    def current_pick(self) -> int:
        return self.total_picks + 1

    @property
    def current_round(self) -> int:
        return (self.total_picks // self.n_teams) + 1

    @property
    def on_the_clock(self) -> bool:
        return self.current_pick in self.my_pick_numbers()

    def picks_until_my_turn(self) -> int:
        """Picks by other managers before my next selection.

        When it is my turn this is the gap to my *following* pick, which is the
        number that matters: the question being asked is what I lose by passing
        on a position now.
        """
        ahead = [p for p in self.my_pick_numbers() if p >= self.current_pick]
        if not ahead:
            return self.n_teams
        if ahead[0] == self.current_pick:
            return ahead[1] - ahead[0] - 1 if len(ahead) > 1 else self.n_teams
        return ahead[0] - self.current_pick

    def seat_of(self, number: int) -> int:
        """The seat that makes overall pick `number`."""
        rnd = (number - 1) // self.n_teams + 1
        within = (number - 1) % self.n_teams + 1
        return self.n_teams + 1 - within if (self.snake and rnd % 2 == 0) else within

    def seats_before_my_turn(self) -> list[int]:
        """Who picks, in order, in the `picks_until_my_turn()` picks ahead."""
        start = self.current_pick + (1 if self.on_the_clock else 0)
        return [self.seat_of(start + i) for i in range(self.picks_until_my_turn())]

    def rosters_by_seat(self) -> dict[int, list[int]]:
        """The board players each seat holds, as far as the draft order says.

        A pick without a number is left out rather than guessed at - except my
        own, which is mine whatever its number.
        """
        out: dict[int, list[int]] = {}
        for pick in self.picks:
            if pick.by_me:
                out.setdefault(self.slot, []).append(pick.player_id)
            elif pick.number is not None:
                out.setdefault(self.seat_of(pick.number), []).append(pick.player_id)
        return out

    def order_check(self) -> tuple[str, str]:
        """Whether the observed order can be believed, from my own picks.

        My picks are the ones whose seat is known independently: they are mine.
        Every one of them the observation numbered must fall on my seat. One
        that does not means the list was not in draft order - a board rendered
        out of sequence, a duplicated row - and every other team's roster read
        from it would be wrong.
        """
        numbered = [p for p in self.picks if p.by_me and p.number is not None]
        if not numbered:
            return "unchecked", "none of my picks has a draft number yet"
        wrong = [p.number for p in numbered if self.seat_of(p.number) != self.slot]
        if wrong:
            return (
                "inconsistent",
                f"my pick(s) at {wrong[:4]} belong to seat(s) "
                f"{sorted({self.seat_of(n) for n in wrong})[:4]}, not {self.slot}",
            )
        return "consistent", f"{len(numbered)} of my picks sit on seat {self.slot}"

    def reconcile(
        self,
        observed: list[int],
        mine: list[int] | None = None,
        unidentified: int = 0,
        order: dict[int, int] | None = None,
    ) -> dict:
        """Make the state match a complete observation of the board.

        Returns what changed, so a caller can log a correction rather than
        discover one. Removals are reported separately from additions because
        a removal means the previous read was wrong - a misparsed name, a pick
        that was undone - and that is worth noticing rather than silently
        absorbing.

        `unidentified` is how many drafted players in *this* observation could
        not be resolved to a board id. It is set, not added, for the same reason
        the rest of this is a diff: the observation is a complete statement, so
        a name that resolves on the next poll stops being counted rather than
        being counted twice.

        `order` maps a player to his place in the observation, when it was sent
        in draft order. It numbers the picks, and is re-read in full every time
        for the same reason.
        """
        # De-duplicate while preserving the order first seen. The same player
        # reaches here twice more often than it looks: two spellings of one
        # name both resolve, a poll overlaps the previous one, a board renders
        # a pick in two places. Appending a Pick per occurrence would inflate
        # the pick count, and the pick count is what the round number and every
        # "picks until my turn" answer are derived from - so one duplicate read
        # would quietly shift the whole draft clock.
        observed_unique: list[int] = []
        seen: set[int] = set()
        for pid in observed:
            if int(pid) not in seen:
                seen.add(int(pid))
                observed_unique.append(int(pid))
        observed_set = seen
        mine_set = {int(p) for p in (mine or [])} | {p.player_id for p in self.picks if p.by_me}
        known = self.drafted

        added = [p for p in observed_unique if p not in known]
        removed = sorted(known - observed_set)

        if removed:
            logger.warning(
                "%d player(s) were marked drafted and are no longer on the observed "
                "board; dropping them: %s",
                len(removed),
                removed,
            )
        kept = [p for p in self.picks if p.player_id in observed_set]
        for pid in added:
            kept.append(Pick(player_id=int(pid), by_me=int(pid) in mine_set))
        # A player can be reported as mine after first being seen as someone's.
        for pick in kept:
            if pick.player_id in mine_set:
                pick.by_me = True
        # With unidentified picks on the board, the order seen is not the
        # draft's numbering - it depends where those fell. The clock uses
        # `total_picks`, which is. An ordered observation carries the real
        # numbering, because an unidentified name still holds its place there,
        # and that goes on `number`.
        for n, pick in enumerate(kept, start=1):
            pick.overall = n
            if order is not None:
                pick.number = order.get(pick.player_id)
        self.picks = kept
        self.unidentified = max(0, int(unidentified))
        if self.unidentified:
            logger.info(
                "%d observed pick(s) could not be identified; counted toward the clock "
                "but attached to nobody. Draft is at pick %d (round %d).",
                self.unidentified,
                self.current_pick,
                self.current_round,
            )
        return {
            "added": [int(p) for p in added],
            "removed": removed,
            "total": len(self.picks),
            "unidentified": self.unidentified,
            "picks_on_the_board": self.total_picks,
        }

    def take(self, player_id: int, by_me: bool) -> bool:
        """Record one pick. False when it was already known."""
        if int(player_id) in self.drafted:
            for pick in self.picks:
                if pick.player_id == int(player_id) and by_me:
                    pick.by_me = True
            return False
        self.picks.append(Pick(player_id=int(player_id), by_me=by_me, overall=len(self.picks) + 1))

        return True

    def reset(self) -> None:
        self.picks = []
        self.unidentified = 0
