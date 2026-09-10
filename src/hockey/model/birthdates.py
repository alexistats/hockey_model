"""Birth dates for a set of players, for the aging curve."""

from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from hockey.models import Player


def player_birth_dates(session: Session, player_ids: list[int]) -> dict[int, date]:
    return dict(
        session.execute(
            select(Player.nhl_id, Player.birth_date).where(Player.nhl_id.in_(player_ids))
        ).all()
    )
