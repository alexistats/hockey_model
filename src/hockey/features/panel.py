"""Warehouse -> tidy modelling panel.

This module is what replaces the old model's data layer: hand-exported
Hockey-Reference text files, a manually maintained schedule file, a fabricated
all-zero "dummy row" to force a current-season index, and a positional slice of
the schedule to guess which games were still to come. All of that is now a
query.

Two things the old layer could not do at all and this one does:

- The opponent and home/away come from the team the player actually dressed for
  that night (skater_game_logs.team_abbrev), so a traded player's games are
  attributed correctly on both sides of the trade.
- The remaining schedule is selected by date, not by counting how many games a
  player has played, so a player who missed time does not shift the whole
  remaining schedule by the number of games they were out.

Index conventions: seasons and teams are mapped to contiguous integer indices
because PyMC's GaussianRandomWalk and AR processes are indexed positionally.
Both maps are built from the data, never hardcoded, so the 31-vs-32 team bug in
the old model cannot recur.
"""

from dataclasses import dataclass
from datetime import date

import pandas as pd
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from hockey.models import NhlTeam
from hockey.seasons import FITTING_SEASONS, SEASON_LENGTH

REGULAR_SEASON = 2

# The columns the model reads, named as hockey.scoring.categories keys so a
# stat line lifted straight out of this panel can be scored without renaming.
SKATER_STATS = [
    "goals",
    "assists",
    "plus_minus",
    "ppp",
    "shp",
    "sog",
    "hits",
    "blocks",
]
GOALIE_STATS = [
    "games_started",
    "wins",
    "goals_against",
    "saves",
    "shots_against",
    "shutouts",
]


@dataclass(frozen=True)
class IndexMaps:
    """Positional indices for the model's array dimensions.

    Teams include franchises that no longer exist (Arizona) and ones that did
    not exist for the whole window (Seattle, Utah). Their unobserved seasons
    are simply missing data, which the AR process handles; dropping them would
    misattribute every game played against them.
    """

    seasons: list[int]
    teams: list[str]

    @property
    def season_index(self) -> dict[int, int]:
        return {s: i for i, s in enumerate(self.seasons)}

    @property
    def team_index(self) -> dict[str, int]:
        return {t: i for i, t in enumerate(self.teams)}

    @property
    def n_seasons(self) -> int:
        return len(self.seasons)

    @property
    def n_teams(self) -> int:
        return len(self.teams)


def build_index_maps(session: Session, seasons: list[int] | None = None) -> IndexMaps:
    """Index maps over the fitting window.

    Defaults to hockey.seasons.FITTING_SEASONS rather than to every season in
    the database. The projection season's schedule is loaded too, and if it
    were indexed as an observed season the model would read "not played yet"
    as "played and produced nothing".
    """
    teams = sorted(session.scalars(select(NhlTeam.abbrev)))
    return IndexMaps(seasons=sorted(seasons or FITTING_SEASONS), teams=teams)


_SKATER_SQL = """
SELECT s.player_id,
       s.game_id,
       g.date,
       g.season,
       g.game_type,
       s.team_abbrev,
       CASE WHEN s.team_abbrev = g.home_team_abbrev
            THEN g.away_team_abbrev ELSE g.home_team_abbrev END AS opponent_abbrev,
       (s.team_abbrev = g.home_team_abbrev)::int                AS is_home,
       s.position,
       s.goals, s.assists, s.plus_minus, s.ppp, s.shp,
       s.sog, s.hits, s.blocks, s.toi_seconds
  FROM skater_game_logs s
  JOIN nhl_games g ON g.nhl_game_id = s.game_id
 WHERE g.game_type = :game_type
   AND s.team_abbrev IS NOT NULL
   {season_filter}
   {player_filter}
 ORDER BY s.player_id, g.date, s.game_id
"""

_GOALIE_SQL = """
SELECT gl.player_id,
       gl.game_id,
       g.date,
       g.season,
       g.game_type,
       gl.team_abbrev,
       CASE WHEN gl.team_abbrev = g.home_team_abbrev
            THEN g.away_team_abbrev ELSE g.home_team_abbrev END AS opponent_abbrev,
       (gl.team_abbrev = g.home_team_abbrev)::int               AS is_home,
       gl.started::int AS games_started,
       gl.wins, gl.goals_against, gl.saves, gl.shots_against, gl.shutouts,
       gl.toi_seconds
  FROM goalie_game_logs gl
  JOIN nhl_games g ON g.nhl_game_id = gl.game_id
 WHERE g.game_type = :game_type
   AND gl.team_abbrev IS NOT NULL
   {season_filter}
   {player_filter}
 ORDER BY gl.player_id, g.date, gl.game_id
"""


def _build_query(
    template: str, seasons: list[int] | None, player_ids: list[int] | None
) -> tuple[str, dict]:
    params: dict = {"game_type": REGULAR_SEASON}
    season_filter = ""
    if seasons:
        season_filter = "AND g.season = ANY(:seasons)"
        params["seasons"] = list(seasons)
    player_filter = ""
    if player_ids:
        # Column prefix differs between the two templates; both alias the log
        # table so `player_id` is unambiguous once qualified by the template.
        player_filter = "AND {alias}.player_id = ANY(:player_ids)"
        params["player_ids"] = list(player_ids)
    alias = "s" if "skater_game_logs s" in template else "gl"
    return (
        template.format(
            season_filter=season_filter,
            player_filter=player_filter.format(alias=alias) if player_filter else "",
        ),
        params,
    )


def _attach_indices(frame: pd.DataFrame, maps: IndexMaps) -> pd.DataFrame:
    """Add the positional indices the model's arrays are keyed on.

    A season or team outside the maps becomes NaN rather than silently
    mapping to 0, which would attribute those games to the first team in the
    list. The caller is expected to have built the maps from the same window.
    """
    frame = frame.copy()
    frame["season_idx"] = frame["season"].map(maps.season_index)
    frame["opponent_idx"] = frame["opponent_abbrev"].map(maps.team_index)
    frame["team_idx"] = frame["team_abbrev"].map(maps.team_index)
    unmapped = frame[frame[["season_idx", "opponent_idx", "team_idx"]].isna().any(axis=1)]
    if not unmapped.empty:
        raise ValueError(
            f"{len(unmapped)} rows reference a season or team outside the index maps "
            f"(seasons {sorted(unmapped['season'].unique())}, "
            f"teams {sorted(set(unmapped['opponent_abbrev']) | set(unmapped['team_abbrev']))}). "
            f"Rebuild the maps over the same window as the panel."
        )
    for column in ("season_idx", "opponent_idx", "team_idx"):
        frame[column] = frame[column].astype("int32")
    return frame


def skater_panel(
    session: Session,
    maps: IndexMaps,
    player_ids: list[int] | None = None,
    seasons: list[int] | None = None,
) -> pd.DataFrame:
    """One row per skater-game, in chronological order per player."""
    sql, params = _build_query(_SKATER_SQL, seasons, player_ids)
    frame = pd.DataFrame(session.execute(text(sql), params).mappings().all())
    if frame.empty:
        return frame
    # ppp and shp are NULL for a skater who dressed but never took a shift.
    # That skater did score zero of each, so zero is the correct reading -
    # the same one hockey.scoring.score_statline takes.
    frame[["ppp", "shp"]] = frame[["ppp", "shp"]].fillna(0).astype("int32")
    return _attach_indices(frame, maps)


def goalie_panel(
    session: Session,
    maps: IndexMaps,
    player_ids: list[int] | None = None,
    seasons: list[int] | None = None,
) -> pd.DataFrame:
    """One row per goalie-appearance, in chronological order per goalie."""
    sql, params = _build_query(_GOALIE_SQL, seasons, player_ids)
    frame = pd.DataFrame(session.execute(text(sql), params).mappings().all())
    if frame.empty:
        return frame
    frame["shutouts"] = frame["shutouts"].fillna(0).astype("int32")
    return _attach_indices(frame, maps)


def team_schedule(
    session: Session,
    maps: IndexMaps,
    team_abbrev: str,
    season: int,
    after: date | None = None,
) -> pd.DataFrame:
    """A team's remaining regular-season games, selected by date.

    The old model took `schedule.iloc[games_played:]`, which assumes the player
    appeared in every one of their team's games to date. For anyone who missed
    time - which is exactly the population the availability component exists
    for - that slices off the wrong games and shifts the whole projection.
    """
    sql = """
    SELECT g.nhl_game_id AS game_id,
           g.date,
           g.season,
           CASE WHEN g.home_team_abbrev = :team
                THEN g.away_team_abbrev ELSE g.home_team_abbrev END AS opponent_abbrev,
           (g.home_team_abbrev = :team)::int AS is_home
      FROM nhl_games g
     WHERE g.season = :season
       AND g.game_type = :game_type
       AND (g.home_team_abbrev = :team OR g.away_team_abbrev = :team)
       {date_filter}
     ORDER BY g.date, g.nhl_game_id
    """
    params = {"team": team_abbrev, "season": season, "game_type": REGULAR_SEASON}
    date_filter = ""
    if after is not None:
        date_filter = "AND g.date > :after"
        params["after"] = after
    frame = pd.DataFrame(
        session.execute(text(sql.format(date_filter=date_filter)), params).mappings().all()
    )
    if frame.empty:
        return frame
    frame["team_abbrev"] = team_abbrev
    # The projected season is deliberately outside the fitting window, so only
    # the team indices are attached here; the model indexes these games at its
    # own extra walk step.
    frame["opponent_idx"] = frame["opponent_abbrev"].map(maps.team_index).astype("int32")
    frame["team_idx"] = frame["team_abbrev"].map(maps.team_index).astype("int32")
    return frame


def availability_panel(
    session: Session,
    player_ids: list[int] | None = None,
    seasons: list[int] | None = None,
) -> pd.DataFrame:
    """Games played against games available, per player-season.

    This is what separates a player who scores at a high rate from a player who
    scores at a high rate and is on the ice in April. Two players with the same
    per-game projection are not the same pick if one of them plays 82 games and
    the other plays 58.

    A real caveat, worth knowing before trusting a number: a season in which a
    player was called up in January looks identical here to a season in which
    they were injured until January. Both show as low availability. The model
    partially pools and lets the walk favour recent seasons, which softens it,
    but a young player's early seasons will read as less available than they
    truly were.
    """
    sql = """
    SELECT s.player_id,
           g.season,
           count(*) AS games_played
      FROM skater_game_logs s
      JOIN nhl_games g ON g.nhl_game_id = s.game_id
     WHERE g.game_type = :game_type
       {season_filter}
       {player_filter}
     GROUP BY s.player_id, g.season
     ORDER BY s.player_id, g.season
    """
    params: dict = {"game_type": REGULAR_SEASON}
    season_filter = ""
    if seasons:
        season_filter = "AND g.season = ANY(:seasons)"
        params["seasons"] = list(seasons)
    player_filter = ""
    if player_ids:
        player_filter = "AND s.player_id = ANY(:player_ids)"
        params["player_ids"] = list(player_ids)

    frame = pd.DataFrame(
        session.execute(
            text(sql.format(season_filter=season_filter, player_filter=player_filter)), params
        )
        .mappings()
        .all()
    )
    if frame.empty:
        return frame
    # Games the player could have played. Each season's real length, not 82:
    # 2020-21 was 56 games, and treating it as 82 would read every player in it
    # as chronically unavailable.
    frame["games_available"] = frame["season"].map(SEASON_LENGTH)
    if frame["games_available"].isna().any():
        missing = sorted(frame.loc[frame["games_available"].isna(), "season"].unique())
        raise ValueError(f"no known season length for {missing}; add it to hockey/seasons.py")
    frame["games_available"] = frame["games_available"].astype(int)
    # A player cannot dress for more games than the schedule holds; a trade
    # mid-season can otherwise push the count past it.
    frame["games_played"] = frame[["games_played", "games_available"]].min(axis=1)
    frame["share"] = frame["games_played"] / frame["games_available"]
    return frame


def player_directory(session: Session, player_ids: list[int] | None = None) -> pd.DataFrame:
    """Names and positions, for labelling output. Never used for modelling -
    a player's team comes from the game log, not from here."""
    sql = """
    SELECT p.nhl_id AS player_id,
           p.first_name || ' ' || p.last_name AS name,
           p.position,
           p.team_abbrev AS current_team
      FROM players p
     {where}
     ORDER BY p.last_name, p.first_name
    """
    params: dict = {}
    where = ""
    if player_ids:
        where = "WHERE p.nhl_id = ANY(:player_ids)"
        params["player_ids"] = list(player_ids)
    return pd.DataFrame(session.execute(text(sql.format(where=where)), params).mappings().all())


def find_player(session: Session, name: str) -> pd.DataFrame:
    """Look a player up by name, for the command-line entry points."""
    sql = """
    SELECT p.nhl_id AS player_id,
           p.first_name || ' ' || p.last_name AS name,
           p.position, p.team_abbrev AS current_team,
           (SELECT count(*) FROM skater_game_logs s WHERE s.player_id = p.nhl_id)
             + (SELECT count(*) FROM goalie_game_logs g WHERE g.player_id = p.nhl_id) AS games
      FROM players p
     WHERE lower(p.first_name || ' ' || p.last_name) LIKE lower(:pattern)
     ORDER BY games DESC
    """
    return pd.DataFrame(session.execute(text(sql), {"pattern": f"%{name}%"}).mappings().all())
