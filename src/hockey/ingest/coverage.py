"""What the warehouse actually holds, per season.

The point of this report is to make gaps visible before the model runs on them.
A season that quietly ingested 80% of its games produces projections that look
entirely normal and are wrong, so every number here is one that would expose
that: games expected against games with logs, and null rates on the columns the
league scores.

Run: python -m hockey.ingest.coverage
"""

import logging
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session

from hockey.seasons import season_label

logger = logging.getLogger(__name__)

# Regular-season games actually played. The two short seasons are real, not
# gaps: 2019-20 stopped at 1082 of 1271 when the season was suspended, and
# 2020-21 was a planned 56-game season on divisional-only schedules. 2026-27 is
# 1344 because the regular season expands to 84 games per team.
EXPECTED_REGULAR_SEASON_GAMES = {
    20182019: 1271,
    20192020: 1082,
    20202021: 868,
    20212022: 1312,
    20222023: 1312,
    20232024: 1312,
    20242025: 1312,
    20252026: 1312,
    20262027: 1344,
}

# A few percent of NULL ppp/shp is expected: the per-player game-log endpoint
# omits a skater who dressed but never took a shift. Much more than that means
# the player-stats pass did not finish.
NULL_TOLERANCE_PCT = 5.0


@dataclass
class SeasonCoverage:
    season: int
    regular_games: int
    regular_completed: int
    playoff_games: int
    games_with_logs: int
    skater_rows: int
    goalie_rows: int
    ppp_null_pct: float
    shp_null_pct: float
    save_pctg_null_pct: float
    shutouts_null_pct: float

    @property
    def played(self) -> bool:
        """A future season has a schedule and no results. Nothing about its
        stat columns is a gap."""
        return self.regular_completed > 0

    @property
    def missing_games(self) -> int:
        return max(self.regular_completed - self.games_with_logs, 0)

    @property
    def expected_delta(self) -> int | None:
        expected = EXPECTED_REGULAR_SEASON_GAMES.get(self.season)
        if expected is None or not self.played:
            return None
        return self.regular_completed - expected


# Regular season and playoffs are counted apart because the expected counts
# above are regular season only; mixing them made every complete season look
# like it had 87 extra games.
_COVERAGE_SQL = text("""
SELECT g.season,
       count(*) FILTER (WHERE g.game_type = 2)                       AS regular_games,
       count(*) FILTER (WHERE g.game_type = 2
                         AND g.game_state IN ('OFF','FINAL'))        AS regular_completed,
       count(*) FILTER (WHERE g.game_type = 3)                       AS playoff_games
  FROM nhl_games g
 GROUP BY g.season
""")

_LOGGED_SQL = text("""
SELECT g.season, count(DISTINCT s.game_id) AS games_with_logs
  FROM skater_game_logs s
  JOIN nhl_games g ON g.nhl_game_id = s.game_id
 WHERE g.game_type = 2
 GROUP BY g.season
""")

_SKATER_SQL = text("""
SELECT g.season,
       count(*) AS skater_rows,
       100.0 * count(*) FILTER (WHERE s.ppp IS NULL) / greatest(count(*), 1) AS ppp_null_pct,
       100.0 * count(*) FILTER (WHERE s.shp IS NULL) / greatest(count(*), 1) AS shp_null_pct
  FROM skater_game_logs s JOIN nhl_games g ON g.nhl_game_id = s.game_id
 GROUP BY g.season
""")

_GOALIE_SQL = text("""
SELECT g.season,
       count(*) AS goalie_rows,
       100.0 * count(*) FILTER (WHERE gl.save_pctg IS NULL) / greatest(count(*), 1)
           AS save_pctg_null_pct,
       100.0 * count(*) FILTER (WHERE gl.shutouts IS NULL) / greatest(count(*), 1)
           AS shutouts_null_pct
  FROM goalie_game_logs gl JOIN nhl_games g ON g.nhl_game_id = gl.game_id
 GROUP BY g.season
""")


def collect(session: Session) -> list[SeasonCoverage]:
    base = {r.season: r for r in session.execute(_COVERAGE_SQL)}
    logged = {r.season: r.games_with_logs for r in session.execute(_LOGGED_SQL)}
    skaters = {r.season: r for r in session.execute(_SKATER_SQL)}
    goalies = {r.season: r for r in session.execute(_GOALIE_SQL)}

    out = []
    for season, row in sorted(base.items()):
        skater = skaters.get(season)
        goalie = goalies.get(season)
        out.append(
            SeasonCoverage(
                season=season,
                regular_games=row.regular_games,
                regular_completed=row.regular_completed,
                playoff_games=row.playoff_games,
                games_with_logs=logged.get(season, 0),
                skater_rows=skater.skater_rows if skater else 0,
                goalie_rows=goalie.goalie_rows if goalie else 0,
                ppp_null_pct=float(skater.ppp_null_pct) if skater else 0.0,
                shp_null_pct=float(skater.shp_null_pct) if skater else 0.0,
                save_pctg_null_pct=float(goalie.save_pctg_null_pct) if goalie else 0.0,
                shutouts_null_pct=float(goalie.shutouts_null_pct) if goalie else 0.0,
            )
        )
    return out


def render(rows: list[SeasonCoverage]) -> str:
    header = (
        "season    reg  done  logged  gap  vs_exp   playoff  skaters  goalies  "
        "ppp_null  shp_null  svp_null   so_null"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        delta = "" if r.expected_delta is None else f"{r.expected_delta:+d}"
        lines.append(
            f"{season_label(r.season):7s}  {r.regular_games:4d}  {r.regular_completed:4d}  "
            f"{r.games_with_logs:6d}  {r.missing_games:3d}  {delta:>6s}   "
            f"{r.playoff_games:7d}  {r.skater_rows:7d}  {r.goalie_rows:7d}  "
            f"{r.ppp_null_pct:7.2f}%  {r.shp_null_pct:7.2f}%  "
            f"{r.save_pctg_null_pct:7.2f}%  {r.shutouts_null_pct:7.2f}%"
        )
    return "\n".join(lines)


def problems(rows: list[SeasonCoverage]) -> list[str]:
    """Anything worth acting on before the model reads this data.

    A season that has not been played yet is skipped entirely: it has a
    schedule and no results, which is not a gap.
    """
    found = []
    for r in rows:
        label = season_label(r.season)
        if not r.played:
            continue
        if r.missing_games:
            found.append(
                f"{label}: {r.missing_games} completed games have no logs. "
                f"Re-run: python -m hockey.ingest game-logs --season {r.season}"
            )
        if r.expected_delta not in (None, 0):
            found.append(
                f"{label}: {r.regular_completed} completed regular-season games against "
                f"{EXPECTED_REGULAR_SEASON_GAMES[r.season]} expected "
                f"({r.expected_delta:+d}); the schedule sync may be incomplete."
            )
        if r.ppp_null_pct > NULL_TOLERANCE_PCT or r.shp_null_pct > NULL_TOLERANCE_PCT:
            found.append(
                f"{label}: ppp NULL {r.ppp_null_pct:.1f}%, shp NULL {r.shp_null_pct:.1f}%. "
                f"Re-run: python -m hockey.ingest player-stats --season {r.season}"
            )
        if r.shutouts_null_pct > NULL_TOLERANCE_PCT:
            found.append(
                f"{label}: goalie shutouts NULL {r.shutouts_null_pct:.1f}%. "
                f"Re-run: python -m hockey.ingest player-stats --season {r.season}"
            )
    return found


def main() -> None:
    from hockey.db import SessionLocal

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    with SessionLocal() as session:
        rows = collect(session)
    print(render(rows))
    print()
    issues = problems(rows)
    if issues:
        print(f"{len(issues)} thing(s) to look at:\n")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("No gaps found in any played season.")


if __name__ == "__main__":
    main()
