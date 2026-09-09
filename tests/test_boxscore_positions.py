"""A dressed backup goalie listed among the skaters must not become a skater row.

This is a real NHL boxscore quirk, not a hypothetical: game 2019030001 listed
Dan Vladar (a goalie) among Boston's skaters with 0:00 played. One row in
92,520 - and it stopped an eight-season backfill dead, because the follow-up
pass then asked the per-player endpoint for that goalie's *skater* game log,
got a goalie-shaped payload with no powerPlayPoints, and the schema broke
loudly as designed.

Loudly is right. Silently writing a goalie into skater_game_logs would put a
zero-everything row into the model's panel for a player who never took a shift.
"""

import pytest

from hockey.ingest import schemas


def skater(player_id: int, position: str, toi: str = "12:00") -> dict:
    return {
        "playerId": player_id,
        "name": {"default": "A. Player"},
        "position": position,
        "goals": 0,
        "assists": 0,
        "points": 0,
        "plusMinus": 0,
        "pim": 0,
        "hits": 1,
        "blockedShots": 1,
        "sog": 2,
        "toi": toi,
    }


def goalie(player_id: int, toi: str = "60:00") -> dict:
    return {
        "playerId": player_id,
        "name": {"default": "A. Goalie"},
        "position": "G",
        "goalsAgainst": 2,
        "saves": 28,
        "shotsAgainst": 30,
        "toi": toi,
        "starter": True,
        "decision": "W",
    }


def boxscore_payload(home_skaters, home_goalies) -> dict:
    empty = {"forwards": [], "defense": [], "goalies": []}
    return {
        "id": 2019030001,
        "season": 20192020,
        "gameType": 3,
        "gameState": "OFF",
        "homeTeam": {"id": 6, "abbrev": "BOS", "score": 3},
        "awayTeam": {"id": 14, "abbrev": "TBL", "score": 2},
        "playerByGameStats": {
            "homeTeam": {
                "forwards": home_skaters,
                "defense": [],
                "goalies": home_goalies,
            },
            "awayTeam": empty,
        },
    }


def split_players(raw: schemas.RawBoxscore):
    """The same partition apply_boxscore performs, without a database."""
    skaters, goalies = [], []
    for side, team in (
        (raw.player_by_game_stats.home_team, raw.home_team.abbrev),
        (raw.player_by_game_stats.away_team, raw.away_team.abbrev),
    ):
        skaters.extend((s, team) for s in side.forwards + side.defense if s.position != "G")
        goalies.extend((g, team) for g in side.goalies)
    return skaters, goalies


def test_a_goalie_listed_among_the_skaters_is_not_a_skater():
    """The exact shape of game 2019030001."""
    raw = schemas.RawBoxscore.model_validate(
        boxscore_payload(
            home_skaters=[skater(8478402, "C"), skater(8478435, "G", toi="00:00")],
            home_goalies=[goalie(8478435, toi="28:42")],
        )
    )
    skaters, goalies = split_players(raw)
    assert [s.player_id for s, _ in skaters] == [8478402]
    assert [g.player_id for g, _ in goalies] == [8478435]


def test_real_skaters_are_all_kept():
    raw = schemas.RawBoxscore.model_validate(
        boxscore_payload(
            home_skaters=[skater(1, "C"), skater(2, "L"), skater(3, "R"), skater(4, "D")],
            home_goalies=[goalie(9)],
        )
    )
    skaters, _ = split_players(raw)
    assert [s.player_id for s, _ in skaters] == [1, 2, 3, 4]


def test_the_team_comes_from_the_side_the_player_was_listed_on():
    """This is the only record of who a player dressed for that night."""
    raw = schemas.RawBoxscore.model_validate(
        boxscore_payload(home_skaters=[skater(1, "C")], home_goalies=[goalie(9)])
    )
    skaters, goalies = split_players(raw)
    assert skaters[0][1] == "BOS"
    assert goalies[0][1] == "BOS"


def test_the_skater_schema_still_breaks_loudly_on_a_missing_field():
    """The fix filters goalies out; it must not have loosened the schema. A
    real skater row missing powerPlayPoints is still an error."""
    with pytest.raises(Exception, match="powerPlayPoints|power_play_points"):
        schemas.RawSkaterGameLog.model_validate(
            {"gameLog": [{"gameId": 1, "shorthandedPoints": 0}]}
        )


def test_the_skater_game_log_requires_shorthanded_points():
    """SHP is scored at 1.5 in this league, so a payload without it is a
    schema change worth stopping for, not a zero."""
    with pytest.raises(Exception, match="shorthandedPoints|shorthanded_points"):
        schemas.RawSkaterGameLog.model_validate({"gameLog": [{"gameId": 1, "powerPlayPoints": 2}]})


def test_an_empty_game_log_is_allowed():
    """The endpoint omits gameLog entirely for a player with no games that
    season, which is a documented empty case rather than a break."""
    assert schemas.RawSkaterGameLog.model_validate({"playerStatsSeasons": []}).game_log == []


# --- the 2020-21 standings, which had no conferences at all ---


def test_a_season_without_conferences_parses():
    """2020-21 was played in four temporary divisions with no conference
    structure. The field is genuinely absent from the data, not from the
    payload, so requiring it stopped the backfill on a correct response."""
    team = schemas.RawStandingsTeam.model_validate(
        {
            "teamAbbrev": {"default": "TOR"},
            "teamName": {"default": "Toronto Maple Leafs"},
            "divisionName": "Scotia North",
        }
    )
    assert team.conference_name is None
    assert team.division_name == "Scotia North"


def test_a_row_from_a_conferenceless_season_omits_the_conference_key():
    """Omitted, not None: upsert_rows only writes the columns a row provides,
    so a neighbouring season's conference is left intact rather than erased."""
    from hockey.ingest.mappers import team_row

    team = schemas.RawStandingsTeam.model_validate(
        {
            "teamAbbrev": {"default": "TOR"},
            "teamName": {"default": "Toronto Maple Leafs"},
            "divisionName": "Scotia North",
        }
    )
    row = team_row(team)
    assert "conference" not in row
    assert row["division"] == "Scotia North"


def test_a_normal_season_still_carries_its_conference():
    from hockey.ingest.mappers import team_row

    team = schemas.RawStandingsTeam.model_validate(
        {
            "teamAbbrev": {"default": "TOR"},
            "teamName": {"default": "Toronto Maple Leafs"},
            "conferenceName": "Eastern",
            "divisionName": "Atlantic",
        }
    )
    assert team_row(team)["conference"] == "Eastern"
