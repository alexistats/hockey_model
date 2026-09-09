"""The coverage report's judgement about what counts as a gap.

This report is the only thing standing between a half-ingested season and a
model that trains on it without complaint, so what it chooses to flag - and
what it chooses not to - is worth testing directly.
"""

from hockey.ingest.coverage import SeasonCoverage, problems, render
from hockey.seasons import season_label


def coverage(**overrides) -> SeasonCoverage:
    """A complete, healthy 2018-19 by default; override one thing per test."""
    base = {
        "season": 20182019,
        "regular_games": 1271,
        "regular_completed": 1271,
        "playoff_games": 87,
        "games_with_logs": 1271,
        "skater_rows": 48888,
        "goalie_rows": 2901,
        "ppp_null_pct": 1.2,
        "shp_null_pct": 1.2,
        "save_pctg_null_pct": 0.3,
        "shutouts_null_pct": 0.3,
    }
    return SeasonCoverage(**(base | overrides))


def test_a_complete_season_is_not_flagged():
    assert problems([coverage()]) == []


def test_playoff_games_do_not_count_against_the_regular_season_expectation():
    """The expected counts are regular season only. Counting the 87 playoff
    games too made every complete season look 87 games over."""
    assert coverage().expected_delta == 0
    assert problems([coverage()]) == []


def test_missing_game_logs_are_flagged_with_the_command_to_fix_them():
    issues = problems([coverage(games_with_logs=1200)])
    assert len(issues) == 1
    assert "71 completed games have no logs" in issues[0]
    assert "game-logs --season 20182019" in issues[0]


def test_a_short_schedule_is_flagged():
    issues = problems([coverage(regular_completed=1100, games_with_logs=1100)])
    assert any("schedule sync may be incomplete" in i for i in issues)


def test_high_null_rate_on_a_scored_category_is_flagged():
    """ppp and shp come from a second pass. If it did not run, every player
    scores zero for two categories and nothing else looks wrong."""
    issues = problems([coverage(ppp_null_pct=100.0, shp_null_pct=100.0)])
    assert any("ppp NULL 100.0%" in i for i in issues)
    assert any("player-stats --season 20182019" in i for i in issues)


def test_a_small_null_rate_is_tolerated():
    """A skater who dressed and never took a shift is omitted by the
    per-player endpoint. A few percent is normal, not a gap."""
    assert problems([coverage(ppp_null_pct=2.0, shp_null_pct=2.0)]) == []


def test_an_unplayed_season_is_not_flagged_at_all():
    """2026-27 has a full schedule and no results. Nothing about it is a gap,
    and flagging it would train the reader to ignore the report."""
    future = coverage(
        season=20262027,
        regular_games=1344,
        regular_completed=0,
        playoff_games=0,
        games_with_logs=0,
        skater_rows=0,
        goalie_rows=0,
        ppp_null_pct=0.0,
        shp_null_pct=0.0,
        save_pctg_null_pct=0.0,
        shutouts_null_pct=0.0,
    )
    assert not future.played
    assert future.expected_delta is None
    assert problems([future]) == []


def test_the_84_game_season_expects_1344_games():
    """2026-27 expands to 84 games per team under the new agreement."""
    played_out = coverage(
        season=20262027, regular_games=1344, regular_completed=1344, games_with_logs=1344
    )
    assert played_out.expected_delta == 0


def test_render_produces_one_line_per_season_plus_a_header():
    text = render(
        [
            coverage(),
            coverage(
                season=20192020, regular_completed=1082, games_with_logs=1082, regular_games=1271
            ),
        ]
    )
    lines = text.splitlines()
    assert len(lines) == 4  # header, rule, two seasons
    assert "2018-19" in lines[2]
    assert "2019-20" in lines[3]


def test_season_label():
    assert season_label(20182019) == "2018-19"
    assert season_label(20262027) == "2026-27"
    assert season_label(20092010) == "2009-10"
