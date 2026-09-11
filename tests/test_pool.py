"""The draftable pool has to match the roster it is drafted into.

A pool chosen by goals and assists got this wrong twice: it ranked players on
categories that are a fraction of what this league scores, and it supplied 64
defencemen for a league that drafts 84, which put replacement level outside the
pool entirely. Both failures are invisible downstream - the board still looks
complete and every defenceman is overvalued.

These run against the real warehouse and skip when it is not up, since the
thing being tested is a query.
"""

import pytest
from sqlalchemy import text

from hockey.export import replacement_slots
from hockey.model.run import _points_expression, choose_pool
from hockey.yahoo.settings import load_roster_from_yaml, load_scoring_from_yaml

N_TEAMS = 14


@pytest.fixture(scope="module")
def session():
    from sqlalchemy.exc import OperationalError

    from hockey.db import SessionLocal

    try:
        with SessionLocal() as s:
            s.execute(text("SELECT 1"))
            yield s
    except OperationalError:
        pytest.skip("warehouse not running")


@pytest.fixture(scope="module")
def quotas():
    slots = replacement_slots(load_roster_from_yaml(), N_TEAMS)
    return {p: int(round(n * 1.4)) for p, n in slots.items() if p != "G"}


def test_the_points_expression_covers_every_scored_category():
    scoring = load_scoring_from_yaml()
    expression, weights = _points_expression(scoring)
    assert len(weights) == len(scoring.skater_rules)
    for rule in scoring.skater_rules:
        assert weights[f"w_{rule.key}"] == rule.modifier
    # Hits and blocks are in the ranking, which is the whole point: at half a
    # point each they outweigh a scoring category over two seasons.
    assert "s.hits" in expression and "s.blocks" in expression


def test_a_category_with_no_column_is_refused():
    """Ranking the pool on a category the warehouse cannot serve would drop it
    silently and quietly reorder the entire draft."""
    scoring = load_scoring_from_yaml()
    broken = type(scoring)(
        league_key=scoring.league_key,
        skater_rules=scoring.skater_rules[:1] + (_fake_rule(),),
        goalie_rules=scoring.goalie_rules,
    )
    with pytest.raises(KeyError, match="no game-log column"):
        _points_expression(broken)


def _fake_rule():
    from hockey.scoring.points import ScoringRule

    category = type(
        "Cat", (), {"key": "faceoff_wins", "position_type": "P", "display_name": "FW"}
    )()
    return ScoringRule(category=category, modifier=1.0)


def test_quotas_fill_every_position_the_league_drafts(session, quotas):
    pool = choose_pool(session, 300, quotas=quotas)
    counts = pool["position"].value_counts().to_dict()
    for position, wanted in quotas.items():
        assert counts.get(position, 0) == wanted, f"{position}: {counts.get(position, 0)}"


def test_the_pool_runs_deeper_than_the_league_drafts(session, quotas):
    """Replacement level is read from inside the pool. A pool that ends before
    the league does forces it to be read off the pool's worst player, which
    overstates every player at that position."""
    slots = replacement_slots(load_roster_from_yaml(), N_TEAMS)
    pool = choose_pool(session, 300, quotas=quotas)
    counts = pool["position"].value_counts().to_dict()
    for position, drafted in slots.items():
        if position == "G":
            continue
        assert counts[position] > drafted, f"{position} pool {counts[position]} <= {drafted}"


def test_ranking_on_league_points_promotes_the_hitters_and_blockers(session, quotas):
    """The check on the old ranking. Scoring alone put defencemen far down a
    combined list; scored properly, the best of them are top-ten players."""
    pool = choose_pool(session, 300, quotas=quotas)
    top_ten = pool.head(10)
    assert "D" in set(top_ten["position"]), list(top_ten["name"])


def test_without_quotas_the_limit_still_applies(session):
    pool = choose_pool(session, 50)
    assert len(pool) == 50
    assert pool["recent_points"].is_monotonic_decreasing
