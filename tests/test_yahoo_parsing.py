"""Yahoo's JSON shapes, and the crosswalk matching rules.

Everything here is pure - no credentials, no network. The payload fixtures
reproduce Yahoo's actual nesting: numeric-keyed collections, records split
across dicts, and single-key wrappers around every collection entry.
"""

import pytest

from hockey.yahoo.client import YahooShapeError, flatten, merge_fragments, unwrap
from hockey.yahoo.crosswalk import (
    _eligible_positions,
    _match,
    _merge_player_fragments,
    _split_name,
    normalize_team,
)

# --- flatten ---


def test_flatten_turns_numeric_keyed_dict_into_list():
    assert flatten({"0": {"a": 1}, "1": {"a": 2}, "count": 2}) == [{"a": 1}, {"a": 2}]


def test_flatten_orders_numerically_not_lexically():
    """Yahoo keys go past '9', where string sorting puts '10' before '2'."""
    node = {str(i): {"i": i} for i in range(12)}
    node["count"] = 12
    assert [d["i"] for d in flatten(node)] == list(range(12))


def test_flatten_recurses_into_nested_collections():
    node = {"league": {"0": {"name": "x"}, "1": {"players": {"0": {"p": 1}, "count": 1}}}}
    assert flatten(node) == {"league": [{"name": "x"}, {"players": [{"p": 1}]}]}


def test_flatten_leaves_ordinary_dicts_alone():
    assert flatten({"name": "x", "id": 3}) == {"name": "x", "id": 3}


# --- merge_fragments ---


def test_merge_fragments_joins_a_split_record():
    node = [{"league_key": "465.l.1", "name": "L"}, {"settings": {"draft_type": "live"}}]
    assert merge_fragments(node) == {
        "league_key": "465.l.1",
        "name": "L",
        "settings": {"draft_type": "live"},
    }


def test_merge_fragments_ignores_padding():
    assert merge_fragments([{"a": 1}, [], {"b": 2}]) == {"a": 1, "b": 2}


def test_merge_fragments_refuses_to_collapse_a_collection():
    """Two entries with the same key are a collection, not a split record.
    Merging would silently keep only the last roster position."""
    node = [{"roster_position": {"position": "C"}}, {"roster_position": {"position": "D"}}]
    with pytest.raises(YahooShapeError, match="collection"):
        merge_fragments(node)


def test_merge_fragments_passes_a_dict_through():
    assert merge_fragments({"a": 1}) == {"a": 1}


# --- unwrap ---


def test_unwrap_strips_the_single_key_wrapper():
    node = [{"stat": {"stat_id": 1}}, {"stat": {"stat_id": 2}}]
    assert unwrap(node, "stat") == [{"stat_id": 1}, {"stat_id": 2}]


def test_unwrap_accepts_a_lone_dict():
    assert unwrap({"stat": {"stat_id": 1}}, "stat") == [{"stat_id": 1}]


def test_unwrap_of_none_is_empty():
    assert unwrap(None, "stat") == []


def test_unwrap_raises_on_an_unexpected_entry():
    """Skipping quietly would understate the league's categories."""
    with pytest.raises(YahooShapeError, match="no 'stat' key"):
        unwrap([{"stat": {"stat_id": 1}}, {"other": {}}], "stat")


# --- a realistic settings payload, end to end through the parsing helpers ---

SETTINGS_PAYLOAD = {
    "fantasy_content": {
        "league": {
            "0": {
                "league_key": "465.l.12345",
                "league_id": "12345",
                "name": "Test League",
                "num_teams": 14,
                "scoring_type": "headpoint",
                "season": "2026",
            },
            "1": {
                "settings": {
                    "roster_positions": {
                        "0": {"roster_position": {"position": "C", "count": 2}},
                        "1": {"roster_position": {"position": "D", "count": 4}},
                        "2": {"roster_position": {"position": "BN", "count": 5}},
                        "count": 3,
                    },
                    "stat_categories": {
                        "stats": {
                            "0": {
                                "stat": {
                                    "stat_id": 1,
                                    "name": "Goals",
                                    "display_name": "G",
                                    "position_type": "P",
                                }
                            },
                            "1": {
                                "stat": {
                                    "stat_id": 31,
                                    "name": "Hits",
                                    "display_name": "HIT",
                                    "position_type": "P",
                                }
                            },
                            "count": 2,
                        }
                    },
                    "stat_modifiers": {
                        "stats": {
                            "0": {"stat": {"stat_id": 1, "value": "5"}},
                            "1": {"stat": {"stat_id": 31, "value": "0.5"}},
                            "count": 2,
                        }
                    },
                }
            },
            "count": 2,
        }
    }
}


def test_settings_payload_parses_into_categories_and_roster():
    from hockey.yahoo.settings import _roster_rows, _stat_rows

    league = merge_fragments(flatten(SETTINGS_PAYLOAD)["fantasy_content"]["league"])
    assert league["name"] == "Test League"
    assert league["num_teams"] == 14

    stats = _stat_rows("465.l.12345", league)
    assert [(s["display_name"], s["modifier"]) for s in stats] == [("G", 5.0), ("HIT", 0.5)]
    # The catalogue resolves each category to the column that serves it.
    assert {s["display_name"]: s["source_column"] for s in stats} == {
        "G": "skater_game_logs.goals",
        "HIT": "skater_game_logs.hits",
    }

    roster = _roster_rows("465.l.12345", league)
    assert [(r["position"], r["count"], r["is_starting"]) for r in roster] == [
        ("C", 2, True),
        ("D", 4, True),
        ("BN", 5, False),
    ]


# --- crosswalk matching ---

# Two Sebastian Ahos, as the app repo's ESPN tests use: the Carolina forward
# and the New York Islanders defenceman. This is the case that makes guessing
# dangerous.
INDEX = {
    ("connor", "mcdavid"): [(8478402, "EDM")],
    ("sebastian", "aho"): [(8478427, "CAR"), (8480222, "NYI")],
    ("tim", "stutzle"): [(8482116, "OTT")],
}


def test_single_name_match_is_exact():
    assert _match(INDEX, "Connor McDavid", "EDM") == (8478402, "exact")


def test_single_match_does_not_need_the_team():
    assert _match(INDEX, "Connor McDavid", None) == (8478402, "exact")


def test_duplicate_names_are_split_by_team():
    assert _match(INDEX, "Sebastian Aho", "CAR") == (8478427, "team_tiebreak")
    assert _match(INDEX, "Sebastian Aho", "NYI") == (8480222, "team_tiebreak")


def test_duplicate_names_without_a_team_are_a_recorded_miss():
    """A wrong id produces a complete, plausible projection for the wrong
    player, which nothing downstream can detect. A miss is visible."""
    assert _match(INDEX, "Sebastian Aho", None) == (None, "ambiguous")


def test_duplicate_names_on_an_unrelated_team_are_a_recorded_miss():
    assert _match(INDEX, "Sebastian Aho", "TOR") == (None, "ambiguous")


def test_no_name_match_is_unmatched():
    assert _match(INDEX, "Totally Unknown", "TOR") == (None, "unmatched")


def test_accents_and_punctuation_fold():
    """Yahoo spells it Stützle; our players table may hold either spelling."""
    assert _match(INDEX, "Tim Stützle", "OTT") == (8482116, "exact")


def test_multi_word_surnames_stay_intact():
    assert _split_name("Pierre-Luc Dubois") == ("pierre luc", "dubois")
    assert _split_name("Ryan Nugent-Hopkins") == ("ryan", "nugent hopkins")


@pytest.mark.parametrize(
    ("yahoo", "nhl"),
    [("LA", "LAK"), ("SJ", "SJS"), ("TB", "TBL"), ("NJ", "NJD"), ("EDM", "EDM")],
)
def test_yahoo_team_abbreviations_normalize(yahoo, nhl):
    assert normalize_team(yahoo) == nhl


def test_normalize_team_of_none():
    assert normalize_team(None) is None


# --- Yahoo's split player record ---


def test_player_fragments_merge_into_the_fields_we_need():
    entry = [
        {"player_key": "465.p.6743", "player_id": "6743"},
        {"name": {"full": "Connor McDavid", "first": "Connor", "last": "McDavid"}},
        {"editorial_team_abbr": "Edm"},
        {"eligible_positions": [{"position": "C"}, {"position": "LW"}]},
    ]
    player = _merge_player_fragments(entry)
    assert player["player_id"] == "6743"
    assert player["name"] == "Connor McDavid"
    assert player["positions"] == ["C", "LW"]


def test_player_padding_is_ignored():
    assert _merge_player_fragments([[], {"nothing": 1}]) is None


def test_eligible_positions_accepts_both_shapes():
    assert _eligible_positions([{"position": "C"}, {"position": "D"}]) == ["C", "D"]
    assert _eligible_positions(["C", "D"]) == ["C", "D"]
    assert _eligible_positions(None) == []


def test_flatten_keeps_a_real_count_field():
    """Yahoo overloads 'count': it marks collection size, and it is also the
    number of a roster slot. Dropping it everywhere turned four defence slots
    into none."""
    node = {"0": {"roster_position": {"position": "D", "count": 4}}, "count": 1}
    assert flatten(node) == [{"roster_position": {"position": "D", "count": 4}}]


def test_flatten_turns_an_empty_collection_into_a_list():
    assert flatten({"count": 0}) == []
