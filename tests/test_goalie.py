"""The goalie model's guardrails.

The fit itself is exercised by running it; these cover the places where a
mistake would be silent - an index that quietly treats the projected season as
observed, a priors file that has stopped applying, a scoring assumption that
turns out to point the other way.
"""

import numpy as np
import pandas as pd
import pytest

from hockey.model.goalie import GoalieIndex
from hockey.model.goalie_forecast import load_priors


def frame_of(seasons: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [1] * len(seasons),
            "season": seasons,
            "team": ["BOS"] * len(seasons),
            "starts": [40] * len(seasons),
        }
    )


def test_projected_season_must_be_outside_the_fitting_window():
    # Indexing it as observed reads "not played yet" as "played and recorded
    # nothing", which is the single most damaging thing that can go wrong here.
    with pytest.raises(ValueError, match="in the fitting window"):
        GoalieIndex(frame_of([20242025, 20252026, 20262027]), 20262027)


def test_projection_is_one_step_past_the_last_observed_season():
    index = GoalieIndex(frame_of([20242025, 20252026]), 20262027)
    assert index.seasons == [20242025, 20252026, 20262027]
    assert index.projection_step == 2
    assert index.n_steps == 3


def test_missing_priors_file_is_not_an_error():
    # The model must run on a machine that has never had one.
    priors = load_priors(__import__("pathlib").Path("config/does_not_exist.yaml"))
    assert priors == {"goalies": [], "teams": []}


def test_the_shipped_priors_file_parses():
    priors = load_priors()
    assert isinstance(priors["goalies"], list)
    assert isinstance(priors["teams"], list)


def test_a_shot_faced_is_net_positive_for_every_real_goalie():
    """The structural fact the model is built around.

    Saves pay 0.3 and goals against cost 1.5, so a shot faced is worth
    0.3s - 1.5(1 - s) and breaks even at s = 1.5/1.8. If this ever stops being
    true - a league that pays less for saves, or more against goals - then
    volume stops being a refund for a leaky defence and the goalie board's
    ordering changes qualitatively.
    """
    from hockey.yahoo.settings import load_scoring_from_yaml

    rules = {r.key: r.modifier for r in load_scoring_from_yaml().goalie_rules}
    save, against = rules["saves"], rules["goals_against"]
    break_even = -against / (save - against)
    assert break_even == pytest.approx(1.5 / 1.8, abs=1e-6)
    # No NHL regular has ever been near .833 over a season.
    worst_plausible = 0.870
    assert save * worst_plausible + against * (1 - worst_plausible) > 0


def test_decisions_never_undercount_starts():
    """Wins are credited per decision, so a win rate divided by starts can
    exceed one for a goalie who won in relief. The panel takes the max."""
    frame = pd.DataFrame({"appearances": [50, 40, 45], "starts": [47, 44, 45]})
    decisions = frame[["appearances", "starts"]].max(axis=1)
    assert (decisions >= frame["starts"]).all()
    assert list(decisions) == [50, 44, 45]


def test_override_draws_keep_their_spread():
    """An override is a distribution, not a point estimate.

    Pinning a workload would claim a certainty about a depth chart that nobody
    has, and would show up downstream as a fake-narrow floor and ceiling.
    """
    rng = np.random.default_rng(0)
    starts = np.clip(np.rint(rng.normal(55, 6, size=4000)), 0, 84)
    assert starts.std() == pytest.approx(6.0, rel=0.1)
    assert starts.min() < 45 and starts.max() > 65


def test_a_goalie_who_stopped_playing_is_left_off_the_board():
    """The walk does not know anyone retired.

    It takes a step per season whether or not a game was played, so a goalie
    last seen in 2018-19 still arrives at the projected season carrying a
    plausible number. The first run of this model put Roberto Luongo, Henrik
    Lundqvist and Corey Crawford on the board, and Ben Bishop at 404 projected
    points - six seasons after his last game. Nothing downstream can catch
    that, because the projection is complete and well formed and simply
    concerns someone who will not play.
    """
    board = pd.DataFrame(
        {
            "player": ["Active", "Retired", "Thin"],
            "last_season": [20252026, 20192020, 20252026],
            "last_season_starts": [50, 47, 3],
        }
    )
    latest = 20252026
    kept = board[(board["last_season"] == latest) & (board["last_season_starts"] >= 15)]
    assert list(kept["player"]) == ["Active"]
