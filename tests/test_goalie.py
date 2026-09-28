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
from hockey.model.goalie_forecast import fill_the_net, load_priors, on_the_board, project


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
            "player_id": [1, 2, 3, 4, 5],
            "player": ["Active", "Retired", "Thin", "Rostered backup", "Back from Europe"],
            "last_season": [20252026, 20192020, 20252026, 20252026, 20232024],
            "last_season_starts": [50, 47, 3, 3, 40],
        }
    )
    kept = on_the_board(board, latest=20252026, min_starts=15, rostered=set())
    assert list(kept["player"]) == ["Active"]
    # On a current roster, the cutoffs do not apply: he has not retired, and a
    # backup the board leaves out is a gap on the page.
    kept = on_the_board(board, latest=20252026, min_starts=15, rostered={4, 5})
    assert list(kept["player"]) == ["Active", "Rostered backup", "Back from Europe"]


# --- whose net, and how full ---------------------------------------------------


def test_a_full_net_squeezes_the_goalie_the_model_has_playing_least():
    # Philadelphia, September 2026: Vladar's walk says 49.5 starts, Woll's still
    # carries Toronto's 39.8, and there are 84 games.
    scale = fill_the_net({1: 49.5, 2: 39.8}, set(), {"PHI": [1, 2]}, 84)
    assert scale[1] == 1.0
    assert 49.5 + 39.8 * scale[2] == pytest.approx(84)


def test_a_net_with_room_is_left_alone():
    # The rest go to call-ups the pool has never seen.
    assert fill_the_net({1: 50, 2: 25}, set(), {"BOS": [1, 2]}, 84) == {1: 1.0, 2: 1.0}


def test_four_on_a_september_roster_squeeze_the_last_ones_not_the_starter():
    scale = fill_the_net({1: 44, 2: 24, 3: 21.5, 4: 18.7}, set(), {"VAN": [1, 2, 3, 4]}, 84)
    assert scale[1] == scale[2] == 1.0
    assert 21.5 * scale[3] == pytest.approx(16)
    assert scale[4] == 0.0


def test_the_priors_file_takes_its_starts_first():
    scale = fill_the_net({1: 60, 2: 45}, {2}, {"PHI": [1, 2]}, 84)
    assert 2 not in scale  # a human's number is not rescaled
    assert 60 * scale[1] == pytest.approx(84 - 45)
    with pytest.raises(SystemExit, match="starts in a 84-game season"):
        fill_the_net({1: 50, 2: 45}, {1, 2}, {"PHI": [1, 2]}, 84)


def _posterior(shares: list[float], n_teams: int, n_draws: int = 2000):
    """A posterior that is nothing but workload: every goalie's share fixed, every
    other effect zero."""
    import types

    import xarray as xr

    def const(value, *shape):
        return np.full((1, n_draws, *shape), value, dtype=float)

    n = len(shares)
    logit = np.log(np.array(shares) / (1 - np.array(shares)))
    share_walk = np.repeat(logit[:, None], 2, axis=1)[None, None].repeat(n_draws, axis=1)
    names = {
        "mu_save": const(2.3),
        "mu_shots": const(np.log(28)),
        "mu_win": const(0.0),
        "mu_share": const(0.0),
        "mu_shutout": const(-3.0),
        "beta_save_on_win": const(0.0),
        "beta_save_on_shutout": const(0.0),
        "beta_shots_on_shutout": const(0.0),
        "alpha_shots": const(50.0),
        "save_walk": const(0.0, n, 2),
        "share_walk": share_walk,
        "team_save": const(0.0, n_teams, 2),
        "team_shots": const(0.0, n_teams, 2),
        "team_win": const(0.0, n_teams, 2),
        "goalie_shots": const(0.0, n),
    }
    ds = xr.Dataset(
        {
            k: (("chain", "draw", *(f"{k}_{i}" for i in range(v.ndim - 2))), v)
            for k, v in names.items()
        }
    )
    return types.SimpleNamespace(posterior=ds)


def _traded_frame() -> pd.DataFrame:
    # Goalie 1 went Pittsburgh then Edmonton in 2025-26; 2 is Philadelphia's
    # starter; 3 finished in Toronto.
    return pd.DataFrame(
        {
            "player_id": [1, 1, 2, 3],
            "season": [20252026] * 4,
            "team": ["EDM", "PIT", "PHI", "TOR"],
            "last_game": pd.to_datetime(["2026-04-15", "2025-12-10", "2026-04-16", "2026-04-16"]),
            "starts": [16, 13, 50, 40],
        }
    )


def _project(current=None, priors=None):
    frame = _traded_frame()
    index = GoalieIndex(frame, 20262027)
    idata = _posterior([0.35, 0.6, 0.5], index.n_teams)
    board, _ = project(idata, frame, index, {"wins": 1.0}, 84, priors=priors, current=current)
    return board.set_index("player_id")


def test_a_goalie_traded_mid_season_finished_where_his_last_game_was():
    # Sorting his two stints by season alone put him on whichever sorted last,
    # with that stint's starts: Jarry, Pittsburgh, 13, and off the board.
    board = _project()
    assert board.loc[1, "team"] == "EDM"
    assert board.loc[1, "last_season_starts"] == 29


def test_a_goalie_on_a_new_roster_is_projected_there_and_shares_its_net():
    before = _project()
    assert before.loc[3, "team"] == "TOR"
    assert before.loc[3, "exp_starts"] == pytest.approx(0.5 * 84, rel=0.03)

    after = _project(current={1: "EDM", 2: "PHI", 3: "PHI"})
    assert (after.loc[3, "team"], after.loc[3, "last_team"]) == ("PHI", "TOR")
    assert after.loc[2, "exp_starts"] == pytest.approx(0.6 * 84, rel=0.03)  # the starter keeps his
    assert after.loc[3, "own_starts"] == pytest.approx(0.5 * 84, rel=1e-6)
    assert after.loc[[2, 3], "exp_starts"].sum() == pytest.approx(84, rel=0.02)


def test_a_workload_in_the_priors_file_stands_and_the_model_fills_around_it():
    priors = {
        "goalies": [{"name": "Backup", "nhl_id": 3, "starts": 28, "starts_sd": 8}],
        "teams": [],
    }
    board = _project(current={1: "EDM", 2: "PHI", 3: "PHI"}, priors=priors)
    assert board.loc[3, "exp_starts"] == pytest.approx(28, rel=0.05)
    assert board.loc[2, "exp_starts"] == pytest.approx(0.6 * 84, rel=0.03)
