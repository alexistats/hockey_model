"""Calibration scores, checked against cases where the answer is known.

These functions decide whether the model's floor and ceiling can be quoted at
all, so they are tested against analytic values and against deliberately
miscalibrated forecasts rather than only against themselves.
"""

import numpy as np
import pytest

from hockey.calibration.scores import calibration_report as report_of
from hockey.calibration.scores import (
    coverage_table,
    crps,
    crps_ensemble,
    interval_coverage,
    pit_deviation,
    pit_values,
    render_report,
)


def test_crps_of_a_point_forecast_is_absolute_error():
    """CRPS reduces to absolute error when the forecast has no spread, which
    is what lets a point model and a distribution be compared on one scale."""
    samples = np.full(500, 7.0)
    assert crps_ensemble(samples, 10.0) == pytest.approx(3.0)
    assert crps_ensemble(samples, 7.0) == pytest.approx(0.0)


def test_crps_matches_the_closed_form_for_a_uniform_forecast():
    """For X ~ Uniform(0,1) and an outcome at 0.5, CRPS is analytically
    E|X-y| - 0.5*E|X-X'| = 0.25 - 0.5*(1/3) = 1/12."""
    x = np.linspace(0, 1, 20001)
    assert crps_ensemble(x, 0.5) == pytest.approx(1.0 / 12.0, abs=1e-3)


def test_crps_prefers_the_sharper_forecast_when_both_are_centred_correctly():
    rng = np.random.default_rng(0)
    truth = 50.0
    sharp = rng.normal(truth, 2.0, 20000)
    vague = rng.normal(truth, 20.0, 20000)
    assert crps_ensemble(sharp, truth) < crps_ensemble(vague, truth)


def test_crps_penalises_a_sharp_forecast_that_is_wrong():
    """Sharpness is only rewarded when it is earned. A confident miss should
    score worse than an honest wide forecast."""
    rng = np.random.default_rng(1)
    confident_and_wrong = rng.normal(20.0, 1.0, 20000)
    wide_and_centred = rng.normal(50.0, 20.0, 20000)
    assert crps_ensemble(confident_and_wrong, 50.0) > crps_ensemble(wide_and_centred, 50.0)


def test_crps_is_never_negative():
    rng = np.random.default_rng(2)
    for _ in range(20):
        samples = rng.normal(0, 3, 500)
        assert crps_ensemble(samples, float(rng.normal(0, 5))) >= 0


def test_crps_vectorised_matches_the_single_case_version():
    rng = np.random.default_rng(3)
    samples = rng.normal(10, 3, (400, 5))
    observed = rng.normal(10, 3, 5)
    expected = [crps_ensemble(samples[:, i], observed[i]) for i in range(5)]
    np.testing.assert_allclose(crps(samples, observed), expected)


def test_crps_refuses_mismatched_shapes():
    with pytest.raises(ValueError, match="line up case for case"):
        crps(np.zeros((10, 3)), np.zeros(4))


def test_crps_refuses_an_empty_ensemble():
    with pytest.raises(ValueError, match="empty ensemble"):
        crps_ensemble(np.array([]), 1.0)


# --- coverage ---


# Each case has a latent value the forecaster cannot see. The outcome is one
# draw around it; the forecast is an ensemble around the same latent. That is
# the situation a real projection is in, and it is the only setup where
# coverage measures anything: a forecast centred on the outcome itself covers
# 100% of the time no matter how narrow it is.
TRUE_SPREAD = 5.0


def _case(rng, n_cases, forecast_spread, draws=600):
    latent = rng.normal(50, 10, n_cases)
    truth = rng.normal(latent, TRUE_SPREAD)
    samples = rng.normal(latent[None, :], forecast_spread, (draws, n_cases))
    return samples, truth


def test_a_well_calibrated_forecast_covers_at_about_its_nominal_rate():
    rng = np.random.default_rng(4)
    samples, truth = _case(rng, 4000, TRUE_SPREAD)
    for level in (0.5, 0.8, 0.9):
        result = interval_coverage(samples, truth, level)
        assert result["coverage"] == pytest.approx(level, abs=0.025)


def test_an_overconfident_forecast_under_covers():
    """The failure mode that matters: a 90% interval that holds far fewer than
    90% of outcomes makes the floor and ceiling actively misleading."""
    rng = np.random.default_rng(5)
    samples, truth = _case(rng, 4000, TRUE_SPREAD / 5)
    assert interval_coverage(samples, truth, 0.9)["coverage"] < 0.5


def test_a_hedging_forecast_over_covers_and_is_wide():
    rng = np.random.default_rng(6)
    honest, truth = _case(rng, 3000, TRUE_SPREAD)
    rng = np.random.default_rng(6)
    hedging, truth2 = _case(rng, 3000, TRUE_SPREAD * 6)
    np.testing.assert_allclose(truth, truth2)  # same outcomes, wider forecast
    assert interval_coverage(hedging, truth, 0.9)["coverage"] > 0.99
    # Coverage alone is trivially satisfiable, which is why width is reported.
    assert (
        interval_coverage(hedging, truth, 0.9)["mean_width"]
        > interval_coverage(honest, truth, 0.9)["mean_width"]
    )


def test_coverage_table_covers_every_requested_level():
    rng = np.random.default_rng(7)
    truth = rng.normal(0, 1, 500)
    samples = rng.normal(truth[None, :], 1.0, (300, 500))
    rows = coverage_table(samples, truth, levels=(0.5, 0.8, 0.9))
    assert [r["level"] for r in rows] == [0.5, 0.8, 0.9]
    # Wider nominal levels must give wider intervals.
    widths = [r["mean_width"] for r in rows]
    assert widths == sorted(widths)


def test_interval_coverage_rejects_a_nonsense_level():
    with pytest.raises(ValueError, match="between 0 and 1"):
        interval_coverage(np.zeros((5, 5)), np.zeros(5), 1.5)


# --- PIT ---


def test_pit_is_flat_for_a_calibrated_forecast():
    rng = np.random.default_rng(8)
    n_cases = 4000
    truth = rng.poisson(30, n_cases)
    samples = rng.poisson(30, (600, n_cases))
    assert pit_deviation(pit_values(samples, truth, seed=0)) < 0.12


def test_pit_is_not_flat_for_an_overconfident_forecast():
    """Too-narrow forecasts pile the PIT up at both edges: the outcome keeps
    landing outside the ensemble entirely."""
    rng = np.random.default_rng(9)
    too_narrow, truth = _case(rng, 4000, TRUE_SPREAD / 10)
    assert pit_deviation(pit_values(too_narrow, truth, seed=0)) > 0.5


def test_pit_randomisation_matters_for_counts():
    """Without randomisation the PIT of a discrete forecast is lumpy even when
    the model is right, which would read as misspecification."""
    rng = np.random.default_rng(10)
    truth = rng.poisson(3, 4000)  # small counts: many ties
    samples = rng.poisson(3, (500, 4000))
    randomized = pit_deviation(pit_values(samples, truth, seed=0))
    plain = pit_deviation((samples <= truth[None, :]).mean(axis=0))
    assert randomized < plain


# --- report ---


def test_report_bundles_the_numbers_needed_to_judge_the_intervals():
    rng = np.random.default_rng(11)
    truth = rng.poisson(40, 800)
    samples = rng.poisson(40, (500, 800))
    result = report_of(samples, truth)
    assert result["n"] == 800
    assert result["crps_mean"] > 0
    assert len(result["coverage"]) == 3
    assert len(result["pit_histogram"]) == 10
    text = render_report(result, "held-out season")
    assert "held-out season" in text
    assert "CRPS mean" in text
    assert "PIT deviation" in text
