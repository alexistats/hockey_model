"""Proper scoring rules and calibration diagnostics.

A floor and a ceiling are only worth having if the intervals they come from are
calibrated. An uncalibrated 90% interval that covers 60% of outcomes is worse
than no interval, because it invites confident decisions that are wrong more
often than the number implies. So these are the numbers that decide whether the
model's output can be trusted, and they get published rather than asserted.

Everything here is a pure function over sample ensembles, so it works the same
on posterior draws, on a bootstrap, or on a benchmark model's output - which
is what lets a non-Bayesian benchmark be compared like for like.
"""

import numpy as np

# --- CRPS ---------------------------------------------------------------


def crps_ensemble(samples: np.ndarray, observed: float) -> float:
    """Continuous ranked probability score for one forecast and one outcome.

    CRPS rewards a distribution for being both accurate and sharp: a forecast
    that hedges by being very wide is penalized, and so is a confident forecast
    that misses. It reduces to absolute error when the forecast is a point, so
    a probabilistic model and a point model can be compared on one scale.
    Lower is better, and it is in the units of the thing being predicted.

    Computed from the sorted ensemble in O(m log m):

        CRPS = mean|X - y| - (1/2) mean|X - X'|

    where the second term is the expected spread within the ensemble itself.
    The naive form of that term is an m-by-m double sum; the sorted identity
    below is the same quantity in linear time, which matters when scoring
    hundreds of players against thousands of draws.
    """
    x = np.sort(np.asarray(samples, dtype=float))
    m = x.size
    if m == 0:
        raise ValueError("cannot score an empty ensemble")
    accuracy = np.abs(x - observed).mean()
    # sum_{i<j} (x_j - x_i) = sum_i (2i - m - 1) * x_(i), one-indexed.
    weights = 2.0 * np.arange(1, m + 1) - m - 1
    spread = float(np.dot(weights, x)) / (m * m)
    return float(accuracy - spread)


def crps(samples: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """CRPS per case. `samples` is (draws, cases); `observed` is (cases,)."""
    samples = np.asarray(samples, dtype=float)
    observed = np.asarray(observed, dtype=float)
    if samples.shape[1] != observed.shape[0]:
        raise ValueError(
            f"{samples.shape[1]} forecast columns against {observed.shape[0]} outcomes; "
            f"they must line up case for case or the scores are meaningless"
        )
    return np.array([crps_ensemble(samples[:, i], observed[i]) for i in range(observed.shape[0])])


# --- interval coverage --------------------------------------------------


def interval_coverage(samples: np.ndarray, observed: np.ndarray, level: float) -> dict[str, float]:
    """How often the central interval at `level` actually contains the outcome.

    A 90% interval should contain about 90% of outcomes. Materially below that
    means the model is overconfident and its floor and ceiling are too narrow;
    materially above means it is hedging and the interval is not saying much.

    Also reports the mean interval width, because coverage on its own is
    trivially satisfiable by predicting an enormous range.
    """
    if not 0 < level < 1:
        raise ValueError(f"level must be between 0 and 1, got {level}")
    tail = (1.0 - level) / 2.0
    lower = np.percentile(samples, 100 * tail, axis=0)
    upper = np.percentile(samples, 100 * (1.0 - tail), axis=0)
    inside = (observed >= lower) & (observed <= upper)
    return {
        "level": level,
        "coverage": float(inside.mean()),
        "mean_width": float((upper - lower).mean()),
        "n": int(observed.size),
    }


def coverage_table(
    samples: np.ndarray, observed: np.ndarray, levels=(0.5, 0.8, 0.9)
) -> list[dict[str, float]]:
    return [interval_coverage(samples, observed, level) for level in levels]


# --- PIT ----------------------------------------------------------------


def pit_values(samples: np.ndarray, observed: np.ndarray, seed: int = 0) -> np.ndarray:
    """Randomized probability integral transform, one value per case.

    If the forecasts are calibrated these are uniform on [0, 1]: a histogram
    that piles up at the edges means the intervals are too narrow, and one that
    bulges in the middle means they are too wide.

    Randomized because the forecasts are counts. For a discrete distribution
    the plain PIT is lumpy even under perfect calibration, so the convention is
    to draw uniformly between the probability strictly below the outcome and
    the probability at or below it. Without that correction the histogram looks
    misspecified when nothing is wrong.
    """
    rng = np.random.default_rng(seed)
    samples = np.asarray(samples, dtype=float)
    observed = np.asarray(observed, dtype=float)
    below = (samples < observed[None, :]).mean(axis=0)
    at_or_below = (samples <= observed[None, :]).mean(axis=0)
    return below + rng.random(observed.shape[0]) * (at_or_below - below)


def pit_histogram(pit: np.ndarray, bins: int = 10) -> np.ndarray:
    """Counts per bin, normalized so a calibrated model gives roughly 1.0
    everywhere."""
    counts, _ = np.histogram(pit, bins=bins, range=(0.0, 1.0))
    return counts / (pit.size / bins)


def pit_deviation(pit: np.ndarray, bins: int = 10) -> float:
    """One number for how far the PIT histogram is from flat.

    Mean absolute deviation from 1.0 across bins. Zero is perfect; roughly 0.1
    is a mild wobble; above about 0.3 the intervals are the wrong shape and the
    floor and ceiling should not be quoted.
    """
    return float(np.abs(pit_histogram(pit, bins) - 1.0).mean())


# --- summary ------------------------------------------------------------


def calibration_report(
    samples: np.ndarray, observed: np.ndarray, levels=(0.5, 0.8, 0.9), seed: int = 0
) -> dict:
    """Everything needed to decide whether the intervals can be quoted."""
    scores = crps(samples, observed)
    pit = pit_values(samples, observed, seed=seed)
    return {
        "n": int(observed.size),
        "crps_mean": float(scores.mean()),
        "crps_median": float(np.median(scores)),
        "mae_of_mean": float(np.abs(samples.mean(axis=0) - observed).mean()),
        "coverage": coverage_table(samples, observed, levels),
        "pit_deviation": pit_deviation(pit),
        "pit_histogram": pit_histogram(pit).tolist(),
    }


def render_report(report: dict, title: str = "calibration") -> str:
    lines = [f"=== {title} (n={report['n']}) ===", ""]
    lines.append(f"  CRPS mean          {report['crps_mean']:8.3f}   (lower is better)")
    lines.append(f"  CRPS median        {report['crps_median']:8.3f}")
    lines.append(
        f"  MAE of posterior mean {report['mae_of_mean']:5.3f}   "
        f"(point-forecast error, for reference)"
    )
    lines.append("")
    lines.append("  interval   nominal   actual   mean width")
    for row in report["coverage"]:
        lines.append(
            f"  {'central':9s}  {row['level']:6.0%}   {row['coverage']:6.1%}   "
            f"{row['mean_width']:10.1f}"
        )
    lines.append("")
    deviation = report["pit_deviation"]
    verdict = "flat" if deviation < 0.15 else ("wobbly" if deviation < 0.3 else "misshapen")
    lines.append(f"  PIT deviation      {deviation:8.3f}   ({verdict}; 0 is perfectly flat)")
    return "\n".join(lines)
