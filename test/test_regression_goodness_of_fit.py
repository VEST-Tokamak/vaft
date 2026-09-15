"""The inlined regression metrics reproduce what scikit-learn reported (#426).

`compute_metrics` used to import scikit-learn for two one-line metrics, which
made an optional dependency a hard one.  Replacing a metric is only defensible
if the replacement returns the same number, including at the edges where the
two could plausibly differ -- which is where the inlining first went wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from vaft.process.statistical_analysis import _goodness_of_fit


def test_the_metrics_match_their_textbook_definitions():
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    predicted = np.array([1.1, 1.9, 3.2, 3.7])

    r2, rmse, mae = _goodness_of_fit(actual, predicted)

    residual = actual - predicted
    expected_r2 = 1.0 - np.sum(residual**2) / np.sum((actual - actual.mean()) ** 2)
    assert r2 == pytest.approx(expected_r2)
    assert rmse == pytest.approx(np.sqrt(np.mean(residual**2)))
    assert mae == pytest.approx(np.mean(np.abs(residual)))


def test_a_perfect_fit_scores_one_and_errs_by_nothing():
    actual = np.array([1.0, 2.0, 3.0])
    r2, rmse, mae = _goodness_of_fit(actual, actual)
    assert (r2, rmse, mae) == (1.0, 0.0, 0.0)


@pytest.mark.parametrize(
    "predicted, expected_r2",
    [
        (np.array([2.0, 2.0, 2.0]), 1.0),   # exact, on a series with no variance
        (np.array([2.0, 2.1, 1.9]), 0.0),   # inexact, with nothing to explain
    ],
)
def test_a_constant_series_scores_the_way_scikit_learn_scored_it(predicted, expected_r2):
    """Not NaN.

    A constant observed series has zero total sum of squares, so the ratio is
    undefined.  ``r2_score`` reports 1.0 for an exact prediction and 0.0
    otherwise; callers threshold on that, and a NaN would pass every comparison
    silently rather than failing one.
    """
    r2, _, _ = _goodness_of_fit(np.array([2.0, 2.0, 2.0]), predicted)
    assert r2 == expected_r2
    assert not np.isnan(r2)
