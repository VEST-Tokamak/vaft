"""Unit coverage for the pure statistical kernels (issue #186).

Array and scalar inputs only -- nothing here constructs an ODS, so these tests
pin the mathematics independently of EFIT, OMAS and the validation pipeline.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from vaft.formula.statistics import (
    bias_standard_error,
    chi_squared,
    dynamic_range,
    fractional_rms_improvement,
    lag1_autocorrelation,
    linear_trend,
    log10_decay_rate,
    median_absolute_deviation,
    monotonic_fraction,
    noise_band,
    normalized_residual,
    outlier_fraction,
    pearson_correlation,
    reduced_chi_squared,
    relative_spread,
    residual_bias,
    rms,
    robust_z_scores,
    runs_test_z,
    sigma_threshold_crossing,
    sigma_unit_factor,
)


# ---------------------------------------------------------------------------
# Residual magnitude
# ---------------------------------------------------------------------------

def test_rms_matches_the_analytic_quadratic_mean():
    assert rms([3.0, 4.0]) == pytest.approx(math.sqrt(12.5))
    assert rms([-2.0, 2.0, -2.0]) == pytest.approx(2.0)
    assert rms(np.zeros(5)) == 0.0


def test_rms_ignores_non_finite_entries_rather_than_propagating_them():
    assert rms([3.0, np.nan, 4.0, np.inf]) == pytest.approx(math.sqrt(12.5))


def test_rms_is_nan_for_empty_and_all_non_finite_input_without_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert math.isnan(rms([]))
        assert math.isnan(rms([np.nan, np.nan]))
        assert math.isnan(rms(np.array([], dtype=float)))


def test_fractional_rms_improvement_spans_perfect_neutral_and_worse():
    baseline = np.array([1.0, -1.0, 1.0, -1.0])

    assert fractional_rms_improvement(baseline, np.zeros(4)) == pytest.approx(1.0)
    assert fractional_rms_improvement(baseline, baseline) == pytest.approx(0.0)
    assert fractional_rms_improvement(baseline, 2.0 * baseline) == pytest.approx(-1.0)


def test_fractional_rms_improvement_is_nan_without_a_usable_baseline():
    assert math.isnan(fractional_rms_improvement(np.zeros(4), np.ones(4)))
    assert math.isnan(fractional_rms_improvement([], np.ones(4)))


# ---------------------------------------------------------------------------
# Goodness of fit
# ---------------------------------------------------------------------------

def test_normalized_residual_divides_by_the_effective_uncertainty():
    z = normalized_residual([2.0, -4.0], [0.5, 0.25], 2.0)

    np.testing.assert_allclose(z, [0.5, -0.5])


def test_normalized_residual_is_all_nan_when_the_unit_factor_is_unusable():
    for k in (0.0, float("nan"), float("inf")):
        z = normalized_residual(np.ones(3), np.ones(3), k)
        assert z.shape == (3,)
        assert np.all(np.isnan(z))


def test_chi_squared_sums_squared_normalized_terms_and_drops_zero_sigma():
    assert chi_squared([3.0, 4.0], [1.0, 2.0]) == pytest.approx(9.0 + 4.0)
    assert chi_squared([3.0, 4.0, 1.0], [1.0, 2.0, 0.0]) == pytest.approx(13.0)
    assert math.isnan(chi_squared([1.0], [0.0]))


def test_reduced_chi_squared_divides_by_the_degrees_of_freedom():
    assert reduced_chi_squared(24.0, 12.0) == pytest.approx(2.0)
    for dof in (0.0, -1.0, float("nan")):
        assert math.isnan(reduced_chi_squared(24.0, dof))


def test_sigma_unit_factor_recovers_a_common_factor_with_zero_spread():
    residual = np.array([1.0, -2.0, 4.0])
    weight = np.array([2.0, 1.0, 0.5])
    k = 2.0
    chi = (residual * weight / k) ** 2

    recovered, spread = sigma_unit_factor(residual, weight, chi)

    assert recovered == pytest.approx(k)
    assert spread == pytest.approx(0.0)


def test_sigma_unit_factor_reports_scatter_when_one_channel_disagrees():
    residual = np.array([1.0, 1.0, 1.0, 10.0])
    weight = np.ones(4)
    chi = np.array([1.0, 1.0, 1.0, 1.0])

    recovered, spread = sigma_unit_factor(residual, weight, chi)

    assert recovered == pytest.approx(1.0)
    assert spread == pytest.approx(9.0)


def test_sigma_unit_factor_is_nan_when_no_channel_qualifies():
    for weight, chi in (
        (np.zeros(3), np.ones(3)),          # no positive weight
        (np.ones(3), np.zeros(3)),          # no positive chi-square
        (np.ones(3), np.full(3, np.nan)),   # no finite chi-square
    ):
        recovered, spread = sigma_unit_factor(np.ones(3), weight, chi)
        assert math.isnan(recovered)
        assert math.isnan(spread)


# ---------------------------------------------------------------------------
# Bias and structure
# ---------------------------------------------------------------------------

def test_residual_bias_is_the_signed_mean_over_finite_entries():
    assert residual_bias([1.0, 3.0, np.nan]) == pytest.approx(2.0)
    assert residual_bias([-1.0, 1.0]) == pytest.approx(0.0)
    assert math.isnan(residual_bias([np.nan]))


def test_bias_standard_error_is_one_over_root_n():
    assert bias_standard_error(4) == pytest.approx(0.5)
    assert bias_standard_error(100) == pytest.approx(0.1)
    assert math.isnan(bias_standard_error(0))
    assert math.isnan(bias_standard_error(-3))


def test_outlier_fraction_counts_the_tail_beyond_a_threshold():
    values = np.array([0.5, 2.5, -3.5, 1.0])

    assert outlier_fraction(values, 2.0) == pytest.approx(0.5)
    assert outlier_fraction(values, 3.0) == pytest.approx(0.25)
    assert outlier_fraction(values, 10.0) == 0.0
    assert math.isnan(outlier_fraction([np.nan, np.nan], 2.0))


def test_lag1_autocorrelation_is_one_for_a_ramp_and_negative_for_alternation():
    # The denominator runs over all n terms and the numerator over n-1, so a
    # perfectly smooth ramp lands just short of 1 rather than exactly on it.
    ramp = np.arange(101, dtype=float)
    assert lag1_autocorrelation(ramp) == pytest.approx(0.9702970297, abs=1e-9)

    alternating = np.array([1.0, -1.0] * 8)
    assert lag1_autocorrelation(alternating) == pytest.approx(-15.0 / 16.0)


def test_lag1_autocorrelation_guards_short_and_constant_samples():
    assert math.isnan(lag1_autocorrelation([1.0, 2.0]))
    assert math.isnan(lag1_autocorrelation(np.full(10, 3.0)))
    assert math.isnan(lag1_autocorrelation([1.0, np.nan, 2.0]))


def test_runs_test_z_is_strongly_negative_for_a_clustered_sign_sequence():
    clustered = np.array([1.0] * 10 + [-1.0] * 10)

    # R = 2, E[R] = 11, Var[R] = 2*100*(200-20)/(400*19)
    expected = (2.0 - 11.0) / math.sqrt(2.0 * 100.0 * 180.0 / (400.0 * 19.0))
    assert runs_test_z(clustered) == pytest.approx(expected)
    assert runs_test_z(clustered) < -2.0


def test_runs_test_z_is_strongly_positive_for_a_perfectly_alternating_sequence():
    alternating = np.array([1.0, -1.0] * 10)

    assert runs_test_z(alternating) > 2.0


def test_runs_test_z_guards_short_single_signed_and_zero_only_input():
    assert math.isnan(runs_test_z([1.0, -1.0]))
    assert math.isnan(runs_test_z(np.ones(10)))
    assert math.isnan(runs_test_z(np.zeros(10)))
    assert math.isnan(runs_test_z([np.nan] * 10))


def test_relative_spread_normalizes_the_range_by_the_largest_magnitude():
    assert relative_spread([8.0, 10.0]) == pytest.approx(0.2)
    assert relative_spread([1.0, 1.0, 1.0]) == 0.0
    # Exact agreement at zero is agreement, not an undefined ratio.
    assert relative_spread([0.0, 0.0]) == 0.0
    assert math.isnan(relative_spread([1.0]))
    assert math.isnan(relative_spread([1.0, np.nan]))


# ---------------------------------------------------------------------------
# Threshold crossing
# ---------------------------------------------------------------------------

def test_noise_band_returns_the_population_mean_and_standard_deviation():
    baseline, noise = noise_band([1.0, 3.0, np.nan])

    assert baseline == pytest.approx(2.0)
    assert noise == pytest.approx(1.0)


def test_noise_band_is_nan_for_an_empty_reference_without_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        baseline, noise = noise_band([])
    assert math.isnan(baseline) and math.isnan(noise)


def test_sigma_threshold_crossing_finds_the_first_emergence_after_the_window():
    time = np.arange(10, dtype=float)
    values = np.array([0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 0.0, 20.0, 30.0])
    window = time < 6.0

    assert sigma_threshold_crossing(time, values, window, sigma=3.0) == 8.0


def test_sigma_threshold_crossing_ignores_excursions_inside_the_window():
    time = np.arange(6, dtype=float)
    values = np.array([0.0, 50.0, -1.0, 1.0, 0.0, 0.0])
    window = time < 4.0

    assert math.isnan(sigma_threshold_crossing(time, values, window, sigma=3.0))


def test_sigma_threshold_crossing_guards_short_and_noiseless_references():
    time = np.arange(5, dtype=float)
    values = np.array([0.0, 0.0, 0.0, 0.0, 100.0])

    # Zero noise inside the window: no scale to measure the excursion against.
    assert math.isnan(
        sigma_threshold_crossing(time, values, time < 4.0, sigma=3.0)
    )
    # Fewer than two reference samples.
    assert math.isnan(
        sigma_threshold_crossing(time, values, time < 1.0, sigma=3.0)
    )


# ---------------------------------------------------------------------------
# Convergence history
# ---------------------------------------------------------------------------

def test_monotonic_fraction_counts_decreasing_steps():
    assert monotonic_fraction([4.0, 3.0, 2.0, 1.0]) == pytest.approx(1.0)
    assert monotonic_fraction([1.0, 2.0, 3.0]) == 0.0
    assert monotonic_fraction([2.0, 1.0, 2.0, 1.0]) == pytest.approx(2.0 / 3.0)
    assert math.isnan(monotonic_fraction([1.0]))


def test_monotonic_fraction_treats_a_non_finite_comparison_as_no_decrease():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # Both pairs involve the nan, and a nan comparison is never a decrease.
        assert monotonic_fraction([4.0, np.nan, 1.0]) == 0.0


def test_log10_decay_rate_recovers_the_slope_of_a_geometric_decay():
    history = 10.0 ** (-0.5 * np.arange(8, dtype=float))

    assert log10_decay_rate(history) == pytest.approx(-0.5)


def test_log10_decay_rate_uses_only_the_requested_tail():
    # A steep early transient followed by a flat tail: the tail is what counts.
    history = np.array([1e3, 1e1, 1.0, 1.0, 1.0, 1.0, 1.0])

    assert log10_decay_rate(history, tail=4) == pytest.approx(0.0)


def test_log10_decay_rate_guards_short_and_non_positive_histories():
    assert math.isnan(log10_decay_rate([1.0, 0.1]))
    assert math.isnan(log10_decay_rate(np.zeros(6)))
    assert math.isnan(log10_decay_rate([1.0, -1.0, 0.0, 1e-3]))
    assert math.isnan(log10_decay_rate(10.0 ** -np.arange(6.0), tail=0))


# ---------------------------------------------------------------------------
# Robust scale, trend and agreement
# ---------------------------------------------------------------------------

def test_median_absolute_deviation_is_unmoved_by_a_corrupted_minority():
    """The property the standard deviation lacks: half the sample may be
    arbitrarily bad before the estimate moves at all.
    """
    clean = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    corrupted = np.array([1.0, 2.0, 3.0, 4.0, 1.0e6])

    assert median_absolute_deviation(clean) == pytest.approx(1.0)
    assert median_absolute_deviation(corrupted) == pytest.approx(1.0)
    assert np.std(corrupted) > 1.0e5


def test_median_absolute_deviation_is_zero_for_a_repeated_majority():
    assert median_absolute_deviation([1.0, 1.0, 1.0, 5.0]) == 0.0
    assert math.isnan(median_absolute_deviation([]))
    assert math.isnan(median_absolute_deviation([np.nan, np.nan]))


def test_robust_z_scores_expose_a_spike_a_plain_z_score_would_hide():
    values = np.concatenate([np.zeros(40), [50.0], np.zeros(40)])

    robust = robust_z_scores(values)
    plain = (values - values.mean()) / values.std()

    assert np.isinf(robust[40])
    # The spike is 9 sigma out on its own inflated scale -- large, but the same
    # scale it created, so a fixed threshold in those units drifts with the
    # anomaly it is meant to catch.
    assert plain[40] == pytest.approx(math.sqrt(80), rel=0.05)


def test_robust_z_scores_keep_their_position_and_mark_non_finite_samples():
    scores = robust_z_scores([1.0, 2.0, 3.0, 100.0, np.nan])

    assert scores.shape == (5,)
    assert math.isnan(scores[4])
    assert scores[3] > 10.0
    assert abs(scores[1]) < 1.0


def test_robust_z_scores_report_infinity_rather_than_nan_for_a_zero_scale():
    """A perfectly linear ramp differenced has zero MAD, yet a single step out
    of it is the clearest possible anomaly -- not an undefined one.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scores = robust_z_scores(np.diff([0.0, 1.0, 2.0, 3.0, 9.0, 10.0]))

    assert scores.tolist() == [0.0, 0.0, 0.0, math.inf, 0.0]
    assert np.isnan(robust_z_scores([np.nan, np.nan])).all()


def test_linear_trend_recovers_a_known_slope_and_ignores_offset():
    time = np.linspace(0.0, 2.0, 21)

    assert linear_trend(time, 3.0 * time + 7.0) == pytest.approx(3.0)
    assert linear_trend(time, 3.0 * time - 1000.0) == pytest.approx(3.0)
    # Unlike a mean, a pure offset is invisible to it: "offset" and "walking"
    # are different faults and need different statistics.
    assert linear_trend(time, np.full_like(time, 5.0)) == pytest.approx(0.0)


def test_linear_trend_guards_degenerate_and_mismatched_inputs():
    assert math.isnan(linear_trend([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))
    assert math.isnan(linear_trend([1.0], [2.0]))
    assert math.isnan(linear_trend([1.0, np.nan], [np.nan, 2.0]))
    with pytest.raises(ValueError, match="same length"):
        linear_trend([1.0, 2.0], [1.0])


def test_pearson_correlation_separates_shape_from_amplitude():
    """A model with the right dynamics and the wrong gain correlates perfectly
    while leaving a large residual -- the distinction an RMS alone hides.
    """
    time = np.linspace(0.0, 1.0, 50)
    measured = np.sin(2.0 * np.pi * time)

    assert pearson_correlation(measured, 4.0 * measured) == pytest.approx(1.0)
    assert pearson_correlation(measured, -measured) == pytest.approx(-1.0)
    assert rms(measured - 4.0 * measured) > rms(measured)


def test_pearson_correlation_guards_zero_variance_and_short_inputs():
    assert math.isnan(pearson_correlation([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))
    assert math.isnan(pearson_correlation([1.0], [2.0]))
    with pytest.raises(ValueError, match="same length"):
        pearson_correlation([1.0, 2.0], [1.0])


def test_dynamic_range_stays_meaningful_for_a_signal_that_straddles_zero():
    """`relative_spread` divides by `max|x|`, which a symmetric swing collapses;
    the raw span is what a residual is compared against.
    """
    swing = np.array([-1.0, 0.0, 1.0])

    assert dynamic_range(swing) == pytest.approx(2.0)
    assert relative_spread(swing) == pytest.approx(2.0)
    assert dynamic_range(np.array([9.0, 10.0])) == pytest.approx(1.0)
    assert relative_spread(np.array([9.0, 10.0])) == pytest.approx(0.1)


def test_dynamic_range_guards_empty_and_constant_input():
    assert dynamic_range([4.0, 4.0, 4.0]) == 0.0
    assert math.isnan(dynamic_range([]))
    assert math.isnan(dynamic_range([np.nan]))
