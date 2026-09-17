"""Degenerate denominators in vaft.formula return nan and say so (issue #357).

Four ratios divided without a guard, so a degenerate input propagated ``inf``
downstream with nothing to say where it started. The degenerate case is not
hypothetical: packaged VEST samples contain an equilibrium slice whose axis and
boundary flux are equal, which is why ``vaft/omas/update.py`` handles it.

``rhoN_from_qpsiN`` additionally rebuilt its cumulative trapezoid from scratch at
every sample. The replacement has to be numerically identical, not merely
close-looking, so that is asserted rather than assumed.
"""

import warnings

import numpy as np
import pytest

from vaft.formula.equilibrium import peaking_factor, psi_normalised, rhoN_from_qpsiN
from vaft.formula.utils import (
    calculate_peaking_factor,
    calculate_volume_weighted_average,
    normalize_profile,
    trapz_integral,
)


def _loop_rhoN(psiN, qpsiN):
    """The O(N^2) implementation this replaced, kept as the reference."""
    num = np.array([trapz_integral(psiN[: i + 1], qpsiN[: i + 1]) for i in range(len(psiN))])
    return np.sqrt(num / trapz_integral(psiN, qpsiN))


@pytest.mark.parametrize(
    "call, because",
    [
        (lambda: normalize_profile(0.5, 1.0, 1.0), "x_boundary - x_axis"),
        (lambda: psi_normalised(0.5, 1.0, 1.0), "x_boundary - x_axis"),
        (lambda: calculate_peaking_factor(3.0, 0.0), "volume_avg"),
        (lambda: peaking_factor(3.0, 0.0), "volume_avg"),
        (
            lambda: calculate_volume_weighted_average(
                np.array([1.0, 2.0]), np.array([0.0, 0.0])
            ),
            "sum(V)",
        ),
    ],
)
def test_a_vanishing_denominator_warns_and_returns_nan(call, because):
    with pytest.warns(RuntimeWarning, match="returning nan"):
        result = call()
    assert np.isnan(result)


def test_the_degenerate_slice_no_longer_propagates_inf():
    """The negative control: an unguarded divide is what this used to do."""
    with np.errstate(divide="ignore", invalid="ignore"):
        unguarded = np.float64(0.5 - 1.0) / np.float64(1.0 - 1.0)
    assert np.isinf(unguarded)
    with pytest.warns(RuntimeWarning):
        assert np.isnan(normalize_profile(0.5, 1.0, 1.0))


def test_a_healthy_profile_is_untouched_and_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert normalize_profile(0.5, 0.0, 2.0) == pytest.approx(0.25)
        assert calculate_peaking_factor(3.0, 1.5) == pytest.approx(2.0)
        assert calculate_volume_weighted_average(
            np.array([1.0, 3.0]), np.array([1.0, 1.0])
        ) == pytest.approx(2.0)


@pytest.mark.parametrize("n", [5, 33, 257])
def test_the_vectorised_cumulative_integral_reproduces_the_loop(n):
    """Equivalence to floating-point noise, so the speedup changes no science."""
    rng = np.random.default_rng(42)
    psiN = np.sort(rng.uniform(0.0, 1.0, n))
    psiN[0], psiN[-1] = 0.0, 1.0
    qpsiN = 1.0 + 3.0 * psiN**2
    np.testing.assert_allclose(rhoN_from_qpsiN(psiN, qpsiN), _loop_rhoN(psiN, qpsiN), rtol=0, atol=1e-15)


def test_rhoN_is_zero_on_axis_one_at_the_edge_and_monotonic():
    psiN = np.linspace(0.0, 1.0, 65)
    rho = rhoN_from_qpsiN(psiN, 1.0 + psiN**2)
    assert rho[0] == pytest.approx(0.0)
    assert rho[-1] == pytest.approx(1.0)
    assert np.all(np.diff(rho) >= 0.0)


def test_a_uniformly_negative_q_is_fine_because_both_integrals_flip():
    """A COCOS sign on q cancels in the ratio; only a sign *change* does not."""
    psiN = np.linspace(0.0, 1.0, 17)
    positive = rhoN_from_qpsiN(psiN, 1.0 + psiN**2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        negative = rhoN_from_qpsiN(psiN, -(1.0 + psiN**2))
    np.testing.assert_allclose(negative, positive)


def test_a_sign_changing_q_warns_rather_than_returning_a_bare_nan():
    """sqrt of a negative cumulative ratio is nan; that should not be silent."""
    psiN = np.linspace(0.0, 1.0, 17)
    # Positive near the axis, negative outside it, so the running integral and
    # the total end up with opposite signs and the ratio goes negative.
    qpsiN = 0.8 - 2.0 * psiN
    with pytest.warns(RuntimeWarning, match="changes sign"):
        rho = rhoN_from_qpsiN(psiN, qpsiN)
    assert np.isnan(rho).any()


def test_a_zero_total_integral_warns_rather_than_dividing():
    psiN = np.linspace(0.0, 1.0, 9)
    with pytest.warns(RuntimeWarning, match="total integral"):
        assert np.isnan(rhoN_from_qpsiN(psiN, np.zeros_like(psiN))).all()
