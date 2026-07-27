"""Regression coverage for the Green-function elliptic-integral wrappers."""

import numpy as np
from scipy.special import ellipe, ellipk

from vaft.formula.green import (
    complete_elliptic_integral_e,
    complete_elliptic_integral_k,
    greens_function_2d,
    greens_function_3d,
    greens_integral_2d,
    greens_integral_3d,
)


def test_complete_elliptic_integrals_match_scipy_for_scalars_and_arrays():
    values = np.array([0.0, 0.25, 0.5, 0.9])

    assert complete_elliptic_integral_k(0.5) == ellipk(0.5)
    assert complete_elliptic_integral_e(0.5) == ellipe(0.5)
    np.testing.assert_allclose(complete_elliptic_integral_k(values), ellipk(values))
    np.testing.assert_allclose(complete_elliptic_integral_e(values), ellipe(values))


def test_green_functions_and_integrals_evaluate_without_numpy_attribute_errors():
    radius = np.array([0.8, 1.0, 1.2])
    height = np.array([-0.2, 0.0, 0.2])
    angle = np.array([0.1, 0.2, 0.3])
    source = np.array([1.0, 2.0, 3.0])

    g2d = greens_function_2d(radius, height, R0=1.1, Z0=0.05)
    g3d = greens_function_3d(radius, height, angle, R0=1.1, Z0=0.05, phi0=0.0)

    assert np.all(np.isfinite(g2d))
    assert np.all(np.isfinite(g3d))
    assert np.isfinite(greens_integral_2d(radius, height, 1.1, 0.05, source))
    assert np.isfinite(greens_integral_3d(radius, height, angle, 1.1, 0.05, 0.0, source))
