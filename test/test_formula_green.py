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


def test_the_3d_kernel_reduces_to_the_ring_kernel_at_zero_toroidal_separation():
    """The one property of `greens_function_3d` with a derivation behind it.

    A point source replaces the ring kernel's ``4*R*R0`` with
    ``4*R*R0*sin^2(dphi/2)``; this expression adds the second to the first, so
    they agree only where the sine vanishes (issue #356). Pinning that keeps a
    future correction honest about what it is allowed to change.
    """
    radius = np.array([0.9, 1.3, 1.7])
    height = np.array([0.0, 0.05, -0.2])
    ring = greens_function_2d(radius, height, R0=1.1, Z0=0.05)
    at_same_angle = greens_function_3d(
        radius, height, np.zeros_like(radius), R0=1.1, Z0=0.05, phi0=0.0
    )
    np.testing.assert_allclose(at_same_angle, ring, rtol=0, atol=0)

    apart = greens_function_3d(
        radius, height, np.full_like(radius, np.pi / 2), R0=1.1, Z0=0.05, phi0=0.0
    )
    assert np.all(apart < ring), "the extra sin^2 only ever enlarges the denominator"
