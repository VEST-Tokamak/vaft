"""Numeric coverage for the ``vaft.formula`` utils and green kernels no test called.

A caller-name sweep over ``test/`` turned up four: the two flux primitives,
exercised until now only through their ``equilibrium`` wrappers so a change to
either primitive alone was invisible; ``make_fit_function``, whose four models
and their aliases had no test at all; and ``calculate_distance``.

``pytest.approx`` passes when *either* its relative or its absolute tolerance is
met, and its default ``abs=1e-12`` swamps anything smaller, so every comparison
of a small value passes ``abs=0.0`` explicitly.
"""

import numpy as np
import pytest

from vaft.formula import (
    calculate_distance,
    calculate_poloidal_flux,
    calculate_toroidal_flux,
    make_fit_function,
    phi_from_Bphi,
    psi_from_RBtheta,
)


# --------------------------------------------------------------------------
# The two flux primitives
# --------------------------------------------------------------------------

def test_poloidal_flux_integrates_a_constant_integrand_exactly():
    l = np.linspace(0.0, 2.0, 101)
    R = np.full_like(l, 1.4)
    B_theta = np.full_like(l, 0.05)
    assert calculate_poloidal_flux(R, B_theta, l) == pytest.approx(
        1.4 * 0.05 * 2.0, rel=1e-12, abs=0.0
    )


def test_poloidal_flux_integrates_a_linear_integrand_exactly():
    # The trapezoidal rule is exact on a linear integrand, so this pins the
    # quadrature rather than its discretisation error.
    l = np.linspace(0.0, 2.0, 101)
    R = np.full_like(l, 1.0)
    B_theta = 0.01 + 0.03 * l
    assert calculate_poloidal_flux(R, B_theta, l) == pytest.approx(
        0.01 * 2.0 + 0.03 * 2.0**2 / 2, rel=1e-12, abs=0.0
    )


def test_poloidal_flux_adds_the_axis_offset_and_follows_the_path_direction():
    l = np.linspace(0.0, 2.0, 101)
    R = np.full_like(l, 1.4)
    B_theta = np.full_like(l, 0.05)
    base = calculate_poloidal_flux(R, B_theta, l)
    assert calculate_poloidal_flux(R, B_theta, l, psi_axis=0.25) == pytest.approx(
        base + 0.25, rel=1e-12, abs=0.0
    )
    # Traversing the path backwards flips the sign, which is the documented
    # convention rather than an accident of the quadrature.
    assert calculate_poloidal_flux(R, B_theta, l[::-1]) == pytest.approx(
        -base, rel=1e-12, abs=0.0
    )


def test_toroidal_flux_is_the_area_weighted_sum():
    B_phi = np.array([0.18, 0.20, 0.22])
    dA = np.array([0.01, 0.02, 0.03])
    assert calculate_toroidal_flux(B_phi, dA) == pytest.approx(
        float(np.sum(B_phi * dA)), rel=1e-13, abs=0.0
    )


def test_toroidal_flux_is_a_riemann_sum_not_a_quadrature_rule():
    # Documented in #358: no trapezoidal weighting, so a uniform field over a
    # uniform grid gives exactly B times the total area, with no end correction.
    dA = np.full(5, 0.02)
    assert calculate_toroidal_flux(np.full(5, 0.3), dA) == pytest.approx(
        0.3 * 0.1, rel=1e-13, abs=0.0
    )


def test_the_equilibrium_wrappers_are_these_primitives():
    # psi_from_RBtheta and phi_from_Bphi are thin physics-layer names for the
    # two above.  Pin that, so a change to one side cannot drift from the other.
    l = np.linspace(0.0, 2.0, 51)
    R = 1.4 + 0.1 * l
    B_theta = 0.05 - 0.01 * l
    assert psi_from_RBtheta(R, B_theta, l, psi_axis=0.3) == (
        calculate_poloidal_flux(R, B_theta, l, psi_axis=0.3)
    )
    B_phi = np.array([0.18, 0.20, 0.22])
    dA = np.array([0.01, 0.02, 0.03])
    assert phi_from_Bphi(B_phi, dA) == calculate_toroidal_flux(B_phi, dA)


# --------------------------------------------------------------------------
# make_fit_function
# --------------------------------------------------------------------------

COEFFS = (2.0, -1.5, 0.4)
X = np.array([0.0, 0.25, 0.5, 0.75, 1.0])


def _poly(x):
    return sum(c * x**k for k, c in enumerate(COEFFS))


def test_polynomial_mode_multiplies_the_series_by_the_edge_factor():
    f = make_fit_function("polynomial")
    assert f(X, *COEFFS) == pytest.approx((1.0 - X) * _poly(X), rel=1e-13, abs=0.0)


def test_free_polynomial_mode_is_the_bare_series():
    f = make_fit_function("free_polynomial")
    assert f(X, *COEFFS) == pytest.approx(_poly(X), rel=1e-13, abs=0.0)


def test_exponential_mode_multiplies_the_exponential_by_the_edge_factor():
    f = make_fit_function("exponential")
    assert f(X, *COEFFS) == pytest.approx(
        (1.0 - X) * np.exp(_poly(X)), rel=1e-13, abs=0.0
    )


def test_free_exponential_mode_is_the_bare_exponential():
    f = make_fit_function("free_exponential")
    assert f(X, *COEFFS) == pytest.approx(np.exp(_poly(X)), rel=1e-13, abs=0.0)


@pytest.mark.parametrize("mode", ["polynomial", "exponential"])
def test_the_constrained_modes_vanish_at_the_edge(mode):
    # The (1 - x) factor is the whole point of the constrained modes: whatever
    # the coefficients, the profile is zero at x = 1.
    f = make_fit_function(mode)
    assert float(f(1.0, *COEFFS)) == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("mode", ["free_polynomial", "free_exponential"])
def test_the_free_modes_do_not_vanish_at_the_edge(mode):
    f = make_fit_function(mode)
    assert abs(float(f(1.0, *COEFFS))) > 1e-6


@pytest.mark.parametrize("mode", ["exponential", "free_exponential"])
def test_the_exponential_modes_are_positive_inside_the_domain(mode):
    f = make_fit_function(mode)
    inside = np.linspace(0.0, 0.99, 50)
    assert np.all(f(inside, *COEFFS) > 0.0)


@pytest.mark.parametrize(
    "alias, canonical",
    [
        ("polynomial_unconstrained", "free_polynomial"),
        ("unconstrained_polynomial", "free_polynomial"),
        ("exp_free", "free_exponential"),
        ("exponential_unconstrained", "free_exponential"),
    ],
)
def test_every_documented_alias_resolves_to_its_canonical_model(alias, canonical):
    assert make_fit_function(alias)(X, *COEFFS) == pytest.approx(
        make_fit_function(canonical)(X, *COEFFS), rel=1e-13, abs=0.0
    )


@pytest.mark.parametrize("mode", ["POLYNOMIAL", "Free_Exponential", "ExPoNeNtIaL"])
def test_the_mode_name_is_case_insensitive(mode):
    assert callable(make_fit_function(mode))


def test_an_unknown_mode_is_rejected_rather_than_silently_defaulted():
    with pytest.raises(ValueError, match="Invalid fitting function"):
        make_fit_function("tanh")


def test_a_model_accepts_any_number_of_coefficients():
    f = make_fit_function("free_polynomial")
    assert float(f(2.0, 1.0)) == pytest.approx(1.0, rel=1e-13, abs=0.0)
    assert float(f(2.0, 1.0, 3.0)) == pytest.approx(7.0, rel=1e-13, abs=0.0)
    # No coefficients is a zero profile, not an error.
    assert float(f(2.0)) == pytest.approx(0.0, abs=1e-15)


def test_a_model_accepts_a_scalar_x():
    f = make_fit_function("free_polynomial")
    assert float(f(0.5, *COEFFS)) == pytest.approx(_poly(0.5), rel=1e-13, abs=0.0)


# --------------------------------------------------------------------------
# calculate_distance
# --------------------------------------------------------------------------

def test_distance_is_the_euclidean_norm_in_the_poloidal_plane():
    assert calculate_distance(1.0, 4.0, 2.0, 6.0) == pytest.approx(5.0, rel=1e-13)


def test_distance_is_zero_for_coincident_points_and_symmetric():
    assert calculate_distance(1.4, 1.4, 0.2, 0.2) == pytest.approx(0.0, abs=1e-15)
    assert calculate_distance(1.0, 1.6, 0.2, -0.3) == pytest.approx(
        calculate_distance(1.6, 1.0, -0.3, 0.2), rel=1e-13, abs=0.0
    )


def test_distance_broadcasts_over_arrays():
    r1 = np.array([0.0, 3.0])
    z1 = np.array([0.0, 4.0])
    assert calculate_distance(r1, 0.0, z1, 0.0) == pytest.approx(
        np.array([0.0, 5.0]), rel=1e-13, abs=1e-15
    )
