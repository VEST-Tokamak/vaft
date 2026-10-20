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
    gradient,
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


def test_toroidal_flux_of_a_uniform_field_is_field_times_area():
    R = np.linspace(0.1, 0.8, 8)
    Z = np.linspace(-0.5, 0.5, 11)
    B_phi = np.full((R.size, Z.size), 0.3)
    assert calculate_toroidal_flux(B_phi, R, Z) == pytest.approx(
        0.3 * 0.7 * 1.0, rel=1e-13, abs=0.0
    )


def test_toroidal_flux_of_the_vacuum_field_converges_at_second_order():
    # B_phi = B0 R0 / R over a rectangle: Phi = B0 R0 dZ ln(R2/R1).  The Riemann
    # sum the function used to be (#358) is first order; the trapezoidal rule
    # must quarter the error when the spacing halves.
    B0, R0, R1, R2, Z1, Z2 = 0.15, 0.4, 0.1, 0.8, -0.6, 0.6
    exact = B0 * R0 * (Z2 - Z1) * np.log(R2 / R1)
    errors = []
    for n in (21, 41, 81):
        R = np.linspace(R1, R2, n)
        Z = np.linspace(Z1, Z2, 7)
        B_phi = np.broadcast_to((B0 * R0 / R)[:, None], (n, Z.size))
        errors.append(abs(calculate_toroidal_flux(B_phi, R, Z) - exact))
    assert errors[-1] / exact < 1e-3
    assert errors[0] / errors[1] == pytest.approx(4.0, rel=0.05)
    assert errors[1] / errors[2] == pytest.approx(4.0, rel=0.05)


def test_toroidal_flux_accepts_ij_meshes_and_either_axis_direction():
    R = np.linspace(0.1, 0.8, 30)
    Z = np.linspace(-0.5, 0.5, 25)
    RR, ZZ = np.meshgrid(R, Z, indexing="ij")
    B_phi = 0.06 / RR
    reference = calculate_toroidal_flux(B_phi, R, Z)
    assert calculate_toroidal_flux(B_phi, RR, ZZ) == pytest.approx(reference, rel=1e-14)
    # Storing an axis backwards does not flip the area.
    assert calculate_toroidal_flux(B_phi[::-1], R[::-1], Z) == pytest.approx(reference, rel=1e-14)
    assert calculate_toroidal_flux(B_phi[:, ::-1], R, Z[::-1]) == pytest.approx(reference, rel=1e-14)
    # The sign is the field's.
    assert calculate_toroidal_flux(-B_phi, R, Z) == pytest.approx(-reference, rel=1e-14)


def test_toroidal_flux_over_a_masked_disk_converges_to_field_times_area():
    # The staircase boundary is first order: the error must shrink with the
    # grid, and be small on a fine one.
    a, R0, B = 0.2, 0.45, 0.1
    errors = []
    for n in (41, 161):
        R = np.linspace(R0 - 1.2 * a, R0 + 1.2 * a, n)
        Z = np.linspace(-1.2 * a, 1.2 * a, n)
        RR, ZZ = np.meshgrid(R, Z, indexing="ij")
        inside = (RR - R0) ** 2 + ZZ**2 <= a**2
        flux = calculate_toroidal_flux(np.full(RR.shape, B), R, Z, mask=inside)
        errors.append(abs(flux - B * np.pi * a**2) / (B * np.pi * a**2))
    assert errors[1] < errors[0]
    assert errors[1] < 0.02


def test_toroidal_flux_rejects_a_grid_that_does_not_match():
    R = np.linspace(0.1, 0.8, 8)
    Z = np.linspace(-0.5, 0.5, 11)
    B_phi = np.ones((R.size, Z.size))
    with pytest.raises(ValueError, match="points"):
        calculate_toroidal_flux(B_phi, Z, R)
    with pytest.raises(ValueError, match="monotonic"):
        calculate_toroidal_flux(B_phi, np.r_[R[:4], R[:4]], Z)
    RR, ZZ = np.meshgrid(R, Z, indexing="xy")
    with pytest.raises(ValueError):
        calculate_toroidal_flux(B_phi, RR, ZZ)
    with pytest.raises(ValueError, match="mask"):
        calculate_toroidal_flux(B_phi, R, Z, mask=np.ones((2, 2), bool))
    with pytest.raises(TypeError):
        calculate_toroidal_flux(B_phi)
    with pytest.raises(TypeError):
        calculate_toroidal_flux(B_phi, R, Z, dA=np.ones_like(B_phi))


def test_toroidal_flux_legacy_area_sum_is_deprecated_but_unchanged():
    B_phi = np.array([0.18, 0.20, 0.22])
    dA = np.array([0.01, 0.02, 0.03])
    expected = float(np.sum(B_phi * dA))
    with pytest.warns(DeprecationWarning, match="#358"):
        assert calculate_toroidal_flux(B_phi, dA) == pytest.approx(expected, rel=1e-13, abs=0.0)
    with pytest.warns(DeprecationWarning, match="#358"):
        assert calculate_toroidal_flux(B_phi, dA=dA) == pytest.approx(expected, rel=1e-13, abs=0.0)


def test_the_equilibrium_wrappers_are_these_primitives():
    # psi_from_RBtheta and phi_from_Bphi are thin physics-layer names for the
    # two above.  Pin that, so a change to one side cannot drift from the other.
    l = np.linspace(0.0, 2.0, 51)
    R = 1.4 + 0.1 * l
    B_theta = 0.05 - 0.01 * l
    assert psi_from_RBtheta(R, B_theta, l, psi_axis=0.3) == (
        calculate_poloidal_flux(R, B_theta, l, psi_axis=0.3)
    )
    R = np.linspace(0.1, 0.8, 8)
    Z = np.linspace(-0.5, 0.5, 11)
    B_phi = 0.06 / R[:, None] + 0.0 * Z[None, :]
    inside = np.abs(Z)[None, :] + 0.0 * R[:, None] < 0.3
    assert phi_from_Bphi(B_phi, R, Z, inside) == calculate_toroidal_flux(B_phi, R, Z, inside)


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


# --------------------------------------------------------------------------
# gradient
# --------------------------------------------------------------------------

def test_gradient_of_a_1d_profile_is_numpy_gradient():
    x = np.array([0.0, 0.1, 0.25, 0.5, 0.6, 1.0])
    y = np.sin(3.0 * x)
    np.testing.assert_array_equal(gradient(x, y), np.gradient(y, x))


def test_gradient_of_a_2d_map_stacks_the_1d_derivatives_of_its_columns():
    # A single coordinate with a 2-D y used to raise "invalid number of
    # arguments" from numpy; the documented contract is the first axis.
    x = np.linspace(0.0, 1.0, 17) ** 1.5
    columns = np.linspace(0.5, 2.0, 5)
    y = np.exp(np.outer(x, columns))
    expected = np.stack([np.gradient(y[:, j], x) for j in range(y.shape[1])], axis=1)
    np.testing.assert_array_equal(gradient(x, y), expected)
