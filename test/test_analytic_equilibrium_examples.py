"""Teaching builders for analytic equilibria: Cerfon-Freidberg Solov'ev and Miller families.

``solovev_example`` must produce the three boundary topologies the boundary
classifier distinguishes, with the X-points where they were asked for and a
flux map that satisfies the Grad-Shafranov equation with its own constant
sources.  ``miller_surfaces`` must return surfaces whose evaluated shape has
the requested elongation and triangularity.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import mu_0

from vaft.data.equilibrium import EquilibriumData, MillerSurface, Topology
from vaft.process import _equilibrium_parametric as parametric
from vaft.process.equilibrium import (
    derive_boundary_representation,
    evaluate_miller,
    grad_shafranov_operator,
    miller_surfaces,
    solovev_example,
)


# --- the Cerfon-Freidberg basis ---------------------------------------------------


def _star(terms, x, y):
    """x d/dx (1/x dpsi/dx) + d2psi/dy2, evaluated from the term list."""
    d1 = parametric._cf_dx(terms)
    return (
        parametric._cf_eval(parametric._cf_dx(d1), x, y)
        - parametric._cf_eval(d1, x, y) / x
        + parametric._cf_eval(parametric._cf_dy(parametric._cf_dy(terms)), x, y)
    )


def test_every_homogeneous_basis_function_is_annihilated_by_delta_star():
    rng = np.random.default_rng(0)
    x = rng.uniform(0.3, 1.8, 200)
    y = rng.uniform(-1.2, 1.2, 200)
    assert len(parametric._CF_BASIS) == 12
    for index, terms in enumerate(parametric._CF_BASIS):
        scale = np.max(np.abs(parametric._cf_eval(terms, x, y))) + 1.0
        assert np.max(np.abs(_star(terms, x, y))) < 1e-10 * scale, index


def test_basis_parity_even_then_odd_in_y():
    x = np.array([0.7, 1.1, 1.4]); y = np.array([0.3, -0.5, 0.8])
    for index, terms in enumerate(parametric._CF_BASIS):
        even = parametric._cf_eval(terms, x, -y)
        sign = 1.0 if index < 7 else -1.0
        np.testing.assert_allclose(even, sign * parametric._cf_eval(terms, x, y), atol=1e-12)


def test_basis_matches_a_numerical_delta_star_too():
    """Independent of the term algebra: a finite-difference Delta* on a grid."""
    x = np.linspace(0.5, 1.5, 201); y = np.linspace(-0.6, 0.6, 201)
    xm, ym = np.meshgrid(x, y, indexing="ij")
    for index, terms in enumerate(parametric._CF_BASIS):
        psi = parametric._cf_eval(terms, xm, ym)
        star = grad_shafranov_operator(psi, x, y)[3:-3, 3:-3]
        # Compare against the size of the terms that cancel, not of psi itself.
        psi_yy = np.gradient(np.gradient(psi, y, axis=1), y, axis=1)[3:-3, 3:-3]
        assert np.max(np.abs(star)) < 1e-3 * (np.max(np.abs(psi_yy)) + 1.0), index


@pytest.mark.parametrize("a", [-0.2, 0.0, 0.5, 0.9])
def test_particular_solution_carries_the_normalized_source(a):
    x = np.linspace(0.4, 1.6, 50); y = np.zeros_like(x)
    np.testing.assert_allclose(_star(parametric._cf_particular(a), x, y), (1 - a) * x**2 + a, atol=1e-12)


# --- solovev_example ----------------------------------------------------------------


_EXPECTED = {
    "limited": Topology.LIMITED,
    "single_null": Topology.LOWER_SINGLE_NULL,
    "double_null": Topology.DOUBLE_NULL,
}


@pytest.fixture(scope="module", params=sorted(_EXPECTED))
def example(request):
    return request.param, solovev_example(request.param)


def test_topology_is_the_one_requested(example):
    topology, eq = example
    assert isinstance(eq, EquilibriumData)
    boundary = derive_boundary_representation(eq)
    assert boundary.topology is _EXPECTED[topology], boundary.reason


def test_active_xpoints_sit_where_requested(example):
    topology, eq = example
    requested = eq.metadata["x_points_requested"]
    active = [point for point in derive_boundary_representation(eq).x_points if point.active]
    assert len(active) == len(requested)
    minor = eq.metadata["shape"]["minor_radius"]
    for r, z in requested:
        distance = min(np.hypot(point.r - r, point.z - z) for point in active)
        assert distance < 0.02 * minor, (topology, r, z, distance)
    if topology == "single_null":
        assert requested[0][1] < eq.magnetic_axis[1]
    if topology == "limited":
        assert requested == ()


def test_lcfs_shape_roughly_matches_the_request(example):
    """The limited boundary has the requested shape; an X-point side is 1.1 times it.

    Cerfon and Freidberg put the X-point at ``(1 - 1.1 delta eps, 1.1 kappa eps)``,
    so on that side the separatrix reaches 10 percent further in height and in
    triangularity than the nominal shape -- by construction, not by error.
    """
    topology, eq = example
    shape = eq.metadata["shape"]
    r, z = eq.lcfs.r, eq.lcfs.z
    a = 0.5 * (r.max() - r.min())
    r_geo = 0.5 * (r.max() + r.min())
    assert a == pytest.approx(shape["minor_radius"], rel=0.01)
    assert r_geo == pytest.approx(shape["major_radius"], rel=0.01)
    upper_factor = 1.1 if topology == "double_null" else 1.0
    lower_factor = 1.0 if topology == "limited" else 1.1
    kappa, delta = shape["elongation"], shape["triangularity"]
    assert z.max() / a == pytest.approx(upper_factor * kappa, rel=0.02)
    assert -z.min() / a == pytest.approx(lower_factor * kappa, rel=0.02)
    assert (r_geo - r[np.argmax(z)]) / a == pytest.approx(upper_factor * delta, abs=0.02)
    assert (r_geo - r[np.argmin(z)]) / a == pytest.approx(lower_factor * delta, abs=0.02)


def test_psi_satisfies_grad_shafranov_with_its_constant_sources(example):
    _, eq = example
    assert eq.convention.cocos == 11 and eq.convention.psi_per_radian is False
    two_pi = 2 * np.pi
    psi = eq.psi / two_pi                       # back to Wb/rad
    pprime = float(eq.pprime[0]) * two_pi       # Pa rad/Wb
    ffprime = float(eq.ffprime[0]) * two_pi
    assert np.ptp(eq.pprime) == 0 and np.ptp(eq.ffprime) == 0
    rm = eq.r[:, None] * np.ones_like(eq.z)[None, :]
    lhs = grad_shafranov_operator(psi, eq.r, eq.z)[2:-2, 2:-2]
    rhs = (-mu_0 * rm**2 * pprime - ffprime)[2:-2, 2:-2]
    assert np.max(np.abs(lhs - rhs)) < 2e-3 * np.max(np.abs(rhs))


def test_profiles_current_and_field_are_physical(example):
    _, eq = example
    assert eq.psi_boundary == 0.0
    assert eq.psi_axis < eq.psi_boundary                  # COCOS 11, Ip > 0: psi increases outward
    assert eq.ip == pytest.approx(eq.metadata["plasma_current_requested"], rel=1e-9)
    assert eq.ip > 0 and eq.bt0 > 0
    assert np.all(eq.pressure >= 0) and eq.pressure[0] > 0
    assert eq.pressure[-1] == pytest.approx(0.0, abs=1e-12)
    assert np.all(eq.f > 0)
    assert eq.f[-1] == pytest.approx(eq.bt0 * eq.r0)
    assert eq.metadata["source_type"] == "solovev"
    assert eq.limiter is not None
    # the axis is an O-point inside the LCFS
    from matplotlib.path import Path

    assert Path(eq.lcfs.points).contains_point(eq.magnetic_axis)


def test_solovev_example_rejects_nonsense():
    with pytest.raises(ValueError):
        solovev_example("upper_single_null")
    with pytest.raises(ValueError):
        solovev_example("limited", aspect_ratio=0.9)
    with pytest.raises(ValueError):
        solovev_example("limited", triangularity=1.0)


def test_solovev_example_convention_argument_round_trips():
    eq = solovev_example("limited", convention=1)
    assert eq.convention.cocos == 1
    eq11 = solovev_example("limited")
    np.testing.assert_allclose(eq.psi * 2 * np.pi, eq11.psi, rtol=0, atol=1e-12 * np.max(np.abs(eq11.psi)))


# --- miller_surfaces -----------------------------------------------------------------


def _measured_shape(surface: MillerSurface):
    theta = np.linspace(0, 2 * np.pi, 20001)
    r, z = evaluate_miller(surface, theta)
    a = 0.5 * (r.max() - r.min())
    kappa = (z.max() - z.min()) / (r.max() - r.min())
    delta = (0.5 * (r.max() + r.min()) - r[np.argmax(z)]) / a
    return a, kappa, delta


def test_miller_surfaces_scan_minor_radius_at_fixed_shape():
    surfaces = miller_surfaces([0.05, 0.1, 0.2], r0=0.4, kappa=1.7, delta=0.35)
    assert isinstance(surfaces, tuple) and len(surfaces) == 3
    for surface, radius in zip(surfaces, [0.05, 0.1, 0.2]):
        assert isinstance(surface, MillerSurface)
        a, kappa, delta = _measured_shape(surface)
        assert a == pytest.approx(radius, rel=1e-6)
        assert kappa == pytest.approx(1.7, rel=1e-6)
        assert delta == pytest.approx(0.35, abs=1e-6)


def test_miller_surfaces_broadcast_shape_sequences():
    kappas = [1.0, 1.5, 2.0]; deltas = [0.0, 0.3, 0.6]
    surfaces = miller_surfaces(0.25, r0=[0.40, 0.42, 0.44], z0=0.01, kappa=kappas, delta=deltas)
    assert [s.r0 for s in surfaces] == [0.40, 0.42, 0.44]
    for surface, k, d in zip(surfaces, kappas, deltas):
        a, kappa, delta = _measured_shape(surface)
        assert surface.z0 == 0.01
        assert kappa == pytest.approx(k, rel=1e-6)
        assert delta == pytest.approx(d, abs=1e-6)


def test_miller_surfaces_scalar_input_gives_one_surface_and_bad_input_raises():
    (only,) = miller_surfaces(0.2, r0=0.4)
    assert (only.kappa, only.delta, only.zeta) == (1.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        miller_surfaces([0.1, 0.2], r0=0.4, kappa=[1.5, 1.6, 1.7])
    with pytest.raises(ValueError):
        miller_surfaces(0.1, r0=0.4, delta=1.2)
    with pytest.raises(ValueError):
        miller_surfaces(-0.1, r0=0.4)
