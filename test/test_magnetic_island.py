"""The straight-field-line map and the analytic island placed on it (#886).

The reference equilibrium is the classic Solov'ev solution
``psi = R^2 Z^2 / kappa^2 + (R^2 - R0^2)^2 / 4`` with a toroidal field strong
enough that ``q = 2`` lies inside the plasma, so every claim below is checked
against an equilibrium whose surfaces are known in closed form.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from vaft.data.equilibrium import EquilibriumConvention, SolovevConstraint
from vaft.formula.constants import MU0
from vaft.process.equilibrium import (
    calculate_q_profile_from_psi,
    make_equilibrium_field_interpolator,
    solovev_to_equilibrium,
    solve_solovev_constraints,
    straight_field_line_angle_on_grid,
    straight_field_line_map,
)
from vaft.process.magnetic_island import (
    MagneticIslandSpec,
    equilibrium_safety_factor,
    island_emissivity,
    magnetic_island_topology,
    resolve_rational_surface,
)

W = 0.03


def _solovev(psi_boundary=0.08, kappa=1.4, f_boundary=2.2):
    pprime = -8.0 * ((1.0 + 1.0 / kappa**2) / 4.0) / MU0
    r_out = np.sqrt(1.0 + 2.0 * np.sqrt(psi_boundary))
    r_in = np.sqrt(1.0 - 2.0 * np.sqrt(psi_boundary))
    z_top = kappa * np.sqrt(psi_boundary)
    constraints = [
        SolovevConstraint(r_out, 0.0, "psi", psi_boundary),
        SolovevConstraint(r_in, 0.0, "psi", psi_boundary),
        SolovevConstraint(1.0, z_top, "psi", psi_boundary),
        SolovevConstraint(1.0, 0.0, "psi", 0.0),
        SolovevConstraint(1.0, 0.0, "dpsi_dr", 0.0),
    ]
    return solve_solovev_constraints(constraints, pprime=pprime, ffprime=0.0, rref=1.0,
                                     psi_boundary=psi_boundary, f_boundary=f_boundary)


def _equilibrium(n=129, **kw):
    z_half = 0.65 if kw.get("kappa", 1.4) > 1.5 else 0.55
    return solovev_to_equilibrium(_solovev(**kw), np.linspace(0.55, 1.4, n),
                                  np.linspace(-z_half, z_half, n))


@pytest.fixture(scope="module")
def eq():
    return _equilibrium()


@pytest.fixture(scope="module")
def sfl(eq):
    return straight_field_line_map(eq.psi, eq.r, eq.z, eq.psi_axis, eq.psi_boundary, eq.magnetic_axis)


def _trace(eq, f, r0, z0, turns=2.0):
    field = make_equilibrium_field_interpolator(eq.r, eq.z, eq.psi, eq.psi_1d, f, cocos=11)

    def rhs(_phi, y):
        b_r, b_z, b_phi = field(y[0], y[1])
        return [y[0] * b_r / b_phi, y[0] * b_z / b_phi]

    phi = np.linspace(0.0, 2.0 * np.pi * turns, 400)
    sol = solve_ivp(rhs, (phi[0], phi[-1]), [r0, z0], t_eval=phi, rtol=1e-10, atol=1e-12,
                    max_step=0.02)
    return phi, sol.y[0], sol.y[1]


def _outboard_width(topology):
    sfl = topology.sfl_map
    z_axis = sfl.magnetic_axis[1]
    r_s = float(sfl.outboard_radius(topology.psi_n_s))
    w = topology.spec.width

    def omega_minus_one(r):  # xi = 0 on the outboard midplane for this island
        x = sfl.outboard_radius(sfl.psi_norm(r, z_axis)) - r_s
        return 8.0 * (x / w) ** 2 - 1.0 - 1.0

    return brentq(omega_minus_one, r_s, r_s + 3 * w) - brentq(omega_minus_one, r_s - 3 * w, r_s)


# --- straight-field-line angle -------------------------------------------------


def test_pest_weights_reproduce_the_contour_safety_factor(eq, sfl):
    levels = np.array([0.3, 0.5, 0.7, 0.9])
    q_contour = calculate_q_profile_from_psi(eq.psi, eq.r, eq.z, (eq.psi_1d, eq.f), eq.psi_axis,
                                             eq.psi_boundary, levels, cocos=11,
                                             axis_rz=eq.magnetic_axis)
    span = (eq.psi_boundary - eq.psi_axis) / (2.0 * np.pi)  # per radian
    for level, q_ref in zip(levels, q_contour):
        weight = sfl.surface(level)["weight"]
        f = np.interp(eq.psi_axis + level * (eq.psi_boundary - eq.psi_axis), eq.psi_1d, eq.f)
        q_rays = abs(f) / (2.0 * np.pi) * np.mean(weight) * 2.0 * np.pi / abs(span)
        assert q_rays == pytest.approx(abs(q_ref), rel=2e-3)


def test_field_lines_are_straight_in_theta_star(eq, sfl):
    surface = sfl.surface(0.5)
    phi, r, z = _trace(eq, eq.f, surface["r"][0], surface["z"][0])
    theta_star = np.unwrap(sfl.theta_star(r, z))
    fit = np.polyfit(phi, theta_star, 1)
    assert np.max(np.abs(theta_star - np.polyval(fit, phi))) < 1e-5
    theta_geo = np.unwrap(sfl.geometric_angle(r, z))
    geo_fit = np.polyfit(phi, theta_geo, 1)
    assert np.max(np.abs(theta_geo - np.polyval(geo_fit, phi))) > 0.1  # the lab angle is not
    q_half = np.interp(0.5, *equilibrium_safety_factor(eq))
    assert abs(fit[0]) == pytest.approx(1.0 / q_half, rel=2e-3)


def test_theta_star_origin_direction_and_symmetry(eq, sfl):
    ra, za = sfl.magnetic_axis
    surface = sfl.surface(0.6)
    assert sfl.theta_star(surface["r"][0], surface["z"][0]) == pytest.approx(0.0, abs=1e-9)
    top = int(np.argmax(surface["z"]))
    assert 0.0 < surface["theta_star"][top] < np.pi
    r_pts = surface["r"][1:40]
    z_pts = surface["z"][1:40]
    up = sfl.theta_star(r_pts, z_pts)
    down = sfl.theta_star(r_pts, 2 * za - z_pts)
    np.testing.assert_allclose(up + down, 2.0 * np.pi, atol=1e-6)


def test_theta_star_on_grid_is_nan_outside_the_boundary(eq):
    theta = straight_field_line_angle_on_grid(eq.psi, eq.r, eq.z, eq.psi_axis, eq.psi_boundary,
                                              eq.magnetic_axis)
    psi_n = (eq.psi - eq.psi_axis) / (eq.psi_boundary - eq.psi_axis)
    assert theta.shape == eq.psi.shape
    assert np.all(np.isnan(theta[psi_n > 1.02]))
    inside = psi_n < 0.98
    assert np.all((theta[inside] >= 0.0) & (theta[inside] < 2.0 * np.pi))


def test_a_boundary_off_the_grid_is_refused(eq):
    keep = eq.r < 1.2
    with pytest.raises(ValueError, match="grid edge"):
        straight_field_line_map(eq.psi[keep], eq.r[keep], eq.z, eq.psi_axis, eq.psi_boundary,
                                eq.magnetic_axis)


# --- rational surface ---------------------------------------------------------


def test_rational_surface_is_the_equilibrium_q_equal_m_over_n(eq, sfl):
    psi_n_s, q_s = resolve_rational_surface(eq, 2, 1)
    assert q_s == pytest.approx(2.0, abs=1e-9)
    # Independently of the profile it was solved on: q from the PEST weights.
    weight = sfl.surface(psi_n_s)["weight"]
    f = np.interp(eq.psi_axis + psi_n_s * (eq.psi_boundary - eq.psi_axis), eq.psi_1d, eq.f)
    span = (eq.psi_boundary - eq.psi_axis) / (2.0 * np.pi)
    assert abs(f) * np.mean(weight) / abs(span) == pytest.approx(2.0, rel=1e-3)


def test_a_missing_rational_surface_fails_explicitly(eq):
    with pytest.raises(ValueError, match="does not occur"):
        resolve_rational_surface(eq, 5, 1)
    with pytest.raises(ValueError, match="does not occur"):
        magnetic_island_topology(eq, MagneticIslandSpec(5, 1, W))


def test_spec_rejects_meaningless_islands():
    with pytest.raises(ValueError):
        MagneticIslandSpec(0, 1, W)
    with pytest.raises(ValueError, match="helicity"):
        MagneticIslandSpec(2, -1, W)
    with pytest.raises(ValueError):
        MagneticIslandSpec(2, 1, -W)
    with pytest.raises(ValueError):
        MagneticIslandSpec(2, 1, W, psi_n_s=1.2)


# --- island topology -----------------------------------------------------------


def test_outboard_full_width_is_the_prescribed_width(eq):
    topology = magnetic_island_topology(eq, MagneticIslandSpec(2, 1, W))
    assert _outboard_width(topology) == pytest.approx(W, abs=1e-6)
    wide = magnetic_island_topology(eq, MagneticIslandSpec(2, 1, 1.5 * W), sfl_map=topology.sfl_map)
    assert _outboard_width(wide) == pytest.approx(1.5 * W, abs=1e-6)


def test_o_and_x_points_sit_at_the_documented_phases(eq):
    topology = magnetic_island_topology(eq, MagneticIslandSpec(2, 1, W))
    sfl = topology.sfl_map
    assert topology.o_points.shape == (2, 2)
    o_theta = np.sort(sfl.theta_star(*topology.o_points.T))
    x_theta = np.sort(sfl.theta_star(*topology.x_points.T))
    np.testing.assert_allclose(o_theta, [0.0, np.pi], atol=2e-3)
    np.testing.assert_allclose(x_theta, [np.pi / 2, 3 * np.pi / 2], atol=2e-3)
    # The helical flux is -1 at an O-point and +1 at an X-point: on the
    # resonant surface x = 0, so Omega = -cos(xi) there.
    r_s = float(sfl.outboard_radius(topology.psi_n_s))
    for points, expected in ((topology.o_points, -1.0), (topology.x_points, 1.0)):
        x = sfl.outboard_radius(sfl.psi_norm(*points.T)) - r_s
        np.testing.assert_allclose(x, 0.0, atol=1e-5)
        xi = 2 * sfl.theta_star(*points.T)
        np.testing.assert_allclose(8 * (x / W) ** 2 - np.cos(xi), expected, atol=1e-4)
    assert np.nanmin(topology.helical_flux) == pytest.approx(-1.0, abs=0.1)
    # The island region is where the two branches enclose it.
    assert topology.inside_separatrix.any()
    assert not topology.inside_separatrix[~topology.inside_boundary].any()


def test_phase_translates_the_island_without_resizing_it(eq):
    base = magnetic_island_topology(eq, MagneticIslandSpec(2, 1, W))
    turned = magnetic_island_topology(eq, MagneticIslandSpec(2, 1, W, phase=np.pi / 2),
                                      sfl_map=base.sfl_map)
    o_theta = np.sort(base.sfl_map.theta_star(*turned.o_points.T))
    np.testing.assert_allclose(o_theta, [np.pi / 4, 5 * np.pi / 4], atol=2e-3)
    assert turned.width_psi_n == base.width_psi_n
    assert _outboard_width(turned) == pytest.approx(W, abs=1e-6)


def test_one_helical_period_returns_the_same_topology(eq):
    omega = 2.0 * np.pi * 5e3
    topology = magnetic_island_topology(eq, MagneticIslandSpec(2, 1, W, angular_frequency=omega))
    later = topology.rephase(time=2.0 * np.pi / omega)
    np.testing.assert_allclose(later.helical_flux, topology.helical_flux, atol=1e-9, equal_nan=True)
    half = topology.rephase(time=np.pi / omega)
    assert not np.allclose(half.helical_flux, topology.helical_flux, equal_nan=True)


@pytest.mark.parametrize("f_sign", [1.0, -1.0])
def test_helical_phase_is_constant_along_resonant_field_lines(eq, f_sign):
    """The helicity comes from the field: reversing B_phi reverses it, and the
    phase stays resonant either way."""
    flipped = dataclasses.replace(eq, f=f_sign * eq.f)
    topology = magnetic_island_topology(flipped, MagneticIslandSpec(2, 1, W))
    sfl = topology.sfl_map
    surface = sfl.surface(topology.psi_n_s)
    phi, r, z = _trace(flipped, flipped.f, surface["r"][0], surface["z"][0], turns=4.0)
    xi = 2 * sfl.theta_star(r, z) - topology.helicity * 1 * phi
    drift = np.angle(np.exp(1j * (xi - xi[0])))
    assert np.max(np.abs(drift)) < 1e-3
    expected = -1 if f_sign > 0 else 1  # dpsi/dR > 0 outboard, COCOS 11
    assert topology.helicity == expected


def test_an_unidentified_convention_needs_an_explicit_helicity(eq):
    anonymous = dataclasses.replace(eq, convention=EquilibriumConvention(
        cocos=None, candidates=(1, 3), psi_per_radian=False))
    with pytest.raises(ValueError, match="helicity"):
        magnetic_island_topology(anonymous, MagneticIslandSpec(2, 1, W))
    topology = magnetic_island_topology(anonymous, MagneticIslandSpec(2, 1, W, helicity=1))
    assert topology.helicity == 1


def test_topology_converges_with_the_grid():
    results = [magnetic_island_topology(_equilibrium(n), MagneticIslandSpec(2, 1, W))
               for n in (65, 129, 257)]
    psi_s = [t.psi_n_s for t in results]
    widths = [t.width_psi_n for t in results]
    o_r = [np.max(t.o_points[:, 0]) for t in results]
    assert abs(psi_s[2] - psi_s[1]) < abs(psi_s[1] - psi_s[0]) + 1e-6
    assert abs(psi_s[2] - psi_s[1]) < 5e-4
    assert widths[2] == pytest.approx(widths[1], rel=1e-3)
    assert o_r[2] == pytest.approx(o_r[1], abs=1e-4)
    for topology in results:
        assert _outboard_width(topology) == pytest.approx(W, abs=1e-6)


def test_the_same_spec_follows_each_equilibrium():
    """No plasma shape parameter enters: a different equilibrium moves the
    resonant surface and changes the flux width, never the outboard width."""
    round_eq = _equilibrium(kappa=1.4, f_boundary=2.2)
    tall_eq = _equilibrium(kappa=1.8, f_boundary=1.9)
    spec = MagneticIslandSpec(2, 1, W)
    round_top = magnetic_island_topology(round_eq, spec)
    tall_top = magnetic_island_topology(tall_eq, spec)
    assert abs(round_top.psi_n_s - tall_top.psi_n_s) > 0.02
    assert abs(round_top.width_psi_n - tall_top.width_psi_n) > 1e-3
    for topology in (round_top, tall_top):
        assert topology.q_s == pytest.approx(2.0, abs=1e-9)
        assert _outboard_width(topology) == pytest.approx(W, abs=1e-6)


def test_island_follows_flux_expansion_away_from_the_outboard_midplane(eq):
    """Uniform in flux: where the surfaces spread, the geometric width grows."""
    topology = magnetic_island_topology(eq, MagneticIslandSpec(2, 1, W, phase=np.pi))
    sfl = topology.sfl_map
    # phase = pi puts an O-point at theta* = pi/2 (m = 2), near the top.
    top = topology.o_points[np.argmax(topology.o_points[:, 1])]
    ra, za = sfl.magnetic_axis
    direction = np.array([top[0] - ra, top[1] - za]) / np.hypot(top[0] - ra, top[1] - za)
    r_s = float(sfl.outboard_radius(topology.psi_n_s))

    def omega_minus_one(s):
        r, z = top + s * direction
        x = sfl.outboard_radius(sfl.psi_norm(r, z)) - r_s
        return 8.0 * (x / W) ** 2 - 1.0 - 1.0

    along_ray = brentq(omega_minus_one, 0.0, 0.2) - brentq(omega_minus_one, -0.2, 0.0)
    grad_out = sfl.grad_psi(r_s, za)
    grad_top = sfl.grad_psi(*top)
    # |grad psi| is smaller at the top, so the island is wider there.
    assert grad_top < grad_out
    assert along_ray > W * 1.05


# --- emissivity -------------------------------------------------------------------


def _profile(psi_n):
    return (1.0 - psi_n) ** 2


def test_zero_amplitude_returns_the_axisymmetric_background(eq):
    topology = magnetic_island_topology(eq, MagneticIslandSpec(2, 1, W))
    eps, delta = island_emissivity(topology, profile=_profile, amplitude=0.0)
    assert np.all(delta == 0.0)
    inside = topology.inside_boundary
    np.testing.assert_allclose(eps[inside], _profile(np.clip(topology.psi_n[inside], 0, 1)))
    assert np.all(eps[~inside] == 0.0)


def test_full_flattening_takes_the_resonant_value_inside_the_island(eq):
    topology = magnetic_island_topology(eq, MagneticIslandSpec(2, 1, 2 * W))
    eps, delta = island_emissivity(topology, profile=_profile, amplitude=1.0, hard_mask=True)
    inside = topology.inside_separatrix
    np.testing.assert_allclose(eps[inside], _profile(topology.psi_n_s))
    assert np.all(delta[~inside] == 0.0)
    # Flattening lowers the emissivity inward of the surface and raises it outward.
    inner = inside & (topology.normal_displacement < 0)
    outer = inside & (topology.normal_displacement > 0)
    assert np.all(delta[inner] <= 0.0) and np.all(delta[outer] >= 0.0)


def test_island_only_model_and_tabulated_profiles(eq):
    topology = magnetic_island_topology(eq, MagneticIslandSpec(2, 1, W))
    eps, delta = island_emissivity(topology, model="island", amplitude=3.0)
    np.testing.assert_array_equal(eps, delta)
    assert np.nanmax(delta) == pytest.approx(3.0, rel=1e-3)
    grid = np.linspace(0.0, 1.0, 201)
    tab, _ = island_emissivity(topology, profile=(grid, _profile(grid)), amplitude=0.5)
    fun, _ = island_emissivity(topology, profile=_profile, amplitude=0.5)
    np.testing.assert_allclose(tab, fun, atol=2e-5)


def test_emissivity_rejects_bad_models(eq):
    topology = magnetic_island_topology(eq, MagneticIslandSpec(2, 1, W))
    with pytest.raises(ValueError, match="model"):
        island_emissivity(topology, profile=_profile, model="gaussian")
    with pytest.raises(ValueError, match="profile"):
        island_emissivity(topology)
    with pytest.raises(ValueError, match="smoothing"):
        island_emissivity(topology, profile=_profile, smoothing=0.0)
