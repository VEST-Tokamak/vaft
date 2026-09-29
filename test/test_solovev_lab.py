"""The VEST-like analytic Solov'ev baseline of the laboratory notebook (#883).

The notebook initializes a Solov'ev equilibrium from the packaged EFIT
boundary of VEST shot 39915 at 319 ms and scans its geometry and sources.
These tests pin the cheap, deterministic parts of that workflow so they do
not depend on executing the notebook, including the pressure scan and its
paramagnetic-diamagnetic zero crossing (#1198).
"""

from __future__ import annotations

import dataclasses
from functools import lru_cache

import numpy as np
import pytest

from vaft.data.eqdsk import from_equilibrium, to_omas
from vaft.data.equilibrium import Contour
from vaft.data.resources import sample_geqdsk
from vaft.formula.constants import MU0
from vaft.process.equilibrium import (
    calculate_q_profile_from_psi, contour_shape_parameters, convert_cocos, derive_global_descriptors, evaluate_solovev,
    fit_miller_sequence, fit_miller_surface, solovev_shape_constraints, solovev_to_equilibrium,
    solve_solovev_constraints,
)

# The notebook calibrates A to the reference beta_p and lands at 0.608; a
# rounded value keeps these tests free of the root find.
A_BASELINE = 0.6


@lru_cache(maxsize=None)
def _reference():
    reference = sample_geqdsk()
    shape = contour_shape_parameters(np.asarray(reference["RBBBS"]), np.asarray(reference["ZBBBS"]))
    target = dict(
        major_radius=0.5 * (shape["r_outboard"] + shape["r_inboard"]),
        minor_radius=0.5 * (shape["r_outboard"] - shape["r_inboard"]),
        elongation=shape["elongation"],
        triangularity=0.5 * (shape["triangularity_upper"] + shape["triangularity_lower"]),
    )
    wall = Contour(np.asarray(reference["RLIM"], float), np.asarray(reference["ZLIM"], float))
    return reference, target, wall


def _solve(a_parameter=A_BASELINE, ip=None, **overrides):
    reference, target, wall = _reference()
    shape = dict(target, **overrides)
    f_boundary = float(reference["BCENTR"] * reference["RCENTR"])
    ip = abs(float(reference["CURRENT"])) if ip is None else ip
    r = np.linspace(0.5 * wall.r.min(), 1.12 * wall.r.max(), 121)
    z = np.linspace(-1.15 * np.abs(wall.z).max(), 1.15 * np.abs(wall.z).max(), 211)
    constraints = solovev_shape_constraints(**shape)
    rref = shape["major_radius"]

    def solve(psi0):
        model = solve_solovev_constraints(
            constraints, basis="cerfon_freidberg_even", rref=rref, f_boundary=f_boundary,
            pprime=-psi0 * (1 - a_parameter) / (MU0 * rref**4), ffprime=-psi0 * a_parameter / rref**2)
        return model, solovev_to_equilibrium(model, r, z, limiter=wall)

    _, unit = solve(1.0)
    model, eq = solve(ip / unit.ip)
    psi_n = (eq.psi_1d - eq.psi_axis) / (eq.psi_boundary - eq.psi_axis)
    q = calculate_q_profile_from_psi(eq.psi, eq.r, eq.z, (eq.psi_1d, eq.f), eq.psi_axis, eq.psi_boundary,
                                     np.clip(psi_n, 0.01, 0.99), axis_rz=eq.magnetic_axis, cocos=eq.convention.cocos)
    return model, dataclasses.replace(eq, q=np.asarray(q, dtype=float))


@lru_cache(maxsize=None)
def _baseline():
    return _solve()


def test_baseline_solve_is_full_rank_well_conditioned_and_closed():
    reference, _, _ = _reference()
    model, eq = _baseline()
    assert model.rank == 7
    assert model.metadata["condition_number"] < 1e5
    assert model.residual_norm < 1e-10
    assert eq.lcfs.closed
    assert np.all(np.isfinite(eq.lcfs.r)) and np.all(np.isfinite(eq.lcfs.z))
    assert eq.ip == pytest.approx(abs(float(reference["CURRENT"])), rel=1e-6)


def test_baseline_descriptors_are_finite_and_match_the_target_shape():
    _, target, _ = _reference()
    _, eq = _baseline()
    desc = derive_global_descriptors(eq)
    for name in ("volume", "cross_section_area", "q0", "q95", "beta_p_boundary_average", "li_virial",
                 "shafranov_shift"):
        assert desc[name].available and np.isfinite(desc[name].value), name
    assert desc["volume"].value > 0 and desc["cross_section_area"].value > 0
    assert desc["elongation"].value == pytest.approx(target["elongation"], rel=5e-3)
    assert desc["major_radius"].value == pytest.approx(target["major_radius"], rel=5e-3)


def test_cocos1_export_round_trip_keeps_the_flux_span():
    """Guard for the #1292 workaround: a per-radian record reaches the ODS in weber once."""
    _, eq = _baseline()
    ods = to_omas(from_equilibrium(convert_cocos(eq, 1)))
    gq = ods["equilibrium.time_slice.0.global_quantities"]
    assert gq["psi_boundary"] - gq["psi_axis"] == pytest.approx(eq.psi_boundary - eq.psi_axis, rel=1e-6)
    assert gq["ip"] == pytest.approx(eq.ip, rel=1e-6)


@pytest.mark.xfail(strict=True, reason="#1292: from_equilibrium writes full-weber flux into a per-radian g-file")
def test_direct_cocos11_export_keeps_the_flux_span():
    _, eq = _baseline()
    ods = to_omas(from_equilibrium(eq))
    gq = ods["equilibrium.time_slice.0.global_quantities"]
    assert gq["psi_boundary"] - gq["psi_axis"] == pytest.approx(eq.psi_boundary - eq.psi_axis, rel=1e-6)


def test_miller_fits_accept_the_interior_and_reach_the_target_at_the_edge():
    _, target, _ = _reference()
    _, eq = _baseline()
    sequence = fit_miller_sequence(eq, (0.2, 0.4, 0.6, 0.8, 0.95))
    assert all(fit.accepted for fit in sequence.fits)
    edge = sequence.fits[-1].surface
    assert edge.kappa == pytest.approx(target["elongation"], rel=0.02)
    assert edge.delta == pytest.approx(target["triangularity"], abs=0.03)
    lcfs = fit_miller_surface(eq.lcfs)
    assert lcfs.accepted
    assert lcfs.surface.kappa == pytest.approx(target["elongation"], rel=0.01)
    assert lcfs.surface.delta == pytest.approx(target["triangularity"], abs=0.01)


def test_elongation_scan_raises_volume_and_q95():
    _, target, _ = _reference()
    volumes, q95 = [], []
    for kappa in (1.0, target["elongation"], 1.9):
        _, eq = _solve(elongation=kappa)
        desc = derive_global_descriptors(eq)
        volumes.append(desc["volume"].value)
        q95.append(desc["q95"].value)
    assert np.all(np.diff(volumes) > 0)
    assert np.all(np.diff(q95) > 0)


def test_source_scan_leaves_the_shape_fixed_and_lowers_beta_p_with_a():
    volumes, beta_p = [], []
    for a_parameter in (0.0, A_BASELINE, 1.0):
        _, eq = _solve(a_parameter=a_parameter)
        desc = derive_global_descriptors(eq)
        volumes.append(desc["volume"].value)
        beta_p.append(desc["beta_p_boundary_average"].value)
    np.testing.assert_allclose(volumes, volumes[1], rtol=1e-3)
    assert np.all(np.diff(beta_p) < 0)


# --- Pressure scan and the paramagnetic-diamagnetic zero crossing (#1198) ---------------------------------------
#
# The notebook raises p' on the fixed baseline boundary at fixed F_boundary and fixed |Ip| (FF' adjusted) and
# root-finds where the LCFS-bounded toroidal-flux perturbation changes sign. These tests re-implement its small
# helpers on the coarse test grid. The canonical calculate_reconstructed_diamagnetic_flux is deliberately not used
# (#871: its psi_N-only mask admits the open-field region).

def _grid():
    _, _, wall = _reference()
    r = np.linspace(0.5 * wall.r.min(), 1.12 * wall.r.max(), 121)
    z = np.linspace(-1.15 * np.abs(wall.z).max(), 1.15 * np.abs(wall.z).max(), 211)
    return r, z


def _solve_sources(pprime, ffprime, r=None, z=None, bt_sign=1.0):
    reference, target, wall = _reference()
    r0, z0 = _grid()
    model = solve_solovev_constraints(
        solovev_shape_constraints(**target), basis="cerfon_freidberg_even", rref=target["major_radius"],
        f_boundary=bt_sign * float(reference["BCENTR"] * reference["RCENTR"]),  # F's sign follows (#1307)
        pprime=pprime, ffprime=ffprime)
    return model, solovev_to_equilibrium(model, r0 if r is None else r, z0 if z is None else z, limiter=wall)


@lru_cache(maxsize=None)
def _baseline_sources():
    model, _ = _baseline()
    return model.pprime, model.ffprime


def _solve_fixed_ip(lam, r=None, z=None):
    """p' = lam * p'_0 with FF' adjusted by Newton steps until |Ip| is the reference current."""
    reference, _, _ = _reference()
    ip = abs(float(reference["CURRENT"]))
    pp0, ff0 = _baseline_sources()
    slope = _solve_sources(0.0, ff0, r, z)[1].ip / ff0
    ffprime = ff0
    for _ in range(8):
        model, eq = _solve_sources(lam * pp0, ffprime, r, z)
        if abs(eq.ip - ip) <= 1e-9 * ip:
            break
        ffprime += (ip - eq.ip) / slope
    return model, eq


def _delta_phi_tor(model, eq, r=None, z=None):
    """LCFS-bounded sign(B_phi) * integral (F/R - F_b/R) dA [Wb], paramagnetic-positive (#1196)."""
    from matplotlib.path import Path

    r = eq.r if r is None else r
    z = eq.z if z is None else z
    rm, zm = np.meshgrid(r, z, indexing="ij")
    inside = Path(np.column_stack([eq.lcfs.r, eq.lcfs.z])).contains_points(
        np.column_stack([rm.ravel(), zm.ravel()])).reshape(rm.shape)
    psi_axis = float(evaluate_solovev(model, *eq.magnetic_axis)["psi"])
    psi_n = np.clip((evaluate_solovev(model, rm, zm)["psi"] - psi_axis) / (model.psi_boundary - psi_axis), 0.0, 1.0)
    psi = psi_axis + psi_n * (model.psi_boundary - psi_axis)
    sign = np.sign(model.f_boundary)
    f = sign * np.sqrt(model.f_boundary**2 + 2 * model.ffprime * (psi - model.psi_boundary))
    da = np.gradient(r)[:, None] * np.gradient(z)[None, :]
    return float(sign * np.sum(np.where(inside, (f - model.f_boundary) / rm, 0.0) * da))


def test_baseline_field_is_positive_and_the_baseline_is_paramagnetic():
    reference, _, _ = _reference()
    assert float(reference["BCENTR"]) > 0  # VEST's positive toroidal field: sign(B_phi) = +1
    model, eq = _solve_fixed_ip(1.0)
    assert _delta_phi_tor(model, eq) > 0
    assert eq.f[0] > eq.f[-1]


def test_fixed_ip_scan_holds_the_current():
    reference, _, _ = _reference()
    ip = abs(float(reference["CURRENT"]))
    for lam in (0.0, 1.0, 2.5, 3.4):
        _, eq = _solve_fixed_ip(lam)
        assert eq.ip == pytest.approx(ip, rel=1e-8), lam


def test_sign_convention_on_clearly_paramagnetic_and_diamagnetic_points():
    low_model, low = _solve_fixed_ip(0.5)
    # lambda_p = 3.4: on this coarse grid j_phi first reverses near 3.43 (3.52 on the notebook's finer grid).
    high_model, high = _solve_fixed_ip(3.4)
    # Paramagnetic: B_phi raised inside the plasma, F_axis > F_boundary, Delta Phi_tor > 0.
    assert _delta_phi_tor(low_model, low) > 5e-4 and low.f[0] > low.f[-1]
    # Diamagnetic: the reverse, and still a valid state (single-signed j_phi).
    assert _delta_phi_tor(high_model, high) < -1e-4 and high.f[0] < high.f[-1]
    from matplotlib.path import Path

    rm, zm = np.meshgrid(high.r, high.z, indexing="ij")
    inside = Path(np.column_stack([high.lcfs.r, high.lcfs.z])).contains_points(
        np.column_stack([rm.ravel(), zm.ravel()])).reshape(rm.shape)
    j_phi = evaluate_solovev(high_model, rm, zm)["j_phi"][inside]
    assert j_phi.min() * j_phi.max() > 0


def test_delta_phi_tor_keeps_its_paramagnetic_positive_meaning_when_the_field_is_reversed():
    pp0, ff0 = _baseline_sources()
    forward_model, forward = _solve_sources(pp0, ff0)
    reversed_model, reversed_eq = _solve_sources(pp0, ff0, bt_sign=-1.0)
    assert reversed_model.f_boundary < 0 and reversed_model.f_sign == -1
    # Negating F_boundary alone reverses both F and bt0 (#1307): the record agrees with itself.
    assert reversed_eq.bt0 < 0 and np.all(reversed_eq.f < 0)
    # psi depends on FF' only, so the reversed baseline is the same plasma with B_phi -> -B_phi ...
    assert _delta_phi_tor(reversed_model, reversed_eq) == pytest.approx(_delta_phi_tor(forward_model, forward), rel=1e-9)
    assert _delta_phi_tor(reversed_model, reversed_eq) > 0
    # ... while F itself changes sign and |F| still rises toward the axis: paramagnetic in either orientation.
    assert reversed_eq.f[0] < 0 and forward.f[0] > 0
    assert abs(reversed_eq.f[0]) > abs(reversed_eq.f[-1])


def test_zero_crossing_is_bracketed_and_sits_where_ffprime_vanishes():
    from scipy.optimize import brentq

    lo, hi = 3.0, 3.5
    dphi = lambda lam: _delta_phi_tor(*_solve_fixed_ip(lam))  # noqa: E731
    assert dphi(lo) > 0 > dphi(hi)
    lam_zero = brentq(dphi, lo, hi, xtol=1e-6)
    model, eq = _solve_fixed_ip(lam_zero)
    _, ff0 = _baseline_sources()
    assert abs(model.ffprime) < 1e-4 * abs(ff0)
    beta_p_zero = derive_global_descriptors(eq, rational_q=())["beta_p_boundary_average"].value
    # The root confirms the identity Delta Phi_tor = 0 <=> FF' = 0 (checked above); its beta_p is that of the
    # pure-p' state, which the low-aspect-ratio VEST-like geometry puts clearly above the large-aspect-ratio 1.
    assert 1.08 < beta_p_zero < 1.16


def test_direct_pprime_scan_at_fixed_ffprime_stays_paramagnetic_while_ip_drifts():
    pp0, ff0 = _baseline_sources()
    values, currents = [], []
    for lam in (0.5, 1.0, 3.0):
        model, eq = _solve_sources(lam * pp0, ff0)
        values.append(_delta_phi_tor(model, eq))
        currents.append(eq.ip)
    assert np.all(np.array(values) > 0) and np.all(np.diff(values) > 0)
    assert np.all(np.diff(currents) > 0)


def test_lcfs_bounded_integral_ignores_grid_extension_beyond_the_lcfs():
    """The #871 failure mode: extending the domain must not change an LCFS-bounded integral."""
    r, z = _grid()
    dr, dz = r[1] - r[0], z[1] - z[0]
    r_ext = np.r_[r, r[-1] + dr * np.arange(1, int((2.0 - r[-1]) / dr))]
    pad = z[-1] + dz * np.arange(1, int((3.0 - z[-1]) / dz))
    z_ext = np.r_[-pad[::-1], z, pad]
    for lam in (1.0, 3.5):
        model, eq = _solve_fixed_ip(lam)
        if lam > 3:  # at high pressure the open-field region holds psi values between axis and boundary ...
            rm, zm = np.meshgrid(r_ext, z_ext, indexing="ij")
            psi_axis = float(evaluate_solovev(model, *eq.magnetic_axis)["psi"])
            psi_n = (evaluate_solovev(model, rm, zm)["psi"] - psi_axis) / (model.psi_boundary - psi_axis)
            area_psi_n_mask = np.count_nonzero((psi_n >= 0) & (psi_n <= 1)) * dr * dz
            assert area_psi_n_mask > 1.1 * derive_global_descriptors(eq, rational_q=())["cross_section_area"].value
        # ... and the LCFS-bounded integral is unchanged by it.
        assert _delta_phi_tor(model, eq, r_ext, z_ext) == pytest.approx(_delta_phi_tor(model, eq), rel=1e-9, abs=1e-12)
