"""The VEST-like analytic Solov'ev baseline of the laboratory notebook (#883).

The notebook initializes a Solov'ev equilibrium from the packaged EFIT
boundary of VEST shot 39915 at 319 ms and scans its geometry and sources.
These tests pin the cheap, deterministic parts of that workflow so they do
not depend on executing the notebook.
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
    calculate_q_profile_from_psi, contour_shape_parameters, convert_cocos, derive_global_descriptors,
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
