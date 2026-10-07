"""Core q-profile, rational-surface, low-shear and boundary context (#1798).

Synthetic profiles with known answers for each case the issue lists, plus the
packaged 39915 slice (a VEST spherical tokamak: q > 1 everywhere, limited).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from vaft.process.core_q_context import core_q_context, core_q_context_from_profiles, low_shear_regions

PSI = np.linspace(0.0, 1.0, 401)


def _monotonic(q0=0.8, qa=4.0):
    return q0 + (qa - q0) * PSI


def _reversed(q_min, psi_min=0.3, q0=None, qa=5.0):
    q0 = q_min + 1.0 if q0 is None else q0
    inner = q_min + (q0 - q_min) * ((psi_min - PSI) / psi_min) ** 2
    outer = q_min + (qa - q_min) * ((PSI - psi_min) / (1 - psi_min)) ** 2
    return np.where(PSI < psi_min, inner, outer)


def test_monotonic_q0_below_one_has_one_q1_surface():
    ctx = core_q_context_from_profiles(PSI, _monotonic())
    assert ctx.q_profile_topology == "monotonic" and ctx.q_min_on_axis
    assert ctx.q_axis == pytest.approx(0.8) and ctx.q_min == ctx.q_axis
    assert ctx.q1_surface_count == 1
    (q1,) = ctx.q1_surfaces
    assert q1.psi_n == pytest.approx(0.2 / 3.2, abs=1e-6) and q1.shear > 0
    assert ctx.double_resonant_pairs == ()


def test_q_above_one_everywhere_has_no_q1_surface_and_a_positive_distance():
    ctx = core_q_context_from_profiles(PSI, _monotonic(q0=1.2))
    assert ctx.q1_surface_count == 0
    prox = {(p.m, p.n): p for p in ctx.rational_proximity}
    assert prox[(1, 1)].delta_q == pytest.approx(0.2) and prox[(1, 1)].crossing_count == 0
    assert not prox[(1, 1)].tangent_candidate


def test_reversed_shear_with_qmin_above_one_is_off_axis_without_q1():
    ctx = core_q_context_from_profiles(PSI, _reversed(1.05))
    assert ctx.q_profile_topology == "reversed_shear" and not ctx.q_min_on_axis
    assert ctx.psi_n_at_q_min == pytest.approx(0.3, abs=1e-3)
    assert ctx.q1_surface_count == 0 and len(ctx.shear_sign_changes_rho) == 1
    # A near-1 low-shear core exists around q_min, named as a region, not a mode.
    near = [r for r in ctx.near_rational_low_shear_regions if (r.m, r.n) == (1, 1)]
    assert near and near[0].rho_start < ctx.rho_at_q_min < near[0].rho_end


def test_reversed_shear_crossing_q1_twice_gives_a_double_resonant_pair_with_opposite_shear():
    ctx = core_q_context_from_profiles(PSI, _reversed(0.9))
    assert ctx.q1_surface_count == 2
    inner, outer = ctx.q1_surfaces
    assert inner.psi_n < 0.3 < outer.psi_n
    assert inner.shear < 0 < outer.shear  # signs kept, never collapsed to |s|
    pairs = [p for p in ctx.double_resonant_pairs if (p.m, p.n) == (1, 1)]
    assert len(pairs) == 1 and pairs[0].delta_rho > 0


def test_multiple_same_helicity_pairs_are_all_kept():
    ctx = core_q_context_from_profiles(PSI, _reversed(1.4, q0=4.0, qa=6.0), rationals=((3, 2), (2, 1), (5, 2)))
    found = {(p.m, p.n) for p in ctx.double_resonant_pairs}
    assert found == {(3, 2), (2, 1), (5, 2)}


def test_a_tangency_between_samples_is_flagged_not_placed():
    grid = np.linspace(0.0, 1.0, 40)  # psi_min = 0.3 is not a node
    q = np.where(grid < 0.3, 1.0 + 4 * (0.3 - grid) ** 2, 1.0 + 6 * (grid - 0.3) ** 2) + 2e-4
    ctx = core_q_context_from_profiles(grid, q)
    prox = {(p.m, p.n): p for p in ctx.rational_proximity}[(1, 1)]
    assert prox.crossing_count == 0 and prox.tangent_candidate


def test_the_sign_of_q_never_changes_the_physics():
    plus = core_q_context_from_profiles(PSI, _reversed(0.9))
    minus = core_q_context_from_profiles(PSI, -_reversed(0.9))
    assert (plus.q_sign, minus.q_sign) == (1, -1)
    assert minus.q_min == plus.q_min and minus.q1_surface_count == plus.q1_surface_count == 2
    assert [c.shear for c in minus.q1_surfaces] == pytest.approx([c.shear for c in plus.q1_surfaces])


def test_low_shear_threshold_is_explicit_and_scannable():
    q = _reversed(1.05)
    narrow, wide = (core_q_context_from_profiles(PSI, q, shear_thresholds=(t,)) for t in (0.05, 0.3))
    width = lambda ctx: max(r.width_rho for r in ctx.low_shear_regions if r.contains_q_min)
    assert width(narrow) < width(wide)
    assert {r.threshold for r in narrow.low_shear_regions} == {0.05}
    both = core_q_context_from_profiles(PSI, q, shear_thresholds=(0.05, 0.3))
    assert {r.threshold for r in both.low_shear_regions} == {0.05, 0.3}
    with pytest.raises(ValueError, match="positive"):
        core_q_context_from_profiles(PSI, q, shear_thresholds=(0.0,))


def test_flat_core_is_weak_shear_not_noise():
    q = 1.2 + 1e-12 * np.random.default_rng(0).normal(size=PSI.size)
    ctx = core_q_context_from_profiles(PSI, q)
    assert ctx.q_profile_topology == "weak_shear" and ctx.shear_sign_changes_rho == ()


def test_the_coordinate_is_named_and_shear_follows_it():
    q = _monotonic()
    pol = core_q_context_from_profiles(PSI, q)
    tor = core_q_context_from_profiles(PSI, q, rho_tor_norm=PSI)  # a different radius
    assert pol.coordinate.startswith("rho_pol_norm") and tor.coordinate == "rho_tor_norm"
    assert pol.q1_surfaces[0].rho == pytest.approx(np.sqrt(pol.q1_surfaces[0].psi_n))
    assert tor.q1_surfaces[0].rho == pytest.approx(tor.q1_surfaces[0].psi_n)
    assert pol.q1_surfaces[0].shear != pytest.approx(tor.q1_surfaces[0].shear)


def test_q_boundary_only_for_a_limited_boundary():
    q = _monotonic()
    limited = core_q_context_from_profiles(PSI, q, boundary_topology="limited")
    diverted = core_q_context_from_profiles(PSI, q, boundary_topology="lower_single_null")
    assert limited.q_boundary == pytest.approx(4.0) and limited.q95 == pytest.approx(0.8 + 3.2 * 0.95)
    assert diverted.q_boundary is None and "separatrix" in diverted.q_boundary_reason
    assert diverted.q95 == limited.q95  # q95 is never swapped for q_boundary


def test_pressure_inside_q1_needs_a_volume_and_matches_the_integral():
    q = _monotonic()
    pressure = 1e3 * (1 - PSI)
    volume = 2.0 * PSI
    without = core_q_context_from_profiles(PSI, q, pressure=pressure)
    assert without.pressure_inside_q1 == () and without.pressure_inside_q1_reason == "no volume profile"
    ctx = core_q_context_from_profiles(PSI, q, pressure=pressure, volume=volume)
    (inside,) = ctx.pressure_inside_q1
    psi1 = inside.psi_n
    assert inside.volume == pytest.approx(2 * psi1)
    assert inside.integrated_pressure == pytest.approx(2e3 * (psi1 - psi1**2 / 2), rel=1e-4)


def test_low_shear_regions_carry_pressure_context_and_q1_containment():
    rho = np.linspace(0, 1, 101)
    shear = np.where(rho < 0.4, 0.01, 1.0)
    q = 0.95 + 0.2 * rho
    (core,) = low_shear_regions(rho, shear, 0.1, q=q, pressure=1e3 * (1 - rho))
    assert core.contains_axis and core.contains_q1_surface and core.rho_end == pytest.approx(0.39)
    assert core.pressure_drop == pytest.approx(390.0) and core.mean_abs_pressure_gradient == pytest.approx(1e3)


def test_record_is_strict_json():
    record = core_q_context_from_profiles(PSI, _reversed(0.9)).as_record()
    json.dumps(record, allow_nan=False)
    assert record["q1_surface_count"] == 2 and record["double_resonant_pairs"][0]["delta_rho"] > 0


def test_packaged_39915_slice():
    pytest.importorskip("omas")
    from _sample_fixtures import sample_ods

    ods = sample_ods()
    ctx = core_q_context(ods, 0)
    assert ctx.coordinate == "rho_tor_norm" and ctx.boundary_topology == "limited"
    assert ctx.q_min > 1 and ctx.q1_surface_count == 0  # a VEST ST slice: no q=1 surface
    assert ctx.q_boundary is not None and ctx.q95 is not None
    assert ctx.pressure_inside_q1_reason == "no volume profile"  # never synthesized
    assert "equilibrium" in ods  # reading created nothing it should not have
    json.dumps(ctx.as_record(), allow_nan=False)


def test_missing_profiles_raise_rather_than_guess():
    pytest.importorskip("omas")
    from omas import ODS

    with pytest.raises(KeyError, match="profiles_1d.psi"):
        core_q_context(ODS(consistency_check=False), 0)
