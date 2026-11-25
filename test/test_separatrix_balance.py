"""Signed dRsep and divertor X-point selection (#2324, #2325).

dRsep = R_out(psi_X,lower) - R_out(psi_X,upper) at the magnetic-axis height:
negative for a lower single null, zero for a balanced double null, positive
for an upper single null (Petrie et al., DIII-D; EFIT ``DRSEP``).

The analytic family ``psi = (R-1)^2 + z^2 - z^4/a^2 + skew z^3`` has an O-point
at (1, 0) and two saddles on R = 1; ``skew`` biases them.  On the outboard
midplane ``psi = (R-1)^2``, so a separatrix of flux psi_X crosses it at
``R = 1 + sqrt(psi_X)`` and the exact dRsep is known in closed form.
"""

from __future__ import annotations

import numpy as np
import pytest

from vaft.data.equilibrium import Contour, EquilibriumConvention, EquilibriumData, Topology
from vaft.formula.equilibrium import (
    separatrix_radial_separation,
    separatrix_separation_from_flux_imbalance,
    xpoint_flux_imbalance,
)
from vaft.formula.sol import (
    secondary_separatrix_exponential_factor,
    separatrix_imbalance_to_heat_flux_width,
)
from vaft.process.equilibrium import derive_boundary_representation, extract_flux_surface_contours

A = 0.65
CONVENTION = EquilibriumConvention(1, (1,), True, False, 1, 1, 1, "test")


def _saddle_fluxes(skew):
    """Lower and upper saddle fluxes of the analytic family, exactly."""
    roots = np.roots([4.0 / A**2, -3.0 * skew, -2.0])
    by_z = sorted(roots)
    return [float(zz**2 - zz**4 / A**2 + skew * zz**3) for zz in by_z]  # lower, upper


def _equilibrium(skew=0.08, n=201, *, inset=0.0, flip=False, scale=1.0, offset=0.0,
                 wall=None, extra=None, axis=True):
    r = np.linspace(0.5, 1.5, n)
    z = np.linspace(-0.8, 0.8, n + 40 if n > 100 else n)
    rm, zm = np.meshgrid(r, z, indexing="ij")
    zz = -zm if flip else zm
    base = (rm - 1) ** 2 + zz**2 - zz**4 / A**2 + skew * zz**3
    if extra is not None:
        base = base + extra(rm, zm)
    psi_lower, psi_upper = _saddle_fluxes(skew)
    if flip:
        psi_lower, psi_upper = psi_upper, psi_lower
    primary = min(psi_lower, psi_upper) * (1.0 - inset)
    raw = extract_flux_surface_contours(base, r, z, 0.0, primary, [1.0])[1.0]
    rb, zb = max(raw, key=lambda pair: pair[0].size)
    eq = EquilibriumData(
        r=r, z=z, psi=scale * base + offset, psi_axis=offset, psi_boundary=scale * primary + offset,
        magnetic_axis=(1.0, 0.0) if axis else None, lcfs=Contour(rb, zb), limiter=wall, convention=CONVENTION,
    )
    return eq, psi_lower, psi_upper


def _exact(psi_lower, psi_upper):
    return np.sqrt(psi_lower) - np.sqrt(psi_upper)


# -- formulas -----------------------------------------------------------------


def test_formulas_compute_the_signed_values():
    assert separatrix_radial_separation(1.10, 1.12) == pytest.approx(-0.02)
    np.testing.assert_allclose(separatrix_radial_separation([1.1, 1.2], [1.0, 1.2]), [0.1, 0.0])
    assert xpoint_flux_imbalance(0.10, 0.12) == pytest.approx(-0.02)
    # First order: (psi_L - psi_U)/dpsi_dR.  Negating the flux convention
    # negates both, so the geometric sign is unchanged.
    assert separatrix_separation_from_flux_imbalance(0.10, 0.12, 2.0) == pytest.approx(-0.01)
    assert separatrix_separation_from_flux_imbalance(-0.10, -0.12, -2.0) == pytest.approx(-0.01)
    assert separatrix_imbalance_to_heat_flux_width(-0.004, 0.002) == pytest.approx(-2.0)
    assert secondary_separatrix_exponential_factor(-0.004, 0.002) == pytest.approx(np.exp(-2.0))
    assert secondary_separatrix_exponential_factor(0.0, 0.002) == pytest.approx(1.0)


@pytest.mark.parametrize("call", [
    lambda: separatrix_radial_separation(np.nan, 1.0),
    lambda: xpoint_flux_imbalance(1.0, np.inf),
    lambda: separatrix_separation_from_flux_imbalance(0.1, 0.2, 0.0),
    lambda: separatrix_imbalance_to_heat_flux_width(0.01, 0.0),
    lambda: secondary_separatrix_exponential_factor(0.01, -1.0),
])
def test_formulas_reject_invalid_inputs(call):
    with pytest.raises(ValueError):
        call()


def test_linearized_and_exact_separation_agree_for_a_small_imbalance():
    # psi = (R-1)^2 on the midplane: dpsi/dR = 2 sqrt(psi) at the midpoint.
    psi_l, psi_u = 0.100, 0.102
    exact = np.sqrt(psi_l) - np.sqrt(psi_u)
    gradient = 2.0 * np.sqrt(0.5 * (psi_l + psi_u))
    assert separatrix_separation_from_flux_imbalance(psi_l, psi_u, gradient) == pytest.approx(exact, rel=1e-3)


# -- sign and symmetry ---------------------------------------------------------


def test_lower_single_null_is_negative_and_its_mirror_is_positive():
    eq, psi_l, psi_u = _equilibrium(0.08)
    lsn = derive_boundary_representation(eq)
    assert lsn.topology == Topology.LOWER_SINGLE_NULL
    assert lsn.d_r_sep.available
    assert lsn.d_r_sep.value == pytest.approx(_exact(psi_l, psi_u), abs=1e-4)
    assert lsn.d_r_sep.value < 0
    assert lsn.d_r_sep.quality["applicability"] == "diverted_balance"
    assert lsn.d_r_sep.quality["lower_role"] == "primary"
    assert lsn.d_r_sep.quality["upper_role"] == "secondary"
    roles = {point.branch: point.role for point in lsn.x_points if point.role}
    assert roles == {"lower": "primary", "upper": "secondary"}
    assert set(lsn.separatrix_x_points) == {"upper", "lower"}

    mirrored, _, _ = _equilibrium(0.08, flip=True)
    usn = derive_boundary_representation(mirrored)
    assert usn.topology == Topology.UPPER_SINGLE_NULL
    assert usn.d_r_sep.value == pytest.approx(-lsn.d_r_sep.value, abs=1e-6)


def test_balanced_double_null_is_zero_within_its_resolution():
    eq, _, _ = _equilibrium(0.0)
    rep = derive_boundary_representation(eq)
    assert rep.topology == Topology.DOUBLE_NULL
    assert rep.d_r_sep.available
    assert abs(rep.d_r_sep.value) <= max(1e-6, rep.d_r_sep.quality["resolution_m"])


@pytest.mark.parametrize("scale, offset", [(2.0 * np.pi, 0.0), (-1.0, 0.3), (-3.5, -1.2)])
def test_affine_flux_transforms_leave_the_geometric_separation_unchanged(scale, offset):
    eq, _, _ = _equilibrium(0.08)
    reference = derive_boundary_representation(eq)
    transformed, _, _ = _equilibrium(0.08, scale=scale, offset=offset)
    rep = derive_boundary_representation(transformed)
    assert rep.topology == reference.topology
    assert rep.d_r_sep.value == pytest.approx(reference.d_r_sep.value, abs=1e-6)
    # Flux metadata transforms with psi; the imbalance carries the scale.
    assert rep.d_r_sep.quality["flux_imbalance"] == pytest.approx(
        scale * reference.d_r_sep.quality["flux_imbalance"], rel=1e-6)


# -- near double null and coarse grids (#2324) ---------------------------------


@pytest.mark.parametrize("n", [201, 129, 65, 33])
@pytest.mark.parametrize("inset", [0.0, 3e-3])
def test_near_double_null_keeps_its_signed_value_on_coarse_grids(n, inset):
    """A reconstruction places psi_boundary a little inside the separatrix;
    the X-point must still be found, and the topology must not depend on the
    grid."""
    eq, psi_l, psi_u = _equilibrium(0.01, n, inset=inset)
    rep = derive_boundary_representation(eq)
    assert rep.d_r_sep.available
    assert rep.d_r_sep.value == pytest.approx(_exact(psi_l, psi_u), abs=5e-4)
    assert rep.d_r_sep.value < 0
    if rep.topology == Topology.DOUBLE_NULL:
        # The 3 mm imbalance is below what a 5 cm grid resolves in flux: a
        # double null is the honest classification, flagged as numerical, and
        # the signed value survives it.
        assert rep.topology_evidence["double_null_resolution_limited"] is True
        assert abs(rep.d_r_sep.value) <= rep.d_r_sep.quality["resolution_m"]
    else:
        assert rep.topology == Topology.LOWER_SINGLE_NULL


def test_inset_boundary_near_a_divertor_wall_is_not_called_limited():
    """The #2324 failure: a coarse grid, a boundary traced just inside the
    separatrix, and a divertor wall within a few cells of the X-point."""
    theta = np.linspace(0.0, 2.0 * np.pi, 240, endpoint=False)
    wall = Contour(1.0 + 0.45 * np.cos(theta), 0.49 * np.sin(theta))
    eq, psi_l, psi_u = _equilibrium(0.01, 65, inset=3e-3, wall=wall)
    rep = derive_boundary_representation(eq)
    assert rep.wall_contact_distance.value <= rep.topology_evidence["wall_contact_tolerance_m"]
    assert rep.topology == Topology.LOWER_SINGLE_NULL
    assert rep.d_r_sep.available and rep.d_r_sep.value < 0


def test_a_limited_plasma_with_external_nulls_is_geometric_only():
    """The wall cuts the plasma well inside both separatrices: the separation
    exists geometrically but is not a diverted magnetic-balance sample."""
    r = np.linspace(0.5, 1.5, 201)
    z = np.linspace(-0.8, 0.8, 241)
    rm, zm = np.meshgrid(r, z, indexing="ij")
    skew = 0.08
    psi = (rm - 1) ** 2 + zm**2 - zm**4 / A**2 + skew * zm**3
    psi_l, psi_u = _saddle_fluxes(skew)
    boundary = 0.6 * psi_l
    raw = extract_flux_surface_contours(psi, r, z, 0.0, boundary, [1.0])[1.0]
    rb, zb = max(raw, key=lambda pair: pair[0].size)
    # A limiter touching the boundary at its outboard midplane.
    theta = np.linspace(0.0, 2.0 * np.pi, 240, endpoint=False)
    r_touch = float(np.max(rb))
    wall = Contour(r_touch + 0.2 * (np.cos(theta) - 1.0), 0.6 * np.sin(theta))
    eq = EquilibriumData(r=r, z=z, psi=psi, psi_axis=0.0, psi_boundary=boundary, magnetic_axis=(1.0, 0.0),
                         lcfs=Contour(rb, zb), limiter=wall, convention=CONVENTION)
    rep = derive_boundary_representation(eq)
    assert rep.topology == Topology.LIMITED
    assert not rep.d_r_sep.available
    assert rep.d_r_sep.quality["applicability"] == "geometric_only"
    assert rep.d_r_sep.value == pytest.approx(_exact(psi_l, psi_u), abs=1e-4)
    assert "geometric only" in rep.d_r_sep.reason


# -- X-point selection -----------------------------------------------------------


def test_an_interior_saddle_is_never_the_secondary_x_point():
    """A local saddle inside the separatrix (an in-vessel coil's) must not
    stand in for the upper divertor X-point."""
    def bump(rm, zm):
        # A localized O/X pair inside the plasma above the axis.
        return -0.06 * np.exp(-((rm - 1.12) ** 2 + (zm - 0.25) ** 2) / 0.003)

    eq, psi_l, psi_u = _equilibrium(0.08, extra=bump)
    rep = derive_boundary_representation(eq)
    interior = [point for point in rep.stationary_points if point.kind == "x" and point.psi_n < 0.98]
    assert interior, "the fixture must contain an interior saddle"
    upper = rep.separatrix_x_points["upper"]
    assert upper.psi_n > 1.0
    assert all((point.r, point.z) != (upper.r, upper.z) for point in interior)
    assert rep.d_r_sep.value == pytest.approx(_exact(psi_l, psi_u), abs=2e-3)


def test_a_single_relevant_null_leaves_drsep_unavailable_not_zero():
    """Without a second divertor X-point there is no secondary separatrix."""
    r = np.linspace(0.5, 1.5, 201)
    z = np.linspace(-0.8, 0.6, 221)
    rm, zm = np.meshgrid(r, z, indexing="ij")
    psi = (rm - 1) ** 2 + zm**2 + 1.2 * zm**3  # one saddle, below the axis
    zx = -2.0 / (3 * 1.2)
    psi_x = zx**2 + 1.2 * zx**3
    raw = extract_flux_surface_contours(psi, r, z, 0.0, psi_x, [1.0])[1.0]
    rb, zb = max(raw, key=lambda pair: pair[0].size)
    eq = EquilibriumData(r=r, z=z, psi=psi, psi_axis=0.0, psi_boundary=psi_x, magnetic_axis=(1.0, 0.0),
                         lcfs=Contour(rb, zb), convention=CONVENTION)
    rep = derive_boundary_representation(eq)
    assert rep.topology == Topology.LOWER_SINGLE_NULL
    assert not rep.d_r_sep.available
    assert rep.d_r_sep.value is None
    assert "upper" in rep.d_r_sep.reason


def test_missing_axis_gives_no_geometric_centre_fallback():
    eq, _, _ = _equilibrium(0.08, axis=False)
    rep = derive_boundary_representation(eq)
    assert rep.d_r_sep.value is None
    assert "magnetic axis" in rep.d_r_sep.reason


# -- scan consistency ----------------------------------------------------------------


@pytest.mark.parametrize("inset, tolerance, expected", [
    (0.0, 2e-3, "lower_single_null"),
    (1.5e-3, 2e-3, "lower_single_null"),
    # Outside the scan's own active tolerance: its near-null band, by policy.
    (3e-3, 2e-3, "near_null"),
    (3e-3, 5e-3, "lower_single_null"),
])
def test_scan_classification_follows_the_canonical_active_flags(inset, tolerance, expected):
    from vaft.code.tokamaker.topology import classify_boundary

    eq, _, _ = _equilibrium(0.01, 65, inset=inset)
    report = classify_boundary(eq, active_tolerance=tolerance)
    canonical = report.representation
    assert [xp["active"] for xp in report.x_points] == [xp.active for xp in canonical.x_points]
    assert report.topology.value == expected
    if expected == "lower_single_null":
        assert report.d_r_sep is not None and report.d_r_sep < 0
    else:
        assert report.d_r_sep is None


def test_an_unrelated_midplane_saddle_is_not_a_divertor_x_point():
    """A saddle off to the side of the plasma (a coil's) is outside the divertor
    sector, even when its flux would make it the nearest secondary candidate."""
    from vaft.data.equilibrium import StationaryPoint
    from vaft.process._equilibrium_parametric import (
        _boundary_relevant_saddles, _flux_resolution, _grid_spacing, _select_separatrix_x_points,
        _stationary_points,
    )

    eq, psi_l, psi_u = _equilibrium(0.08)
    saddles = [point for point in _stationary_points(eq) if point.kind == "x"]
    resolution, spacing = _flux_resolution(eq), _grid_spacing(eq)
    minor = 0.5 * float(np.ptp(eq.lcfs.r))
    active, _ = _boundary_relevant_saddles(eq, saddles, None, resolution, minor)
    decoy = StationaryPoint(1.42, 0.05, 1.001 * psi_l, 1.001, "x", -1.0, 2.0)
    chosen, why, evidence = _select_separatrix_x_points(
        eq, [*saddles, decoy], active, None, resolution, spacing, minor)
    assert chosen["upper"].psi_n > 1.1
    assert any("divertor sector" in note for note in evidence["upper_rejected"])


@pytest.mark.parametrize("skew", [0.002, 0.005])
@pytest.mark.parametrize("inset", [0.0, 3e-3])
def test_a_resolved_secondary_within_one_percent_stays_secondary(skew, inset):
    """The LCFS-anchored rule must not promote a secondary X-point that the
    grid resolves from the primary into a second primary (cold review of
    #2325): a near-double null is a single null with a signed dRsep."""
    eq, psi_l, psi_u = _equilibrium(skew, 201, inset=inset)
    rep = derive_boundary_representation(eq)
    assert rep.topology == Topology.LOWER_SINGLE_NULL
    assert rep.separatrix_x_points["upper"].role == "secondary"
    assert rep.d_r_sep.value == pytest.approx(_exact(psi_l, psi_u), abs=2e-4)
    assert rep.d_r_sep.value < -rep.d_r_sep.quality["resolution_m"]


def test_an_explicit_flux_tolerance_caps_the_anchored_offset():
    eq, _, _ = _equilibrium(0.01, 201, inset=3e-3)
    assert derive_boundary_representation(eq, flux_tolerance=5e-3).topology == Topology.LOWER_SINGLE_NULL
    strict = derive_boundary_representation(eq, flux_tolerance=1e-4)
    assert not strict.topology.is_diverted
    assert not any(point.active for point in strict.x_points)


def test_a_wall_contact_away_from_the_saddle_keeps_the_plasma_limited():
    """An inset boundary that also rests on an outboard limiter is limited by
    that limiter, however close its X-point."""
    eq, _, _ = _equilibrium(0.01, 201, inset=3e-3)
    r_touch = float(np.max(eq.lcfs.r))
    theta = np.linspace(0.0, 2.0 * np.pi, 240, endpoint=False)
    wall = Contour(r_touch + 0.3 * (np.cos(theta) - 1.0), 0.75 * np.sin(theta))
    limited, _, _ = _equilibrium(0.01, 201, inset=3e-3, wall=wall)
    rep = derive_boundary_representation(limited)
    assert rep.topology == Topology.LIMITED
    assert any("rejected" in note for note in rep.topology_evidence["lcfs_anchored"])
    assert not rep.d_r_sep.available
