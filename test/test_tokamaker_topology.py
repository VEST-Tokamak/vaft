"""Unit tests for the scan-grade topology classification (OFT-free)."""

from dataclasses import replace

import numpy as np
import pytest

# The analytic limited / double-null fixtures from the issue-65 descriptor
# tests are exactly the equilibria these classifications must nail; reuse
# them rather than re-deriving the analytic psi maps.
from test_parametric_equilibrium import _analytic_equilibrium

from vaft.code.tokamaker.topology import ScanTopology, classify_boundary
from vaft.data.equilibrium import EquilibriumData


def test_limited_equilibrium_classifies_limited_with_contact():
    report = classify_boundary(_analytic_equilibrium("limited"))

    assert report.topology is ScanTopology.LIMITED
    assert report.limiter_contact is not None
    assert report.limiter_contact.distance >= 0.0
    assert np.isfinite(report.limiter_contact.distance)
    assert report.reason is None

    payload = report.to_dict()
    assert payload["topology"] == "limited"
    assert payload["limiter_contact"]["distance"] == pytest.approx(
        report.limiter_contact.distance)


def test_double_null_classifies_at_default_tolerance():
    # the analytic DN has its X-points exactly on psi_boundary (psi_n == 1)
    report = classify_boundary(_analytic_equilibrium("double"))

    assert report.topology is ScanTopology.DOUBLE_NULL
    assert report.null_margin == pytest.approx(0.0, abs=1e-6)
    actives = [xp for xp in report.x_points if xp["active"]]
    assert {xp["z"] > 0 for xp in actives} == {True, False}
    # balanced DN: separatrix balance ~ 0
    assert report.d_r_sep is not None
    assert report.d_r_sep == pytest.approx(0.0, abs=5e-3)
    assert report.limiter_contact is None            # diverted: no contact search


def test_near_null_band_between_active_and_limited():
    base = _analytic_equilibrium("double")
    # pull the LCFS inside the separatrix: X-points sit at psi_n slightly > 1
    eq = replace(base, psi_boundary=base.psi_boundary * 0.98)
    report = classify_boundary(eq, active_tolerance=2.0e-3, near_null_band=5.0e-2)

    assert report.topology is ScanTopology.NEAR_NULL
    assert report.null_margin is not None
    assert 2.0e-3 < report.null_margin <= 5.0e-2
    # near-null states still report the wall clearance
    assert report.limiter_contact is not None


def test_far_null_classifies_limited():
    base = _analytic_equilibrium("double")
    eq = replace(base, psi_boundary=base.psi_boundary * 0.5)
    report = classify_boundary(eq, near_null_band=5.0e-2)

    assert report.topology is ScanTopology.LIMITED
    # a null this far out either exceeds the band or leaves the descriptor
    # layer's psi_n retention window entirely (margin unknown)
    assert report.null_margin is None or report.null_margin > 5.0e-2


def test_tolerance_knobs_move_the_classification():
    base = _analytic_equilibrium("double")
    eq = replace(base, psi_boundary=base.psi_boundary * 0.98)

    generous = classify_boundary(eq, active_tolerance=0.1)
    assert generous.topology is ScanTopology.DOUBLE_NULL

    strict = classify_boundary(eq, active_tolerance=1e-4, near_null_band=1e-3)
    assert strict.topology is ScanTopology.LIMITED


def test_unreadable_source_reports_unknown_instead_of_raising():
    report = classify_boundary(EquilibriumData())

    assert report.topology is ScanTopology.UNKNOWN
    assert report.reason
    assert report.to_dict()["topology"] == "unknown"


def test_far_side_vacuum_saddles_do_not_register_as_near_null():
    # a saddle with psi_n well below 1 sits on the far side of the LCFS flux;
    # it must not produce an approach margin (would misreport NEAR_NULL)
    base = _analytic_equilibrium("double")
    # push the LCFS OUTSIDE the separatrix: X-points at psi_n < 1
    eq = replace(base, psi_boundary=base.psi_boundary * 1.10)
    report = classify_boundary(eq, active_tolerance=2.0e-3, near_null_band=5.0e-2)

    assert report.topology is ScanTopology.LIMITED
    assert report.null_margin is None


def _with_limiter(eq, r, z):
    from vaft.data.equilibrium import Contour

    return replace(eq, limiter=Contour(r, z, closed=True))


def test_limited_contact_reports_its_side():
    report = classify_boundary(_analytic_equilibrium("limited"))

    # the analytic limited plasma rests on the outboard midplane of its limiter
    assert report.limiter_contact.side == "outboard"
    assert report.to_dict()["limiter_contact"]["side"] == "outboard"
    assert report.contact_tolerance > 0.0


def test_inboard_contact_is_named_inboard():
    eq = _analytic_equilibrium("limited")
    # the fixture limiter touches the LCFS at both midplanes; pull its outboard
    # half 5 cm back so only the inboard (high-field-side) contact remains
    r0 = eq.magnetic_axis[0]
    r = np.where(eq.limiter.r > r0, eq.limiter.r + 0.05, eq.limiter.r)
    report = classify_boundary(_with_limiter(eq, r, eq.limiter.z))

    assert report.topology is ScanTopology.LIMITED
    assert report.limiter_contact.side == "inboard"
    assert report.limiter_contact.distance <= report.contact_tolerance


def test_null_free_boundary_clear_of_the_wall_is_not_limited():
    # No X-point and no wall contact: neither sets the boundary, so the contour
    # cannot be a free-boundary LCFS. Calling it LIMITED hid a TokaMaker run
    # whose boundary floated 6 cm off the VEST inboard limiter (#1469).
    eq = _analytic_equilibrium("limited")
    # the same wall, enlarged 20 % about the magnetic axis: 7 cm clear everywhere
    r0, z0 = eq.magnetic_axis
    wall = (r0 + 1.2 * (eq.limiter.r - r0), z0 + 1.2 * (eq.limiter.z - z0))
    report = classify_boundary(_with_limiter(eq, *wall))

    assert report.topology is ScanTopology.AMBIGUOUS
    assert report.limiter_contact is not None
    assert report.limiter_contact.distance > report.contact_tolerance
    assert "clear of the limiter" in report.reason

    # an explicit tolerance can accept the gap
    loose = classify_boundary(_with_limiter(eq, *wall), contact_tolerance=0.1)
    assert loose.topology is ScanTopology.LIMITED
