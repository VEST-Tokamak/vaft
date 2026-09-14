"""Toroidal placements written without an installation record (issue #746).

These are best-available inferences, not measurements. They are pinned so that
a later correction is a deliberate edit with a failing test to update, rather
than a silent drift -- and so the reasoning behind each stays attached to a
value someone will otherwise read as surveyed.

Each constant under test has ``PROVISIONAL`` in its name or docstring, so
`grep -rn PROVISIONAL vaft/machine_mapping/` finds the whole set.
"""

from __future__ import annotations

import numpy as np
import pytest

from vaft.machine_mapping.registry import port_phi


# ---------------------------------------------------------------------------
# Soft X-ray: the geometry table's angle is read as a VEST clock angle
# ---------------------------------------------------------------------------


def test_sxr_geometry_column_is_converted_not_passed_through():
    from vaft.machine_mapping.soft_x_rays import (
        SXR_GEOMETRY_PHI_IS_VEST_CLOCK_ANGLE,
        _geometry_phi_to_imas,
    )

    assert SXR_GEOMETRY_PHI_IS_VEST_CLOCK_ANGLE is True
    # 12 o'clock is its own reflection, so the 17592 arrays are unaffected.
    assert _geometry_phi_to_imas(0.0) == pytest.approx(0.0)
    # 120 deg of clock angle is 4 o'clock, which is phi = 240.
    assert np.rad2deg(_geometry_phi_to_imas(np.deg2rad(120.0))) == pytest.approx(240.0)


def test_sxr_arrays_land_where_the_conversion_says():
    from vaft.machine_mapping.soft_x_rays import load_sxr_geometry_table

    by_array = {}
    for (array, _), row in load_sxr_geometry_table().items():
        by_array.setdefault(array, float(row["phi"]))

    # The loader already returns IMAS phi.
    resolved = {array: round(float(np.rad2deg(phi)), 6) for array, phi in by_array.items()}
    assert resolved == {
        "horizontal": 0.0,   # 12MM10, corroborated by the port document
        "vertical": 0.0,     # 12MM10
        "lowermid": 240.0,   # PROVISIONAL: 4 o'clock, not in the port document
        "bottom": 240.0,     # PROVISIONAL
    }


def test_the_12_oclock_sxr_arrays_agree_with_their_documented_port():
    """The half of the SXR placement that is not a guess."""
    from vaft.machine_mapping.soft_x_rays import _geometry_phi_to_imas

    assert _geometry_phi_to_imas(0.0) == pytest.approx(port_phi("12MM10"))


# ---------------------------------------------------------------------------
# CES: one of two documented ports, chosen on beam geometry
# ---------------------------------------------------------------------------


def test_ces_port_is_one_of_the_two_the_document_lists():
    from vaft.machine_mapping.charge_exchange import CES_CANDIDATE_PORTS, CES_PORT

    assert CES_PORT in CES_CANDIDATE_PORTS


def test_the_chosen_ces_port_is_the_one_nearer_the_beam():
    """The whole basis for the choice, restated as an assertion.

    If the beam geometry or either port angle ever changes, the reasoning
    behind ``CES_PORT`` should be rechecked rather than silently invalidated.
    """
    from vaft.machine_mapping.charge_exchange import CES_CANDIDATE_PORTS, CES_PORT

    entry, exit_ = port_phi("2MR"), port_phi("7M12")
    chord = exit_ + 0.5 * np.angle(np.exp(1j * (entry - exit_)))

    def separation(port):
        return abs(np.angle(np.exp(1j * (port_phi(port) - chord))))

    assert CES_PORT == min(CES_CANDIDATE_PORTS, key=separation)
    assert np.rad2deg(separation(CES_PORT)) == pytest.approx(45.0)


def test_ces_channels_are_written_at_that_port():
    from vaft.machine_mapping.charge_exchange import CES_PORT

    assert np.rad2deg(port_phi(CES_PORT)) == pytest.approx(270.0)


# ---------------------------------------------------------------------------
# Camera: its own port is documented; the projection landmarks are not
# ---------------------------------------------------------------------------


def test_the_camera_views_through_the_entrance_port():
    from vaft.machine_mapping.camera_visible import CAMERA_PORT

    assert CAMERA_PORT == "6MR"
    assert np.rad2deg(port_phi(CAMERA_PORT)) == pytest.approx(180.0)


def test_the_projection_landmarks_are_still_in_their_own_frame():
    """Pins the discrepancy rather than the fix.

    The rectangular ports really are 120 deg apart, but at 60/180/300, and the
    projection uses 30/150. Changing the projection means redoing a camera
    calibration, so this test records that it has not happened yet -- and will
    fail loudly when someone does it, which is when the calibration needs
    rechecking.
    """
    from vaft.machine_mapping.camera_visible import (
        PORT_FIRST_CENTRE_RAD,
        PORT_SEPARATION_RAD,
    )

    assert np.rad2deg(PORT_FIRST_CENTRE_RAD) == pytest.approx(30.0)
    assert np.rad2deg(PORT_SEPARATION_RAD) == pytest.approx(120.0)
    rectangular = sorted(np.rad2deg(port_phi(name)) for name in ("2MR", "6MR", "10MR"))
    assert rectangular == pytest.approx([60.0, 180.0, 300.0])
    assert not np.isclose(np.rad2deg(PORT_FIRST_CENTRE_RAD), rectangular).any()
