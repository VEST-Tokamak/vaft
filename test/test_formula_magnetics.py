"""The probe projection rule, in one place, read the way the IMAS DD reads it.

Issue #288 found VAFT storing ``pi/2`` and projecting with ``(cos, +sin)`` --
two errors that cancelled.  #293 fixed the stored angle; this module is where
the projection lives so that no consumer can carry the other half of the
mistake on its own (a downstream solver did, after #293 moved the angle).
"""

import math

import numpy as np
import pytest

from vaft.formula.magnetics import probe_axis, project_poloidal_field
from vaft.machine_mapping.magnetics import POLOIDAL_ANGLE


def test_the_stored_vest_angle_means_plus_bz():
    c_r, c_z = probe_axis(POLOIDAL_ANGLE)
    assert c_r == pytest.approx(0.0, abs=1e-12)
    assert c_z == pytest.approx(1.0)


def test_clockwise_from_plus_r_turns_toward_minus_z():
    c_r, c_z = probe_axis(np.array([0.0, math.pi / 2, math.pi]))
    np.testing.assert_allclose(c_r, [1.0, 0.0, -1.0], atol=1e-12)
    np.testing.assert_allclose(c_z, [0.0, -1.0, 0.0], atol=1e-12)


def test_projection_reads_bz_for_a_plus_bz_probe_and_broadcasts_over_rows():
    b_r = np.array([[1.0, 2.0], [3.0, 4.0]])
    b_z = np.array([[10.0, 20.0], [30.0, 40.0]])
    angles = np.array([POLOIDAL_ANGLE, 0.0])[:, None]
    projected = project_poloidal_field(b_r, b_z, angles)
    np.testing.assert_allclose(projected[0], b_z[0])   # +Bz probe
    np.testing.assert_allclose(projected[1], b_r[1])   # +Br probe


def test_the_counter_clockwise_reading_would_invert_every_plus_bz_probe():
    wrong = np.cos(POLOIDAL_ANGLE) * 0.0 + np.sin(POLOIDAL_ANGLE) * 1.0
    right = project_poloidal_field(0.0, 1.0, POLOIDAL_ANGLE)
    assert wrong == pytest.approx(-right)
