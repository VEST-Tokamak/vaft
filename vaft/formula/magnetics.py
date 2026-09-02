"""Magnetic-probe geometry shared by every forward model.

One home for the rule that turns a stored IMAS ``poloidal_angle`` into a
sensitive axis, so the angle written into the IDS and the projection a
consumer applies cannot move apart (issue #288).  The Data Dictionary
defines ``poloidal_angle`` as a **clockwise** angle from the horizontal
plane, zero when the sensor normal points toward increasing major radius.
Clockwise from ``+R`` turns toward ``-Z``, so the axis is
``(cos(angle), -sin(angle))`` in ``(R, Z)`` and a probe measuring ``+Bz``
is stored as ``3*pi/2``.  Projecting with ``(cos, +sin)`` -- the
counter-clockwise reading -- inverts every probe while leaving flux loops
untouched, which is exactly how the mistake hides: VAFT carried it until
#288, and a downstream solver carried it after #293 moved the angle.
"""

from __future__ import annotations

import numpy as np

__all__ = ["probe_axis", "project_poloidal_field"]


def probe_axis(poloidal_angle):
    """``(c_R, c_Z)`` direction cosines of a probe's sensitive axis.

    Accepts a scalar or an array of angles [rad]; returns arrays of the same
    shape.  ``probe_axis(3*pi/2) == (0, +1)``: VEST's +Bz probes.
    """
    angle = np.asarray(poloidal_angle, dtype=float)
    return np.cos(angle), -np.sin(angle)


def project_poloidal_field(b_r, b_z, poloidal_angle):
    """The component of ``(B_R, B_Z)`` a probe at ``poloidal_angle`` reads.

    ``b_r`` and ``b_z`` may carry any trailing shape (a time series, a
    response row); ``poloidal_angle`` must broadcast against them -- pass a
    scalar per probe, or ``angles[:, None]`` for a stack of rows.
    """
    c_r, c_z = probe_axis(poloidal_angle)
    return np.asarray(b_r, dtype=float) * c_r + np.asarray(b_z, dtype=float) * c_z
