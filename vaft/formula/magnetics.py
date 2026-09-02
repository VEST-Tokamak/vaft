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
    r"""Direction cosines $(c_R, c_Z)$ of a magnetic probe's sensitive axis.

    $$(c_R, c_Z) = \big(\cos\theta_{pol},\ -\sin\theta_{pol}\big)$$

    Parameters
    ----------
    poloidal_angle : float or np.ndarray
        Probe poloidal angle as the Data Dictionary stores it [rad].

    Returns
    -------
    c_R : np.ndarray
        Radial direction cosine, the shape of the input [-].
    c_Z : np.ndarray
        Vertical direction cosine, the shape of the input [-].

    Convention
    ----------
    The Data Dictionary defines ``poloidal_angle`` as a **clockwise** angle from
    the horizontal plane, zero when the sensor normal points toward increasing
    major radius. Clockwise from $+R$ turns toward $-Z$, which is where the minus
    sign comes from, and why a probe measuring $+B_Z$ is stored as $3\pi/2$ and
    ``probe_axis(3*pi/2)`` is $(0, +1)$.

    Projecting with $(\cos, +\sin)$ -- reading the angle counter-clockwise --
    inverts every probe while leaving flux loops untouched, which is exactly how
    the mistake hides. VAFT carried it until #288, and a downstream solver carried
    it after #293 moved the angle.

    References
    ----------
    .. [1] IMAS Data Dictionary, ``magnetics.b_field_pol_probe[:].poloidal_angle``.

    See Also
    --------
    project_poloidal_field
    """
    angle = np.asarray(poloidal_angle, dtype=float)
    return np.cos(angle), -np.sin(angle)


def project_poloidal_field(b_r, b_z, poloidal_angle):
    r"""The component of a poloidal field a probe at ``poloidal_angle`` reads.

    $$B_{\mathrm{probe}} = B_R\,c_R + B_Z\,c_Z$$

    with $(c_R, c_Z)$ from :func:`probe_axis`.

    Parameters
    ----------
    b_r : float or np.ndarray
        Radial field at the probe [T].
    b_z : float or np.ndarray
        Vertical field at the probe [T].
    poloidal_angle : float or np.ndarray
        Probe poloidal angle, broadcast against the fields [rad].

    Returns
    -------
    np.ndarray
        Field along the probe's sensitive axis [T].

    Convention
    ----------
    Clockwise ``poloidal_angle``, as :func:`probe_axis` documents. This is the one
    place the projection rule lives, so the angle written into the IDS and the
    projection a consumer applies cannot drift apart (#288).

    Assumptions
    -----------
    ``b_r`` and ``b_z`` may carry any trailing shape -- a time series, a response
    row -- and ``poloidal_angle`` must broadcast against them: pass a scalar per
    probe, or ``angles[:, None]`` for a stack of rows.

    References
    ----------
    .. [1] IMAS Data Dictionary, ``magnetics.b_field_pol_probe[:].poloidal_angle``.
    """
    c_r, c_z = probe_axis(poloidal_angle)
    return np.asarray(b_r, dtype=float) * c_r + np.asarray(b_z, dtype=float) * c_z
