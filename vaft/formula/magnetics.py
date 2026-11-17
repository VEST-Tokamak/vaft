"""Magnetic-probe geometry shared by every forward model.

One home for the rule that turns a stored IMAS ``poloidal_angle`` into a
sensitive axis, so the angle written into the IDS and the projection a
consumer applies cannot move apart.  The Data Dictionary
defines ``poloidal_angle`` as a **clockwise** angle from the horizontal
plane, zero when the sensor normal points toward increasing major radius.
Clockwise from ``+R`` turns toward ``-Z``, so the axis is
``(cos(angle), -sin(angle))`` in ``(R, Z)`` and a probe measuring ``+Bz``
is stored as ``3*pi/2``.  Projecting with ``(cos, +sin)`` -- the
counter-clockwise reading -- flips the vertical-field part of every probe
(a purely vertical one reads inverted) while leaving flux loops untouched,
which is exactly how the mistake hides: VAFT itself once carried it, and a
downstream solver carried it after VAFT moved its stored angle to the DD
convention.
"""

from __future__ import annotations

import numpy as np

__all__ = ["probe_axis", "project_poloidal_field"]


def probe_axis(poloidal_angle):
    r"""Direction cosines ``(c_R, c_Z)`` of a poloidal-field probe's sensitive axis.
    
    $$c_R = \cos\theta, \qquad c_Z = -\sin\theta$$
    
    ``probe_axis(3*pi/2) == (0, +1)``: VEST's +Bz probes.
    
    Parameters
    ----------
    poloidal_angle : float or ndarray
        Probe orientation angle in the DD convention; scalar or array [rad].
    
    Returns
    -------
    tuple of ndarray
        ``(c_R, c_Z)`` direction cosines, each the shape of the input [-].

    Convention
    ----------
    The Data Dictionary measures ``poloidal_angle`` **clockwise** from $+R$,
    i.e. turning toward $-Z$, so the axis is $(\cos\theta, -\sin\theta)$.
    Reading the angle counter-clockwise, $(\cos\theta, +\sin\theta)$, mirrors
    every axis about the midplane: a $+B_Z$ probe stored at $3\pi/2$ becomes a
    $-B_Z$ probe.  The $B_Z$ contribution of every probe with a vertical
    component then enters with the wrong sign -- a purely vertical probe reads
    exactly inverted, an oblique one simply wrong -- while purely radial
    probes and all flux loops, which carry no angle, are unaffected.  That
    asymmetry is how the error hides in a reconstruction: VAFT itself once
    carried it, and a downstream solver did after VAFT moved its stored angle
    to the DD convention.

    References
    ----------
    .. [1] IMAS Data Dictionary, ``magnetics.b_field_pol_probe[:].poloidal_angle``.
    """
    angle = np.asarray(poloidal_angle, dtype=float)
    return np.cos(angle), -np.sin(angle)


def project_poloidal_field(b_r, b_z, poloidal_angle):
    r"""The component of ``(B_R, B_Z)`` a probe oriented at ``poloidal_angle`` reads.
    
    $$B_{\mathrm{probe}} = B_R \cos\theta - B_Z \sin\theta$$
    
    ``b_r`` and ``b_z`` may carry any trailing shape (a time series, a response
    row); ``poloidal_angle`` must broadcast against them -- pass a scalar per
    probe, or ``angles[:, None]`` for a stack of rows.
    
    Parameters
    ----------
    b_r : float or ndarray
        Radial field component [T].
    b_z : float or ndarray
        Vertical field component [T].
    poloidal_angle : float or ndarray
        Probe orientation angle in the DD convention [rad].
    
    Returns
    -------
    ndarray
        Projected field along the probe axis [T].

    Convention
    ----------
    :func:`probe_axis`'s DD rule applied: ``poloidal_angle`` is clockwise from
    $+R$, so the reading is $B_R\cos\theta - B_Z\sin\theta$.  The
    counter-clockwise reading, $B_R\cos\theta + B_Z\sin\theta$, returns the
    opposite of the measured signal for a vertical probe ($\theta = \pi/2$ or
    $3\pi/2$), a wrong value for an oblique one and the right one for a radial
    probe, so a synthetic diagnostic built on it disagrees with the
    measurement on every probe with a vertical component.

    References
    ----------
    .. [1] IMAS Data Dictionary, ``magnetics.b_field_pol_probe[:].poloidal_angle``.
    """
    c_r, c_z = probe_axis(poloidal_angle)
    return np.asarray(b_r, dtype=float) * c_r + np.asarray(b_z, dtype=float) * c_z
