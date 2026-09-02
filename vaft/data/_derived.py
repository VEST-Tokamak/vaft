"""Derived equilibrium profiles shared by the format adapters.

``vaft.data.eqdsk`` and ``vaft.data.vfit`` both have to turn a stored ``q``
profile into a toroidal-flux coordinate.  Doing it twice is how the two
conventions in issue #276 diverged, so the guard logic lives here and both
adapters call it -- as does the ODS-side updater in :mod:`vaft.omas.update`.

The pure mathematics is in :mod:`vaft.formula.equilibrium`; what this module
adds is the *policy*: when the stored profile cannot support a real toroidal
coordinate, refuse rather than fill the field with something else.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Optional

import numpy as np

__all__ = [
    "AXIS_Q_OUTLIER_RATIO",
    "RhoTorProfile",
    "is_rho_pol_proxy",
    "rho_tor_profile",
]


class RhoTorProfile(NamedTuple):
    """Toroidal-flux coordinate of one time slice.

    ``rho_tor`` is ``None`` when no ``B0`` was supplied: the normalized
    coordinate does not need one, the dimensional coordinate does.
    """

    phi: np.ndarray
    rho_tor: Optional[np.ndarray]
    rho_tor_norm: np.ndarray


def rho_tor_profile(q: Any, psi_wb: Any, b0: Any = None) -> Optional[RhoTorProfile]:
    """Integrate ``q`` into toroidal flux and the rho_tor coordinate.

    ``psi_wb`` is the poloidal flux **in weber** -- what the IMAS DD stores since
    issue #236.  ``Phi = integral(q dpsi_wb)``, ``rho_tor = sqrt(|Phi|/(pi|B0|))``
    and ``rho_tor_norm = rho_tor/rho_tor[-1]``.  This is the coordinate OMFIT's
    ``fluxSurfaces`` produces -- they agree to within 1.5e-3 in normalized units
    on VEST g-files.  The ``sqrt(psi_N)`` proxy VAFT wrote before is not: that is
    ``rho_pol``, off by up to 0.126 on the packaged samples, and every kinetic
    profile is mapped onto this grid (issue #276).

    ``b0`` is optional because the normalized coordinate does not need it -- the
    field cancels in the ratio.  Without it ``rho_tor`` is ``None`` and only
    ``phi``/``rho_tor_norm`` are available.

    Returns ``None`` when the profile cannot support a toroidal coordinate.  The
    caller must then leave ``rho_tor_norm`` unset and write the poloidal quantity
    instead -- ``profiles_1d.psi_norm``, since the DD gives equilibrium
    ``profiles_1d`` no ``rho_pol_norm`` sibling (``omas_info_node`` returns an
    empty record for it, where ``psi_norm`` is a documented leaf).

    That happens when the inputs are degenerate (mismatched or too-short arrays,
    non-finite values, zero edge flux), and when the cumulative flux is not
    monotonic -- a ``q`` that changes sign integrates to a non-monotonic ``Phi``,
    which is not a radial coordinate at all.  ``derive_radial_coordinates`` in
    :mod:`vaft.process._equilibrium_parametric` refuses on the same grounds.
    """
    from vaft.formula.equilibrium import rho_tor_from_phi, toroidal_flux_from_q_psi

    q = np.asarray(q, dtype=float).reshape(-1)
    psi_wb = np.asarray(psi_wb, dtype=float).reshape(-1)
    if q.size != psi_wb.size or q.size < 2:
        return None
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(psi_wb)):
        return None
    _warn_on_axis_q_outlier(q)

    phi = toroidal_flux_from_q_psi(q, psi_wb)
    edge = float(phi[-1])
    if not np.isfinite(edge) or edge == 0.0:
        return None
    # A monotonic |Phi| is what makes sqrt(|Phi|/Phi_edge) a coordinate.  Allow
    # rounding-scale inversions, the same tolerance the volume profile uses.
    fraction = phi / edge
    if np.any(fraction < -1e-12) or np.any(np.diff(fraction) < -1e-10):
        return None

    rho_tor_norm = np.sqrt(np.clip(fraction, 0.0, None))
    edge_norm = float(rho_tor_norm[-1])
    if not np.isfinite(edge_norm) or edge_norm <= 0.0:
        return None
    rho_tor_norm = rho_tor_norm / edge_norm

    rho_tor = None
    if b0 is not None and abs(float(b0)) > 0.0:
        rho_tor = rho_tor_from_phi(phi, b0)
        if not np.all(np.isfinite(rho_tor)):
            rho_tor = None
    return RhoTorProfile(phi, rho_tor, rho_tor_norm)


#: How close a stored profile has to sit to ``sqrt(psi_N)`` to be called the
#: proxy.  The margin is ten orders of magnitude, not a tuning knob: on the
#: packaged samples ``max|rho - sqrt(psi_N)|`` is 1.1e-16, while a real
#: coordinate is 0.114 away (the kineticEfit reference) or 0.154 (a fresh
#: ``to_omas``).  Anything in between is not a coordinate anyone produced.
RHO_POL_PROXY_TOLERANCE = 1e-6


def is_rho_pol_proxy(rho_tor_norm: Any, psi_norm: Any = None) -> bool:
    """Whether a stored ``rho_tor_norm`` is really the ``sqrt(psi_N)`` proxy.

    Producers stopped writing the proxy in issue #276, but files written before
    that still carry it, and it is indistinguishable from the real coordinate by
    name alone.  A reader that trusts it plots a poloidal coordinate under a
    toroidal label; every packaged sample is in that state today.

    ``psi_norm`` defaults to a uniform grid, which is what an EFIT ``profiles_1d``
    is on -- the proxy was written as ``sqrt(linspace(0, 1, n))``.  Pass the
    slice's own ``psi_norm`` when it is not uniform.

    A profile that is not a plausible normalized coordinate at all -- wrong
    length, non-finite, not running 0 to 1 -- is *not* reported as the proxy:
    this answers one narrow question, and the caller's own validity checks
    remain theirs.
    """
    rho = np.asarray(rho_tor_norm, dtype=float).reshape(-1)
    if rho.size < 2 or not np.all(np.isfinite(rho)):
        return False
    if psi_norm is None:
        reference = np.linspace(0.0, 1.0, rho.size)
    else:
        reference = np.asarray(psi_norm, dtype=float).reshape(-1)
        if reference.size != rho.size or not np.all(np.isfinite(reference)):
            return False
    return bool(
        np.max(np.abs(rho - np.sqrt(np.clip(reference, 0.0, None))))
        < RHO_POL_PROXY_TOLERANCE
    )


#: How far ``q[0]`` may stand above its immediate neighbours before it is called
#: an outlier. A physical q profile is smooth, so a factor of two against the
#: points right beside it is a solver or discretization artifact rather than
#: reversed shear -- the comparison is local, not against ``q_min``.
AXIS_Q_OUTLIER_RATIO = 2.0


def _warn_on_axis_q_outlier(q: np.ndarray) -> None:
    """Warn when ``q`` on axis contradicts the profile it belongs to.

    EFIT's ``q[0]`` is often unreliable, and ``Phi = integral(q dpsi)`` carries it
    into ``rho_tor``, ``rho_tor_norm`` and every kinetic profile mapped onto that
    grid. On the packaged kineticEfit reference ``q[0] = 8.07`` against a
    neighbourhood of 1.89 -- 4.3x -- and it moves ``rho_tor_norm`` by up to 0.043
    near the axis and 0.0045 at mid-radius.

    It is **not** repaired here. Extrapolating ``q[0]`` from its neighbours does
    bring the coordinate 5.8x closer to what that reference stores (0.0369 to
    0.0064), which is good evidence that the outlier is what the two disagree
    about -- and is exactly why silently repairing it would be wrong. That would
    replace a measurement with a guess, and hide a solver problem behind a
    coordinate that looks fine. Issue #317 records the finding; the caller is
    told, and decides.
    """
    import warnings

    if q.size < 4:
        return
    neighbourhood = np.median(np.abs(q[1:5]))
    if neighbourhood <= 0.0 or not np.isfinite(neighbourhood):
        return
    ratio = abs(float(q[0])) / float(neighbourhood)
    if ratio <= AXIS_Q_OUTLIER_RATIO:
        return
    warnings.warn(
        f"q on axis is {q[0]:.3g}, {ratio:.1f}x its immediate neighbours "
        f"(median {neighbourhood:.3g}); the toroidal flux integral carries that "
        f"into rho_tor_norm. The profile is used as given -- see issue #317.",
        RuntimeWarning,
        stacklevel=3,
    )
