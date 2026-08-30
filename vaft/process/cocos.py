"""Operational COCOS handling: consistency checking against Sauter Eq. 23.

:mod:`vaft.data.cocos` declares *what* each convention and each external code is.
This module answers *whether a given equilibrium is actually consistent with the
convention it claims*, which is the check that catches a mislabelled file before
its signs propagate into derived quantities.

Sauter & Medvedev 2013 Eq. 23 gives six relations that any equilibrium in a given
COCOS must satisfy, in terms of the signs of the plasma current and the vacuum
toroidal field:

===============  =====================================
quantity         required sign
===============  =====================================
``F``            ``sigma_B0``
``Phi_tor``      ``sigma_B0``
``psi_edge -``   ``sigma_Ip * sigma_Bp``
``psi_axis``
``dp/dpsi``      ``-sigma_Ip * sigma_Bp``
``j_phi``        ``sigma_Ip``
``q``            ``sigma_Ip * sigma_B0 * sigma_rhotheta``
===============  =====================================

The ``q`` relation is reported as a warning rather than an error: Sauter Sect. IV
notes that codes frequently emit ``abs(q)``, so a mismatch there is common and is
not on its own evidence of a wrong index.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vaft.data.cocos import cocos_spec
from vaft.data.equilibrium import ValidationIssue, ValidationReport

__all__ = ["cocos_consistency_signs", "identify_convention", "identify_flux_exponent", "validate_cocos"]

#: Relations Eq. 23 defines, in report order, with the field each one inspects.
_RELATIONS = (
    ("f", "cocos_sign_f", "f", "F = R*B_phi"),
    ("dpsi", "cocos_sign_dpsi", "psi_boundary", "psi_boundary - psi_axis"),
    ("pprime", "cocos_sign_pprime", "pressure", "dp/dpsi"),
    ("q", "cocos_sign_q", "q", "q"),
    ("j_phi", "cocos_sign_jphi", "j_phi", "toroidal current density"),
    ("phi_tor", "cocos_sign_phi_tor", "phi_tor", "toroidal flux"),
)


def _sign(value: Any) -> int | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float).reshape(-1)
    array = array[np.isfinite(array)]
    if not array.size:
        return None
    # The bulk sign: a profile that crosses zero is judged by where its weight is.
    total = float(np.nanmedian(array))
    if not np.isfinite(total) or abs(total) < 1e-30:
        return None
    return 1 if total > 0 else -1


def cocos_consistency_signs(equilibrium: Any) -> dict[str, int | None]:
    """Observed sign of each Eq. 23 quantity, or ``None`` where not determinable.

    ``dp/dpsi`` is taken as the bulk slope ``(p_edge - p_axis)/(psi_edge -
    psi_axis)`` rather than a pointwise derivative, which is what Sauter
    recommends: pressure is much larger on axis than at the edge, so the overall
    slope is the meaningful sign even where the profile is not monotonic.
    """
    eq = equilibrium
    observed: dict[str, int | None] = {
        "f": _sign(getattr(eq, "f", None)),
        "q": _sign(getattr(eq, "q", None)),
        "j_phi": _sign(getattr(eq, "j_phi", None)),
        "phi_tor": _sign(getattr(eq, "phi_tor", None)),
        "dpsi": None,
        "pprime": None,
    }

    psi_axis, psi_boundary = getattr(eq, "psi_axis", None), getattr(eq, "psi_boundary", None)
    delta_psi = None
    if psi_axis is not None and psi_boundary is not None:
        delta_psi = float(psi_boundary) - float(psi_axis)
        observed["dpsi"] = _sign(delta_psi)

    pprime = getattr(eq, "pprime", None)
    if pprime is not None:
        observed["pprime"] = _sign(pprime)
    else:
        pressure, psi_1d = getattr(eq, "pressure", None), getattr(eq, "psi_1d", None)
        if pressure is not None and psi_1d is not None and delta_psi:
            pressure = np.asarray(pressure, dtype=float).reshape(-1)
            psi_1d = np.asarray(psi_1d, dtype=float).reshape(-1)
            if pressure.size == psi_1d.size and pressure.size >= 2:
                # Order axis-to-edge so the slope is taken in a known direction.
                order = np.argsort((psi_1d - float(psi_axis)) / delta_psi)
                span = float(psi_1d[order][-1] - psi_1d[order][0])
                if span:
                    observed["pprime"] = _sign(
                        (float(pressure[order][-1]) - float(pressure[order][0])) / span
                    )
    return observed


def validate_cocos(
    equilibrium: Any, cocos: int | None = None, *,
    sigma_ip: int | None = None, sigma_b0: int | None = None,
) -> ValidationReport:
    """Check ``equilibrium`` against the Eq. 23 relations for ``cocos``.

    ``cocos`` defaults to the index recorded on the equilibrium's convention.
    ``sigma_ip``/``sigma_b0`` default to the signs of ``ip`` and ``bt0``.

    Returns a report; it never raises on an inconsistency, so a caller can decide
    whether a mismatch is fatal.  Relations whose inputs are unavailable are
    reported once as a single ``cocos_unverifiable`` warning rather than one
    issue each.
    """
    issues: list[ValidationIssue] = []

    if cocos is None:
        convention = getattr(equilibrium, "convention", None)
        cocos = getattr(convention, "cocos", None)
        if cocos is None:
            candidates = tuple(getattr(convention, "candidates", ()) or ())
            if len(candidates) == 1:
                cocos = candidates[0]
    if cocos is None:
        issues.append(ValidationIssue(
            "error", "cocos_undeclared", "convention",
            "no COCOS index is declared or uniquely identified, so the sign "
            "relations cannot be checked; pass cocos= explicitly",
        ))
        return ValidationReport(tuple(issues))

    spec = cocos_spec(int(cocos))
    if sigma_ip is None:
        sigma_ip = _sign(getattr(equilibrium, "ip", None))
    if sigma_b0 is None:
        sigma_b0 = _sign(getattr(equilibrium, "bt0", None))
    if sigma_ip is None or sigma_b0 is None:
        missing = ", ".join(
            name for name, value in (("ip", sigma_ip), ("bt0", sigma_b0)) if value is None
        )
        issues.append(ValidationIssue(
            "warning", "cocos_unverifiable", "convention",
            f"the sign of {missing} is unavailable, so the COCOS {cocos} sign "
            "relations cannot be checked",
        ))
        return ValidationReport(tuple(issues))

    observed = cocos_consistency_signs(equilibrium)
    unverifiable: list[str] = []
    for quantity, code, field, label in _RELATIONS:
        seen = observed.get(quantity)
        if seen is None:
            unverifiable.append(label)
            continue
        expected = spec.expected_sign(quantity, sigma_ip=sigma_ip, sigma_b0=sigma_b0)
        if seen == expected:
            continue
        # Codes commonly emit abs(q); Sauter Sect. IV says warn, do not reject.
        severity = "warning" if quantity == "q" else "error"
        issues.append(ValidationIssue(
            severity, code, field,
            f"COCOS {cocos} requires sign({label}) = {expected:+d} for "
            f"sigma_Ip={sigma_ip:+d}, sigma_B0={sigma_b0:+d}, but it is {seen:+d}",
        ))
    if unverifiable:
        issues.append(ValidationIssue(
            "warning", "cocos_unverifiable", "convention",
            f"COCOS {cocos} relations not checked because their inputs are "
            f"unavailable: {', '.join(unverifiable)}",
        ))
    return ValidationReport(tuple(issues))


def identify_flux_exponent(equilibrium: Any) -> tuple[int | None, float | None]:
    """Decide whether psi is stored in weber or weber/radian, from Ampere's law.

    Returns ``(e_Bp, ratio)``.  ``e_Bp`` is 0 for a weber-per-radian psi
    (COCOS 1-8) and 1 for a weber psi (COCOS 11-18); ``None`` when the inputs
    needed are unavailable.

    The loop integral of the poloidal field around the LCFS equals ``mu0*|Ip|``.
    Computing that field from psi as if it were weber-per-radian therefore gives
    a ratio of 1 when the assumption holds and 2*pi when psi is really in weber.
    The two outcomes differ by a factor of 2*pi, so the test is decisive.

    This replaces the ``a`` argument of :func:`omas.identify_cocos`, which is not
    usable here.  That routine evaluates a cylindrical estimate
    ``pi*B0*(a[i]-a[0])**2/(psi[i]-psi[0])`` at ``i = argmin|q|`` -- the node
    adjacent to the axis for a monotonic q.  Near the axis ``a`` goes as
    ``sqrt(psi)``, so any linear reconstruction of the minor-radius profile
    underestimates ``a[1]`` badly, and the estimate depends on its square.  On
    the packaged VEST sample it selects the wrong family on a margin of 0.22
    against 1.67, where the loop integral separates 1.004 from 6.31.
    """
    eq = equilibrium
    ip = getattr(eq, "ip", None)
    lcfs = getattr(eq, "lcfs", None)
    if ip is None or not ip or lcfs is None or getattr(lcfs, "r", None) is None:
        return None, None
    if eq.r is None or eq.z is None or eq.psi is None:
        return None, None
    if lcfs.r.size < 3 or eq.psi.shape != (eq.r.size, eq.z.size):
        return None, None

    from scipy.constants import mu_0 as MU0

    from vaft.process.equilibrium import poloidal_field_at_boundary

    r_b = np.r_[lcfs.r, lcfs.r[0]]
    z_b = np.r_[lcfs.z, lcfs.z[0]]
    try:
        # cocos=None is the k = -1, weber-per-radian form; only |B_p| matters here.
        b_p, _, _ = poloidal_field_at_boundary(eq.r, eq.z, eq.psi, r_b, z_b, cocos=None)
    except Exception:
        return None, None
    length = np.hypot(np.diff(r_b), np.diff(z_b))
    loop = float(np.sum(0.5 * (np.asarray(b_p)[:-1] + np.asarray(b_p)[1:]) * length))
    expected = MU0 * abs(float(ip))
    if not expected or not np.isfinite(loop):
        return None, None
    ratio = loop / expected
    if not np.isfinite(ratio) or ratio <= 0:
        return None, None
    return (0 if abs(ratio - 1.0) < abs(ratio - 2.0 * np.pi) else 1), float(ratio)


def identify_convention(
    equilibrium: Any, *, clockwise_phi: bool | None = None,
) -> tuple[int, ...]:
    """Candidate COCOS indices for ``equilibrium``, from its observable signs.

    The sign family (which of the eight orientations) comes from
    :func:`omas.identify_cocos`, which reads it off sign(Ip), sign(B0),
    sign(q) and sign(dpsi).  The remaining freedom -- whether psi carries the
    2*pi -- is settled by :func:`identify_flux_exponent` rather than by the
    ``a`` argument of ``identify_cocos``; see that function for why.

    ``clockwise_phi`` distinguishes odd from even indices and is a fact about
    the machine, not about the data.  Without it, both are returned.
    """
    eq = equilibrium
    if eq.bt0 is None or eq.ip is None or eq.q is None or eq.psi_1d is None:
        return ()
    psi_1d = np.asarray(eq.psi_1d, dtype=float).reshape(-1)
    q = np.asarray(eq.q, dtype=float).reshape(-1)
    if psi_1d.size < 2 or q.size != psi_1d.size:
        return ()

    # identify_cocos reads sign(gradient(psi))[0], so the profile has to run
    # axis to edge.  A boundary-first profile would invert sigma_Bp silently.
    if eq.psi_axis is not None and abs(psi_1d[0] - float(eq.psi_axis)) > abs(
        psi_1d[-1] - float(eq.psi_axis)
    ):
        psi_1d, q = psi_1d[::-1], q[::-1]

    from omas import identify_cocos

    try:
        candidates = {
            int(value)
            for value in identify_cocos(eq.bt0, eq.ip, q, psi_1d, clockwise_phi=clockwise_phi)
        }
    except Exception:
        return ()
    if not candidates:
        return ()

    exponent, _ = identify_flux_exponent(eq)
    if exponent is not None:
        wanted = range(1, 9) if exponent == 0 else range(11, 19)
        narrowed = {value for value in candidates if value in wanted}
        if narrowed:
            candidates = narrowed
    return tuple(sorted(candidates))
