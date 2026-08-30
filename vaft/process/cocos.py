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

__all__ = ["cocos_consistency_signs", "validate_cocos"]

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
