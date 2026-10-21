"""COCOS consistency: an equilibrium against the Sauter Eq. 23 sign relations.

Whether an equilibrium's signs are consistent with the COCOS index it claims is
a verification question -- *representation consistency* -- so it lives in the
validation layer (#253, #337).  The measurements it compares are made in
:mod:`vaft.process.cocos` (:func:`~vaft.process.cocos.cocos_consistency_signs`,
which also documents the six relations), and the per-convention expected signs
come from :meth:`vaft.data.cocos.CocosSpec.expected_sign`.  Nothing here
identifies or converts a convention.

:func:`vaft.process.cocos.validate_cocos` remains as a deprecated alias.
"""

from __future__ import annotations

from typing import Any

from vaft.data.cocos import cocos_spec
from vaft.data.equilibrium import ValidationIssue, ValidationReport
from vaft.process.cocos import _sign, cocos_consistency_signs

__all__ = ["validate_cocos"]

#: Relations Eq. 23 defines, in report order, with the field each one inspects.
_RELATIONS = (
    ("f", "cocos_sign_f", "f", "F = R*B_phi"),
    ("dpsi", "cocos_sign_dpsi", "psi_boundary", "psi_boundary - psi_axis"),
    ("pprime", "cocos_sign_pprime", "pressure", "dp/dpsi"),
    ("q", "cocos_sign_q", "q", "q"),
    ("j_phi", "cocos_sign_jphi", "j_phi", "toroidal current density"),
    ("phi_tor", "cocos_sign_phi_tor", "phi_tor", "toroidal flux"),
)


def validate_cocos(
    equilibrium: Any, cocos: int | None = None, *,
    sigma_ip: int | None = None, sigma_b0: int | None = None,
) -> ValidationReport:
    """Check an equilibrium against the Sauter Eq. 23 relations for one COCOS index.

    The check that catches a mislabelled file before its signs propagate into
    derived quantities.  Never raises on an inconsistency: it returns a report and
    lets the caller decide whether a mismatch is fatal.

    Parameters
    ----------
    equilibrium : EquilibriumData
        The equilibrium to check [-].
    cocos : int, optional
        The convention to check against, 1 to 18.  Defaults to the index recorded
        on the equilibrium's own ``convention`` [-].
    sigma_ip : int, optional
        Sign of the plasma current, ``+1`` or ``-1``.  Defaults to the sign of the
        equilibrium's ``ip`` [-].
    sigma_b0 : int, optional
        Sign of the vacuum toroidal field.  Defaults to the sign of ``bt0`` [-].

    Returns
    -------
    ValidationReport
        One issue per violated relation, plus at most one
        ``cocos_unverifiable`` warning covering every relation whose inputs were
        missing.  An empty report means every checkable relation held [-].

    Processing steps
    ----------------
    1. Resolve the target index and the two reference signs from the arguments,
       falling back to the equilibrium's own convention and field signs.
    2. Measure the observed signs with :func:`cocos_consistency_signs`.
    3. Compare each against the sign Eq. 23 requires for that index, collecting
       the mismatches.
    4. Collapse every unverifiable relation into a single warning rather than
       one issue each.

    Convention
    ----------
    Checks consistency *with* a convention; it does not identify one and does not
    convert. A mismatch on ``q`` is reported at warning severity rather than
    error, because codes commonly emit ``abs(q)`` and a sign disagreement there is
    not on its own evidence of a wrong index.  Every other relation is an error.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Only four of Eq. 23's six relations are checkable against
    :class:`~vaft.data.equilibrium.EquilibriumData` as it stands; ``j_phi`` and
    ``phi_tor`` have no field to read and are always unverifiable.  A passing
    report therefore means "consistent as far as the stored fields can say", not
    "the index is correct".

    Provenance
    ----------
    .. [1] Sauter and Medvedev (2013), Eq. 23 for the relations and Sect. IV for
       the rule that a ``q`` sign mismatch is a warning, not a rejection.
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
