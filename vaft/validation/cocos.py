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

__all__ = ["check_cocos_transform", "validate_cocos"]

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


#: Relative tolerance on the quantities a COCOS transform must leave unchanged
#: in magnitude.  The transform multiplies by exact signs and powers of 2*pi, so
#: anything above rounding is a wrong factor, not noise.
TRANSFORM_RTOL = 1e-9


def check_cocos_transform(before: Any, after: Any) -> ValidationReport:
    """Check that a COCOS conversion preserved what no convention can change.

    The registry check ``verification.cocos_transform_invariants``: the
    transformation-axis evidence that a conversion was carried out correctly,
    as opposed to merely declared (#2165).

    Parameters
    ----------
    before : EquilibriumData
        The equilibrium in its source convention [-].
    after : EquilibriumData
        The same equilibrium after the conversion [-].

    Returns
    -------
    ValidationReport
        An error per broken invariant, a warning per invariant that could not
        be checked; empty when every checkable invariant held [-].

    Raises
    ------
    ValueError
        Either equilibrium carries no COCOS index.

    Processing steps
    ----------------
    1. Both conventions must carry an index.
    2. ``|I_p|``, ``|B_0|``, ``|q|``, ``|F|`` and the pressure are invariant
       in magnitude under any COCOS change.
    3. ``|psi_boundary - psi_axis|`` and the grid ``|psi - psi_axis|`` scale by
       ``k = (2*pi)**(e_Bp,after - e_Bp,before)``; ``|p'|`` and ``|FF'|`` by
       ``1/k``.
    4. The converted equilibrium satisfies the Sauter Eq. 23 sign relations of
       its new index, as far as :func:`validate_cocos` can check them.

    A missing or non-finite input is a warning, never a pass: the report then
    reads ``warn`` because something was not checked.

    Convention
    ----------
    Mathematical consistency of the transform only.  A passing report says the
    conversion was faithful to the source convention; it says nothing about
    whether that source convention was right, and so nothing about the VEST
    machine polarity (#288).

    Limitations
    -----------
    The standing ``cocos_unverifiable`` warning of :func:`validate_cocos`
    about ``j_phi`` and ``phi_tor``, which no ``EquilibriumData`` field can
    carry, is not repeated here; every other finding of it is, including a
    missing current or field sign that leaves no relation checkable.

    Provenance
    ----------
    .. [1] Sauter and Medvedev (2013), Table I (``e_Bp``) and Eq. 23.
    """
    import numpy as np

    from vaft.data.cocos import psi_is_per_radian

    issues: list[ValidationIssue] = []
    source = getattr(getattr(before, "convention", None), "cocos", None)
    target = getattr(getattr(after, "convention", None), "cocos", None)
    if source is None or target is None:
        raise ValueError("both equilibria must carry a COCOS index; a conversion that produced "
                         "an undeclared convention is not a conversion to check")

    exponent = int(not psi_is_per_radian(target)) - int(not psi_is_per_radian(source))
    flux_scale = (2.0 * np.pi) ** exponent

    def unverifiable(field: str, why: str) -> None:
        issues.append(ValidationIssue("warning", "cocos_transform_unverifiable", field,
                                      f"{field} {why}, so it is not checked"))

    def scaled(field: str, a, b, factor: float, label: str) -> None:
        # |b| == factor * |a| elementwise; a missing or non-finite input is never a pass
        if a is None or b is None:
            unverifiable(field, "is missing")
            return
        a, b = np.abs(np.asarray(a, dtype=float)), np.abs(np.asarray(b, dtype=float))
        if a.shape != b.shape:
            issues.append(ValidationIssue("error", f"cocos_transform_{field}", field,
                                          f"{field} changed shape under COCOS {source} -> {target}"))
            return
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            unverifiable(field, "holds non-finite values")
            return
        if not np.allclose(b, factor * a, rtol=TRANSFORM_RTOL, atol=0.0):
            issues.append(ValidationIssue("error", f"cocos_transform_{field}", field,
                                          f"{label} under COCOS {source} -> {target}"))

    for field in ("ip", "bt0", "q", "f", "pressure"):
        scaled(field, getattr(before, field, None), getattr(after, field, None), 1.0, f"|{field}| changed")

    def flux_from_axis(e, field):
        value, axis = getattr(e, field, None), getattr(e, "psi_axis", None)
        return None if value is None or axis is None else np.asarray(value, dtype=float) - float(axis)

    for field in ("psi_boundary", "psi"):
        scaled(field, flux_from_axis(before, field), flux_from_axis(after, field), flux_scale,
               f"the flux from the axis did not scale by {flux_scale:.12g}")
    for field in ("pprime", "ffprime"):
        scaled(field, getattr(before, field, None), getattr(after, field, None), 1.0 / flux_scale,
               f"d/dpsi did not scale by {1.0 / flux_scale:.12g}")

    standing = {"toroidal current density", "toroidal flux"}
    for issue in validate_cocos(after, target).issues:
        if issue.code == "cocos_unverifiable" and issue.message.startswith(f"COCOS {target} relations not checked"):
            if set(issue.message.split(": ", 1)[-1].split(", ")) <= standing:
                continue  # the two relations no EquilibriumData field can ever carry
        issues.append(issue)
    return ValidationReport(tuple(issues))
