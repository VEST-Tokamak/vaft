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

Only four of the six are checkable against :class:`~vaft.data.equilibrium.EquilibriumData`
as it stands: it carries no toroidal current density and no toroidal flux, so
``j_phi`` and ``Phi_tor`` are always reported as unverifiable.  They are listed
here because they are part of Eq. 23 and become checkable if those fields are
ever added, not because they are being tested today.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vaft.data.cocos import cocos_spec
from vaft.data.equilibrium import ValidationIssue, ValidationReport

__all__ = [
    "FLUX_EXPONENT_TOLERANCE", "cocos_consistency_signs", "identify_convention",
    "identify_flux_exponent", "validate_cocos",
]

#: Relative band around 1 and 2*pi within which the Ampere ratio is accepted.
#: A correct equilibrium lands within about half a percent -- the residual is
#: LCFS discretization -- so 15% is roughly thirty times the observed numerical
#: error while still rejecting anything a fifth of the way to the other answer.
#: The two outcomes are a factor 2*pi apart, so the bands stay far apart:
#: [0.85, 1.15] against [5.34, 7.23], with everything between an abstention.
FLUX_EXPONENT_TOLERANCE = 0.15

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
    """Observed sign of each Sauter Eq. 23 quantity, or ``None`` where not determinable.

    Reads signs off the equilibrium as it stands.  Nothing is compared against a
    convention here and nothing is converted; :func:`validate_cocos` does the
    comparison and :func:`vaft.process.equilibrium.convert_cocos` is the only
    transform in the package.

    Parameters
    ----------
    equilibrium : EquilibriumData
        The equilibrium to inspect.  Missing fields yield ``None`` for the
        relations that need them rather than an error [-].

    Returns
    -------
    dict of str to int or None
        Keys ``f``, ``q``, ``j_phi``, ``phi_tor``, ``dpsi``, ``pprime``; each
        ``+1``, ``-1``, or ``None`` when the field is absent or all-zero [-].

    Processing steps
    ----------------
    1. ``f``, ``q``, ``j_phi`` and ``phi_tor``: the sign of the profile's median,
       with a deadband so a numerically-zero profile reports ``None``.
    2. ``dpsi``: the sign of ``psi_boundary - psi_axis``.
    3. ``pprime``: the stored ``pprime`` profile's sign when the equilibrium
       carries one, which for a GEQDSK or an ODS it essentially always does.
       Only when it does not is the bulk slope
       ``(p_edge - p_axis)/(psi_edge - psi_axis)`` used instead, taken on a
       profile reordered axis-to-edge so the direction is known.

    Convention
    ----------
    Signs only, in whatever convention the equilibrium is already in; this
    function neither assumes nor imposes a COCOS.  ``j_phi`` and ``phi_tor`` are
    always ``None`` today because :class:`~vaft.data.equilibrium.EquilibriumData`
    carries neither a toroidal current density nor a toroidal flux; they are
    reported so that Eq. 23 is represented in full, not because they are tested.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    A bulk slope is the sign of the overall trend, not of every point: a
    non-monotonic pressure profile still reports one sign.  Because the stored
    ``pprime`` takes precedence, the fallback rarely runs in practice; that the
    two paths can disagree for a hand-built equilibrium is tracked in #605.

    Provenance
    ----------
    .. [1] Sauter and Medvedev, *Tokamak Coordinate Conventions: COCOS*, Comput.
       Phys. Commun. 184, 293 (2013), Eq. 23, which defines the six sign
       relations this function measures.  The preference for a bulk pressure
       slope over a pointwise derivative is that paper's own recommendation.
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


def identify_flux_exponent(equilibrium: Any) -> tuple[int | None, float | None]:
    """Decide whether psi is stored in weber or weber per radian, from Ampere's law.

    The loop integral of the poloidal field around the last closed flux surface
    equals ``mu0*|Ip|``.  Computing that field from psi *as if* it were weber per
    radian gives a ratio of 1 when the assumption holds and ``2*pi`` when psi is
    really in weber, so the two answers are a factor ``2*pi`` apart and the test is
    decisive rather than a threshold on a continuum.

    Parameters
    ----------
    equilibrium : EquilibriumData
        Must carry ``ip``, a closed ``lcfs``, and a ``psi`` map on its ``r`` and
        ``z`` grid; anything missing yields ``(None, None)`` [-].

    Returns
    -------
    tuple of (int or None, float or None)
        ``(e_Bp, ratio)``.  ``e_Bp`` is 0 for a weber-per-radian psi (COCOS 1-8)
        and 1 for a weber psi (COCOS 11-18), or ``None`` when the inputs are
        unavailable or the ratio lands near neither answer.  ``ratio`` is the
        measured loop integral over ``mu0*|Ip|`` [-].

    Processing steps
    ----------------
    1. Close the LCFS contour and compute the poloidal field on it with
       :func:`vaft.process.equilibrium.poloidal_field_at_boundary` in the
       weber-per-radian form; only the magnitude matters.
    2. Integrate it along the contour by the trapezoidal rule.
    3. Divide by ``mu0*|Ip|``.
    4. Accept 0 or 1 if the ratio is within
       :data:`FLUX_EXPONENT_TOLERANCE` of 1 or ``2*pi``; otherwise abstain.

    Defaults
    --------
    :data:`FLUX_EXPONENT_TOLERANCE` is 0.15, a numerical convenience rather than
    a physical threshold: a correct equilibrium lands within about half a
    percent, the residual being LCFS discretization, so the band is roughly
    thirty times the observed error while leaving the two acceptance windows
    far apart, at 0.85 to 1.15 against 5.34 to 7.23.

    Convention
    ----------
    Decides which storage family psi is in; it does not rescale psi and does not
    say which of the eight sign orientations applies.  ``e_Bp`` is Sauter's flux
    exponent: 0 means the ``2*pi`` is not carried in psi, 1 means it is.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Abstains rather than picking the nearer of two wrong answers.  A ratio near
    neither value is evidence the input is broken, a truncated LCFS or an ``ip``
    that disagrees with the psi map, not evidence of a convention.  Needs a
    2-D psi map and an LCFS, so a profiles-only equilibrium cannot be classified.

    Provenance
    ----------
    .. [1] Sauter and Medvedev (2013) for ``e_Bp`` and the storage families.
    .. [2] Replaces the ``a`` argument of ``omas.identify_cocos``, which is not
       usable here: it evaluates ``pi*B0*(a[i]-a[0])**2/(psi[i]-psi[0])`` at the
       node adjacent to the axis, where the minor radius goes as ``sqrt(psi)``, so
       a linear reconstruction underestimates it and the estimate depends on its
       square.  On the packaged VEST sample that picks the wrong family on a
       margin of 0.22 against 1.67, where this loop integral separates 1.004 from
       6.31.
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

    # The whole value of the test is that the two answers are 2*pi apart, so a
    # ratio that is near neither is evidence the input is broken -- a truncated
    # LCFS, an Ip that disagrees with the psi map, a rescaled psi -- not evidence
    # of a convention.  Abstain rather than pick the nearer of two wrong answers.
    for exponent, expected_ratio in ((0, 1.0), (1, 2.0 * np.pi)):
        if abs(ratio - expected_ratio) <= FLUX_EXPONENT_TOLERANCE * expected_ratio:
            return exponent, float(ratio)
    return None, float(ratio)


def identify_convention(
    equilibrium: Any, *, clockwise_phi: bool | None = None,
) -> tuple[int, ...]:
    """Candidate COCOS indices for an equilibrium, from its observable signs.

    Narrows rather than decides: the return is every index consistent with what
    the data shows, which is often more than one.  A caller that needs a single
    index supplies the missing fact instead of guessing.

    Parameters
    ----------
    equilibrium : EquilibriumData
        Must carry ``bt0``, ``ip``, ``q`` and ``psi_1d``; without them the
        candidate set is empty [-].
    clockwise_phi : bool, optional
        Whether the machine's toroidal angle runs clockwise seen from above.  A
        fact about the machine, not about the data; without it both the odd and
        the even index of each pair are returned [-].

    Returns
    -------
    tuple of int
        The candidate COCOS indices in increasing order, empty when the required
        fields are missing or the sign identification fails [-].

    Processing steps
    ----------------
    1. Reorder ``psi_1d`` and ``q`` axis-to-edge when the stored profile runs the
       other way.
    2. Get the sign family from ``omas.identify_cocos``, which reads it off the
       signs of ``ip``, ``bt0``, ``q`` and the psi gradient.
    3. Narrow to 1-8 or 11-18 using :func:`identify_flux_exponent`, keeping the
       full set when that abstains.

    Convention
    ----------
    Returns indices in the standard 1 to 18 numbering; 9 and 10 do not exist.
    The sign family and the storage family are identified by different evidence,
    signs for the first and an Ampere loop integral for the second, and either can
    be inconclusive on its own.  What each index means is declared in
    :mod:`vaft.data.cocos`, not here.

    Applicability
    -------------
    Machine-independent.  ``clockwise_phi`` is where a machine's own geometry
    enters.

    Limitations
    -----------
    The axis-to-edge reorder exists because ``identify_cocos`` reads the psi
    gradient at the first node, so a boundary-first profile would invert the
    poloidal-flux sign silently.  When the flux exponent abstains the result spans
    both storage families, and a caller that then converts must not assume the
    candidates share one.  Any exception from the underlying identification is
    reported as no candidates rather than raised.

    Provenance
    ----------
    .. [1] Sauter and Medvedev (2013) for the index numbering and the sign
       relations the identification rests on.
    .. [2] ``omas.identify_cocos`` supplies the sign family; its flux-exponent
       argument is deliberately not used, see :func:`identify_flux_exponent`.
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
