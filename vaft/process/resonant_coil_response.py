"""Resonant-field response of coil-group phasors: a verified linear operator and its optima.

On a fixed equilibrium, toroidal mode ``n`` and plasma-response model, the
complex resonant flux GPEC writes at the rational surfaces (``Phi_res`` total,
``Phi_res_v`` vacuum; area-normalized, tesla) responds linearly to the coil
excitation.  This module holds that operator, built from per-group unit runs
and **verified** on held-out runs rather than assumed, and the optimizations
built on it.  It runs no solver.

    resonant_response_operator ─▶ H from the unit runs e_k, Phi = H conj(c)
    qualify_response_operator ──▶ held-out complex runs -> qualified / rejected / insufficient
    window_weights ─────────────▶ W: mean |Phi|^2 over a resonant window (RMS^2)
    ResonantResponseOperator.quadratic(W) ─▶ Q with J = c^H Q c, for the generic optima of
                                    vaft.process.ntv_coil_response (two-group, budgeted, fixed-amplitude)
    match_resonant_target ──────▶ stage B: regularized complex least squares, with rank and SVD
    constrained_window_extremum ─▶ stage D: maximize one window under a cap on another and a budget

Notation
--------
K        : number of independently driven coil groups                     [-]
N_s      : number of rational surfaces                                    [-]
c        : VAFT excitation phasor per group, c = A exp(+i delta)        [phasor]
Phi      : complex resonant flux per surface, area-normalized              [T]
H        : N_s x K operator, Phi = H conj(c)                        [T / phasor]
W        : N_s x N_s Hermitian positive-semidefinite surface weights        [-]
J        : quadratic resonant metric, (H conj c)^H W (H conj c) = c^H Q c  [T^2]
D        : K x K Hermitian positive-definite excitation metric             [-]

Conventions
-----------
**The response is linear in conj(c).**  VAFT's excitation
(:data:`vaft.process.coil_operating_space.COIL_EXCITATION_CONVENTION`) is
``I_k = A cos(n phi_k + delta) = Re(c exp(+i n phi_k))``; GPEC decomposes in
``exp(-i n phi)``, whose coefficient is ``conj(c) / 2``.  Measured on the
DIII-D-like example: ``Phi(i e_k) = -i Phi(e_k)`` exactly, and
``Phi = H conj(c)`` reproduces every held-out run to 4e-6.  Hence ``Q =
conj(H^H W H)`` for a quadratic metric in ``c``; using ``H^H W H`` instead
would mirror every optimal relative phase.

**Squared, not RMS, is optimized.**  ``J`` with :func:`window_weights` is the
window's mean squared magnitude; its square root is
:func:`vaft.process.perturbation.rms_resonant_field` for the same surfaces,
so the ordering is the same for a fixed non-empty window.  Results report
which one they hold.

**Total and vacuum stay apart.**  ``Phi_res`` and ``Phi_res_v`` get separate
operators; nothing here subtracts them.

Provenance
----------
.. [1884] Issue #1884, resonant-response optimization (stages A-D).
.. [1178] Issue #1178, the coil operating-space umbrella and its result schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

__all__ = [
    "RESONANT_FIELDS",
    "ResonantResponseOperator",
    "ResponseOperatorQualification",
    "constrained_window_extremum",
    "match_resonant_target",
    "qualify_response_operator",
    "resonant_operating_space",
    "resonant_response_operator",
    "window_weights",
]

#: GPEC's rational-surface fluxes this operator may represent.
RESONANT_FIELDS = ("Phi_res", "Phi_res_v")


@dataclass(frozen=True)
class ResonantResponseOperator:
    """``Phi = H conj(c)`` for one resonant field on fixed surfaces."""

    #: ``Phi_res`` (total) or ``Phi_res_v`` (vacuum).
    field: str
    #: ``(N_s, K)`` complex operator [T / phasor].
    H: np.ndarray
    #: Normalized poloidal flux of each rational surface [-].
    psi_n_rational: np.ndarray
    #: Poloidal mode number of each surface [-].
    m_rational: np.ndarray
    #: Toroidal mode number [-].
    n: int
    #: Coil-group names, in phasor order [-].
    groups: tuple[str, ...]
    #: Cosine amplitude of one phasor unit [A per turn].
    amplitude_unit: float
    provenance: Mapping[str, object] = field(default_factory=dict)

    def response(self, phasors) -> np.ndarray:
        """Complex ``Phi`` per surface for one ``(K,)`` or ``(N, K)`` excitation [T]."""
        c = np.asarray(phasors, dtype=complex)
        if c.shape[-1] != self.H.shape[1]:
            raise ValueError(f"phasors of length {c.shape[-1]} for {self.H.shape[1]} groups")
        return np.conj(c) @ self.H.T

    def quadratic(self, weights) -> np.ndarray:
        """``Q`` with ``(H conj c)^H W (H conj c) = c^H Q c``: ``conj(H^H W H)`` [T^2 / phasor^2]."""
        w = np.asarray(weights, dtype=complex)
        if w.shape != (self.H.shape[0],) * 2:
            raise ValueError(f"weights must be {self.H.shape[0]} x {self.H.shape[0]}")
        q = np.conj(self.H.conj().T @ w @ self.H)
        return 0.5 * (q + q.conj().T)


@dataclass(frozen=True)
class ResponseOperatorQualification:
    """Whether ``Phi = H conj(c)`` reproduces runs it was not built from."""

    status: str
    #: ``||Phi_model - Phi_direct|| / ||Phi_direct||`` per compared run [-].
    relative_errors: Mapping[str, float]
    reasons: tuple[str, ...]


def resonant_response_operator(
    unit_responses: Sequence[Sequence[complex]],
    *,
    field: str,
    psi_n_rational: Sequence[float],
    m_rational: Sequence[int],
    n: int,
    groups: Sequence[str],
    amplitude_unit: float,
    provenance: Mapping[str, object] | None = None,
) -> ResonantResponseOperator:
    """The operator ``H`` from the responses to the unit excitations ``c = e_k``.

    Parameters
    ----------
    unit_responses : sequence of sequence of complex
        ``Phi`` of the run with ``c = e_k`` (group ``k`` at one phasor unit,
        phase 0, the others off), one per group, in group order [T].
    field : str
        ``Phi_res`` or ``Phi_res_v`` [-].
    psi_n_rational : sequence of float
        Surface positions, the same in every run [-].
    m_rational : sequence of int
        Poloidal mode number per surface [-].
    n : int
        Toroidal mode number [-].
    groups : sequence of str
        Coil-group names in phasor order [-].
    amplitude_unit : float
        Cosine amplitude of one phasor unit [A per turn].
    provenance : mapping, optional
        Equilibrium, code version, run identities [-].

    Returns
    -------
    operator : ResonantResponseOperator
        ``H[:, k] = Phi(e_k)``, so ``Phi = H conj(c)`` [-].

    Raises
    ------
    ValueError
        Unknown field, mismatched lengths, a non-finite response, or no group.

    Convention
    ----------
    Because ``e_k`` is real, ``Phi(e_k) = H e_k``; the conjugate shows only
    for complex excitations.  :func:`qualify_response_operator` is what
    establishes the relation for complex ``c``.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Built from ``K`` runs; trustworthy only once qualified on held-out complex
    runs at the same equilibrium, ``n`` and numerical settings.

    Provenance
    ----------
    .. [1884] Issue #1884, the verified linear operator.
    """
    if field not in RESONANT_FIELDS:
        raise ValueError(f"field must be one of {RESONANT_FIELDS}, got {field!r}")
    columns = [np.asarray(r, dtype=complex).ravel() for r in unit_responses]
    psi = np.asarray(psi_n_rational, dtype=float).ravel()
    m = np.asarray(m_rational, dtype=int).ravel()
    if not columns or len(columns) != len(groups):
        raise ValueError(f"{len(columns)} unit responses for {len(groups)} groups")
    if any(col.shape != psi.shape for col in columns) or m.shape != psi.shape:
        raise ValueError("every unit response must have one value per rational surface")
    h = np.column_stack(columns)
    if not np.all(np.isfinite(h)):
        raise ValueError("a unit response is not finite")
    if not amplitude_unit > 0:
        raise ValueError("amplitude_unit must be positive")
    return ResonantResponseOperator(field=field, H=h, psi_n_rational=psi, m_rational=m, n=int(n),
                                    groups=tuple(str(g) for g in groups), amplitude_unit=float(amplitude_unit),
                                    provenance=dict(provenance or {}))


def qualify_response_operator(
    operator: ResonantResponseOperator,
    phasors: np.ndarray,
    responses: Sequence[Sequence[complex]],
    *,
    sample_ids: Sequence[str],
    rtol: float,
) -> ResponseOperatorQualification:
    """Qualify or reject ``Phi = H conj(c)`` against held-out runs.

    Parameters
    ----------
    operator : ResonantResponseOperator
        From :func:`resonant_response_operator` [-].
    phasors : numpy.ndarray
        ``(N, K)`` excitations not among the unit runs, complex ones included [phasor].
    responses : sequence of sequence of complex
        Their direct ``Phi``; an all-NaN row marks a failed run [T].
    sample_ids : sequence of str
        One id per run [-].
    rtol : float
        Largest accepted ``||Phi_model - Phi_direct|| / ||Phi_direct||`` [-].

    Returns
    -------
    qualification : ResponseOperatorQualification
        ``rejected`` when any run exceeds ``rtol``; ``insufficient`` when any
        failed or none was given, or no run had a non-real phasor (the
        conjugate relation would be untested); otherwise ``qualified`` [-].

    Raises
    ------
    ValueError
        Shapes disagree or ``rtol`` is negative.

    Processing steps
    ----------------
    1. Set aside failed (non-finite) runs and count them.
    2. Relative vector error of ``H conj(c)`` against each direct ``Phi``.
    3. One status, requiring at least one run with a non-real phasor.

    Convention
    ----------
    The comparison is on the complex vector, phase included; matching the
    magnitudes alone would not catch the ``conj`` relation.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [1884] Issue #1884, independent combined-current validation.
    """
    if not rtol >= 0:
        raise ValueError("rtol must not be negative")
    c = np.atleast_2d(np.asarray(phasors, dtype=complex))
    direct = np.atleast_2d(np.asarray(responses, dtype=complex))
    ids = [str(s) for s in sample_ids]
    if c.shape != (len(ids), operator.H.shape[1]) or direct.shape != (len(ids), operator.H.shape[0]):
        raise ValueError("phasors, responses and sample_ids disagree with the operator")
    finite = np.all(np.isfinite(direct), axis=1)
    model = operator.response(c[finite])
    errors = {i: float(np.linalg.norm(m - d) / np.linalg.norm(d))
              for i, m, d in zip(np.asarray(ids)[finite], model, direct[finite])}
    reasons: list[str] = []
    failed = [i for i, e in errors.items() if e > rtol]
    if failed:
        status = "rejected"
        reasons.append(f"relative error above {rtol:g}: {failed}")
    elif not finite.all() or not finite.any():
        status = "insufficient"
        reasons.append(f"{int((~finite).sum())} run(s) failed; the operator is identified, not validated")
    elif not np.any(np.abs(c[finite].imag) > 0):
        status = "insufficient"
        reasons.append("no held-out run had a non-real phasor, so Phi = H conj(c) is untested")
    else:
        status = "qualified"
    return ResponseOperatorQualification(status=status, relative_errors=errors, reasons=tuple(reasons))


def window_weights(operator: ResonantResponseOperator, window) -> np.ndarray:
    """Surface weights whose quadratic metric is the window's mean squared ``|Phi|``.

    Parameters
    ----------
    operator : ResonantResponseOperator
        Supplies the surface positions [-].
    window : vaft.process.perturbation.ResonantWindow
        From :func:`vaft.process.perturbation.resonant_windows` [-].

    Returns
    -------
    W : numpy.ndarray
        Diagonal ``N_s x N_s``, ``1 / N_in`` on the surfaces inside [-].

    Raises
    ------
    ValueError
        The window holds no surface: an empty window is not a zero response.

    Convention
    ----------
    Endpoints included, as :meth:`ResonantWindow.mask`; a surface on the
    core/edge boundary is in both.  ``sqrt(c^H Q c)`` with ``Q =
    operator.quadratic(W)`` is :func:`vaft.process.perturbation.rms_resonant_field`.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [1884] Issue #1884, Sec. 3 (core/edge metrics).
    """
    inside = window.mask(operator.psi_n_rational)
    if not inside.any():
        raise ValueError(f"the {window.name} window [{window.low:g}, {window.high:g}] holds no rational surface")
    return np.diag(inside / inside.sum()).astype(float)


def match_resonant_target(
    operator: ResonantResponseOperator,
    target: Sequence[complex],
    *,
    weights: np.ndarray,
    regularization: float,
    metric: np.ndarray,
) -> dict[str, object]:
    """Stage B: the excitation whose resonant response best matches a complex target.

    Parameters
    ----------
    operator : ResonantResponseOperator
        A qualified operator [-].
    target : sequence of complex
        Target ``Phi`` per surface, phase included [T].
    weights : numpy.ndarray
        Hermitian positive-semidefinite surface weights ``W`` [-].
    regularization : float
        ``lambda >= 0`` of the excitation penalty [T^2 / phasor^2].
    metric : numpy.ndarray
        Hermitian positive-definite excitation metric ``D`` of the penalty [-].

    Returns
    -------
    solution : dict
        ``c`` [phasor]; ``response`` [T]; ``residual`` ``||W^1/2 (H conj c - t)||``
        and ``relative_residual`` against ``||W^1/2 t||`` [T, -];
        ``singular_values`` of ``W^1/2 H`` [T / phasor], ``rank`` and
        ``condition`` (largest over smallest *nonzero* singular value; finite
        even when rank deficient) [-]; ``null_space`` (columns of excitations with no
        weighted response) [phasor].

    Raises
    ------
    ValueError
        Shapes disagree, ``W`` or ``D`` is not Hermitian, ``D`` is not
        positive definite, or ``regularization`` is negative.

    Processing steps
    ----------------
    1. With ``g = conj(c)``: minimize ``||W^1/2 (H g - t)||^2 + lambda g^H conj(D) g``.
    2. Solve the stacked system ``[W^1/2 H; sqrt(lambda) L^H] g = [W^1/2 t; 0]``
       (``conj(D) = L L^H``) by least squares, never by an explicit inverse.
    3. ``c = conj(g)``; report the SVD of ``W^1/2 H`` and its null space.

    Convention
    ----------
    The target is complex: its phase is physical here, so no group's phase is
    fixed.  With ``lambda = 0`` and a rank-deficient ``W^1/2 H`` the returned
    ``c`` is the minimum-norm one, and the null space says what else fits.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    A phasor-space answer: actual sector currents, winding signs and circuit
    limits must be checked separately.  A zero target with ``lambda > 0``
    returns ``c = 0``; ask for a nontrivial target instead.

    Provenance
    ----------
    .. [1884] Issue #1884, Sec. 4 stage B.
    """
    h = operator.H
    t = np.asarray(target, dtype=complex).ravel()
    w = _hermitian_psd(weights, "weights")
    d = _hermitian_psd(metric, "metric")
    if t.shape != (h.shape[0],) or w.shape != (h.shape[0],) * 2 or d.shape != (h.shape[1],) * 2:
        raise ValueError("target, weights or metric do not match the operator")
    if not regularization >= 0:
        raise ValueError("regularization must not be negative")
    if np.linalg.eigvalsh(d).min() <= 0:
        raise ValueError("metric must be positive definite")
    w_half = _sqrt_psd(w)
    a = w_half @ h
    rhs = w_half @ t
    if regularization > 0:
        lower = np.linalg.cholesky(np.conj(d))
        a_full = np.vstack([a, np.sqrt(regularization) * lower.conj().T])
        rhs_full = np.concatenate([rhs, np.zeros(h.shape[1], dtype=complex)])
    else:
        a_full, rhs_full = a, rhs
    g, *_ = np.linalg.lstsq(a_full, rhs_full, rcond=None)
    u, s, vh = np.linalg.svd(a)
    tol = s.max(initial=0.0) * max(a.shape) * np.finfo(float).eps
    rank = int(np.sum(s > tol))
    residual = float(np.linalg.norm(a @ g - rhs))
    norm_t = float(np.linalg.norm(rhs))
    c = np.conj(g)
    return {
        "c": c,
        "response": operator.response(c),
        "residual": residual,
        "relative_residual": residual / norm_t if norm_t > 0 else float("nan"),
        "singular_values": s,
        "rank": rank,
        "condition": float(s[0] / s[rank - 1]) if rank else float("inf"),
        "null_space": np.conj(vh[rank:].conj().T),
    }


def constrained_window_extremum(
    Q_objective: np.ndarray,
    Q_constraint: np.ndarray,
    *,
    metric: np.ndarray,
    budget: float,
    limit: float,
    tolerance: float = 1e-10,
) -> dict[str, object]:
    """Stage D: maximize ``c^H Q_o c`` with ``c^H Q_c c <= limit`` and ``c^H D c = budget``.

    Parameters
    ----------
    Q_objective : numpy.ndarray
        Hermitian positive-semidefinite metric to maximize, e.g. the edge
        window's ``operator.quadratic(W_edge)`` [T^2 / phasor^2].
    Q_constraint : numpy.ndarray
        Hermitian positive-semidefinite metric to cap, e.g. the core window's [T^2 / phasor^2].
    metric : numpy.ndarray
        Hermitian positive-definite excitation metric ``D`` [-].
    budget : float
        ``P_0 > 0`` of ``c^H D c = P_0`` [phasor^2].
    limit : float
        Cap on ``c^H Q_c c`` [T^2].
    tolerance : float, optional
        Relative tolerance of the multiplier bisection; a numerical-method
        control [-].

    Returns
    -------
    solution : dict
        ``c`` [phasor], ``objective`` and ``constraint`` values [T^2],
        ``multiplier`` ``mu >= 0`` [-], and ``status``: ``unconstrained``
        (the cap is not active), ``active`` (the constraint sits on the cap),
        ``at_floor`` (only the least-constraint excitations meet it; the
        multiplier diverges) or ``infeasible`` (none does; ``c`` is then the
        least-constraint excitation) [-].

    Raises
    ------
    ValueError
        Shapes disagree, a matrix is not Hermitian, ``D`` is not positive
        definite, or ``budget``/``limit`` is not positive / negative.

    Processing steps
    ----------------
    1. ``infeasible`` when even the least ``c^H Q_c c`` on the budget (the
       bottom generalized eigenvector of ``(Q_c, D)``) exceeds the cap.
    2. For a multiplier ``mu``, the top generalized eigen*space* of
       ``(Q_o - mu Q_c, D)``; ``mu = 0`` if some excitation in it meets the cap.
    3. Otherwise bisect ``mu`` on the least constraint within that space, and
       at the converged ``mu`` combine the space's extreme directions so the
       constraint lands on the cap (the objective is then maximal).

    Convention
    ----------
    Squared metrics; report ``sqrt`` for an RMS.  The answer is defined up to
    a common phase.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    The multiplier route is exact for two Hermitian quadratic forms on a
    complex sphere (no duality gap); degenerate top eigenvalues, which occur
    whenever the cap's null space lies inside the objective's (e.g. more
    groups than capped surfaces), are handled by working on the eigenspace.
    Eigenvalues within ``1e-9`` of the spectral scale are treated as equal.
    Sector-current limits are not imposed.

    Provenance
    ----------
    .. [1884] Issue #1884, Sec. 4 stage D.
    """
    from scipy.linalg import eigh

    qo = _hermitian_psd(Q_objective, "Q_objective")
    qc = _hermitian_psd(Q_constraint, "Q_constraint")
    d = _hermitian_psd(metric, "metric")
    if not (qo.shape == qc.shape == d.shape) or qo.shape[0] != qo.shape[1]:
        raise ValueError("Q_objective, Q_constraint and metric must be square and of one size")
    if not budget > 0 or not limit >= 0:
        raise ValueError("budget must be positive and limit not negative")
    if np.linalg.eigvalsh(d).min() <= 0:
        raise ValueError("metric must be positive definite")
    scale = max(float(np.max(np.abs(qo))), float(np.max(np.abs(qc))), np.finfo(float).tiny)
    slack = 1e-10 * float(np.max(np.abs(qc))) * budget  # round-off of a constraint value

    def objective(c):
        return float(np.real(c.conj() @ qo @ c))

    def constraint(c):
        return float(np.real(c.conj() @ qc @ c))

    def top_space(mu):
        # The top generalized eigenspace, not one eigenvector: when the cap's null
        # space lies inside the objective's, the top eigenvalue is degenerate at the
        # active multiplier and a single eigenvector can sit far below the cap.
        values, vectors = eigh(qo - mu * qc, d)  # D-orthonormal columns
        keep = values >= values[-1] - 1e-9 * scale * max(1.0, mu)
        return vectors[:, keep]

    def pick(space, cap):
        """Budget-scaled excitation in ``space`` with the largest constraint <= cap (or the least, if none)."""
        m = space.conj().T @ qc @ space
        values, vectors = np.linalg.eigh(0.5 * (m + m.conj().T))
        low, high = values[0] * budget, values[-1] * budget
        if low > cap + slack:
            a = vectors[:, 0]
        elif high <= cap + slack:
            a = vectors[:, -1]
        else:  # mix the extremes to land on the cap
            share = float(np.clip((cap - low) / (high - low), 0.0, 1.0))
            a = np.sqrt(share) * vectors[:, -1] + np.sqrt(1.0 - share) * vectors[:, 0]
        return space @ (a * np.sqrt(budget / float(np.real(a.conj() @ a))))

    def least(mu):
        m = top_space(mu)
        mm = m.conj().T @ qc @ m
        return float(np.linalg.eigvalsh(0.5 * (mm + mm.conj().T))[0]) * budget

    def result(c, mu, status):
        return {"c": c, "objective": objective(c), "constraint": constraint(c), "multiplier": mu, "status": status}

    floor_vector = eigh(qc, d)[1][:, 0]
    floor_vector = floor_vector * np.sqrt(budget / float(np.real(floor_vector.conj() @ d @ floor_vector)))
    if constraint(floor_vector) > limit + slack:
        return result(floor_vector, float("nan"), "infeasible")
    free = pick(top_space(0.0), limit)
    if constraint(free) <= limit + slack:
        return result(free, 0.0, "unconstrained")
    # The bisection itself uses the exact cap; the slack only decides feasibility.
    low, high = 0.0, 1.0
    while least(high) > limit:
        high *= 2.0
        if high > 1e12:
            # Only the least-constraint excitations meet the cap: the multiplier diverges.
            return result(floor_vector, float("inf"), "at_floor")
    while high - low > tolerance * max(high, 1.0):
        mid = 0.5 * (low + high)
        low, high = (mid, high) if least(mid) > limit else (low, mid)
    return result(pick(top_space(high), limit), high, "active")


def resonant_operating_space(
    groups: Sequence[str],
    n: int,
    phasors: np.ndarray,
    responses: Mapping[str, np.ndarray],
    *,
    psi_n_rational: Sequence[float],
    windows: Mapping[str, object],
    sample_ids: Sequence[str],
    roles: Sequence[str],
    amplitude_unit: float,
    frame: str,
    provenance: Mapping[str, object],
    operators: Mapping[str, tuple[ResonantResponseOperator, ResponseOperatorQualification]] | None = None,
):
    """Direct resonant-field RMS per window (and a qualified operator's predictions) as a coil operating-space result.

    Parameters
    ----------
    groups : sequence of str
        Coil-group names in phasor order [-].
    n : int
        Toroidal mode number [-].
    phasors : numpy.ndarray
        ``(N, K)`` excitations [phasor].
    responses : mapping of str to numpy.ndarray
        ``(N, N_s)`` direct complex flux per field, keyed ``Phi_res`` and/or
        ``Phi_res_v``; a row of NaN marks a failed run [T].
    psi_n_rational : sequence of float
        Surface positions [-].
    windows : mapping of str to ResonantWindow
        Windows to reduce over, e.g. ``core`` and ``edge`` of
        :func:`vaft.process.perturbation.resonant_windows` [-].
    sample_ids : sequence of str
        One id per sample [-].
    roles : sequence of str
        Role per sample [-].
    amplitude_unit : float
        Cosine amplitude of one phasor unit [A per turn].
    frame : str
        The toroidal frame of the sector angles [-].
    provenance : mapping
        Equilibrium, code, pedestal source [-].
    operators : mapping, optional
        Per field, a ``(operator, qualification)`` pair; a ``qualified`` one
        adds ``<metric>_model`` rows [-].

    Returns
    -------
    result : vaft.process.coil_operating_space.CoilOperatingSpace
        Metrics ``phi_res_rms_<window>`` / ``phi_res_v_rms_<window>`` [T] and
        their ``_model`` counterparts [-].

    Raises
    ------
    ValueError
        Lengths disagree, an unknown field, or an operator that is not qualified.

    Processing steps
    ----------------
    1. Per sample, field and window, :func:`vaft.process.perturbation.rms_resonant_field`
       of the direct response; ``failed`` (NaN) for a failed run, ``undefined``
       for a window with no surface.
    2. With a qualified operator, the same reduction of ``H conj(c)``.
    3. Window bounds and sources go into provenance.

    Convention
    ----------
    Reported statistic is the RMS (tesla), the square root of the optimized
    mean square; measured and modelled values never share a metric.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [1884] Issue #1884; [1886] the result schema's ``phi_res_rms_edge`` naming.
    """
    import pandas as pd

    from .coil_operating_space import COIL_EXCITATION_CONVENTION, CoilOperatingSpace
    from .perturbation import rms_resonant_field

    c = np.atleast_2d(np.asarray(phasors, dtype=complex))
    psi = np.asarray(psi_n_rational, dtype=float)
    ids, role = [str(s) for s in sample_ids], [str(r) for r in roles]
    if not (c.shape[0] == len(ids) == len(role)) or c.shape[1] != len(groups):
        raise ValueError("phasors, sample_ids, roles and groups disagree in length")
    samples = pd.DataFrame({"sample_id": ids, "role": role})
    for index, group in enumerate(groups):
        samples[f"amplitude_{group}"] = np.abs(c[:, index]) * float(amplitude_unit)
        samples[f"phase_{group}"] = np.mod(np.angle(c[:, index]), 2 * np.pi)
    rows, record = [], {}
    unused = set(operators or {}) - set(responses)
    if unused:
        raise ValueError(f"operators given for fields without responses: {sorted(unused)}")
    for field, values in responses.items():
        if field not in RESONANT_FIELDS:
            raise ValueError(f"unknown field {field!r}")
        values = np.atleast_2d(np.asarray(values, dtype=complex))
        if values.shape != (len(ids), psi.size):
            raise ValueError(f"{field} responses must be {len(ids)} x {psi.size}")
        prefix = field.lower()
        model = None
        if operators and field in operators:
            operator, qualification = operators[field]
            if qualification.status != "qualified":
                raise ValueError(f"the {field} operator is {qualification.status!r}; it adds no prediction rows")
            model = operator.response(c)
        for name, window in windows.items():
            metric = f"{prefix}_rms_{name}"
            record[metric] = {**dict(provenance), "field": field, "window": [window.low, window.high],
                              "window_source": window.source, "statistic": "rms over the window's surfaces"}
            empty = not window.mask(psi).any()
            for i, sample in enumerate(ids):
                ok = bool(np.all(np.isfinite(values[i])))
                status = "undefined" if empty else ("valid" if ok else "failed")
                value = rms_resonant_field(psi, values[i], window=window) if status == "valid" else np.nan
                rows.append({"sample_id": sample, "metric": metric, "value": value, "unit": "T", "status": status})
                if model is not None:
                    rows.append({"sample_id": sample, "metric": f"{metric}_model", "unit": "T",
                                 "value": rms_resonant_field(psi, model[i], window=window) if not empty else np.nan,
                                 "status": "undefined" if empty else "valid"})
            if model is not None:
                record[f"{metric}_model"] = {
                    **record[metric], "model": "Phi = H conj(c)", "qualification": operators[field][1].status,
                    "held_out_errors": dict(operators[field][1].relative_errors),
                    "note": "predicted for every sample, including the unit runs H was built from and runs "
                            "whose direct evaluation failed",
                }
    return CoilOperatingSpace(
        n=n, groups=tuple(groups), samples=samples, values=pd.DataFrame(rows),
        convention={**COIL_EXCITATION_CONVENTION, "frame": frame}, grid={"kind": "irregular"}, provenance=record,
    )


def _hermitian_psd(matrix, name: str) -> np.ndarray:
    m = np.asarray(matrix, dtype=complex)
    if m.ndim != 2 or m.shape[0] != m.shape[1] or not np.all(np.isfinite(m)):
        raise ValueError(f"{name} must be a finite square matrix")
    scale = max(float(np.max(np.abs(m))), np.finfo(float).tiny)
    if not np.allclose(m, m.conj().T, rtol=0.0, atol=1e-9 * scale):
        raise ValueError(f"{name} is not Hermitian")
    m = 0.5 * (m + m.conj().T)
    if np.linalg.eigvalsh(m).min() < -1e-9 * scale:
        raise ValueError(f"{name} is not positive semidefinite")
    return m


def _sqrt_psd(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * np.sqrt(np.clip(values, 0.0, None))) @ vectors.conj().T
