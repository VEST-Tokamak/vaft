"""Quadratic coil-response model of a signed NTV torque, and its qualification.

A non-axisymmetric coil set with ``K`` independently driven groups is described
by one complex phasor per group, ``c_k = A_k exp(+i phi_k)``.  On a *fixed*
plasma and kinetic background, the signed toroidal torque PENTRC computes from
the GPEC displacement that excitation drives may -- or may not -- be a
Hermitian quadratic form of the phasors (issue #1887):

    T(c) = c^H Q c,        Q = Q^H.

This module holds the algebra for testing that hypothesis and, only where it
holds, for using it.  It runs no solver: each torque it takes is a number some
GPEC -> PENTRC evaluation produced (:func:`vaft.code.gpec.torque_profile`), and
each phasor is the excitation that evaluation was run with.

    quadratic_probe_design ──▶ K^2 probes  (e_i, e_i+e_j, e_i+i e_j)
    hermitian_torque_matrix ─▶ Q from the probes' torques
    quadratic_invariance_checks ─▶ T(a c) = a^2 T(c), T(-c) = T(c), T(e^{i chi} c) = T(c)
    qualify_torque_matrix ───▶ held-out error -> "qualified" / "rejected" / "insufficient"
    fit_inhomogeneous_torque ─▶ T0 + 2 Re(b^H c) + c^H Q c, to see whether T0 and b vanish
    two_group_phase_extrema, budgeted_torque_extrema, fixed_amplitude_phase_extrema

A rejected hypothesis is a result, not a failure of this module: the direct
scan it was tested on is then the operating-space study, and no optimizer here
should be fed that ``Q``.

Notation
--------
K       : number of independently driven coil groups                       [-]
c       : complex excitation phasor per group, c_k = A_k exp(+i phi_k)  [phasor]
A_k     : declared amplitude normalization of group k (not a sector-peak
          current unless the caller declares it so)                       [phasor]
phi_k   : phase of group k                                                 [rad]
T       : signed scalar torque of one evaluation (e.g. the enclosed edge
          value of PENTRC's real(sum_ell T))                               [N m]
Q       : Hermitian torque matrix, T = c^H Q c                      [N m / phasor^2]
D       : Hermitian positive-definite excitation metric of a budget c^H D c [-]

Conventions
-----------
**Phasor sign.**  ``c_k = A_k exp(+i phi_k)`` and ``T = c^H Q c``.  The
cross-term recovery below depends on it: with the opposite phasor sign the
imaginary part of every ``Q_ij`` changes sign.

**Signed torque.**  ``T`` keeps PENTRC's sign; nothing here takes ``|T|``.  A
failed or uncomputed evaluation is NaN, never zero.

**Common phase.**  For a single toroidal mode, ``T(e^{i chi} c) = T(c)``, so a
``K``-group phase optimum has ``K - 1`` free phases; group 0's phase is held
at zero.

Provenance
----------
.. [1887] Issue #1887, the PENTRC NTV torque phase/amplitude study and the
   probe identities used here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np

__all__ = [
    "NTV_QUADRATIC_STATUSES",
    "QuadraticTorqueCheck",
    "TorqueMatrixQualification",
    "budgeted_torque_extrema",
    "coil_phasors",
    "fit_inhomogeneous_torque",
    "fixed_amplitude_phase_extrema",
    "hermitian_torque_matrix",
    "quadratic_invariance_checks",
    "quadratic_probe_design",
    "quadratic_torque",
    "qualify_torque_matrix",
    "two_group_phase_extrema",
]

#: What :func:`qualify_torque_matrix` can conclude.
NTV_QUADRATIC_STATUSES = ("qualified", "rejected", "insufficient")


@dataclass(frozen=True)
class QuadraticTorqueCheck:
    """One test of the quadratic hypothesis against direct evaluations."""

    name: str
    passed: bool
    #: Largest ``|T_compared - T_expected|`` over the samples [N m].
    max_abs_error: float
    #: Largest ``|T_compared - T_expected| / |T_expected|`` (``inf`` at zero) [-].
    max_rel_error: float
    samples: int


@dataclass(frozen=True)
class TorqueMatrixQualification:
    """Whether a reconstructed ``Q`` reproduces evaluations it was not built from."""

    status: str
    Q: np.ndarray
    #: Eigenvalues of ``Q``, ascending; mixed signs mean an indefinite ``Q`` [N m / phasor^2].
    eigenvalues: np.ndarray
    checks: tuple[QuadraticTorqueCheck, ...]
    reasons: tuple[str, ...]
    provenance: Mapping[str, object] = field(default_factory=dict)


def coil_phasors(amplitudes: Sequence[float], phases: Sequence[float]) -> np.ndarray:
    """Complex excitation phasors ``A_k exp(+i phi_k)`` of the coil groups.

    Parameters
    ----------
    amplitudes : sequence of float
        Declared amplitude normalization of each group, not negative [phasor].
    phases : sequence of float
        Phase of each group [rad].

    Returns
    -------
    c : numpy.ndarray
        Complex phasor per group [phasor].

    Raises
    ------
    ValueError
        The two sequences differ in length or an amplitude is negative or not
        finite.

    Convention
    ----------
    ``c_k = A_k exp(+i phi_k)``; the sign of the exponent is the one
    :func:`hermitian_torque_matrix` recovers cross terms in.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [1887] Issue #1887, phasor convention.
    """
    a = np.asarray(amplitudes, dtype=float).ravel()
    phi = np.asarray(phases, dtype=float).ravel()
    if a.shape != phi.shape:
        raise ValueError(f"{a.size} amplitudes for {phi.size} phases")
    if not np.all(np.isfinite(a)) or np.any(a < 0):
        raise ValueError("amplitudes must be finite and not negative")
    return a * np.exp(1j * phi)


def quadratic_probe_design(groups: int) -> tuple[tuple[str, np.ndarray], ...]:
    """The ``K^2`` unit excitations that determine a Hermitian ``Q``.

    Parameters
    ----------
    groups : int
        Number of independently driven coil groups ``K``, at least 1 [-].

    Returns
    -------
    probes : tuple of (str, numpy.ndarray)
        ``("e0", e_0)``, ..., then for every ``i < j`` ``("e{i}+e{j}", e_i + e_j)``
        and ``("e{i}+ie{j}", e_i + i e_j)``: ``K + 2 C(K, 2) = K^2`` phasors [phasor].

    Raises
    ------
    ValueError
        ``groups`` is below 1.

    Convention
    ----------
    The mixed probes put ``+i`` on the higher-index group, matching the
    ``exp(+i phi)`` phasor of :func:`coil_phasors`.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    ``K^2`` evaluations identify ``Q`` only; validating it needs further
    held-out evaluations (:func:`qualify_torque_matrix`).

    Provenance
    ----------
    .. [1887] Issue #1887, Sec. 6 (probe identities and their count).
    """
    k = int(groups)
    if k < 1:
        raise ValueError("at least one coil group is needed")
    unit = np.eye(k, dtype=complex)
    probes = [(f"e{i}", unit[i]) for i in range(k)]
    for i, j in combinations(range(k), 2):
        probes.append((f"e{i}+e{j}", unit[i] + unit[j]))
        probes.append((f"e{i}+ie{j}", unit[i] + 1j * unit[j]))
    return tuple(probes)


def hermitian_torque_matrix(torques: Mapping[str, float], groups: int) -> np.ndarray:
    """The Hermitian ``Q`` of ``T = c^H Q c`` from the probe torques.

    Parameters
    ----------
    torques : mapping of str to float
        Signed torque of each probe of :func:`quadratic_probe_design`, keyed
        by its label [N m].
    groups : int
        Number of coil groups ``K`` [-].

    Returns
    -------
    Q : numpy.ndarray
        ``K x K`` Hermitian matrix [N m / phasor^2].

    Raises
    ------
    KeyError
        A probe label is missing.
    ValueError
        A probe torque is not finite (a failed evaluation cannot identify ``Q``).

    Processing steps
    ----------------
    1. ``Q_ii = T(e_i)``.
    2. ``Re Q_ij = [T(e_i + e_j) - T(e_i) - T(e_j)] / 2``.
    3. ``Im Q_ij = [T(e_i) + T(e_j) - T(e_i + i e_j)] / 2``.
    4. ``Q_ji = conj(Q_ij)``.

    Convention
    ----------
    With ``c = e_i + i e_j``, ``c^H Q c = Q_ii + Q_jj - 2 Im Q_ij``, which is
    step 3; it holds for the ``exp(+i phi)`` phasor of :func:`coil_phasors`.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    A ``Q`` that reproduces its own probes exactly is not thereby validated:
    the probes determine it.  Only :func:`qualify_torque_matrix` on held-out
    evaluations can qualify it.

    Provenance
    ----------
    .. [1887] Issue #1887, Sec. 6.
    """
    k = int(groups)
    labels = [label for label, _ in quadratic_probe_design(k)]
    values = {label: float(torques[label]) for label in labels}
    bad = sorted(label for label, value in values.items() if not np.isfinite(value))
    if bad:
        raise ValueError(f"probe torques are not finite: {bad}")
    q = np.zeros((k, k), dtype=complex)
    for i in range(k):
        q[i, i] = values[f"e{i}"]
    for i, j in combinations(range(k), 2):
        real = 0.5 * (values[f"e{i}+e{j}"] - values[f"e{i}"] - values[f"e{j}"])
        imag = 0.5 * (values[f"e{i}"] + values[f"e{j}"] - values[f"e{i}+ie{j}"])
        q[i, j] = real + 1j * imag
        q[j, i] = real - 1j * imag
    return q


def quadratic_torque(Q: np.ndarray, phasors: np.ndarray) -> np.ndarray:
    """Signed torque ``c^H Q c`` of one excitation or a stack of them.

    Parameters
    ----------
    Q : numpy.ndarray
        ``K x K`` Hermitian torque matrix [N m / phasor^2].
    phasors : numpy.ndarray
        One phasor vector of length ``K``, or an ``(N, K)`` stack [phasor].

    Returns
    -------
    torque : numpy.ndarray
        Real signed torque, scalar or length ``N`` [N m].

    Raises
    ------
    ValueError
        ``Q`` is not square and Hermitian, or the phasors do not match it.

    Convention
    ----------
    ``T = c^H Q c`` with ``c_k = A_k exp(+i phi_k)``; the imaginary part,
    zero for a Hermitian ``Q``, is dropped after the check.

    Applicability
    -------------
    Machine-independent.
    """
    q = _hermitian(Q)
    c = np.asarray(phasors, dtype=complex)
    if c.shape[-1] != q.shape[0]:
        raise ValueError(f"phasors of length {c.shape[-1]} for a {q.shape[0]}-group Q")
    return np.real(np.einsum("...i,ij,...j->...", c.conj(), q, c))


def quadratic_invariance_checks(
    base_torque: float,
    *,
    scaled: Sequence[tuple[float, float]] = (),
    reversed_torque: float | None = None,
    rotated: Sequence[tuple[float, float]] = (),
    atol: float,
    rtol: float,
) -> tuple[QuadraticTorqueCheck, ...]:
    """Tests of a pure quadratic response that need no ``Q``: scaling, sign, common phase.

    Parameters
    ----------
    base_torque : float
        Signed torque of the reference excitation ``c`` [N m].
    scaled : sequence of (float, float)
        ``(a, T(a c))`` for positive real factors ``a`` [-, N m].
    reversed_torque : float, optional
        ``T(-c)`` [N m].
    rotated : sequence of (float, float)
        ``(chi, T(exp(i chi) c))`` for common phase rotations [rad, N m].
    atol : float
        Absolute tolerance, the scale below which a torque is near zero [N m].
    rtol : float
        Relative tolerance away from zero [-].

    Returns
    -------
    checks : tuple of QuadraticTorqueCheck
        ``amplitude_scaling`` (expects ``a^2 T(c)``), ``sign_reversal``
        (expects ``T(c)``) and ``common_phase`` (expects ``T(c)``), each only
        when its samples were given [-].

    Raises
    ------
    ValueError
        A tolerance is negative, a factor ``a`` is not positive, or a torque
        is not finite.

    Processing steps
    ----------------
    1. Expected torque per sample from the homogeneous-quadratic hypothesis.
    2. A sample passes when ``|T - T_expected| <= atol + rtol |T_expected|``.
    3. One check per kind, passed only if every sample of that kind passes.

    Convention
    ----------
    The common-phase test assumes a single toroidal mode ``n``; a mixed-``n``
    excitation is not phase-invariant and should not be given here.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    A failed check rejects the *pure* quadratic form; a constant or linear
    term (a background field) is examined by :func:`fit_inhomogeneous_torque`.
    The tolerances are the caller's: no universal NTV tolerance exists.

    Provenance
    ----------
    .. [1887] Issue #1887, Sec. 5.
    """
    _tolerances(atol, rtol)
    base = _finite(base_torque, "base_torque")
    checks: list[QuadraticTorqueCheck] = []
    if scaled:
        factors = np.array([float(a) for a, _ in scaled])
        if np.any(factors <= 0):
            raise ValueError("scaling factors must be positive")
        values = np.array([_finite(t, "scaled torque") for _, t in scaled])
        checks.append(_check("amplitude_scaling", values, factors**2 * base, atol, rtol))
    if reversed_torque is not None:
        checks.append(_check("sign_reversal", np.array([_finite(reversed_torque, "reversed_torque")]),
                             np.array([base]), atol, rtol))
    if rotated:
        values = np.array([_finite(t, "rotated torque") for _, t in rotated])
        checks.append(_check("common_phase", values, np.full(values.shape, base), atol, rtol))
    return tuple(checks)


def qualify_torque_matrix(
    Q: np.ndarray,
    held_out_phasors: np.ndarray,
    held_out_torques: Sequence[float],
    *,
    atol: float,
    rtol: float,
    invariance_checks: Sequence[QuadraticTorqueCheck] = (),
) -> TorqueMatrixQualification:
    """Qualify or reject a ``Q`` against direct evaluations it was not built from.

    Parameters
    ----------
    Q : numpy.ndarray
        Hermitian torque matrix, e.g. from :func:`hermitian_torque_matrix` [N m / phasor^2].
    held_out_phasors : numpy.ndarray
        ``(N, K)`` excitations not among the probes that built ``Q`` [phasor].
    held_out_torques : sequence of float
        Their direct signed torques; NaN marks a failed evaluation [N m].
    atol : float
        Absolute tolerance, the near-zero torque scale [N m].
    rtol : float
        Relative tolerance away from zero [-].
    invariance_checks : sequence of QuadraticTorqueCheck, optional
        From :func:`quadratic_invariance_checks`; any failure rejects [-].

    Returns
    -------
    qualification : TorqueMatrixQualification
        ``status`` is ``qualified`` only when every held-out evaluation and
        every given invariance check passes; ``insufficient`` with no finite
        held-out evaluation; otherwise ``rejected`` with the reasons [-].

    Raises
    ------
    ValueError
        Shapes disagree, ``Q`` is not Hermitian, or a tolerance is negative.

    Processing steps
    ----------------
    1. Drop held-out samples whose direct torque is NaN, and say how many.
    2. Compare ``c^H Q c`` with the direct torque per sample, passing when
       ``|dT| <= atol + rtol |T_direct|``.
    3. Combine with the invariance checks into one status.

    Convention
    ----------
    The comparison is on the signed torque; a model that matches ``|T|`` but
    not its sign fails.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Qualification holds for the background, ``n``, PENTRC method, grid and
    amplitude range the evaluations covered, and nowhere else.

    Provenance
    ----------
    .. [1887] Issue #1887, Sec. 6 (held-out validation).
    """
    _tolerances(atol, rtol)
    q = _hermitian(Q)
    c = np.atleast_2d(np.asarray(held_out_phasors, dtype=complex))
    t = np.asarray(held_out_torques, dtype=float).ravel()
    if c.shape != (t.size, q.shape[0]):
        raise ValueError(f"held-out phasors {c.shape} do not match {t.size} torques and a {q.shape[0]}-group Q")
    finite = np.isfinite(t)
    reasons: list[str] = []
    checks = list(invariance_checks)
    if (~finite).any():
        reasons.append(f"{int((~finite).sum())} held-out evaluation(s) failed and were not compared")
    if finite.any():
        checks.append(_check("held_out", quadratic_torque(q, c[finite]), t[finite], atol, rtol))
    failed = [check.name for check in checks if not check.passed]
    if failed:
        status = "rejected"
        reasons.append(f"failed: {', '.join(failed)}")
    elif not finite.any():
        status = "insufficient"
        reasons.append("no finite held-out evaluation: Q is identified, not validated")
    else:
        status = "qualified"
    return TorqueMatrixQualification(
        status=status,
        Q=q,
        eigenvalues=np.linalg.eigvalsh(q),
        checks=tuple(checks),
        reasons=tuple(reasons),
        provenance={"atol": float(atol), "rtol": float(rtol), "held_out": int(finite.sum())},
    )


def fit_inhomogeneous_torque(phasors: np.ndarray, torques: Sequence[float]) -> dict[str, object]:
    """Least-squares ``T = T0 + 2 Re(b^H c) + c^H Q c``, to test whether ``T0`` and ``b`` vanish.

    Parameters
    ----------
    phasors : numpy.ndarray
        ``(N, K)`` excitations with finite direct torques [phasor].
    torques : sequence of float
        Their signed torques [N m].

    Returns
    -------
    fit : dict
        ``T0`` [N m], ``b`` (length ``K``) [N m / phasor], ``Q`` (Hermitian)
        [N m / phasor^2], ``residual_rms`` [N m], ``rank`` and ``unknowns`` [-].

    Raises
    ------
    ValueError
        Fewer finite samples than the ``1 + 2K + K^2`` real unknowns, or
        shapes disagree.

    Processing steps
    ----------------
    1. Real unknowns: ``T0``, ``Re b``, ``Im b``, ``Q_ii`` and, for ``i < j``,
       ``Re Q_ij`` and ``Im Q_ij``.
    2. Each sample contributes one real equation; solve by least squares.
    3. Report the residual and the rank so an under-determined design shows.

    Convention
    ----------
    ``2 Re(b^H c) = 2 (Re b . Re c + Im b . Im c)`` and
    ``c^H Q c = sum_i Q_ii |c_i|^2 + 2 sum_{i<j} Re(conj(c_i) Q_ij c_j)``.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    A nonzero ``T0`` or ``b`` means a background (an error field, another
    coil) is present; the pure form ``c^H Q c`` then does not apply as is.

    Provenance
    ----------
    .. [1887] Issue #1887, Sec. 5 (the inhomogeneous form tested first).
    """
    c = np.atleast_2d(np.asarray(phasors, dtype=complex))
    t = np.asarray(torques, dtype=float).ravel()
    if c.shape[0] != t.size:
        raise ValueError(f"{c.shape[0]} phasors for {t.size} torques")
    keep = np.isfinite(t)
    c, t = c[keep], t[keep]
    k = c.shape[1]
    pairs = list(combinations(range(k), 2))
    unknowns = 1 + 2 * k + k + 2 * len(pairs)
    if t.size < unknowns:
        raise ValueError(f"{t.size} finite samples for {unknowns} unknowns")
    columns = [np.ones(t.size), 2 * c.real.T, 2 * c.imag.T, (np.abs(c) ** 2).T]
    cross = []
    for i, j in pairs:
        product = c[:, i].conj() * c[:, j]
        cross.append(2 * product.real)    # coefficient of Re Q_ij
        cross.append(-2 * product.imag)   # coefficient of Im Q_ij
    design = np.column_stack([columns[0], *columns[1], *columns[2], *columns[3], *cross])
    solution, _, rank, _ = np.linalg.lstsq(design, t, rcond=None)
    t0 = float(solution[0])
    b = solution[1:1 + k] + 1j * solution[1 + k:1 + 2 * k]
    q = np.diag(solution[1 + 2 * k:1 + 3 * k]).astype(complex)
    for index, (i, j) in enumerate(pairs):
        value = solution[1 + 3 * k + 2 * index] + 1j * solution[2 + 3 * k + 2 * index]
        q[i, j], q[j, i] = value, np.conj(value)
    residual = t - design @ solution
    return {
        "T0": t0,
        "b": b,
        "Q": q,
        "residual_rms": float(np.sqrt(np.mean(residual**2))),
        "rank": int(rank),
        "unknowns": unknowns,
    }


def two_group_phase_extrema(Q: np.ndarray, amplitudes: Sequence[float]) -> dict[str, object]:
    """Relative phases of the largest and smallest signed torque of two groups.

    Parameters
    ----------
    Q : numpy.ndarray
        ``2 x 2`` Hermitian torque matrix of a qualified model [N m / phasor^2].
    amplitudes : sequence of float
        Fixed amplitudes ``(A_1, A_2)``, not negative [phasor].

    Returns
    -------
    extrema : dict
        ``delta_phi_max`` and ``delta_phi_min`` in ``[0, 2 pi)`` [rad], or
        ``None`` when the torque does not depend on the phase; ``T_max`` and
        ``T_min`` [N m]; ``phase_independent`` [-].

    Raises
    ------
    ValueError
        ``Q`` is not ``2 x 2`` Hermitian or an amplitude is negative.

    Processing steps
    ----------------
    1. ``T(dphi) = Q_11 A_1^2 + Q_22 A_2^2 + 2 A_1 A_2 |Q_12| cos(dphi + arg Q_12)``.
    2. Maximum at ``dphi = -arg Q_12``, minimum at ``pi - arg Q_12``.
    3. With ``Q_12 = 0`` or a zero amplitude no phase is preferred.

    Convention
    ----------
    ``c = (A_1, A_2 exp(+i dphi))``; the extrema are of the *signed* torque,
    not of ``|T|`` or of a braking criterion.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Valid only where :func:`qualify_torque_matrix` qualified ``Q``.

    Provenance
    ----------
    .. [1887] Issue #1887, Sec. 8.
    """
    q = _hermitian(Q)
    if q.shape != (2, 2):
        raise ValueError("two_group_phase_extrema needs a 2 x 2 Q")
    a1, a2 = (float(a) for a in amplitudes)
    if a1 < 0 or a2 < 0:
        raise ValueError("amplitudes must not be negative")
    base = float(np.real(q[0, 0])) * a1**2 + float(np.real(q[1, 1])) * a2**2
    swing = 2 * a1 * a2 * abs(q[0, 1])
    scale = max(abs(base), abs(q[0, 0]) * a1**2, abs(q[1, 1]) * a2**2, np.finfo(float).tiny)
    if swing <= 1e-12 * scale:
        return {"delta_phi_max": None, "delta_phi_min": None, "T_max": base, "T_min": base,
                "phase_independent": True}
    arg = float(np.angle(q[0, 1]))
    return {
        "delta_phi_max": float(np.mod(-arg, 2 * np.pi)),
        "delta_phi_min": float(np.mod(np.pi - arg, 2 * np.pi)),
        "T_max": base + swing,
        "T_min": base - swing,
        "phase_independent": False,
    }


def budgeted_torque_extrema(Q: np.ndarray, D: np.ndarray, budget: float) -> dict[str, object]:
    """Largest and smallest signed torque over free complex excitations with ``c^H D c = budget``.

    Parameters
    ----------
    Q : numpy.ndarray
        Hermitian torque matrix of a qualified model, possibly indefinite [N m / phasor^2].
    D : numpy.ndarray
        Hermitian positive-definite excitation metric [-].
    budget : float
        The excitation budget ``P_0 > 0`` [phasor^2].

    Returns
    -------
    extrema : dict
        ``T_max``/``T_min`` [N m], ``c_max``/``c_min`` meeting the budget
        [phasor], and the generalized ``eigenvalues`` ascending [N m / phasor^2].

    Raises
    ------
    ValueError
        ``D`` is not positive definite, shapes disagree, or the budget is not
        positive.

    Processing steps
    ----------------
    1. Solve ``Q v = lambda D v`` (generalized Hermitian eigenproblem).
    2. Scale the extreme eigenvectors to ``v^H D v = P_0``; ``T = lambda P_0``.

    Convention
    ----------
    Signed: with an indefinite ``Q`` the minimum is a negative torque, not a
    small magnitude.  The returned phasors are defined up to a common phase.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    A mathematical optimum of the phasor space: actual sector or circuit
    current limits, winding signs and fixed group amplitudes are not imposed
    and must be checked separately.

    Provenance
    ----------
    .. [1887] Issue #1887, Sec. 8 (free complex excitation with a budget).
    """
    from scipy.linalg import eigh

    q = _hermitian(Q)
    d = _hermitian(D)
    if d.shape != q.shape:
        raise ValueError("Q and D differ in shape")
    if not budget > 0:
        raise ValueError("the budget must be positive")
    if np.linalg.eigvalsh(d).min() <= 0:
        raise ValueError("D must be positive definite")
    values, vectors = eigh(q, d)

    def scaled(v):
        return v * np.sqrt(budget / float(np.real(v.conj() @ d @ v)))

    return {
        "T_max": float(values[-1] * budget),
        "T_min": float(values[0] * budget),
        "c_max": scaled(vectors[:, -1]),
        "c_min": scaled(vectors[:, 0]),
        "eigenvalues": values,
    }


def fixed_amplitude_phase_extrema(
    Q: np.ndarray,
    amplitudes: Sequence[float],
    *,
    grid_points: int,
    refine: int,
) -> dict[str, object]:
    """Phases of the largest and smallest signed torque with every group amplitude fixed.

    Parameters
    ----------
    Q : numpy.ndarray
        ``K x K`` Hermitian torque matrix of a qualified model [N m / phasor^2].
    amplitudes : sequence of float
        Fixed amplitude of each group, not negative [phasor].
    grid_points : int
        Phase samples per free phase of the deterministic grid, at least 4 [-].
    refine : int
        How many of the best grid points per extremum to polish with a local
        optimizer; 0 keeps the grid answer [-].

    Returns
    -------
    extrema : dict
        ``phases_max``/``phases_min`` (group 0 held at 0) [rad], ``T_max``/
        ``T_min`` [N m] and ``evaluations``, the model evaluations used [-].

    Raises
    ------
    ValueError
        Shapes disagree, an amplitude is negative, or ``grid_points < 4``.

    Processing steps
    ----------------
    1. Hold group 0's phase at 0 (common phase is irrelevant for one ``n``).
    2. Evaluate ``c^H Q c`` on a ``grid_points^(K-1)`` phase grid.
    3. Polish the ``refine`` best points of each extremum with L-BFGS-B on the
       free phases and keep the best.

    Convention
    ----------
    ``c_k = A_k exp(+i phi_k)``; extrema of the signed torque.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Nonconvex for ``K >= 3``: the grid plus local polish is a deterministic
    reference, not a proof of the global optimum at coarse grids.  The grid
    grows as ``grid_points^(K-1)``.

    Provenance
    ----------
    .. [1887] Issue #1887, Sec. 8 (multi-group fixed-amplitude optimization).
    """
    q = _hermitian(Q)
    a = np.asarray(amplitudes, dtype=float).ravel()
    if a.size != q.shape[0]:
        raise ValueError(f"{a.size} amplitudes for a {q.shape[0]}-group Q")
    if np.any(a < 0):
        raise ValueError("amplitudes must not be negative")
    if int(grid_points) < 4:
        raise ValueError("grid_points must be at least 4")
    k = a.size

    def torque(free):
        return float(quadratic_torque(q, coil_phasors(a, np.concatenate(([0.0], free)))))

    if k == 1:
        value = torque(np.empty(0))
        return {"phases_max": np.zeros(1), "phases_min": np.zeros(1), "T_max": value, "T_min": value,
                "evaluations": 1}
    axis = np.linspace(0.0, 2 * np.pi, int(grid_points), endpoint=False)
    mesh = np.stack(np.meshgrid(*([axis] * (k - 1)), indexing="ij"), axis=-1).reshape(-1, k - 1)
    phasors = a * np.exp(1j * np.concatenate((np.zeros((mesh.shape[0], 1)), mesh), axis=1))
    values = quadratic_torque(q, phasors)
    evaluations = values.size
    order = np.argsort(values)
    result = {}
    for name, sign, candidates in (("max", -1.0, order[::-1]), ("min", 1.0, order)):
        best_free, best_value = mesh[candidates[0]], float(values[candidates[0]])
        if refine > 0:
            from scipy.optimize import minimize

            for start in candidates[: int(refine)]:
                found = minimize(lambda x: sign * torque(x), mesh[start], method="L-BFGS-B")
                evaluations += int(found.nfev)
                value = sign * float(found.fun)
                if (value > best_value) if sign < 0 else (value < best_value):
                    best_free, best_value = np.mod(found.x, 2 * np.pi), value
        result[f"phases_{name}"] = np.concatenate(([0.0], best_free))
        result[f"T_{name}"] = best_value
    result["evaluations"] = evaluations
    return result


def _hermitian(matrix: np.ndarray) -> np.ndarray:
    q = np.asarray(matrix, dtype=complex)
    if q.ndim != 2 or q.shape[0] != q.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {q.shape}")
    scale = max(float(np.max(np.abs(q))), np.finfo(float).tiny)
    if not np.allclose(q, q.conj().T, rtol=0.0, atol=1e-12 * scale):
        raise ValueError("the matrix is not Hermitian")
    return 0.5 * (q + q.conj().T)


def _tolerances(atol: float, rtol: float) -> None:
    if not (atol >= 0 and rtol >= 0):
        raise ValueError("tolerances must not be negative")


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} is not finite")
    return value


def _check(name: str, compared: np.ndarray, expected: np.ndarray, atol: float, rtol: float) -> QuadraticTorqueCheck:
    error = np.abs(compared - expected)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(expected != 0, error / np.abs(expected), np.where(error == 0, 0.0, np.inf))
    return QuadraticTorqueCheck(
        name=name,
        passed=bool(np.all(error <= atol + rtol * np.abs(expected))),
        max_abs_error=float(error.max()) if error.size else 0.0,
        max_rel_error=float(relative.max()) if relative.size else 0.0,
        samples=int(error.size),
    )
