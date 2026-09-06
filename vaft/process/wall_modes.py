"""Segment-wise eigenmode basis of a passive wall, with global coupling (vaft #473).

The reduced vessel-wall contract (VEST-Tokamak/vfit#8) in one line: **local
eigenbasis, global electromagnetic dynamics.**  The wall's current vector is
partitioned by physical segment, each segment gets its own L/R eigenbasis,
the bases are block-assembled into ``V_seg``, and every reduced operator is
then a projection of the *full* matrices -- so the mutual inductance between
segments survives in the off-diagonal blocks of the reduced inductance.

Naming trap, stated once.  Throughout :mod:`vaft.process.electromagnetics`
the passive-passive inductance is called ``M_mat`` and the passive-to-source
coupling ``L_mat``; this module keeps those argument names for compatibility
and says which is which at every signature.  In the formulas below ``L`` is
the physics inductance (code ``M_mat``), ``R`` the diagonal resistance (code
``R_mat``) and ``M`` the source coupling (code ``L_mat``).

Per segment ``g`` with diagonal ``R_gg`` and symmetric-definite ``L_gg``::

    S_g = R_gg^{-1/2} L_gg R_gg^{-1/2}          (symmetric)
    S_g q = tau q                                (eigh; tau ARE the L/R times)
    v     = R_gg^{-1/2} q                        (R-orthonormal)

so that ``v^T R_gg v = 1``, ``v^T L_gg v = tau`` and ``R_gg v = L_gg v / tau``.
The eigenvalues are the local decay times; modes are ordered by descending
``tau`` within each segment and the largest-|component| entry of every mode
is positive.  Assembled over segments (scattered by element index, so an
interleaved loop order needs no permutation)::

    V_seg = blockdiag(V_1, ..., V_G)             (N, M_tot)
    R_r   = V_seg^T R V_seg = I                  (computed and checked)
    L_r   = V_seg^T L V_seg                      diag blocks diag(tau_g), off-diagonal coupling
    M_r   = V_seg^T M                            (M_tot, n_src)
    G_red = G_full V_seg                         for any (n_obs, N) response

The projection is exact in the R inner product: ``a = V_seg^T R I_w``, which
for a truncated basis is the R-orthogonal (least-dissipation) projection.
Amplitudes carry units of sqrt(W): ``a^T a`` is the ohmic dissipation of the
wall current.  Conversions -- Euclidean-normalized modes have ``a_E = a
||v||_2``, inductance-normalized ones ``a_L = a sqrt(tau)``.

This module chooses no reduced order.  It returns every mode of every segment
and offers selection helpers; which modes to keep is the validation study's
question (vfit #10, vaft #494), and the tools for that question live here too:
the reduced circuit solve (:func:`solve_reduced_eddy`), rankings of the modes
by response rather than by decay time (:func:`mode_scores`), a greedy
segment-wise allocation to a tolerance (:func:`allocate_per_segment`), and the
drive-independent moment patterns (:func:`moment_patterns`) that the study
compares the eigenbasis against.

Nothing here reads an ODS; :func:`vaft.omas.process_wrapper.compute_wall_mode_basis_ods`
does the mapping.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import warnings
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "ReducedWall",
    "SegmentModes",
    "WallModeBasis",
    "WallModeError",
    "allocate_per_segment",
    "build_wall_mode_basis",
    "canonical_sign",
    "check_wall_mode_basis",
    "combined_operators",
    "global_time_constants",
    "mode_scores",
    "moment_patterns",
    "orthonormalize_r",
    "project",
    "reconstruct",
    "reconstruction_error",
    "reduce_response",
    "reduced_operators",
    "segment_eigenmodes",
    "select_all",
    "select_by_score",
    "select_slowest",
    "select_tau_range",
    "solve_reduced_eddy",
    "subspace_angles_r",
    "symmetrize_inductance",
]

NORMALIZATION = "R_orthonormal"
SIGN_RULE = "max_abs_positive"


class WallModeError(ValueError):
    """The wall matrices or the segment map cannot support a well-posed basis."""


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def symmetrize_inductance(
    L: np.ndarray, *, rtol: float = 1e-8, reject: float = 1e-6
) -> tuple[np.ndarray, float]:
    """Fold a mutual-inductance matrix onto its symmetric part, with a refusal.

    Reciprocity makes a mutual-inductance matrix symmetric, so a departure is a
    defect in the asset, not noise.  The sanctioned place to repair reciprocity is
    the coupling mapper (:func:`vaft.machine_mapping.em_coupling.em_coupling`,
    issue #373), which is why this one refuses rather than repairs beyond a
    rounding-sized asymmetry.

    Parameters
    ----------
    L : numpy.ndarray
        The inductance matrix, ``(N, N)`` [H].
    rtol : float, optional
        Below this relative asymmetry the fold is silent [-].
    reject : float, optional
        Above this relative asymmetry the matrix is refused [-].

    Returns
    -------
    L_sym : numpy.ndarray
        ``(L + L^T) / 2`` [H].
    asymmetry : float
        ``max|L - L^T| / max|L|``, recorded in the basis provenance [-].

    Raises
    ------
    WallModeError
        When the asymmetry exceeds ``reject``.

    Warnings
    --------
    An asymmetry between ``rtol`` and ``reject`` is folded with a
    ``UserWarning``, so the repair reaches the provenance rather than
    happening silently.

    Defaults
    --------
    ``rtol = 1e-8`` is a numerical convenience, the size of double-precision
    round-off accumulated over an assembly of this scale.  ``reject = 1e-6`` is
    empirical: the asymmetry of the packaged VEST coupling asset once its
    one-sided material factor was repaired (issue #373).

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [473] Issue #473, which built the segment-wise eigenbasis, and #500, which
       added the selection, scoring and moment tools.
    .. [vfit8] The reduced vessel-wall contract, VEST-Tokamak/vfit#8: local
       eigenbasis, global electromagnetic dynamics.
    """
    L = np.asarray(L, dtype=float)
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise WallModeError(f"inductance must be square, got shape {L.shape}")
    if not np.all(np.isfinite(L)):
        raise WallModeError("inductance carries non-finite entries")
    scale = float(np.max(np.abs(L))) if L.size else 0.0
    if scale == 0.0:
        raise WallModeError("inductance is identically zero")
    asymmetry = float(np.max(np.abs(L - L.T)) / scale)
    if asymmetry > reject:
        raise WallModeError(
            f"inductance is asymmetric by {asymmetry:.3g} (relative), above {reject:.0e}; "
            "re-map em_coupling through vaft.machine_mapping.em_coupling (issues #347/#373) "
            "rather than symmetrizing here"
        )
    if asymmetry > rtol:
        warnings.warn(
            f"inductance asymmetric by {asymmetry:.3g} (relative); symmetrized for the "
            "wall-mode basis and recorded in its provenance",
            RuntimeWarning,
            stacklevel=2,
        )
    return 0.5 * (L + L.T), asymmetry


def _diagonal_resistance(R: np.ndarray, *, where: str = "R") -> np.ndarray:
    R = np.asarray(R, dtype=float)
    if R.ndim == 1:
        r = R
    elif R.ndim == 2 and R.shape[0] == R.shape[1]:
        off = R - np.diag(np.diag(R))
        if np.any(off != 0.0):
            raise WallModeError(
                f"{where} must be diagonal for the symmetric-definite pencil; "
                "a coupled resistance is not a passive-wall circuit"
            )
        r = np.diag(R)
    else:
        raise WallModeError(f"{where} must be a diagonal matrix or a vector, got shape {R.shape}")
    if not np.all(np.isfinite(r)) or np.any(r <= 0.0):
        raise WallModeError(f"{where} must be finite and positive on the diagonal (zero resistance has no decay time)")
    return r


def canonical_sign(V: np.ndarray) -> np.ndarray:
    """Flip each column so its largest-magnitude entry is positive.

    An eigenvector is defined up to sign; without a rule two runs of the same
    build can disagree, and a stored basis cannot be compared with a fresh one.

    Parameters
    ----------
    V : numpy.ndarray
        Modes as columns, ``(n, m)`` [any].

    Returns
    -------
    numpy.ndarray
        The same modes with each column's sign fixed [any].

    Convention
    ----------
    The sign rule of this module (``SIGN_RULE = "max_abs_positive"``): the
    largest-magnitude entry of every column is made positive, ties going to the
    lowest index (``argmax`` on ``|v|``).  A pure function of the column, so
    repeated runs agree bitwise.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Two entries of exactly equal magnitude and opposite sign make the rule
    index-dependent rather than ill-defined; a near-degenerate *eigenspace* is
    the real ambiguity, and :func:`build_wall_mode_basis` refuses that case
    instead of signing it.

    Provenance
    ----------
    .. [473] Issue #473, which built the segment-wise eigenbasis, and #500, which
       added the selection, scoring and moment tools.
    .. [vfit8] The reduced vessel-wall contract, VEST-Tokamak/vfit#8: local
       eigenbasis, global electromagnetic dynamics.
    """
    V = np.array(V, dtype=float, copy=True)
    if V.ndim != 2:
        raise WallModeError("modes must be a 2-D array of columns")
    pivots = np.argmax(np.abs(V), axis=0)
    signs = np.sign(V[pivots, np.arange(V.shape[1])])
    signs[signs == 0.0] = 1.0
    return V * signs[None, :]


def segment_eigenmodes(
    R_gg: np.ndarray,
    L_gg: np.ndarray,
    *,
    residual_atol: float = 1e-10,
    cond_max: float = 1e12,
    tau_floor_rel: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve one segment's pencil ``R_gg v = L_gg v / tau``.

    The local half of the contract: each segment's own L/R eigenbasis, from which
    the global dynamics are then built by projecting the *full* matrices.

    Processing steps
    ----------------
    1. Whiten by the resistance: ``S = R^{-1/2} L R^{-1/2}``, symmetric.
    2. ``eigh(S)``; the eigenvalues *are* the local L/R decay times.
    3. Map back, ``v = R^{-1/2} q``, so ``v^T R v = 1`` and ``v^T L v = tau``.
    4. Order by descending ``tau`` and fix the signs (:func:`canonical_sign`).

    Parameters
    ----------
    R_gg : numpy.ndarray
        The segment's diagonal resistance, as a matrix or as its diagonal
        [Ohm].
    L_gg : numpy.ndarray
        The segment's symmetric inductance block [H].
    residual_atol : float, optional
        Largest relative pencil residual accepted [-].
    cond_max : float, optional
        Largest accepted spread of decay times within the segment [-].
    tau_floor_rel : float, optional
        Smallest accepted decay time, relative to the largest [-].

    Returns
    -------
    tau : numpy.ndarray
        Local decay times, descending, ``(n_g,)`` [s].
    V : numpy.ndarray
        R-orthonormal, sign-canonical modes as columns, ``(n_g, n_g)``
        [A/sqrt(Ohm)].
    residual : float
        ``max|R V - L V / tau| / max|R V|`` [-].

    Raises
    ------
    WallModeError
        When the block is not a usable inductance (a non-positive eigenvalue is
        a current pattern with no stored energy), when the spread exceeds
        ``cond_max`` so the fastest mode is numerically indistinguishable from
        zero, or when the residual exceeds ``residual_atol``.

    Defaults
    --------
    All three are numerical convenience, set where double precision stops
    supporting the answer rather than where physics changes.

    Convention
    ----------
    Modes are R-orthonormal (``v^T R v = 1``) and sign-canonical
    (:func:`canonical_sign`), so amplitudes carry units of sqrt(W) and ``a^T a``
    is the wall current's ohmic dissipation.  Ordering is by descending ``tau`` within the
    segment, which is the mode index every ``keep`` and every label refers to.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [473] Issue #473, which built the segment-wise eigenbasis, and #500, which
       added the selection, scoring and moment tools.
    .. [vfit8] The reduced vessel-wall contract, VEST-Tokamak/vfit#8: local
       eigenbasis, global electromagnetic dynamics.
    """
    r = _diagonal_resistance(R_gg, where="R_gg")
    L_gg = np.asarray(L_gg, dtype=float)
    if L_gg.shape != (r.size, r.size):
        raise WallModeError(f"L_gg shape {L_gg.shape} does not match {r.size} elements")
    s = 1.0 / np.sqrt(r)
    S = (L_gg * s[:, None]) * s[None, :]
    S = 0.5 * (S + S.T)
    tau, Q = np.linalg.eigh(S)
    if not np.all(np.isfinite(tau)):
        raise WallModeError("segment pencil produced non-finite eigenvalues")
    tau_max = float(tau.max())
    if tau_max <= 0.0 or float(tau.min()) <= tau_floor_rel * tau_max:
        raise WallModeError(
            "segment inductance block is not positive definite (a mode with no stored "
            f"energy): tau range [{tau.min():.3g}, {tau_max:.3g}] s"
        )
    if tau_max / float(tau.min()) > cond_max:
        raise WallModeError(
            f"segment pencil condition number {tau_max / tau.min():.3g} exceeds {cond_max:.0e}"
        )
    order = np.argsort(-tau, kind="stable")
    tau = tau[order]
    V = canonical_sign(s[:, None] * Q[:, order])
    R_V = r[:, None] * V
    residual = float(np.max(np.abs(R_V - (L_gg @ V) / tau[None, :])) / np.max(np.abs(R_V)))
    if residual > residual_atol:
        raise WallModeError(f"segment eigen-residual {residual:.3g} exceeds {residual_atol:.0e}")
    return tau, V, residual


# ---------------------------------------------------------------------------
# The basis
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SegmentModes:
    """One segment's complete local eigenbasis."""

    id: str
    index: np.ndarray
    """Element indices into the full wall, ``(n_g,)``."""
    tau: np.ndarray
    """Local decay times [s], descending, ``(n_g,)``."""
    V: np.ndarray
    """R-orthonormal modes as columns [A/sqrt(Ohm)], ``(n_g, n_g)``."""
    residual: float
    min_relative_gap: float
    """``min_k (tau_k - tau_{k+1}) / tau_k``; small means a near-degenerate pair."""

    def __post_init__(self) -> None:
        for name in ("index", "tau", "V"):
            array = np.asarray(getattr(self, name))
            array = array.astype(np.int64 if name == "index" else float, copy=True)
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @property
    def size(self) -> int:
        return int(self.index.size)


@dataclass(frozen=True)
class ReducedWall:
    """Reduced operators for one retained selection ``keep``."""

    L_r: np.ndarray
    """Reduced inductance [H], ``(M_tot, M_tot)``: ``diag(tau_g)`` blocks plus
    the inter-segment coupling blocks."""
    R_r: np.ndarray
    """Reduced resistance [Ohm], ``(M_tot, M_tot)``; the identity to rounding."""
    M_r: np.ndarray | None
    """Reduced source coupling [H], ``(M_tot, n_src)``, when a source coupling
    was supplied."""
    labels: tuple[tuple[str, int], ...]
    """``(segment_id, k)`` for every reduced coefficient, segment-major."""
    keep: tuple[np.ndarray, ...]

    @property
    def n_modes(self) -> tuple[int, ...]:
        """``M_repr = (M_1, ..., M_G)``."""
        return tuple(int(k.size) for k in self.keep)


@dataclass(frozen=True)
class WallModeBasis:
    """Every segment's eigenbasis, plus how it was made."""

    segments: tuple[SegmentModes, ...]
    n_elements: int
    provenance: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "segments", tuple(self.segments))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    # -- shape -------------------------------------------------------------
    def n_modes(self) -> tuple[int, ...]:
        """Full ranks ``(n_1, ..., n_G)``; they sum to ``n_elements``."""
        return tuple(seg.size for seg in self.segments)

    def _keep(self, keep: Sequence[np.ndarray] | None) -> tuple[np.ndarray, ...]:
        if keep is None:
            return tuple(np.arange(seg.size) for seg in self.segments)
        if len(keep) != len(self.segments):
            raise WallModeError(f"keep has {len(keep)} entries for {len(self.segments)} segments")
        out = []
        for seg, k in zip(self.segments, keep):
            k = np.asarray(k, dtype=np.int64).reshape(-1)
            if k.size and (k.min() < 0 or k.max() >= seg.size):
                raise WallModeError(f"keep for segment {seg.id!r} indexes outside its {seg.size} modes")
            out.append(k)
        return tuple(out)

    def V(self, keep: Sequence[np.ndarray] | None = None) -> np.ndarray:
        """``V_seg`` for the retained modes, ``(n_elements, M_tot)``, scattered
        by element index so an interleaved loop order needs no permutation."""
        keep = self._keep(keep)
        total = sum(int(k.size) for k in keep)
        V = np.zeros((self.n_elements, total))
        col = 0
        for seg, k in zip(self.segments, keep):
            V[np.ix_(seg.index, np.arange(col, col + k.size))] = seg.V[:, k]
            col += k.size
        return V

    def tau(self, keep: Sequence[np.ndarray] | None = None) -> np.ndarray:
        keep = self._keep(keep)
        return np.concatenate([seg.tau[k] for seg, k in zip(self.segments, keep)]) if keep else np.empty(0)

    def labels(self, keep: Sequence[np.ndarray] | None = None) -> tuple[tuple[str, int], ...]:
        keep = self._keep(keep)
        return tuple((seg.id, int(j)) for seg, k in zip(self.segments, keep) for j in k)

    def segment(self, seg_id: str) -> SegmentModes:
        for seg in self.segments:
            if seg.id == seg_id:
                return seg
        raise KeyError(seg_id)

    # -- identity ----------------------------------------------------------
    def digest(self) -> str:
        """12-hex fingerprint of the segment ids, decay times and modes."""
        digest = hashlib.sha1()
        for seg in self.segments:
            digest.update(seg.id.encode("utf-8"))
            digest.update(np.round(seg.tau, 15).tobytes())
            digest.update(np.round(seg.V, 12).tobytes())
        return digest.hexdigest()[:12]

    # -- serialization -----------------------------------------------------
    def to_npz(self, path) -> None:
        payload: dict[str, Any] = {
            "n_elements": np.int64(self.n_elements),
            "segment_ids": np.array([seg.id for seg in self.segments]),
            "provenance_json": np.array(json.dumps(dict(self.provenance), sort_keys=True)),
        }
        for position, seg in enumerate(self.segments):
            payload[f"index_{position}"] = seg.index
            payload[f"tau_{position}"] = seg.tau
            payload[f"V_{position}"] = seg.V
            payload[f"residual_{position}"] = np.float64(seg.residual)
            payload[f"gap_{position}"] = np.float64(seg.min_relative_gap)
        np.savez_compressed(path, **payload)

    @classmethod
    def from_npz(cls, path) -> "WallModeBasis":
        with np.load(path, allow_pickle=False) as data:
            ids = [str(x) for x in data["segment_ids"]]
            segments = tuple(
                SegmentModes(
                    id=seg_id,
                    index=data[f"index_{p}"],
                    tau=data[f"tau_{p}"],
                    V=data[f"V_{p}"],
                    residual=float(data[f"residual_{p}"]),
                    min_relative_gap=float(data[f"gap_{p}"]),
                )
                for p, seg_id in enumerate(ids)
            )
            provenance = json.loads(str(data["provenance_json"]))
            return cls(segments=segments, n_elements=int(data["n_elements"]), provenance=provenance)


def _relative_gaps(tau: np.ndarray) -> float:
    if tau.size < 2:
        return float("inf")
    return float(np.min((tau[:-1] - tau[1:]) / tau[:-1]))


def build_wall_mode_basis(
    R_mat: np.ndarray,
    M_mat: np.ndarray,
    segments: Sequence[Any],
    *,
    symmetry_rtol: float = 1e-8,
    symmetry_reject: float = 1e-6,
    cluster_rtol: float = 1e-6,
    on_cluster: str = "raise",
    residual_atol: float = 1e-10,
    cond_max: float = 1e12,
    provenance: Mapping[str, str] | None = None,
) -> WallModeBasis:
    """Build the segment-wise eigenbasis of a wall.

    Partitions the wall's current vector by physical segment, gives each segment
    its own eigenbasis (:func:`segment_eigenmodes`), and records how it was done.
    No coupling is discarded here: the inter-segment mutual inductance survives
    because every reduced operator is later a projection of the full matrices.

    Processing steps
    ----------------
    1. Fold the inductance onto its symmetric part
       (:func:`symmetrize_inductance`) and record the asymmetry.
    2. Read the segment map; the segments must cover every element exactly once.
    3. Solve each segment's pencil.
    4. Check the within-segment decay-time gaps and act on ``on_cluster``.
    5. Record normalization, sign rule, mode order, projection formula, input
       asymmetry and the segment ids in the basis provenance.

    Parameters
    ----------
    R_mat : numpy.ndarray
        Diagonal loop resistance of the wall, ``(N, N)`` or its diagonal [Ohm].
    M_mat : numpy.ndarray
        Passive-passive inductance in code naming; the physics ``L``, ``(N, N)``
        [H].
    segments : sequence
        Objects with ``.id`` and ``.index`` (``WallSegment``) or ``(id, index)``
        pairs, together covering every element exactly once [-].
    symmetry_rtol, symmetry_reject : float, optional
        Passed to :func:`symmetrize_inductance` [-].
    cluster_rtol : float, optional
        Relative decay-time gap below which a within-segment pair counts as
        near-degenerate [-].
    on_cluster : {'raise', 'warn'}, optional
        What to do about such a pair [-].
    residual_atol, cond_max : float, optional
        Passed to :func:`segment_eigenmodes` [-].
    provenance : mapping of str to str or None, optional
        Extra entries -- the asset the matrices came from, a shot, a build --
        merged into the recorded provenance [-].

    Returns
    -------
    WallModeBasis
        Every segment's complete eigenbasis, the element count, and the
        provenance [-].

    Raises
    ------
    WallModeError
        When the segment map does not cover the elements exactly once, when a
        segment's pencil is refused, or, with ``on_cluster="raise"``, when a
        near-degenerate pair is found.

    Defaults
    --------
    ``cluster_rtol = 1e-6`` is a numerical convenience: inside a closer pair the
    individual modes are arbitrary up to rotation, so the default refuses rather
    than let the provenance claim a determinism it does not have.  The remaining
    tolerances are those of the routines they are passed to.

    Convention
    ----------
    Argument naming follows :mod:`vaft.process.electromagnetics`, not the
    physics: ``M_mat`` is the passive-passive inductance (physics ``L``) and
    ``L_mat`` the passive-to-source coupling (physics ``M``).  ``R_mat`` is the
    diagonal loop resistance either way.

    Applicability
    -------------
    Machine-independent.  The VEST wall's segment map comes from
    ``vaft.machine_mapping``; nothing here knows a VEST number.

    Limitations
    -----------
    A near-degenerate eigenspace is refused, not resolved: the honest comparison
    between two such bases is :func:`subspace_angles_r`, not a column-by-column
    one.

    Provenance
    ----------
    .. [473] Issue #473, which built the segment-wise eigenbasis, and #500, which
       added the selection, scoring and moment tools.
    .. [vfit8] The reduced vessel-wall contract, VEST-Tokamak/vfit#8: local
       eigenbasis, global electromagnetic dynamics.
    """
    if on_cluster not in ("raise", "warn"):
        raise ValueError("on_cluster must be 'raise' or 'warn'")
    r = _diagonal_resistance(R_mat, where="R_mat")
    n = r.size
    L, asymmetry = symmetrize_inductance(M_mat, rtol=symmetry_rtol, reject=symmetry_reject)
    if L.shape != (n, n):
        raise WallModeError(f"M_mat shape {L.shape} does not match {n} resistances")

    parsed: list[tuple[str, np.ndarray]] = []
    for item in segments:
        if hasattr(item, "id") and hasattr(item, "index"):
            parsed.append((str(item.id), np.asarray(item.index, dtype=np.int64).reshape(-1)))
        else:
            seg_id, index = item
            parsed.append((str(seg_id), np.asarray(index, dtype=np.int64).reshape(-1)))
    membership = np.full(n, -1, dtype=np.int64)
    for position, (seg_id, index) in enumerate(parsed):
        if index.size == 0:
            raise WallModeError(f"segment {seg_id!r} has no elements")
        if index.min() < 0 or index.max() >= n:
            raise WallModeError(f"segment {seg_id!r} indexes outside the {n} elements")
        if np.any(membership[index] >= 0):
            raise WallModeError(f"segment {seg_id!r} shares elements with another segment")
        membership[index] = position
    if np.any(membership < 0):
        raise WallModeError(f"{int(np.sum(membership < 0))} element(s) belong to no segment")

    modes: list[SegmentModes] = []
    clusters: list[str] = []
    for seg_id, index in parsed:
        tau, V, residual = segment_eigenmodes(
            r[index], L[np.ix_(index, index)],
            residual_atol=residual_atol, cond_max=cond_max,
        )
        gap = _relative_gaps(tau)
        if gap < cluster_rtol:
            where = int(np.argmin((tau[:-1] - tau[1:]) / tau[:-1]))
            clusters.append(f"{seg_id}:{where}-{where + 1}")
        modes.append(SegmentModes(id=seg_id, index=index, tau=tau, V=V, residual=residual, min_relative_gap=gap))
    if clusters:
        message = (
            f"near-degenerate decay times (relative gap < {cluster_rtol:.0e}) in {clusters}; "
            "the modes inside such a pair are defined only up to a rotation"
        )
        if on_cluster == "raise":
            raise WallModeError(message + " -- pass on_cluster='warn' to accept and record them")
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    record: dict[str, str] = {
        "normalization": NORMALIZATION,
        "sign_rule": SIGN_RULE,
        "mode_order": "descending_tau_within_segment",
        "projection": "a = V^T R I_w",
        "input_asymmetry": f"{asymmetry:.6e}",
        "n_segments": str(len(modes)),
        "segment_ids": ",".join(seg.id for seg in modes),
    }
    if clusters:
        record["degenerate_pairs"] = ",".join(clusters)
    if provenance:
        record.update({str(k): str(v) for k, v in provenance.items()})
    return WallModeBasis(segments=tuple(modes), n_elements=n, provenance=record)


# ---------------------------------------------------------------------------
# Using the basis
# ---------------------------------------------------------------------------

def reduced_operators(
    basis: WallModeBasis,
    R_mat: np.ndarray,
    M_mat: np.ndarray,
    L_mat: np.ndarray | None = None,
    keep: Sequence[np.ndarray] | None = None,
) -> ReducedWall:
    """Project the full wall matrices onto the retained modes.

    The global half of the contract.  Because the *full* matrices are projected,
    the mutual inductance between segments survives in the off-diagonal blocks of
    ``L_r``; only the diagonal blocks are the local ``diag(tau_g)``.

    Input semantics
    ---------------
    Element space: the assembled wall matrices, one row and column per wall
    element, as :mod:`vaft.process.electromagnetics` builds them.

    Output semantics
    ----------------
    Mode space: operators of the retained modal coordinates, whose amplitudes
    carry units of sqrt(W).  A solve on them is the reduced circuit
    (:func:`solve_reduced_eddy`); to come back to element space, use
    :func:`reconstruct`.

    Parameters
    ----------
    basis : WallModeBasis
        The eigenbasis [-].
    R_mat : numpy.ndarray
        Diagonal loop resistance, ``(N, N)`` or its diagonal [Ohm].
    M_mat : numpy.ndarray
        Passive-passive inductance in code naming, ``(N, N)`` [H].
    L_mat : numpy.ndarray or None, optional
        Passive-to-source coupling in code naming, ``(N, n_src)`` [H].
    keep : sequence of numpy.ndarray or None, optional
        Retained mode indices per segment; every mode when ``None`` [-].

    Returns
    -------
    ReducedWall
        ``L_r`` ``(M_tot, M_tot)`` [H], ``R_r`` ``(M_tot, M_tot)`` [Ohm],
        ``M_r`` ``(M_tot, n_src)`` [H] when a source coupling was given, and the
        ``(segment_id, k)`` label of every coefficient [-].

    Convention
    ----------
    Argument naming follows :mod:`vaft.process.electromagnetics`, not the
    physics: ``M_mat`` is the passive-passive inductance (physics ``L``) and
    ``L_mat`` the passive-to-source coupling (physics ``M``).  ``R_mat`` is the
    diagonal loop resistance either way.

    ``R_r`` is computed rather than assumed to be the identity, so a basis built
    on other matrices than the ones passed here shows up as a non-identity
    instead of silently changing the dynamics.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [473] Issue #473, which built the segment-wise eigenbasis, and #500, which
       added the selection, scoring and moment tools.
    .. [vfit8] The reduced vessel-wall contract, VEST-Tokamak/vfit#8: local
       eigenbasis, global electromagnetic dynamics.
    """
    V = basis.V(keep)
    r = _diagonal_resistance(R_mat, where="R_mat")
    L = np.asarray(M_mat, dtype=float)
    L_r = V.T @ L @ V
    R_r = V.T @ (r[:, None] * V)
    M_r = None if L_mat is None else V.T @ np.asarray(L_mat, dtype=float)
    return ReducedWall(L_r=L_r, R_r=R_r, M_r=M_r, labels=basis.labels(keep), keep=basis._keep(keep))


def project(
    basis: WallModeBasis, I_w: np.ndarray, R_mat: np.ndarray, keep: Sequence[np.ndarray] | None = None
) -> np.ndarray:
    """Modal amplitudes ``a = V^T R I_w``.

    Exact for a current that lies inside the retained subspace; for a truncated
    basis it is the R-orthogonal projection, which is the least-dissipation one.

    Input semantics
    ---------------
    Element space: a wall current, one entry per element, measured or solved.

    Output semantics
    ----------------
    Mode space: amplitudes in the R inner product, so ``a^T a`` is the current's
    ohmic dissipation.

    Parameters
    ----------
    basis : WallModeBasis
        The eigenbasis [-].
    I_w : numpy.ndarray
        Wall current, ``(N,)`` or ``(n_times, N)`` [A].
    R_mat : numpy.ndarray
        Diagonal loop resistance, ``(N, N)`` or its diagonal [Ohm].
    keep : sequence of numpy.ndarray or None, optional
        Retained mode indices per segment; every mode when ``None`` [-].

    Returns
    -------
    numpy.ndarray
        Amplitudes, ``(M_tot,)`` or ``(n_times, M_tot)`` [W**0.5].

    Convention
    ----------
    Modes are R-orthonormal (``v^T R v = 1``) and sign-canonical
    (:func:`canonical_sign`), so amplitudes carry units of sqrt(W) and ``a^T a``
    is the wall current's ohmic dissipation.  Two other normalizations are in use elsewhere and
    convert from this one: Euclidean-normalized modes have ``a_E = a ||v||_2``,
    inductance-normalized ones ``a_L = a sqrt(tau)``.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [473] Issue #473, which built the segment-wise eigenbasis, and #500, which
       added the selection, scoring and moment tools.
    .. [vfit8] The reduced vessel-wall contract, VEST-Tokamak/vfit#8: local
       eigenbasis, global electromagnetic dynamics.
    """
    V = basis.V(keep)
    r = _diagonal_resistance(R_mat, where="R_mat")
    I_w = np.asarray(I_w, dtype=float)
    if I_w.ndim == 1:
        return V.T @ (r * I_w)
    return (I_w * r[None, :]) @ V


def reconstruct(basis: WallModeBasis, a: np.ndarray, keep: Sequence[np.ndarray] | None = None) -> np.ndarray:
    """Wall currents ``I_w = V a`` from modal amplitudes.

    The inverse direction of :func:`project`; exact only for the retained
    subspace, which is what :func:`reconstruction_error` measures.

    Input semantics
    ---------------
    Mode space: amplitudes of the retained modes.

    Output semantics
    ----------------
    Element space: a wall current on every element, whatever the retained order.

    Parameters
    ----------
    basis : WallModeBasis
        The eigenbasis [-].
    a : numpy.ndarray
        Amplitudes, ``(M_tot,)`` or ``(n_times, M_tot)`` [W**0.5].
    keep : sequence of numpy.ndarray or None, optional
        Retained mode indices per segment; every mode when ``None`` [-].

    Returns
    -------
    numpy.ndarray
        Wall current, ``(N,)`` or ``(n_times, N)`` [A].

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [473] Issue #473, which built the segment-wise eigenbasis, and #500, which
       added the selection, scoring and moment tools.
    .. [vfit8] The reduced vessel-wall contract, VEST-Tokamak/vfit#8: local
       eigenbasis, global electromagnetic dynamics.
    """
    V = basis.V(keep)
    a = np.asarray(a, dtype=float)
    return V @ a if a.ndim == 1 else a @ V.T


def reconstruction_error(
    I_w: np.ndarray, I_rec: np.ndarray, R_mat: np.ndarray, basis: WallModeBasis | None = None
) -> dict[str, Any]:
    """How far a reconstruction is from the full current, in the norms that matter.

    Parameters
    ----------
    I_w : numpy.ndarray
        The full wall current, ``(N,)`` or ``(n_times, N)`` [A].
    I_rec : numpy.ndarray
        The reconstruction, same shape [A].
    R_mat : numpy.ndarray
        Diagonal loop resistance, ``(N, N)`` or its diagonal [Ohm].
    basis : WallModeBasis or None, optional
        When given, the same measures are reported per segment [-].

    Returns
    -------
    dict
        ``relative_l2``, ``relative_dissipation`` and ``max_abs``, plus one
        such entry per segment id when a basis was given [-].

        ``max_abs`` is the largest absolute element error, in amperes; the
        other two are relative.

    Convention
    ----------
    ``relative_dissipation`` weighs each element by its resistance -- the natural
    norm of this basis, where the error's ohmic power is compared with the
    current's -- and ``relative_l2`` is the plain Euclidean measure.  They differ
    by however unevenly the wall's resistance is distributed, so a reported
    error means nothing without saying which of the two it is.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [473] Issue #473, which built the segment-wise eigenbasis, and #500, which
       added the selection, scoring and moment tools.
    .. [vfit8] The reduced vessel-wall contract, VEST-Tokamak/vfit#8: local
       eigenbasis, global electromagnetic dynamics.
    """
    r = _diagonal_resistance(R_mat, where="R_mat")
    I_w = np.asarray(I_w, dtype=float).reshape(-1, r.size)
    I_rec = np.asarray(I_rec, dtype=float).reshape(-1, r.size)
    err = I_rec - I_w

    def _measures(e: np.ndarray, ref: np.ndarray, w: np.ndarray) -> dict[str, float]:
        l2 = float(np.linalg.norm(e) / max(np.linalg.norm(ref), 1e-300))
        diss = float(np.sqrt(np.sum(e**2 * w) / max(np.sum(ref**2 * w), 1e-300)))
        return {"relative_l2": l2, "relative_dissipation": diss, "max_abs": float(np.max(np.abs(e)))}

    out: dict[str, Any] = _measures(err, I_w, r[None, :])
    if basis is not None:
        out["segments"] = {
            seg.id: _measures(err[:, seg.index], I_w[:, seg.index], r[None, seg.index])
            for seg in basis.segments
        }
    return out


def reduce_response(
    G_full: np.ndarray, basis: WallModeBasis, keep: Sequence[np.ndarray] | None = None
) -> np.ndarray:
    """Reduce any linear response with the wall elements as its columns.

    Probe field, flux-loop flux, grid psi or B, an EFIT table -- one matmul, no
    per-class logic, so every response class is reduced by the same basis.

    Input semantics
    ---------------
    Element space: a response matrix mapping every wall element to observations.

    Output semantics
    ----------------
    Mode space: the same map from the retained modal amplitudes.

    Parameters
    ----------
    G_full : numpy.ndarray
        Response with wall elements as columns, ``(n_obs, N)`` [any/A].
    basis : WallModeBasis
        The eigenbasis [-].
    keep : sequence of numpy.ndarray or None, optional
        Retained mode indices per segment; every mode when ``None`` [-].

    Returns
    -------
    numpy.ndarray
        ``G_red = G_full V_seg``, ``(n_obs, M_tot)`` [any/W**0.5].

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [473] Issue #473, which built the segment-wise eigenbasis, and #500, which
       added the selection, scoring and moment tools.
    .. [vfit8] The reduced vessel-wall contract, VEST-Tokamak/vfit#8: local
       eigenbasis, global electromagnetic dynamics.
    """
    G_full = np.asarray(G_full, dtype=float)
    if G_full.shape[-1] != basis.n_elements:
        raise WallModeError(
            f"response has {G_full.shape[-1]} wall columns, basis has {basis.n_elements} elements"
        )
    return G_full @ basis.V(keep)


# ---------------------------------------------------------------------------
# Selection and inspection (no order is chosen here)
# ---------------------------------------------------------------------------

def select_all(basis: WallModeBasis) -> tuple[np.ndarray, ...]:
    """Keep every mode of every segment: the full-rank selection.

    Parameters
    ----------
    basis : WallModeBasis
        The eigenbasis [-].

    Returns
    -------
    tuple of numpy.ndarray
        One index array per segment, holding every mode index [-].

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [473] Issue #473, which built the segment-wise eigenbasis, and #500, which
       added the selection, scoring and moment tools.
    .. [vfit8] The reduced vessel-wall contract, VEST-Tokamak/vfit#8: local
       eigenbasis, global electromagnetic dynamics.
    """
    return basis._keep(None)


def select_slowest(basis: WallModeBasis, M: int | Sequence[int]) -> tuple[np.ndarray, ...]:
    """Keep the slowest modes, globally or per segment.

    The selection the spectrum alone suggests; whether it is the right one for a
    given drive is what :func:`mode_scores` exists to question.

    Parameters
    ----------
    basis : WallModeBasis
        The eigenbasis [-].
    M : int or sequence of int
        A count taken globally across segments by descending decay time, or one
        count per segment [-].

    Returns
    -------
    tuple of numpy.ndarray
        One index array per segment [-].

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [473] Issue #473, which built the segment-wise eigenbasis, and #500, which
       added the selection, scoring and moment tools.
    .. [vfit8] The reduced vessel-wall contract, VEST-Tokamak/vfit#8: local
       eigenbasis, global electromagnetic dynamics.
    .. [vfit10] The reduced-order validation study (VEST-Tokamak/vfit#10, vaft
       #494), which asks which modes to keep; this module answers no part of that
       question by itself.
    """
    if isinstance(M, (int, np.integer)):
        tau = basis.tau()
        labels = basis.labels()
        chosen = set(labels[i] for i in np.argsort(-tau, kind="stable")[: int(M)])
        return tuple(
            np.array([k for k in range(seg.size) if (seg.id, k) in chosen], dtype=np.int64)
            for seg in basis.segments
        )
    counts = list(M)
    if len(counts) != len(basis.segments):
        raise WallModeError(f"per-segment M has {len(counts)} entries for {len(basis.segments)} segments")
    return tuple(np.arange(min(int(m), seg.size), dtype=np.int64) for seg, m in zip(basis.segments, counts))


def select_tau_range(basis: WallModeBasis, tau_min: float, tau_max: float = np.inf) -> tuple[np.ndarray, ...]:
    """Keep the modes whose decay time lies in a range, per segment.

    Parameters
    ----------
    basis : WallModeBasis
        The eigenbasis [-].
    tau_min : float
        Slowest excluded bound from below; modes at or above it are kept [s].
    tau_max : float, optional
        Upper bound, inclusive [s].

    Returns
    -------
    tuple of numpy.ndarray
        One index array per segment [-].

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [473] Issue #473, which built the segment-wise eigenbasis, and #500, which
       added the selection, scoring and moment tools.
    .. [vfit8] The reduced vessel-wall contract, VEST-Tokamak/vfit#8: local
       eigenbasis, global electromagnetic dynamics.
    """
    return tuple(
        np.flatnonzero((seg.tau >= tau_min) & (seg.tau <= tau_max)).astype(np.int64)
        for seg in basis.segments
    )


def global_time_constants(basis: WallModeBasis, M_mat: np.ndarray) -> np.ndarray:
    """The whole wall's decay times, from the full-rank reduced inductance.

    With ``R_r = I`` the reduced pencil is an ordinary symmetric eigenproblem, so
    ``eigvalsh(L_r)`` are the global L/R times.  They are not the segments' local
    times: the difference between the two is exactly the inter-segment coupling
    the basis keeps.

    Parameters
    ----------
    basis : WallModeBasis
        The eigenbasis [-].
    M_mat : numpy.ndarray
        Passive-passive inductance in code naming, ``(N, N)`` [H].

    Returns
    -------
    numpy.ndarray
        Global decay times, descending, ``(N,)`` [s].

    Convention
    ----------
    Full rank by construction: every mode of every segment takes part, because a
    truncated basis would report the decay times of a different wall.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [473] Issue #473, which built the segment-wise eigenbasis, and #500, which
       added the selection, scoring and moment tools.
    .. [vfit8] The reduced vessel-wall contract, VEST-Tokamak/vfit#8: local
       eigenbasis, global electromagnetic dynamics.
    """
    V = basis.V(None)
    L_r = V.T @ np.asarray(M_mat, dtype=float) @ V
    L_r = 0.5 * (L_r + L_r.T)
    return np.sort(np.linalg.eigvalsh(L_r))[::-1]


def check_wall_mode_basis(basis: WallModeBasis, R_mat: np.ndarray, M_mat: np.ndarray) -> dict[str, Any]:
    """Metrics of a basis against the matrices it claims to diagonalize; no verdict.

    Numbers a caller or a report can threshold as it sees fit; nothing here
    decides whether a basis is good enough.

    Parameters
    ----------
    basis : WallModeBasis
        The eigenbasis [-].
    R_mat : numpy.ndarray
        Diagonal loop resistance, ``(N, N)`` or its diagonal [Ohm].
    M_mat : numpy.ndarray
        Passive-passive inductance in code naming, ``(N, N)`` [H].

    Returns
    -------
    dict
        ``n_elements``, ``n_modes``, ``r_r_identity_error``
        (``max|V^T R V - I|``), ``l_r_symmetry_error``,
        ``max_segment_residual``, ``min_relative_gap`` and ``coupling`` [-].

        ``coupling`` holds, per segment pair, the Frobenius norm of that
        off-diagonal block of ``L_r`` relative to the geometric mean of the two
        diagonal blocks, so a reader can see which segments actually talk to
        each other [-].

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [473] Issue #473, which built the segment-wise eigenbasis, and #500, which
       added the selection, scoring and moment tools.
    .. [vfit8] The reduced vessel-wall contract, VEST-Tokamak/vfit#8: local
       eigenbasis, global electromagnetic dynamics.
    """
    ops = reduced_operators(basis, R_mat, M_mat)
    M_tot = ops.L_r.shape[0]
    identity_error = float(np.max(np.abs(ops.R_r - np.eye(M_tot))))
    offsets = np.cumsum([0] + [seg.size for seg in basis.segments])
    coupling: dict[str, float] = {}
    for i, a in enumerate(basis.segments):
        for j, b in enumerate(basis.segments):
            if j <= i:
                continue
            block = ops.L_r[offsets[i]:offsets[i + 1], offsets[j]:offsets[j + 1]]
            scale = np.sqrt(
                np.linalg.norm(ops.L_r[offsets[i]:offsets[i + 1], offsets[i]:offsets[i + 1]])
                * np.linalg.norm(ops.L_r[offsets[j]:offsets[j + 1], offsets[j]:offsets[j + 1]])
            )
            coupling[f"{a.id}-{b.id}"] = float(np.linalg.norm(block) / scale) if scale > 0 else float("nan")
    return {
        "n_elements": basis.n_elements,
        "n_modes": basis.n_modes(),
        "r_r_identity_error": identity_error,
        "l_r_symmetry_error": float(np.max(np.abs(ops.L_r - ops.L_r.T)) / np.max(np.abs(ops.L_r))),
        "max_segment_residual": max(seg.residual for seg in basis.segments),
        "min_relative_gap": min(seg.min_relative_gap for seg in basis.segments),
        "coupling": coupling,
    }


def subspace_angles_r(V_a: np.ndarray, V_b: np.ndarray, R_mat: np.ndarray) -> np.ndarray:
    """Principal angles between two mode subspaces in the R inner product.

    Two bases of a near-degenerate eigenspace differ by a rotation, so the right
    question is whether they span the same space, not whether their columns
    match.

    Parameters
    ----------
    V_a : numpy.ndarray
        First set of modes as columns, ``(N, m_a)`` [A/sqrt(Ohm)].
    V_b : numpy.ndarray
        Second set, ``(N, m_b)`` [A/sqrt(Ohm)].
    R_mat : numpy.ndarray
        Diagonal loop resistance, ``(N, N)`` or its diagonal [Ohm].

    Returns
    -------
    numpy.ndarray
        Principal angles, ascending, ``(min(m_a, m_b),)`` [rad].

    Convention
    ----------
    The angles are taken in the R metric, not the Euclidean one: both sets are
    mapped by ``R^{1/2}`` first, so the ordinary angles of the mapped columns are
    the R-metric angles of the originals.  A zero angle means a shared direction;
    ``pi/2`` means an R-orthogonal one.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [473] Issue #473, which built the segment-wise eigenbasis, and #500, which
       added the selection, scoring and moment tools.
    .. [vfit8] The reduced vessel-wall contract, VEST-Tokamak/vfit#8: local
       eigenbasis, global electromagnetic dynamics.
    """
    from scipy.linalg import subspace_angles

    r = _diagonal_resistance(R_mat, where="R_mat")
    w = np.sqrt(r)[:, None]
    return subspace_angles(w * np.asarray(V_a, dtype=float), w * np.asarray(V_b, dtype=float))


# ---------------------------------------------------------------------------
# Reduced dynamics and response-ranked selection (vaft #494, vfit #10)
# ---------------------------------------------------------------------------

def solve_reduced_eddy(
    reduced: ReducedWall,
    drive: np.ndarray,
    time: np.ndarray,
    *,
    V: np.ndarray | None = None,
    dt_sub: float = 5.0e-5,
    method: str = "auto",
) -> tuple[np.ndarray, np.ndarray | None]:
    """Integrate the reduced circuit ``L_r da/dt + R_r a = -M_r dI_src/dt``.

    The same integrator as the full wall
    (:func:`vaft.process.electromagnetics.solve_eddy_currents`), fed the projected
    operators, so the only difference between the two solutions is the retained
    subspace: with every mode kept they agree to rounding.

    Input semantics
    ---------------
    Mode space: reduced operators and a source drive in element-independent
    coordinates.

    Output semantics
    ----------------
    Mode space amplitudes, and -- when the retained ``V`` is given -- the
    reconstructed element-space wall current alongside them.

    Parameters
    ----------
    reduced : ReducedWall
        The projected operators; ``M_r`` must be present [-].
    drive : numpy.ndarray
        Source currents, ``(n_times, n_src)`` on ``time`` [A].
    time : numpy.ndarray
        Time grid of the drive [s].
    V : numpy.ndarray or None, optional
        The retained modes, ``(N, M_tot)``; when given, the wall current is
        reconstructed too [A/sqrt(Ohm)].
    dt_sub : float, optional
        Sub-step of the integrator [s].
    method : str, optional
        Integration method, forwarded to the full-wall solver [-].

    Returns
    -------
    a : numpy.ndarray
        Modal amplitudes, ``(n_times, M_tot)`` [W**0.5].
    I_w : numpy.ndarray or None
        ``V a``, ``(n_times, N)``, or ``None`` when no ``V`` was given [A].

    Defaults
    --------
    ``dt_sub = 50 us`` is a validated-workflow default: the sub-step the full-wall
    solver is used with in the VEST eddy-current workflow.  ``method="auto"``
    leaves the choice to that solver.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [473] Issue #473, which built the segment-wise eigenbasis, and #500, which
       added the selection, scoring and moment tools.
    .. [vfit8] The reduced vessel-wall contract, VEST-Tokamak/vfit#8: local
       eigenbasis, global electromagnetic dynamics.
    .. [vfit10] The reduced-order validation study (VEST-Tokamak/vfit#10, vaft
       #494), which asks which modes to keep; this module answers no part of that
       question by itself.
    """
    from vaft.process.electromagnetics import solve_eddy_currents

    if reduced.M_r is None:
        raise WallModeError("solve_reduced_eddy needs the projected source coupling M_r")
    a = solve_eddy_currents(
        reduced.R_r, reduced.M_r, reduced.L_r,
        np.asarray(drive, dtype=float), np.asarray(time, dtype=float),
        dt_sub=dt_sub, method=method,
    )
    I_w = None if V is None else a @ np.asarray(V, dtype=float).T
    return a, I_w


def mode_scores(
    basis: WallModeBasis,
    R_mat: np.ndarray,
    M_mat: np.ndarray,
    L_mat: np.ndarray,
    *,
    G: np.ndarray | None = None,
    drive: np.ndarray | None = None,
    time: np.ndarray | None = None,
    keep: Sequence[np.ndarray] | None = None,
    dt_sub: float = 5.0e-5,
) -> dict[str, np.ndarray]:
    """Rankings of the retained modes, one value per retained coefficient.

    Rankings, not verdicts: each is a different answer to *which modes carry the
    wall's response*, and the order study compares them.

    ``tau``
        the decay time -- what the spectrum alone would rank by;
    ``drive_gain``
        ``tau_k ||M_r[k, :]|| ||G_red[:, k]||`` -- the quasi-static amplitude a
        unit source ramp excites in mode ``k``, times how visible the mode is at
        the observation points (all ones without ``G``); needs no drive;
    ``response_energy``
        the rms of the projected full response ``a = V^T R I_w(t)`` under
        ``drive`` -- how much dissipation each mode actually carried;
    ``output_weight``
        ``response_energy`` times the observability, the ranking that minimizes
        the diagnostic-space error fastest on the packaged wall.

    Parameters
    ----------
    basis : WallModeBasis
        The eigenbasis [-].
    R_mat : numpy.ndarray
        Diagonal loop resistance, ``(N, N)`` or its diagonal [Ohm].
    M_mat : numpy.ndarray
        Passive-passive inductance in code naming, ``(N, N)`` [H].
    L_mat : numpy.ndarray
        Passive-to-source coupling in code naming, ``(N, n_src)`` [H].
    G : numpy.ndarray or None, optional
        Observation response with wall elements as columns, ``(n_obs, N)``;
        without it every mode is equally observable [any/A].
    drive : numpy.ndarray or None, optional
        Source currents, ``(n_times, n_src)``; needed by the last two rankings
        [A].
    time : numpy.ndarray or None, optional
        Time grid of the drive [s].
    keep : sequence of numpy.ndarray or None, optional
        Candidate modes per segment; every mode when ``None`` [-].
    dt_sub : float, optional
        Sub-step of the full-wall solve the last two rankings need [s].

    Returns
    -------
    dict of str to numpy.ndarray
        ``tau``, ``drive_gain`` and, when a drive was given,
        ``response_energy`` and ``output_weight``; each array is aligned with
        ``basis.labels(keep)`` [-].

        ``tau`` is in seconds; the three rankings are relative numbers whose
        scale carries no meaning, only their order does.

    Defaults
    --------
    ``dt_sub = 50 us`` is a validated-workflow default, as in
    :func:`solve_reduced_eddy`.

    Applicability
    -------------
    Machine-independent.  The claim that ``output_weight`` falls fastest is an
    observation on the packaged VEST wall, not a theorem.

    Limitations
    -----------
    The last two rankings cost one full wall solve.  Every ranking is relative to
    the drive and observation set it was computed with; a mode invisible to one
    diagnostic set is not an unimportant mode.

    Provenance
    ----------
    .. [473] Issue #473, which built the segment-wise eigenbasis, and #500, which
       added the selection, scoring and moment tools.
    .. [vfit8] The reduced vessel-wall contract, VEST-Tokamak/vfit#8: local
       eigenbasis, global electromagnetic dynamics.
    .. [vfit10] The reduced-order validation study (VEST-Tokamak/vfit#10, vaft
       #494), which asks which modes to keep; this module answers no part of that
       question by itself.
    """
    ops = reduced_operators(basis, R_mat, M_mat, L_mat, keep)
    tau = basis.tau(keep)
    if G is not None:
        observability = np.linalg.norm(reduce_response(G, basis, keep), axis=0)
    else:
        observability = np.ones(tau.size)
    scores: dict[str, np.ndarray] = {
        "tau": tau,
        "drive_gain": tau * np.linalg.norm(ops.M_r, axis=1) * observability,
    }
    if drive is not None:
        if time is None:
            raise WallModeError("mode_scores needs `time` with `drive`")
        from vaft.process.electromagnetics import solve_eddy_currents

        I_full = solve_eddy_currents(
            np.asarray(R_mat, dtype=float), np.asarray(L_mat, dtype=float),
            np.asarray(M_mat, dtype=float), np.asarray(drive, dtype=float),
            np.asarray(time, dtype=float), dt_sub=dt_sub,
        )
        a = project(basis, I_full, R_mat, keep)
        energy = np.sqrt(np.mean(a**2, axis=0))
        scores["response_energy"] = energy
        scores["output_weight"] = energy * observability
    return scores


def select_by_score(
    basis: WallModeBasis, score: np.ndarray, M: int, keep: Sequence[np.ndarray] | None = None
) -> tuple[np.ndarray, ...]:
    """Keep the highest-scoring modes across segments.

    The global counterpart of :func:`select_slowest`, for any ranking -- usually
    one of :func:`mode_scores`.

    Parameters
    ----------
    basis : WallModeBasis
        The eigenbasis [-].
    score : numpy.ndarray
        One value per coefficient of ``basis.labels(keep)`` [-].
    M : int
        How many modes to keep in total [-].
    keep : sequence of numpy.ndarray or None, optional
        Candidate modes per segment; every mode when ``None`` [-].

    Returns
    -------
    tuple of numpy.ndarray
        One index array per segment [-].

    Convention
    ----------
    Ties resolve toward the earlier label, so the selection is deterministic for
    a given score vector.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [473] Issue #473, which built the segment-wise eigenbasis, and #500, which
       added the selection, scoring and moment tools.
    .. [vfit8] The reduced vessel-wall contract, VEST-Tokamak/vfit#8: local
       eigenbasis, global electromagnetic dynamics.
    .. [vfit10] The reduced-order validation study (VEST-Tokamak/vfit#10, vaft
       #494), which asks which modes to keep; this module answers no part of that
       question by itself.
    """
    labels = basis.labels(keep)
    score = np.asarray(score, dtype=float).reshape(-1)
    if score.size != len(labels):
        raise WallModeError(f"score has {score.size} entries for {len(labels)} candidate modes")
    order = np.argsort(-score, kind="stable")[: max(int(M), 0)]
    chosen = set(labels[i] for i in order)
    return tuple(
        np.array(sorted(k for (seg_id, k) in chosen if seg_id == seg.id), dtype=np.int64)
        for seg in basis.segments
    )


def allocate_per_segment(
    basis: WallModeBasis,
    R_mat: np.ndarray,
    M_mat: np.ndarray,
    L_mat: np.ndarray,
    drive: np.ndarray,
    time: np.ndarray,
    *,
    tolerance: float,
    metric: str = "dissipation",
    G: np.ndarray | None = None,
    score: np.ndarray | None = None,
    step: int = 1,
    max_modes: int | None = None,
    dt_sub: float = 5.0e-5,
) -> tuple[tuple[np.ndarray, ...], list[dict[str, Any]]]:
    """Greedy segment-wise allocation ``M_repr`` to a response tolerance.

    A global allocation for a *given* total is simply :func:`select_by_score`;
    this routine answers the other question -- the smallest total that meets a
    response tolerance, and how it should be split between segments.

    Processing steps
    ----------------
    1. Solve the full wall once under ``drive`` and rank the candidates within
       each segment (the ``output_weight`` ranking of :func:`mode_scores` by
       default, which needs no more than that solve).
    2. Start from no modes at all.
    3. Each round, re-solve the reduced wall, measure the global ``metric`` and
       each segment's share of it, and add ``step`` modes to the segment
       carrying the largest remaining error, in that segment's ranked order.
    4. Stop when the metric drops to ``tolerance``, when ``max_modes`` is
       reached, or when every candidate is used.

    Parameters
    ----------
    basis : WallModeBasis
        The eigenbasis [-].
    R_mat : numpy.ndarray
        Diagonal loop resistance, ``(N, N)`` or its diagonal [Ohm].
    M_mat : numpy.ndarray
        Passive-passive inductance in code naming, ``(N, N)`` [H].
    L_mat : numpy.ndarray
        Passive-to-source coupling in code naming, ``(N, n_src)`` [H].
    drive : numpy.ndarray
        Source currents, ``(n_times, n_src)`` [A].
    time : numpy.ndarray
        Time grid of the drive [s].
    tolerance : float
        Relative error the allocation stops at [-].
    metric : {'dissipation', 'output'}, optional
        Which error to measure [-].
    G : numpy.ndarray or None, optional
        Observation response, required by ``metric="output"`` [any/A].
    score : numpy.ndarray or None, optional
        Ranking within each segment; ``output_weight`` when ``None`` [-].
    step : int, optional
        Modes added per round [-].
    max_modes : int or None, optional
        Cap on the total [-].
    dt_sub : float, optional
        Sub-step of the solves [s].

    Returns
    -------
    keep : tuple of numpy.ndarray
        The allocation, one index array per segment [-].
    history : list of dict
        One row per round: the running ``M_repr``, its total ``M_total``, the
        metric under its own name, and ``by_segment`` [-].

    Defaults
    --------
    ``metric="dissipation"`` is the relative R-energy error of the wall current,
    the norm this basis is built in; it is a conventional choice, not a tuned
    one.  ``dt_sub`` is a validated-workflow default, as in
    :func:`solve_reduced_eddy`.

    Convention
    ----------
    The per-segment error is the segment's share of the global squared error in
    the same norm, so a segment the drive barely reaches never attracts modes.
    ``metric="output"`` measures the relative error of ``G I_w`` instead, which
    ranks a mode by what a diagnostic would see rather than by what it
    dissipates.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Greedy, so the allocation is not proven minimal; it is the smallest total
    this order of additions reaches.  Each round costs a reduced solve, and the
    result is only as general as the ``drive`` it was allocated under.

    Provenance
    ----------
    .. [473] Issue #473, which built the segment-wise eigenbasis, and #500, which
       added the selection, scoring and moment tools.
    .. [vfit8] The reduced vessel-wall contract, VEST-Tokamak/vfit#8: local
       eigenbasis, global electromagnetic dynamics.
    .. [vfit10] The reduced-order validation study (VEST-Tokamak/vfit#10, vaft
       #494), which asks which modes to keep; this module answers no part of that
       question by itself.
    """
    from vaft.process.electromagnetics import solve_eddy_currents

    if metric not in ("dissipation", "output"):
        raise WallModeError("metric must be 'dissipation' or 'output'")
    if metric == "output" and G is None:
        raise WallModeError("metric='output' needs the response G")
    r = _diagonal_resistance(R_mat, where="R_mat")
    drive = np.asarray(drive, dtype=float)
    time = np.asarray(time, dtype=float)
    I_full = solve_eddy_currents(r[:, None] * np.eye(r.size), np.asarray(L_mat, dtype=float),
                                 np.asarray(M_mat, dtype=float), drive, time, dt_sub=dt_sub)
    if score is None:
        a_full = project(basis, I_full, R_mat)
        observability = np.ones(basis.n_elements) if G is None else np.linalg.norm(reduce_response(G, basis), axis=0)
        score = np.sqrt(np.mean(a_full**2, axis=0)) * observability
    score = np.asarray(score, dtype=float).reshape(-1)
    if score.size != basis.n_elements:
        raise WallModeError(f"score has {score.size} entries for {basis.n_elements} modes")

    offsets = np.cumsum([0] + [seg.size for seg in basis.segments])
    ranked = [np.argsort(-score[offsets[i]:offsets[i + 1]], kind="stable") for i in range(len(basis.segments))]
    counts = [0] * len(basis.segments)
    limit = basis.n_elements if max_modes is None else min(int(max_modes), basis.n_elements)
    y_full = None if G is None else I_full @ np.asarray(G, dtype=float).T
    history: list[dict[str, Any]] = []

    def _evaluate(keep: tuple[np.ndarray, ...]) -> tuple[float, np.ndarray]:
        total = sum(int(k.size) for k in keep)
        if total == 0:
            I_red = np.zeros_like(I_full)
        else:
            ops = reduced_operators(basis, R_mat, M_mat, L_mat, keep)
            _, I_red = solve_reduced_eddy(ops, drive, time, V=basis.V(keep), dt_sub=dt_sub)
        err = I_red - I_full
        if metric == "dissipation":
            ref = np.sum(r[None, :] * I_full**2)
            per_segment = np.array([np.sum(r[None, seg.index] * err[:, seg.index]**2) for seg in basis.segments])
            return float(np.sqrt(per_segment.sum() / max(ref, 1e-300))), per_segment / max(ref, 1e-300)
        y_err = err @ np.asarray(G, dtype=float).T
        ref = np.sum(y_full**2)
        # a segment's share of the output error: the output of its own error current
        per_segment = np.array([
            np.sum((err[:, seg.index] @ np.asarray(G, dtype=float)[:, seg.index].T)**2) for seg in basis.segments
        ])
        return float(np.sqrt(np.sum(y_err**2) / max(ref, 1e-300))), per_segment / max(ref, 1e-300)

    while True:
        keep = tuple(np.sort(ranked[i][:counts[i]]).astype(np.int64) for i in range(len(basis.segments)))
        value, per_segment = _evaluate(keep)
        total = sum(counts)
        history.append({"M_repr": tuple(counts), "M_total": total, metric: value,
                        "by_segment": {seg.id: float(v) for seg, v in zip(basis.segments, per_segment)}})
        if value <= tolerance or total >= limit:
            return keep, history
        open_segments = [i for i, seg in enumerate(basis.segments) if counts[i] < seg.size]
        target = max(open_segments, key=lambda i: per_segment[i])
        counts[target] = min(counts[target] + max(int(step), 1), basis.segments[target].size)


def orthonormalize_r(X: np.ndarray, R_mat: np.ndarray, *, rtol: float = 1e-10) -> np.ndarray:
    """An R-orthonormal basis of ``span(X)``, canonical in sign.

    QR of ``R^{1/2} X`` with the dependent columns dropped, so a set of patterns
    that repeats a direction does not return a singular basis.

    Parameters
    ----------
    X : numpy.ndarray
        Patterns as columns, ``(N, m)`` [any].
    R_mat : numpy.ndarray
        Diagonal loop resistance, ``(N, N)`` or its diagonal [Ohm].
    rtol : float, optional
        A column whose triangular-factor diagonal falls below this times the
        largest is dependent and is dropped [-].

    Returns
    -------
    numpy.ndarray
        R-orthonormal, sign-canonical basis, ``(N, m')`` with ``m' <= m``
        [A/sqrt(Ohm)].

    Defaults
    --------
    ``rtol = 1e-10`` is a numerical convenience: the rank cut of the QR in double
    precision.

    Convention
    ----------
    Modes are R-orthonormal (``v^T R v = 1``) and sign-canonical
    (:func:`canonical_sign`), so amplitudes carry units of sqrt(W) and ``a^T a``
    is the wall current's ohmic dissipation.  This is the same normalization the eigenbasis carries, which
    is what lets a pattern basis and an eigenbasis be combined in one reduction
    (:func:`combined_operators`).

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [473] Issue #473, which built the segment-wise eigenbasis, and #500, which
       added the selection, scoring and moment tools.
    .. [vfit8] The reduced vessel-wall contract, VEST-Tokamak/vfit#8: local
       eigenbasis, global electromagnetic dynamics.
    """
    r = _diagonal_resistance(R_mat, where="R_mat")
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] != r.size:
        raise WallModeError(f"patterns have shape {X.shape}; expected ({r.size}, m)")
    w = np.sqrt(r)[:, None]
    Q, T = np.linalg.qr(w * X)
    diag = np.abs(np.diag(T))
    independent = diag > rtol * max(diag.max() if diag.size else 0.0, 1e-300)
    return canonical_sign(Q[:, independent] / w)


def moment_patterns(R_mat: np.ndarray, M_mat: np.ndarray, L_mat: np.ndarray, order: int = 1) -> np.ndarray:
    """Drive-independent wall patterns from the source coupling.

    An R-orthonormal basis of the block Krylov space
    ``span{R^{-1} M, (R^{-1} L) R^{-1} M, ...}`` up to ``order`` blocks.  The
    first block is the resistive limit -- the wall current a constant source ramp
    settles into, ``I_w = -R^{-1} M dI_src/dt`` -- and each further block is the
    next inductive correction of the slowly driven response: the Laplace-domain
    moments of the wall's transfer function, matched at zero frequency.

    Built as a block Arnoldi iteration in the R inner product, so the blocks stay
    independent where the raw powers would collapse onto the slowest mode; a
    block that adds no new direction ends the iteration early and the basis is
    narrower than ``order * n_src``.

    Parameters
    ----------
    R_mat : numpy.ndarray
        Diagonal loop resistance, ``(N, N)`` or its diagonal [Ohm].
    M_mat : numpy.ndarray
        Passive-passive inductance in code naming; the physics ``L``, ``(N, N)``
        [H].
    L_mat : numpy.ndarray
        Passive-to-source coupling in code naming; the physics ``M``,
        ``(N, n_src)`` [H].
    order : int, optional
        How many Krylov blocks to build [-].

    Returns
    -------
    numpy.ndarray
        R-orthonormal patterns as columns, ``(N, m)`` with
        ``m <= order * n_src`` [A/sqrt(Ohm)].

    Convention
    ----------
    Argument naming follows :mod:`vaft.process.electromagnetics`, not the
    physics: ``M_mat`` is the passive-passive inductance (physics ``L``) and
    ``L_mat`` the passive-to-source coupling (physics ``M``).  ``R_mat`` is the
    diagonal loop resistance either way.

    Applicability
    -------------
    Machine-independent.  The comparison quoted below is a VEST observation, not
    a property of the method.

    Limitations
    -----------
    A zero-frequency expansion: on the packaged VEST wall ten resistive patterns
    reproduce the probe response of a real PF drive to about 1 %, where 150
    eigenmodes are needed for the same, but fast transients remain the
    eigenmodes' territory.

    Provenance
    ----------
    .. [473] Issue #473, which built the segment-wise eigenbasis, and #500, which
       added the selection, scoring and moment tools.
    .. [vfit8] The reduced vessel-wall contract, VEST-Tokamak/vfit#8: local
       eigenbasis, global electromagnetic dynamics.
    .. [vfit10] The reduced-order validation study (VEST-Tokamak/vfit#10, vaft
       #494), which asks which modes to keep; this module answers no part of that
       question by itself.
    """
    if int(order) < 1:
        raise WallModeError("order must be at least 1")
    r = _diagonal_resistance(R_mat, where="R_mat")
    L = np.asarray(M_mat, dtype=float)
    Q = orthonormalize_r(np.asarray(L_mat, dtype=float) / r[:, None], R_mat)
    blocks = [Q]
    for _ in range(int(order) - 1):
        X = (L @ blocks[-1]) / r[:, None]
        V = np.hstack(blocks)
        X = X - V @ (V.T @ (r[:, None] * X))          # R-orthogonal to everything so far
        X = X - V @ (V.T @ (r[:, None] * X))          # twice, for the usual reason
        Q = orthonormalize_r(X, R_mat, rtol=1e-8)
        if Q.shape[1] == 0:
            break
        blocks.append(Q)
    return canonical_sign(np.hstack(blocks))


def combined_operators(
    V: np.ndarray,
    R_mat: np.ndarray,
    M_mat: np.ndarray,
    L_mat: np.ndarray | None = None,
    *,
    label: str = "pattern",
) -> ReducedWall:
    """Reduced operators for an arbitrary R-orthonormal basis.

    The same projections as :func:`reduced_operators`, for a basis that is not
    (only) segment eigenmodes -- an enrichment by :func:`moment_patterns`, say,
    or a POD basis.

    Input semantics
    ---------------
    Element space: the assembled wall matrices and a basis whose columns are wall
    patterns.

    Output semantics
    ----------------
    Mode space: operators of the basis's own coordinates, labelled ``(label, k)``
    rather than by segment.

    Parameters
    ----------
    V : numpy.ndarray
        R-orthonormal basis as columns, ``(N, m)`` [A/sqrt(Ohm)].
    R_mat : numpy.ndarray
        Diagonal loop resistance, ``(N, N)`` or its diagonal [Ohm].
    M_mat : numpy.ndarray
        Passive-passive inductance in code naming, ``(N, N)`` [H].
    L_mat : numpy.ndarray or None, optional
        Passive-to-source coupling in code naming, ``(N, n_src)`` [H].
    label : str, optional
        Label component of every coefficient's ``(label, k)`` name [-].

    Returns
    -------
    ReducedWall
        ``L_r`` [H], ``R_r`` [Ohm], ``M_r`` [H] when a source coupling was given,
        and the labels [-].

    Convention
    ----------
    ``R_r`` is computed, not assumed: a basis that is not R-orthonormal in the
    matrices passed here shows up as a non-identity rather than silently changing
    the dynamics.  Use :func:`orthonormalize_r` to make one that is.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [473] Issue #473, which built the segment-wise eigenbasis, and #500, which
       added the selection, scoring and moment tools.
    .. [vfit8] The reduced vessel-wall contract, VEST-Tokamak/vfit#8: local
       eigenbasis, global electromagnetic dynamics.
    """
    V = np.asarray(V, dtype=float)
    r = _diagonal_resistance(R_mat, where="R_mat")
    if V.ndim != 2 or V.shape[0] != r.size:
        raise WallModeError(f"basis has shape {V.shape}; expected ({r.size}, m)")
    L = np.asarray(M_mat, dtype=float)
    return ReducedWall(
        L_r=V.T @ L @ V,
        R_r=V.T @ (r[:, None] * V),
        M_r=None if L_mat is None else V.T @ np.asarray(L_mat, dtype=float),
        labels=tuple((str(label), k) for k in range(V.shape[1])),
        keep=(np.arange(V.shape[1], dtype=np.int64),),
    )
