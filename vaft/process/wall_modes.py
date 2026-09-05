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
    """Return ``(L_sym, asymmetry)`` where ``asymmetry = max|L - L^T| / max|L|``.

    Reciprocity makes a mutual-inductance matrix symmetric; a departure is a
    defect in the asset, not noise (vaft #347).  Below ``rtol`` it is rounding
    and is folded silently; up to ``reject`` it is folded with a warning so
    the provenance can record it; above that the basis refuses, because the
    sanctioned place to repair reciprocity is the coupling mapper
    (:func:`vaft.machine_mapping.em_coupling.em_coupling`, #373), not a
    consumer.
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

    Ties go to the lowest index (``argmax`` on ``|v|``), so the rule is a pure
    function of the column and repeated runs agree bitwise.
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

    ``R_gg`` is the segment's diagonal resistance (matrix or vector), ``L_gg``
    its symmetric inductance block.  Returns ``(tau, V, residual)``: decay
    times descending, R-orthonormal sign-canonical modes as columns, and the
    relative pencil residual ``max|R V - L V / tau| / max|R V|``.

    Refuses explicitly rather than returning something plausible: a
    non-positive eigenvalue means the block is not an inductance (a current
    pattern with no stored energy), and a spread beyond ``cond_max`` means
    the fastest mode is numerically indistinguishable from zero.
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

    ``R_mat`` is the diagonal loop resistance (code naming), ``M_mat`` the
    passive-passive inductance (code naming; physics ``L``).  ``segments`` is
    a sequence of objects with ``.id`` and ``.index`` (``WallSegment``) or of
    ``(id, index)`` pairs; together they must cover every element exactly
    once.

    A within-segment pair of decay times closer than ``cluster_rtol``
    (relative) is a near-degenerate eigenspace, inside which the individual
    modes are arbitrary up to rotation; by default the basis refuses so the
    provenance never claims a determinism it does not have.  Pass
    ``on_cluster="warn"`` to record the pairs instead.
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
    """Project the FULL wall matrices onto the retained modes.

    ``M_mat`` is the passive-passive inductance and ``L_mat`` the passive-to-
    source coupling ``(N, n_src)``, in :mod:`vaft.process.electromagnetics`'s
    naming.  ``R_r`` is computed rather than assumed to be the identity, so a
    basis built on other matrices shows up as a non-identity.
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
    """Modal amplitudes ``a = V^T R I_w`` [sqrt(W)].

    Exact for a current inside the retained subspace; the R-orthogonal
    projection otherwise.  ``I_w`` is ``(N,)`` or ``(n_times, N)``.
    """
    V = basis.V(keep)
    r = _diagonal_resistance(R_mat, where="R_mat")
    I_w = np.asarray(I_w, dtype=float)
    if I_w.ndim == 1:
        return V.T @ (r * I_w)
    return (I_w * r[None, :]) @ V


def reconstruct(basis: WallModeBasis, a: np.ndarray, keep: Sequence[np.ndarray] | None = None) -> np.ndarray:
    """Wall currents ``I_w = V a`` [A]; ``a`` is ``(M_tot,)`` or ``(n_times, M_tot)``."""
    V = basis.V(keep)
    a = np.asarray(a, dtype=float)
    return V @ a if a.ndim == 1 else a @ V.T


def reconstruction_error(
    I_w: np.ndarray, I_rec: np.ndarray, R_mat: np.ndarray, basis: WallModeBasis | None = None
) -> dict[str, Any]:
    """How far a reconstruction is from the full current, in the norms that matter.

    ``relative_l2`` is the Euclidean measure; ``relative_dissipation`` weighs
    each element by its resistance -- the natural norm of this basis, where
    the error's ohmic power is compared to the current's.  Per-segment
    entries follow when a basis is given.
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
    """``G_red = G_full V_seg`` for any response with the wall elements as columns.

    Probe field, flux-loop flux, grid psi/B_R/B_Z, EFIT tables -- one matmul,
    no per-class logic, so every response class is reduced by the same basis.
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
    """Keep every mode of every segment (the full-rank selection)."""
    return basis._keep(None)


def select_slowest(basis: WallModeBasis, M: int | Sequence[int]) -> tuple[np.ndarray, ...]:
    """Keep the ``M`` slowest modes: globally across segments (an ``int``), or
    ``M_g`` per segment (a sequence, one entry per segment)."""
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
    """Keep the modes whose decay time lies in ``[tau_min, tau_max]``, per segment."""
    return tuple(
        np.flatnonzero((seg.tau >= tau_min) & (seg.tau <= tau_max)).astype(np.int64)
        for seg in basis.segments
    )


def global_time_constants(basis: WallModeBasis, M_mat: np.ndarray) -> np.ndarray:
    """The whole wall's decay times, descending, from the full-rank reduced
    inductance: ``eigvalsh(L_r)`` with ``R_r = I`` is the global pencil."""
    V = basis.V(None)
    L_r = V.T @ np.asarray(M_mat, dtype=float) @ V
    L_r = 0.5 * (L_r + L_r.T)
    return np.sort(np.linalg.eigvalsh(L_r))[::-1]


def check_wall_mode_basis(basis: WallModeBasis, R_mat: np.ndarray, M_mat: np.ndarray) -> dict[str, Any]:
    """Metrics of a basis against the matrices it claims to diagonalize; no verdict.

    ``r_r_identity_error`` is ``max|V^T R V - I|``; ``coupling`` holds the
    Frobenius norm of every off-diagonal block of ``L_r`` relative to the
    geometric mean of the diagonal blocks, so a reader can see which
    segments actually talk to each other.
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
    """Principal angles [rad] between two mode subspaces in the R inner product.

    Two bases of a near-degenerate eigenspace differ by a rotation, so the
    right question is whether they span the same space, not whether the
    columns match.  Both sets are mapped by ``R^{1/2}`` so the ordinary
    Euclidean angles are the R-metric ones.
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
    (:func:`vaft.process.electromagnetics.solve_eddy_currents`), fed the
    projected operators, so the only difference between the two solutions is
    the retained subspace: with every mode kept they agree to rounding.
    ``drive`` is ``(n_times, n_src)`` on ``time``.  Returns the amplitudes
    ``a`` ``(n_times, M_tot)`` [sqrt(W)] and, when the retained ``V`` is
    given, the reconstructed wall current ``I_w = V a`` ``(n_times, N)`` [A].
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
    """Rankings of the retained modes, one value per coefficient of ``labels(keep)``.

    Rankings, not verdicts: each is a different answer to "which modes carry
    the wall's response", and the order study (vfit #10) compares them.

    ``tau``
        the decay time -- what the spectrum alone would rank by;
    ``drive_gain``
        ``tau_k ||M_r[k, :]|| ||G_red[:, k]||`` -- the quasi-static amplitude a
        unit source ramp excites in mode ``k`` times how visible the mode is
        at the observation points (all ones without ``G``); needs no drive;
    ``response_energy``
        the rms of the projected full response ``a = V^T R I_w(t)`` under
        ``drive`` -- how much dissipation each mode actually carried;
    ``output_weight``
        ``response_energy`` times the observability, the ranking that
        minimizes the diagnostic-space error fastest on the packaged wall.

    The last two need ``drive`` ``(n_times, n_src)`` and ``time`` and cost
    one full wall solve.
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
    """Keep the ``M`` highest-scoring modes across segments.

    ``score`` is aligned with ``basis.labels(keep)`` (one entry per coefficient
    of the candidate selection, every mode by default); ties resolve toward
    the earlier label so the selection is deterministic.
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
    """Greedy segment-wise allocation ``M_repr = (M_1, ..., M_G)`` to a tolerance.

    Starting from no modes, each round re-solves the reduced wall under
    ``drive`` and adds ``step`` modes to the segment carrying the largest
    remaining error, taken in the order of ``score`` within that segment
    (the ``output_weight`` ranking of :func:`mode_scores` by default, which
    needs no more than the full solve already made here).  It stops when the
    global ``metric`` drops to ``tolerance`` or every candidate is used.

    ``metric="dissipation"`` is the relative R-energy error of the wall
    current, the norm this basis is built in; ``metric="output"`` is the
    relative error of ``G I_w`` and needs ``G``.  The per-segment error is
    the segment's share of the global squared error in the same norm, so a
    segment that the drive barely reaches never attracts modes.

    Returns the selection and the history: one row per round with the
    running ``M_repr``, its total, and the metric.  A global allocation for
    a *given* total is simply :func:`select_by_score`; this routine answers
    the other question, the smallest total that meets a response tolerance.
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

    QR of ``R^{1/2} X`` with the dependent columns (diagonal of the triangular
    factor below ``rtol`` times its largest entry) dropped, so a set of
    patterns that repeats a direction does not return a singular basis.
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
    """Drive-independent wall patterns from the source coupling: an R-orthonormal
    basis of the block Krylov space ``span{R^{-1} M, (R^{-1} L) R^{-1} M, ...}``
    up to ``order`` blocks.

    The first block is the resistive limit -- the wall current a constant
    source ramp settles into, ``I_w = -R^{-1} M dI_src/dt`` -- and each
    further block is the next inductive correction of the slowly driven
    response (the Laplace-domain moments of the wall's transfer function,
    matched at zero frequency).  Built as a block Arnoldi iteration in the R
    inner product, so the blocks stay independent where the raw powers would
    collapse onto the slowest mode; a block that adds no new direction ends
    the iteration early and the basis is narrower than ``order * n_src``.

    On the packaged VEST wall ten resistive patterns reproduce the probe
    response of a real PF drive to ~1 % where 150 eigenmodes are needed for
    the same; fast transients remain the eigenmodes' territory.  ``L`` is the
    physics inductance (code ``M_mat``), ``M`` the source coupling (code
    ``L_mat``), matching the module convention.
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
    """Reduced operators for an arbitrary R-orthonormal basis ``V`` ``(N, m)``.

    The same projections as :func:`reduced_operators`, for a basis that is
    not (only) segment eigenmodes -- an enrichment by
    :func:`moment_patterns`, say, or a POD basis; coefficients are labelled
    ``(label, k)``.  ``R_r`` is computed, not assumed.
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
