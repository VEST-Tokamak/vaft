"""Deterministic segmentation of the VEST passive wall (vfit #8/#9, vaft #473).

The reduced vessel-wall contract asks for "approximately ten physically
meaningful conductor segments", a deterministic map from every vessel element
to exactly one of them, and segment identity as part of the basis provenance.
VAFT's wall is ~950 filament rectangles under ``pf_passive.loop.*`` carrying a
structure name (``W1``..``W11``) and nothing else, so the segmentation is
derived, not stored:

- loops are grouped by ``name`` -- index order is irrelevant, several
  structures interleave in the packaged geometry;
- a group whose loops leave a genuine gap in Z (larger than ``GAP_FACTOR``
  times the group's median loop height) is split at that gap, because the two
  pieces are electrically disconnected conductors that merely share a label
  (the top and bottom lids of ``W2``, for instance).  This is the rule the
  TokaMaker vessel export already applies, factored out here so the two codes
  cannot disagree about what a conductor is.

Every loop lands in exactly one segment; ``W11`` (the inboard limiter tiles)
is a segment like any other here even though the TokaMaker mesh leaves it
out, since the contract is about the current vector the eddy solve actually
carries.  Segments are ordered by their first element index, so the
``(segment, mode)`` indexing of a reduced basis is reproducible from the loop
order alone.

The rule is versioned (:data:`SEGMENT_DEFINITION_VERSION`) and the outcome is
fingerprinted (:func:`segment_digest`), so a basis records exactly which
segmentation it was built on.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

__all__ = [
    "GAP_FACTOR",
    "SEGMENT_DEFINITION_VERSION",
    "WallSegment",
    "cluster_by_z_gap",
    "loop_extents",
    "segment_digest",
    "segment_loops",
    "segment_membership",
]

#: The rule's identity: name grouping, Z-gap split at ``GAP_FACTOR`` median
#: heights, revision 1.  Bump the revision when the rule changes.
SEGMENT_DEFINITION_VERSION = "vest-name-zgap-1.5-v1"

#: A Z gap larger than this multiple of a group's median loop height marks a
#: disconnected conductor.  Shared with the TokaMaker vessel export.
GAP_FACTOR = 1.5


@dataclass(frozen=True)
class WallSegment:
    """One conductor segment: which loops, and where it sits."""

    id: str
    """``W1``, ``W2_L``, ``W2_U``, ... -- the structure name plus a cluster
    suffix when the structure was split."""
    name: str
    """The source ``pf_passive.loop.*.name`` (``W2`` for both ``W2_L`` and
    ``W2_U``)."""
    index: np.ndarray
    """Sorted element indices into ``pf_passive.loop``, shape ``(n_g,)``."""
    r_center: float
    z_center: float

    def __post_init__(self) -> None:
        index = np.asarray(self.index, dtype=np.int64).reshape(-1)
        if index.size == 0:
            raise ValueError(f"segment {self.id!r} has no elements")
        index = np.sort(index)
        index.setflags(write=False)
        object.__setattr__(self, "index", index)

    @property
    def size(self) -> int:
        return int(self.index.size)


def loop_extents(ods: Any) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Names and poloidal extents of every ``pf_passive`` loop.

    Returns ``(names, z_lo, z_hi, r_center, z_center)``.  A loop's extent
    comes from its first element's ``outline`` when present, else from its
    ``rectangle``.  A loop without a name is refused: the segmentation is by
    name, and a nameless loop would silently become its own segment.
    """
    from vaft.ods_access import path_value

    n_loops = len(ods["pf_passive.loop"])
    names: list[str] = []
    z_lo = np.empty(n_loops)
    z_hi = np.empty(n_loops)
    r_center = np.empty(n_loops)
    z_center = np.empty(n_loops)
    for i in range(n_loops):
        base = f"pf_passive.loop.{i}"
        raw = path_value(ods, f"{base}.name")
        name = str(raw).strip().upper() if raw is not None else ""
        if not name:
            raise ValueError(f"pf_passive.loop.{i} carries no name; the wall is segmented by name")
        names.append(name)
        geometry = f"{base}.element.0.geometry"
        outline_z = path_value(ods, f"{geometry}.outline.z")
        if outline_z is not None and np.size(outline_z):
            r = np.asarray(path_value(ods, f"{geometry}.outline.r"), dtype=float)
            z = np.asarray(outline_z, dtype=float)
            z_lo[i], z_hi[i] = float(z.min()), float(z.max())
            r_center[i], z_center[i] = float(r.mean()), float(z.mean())
            continue
        rc = path_value(ods, f"{geometry}.rectangle.r")
        zc = path_value(ods, f"{geometry}.rectangle.z")
        if rc is None or zc is None:
            raise ValueError(f"pf_passive.loop.{i} has neither an outline nor a rectangle centre")
        height = path_value(ods, f"{geometry}.rectangle.height")
        h = float(height) if height is not None else 0.0
        zc = float(zc)
        z_lo[i], z_hi[i] = zc - h / 2.0, zc + h / 2.0
        r_center[i], z_center[i] = float(rc), zc
    return names, z_lo, z_hi, r_center, z_center


def cluster_by_z_gap(
    z_lo: np.ndarray, z_hi: np.ndarray, *, gap_factor: float = GAP_FACTOR
) -> list[np.ndarray]:
    """Split loops into Z-contiguous clusters, bottom-up.

    Loops are ordered by Z centre; a new cluster starts wherever the next
    loop's lower edge sits more than ``gap_factor`` median heights above the
    previous loop's upper edge.  Returns index arrays into the input, each in
    ascending Z-centre order, clusters from the bottom of the machine up.  A
    single loop is one cluster.
    """
    z_lo = np.asarray(z_lo, dtype=float).reshape(-1)
    z_hi = np.asarray(z_hi, dtype=float).reshape(-1)
    if z_lo.size != z_hi.size:
        raise ValueError("z_lo and z_hi must have the same length")
    if z_lo.size == 0:
        return []
    order = np.argsort((z_lo + z_hi) / 2.0, kind="stable")
    heights = z_hi - z_lo
    gap_tol = float(gap_factor) * float(np.median(heights))
    clusters: list[list[int]] = [[int(order[0])]]
    for prev, cur in zip(order, order[1:]):
        if z_lo[cur] - z_hi[prev] > gap_tol:
            clusters.append([int(cur)])
        else:
            clusters[-1].append(int(cur))
    return [np.asarray(c, dtype=np.int64) for c in clusters]


def segment_loops(ods: Any, *, gap_factor: float = GAP_FACTOR) -> tuple[WallSegment, ...]:
    """Segment the passive wall: by name, split at Z gaps; every loop once.

    Segments are ordered by their smallest element index.  Two clusters that
    straddle the midplane are ``<name>_L`` / ``<name>_U``; any other split is
    numbered ``<name>_1`` .. ``<name>_n`` bottom-up, so a suffix never claims a
    mirror symmetry the geometry does not have.
    """
    names, z_lo, z_hi, r_center, z_center = loop_extents(ods)
    by_name: dict[str, list[int]] = {}
    for i, name in enumerate(names):
        by_name.setdefault(name, []).append(i)

    segments: list[WallSegment] = []
    for name, members in by_name.items():
        members_arr = np.asarray(members, dtype=np.int64)
        clusters = cluster_by_z_gap(z_lo[members_arr], z_hi[members_arr], gap_factor=gap_factor)
        if len(clusters) == 1:
            ids = [name]
        elif len(clusters) == 2:
            lower, upper = clusters
            straddles = z_center[members_arr[lower]].max() < 0.0 < z_center[members_arr[upper]].min()
            ids = [f"{name}_L", f"{name}_U"] if straddles else [f"{name}_1", f"{name}_2"]
        else:
            ids = [f"{name}_{k + 1}" for k in range(len(clusters))]
        for seg_id, cluster in zip(ids, clusters):
            index = members_arr[cluster]
            segments.append(
                WallSegment(
                    id=seg_id,
                    name=name,
                    index=index,
                    r_center=float(r_center[index].mean()),
                    z_center=float(z_center[index].mean()),
                )
            )
    segments.sort(key=lambda seg: int(seg.index.min()))
    segment_membership(segments, len(names))  # every loop exactly once
    return tuple(segments)


def segment_membership(segments: Sequence[WallSegment], n_loops: int) -> np.ndarray:
    """Segment position of every loop, ``shape (n_loops,)``.

    Raises when a loop is missing from every segment or claimed by more than
    one -- the contract's "every element to exactly one segment".
    """
    membership = np.full(int(n_loops), -1, dtype=np.int64)
    for position, seg in enumerate(segments):
        if seg.index.min() < 0 or seg.index.max() >= n_loops:
            raise ValueError(f"segment {seg.id!r} indexes outside the {n_loops} loops")
        taken = membership[seg.index] >= 0
        if taken.any():
            raise ValueError(
                f"loop(s) {seg.index[taken][:5].tolist()} belong to more than one segment "
                f"({seg.id!r} and {segments[int(membership[seg.index][taken][0])].id!r})"
            )
        membership[seg.index] = position
    missing = np.flatnonzero(membership < 0)
    if missing.size:
        raise ValueError(f"loop(s) {missing[:5].tolist()} belong to no segment")
    return membership


def segment_digest(segments: Sequence[WallSegment]) -> str:
    """12-hex fingerprint of the segment ids and their element indices."""
    digest = hashlib.sha1()
    for seg in segments:
        digest.update(seg.id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(np.asarray(seg.index, dtype=np.int64).tobytes())
        digest.update(b"\1")
    return digest.hexdigest()[:12]
