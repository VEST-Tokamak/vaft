"""Scan-grade topology classification on top of the #65 boundary descriptors.

``vaft.process.equilibrium.derive_boundary_representation`` (issue #65 / PR
#204) finds X-points, dRsep, gaps, and strike points for one equilibrium, but
its topology enum has no *near-null* state, classifies "limited" only when no
X-point exists anywhere, and hinges on a single ``flux_tolerance``. Free-
boundary scans (issue #67) need to walk ``limited -> near-null -> diverted``
robustly, so this module derives the classification directly from the found
X-points' ``psi_n`` with two explicit thresholds, adds limiter-contact
detection for non-diverted states, and returns a JSON-safe report suitable
for per-case scan manifests. The #65 machinery itself is not modified.

Classification rules (``axis_z`` = magnetic-axis height, fallback 0):

- an X-point is **active** when ``|psi_n - 1| <= active_tolerance``;
- active nulls above and below ``axis_z``  -> ``DOUBLE_NULL``;
- active nulls on one side only            -> ``UPPER/LOWER_SINGLE_NULL``;
- no active null but one within ``near_null_band`` -> ``NEAR_NULL``;
- otherwise                                -> ``LIMITED``;
- an upstream failure (unreadable source, no LCFS) -> ``UNKNOWN`` with the
  reason preserved — a scan records it instead of raising.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional

import numpy as np

_log = logging.getLogger(__name__)


class ScanTopology(str, Enum):
    LIMITED = "limited"
    NEAR_NULL = "near_null"
    LOWER_SINGLE_NULL = "lower_single_null"
    UPPER_SINGLE_NULL = "upper_single_null"
    DOUBLE_NULL = "double_null"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class LimiterContact:
    """Closest approach between the LCFS and the limiter."""

    r: float                  # LCFS-side point [m]
    z: float
    wall_r: float             # nearest wall point [m]
    wall_z: float
    distance: float           # [m]; ~0 for a truly limited plasma


@dataclass
class TopologyReport:
    """JSON-safe per-equilibrium topology classification for scan manifests."""

    topology: ScanTopology
    x_points: tuple[Mapping[str, float], ...] = ()   # {r, z, psi_n, active}
    d_r_sep: Optional[float] = None                  # [m]; sign: + = upper outboard of lower
    null_margin: Optional[float] = None              # min |psi_n - 1| over vacuum-side X-points (psi_n >= 1 - tol)
    limiter_contact: Optional[LimiterContact] = None
    active_tolerance: float = 0.0
    near_null_band: float = 0.0
    reason: Optional[str] = None
    representation: Any = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        contact = None
        if self.limiter_contact is not None:
            contact = {
                "r": self.limiter_contact.r,
                "z": self.limiter_contact.z,
                "wall_r": self.limiter_contact.wall_r,
                "wall_z": self.limiter_contact.wall_z,
                "distance": self.limiter_contact.distance,
            }
        return {
            "topology": self.topology.value,
            "x_points": [dict(xp) for xp in self.x_points],
            "d_r_sep": self.d_r_sep,
            "null_margin": self.null_margin,
            "limiter_contact": contact,
            "active_tolerance": self.active_tolerance,
            "near_null_band": self.near_null_band,
            "reason": self.reason,
        }


def _min_distance_to_polyline(
    points_r: np.ndarray, points_z: np.ndarray, poly_r: np.ndarray, poly_z: np.ndarray
) -> tuple[float, tuple[float, float], tuple[float, float]]:
    """Minimum distance from a point set to a closed polyline.

    Returns ``(distance, (r, z) of the closest point-set point, (r, z) of the
    nearest point on the polyline)``. Vectorized point-to-segment distances.
    """
    p = np.column_stack([points_r, points_z])                       # (N, 2)
    a = np.column_stack([poly_r, poly_z])                           # (M, 2)
    b = np.roll(a, -1, axis=0)
    ab = b - a                                                      # (M, 2)
    ab_len2 = np.einsum("ij,ij->i", ab, ab)
    ab_len2 = np.where(ab_len2 > 0.0, ab_len2, 1.0)                 # degenerate segments

    ap = p[:, None, :] - a[None, :, :]                              # (N, M, 2)
    t = np.clip(np.einsum("nmj,mj->nm", ap, ab) / ab_len2[None, :], 0.0, 1.0)
    closest = a[None, :, :] + t[..., None] * ab[None, :, :]         # (N, M, 2)
    dist = np.linalg.norm(p[:, None, :] - closest, axis=2)          # (N, M)

    n_idx, m_idx = np.unravel_index(int(np.argmin(dist)), dist.shape)
    wall = closest[n_idx, m_idx]
    return float(dist[n_idx, m_idx]), (float(p[n_idx, 0]), float(p[n_idx, 1])), (
        float(wall[0]), float(wall[1]))


def classify_boundary(
    source: Any,
    *,
    active_tolerance: float = 2.0e-3,
    near_null_band: float = 5.0e-2,
    time_index: int = 0,
    gap_angles: Mapping[str, float] | None = None,
) -> TopologyReport:
    """Classify one equilibrium's boundary topology for scan bookkeeping.

    ``source`` is anything ``vaft.process.equilibrium.as_equilibrium`` accepts
    (g-file path, GEQDSK, ODS, EquilibriumData, ...). Never raises: unreadable
    or descriptor-less equilibria come back as ``UNKNOWN`` with a reason, so a
    scan can record the state and continue.
    """
    from vaft.process.equilibrium import as_equilibrium, derive_boundary_representation

    try:
        equilibrium = as_equilibrium(source, time_index=time_index)
        representation = derive_boundary_representation(
            equilibrium,
            flux_tolerance=active_tolerance,
            gap_angles=gap_angles,
        )
    except Exception as exc:
        _log.warning("Boundary classification failed: %s", exc)
        return TopologyReport(
            topology=ScanTopology.UNKNOWN,
            active_tolerance=active_tolerance,
            near_null_band=near_null_band,
            reason=str(exc),
        )

    x_points = tuple(
        {
            "r": float(xp.r), "z": float(xp.z), "psi_n": float(xp.psi_n),
            "active": bool(abs(xp.psi_n - 1.0) <= active_tolerance),
        }
        for xp in representation.x_points
    )
    # The approach distance counts only nulls on the vacuum side of the LCFS
    # (psi_n coming DOWN toward 1). A far-side vacuum saddle can carry
    # psi_n < 1 without being anywhere near the plasma boundary, and must not
    # register as an approaching null.
    approaching = [
        xp["psi_n"] - 1.0
        for xp in x_points
        if xp["psi_n"] >= 1.0 - active_tolerance
    ]
    null_margin = min(abs(m) for m in approaching) if approaching else None
    d_r_sep = (
        float(representation.d_r_sep.value)
        if representation.d_r_sep is not None and representation.d_r_sep.available
        else None
    )

    axis = equilibrium.magnetic_axis
    axis_z = float(axis[1]) if axis is not None else 0.0

    reason = representation.reason
    lcfs = representation.lcfs
    if lcfs is None:
        topology = ScanTopology.UNKNOWN
        reason = reason or "no LCFS contour available"
    else:
        actives = [xp for xp in x_points if xp["active"]]
        upper = [xp for xp in actives if xp["z"] >= axis_z]
        lower = [xp for xp in actives if xp["z"] < axis_z]
        if upper and lower:
            topology = ScanTopology.DOUBLE_NULL
        elif upper:
            topology = ScanTopology.UPPER_SINGLE_NULL
        elif lower:
            topology = ScanTopology.LOWER_SINGLE_NULL
        elif null_margin is not None and null_margin <= near_null_band:
            topology = ScanTopology.NEAR_NULL
        else:
            topology = ScanTopology.LIMITED

    limiter_contact = None
    limiter = representation.limiter
    if (
        topology in (ScanTopology.LIMITED, ScanTopology.NEAR_NULL)
        and lcfs is not None
        and limiter is not None
        and len(lcfs.r) >= 1
        and len(limiter.r) >= 2
    ):
        try:
            distance, plasma_pt, wall_pt = _min_distance_to_polyline(
                lcfs.r, lcfs.z, limiter.r, limiter.z
            )
            limiter_contact = LimiterContact(
                r=plasma_pt[0], z=plasma_pt[1],
                wall_r=wall_pt[0], wall_z=wall_pt[1],
                distance=distance,
            )
        except Exception as exc:  # pragma: no cover - defensive
            _log.warning("Limiter-contact search failed: %s", exc)

    return TopologyReport(
        topology=topology,
        x_points=x_points,
        d_r_sep=d_r_sep,
        null_margin=null_margin,
        limiter_contact=limiter_contact,
        active_tolerance=active_tolerance,
        near_null_band=near_null_band,
        reason=reason,
        representation=representation,
    )
