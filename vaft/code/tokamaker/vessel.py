"""Synthesize TokaMaker conductor regions from the VEST ``pf_passive`` filaments.

VAFT has no continuous vessel description: the shell exists only as ~950 small
filament rectangles (``pf_passive.loop.*``) grouped into named segments
(``W1``..``W11``). This module turns each segment into one or two continuous
conductor polygons suitable for ``gs_Domain.define_region(..., 'conductor')``:

- loops are grouped by ``name`` (index order is irrelevant — several segments
  interleave in the packaged file);
- a segment whose loops are contiguous through Z = 0 (outer cylinder ``W1``,
  central column ``W4``) stays one region; segments with a genuine Z gap are
  split into mirrored ``_U``/``_L`` regions;
- each region's outline is a monotone strip trace along its dominant axis —
  an exact staircase for stepped strips (cones), degenerating to the bounding
  rectangle for straight runs — validated against the summed loop area with a
  bounding-box fallback;
- resistivity defaults to ``TokaMakerConfig.eta_vessel`` (7.8e-7 Ohm·m
  SUS316LN, which exactly reproduces the packaged W2–W10 loop resistances via
  ``R = 2*pi*R*eta/A``) with per-segment/region overrides. The outboard wall
  ``W1`` carries a black-box per-loop calibration in the ODS (issue #191);
  its ODS median is recorded for provenance but NOT applied automatically.

``W11`` (0.1 mm tungsten inboard limiter tiles) is excluded by default: it is
a degenerate sliver and tiles, not vessel structure.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .config import TokaMakerConfig

_log = logging.getLogger(__name__)

# A Z gap larger than this multiple of the median loop height splits a segment
# into separate regions.
_GAP_FACTOR = 1.5
# Accept a traced polygon when its shoelace area matches the summed loop area
# within this relative tolerance; otherwise fall back to the bounding box.
_AREA_RTOL = 0.15


def _loop_rectangles(ods: Any) -> dict[str, list[dict[str, float]]]:
    """Group pf_passive loops into per-segment lists of rectangle descriptors."""
    nloop = len(ods["pf_passive.loop"])
    segments: dict[str, list[dict[str, float]]] = {}
    for i in range(nloop):
        try:
            name = str(ods[f"pf_passive.loop.{i}.name"]).strip().upper()
        except Exception:
            name = f"LOOP{i}"
        r = np.asarray(ods[f"pf_passive.loop.{i}.element.0.geometry.outline.r"], dtype=float)
        z = np.asarray(ods[f"pf_passive.loop.{i}.element.0.geometry.outline.z"], dtype=float)
        rect = {
            "r_lo": float(r.min()), "r_hi": float(r.max()),
            "z_lo": float(z.min()), "z_hi": float(z.max()),
        }
        try:
            rect["resistivity"] = float(ods[f"pf_passive.loop.{i}.resistivity"])
        except Exception:
            rect["resistivity"] = float("nan")
        segments.setdefault(name, []).append(rect)
    return segments


def _split_z_clusters(rects: list[dict[str, float]]) -> list[list[dict[str, float]]]:
    """Split a segment's rectangles at genuine Z gaps (mirrored halves)."""
    rects = sorted(rects, key=lambda rect: (rect["z_lo"] + rect["z_hi"]) / 2.0)
    heights = np.array([rect["z_hi"] - rect["z_lo"] for rect in rects])
    gap_tol = _GAP_FACTOR * float(np.median(heights))
    clusters: list[list[dict[str, float]]] = [[rects[0]]]
    for prev, rect in zip(rects, rects[1:]):
        if rect["z_lo"] - prev["z_hi"] > gap_tol:
            clusters.append([rect])
        else:
            clusters[-1].append(rect)
    return clusters


def _dedupe_chain(points: list[tuple[float, float]], tol: float = 1e-9) -> list[tuple[float, float]]:
    """Drop consecutive duplicates and interior points of axis-aligned runs."""
    if len(points) < 3:
        return points
    out = [points[0]]
    for pt in points[1:]:
        if abs(pt[0] - out[-1][0]) < tol and abs(pt[1] - out[-1][1]) < tol:
            continue
        if len(out) >= 2:
            a, b = out[-2], out[-1]
            same_x = abs(a[0] - b[0]) < tol and abs(b[0] - pt[0]) < tol
            same_y = abs(a[1] - b[1]) < tol and abs(b[1] - pt[1]) < tol
            if same_x or same_y:
                out[-1] = pt
                continue
        out.append(pt)
    return out


def _shoelace_area(contour: np.ndarray) -> float:
    x, y = contour[:, 0], contour[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _bounding_box_contour(rects: list[dict[str, float]]) -> np.ndarray:
    r_lo = min(rect["r_lo"] for rect in rects)
    r_hi = max(rect["r_hi"] for rect in rects)
    z_lo = min(rect["z_lo"] for rect in rects)
    z_hi = max(rect["z_hi"] for rect in rects)
    return np.array([[r_lo, z_lo], [r_hi, z_lo], [r_hi, z_hi], [r_lo, z_hi]])


def _rect_area(rect: dict[str, float]) -> float:
    return (rect["r_hi"] - rect["r_lo"]) * (rect["z_hi"] - rect["z_lo"])


def _trace_strip(rects: list[dict[str, float]], region: str) -> tuple[np.ndarray, bool]:
    """Monotone staircase outline of a strip of rectangles.

    The strip is traversed along its dominant axis (larger center spread):
    one chain follows the high transverse edges, the return chain the low
    edges. For straight runs the result collapses to the bounding rectangle;
    for stepped strips (cones, shoulders) it reproduces the staircase. When
    the traced area disagrees with the summed rectangle area (both computed
    from the rectangles *as passed*, i.e. after any shrink/clamp) by more than
    ``_AREA_RTOL`` — non-monotone geometry — the bounding box is used instead.

    Returns ``(contour, used_bounding_box)``: a bounding-box result can cover
    area beyond the rectangles and must be re-checked against neighbouring
    regions by the caller.
    """
    if len(rects) == 1:
        return _bounding_box_contour(rects), False

    r_centers = np.array([(rect["r_lo"] + rect["r_hi"]) / 2.0 for rect in rects])
    z_centers = np.array([(rect["z_lo"] + rect["z_hi"]) / 2.0 for rect in rects])
    vertical = np.ptp(z_centers) >= np.ptp(r_centers)
    axis_lo, axis_hi = ("z_lo", "z_hi") if vertical else ("r_lo", "r_hi")
    trans_lo, trans_hi = ("r_lo", "r_hi") if vertical else ("z_lo", "z_hi")

    ordered = sorted(rects, key=lambda rect: (rect[axis_lo] + rect[axis_hi]) / 2.0)

    def _point(axis_value: float, trans_value: float) -> tuple[float, float]:
        return (trans_value, axis_value) if vertical else (axis_value, trans_value)

    forward: list[tuple[float, float]] = []
    backward: list[tuple[float, float]] = []
    for rect in ordered:
        forward.append(_point(rect[axis_lo], rect[trans_hi]))
        forward.append(_point(rect[axis_hi], rect[trans_hi]))
    for rect in reversed(ordered):
        backward.append(_point(rect[axis_hi], rect[trans_lo]))
        backward.append(_point(rect[axis_lo], rect[trans_lo]))

    contour = np.array(_dedupe_chain(forward + backward))
    loop_area = sum(_rect_area(rect) for rect in rects)
    if loop_area > 0.0:
        traced = _shoelace_area(contour)
        if abs(traced - loop_area) > _AREA_RTOL * loop_area:
            _log.warning(
                "Vessel region %s: staircase trace area %.3e differs from loop "
                "area %.3e by more than %.0f%%; using the bounding box.",
                region, traced, loop_area, 100 * _AREA_RTOL,
            )
            return _bounding_box_contour(rects), True
    return contour, False


def _shrink_rects(rects: list[dict[str, float]], margin: float) -> list[dict[str, float]]:
    """Shrink every rectangle by ``margin`` per side (turns abutments into gaps)."""
    out = []
    for rect in rects:
        shrunk = dict(rect)
        shrunk["r_lo"] += margin
        shrunk["r_hi"] -= margin
        shrunk["z_lo"] += margin
        shrunk["z_hi"] -= margin
        if shrunk["r_hi"] > shrunk["r_lo"] and shrunk["z_hi"] > shrunk["z_lo"]:
            out.append(shrunk)
    return out


def _cluster_span(rects: list[dict[str, float]]) -> dict[str, float]:
    return {
        "r_lo": min(r["r_lo"] for r in rects), "r_hi": max(r["r_hi"] for r in rects),
        "z_lo": min(r["z_lo"] for r in rects), "z_hi": max(r["z_hi"] for r in rects),
    }


def _is_vertical(rects: list[dict[str, float]]) -> bool:
    span = _cluster_span(rects)
    return (span["z_hi"] - span["z_lo"]) >= (span["r_hi"] - span["r_lo"])


def _clamp_out_of_band(
    rects: list[dict[str, float]], band: dict[str, float], gap: float
) -> list[dict[str, float]]:
    """Clamp rectangles in Z so none enters ``band`` where their R ranges overlap."""
    out = []
    band_mid = 0.5 * (band["z_lo"] + band["z_hi"])
    for rect in rects:
        if rect["r_hi"] <= band["r_lo"] or rect["r_lo"] >= band["r_hi"]:
            out.append(rect)
            continue
        if rect["z_hi"] <= band["z_lo"] or rect["z_lo"] >= band["z_hi"]:
            out.append(rect)
            continue
        clamped = dict(rect)
        rect_mid = 0.5 * (rect["z_lo"] + rect["z_hi"])
        if rect_mid <= band_mid:
            clamped["z_hi"] = min(clamped["z_hi"], band["z_lo"] - gap)
        else:
            clamped["z_lo"] = max(clamped["z_lo"], band["z_hi"] + gap)
        if clamped["z_hi"] > clamped["z_lo"]:
            out.append(clamped)
    return out


def _deconflict_clusters(
    clusters: dict[str, list[dict[str, float]]], gap: float
) -> dict[str, list[dict[str, float]]]:
    """Make region clusters pairwise disjoint.

    The filament segments abut and locally overlap at the vessel's corner
    joints (e.g. the outer cylinder W1 runs through the W8/W9 shoulder bands,
    and the W2/W7 lid layers share a corner strip). Rules, applied at the
    rectangle level so staircase traces stay valid:

    1. every rectangle is pre-shrunk by ``gap/2`` per side (separates clean
       abutments and removes T-junctions);
    2. vertical runs (cylinders, chimneys, columns) yield to horizontal bands
       (lids, decks, shoulders): their rectangles are clamped in Z out of any
       overlapping horizontal cluster's span — corner metal is assigned to
       the horizontal member;
    3. between overlapping horizontal clusters, the smaller (fewer loops)
       yields to the larger by the same Z-clamping.
    """
    shrunk = {name: _shrink_rects(rects, gap / 2.0) for name, rects in clusters.items()}
    shrunk = {name: rects for name, rects in shrunk.items() if rects}

    vertical = {name for name, rects in shrunk.items() if _is_vertical(rects)}
    horizontal = [name for name in shrunk if name not in vertical]

    for name in list(shrunk):
        if name not in vertical:
            continue
        for other in horizontal:
            shrunk[name] = _clamp_out_of_band(shrunk[name], _cluster_span(shrunk[other]), gap / 2.0)
        if not shrunk[name]:
            _log.warning("Vessel region %s vanished during de-conflict; dropped", name)
            del shrunk[name]

    by_size = sorted(horizontal, key=lambda name: len(shrunk.get(name, [])), reverse=True)
    for i, name in enumerate(by_size):
        if name not in shrunk:
            continue
        for larger in by_size[:i]:
            if larger not in shrunk:
                continue
            shrunk[name] = _clamp_out_of_band(shrunk[name], _cluster_span(shrunk[larger]), gap / 2.0)
        if not shrunk[name]:
            _log.warning("Vessel region %s vanished during de-conflict; dropped", name)
            del shrunk[name]
    return shrunk


def _assert_disjoint(clusters: dict[str, list[dict[str, float]]]) -> None:
    """Exact pairwise rectangle-overlap check across regions (defensive)."""
    names = sorted(clusters)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for ra in clusters[a]:
                for rb in clusters[b]:
                    if (ra["r_lo"] < rb["r_hi"] and rb["r_lo"] < ra["r_hi"]
                            and ra["z_lo"] < rb["z_hi"] and rb["z_lo"] < ra["z_hi"]):
                        raise ValueError(
                            f"Vessel regions {a} and {b} still overlap after "
                            "de-conflict; check pf_passive geometry or raise "
                            "TokaMakerConfig.vessel_gap."
                        )


def _resolve_eta(
    region: str, segment: str, rects: list[dict[str, float]], config: TokaMakerConfig
) -> tuple[float, float]:
    """(eta to use, ODS-median eta for provenance)."""
    ods_values = np.array([rect["resistivity"] for rect in rects])
    ods_median = float(np.nanmedian(ods_values)) if np.any(np.isfinite(ods_values)) else float("nan")
    overrides = {str(k).upper(): float(v) for k, v in (config.vessel_eta or {}).items()}
    eta = overrides.get(region, overrides.get(segment, float(config.eta_vessel)))
    return eta, ods_median


def vessel_segments_from_ods(ods: Any, config: TokaMakerConfig) -> dict[str, dict]:
    """Build TokaMaker conductor-region descriptions from ``pf_passive``.

    Returns ``{region_name: {"contour": [[r, z], ...], "eta", "dx",
    "noncontinuous", "segment", "n_loops", "eta_ods_median"}}``.
    """
    try:
        segments = _loop_rectangles(ods)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(
            "include_vessel=True requires pf_passive.loop geometry in the ODS "
            "(load the packaged static geometry via vaft.machine_mapping.pf_passive)."
        ) from exc
    if not segments:
        raise ValueError("No pf_passive loops found to build vessel conductor regions")

    excluded = {name.upper() for name in config.exclude_vessel_segments}
    noncontinuous = {name.upper() for name in config.vessel_noncontinuous}

    named_clusters: dict[str, list[dict[str, float]]] = {}
    segment_of: dict[str, str] = {}
    for segment in sorted(segments):
        if segment in excluded:
            continue
        clusters = _split_z_clusters(segments[segment])
        for cluster_index, cluster in enumerate(clusters):
            if len(clusters) == 1:
                region = segment
            elif len(clusters) == 2:
                # clusters come out of the Z-split in ascending order, so
                # relative position gives unique names even when both halves
                # sit on the same side of Z = 0
                region = f"{segment}_{'L' if cluster_index == 0 else 'U'}"
            else:
                region = f"{segment}_{cluster_index + 1}"
            named_clusters[region] = cluster
            segment_of[region] = segment

    named_clusters = _deconflict_clusters(named_clusters, config.vessel_gap)
    _assert_disjoint(named_clusters)

    regions: dict[str, dict] = {}
    bbox_fallbacks: list[str] = []
    for region, cluster in named_clusters.items():
        segment = segment_of[region]
        contour, used_bbox = _trace_strip(cluster, region)
        if used_bbox:
            bbox_fallbacks.append(region)
        if np.any(contour[:, 0] <= 0.0):
            raise ValueError(f"Vessel region {region} has R <= 0 points")

        thickness = float(np.median([
            min(r["r_hi"] - r["r_lo"], r["z_hi"] - r["z_lo"]) for r in cluster
        ]))
        dx = min(config.dx_conductor, max(thickness, config.dx_conductor_min))
        eta, eta_ods_median = _resolve_eta(region, segment, cluster, config)

        regions[region] = {
            "contour": [[float(r), float(z)] for r, z in contour],
            "eta": eta,
            "dx": dx,
            "noncontinuous": region in noncontinuous or segment in noncontinuous,
            "segment": segment,
            "n_loops": len(cluster),
            "eta_ods_median": eta_ods_median,
        }

    if not regions:
        raise ValueError(
            "All pf_passive segments were excluded; nothing to mesh as the vessel "
            f"(exclude_vessel_segments={config.exclude_vessel_segments!r})"
        )

    # The rectangle-level disjointness assert above does not cover bounding-box
    # fallbacks, whose contour can span area beyond the rectangles it replaced.
    for region in bbox_fallbacks:
        contour = np.asarray(regions[region]["contour"])
        box = {
            "r_lo": float(contour[:, 0].min()), "r_hi": float(contour[:, 0].max()),
            "z_lo": float(contour[:, 1].min()), "z_hi": float(contour[:, 1].max()),
        }
        for other, cluster in named_clusters.items():
            if other == region:
                continue
            for rect in cluster:
                if (box["r_lo"] < rect["r_hi"] and rect["r_lo"] < box["r_hi"]
                        and box["z_lo"] < rect["z_hi"] and rect["z_lo"] < box["z_hi"]):
                    raise ValueError(
                        f"Vessel region {region} fell back to its bounding box, "
                        f"which overlaps region {other}; this geometry cannot be "
                        "represented as monotone strips — split or exclude the "
                        f"segment (exclude_vessel_segments) or adjust vessel_gap."
                    )
    return regions
