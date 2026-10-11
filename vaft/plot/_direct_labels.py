"""Direct labels: name each series beside its own data instead of in a legend (#1946).

A legend makes the reader match a colour swatch to a curve; a direct label sits
on the data it names, in that data's colour.  The cost is placement: labels
must not cover each other, should not cover the data, and must stay inside the
axes.  :func:`add_direct_labels` does that placement, and does it at *draw
time*: a figure laid out afterwards (``tight_layout``, constrained layout, an
equal aspect, a presentation format, another ``savefig`` dpi) changes the axes'
size on the page while the text keeps its size in points, so positions chosen
at call time would no longer be collision-free.  A placement artist attached to
the axes therefore re-places every label, with the renderer's own text extents,
each time the axes draw.

Placement is greedy and deterministic, in the order the labels were added:

* candidates are boxes of the label's measured size around sample points of its
  anchor -- every direction for a curve (outward first), above the points first
  for a cluster -- at increasing distance;
* a box that overlaps an already placed label, or leaves the axes, is refused;
* the rest are scored by how much data they cover (every label's anchor points,
  curves resampled along their length, and any extra ``avoid`` curve), their
  distance from the anchor and the preferred side;
* a label that ends up away from its anchor gets a thin leader line in its own
  colour back to the anchor point it names.

Several calls on one axes share one placement, so a caller can add a label
(e.g. an annotation ring) after a renderer has labelled its series, and both
avoid each other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = ["DirectLabel", "add_direct_labels", "LABEL_GID", "LEADER_GID"]

#: ``gid`` prefixes of the artists this module adds (``f"{prefix}{text}"``).
LABEL_GID = "direct-label:"
LEADER_GID = "direct-label-leader:"

_SIDES = ("above", "outside", "any")
# Unit directions (dx, dy) in display space and their penalty for side="above".
_DIRECTIONS = (
    ((0.0, 1.0), 0.0), ((-0.7071, 0.7071), 0.6), ((0.7071, 0.7071), 0.6),
    ((-1.0, 0.0), 1.6), ((1.0, 0.0), 1.6),
    ((-0.7071, -0.7071), 2.6), ((0.7071, -0.7071), 2.6), ((0.0, -1.0), 3.2),
)
_RADII = 9          # distance steps tried away from an anchor
_SAMPLES = 24       # anchor sample points per label
_ATTACHED_STEPS = 1  # within this many steps the label sits on its anchor, no leader
_LEADER_CHECKS = 120  # displaced candidates whose leader is checked for crossings


@dataclass(frozen=True)
class DirectLabel:
    """One label: its text and colour, and the data points it names.

    Parameters
    ----------
    text : str
        Label text [str].
    color : str
        Text and leader colour, the series' own colour [colour].
    x, y : array_like
        Anchor points in data coordinates: a cluster's points or a curve's
        vertices [data].
    side : str, optional
        ``"above"`` (a scatter cluster: above its points first),
        ``"outside"`` (a closed curve: away from its centre first; the curve is
        also treated as a line to keep clear of) or ``"any"``; default
        ``"above"`` [str].
    """

    text: str
    color: Any
    x: Any
    y: Any
    side: str = "above"

    def __post_init__(self):
        if self.side not in _SIDES:
            raise ValueError(f"side must be one of {_SIDES}; got {self.side!r}")


def add_direct_labels(
    ax,
    labels: Iterable[DirectLabel | Sequence[Any]],
    *,
    avoid: Iterable[tuple[Any, Any]] = (),
    fontsize: Any = None,
    halo: bool = True,
    leader: bool = True,
):
    """Label each series directly on ``ax``, without overlaps; return the texts.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to label [Axes].
    labels : iterable of DirectLabel or tuple
        ``DirectLabel`` or ``(text, color, x, y[, side])`` per series [-].
    avoid : iterable of (x, y), optional
        Extra polylines in data coordinates to keep clear of, e.g. reference
        lines [data].
    fontsize : float or str or None, optional
        Label size; default ``None`` is ``rcParams["legend.fontsize"]`` scaled
        up by 1.15, the size a legend entry had, made legible [pt].
    halo : bool, optional
        White outline behind the text so it reads over data, default ``True``.
    leader : bool, optional
        Draw a leader line when a label is displaced, default ``True``.

    Returns
    -------
    list of matplotlib.text.Text
        One text per label, ``gid`` ``"direct-label:<text>"``; positions are
        final once the figure has been drawn.
    """
    from matplotlib import patheffects
    from matplotlib.font_manager import FontProperties
    from matplotlib.lines import Line2D
    import matplotlib

    if fontsize is None:
        fontsize = 1.15 * FontProperties(size=matplotlib.rcParams["legend.fontsize"]).get_size_in_points()
    placer = _placer(ax)
    texts = []
    for item in labels:
        spec = item if isinstance(item, DirectLabel) else DirectLabel(*item)
        x, y = np.asarray(spec.x, float).ravel(), np.asarray(spec.y, float).ravel()
        text = ax.text(0.0, 0.0, spec.text, color=spec.color, fontsize=fontsize, ha="left", va="bottom",
                       transform=ax.transAxes, zorder=7, clip_on=False, gid=f"{LABEL_GID}{spec.text}")
        text.set_in_layout(False)
        if halo:
            text.set_path_effects([patheffects.withStroke(linewidth=3.0, foreground="white", alpha=0.9)])
        line = None
        if leader:
            line = Line2D([], [], color=spec.color, lw=0.9, transform=ax.transAxes, zorder=6.9,
                          gid=f"{LEADER_GID}{spec.text}", visible=False, clip_on=False)
            line.set_in_layout(False)
            ax.add_line(line)
        placer.entries.append(_Entry(spec.side, x, y, text, line, text.get_fontsize()))
        texts.append(text)
    for xs, ys in avoid:
        placer.avoid.append((np.asarray(xs, float).ravel(), np.asarray(ys, float).ravel()))
    placer.stale = True
    return texts


@dataclass
class _Entry:
    side: str
    x: np.ndarray
    y: np.ndarray
    text: Any
    line: Any
    size: float


def _placer(ax):
    for artist in ax.artists:
        if isinstance(artist, _placer_class()):
            return artist
    placer = _placer_class()()
    ax.add_artist(placer)
    return placer


_PLACER = None


def _placer_class():
    """The placement artist, defined lazily so importing this module needs no matplotlib."""
    global _PLACER
    if _PLACER is not None:
        return _PLACER
    from matplotlib.artist import Artist

    class DirectLabelPlacer(Artist):
        """Invisible artist drawn first in its axes: it positions the labels."""

        def __init__(self):
            super().__init__()
            self.entries: list[_Entry] = []
            self.avoid: list[tuple[np.ndarray, np.ndarray]] = []
            self.set_zorder(-1e9)
            self.set_in_layout(False)

        def draw(self, renderer):
            if self.entries:
                _place(self.axes, self.entries, self.avoid, renderer)
            self.stale = False

        def get_window_extent(self, renderer=None):
            from matplotlib.transforms import Bbox

            return Bbox.null()

    _PLACER = DirectLabelPlacer
    return _PLACER


# ---------------------------------------------------------------------------
# placement


def _display(ax, x, y):
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        return np.empty((0, 2))
    with np.errstate(all="ignore"):
        pts = ax.transData.transform(np.column_stack([x[finite], y[finite]]))
    return pts[np.isfinite(pts).all(axis=1)]


def _resample(pts, step):
    """Points every ``step`` pixels along a polyline (display coordinates)."""
    if len(pts) < 2:
        return pts
    seg = np.diff(pts, axis=0)
    length = np.hypot(seg[:, 0], seg[:, 1])
    out = [pts[:1]]
    for start, d, n in zip(pts[:-1], seg, np.ceil(length / step).astype(int)):
        if n > 0:
            t = (np.arange(1, n + 1) / n)[:, None]
            out.append(start + t * d)
    return np.vstack(out)


def _inside(boxes, frame):
    return ((boxes[:, 0] >= frame[0]) & (boxes[:, 1] >= frame[1])
            & (boxes[:, 2] <= frame[2]) & (boxes[:, 3] <= frame[3]))


def _overlaps(boxes, others, pad):
    if not len(others):
        return np.zeros(len(boxes), bool)
    o = np.asarray(others)
    hit = ((boxes[:, None, 0] < o[None, :, 2] + pad) & (boxes[:, None, 2] > o[None, :, 0] - pad)
           & (boxes[:, None, 1] < o[None, :, 3] + pad) & (boxes[:, None, 3] > o[None, :, 1] - pad))
    return hit.any(axis=1)


def _covered(boxes, points):
    """Number of obstacle points inside each box."""
    if not len(points):
        return np.zeros(len(boxes))
    counts = np.zeros(len(boxes))
    chunk = max(1, 2_000_000 // max(len(points), 1))
    px, py = points[:, 0], points[:, 1]
    for i in range(0, len(boxes), chunk):
        b = boxes[i:i + chunk]
        counts[i:i + chunk] = ((px[None, :] >= b[:, None, 0]) & (px[None, :] <= b[:, None, 2])
                               & (py[None, :] >= b[:, None, 1]) & (py[None, :] <= b[:, None, 3])).sum(axis=1)
    return counts


def _samples(pts, n):
    """Up to ``n`` representative anchor points: spread along a curve / over a cluster."""
    if len(pts) <= n:
        return pts
    index = np.linspace(0, len(pts) - 1, n).round().astype(int)
    return pts[index]


def _cluster_samples(pts, n):
    """Representative points of a cluster: the medoid first, then its core spread by x.

    Only the core (the 80 % of points nearest the median) is sampled, so a label
    names the bulk of the cluster rather than a stray point.
    """
    if len(pts) == 0:
        return pts
    centre = np.median(pts, axis=0)
    if len(pts) > 5:
        distance = np.hypot(*(pts - centre).T)
        pts = pts[distance <= np.quantile(distance, 0.8)]
    order = np.argsort(pts[:, 0], kind="stable")
    spread = _samples(pts[order], n - 1)
    medoid = pts[np.argmin(np.hypot(*(pts - centre).T))]
    return np.vstack([medoid[None, :], spread])


def _candidates(entry, pts, w, h, gap):
    """Candidate boxes (x0, y0, x1, y1), anchor points and preference penalties."""
    boxes, anchors, penalty, steps = [], [], [], []
    if entry.side == "outside":
        sample = _samples(pts, _SAMPLES * 2)
        centre = pts.mean(axis=0)
        normal = sample - centre
        norm = np.hypot(normal[:, 0], normal[:, 1])
        norm[norm == 0] = 1.0
        normal /= norm[:, None]
        # Prefer the top of the curve slightly, so ties resolve the same way every time.
        order_penalty = 0.4 * (1.0 - normal[:, 1])
        directions = [(normal, order_penalty)]
    else:
        sample = _cluster_samples(pts, _SAMPLES)
        centre = np.median(pts, axis=0)
        scale = max(np.ptp(pts[:, 0]) if len(pts) > 1 else 0.0, np.ptp(pts[:, 1]) if len(pts) > 1 else 0.0, h)
        offcentre = 1.5 * np.hypot(*(sample - centre).T) / scale
        directions = []
        for (dx, dy), side_penalty in _DIRECTIONS:
            p = side_penalty if entry.side == "above" else 0.5 * side_penalty
            directions.append((np.tile([dx, dy], (len(sample), 1)), p + offcentre))
    for step in range(_RADII):
        r = gap + step * 0.9 * h
        for unit, pen in directions:
            # The classic eight label positions: the point ``r`` away along
            # ``unit`` touches the box's corner (diagonal) or the middle of its
            # side (axis-aligned), so a wide label never swings far off diagonally.
            sx = np.where(unit[:, 0] > 0.38, 1.0, np.where(unit[:, 0] < -0.38, -1.0, 0.0))
            sy = np.where(unit[:, 1] > 0.38, 1.0, np.where(unit[:, 1] < -0.38, -1.0, 0.0))
            cx = sample[:, 0] + unit[:, 0] * r + sx * 0.5 * w
            cy = sample[:, 1] + unit[:, 1] * r + sy * 0.5 * h
            boxes.append(np.column_stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h]))
            anchors.append(sample)
            penalty.append(np.broadcast_to(pen, len(sample)))
            steps.append(np.full(len(sample), step))
    return np.vstack(boxes), np.vstack(anchors), np.concatenate(penalty), np.concatenate(steps)


def _place(ax, entries, avoid, renderer):
    bbox = ax.bbox
    margin = 3.0
    frame = (bbox.x0 + margin, bbox.y0 + margin, bbox.x1 - margin, bbox.y1 - margin)
    to_axes = ax.transAxes.inverted()

    entry_pts = [_display(ax, e.x, e.y) for e in entries]
    step_px = 2.0
    obstacles = []
    for e, pts in zip(entries, entry_pts):
        obstacles.append(_resample(pts, step_px) if e.side == "outside" else pts)
    for xs, ys in avoid:
        obstacles.append(_resample(_display(ax, xs, ys), step_px))
    points = np.vstack(obstacles) if obstacles else np.empty((0, 2))
    if len(points):
        keep = ((points[:, 0] >= bbox.x0) & (points[:, 0] <= bbox.x1)
                & (points[:, 1] >= bbox.y0) & (points[:, 1] <= bbox.y1))
        points = points[keep]

    placed = []
    for entry, pts in zip(entries, entry_pts):
        text = entry.text
        text.set_position((0.0, 0.0))
        text.set_fontsize(entry.size)
        extent = text.get_window_extent(renderer)
        room = frame[2] - frame[0]
        if extent.width > room > 0:
            # A label wider than the axes is set smaller rather than let out of them.
            text.set_fontsize(entry.size * room / extent.width * 0.98)
            extent = text.get_window_extent(renderer)
        w, h = extent.width, extent.height
        gap = 0.35 * h
        if entry.line is not None:
            entry.line.set_visible(False)
        inside = pts[(pts[:, 0] >= bbox.x0) & (pts[:, 0] <= bbox.x1)
                     & (pts[:, 1] >= bbox.y0) & (pts[:, 1] <= bbox.y1)] if len(pts) else pts
        if not len(inside):
            text.set_visible(False)
            continue
        text.set_visible(True)
        boxes, anchors, penalty, steps = _candidates(entry, inside, w, h, gap)
        ok = _inside(boxes, frame) & ~_overlaps(boxes, placed, 2.0)
        if not ok.any():
            # Nothing free inside the axes: clamp the nearest candidate into the frame.
            boxes = _clamp(boxes, frame)
            ok = ~_overlaps(boxes, placed, 2.0)
            if not ok.any():
                ok = np.ones(len(boxes), bool)
        hits = _covered(boxes, points)
        score = 10.0 * (hits > 0) + 2.0 * np.log1p(hits) + 2.5 * steps + penalty
        score[~ok] = np.inf
        # A leader that cuts through other data is worse than a slightly farther label:
        # rescore the best few displaced candidates by what their leader crosses.
        finite = np.flatnonzero(np.isfinite(score) & (steps >= _ATTACHED_STEPS))
        if len(finite) and len(points):
            top = finite[np.argsort(score[finite], kind="stable")[:_LEADER_CHECKS]]
            crossed = _leader_crossings(anchors[top], boxes[top], points, gap)
            score[top] += 4.0 * (crossed > 0) + 1.5 * np.log1p(crossed)
        best = int(np.argmin(score))  # argmin is first-minimum: deterministic
        box, anchor = boxes[best], anchors[best]
        placed.append(box)
        text.set_position(to_axes.transform((box[0], box[1])))
        if entry.line is not None and steps[best] >= _ATTACHED_STEPS:
            near = np.clip(anchor, box[:2], box[2:])
            if np.hypot(*(near - anchor)) > gap * 1.5:
                ends = to_axes.transform(np.vstack([anchor, near]))
                entry.line.set_data(ends[:, 0], ends[:, 1])
                entry.line.set_visible(True)


def _leader_crossings(anchors, boxes, points, gap):
    """Obstacle points within 2 px of each candidate's leader (anchor to box edge)."""
    near = np.clip(anchors, boxes[:, :2], boxes[:, 2:])
    seg = near - anchors
    length2 = np.maximum((seg ** 2).sum(axis=1), 1e-9)
    rel = points[None, :, :] - anchors[:, None, :]
    t = np.clip((rel * seg[:, None, :]).sum(axis=2) / length2[:, None], 0.0, 1.0)
    closest = anchors[:, None, :] + t[:, :, None] * seg[:, None, :]
    distance = np.hypot(*(points[None, :, :] - closest).transpose(2, 0, 1))
    from_anchor = np.hypot(*rel.transpose(2, 0, 1))
    # Points at the anchor itself are the labelled data, not a crossing.
    return ((distance < 2.0) & (from_anchor > gap)).sum(axis=1)


def _clamp(boxes, frame):
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    x0 = np.clip(boxes[:, 0], frame[0], np.maximum(frame[2] - w, frame[0]))
    y0 = np.clip(boxes[:, 1], frame[1], np.maximum(frame[3] - h, frame[1]))
    return np.column_stack([x0, y0, x0 + w, y0 + h])
