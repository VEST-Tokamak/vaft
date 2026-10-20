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
  colour back to the anchor point it names (owned and drawn by the label's
  text, created only when needed, so ``ax.lines`` is never touched);
* a label wider than the axes is set smaller, down to 0.6 of its size, and
  ellipsized beyond that (the full text stays in ``text.full_text`` and the gid).

Placement is cached per axes on (axes box, view limits, scales, dpi, label
texts, sizes and visibility), so redrawing an unchanged figure reuses it.
Obstacles are counted on a 2 px occupancy grid over the axes with a
summed-area table, so scoring a candidate costs a few lookups whatever the row
count, and memory is bounded by the axes' pixel area, not by the data.  A label the caller
hides with ``set_visible(False)`` stays hidden and is not placed.

Several calls on one axes share one placement, so a caller can add a label
(e.g. an annotation ring) after a renderer has labelled its series, and both
avoid each other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
from matplotlib.artist import Artist
from matplotlib.text import Text

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
_MIN_SHRINK = 0.6      # a label is never set below this fraction of its size; it is ellipsized instead
_ELLIPSIS = "\u2026"


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
        One :class:`DirectLabelText` per label, ``gid`` ``"direct-label:<text>"``;
        positions are final once the figure has been drawn, and ``.leader`` is
        the leader line when one is drawn.
    """
    from matplotlib import patheffects
    from matplotlib.font_manager import FontProperties
    import matplotlib

    if fontsize is None:
        fontsize = 1.15 * FontProperties(size=matplotlib.rcParams["legend.fontsize"]).get_size_in_points()
    placer = _placer(ax)
    texts = []
    for item in labels:
        spec = item if isinstance(item, DirectLabel) else DirectLabel(*item)
        x, y = np.asarray(spec.x, float).ravel(), np.asarray(spec.y, float).ravel()
        text = DirectLabelText(0.0, 0.0, spec.text, color=spec.color, fontsize=fontsize, ha="left",
                               va="bottom", transform=ax.transAxes, zorder=7, clip_on=False,
                               gid=f"{LABEL_GID}{spec.text}")
        text.want_leader = bool(leader)
        text.set_in_layout(False)
        if halo:
            text.set_path_effects([patheffects.withStroke(linewidth=3.0, foreground="white", alpha=0.9)])
        ax._add_text(text)
        placer.entries.append(_Entry(spec.side, x, y, text, text.get_fontsize()))
        texts.append(text)
    for xs, ys in avoid:
        placer.avoid.append((np.asarray(xs, float).ravel(), np.asarray(ys, float).ravel()))
    placer.cache = None
    placer.stale = True
    return texts


class DirectLabelText(Text):
    """A direct label: a text that draws its own leader line, and that placement may suppress.

    ``set_visible`` stays the caller's: placement hides a label it cannot place
    (no anchor inside the view) through ``placement_hidden`` instead, so a label
    the caller hid is never shown again by a redraw.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.full_text = self.get_text()
        self.placement_hidden = False
        self.want_leader = True
        self.leader = None

    def leader_to(self, xy_axes) -> None:
        """Show the leader along ``xy_axes`` (2x2, axes fraction); ``None`` hides it."""
        if xy_axes is None:
            if self.leader is not None:
                self.leader.set_visible(False)
            return
        if not self.want_leader:
            return
        if self.leader is None:
            from matplotlib.lines import Line2D

            self.leader = Line2D([], [], color=self.get_color(), lw=0.9, clip_on=False,
                                 gid=f"{LEADER_GID}{self.full_text}")
            self.leader.set_in_layout(False)
            self.leader.set_figure(self.figure)
            self.leader.axes = self.axes
        self.leader.set_transform(self.get_transform())
        self.leader.set_data(xy_axes[:, 0], xy_axes[:, 1])
        self.leader.set_visible(True)

    def draw(self, renderer):
        if not self.get_visible() or self.placement_hidden:
            return
        if self.leader is not None and self.leader.get_visible():
            self.leader.draw(renderer)
        super().draw(renderer)


@dataclass
class _Entry:
    side: str
    x: np.ndarray
    y: np.ndarray
    text: Any
    size: float


class DirectLabelPlacer(Artist):
    """Invisible artist drawn first in its axes: it positions the labels."""

    def __init__(self):
        super().__init__()
        self.entries: list[_Entry] = []
        self.avoid: list[tuple[np.ndarray, np.ndarray]] = []
        self.cache = None
        self.set_zorder(-1e9)
        self.set_in_layout(False)

    def draw(self, renderer):
        if self.entries and self.axes is not None:
            key = _cache_key(self.axes, self.entries, self.avoid)
            if self.cache is not None and self.cache[0] == key:
                _restore(self.entries, self.cache[1])
            else:
                _place(self.axes, self.entries, self.avoid, renderer)
                self.cache = (key, _snapshot(self.entries))
        self.stale = False

    def get_window_extent(self, renderer=None):
        from matplotlib.transforms import Bbox

        return Bbox.null()


def _placer(ax):
    for artist in ax.artists:
        if isinstance(artist, DirectLabelPlacer):
            return artist
    placer = DirectLabelPlacer()
    ax.add_artist(placer)
    return placer


def _cache_key(ax, entries, avoid):
    return (tuple(np.round(ax.bbox.bounds, 3)), tuple(np.round(ax.viewLim.bounds, 12)),
            ax.get_xscale(), ax.get_yscale(), ax.figure.dpi, len(avoid),
            tuple((e.text.full_text, e.size, e.text.get_visible()) for e in entries))


def _snapshot(entries):
    out = []
    for e in entries:
        t = e.text
        leader = None
        if t.leader is not None and t.leader.get_visible():
            leader = np.asarray(t.leader.get_xydata(), float).copy()
        out.append((t.get_position(), t.get_fontsize(), t.get_text(), t.placement_hidden, leader))
    return out


def _restore(entries, snapshot):
    for e, (position, size, content, hidden, leader) in zip(entries, snapshot):
        t = e.text
        t.set_position(position)
        t.set_fontsize(size)
        t.set_text(content)
        t.placement_hidden = hidden
        t.leader_to(leader)


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


class _Occupancy:
    """Obstacle weight on a pixel grid over the axes, with a summed-area table.

    A box's covered weight is four lookups, a leader's crossing weight a walk
    along its length, so scoring costs O(candidates) whatever the row count.
    """

    def __init__(self, points, weights, bbox, cell=2.0):
        self.x0, self.y0, self.cell = bbox.x0, bbox.y0, cell
        self.nx = max(1, int(np.ceil(bbox.width / cell)))
        self.ny = max(1, int(np.ceil(bbox.height / cell)))
        grid = np.zeros((self.ny, self.nx))
        if len(points):
            ix = np.clip(((points[:, 0] - self.x0) / cell).astype(int), 0, self.nx - 1)
            iy = np.clip(((points[:, 1] - self.y0) / cell).astype(int), 0, self.ny - 1)
            np.add.at(grid, (iy, ix), weights)
        self.grid = grid
        self.table = np.zeros((self.ny + 1, self.nx + 1))
        self.table[1:, 1:] = grid.cumsum(axis=0).cumsum(axis=1)

    def _index(self, value, origin, n):
        return np.clip(np.floor((value - origin) / self.cell).astype(int), 0, n)

    def boxes(self, boxes):
        """Weight inside each (x0, y0, x1, y1) box."""
        i0 = self._index(boxes[:, 0], self.x0, self.nx)
        j0 = self._index(boxes[:, 1], self.y0, self.ny)
        i1 = np.clip(np.ceil((boxes[:, 2] - self.x0) / self.cell).astype(int), i0, self.nx)
        j1 = np.clip(np.ceil((boxes[:, 3] - self.y0) / self.cell).astype(int), j0, self.ny)
        t = self.table
        return t[j1, i1] - t[j0, i1] - t[j1, i0] + t[j0, i0]

    def segments(self, starts, ends, skip):
        """Weight of the cells each segment passes through, ignoring its first ``skip`` px."""
        out = np.zeros(len(starts))
        for k, (a, b) in enumerate(zip(starts, ends)):
            length = float(np.hypot(*(b - a)))
            if length <= skip:
                continue
            t = np.arange(skip, length, self.cell) / length
            p = a + t[:, None] * (b - a)
            ix = np.floor((p[:, 0] - self.x0) / self.cell).astype(int)
            iy = np.floor((p[:, 1] - self.y0) / self.cell).astype(int)
            ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
            cells = np.unique(iy[ok] * self.nx + ix[ok])
            out[k] = self.grid.ravel()[cells].sum()
        return out


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


def _fit_width(text, size, room, renderer):
    """Set ``text`` to fit ``room`` px: shrink to at most ``_MIN_SHRINK``, then ellipsize."""
    text.set_text(text.full_text)
    text.set_fontsize(size)
    extent = text.get_window_extent(renderer)
    if room <= 0 or extent.width <= room:
        return extent
    text.set_fontsize(max(size * room / extent.width * 0.98, _MIN_SHRINK * size))
    extent = text.get_window_extent(renderer)
    content = text.full_text
    while extent.width > room and len(content) > 1:
        content = content[:-1]
        text.set_text(content.rstrip() + _ELLIPSIS)
        extent = text.get_window_extent(renderer)
    return extent


def _place(ax, entries, avoid, renderer):
    bbox = ax.bbox
    live = [e for e in entries if e.text.get_visible()]
    if bbox.width <= 0 or bbox.height <= 0:
        # A collapsed axes has no room and no invertible transform: show nothing.
        for e in live:
            e.text.placement_hidden = True
            e.text.leader_to(None)
        return
    margin = 3.0
    frame = (bbox.x0 + margin, bbox.y0 + margin, bbox.x1 - margin, bbox.y1 - margin)
    to_axes = ax.transAxes.inverted()

    entry_pts = [_display(ax, e.x, e.y) for e in live]
    step_px = 2.0
    obstacles = []
    for e, pts in zip(live, entry_pts):
        obstacles.append(_resample(pts, step_px) if e.side == "outside" else pts)
    for xs, ys in avoid:
        obstacles.append(_resample(_display(ax, xs, ys), step_px))
    points = np.vstack(obstacles) if obstacles else np.empty((0, 2))
    if len(points):
        keep = ((points[:, 0] >= bbox.x0) & (points[:, 0] <= bbox.x1)
                & (points[:, 1] >= bbox.y0) & (points[:, 1] <= bbox.y1))
        points = points[keep]
    occupancy = _Occupancy(points, np.ones(len(points)), bbox)

    placed = []
    for entry, pts in zip(live, entry_pts):
        text = entry.text
        text.set_position((0.0, 0.0))
        text.leader_to(None)
        extent = _fit_width(text, entry.size, frame[2] - frame[0], renderer)
        w, h = extent.width, extent.height
        gap = 0.35 * h
        inside = pts[(pts[:, 0] >= bbox.x0) & (pts[:, 0] <= bbox.x1)
                     & (pts[:, 1] >= bbox.y0) & (pts[:, 1] <= bbox.y1)] if len(pts) else pts
        if not len(inside):
            text.placement_hidden = True
            continue
        text.placement_hidden = False
        boxes, anchors, penalty, steps = _candidates(entry, inside, w, h, gap)
        ok = _inside(boxes, frame) & ~_overlaps(boxes, placed, 2.0)
        if not ok.any():
            # Nothing free inside the axes: clamp the nearest candidate into the frame.
            boxes = _clamp(boxes, frame)
            ok = ~_overlaps(boxes, placed, 2.0)
            if not ok.any():
                ok = np.ones(len(boxes), bool)
        hits = occupancy.boxes(boxes)
        score = 10.0 * (hits > 0) + 2.0 * np.log1p(hits) + 2.5 * steps + penalty
        score[~ok] = np.inf
        # A leader that cuts through other data is worse than a slightly farther label:
        # rescore the best few displaced candidates by what their leader crosses.
        finite = np.flatnonzero(np.isfinite(score) & (steps >= _ATTACHED_STEPS))
        if len(finite) and len(points):
            top = finite[np.argsort(score[finite], kind="stable")[:_LEADER_CHECKS]]
            near = np.clip(anchors[top], boxes[top][:, :2], boxes[top][:, 2:])
            crossed = occupancy.segments(anchors[top], near, gap)
            score[top] += 4.0 * (crossed > 0) + 1.5 * np.log1p(crossed)
        best = int(np.argmin(score))  # argmin is first-minimum: deterministic
        box, anchor = boxes[best], anchors[best]
        placed.append(box)
        text.set_position(to_axes.transform((box[0], box[1])))
        if steps[best] >= _ATTACHED_STEPS:
            near = np.clip(anchor, box[:2], box[2:])
            if np.hypot(*(near - anchor)) > gap * 1.5:
                text.leader_to(to_axes.transform(np.vstack([anchor, near])))


def _clamp(boxes, frame):
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    x0 = np.clip(boxes[:, 0], frame[0], np.maximum(frame[2] - w, frame[0]))
    y0 = np.clip(boxes[:, 1], frame[1], np.maximum(frame[3] - h, frame[1]))
    return np.column_stack([x0, y0, x0 + w, y0 + h])
