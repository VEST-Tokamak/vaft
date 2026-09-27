"""The 2-D chart every stability and operational-space diagram draws on.

Two axes with math labels, optional ticks, boundary curves clipped to the
plotting box, region labels and point markers -- in data coordinates, mapped
to centimetres. Shared so later chart families (bifurcation, collisionality
regimes, CMA) reuse one layout rather than restating it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ._scene import Arrow, Label, Polyline, Scene

#: plotting box of every chart [cm]
CHART_WIDTH = 9.0
CHART_HEIGHT = 6.5


@dataclass(eq=False)
class Chart:
    """What a stability diagram shows, in data coordinates."""

    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    curves: Dict[str, np.ndarray] = field(default_factory=dict)
    points: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    labels: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    parameters: Dict[str, float] = field(default_factory=dict)

    def to_cm(self, xy) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        (x0, x1), (y0, y1) = self.x_range, self.y_range
        return np.stack([(xy[..., 0] - x0) / (x1 - x0) * CHART_WIDTH,
                         (xy[..., 1] - y0) / (y1 - y0) * CHART_HEIGHT], axis=-1)


def star_polygon(center, outer=0.24, inner=0.1) -> np.ndarray:
    angles = np.pi / 2 + np.arange(10) * np.pi / 5
    radii = np.where(np.arange(10) % 2 == 0, outer, inner)
    return np.asarray(center) + np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=-1)


def clip(xy: np.ndarray, chart: Chart) -> List[np.ndarray]:
    """Runs of a polyline that lie inside the chart box."""
    (x0, x1), (y0, y1) = chart.x_range, chart.y_range
    inside = (xy[:, 0] >= x0 - 1e-12) & (xy[:, 0] <= x1 + 1e-12) & (xy[:, 1] >= y0 - 1e-12) & (xy[:, 1] <= y1 + 1e-12)
    runs, start = [], None
    for i, ok in enumerate(inside):
        if ok and start is None:
            start = i
        if (not ok or i == len(inside) - 1) and start is not None:
            end = i + 1 if ok else i
            if end - start >= 2:
                runs.append(xy[start:end])
            start = None
    return runs


def render_chart(chart: Chart, *, x_label: str, y_label: str, curve_styles: Dict[str, str],
                  region_text: Dict[str, str], x_ticks: Sequence[float] = (), y_ticks: Sequence[float] = (),
                  note: str = "", star: str = "") -> Scene:
    items: List = []
    W, H = CHART_WIDTH, CHART_HEIGHT
    for name, style in curve_styles.items():
        for run in clip(chart.curves[name], chart):
            items.append(Polyline.of(chart.to_cm(run), style, role=name))
    if star:
        items.append(Polyline.of(star_polygon(chart.to_cm(chart.points[star])), "star", role=star, closed=True))
    # axis titles clear the tick numbers when there are any
    x_gap = 0.75 if len(x_ticks) else 0.35
    y_gap = 0.95 if len(y_ticks) else 0.35
    items += [
        Arrow((0.0, 0.0), (W + 0.6, 0.0), "chart axis", role="axes"),
        Arrow((0.0, 0.0), (0.0, H + 0.6), "chart axis", role="axes"),
        Label((W * 0.62, -x_gap), x_label, "xlabel", anchor="north", role="axes"),
        Label((-y_gap, H * 0.62), y_label, "ylabel", anchor="south", role="axes"),
    ]
    for x in x_ticks:
        cx = float(chart.to_cm(np.array([x, chart.y_range[0]]))[0])
        items += [Polyline.of([(cx, 0.0), (cx, -0.12)], "tick", role="ticks"),
                  Label((cx, -0.18), f"${x:g}$", "ticklabel", anchor="north", role="ticks")]
    for y in y_ticks:
        cy = float(chart.to_cm(np.array([chart.x_range[0], y]))[1])
        items += [Polyline.of([(0.0, cy), (-0.12, cy)], "tick", role="ticks"),
                  Label((-0.18, cy), f"${y:g}$", "ticklabel", anchor="east", role="ticks")]
    for name, text in region_text.items():
        items.append(Label(tuple(chart.to_cm(chart.labels[name])), text, "region", role=f"region_{name}"))
    if note:
        items.append(Label((W / 2, -x_gap - 1.2), note, "note", role="note"))
    return Scene(tuple(items))


def nice_ticks(top: float) -> List[float]:
    """Round tick values from 0 up to ``top``: 3-6 of them at a 1-2-5 spacing."""
    for step in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0):
        if top / step <= 6.0:
            return [float(v) for v in np.arange(0.0, top, step)]
    return [0.0]
