"""Concept-diagram primitives: boxes, connectors and bands.

Flow charts, taxonomies and process sequences (#1041, #1047, #1090, ...)
are built from these. Each expands into the existing scene items -- a
closed ``Polyline`` and a ``Label`` -- so the renderer is unchanged and a
concept diagram is hashed, rendered and checked like any other scene.
Layout is explicit: coordinates in centimetres, chosen by the builder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from ._scene import Arrow, Label, Polyline

#: inner padding between a box's outline and its text [cm]
_PAD = 0.15


@dataclass(frozen=True)
class Box:
    """A labelled rectangle: its geometry, for connectors, and its scene items."""

    x: float
    y: float
    width: float
    height: float
    items: Tuple

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def boundary_point(self, towards) -> Tuple[float, float]:
        """Where the ray from the centre towards ``towards`` leaves the box."""
        d = np.asarray(towards, dtype=float) - np.array(self.center)
        if not np.any(d):
            return self.center
        scale = min(
            (0.5 * self.width / abs(d[0])) if d[0] else np.inf,
            (0.5 * self.height / abs(d[1])) if d[1] else np.inf,
        )
        return tuple(np.array(self.center) + scale * d)


def box(x: float, y: float, width: float, height: float, text: str, *, style: str = "concept box",
        text_style: str = "concept text", role: str = "") -> Box:
    """A rounded box centred at ``(x, y)`` with ``text`` wrapped to its width."""
    if not width > 2 * _PAD or not height > 0:
        raise ValueError(f"box needs a width above {2 * _PAD} cm and a positive height, not {width} x {height}")
    hw, hh = 0.5 * width, 0.5 * height
    outline = Polyline.of([(x - hw, y - hh), (x + hw, y - hh), (x + hw, y + hh), (x - hw, y + hh)], style,
                          role=role, closed=True)
    label = Label((x, y), text, f"{text_style},text width={width - 2 * _PAD:.2f}cm", role=role)
    return Box(x, y, width, height, (outline, label))


def connector(start: Box, end: Box, *, style: str = "connector", role: str = "", gap: float = 0.08) -> Arrow:
    """An arrow from the edge of ``start`` to the edge of ``end``, along their centre line."""
    a = np.array(start.boundary_point(end.center))
    b = np.array(end.boundary_point(start.center))
    d = b - a
    length = float(np.hypot(*d))
    towards = np.array(end.center) - np.array(start.center)
    # overlapping boxes put the exit points the wrong way round
    if length <= 2 * gap or float(np.dot(d, towards)) <= 0.0:
        raise ValueError("the boxes touch or overlap: nothing to connect")
    u = d / length
    return Arrow(tuple(a + gap * u), tuple(b - gap * u), style, role=role)


def band(x0: float, x1: float, y0: float, y1: float, text: str = "", *, style: str = "concept band",
         role: str = "") -> List:
    """A shaded background strip from ``(x0, y0)`` to ``(x1, y1)``, its label at the top-left."""
    if not (x1 > x0 and y1 > y0):
        raise ValueError("band needs x1 > x0 and y1 > y0")
    items: List = [Polyline.of([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], style, role=role, closed=True)]
    if text:
        items.append(Label((x0 + _PAD, y1 - _PAD), text, "concept band label", anchor="north west", role=role))
    return items
