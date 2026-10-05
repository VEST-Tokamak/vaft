"""Concept-diagram primitives: boxes, connectors and bands.

Flow charts, taxonomies and process sequences (#1041, #1047, #1090, ...)
are built from these. Each expands into the existing scene items -- a
closed ``Polyline`` and a ``Label`` -- so the renderer is unchanged and a
concept diagram is hashed, rendered and checked like any other scene.
Layout is explicit: coordinates in centimetres, chosen by the builder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ._scene import Arrow, Label, Polyline

#: inner padding between a box's outline and its text [cm]
_PAD = 0.15
#: shortest arrow worth drawing between two boxes [cm]
_MIN_ARROW = 0.2

#: LaTeX's special characters, as they must be written to appear literally
_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&", "%": r"\%",
    "#": r"\#", "_": r"\_", "^": r"\textasciicircum{}", "~": r"\textasciitilde{}",
}


def escape_latex(text: str) -> str:
    """``text`` with every LaTeX special character made literal."""
    return "".join(_LATEX_ESCAPES.get(ch, ch) for ch in text)


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
        text_style: str = "concept text", role: str = "", latex: bool = False) -> Box:
    """A rounded box centred at ``(x, y)`` with ``text`` wrapped to its width.

    ``text`` is plain text, escaped so module names, issue numbers and
    percentages print as written; pass ``latex=True`` to typeset it as LaTeX.
    """
    if not width > 2 * _PAD or not height > 0:
        raise ValueError(f"box needs a width above {2 * _PAD} cm and a positive height, not {width} x {height}")
    hw, hh = 0.5 * width, 0.5 * height
    outline = Polyline.of([(x - hw, y - hh), (x + hw, y - hh), (x + hw, y + hh), (x - hw, y + hh)], style,
                          role=role, closed=True)
    label = Label((x, y), text if latex else escape_latex(text),
                  f"{text_style},text width={width - 2 * _PAD:.2f}cm", role=role)
    return Box(x, y, width, height, (outline, label))


def connector(start: Box, end: Box, *, style: str = "connector", role: str = "", gap: float = 0.08) -> Arrow:
    """An arrow from the edge of ``start`` to the edge of ``end``, along their centre line."""
    if (abs(end.x - start.x) < 0.5 * (start.width + end.width)
            and abs(end.y - start.y) < 0.5 * (start.height + end.height)):
        raise ValueError("the boxes overlap: nothing to connect")
    a = np.array(start.boundary_point(end.center), dtype=float)
    b = np.array(end.boundary_point(start.center), dtype=float)
    d = b - a
    length = float(np.hypot(*d))
    if length <= 2 * gap + _MIN_ARROW:
        raise ValueError(f"the boxes are {length:.2f} cm apart: too close for a {gap:g} cm gap and a visible arrow")
    u = d / length
    return Arrow(tuple(float(v) for v in a + gap * u), tuple(float(v) for v in b - gap * u), style, role=role)


def band(x0: float, x1: float, y0: float, y1: float, text: str = "", *, style: str = "concept band",
         role: str = "") -> List:
    """A shaded background strip from ``(x0, y0)`` to ``(x1, y1)``, its label at the top-left."""
    if not (x1 > x0 and y1 > y0):
        raise ValueError("band needs x1 > x0 and y1 > y0")
    items: List = [Polyline.of([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], style, role=role, closed=True)]
    if text:
        items.append(Label((x0 + _PAD, y1 - _PAD), text, "concept band label", anchor="north west", role=role))
    return items


def database(x: float, y: float, width: float, height: float, text: str, *, style: str = "concept database",
             text_style: str = "concept text", role: str = "", latex: bool = False, ry: Optional[float] = None) -> Box:
    """A database cylinder centred at ``(x, y)``: a body and a front rim, drawn as polylines.

    The body is one closed outline -- the top ellipse's back half, the
    sides and the bottom ellipse's front half -- and the rim is the top
    ellipse's front half, so the drum reads as a store without a TikZ shape
    library. Connectors attach to its bounding box like a :func:`box`.
    ``ry`` sets the rim's half-height; by default it scales with the drum.
    """
    if not width > 2 * _PAD or not height > 0:
        raise ValueError(f"database needs a width above {2 * _PAD} cm and a positive height, not {width} x {height}")
    hw, hh = 0.5 * width, 0.5 * height
    if ry is None:
        ry = min(0.18 * width, 0.25 * height)  # ellipse half-height
    elif not 0.0 < ry <= 0.5 * height:
        raise ValueError(f"ry must be in (0, height/2], not {ry!r}")
    t = np.linspace(0.0, np.pi, 25)
    top_back = np.stack([x + hw * np.cos(t), y + hh - ry + ry * np.sin(t)], -1)          # right to left, over
    bottom_front = np.stack([x - hw * np.cos(t), y - hh + ry - ry * np.sin(t)], -1)     # left to right, under
    body = Polyline.of(np.concatenate([top_back, bottom_front]), style, role=role, closed=True)
    rim = Polyline.of(np.stack([x - hw * np.cos(t), y + hh - ry - ry * np.sin(t)], -1), "concept database rim",
                      role=role)
    # centred in the body below the rim: between the rim's lowest point (y + hh - 2 ry) and the base (y - hh)
    label = Label((x, y - ry), text if latex else escape_latex(text),
                  f"{text_style},text width={width - 2 * _PAD:.2f}cm", role=role)
    return Box(x, y, width, height, (body, rim, label))
