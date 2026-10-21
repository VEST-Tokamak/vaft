"""Semantic vector pictograms (#2101).

A pictogram is a small stroke drawing that names a *kind of thing* -- a database, a solver, a
researcher -- so a concept diagram reads at a glance. It is a presentation component, not a
model output: its geometry carries no physical dimension, position or relationship, and it never
stands in for model-derived geometry (flux surfaces, islands, coordinates, field lines), which
stays with the builders and :mod:`vaft.formula`. A pictogram supplements a label; it does not
replace one.

Each symbol has a stable semantic name (``"database"``, ``"signal"``, ...) so the same concept
draws the same way in every figure. The geometry comes from two sources, both converted here into
ordinary :class:`~vaft.diagram._scene.Polyline` items, so every renderer (TikZ/SVG/PDF and the
Matplotlib animation path) draws them without a new primitive and nothing is fetched at run time:

* general-purpose icons vendored verbatim from a pinned Lucide release (:mod:`._lucide`, ISC with
  MIT for the Feather-derived icons; see ``THIRD_PARTY_NOTICES.md``);
* VAFT scientific symbols written here in the same 24-unit SVG grid where a generic icon would
  mislead (``tokamak``, ``signal``, ``reconstruction``, ``inference``, ``synthetic``).

Only a restricted SVG subset is understood: ``path`` (``M L H V C S Q A Z``, absolute and
relative), ``line``, ``polyline``, ``circle``, ``ellipse`` and ``rect`` (with ``rx``). Anything
else raises. Curves and arcs are flattened with a fixed chord-error bound (:data:`TOLERANCE`), so
a symbol always yields the same points; a zero-length stroke (Lucide's dot idiom) becomes a small
closed ring.

>>> from vaft.diagram.pictograms import pictogram
>>> items = pictogram("database", (2.0, 3.0), 0.8, role="symbol:stage:database").items

Roles must start with ``symbol:`` so pictogram items never collide with a builder's ``node:`` /
``edge:`` roles, which the topology tests query.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from ._lucide import ICONS, LUCIDE_VERSION
from ._scene import Label, Point, Polyline, Scene

__all__ = [
    "BADGE",
    "BADGE_MUTED",
    "PICTOGRAM",
    "PICTOGRAM_MUTED",
    "SYMBOLS",
    "TOLERANCE",
    "Symbol",
    "badge",
    "corner",
    "names",
    "pictogram",
    "pictogram_catalog",
    "symbol",
]

#: The pictogram stroke: inline TikZ options (a template style would re-stale every diagram).
#: One weight at every size, so icons of different sizes in one figure match.
PICTOGRAM = "draw=islanddark,line width=0.7pt"
#: For something planned or not yet available: grey and dashed, never the full-weight icon.
PICTOGRAM_MUTED = "draw=black!45,line width=0.7pt,dashed"
#: The disc behind a :func:`badge`: white, so the icon reads where it overlaps a box outline.
BADGE = "draw=islanddark,line width=0.6pt,fill=white"
#: The disc behind a muted (planned) badge.
BADGE_MUTED = "draw=black!45,line width=0.6pt,dashed,fill=white"
#: Maximum distance between a flattened curve and the true curve, in 24-unit grid units
#: (0.01/24 of the icon size: 0.25 um on a 0.6 cm icon).
TOLERANCE = 0.01

_GRID = 24.0
_DOT_RADIUS = 0.5  # grid units: a zero-length stroke drawn as a ring this size


@dataclass(frozen=True)
class Symbol:
    """One registered pictogram.

    ``strokes`` are ``(points, closed)`` pairs in a unit box centred on the origin (the 24-unit
    source grid mapped to ``[-0.5, 0.5]``, y up); ``source`` names where the geometry came from
    (``"lucide:<name>@<version>"`` or ``"vaft"``).
    """

    name: str
    meaning: str
    source: str
    strokes: Tuple[Tuple[Tuple[Point, ...], bool], ...]


# --------------------------------------------------------------------------
# restricted SVG -> strokes
# --------------------------------------------------------------------------

_ELEMENT = re.compile(r"<([a-z]+)\b([^>]*)/>")
_ATTRIBUTE = re.compile(r"([a-z][a-z0-9-]*)=\"([^\"]*)\"")
_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_SEPARATOR = re.compile(r"[\s,]*")


def _segments(angle: float, radius: float) -> int:
    """Chords for an arc of ``angle`` radians and ``radius`` within :data:`TOLERANCE`."""
    if radius <= TOLERANCE:
        return 1
    step = 2.0 * math.acos(1.0 - TOLERANCE / radius)
    return max(1, math.ceil(abs(angle) / step))


def _bezier(points: Sequence[Point]) -> List[Point]:
    """Flatten a quadratic or cubic Bezier (start excluded) within :data:`TOLERANCE`."""
    p = points
    if len(p) == 3:
        bend = math.dist((p[0][0] - 2 * p[1][0] + p[2][0], p[0][1] - 2 * p[1][1] + p[2][1]), (0, 0))
        n = max(1, math.ceil(math.sqrt(2.0 * bend / (8.0 * TOLERANCE))))
        return [((1 - t) ** 2 * p[0][0] + 2 * (1 - t) * t * p[1][0] + t * t * p[2][0],
                 (1 - t) ** 2 * p[0][1] + 2 * (1 - t) * t * p[1][1] + t * t * p[2][1])
                for t in (i / n for i in range(1, n + 1))]
    bend = max(math.dist((p[i][0] - 2 * p[i + 1][0] + p[i + 2][0], p[i][1] - 2 * p[i + 1][1] + p[i + 2][1]), (0, 0))
               for i in (0, 1))
    n = max(1, math.ceil(math.sqrt(6.0 * bend / (8.0 * TOLERANCE))))
    out = []
    for i in range(1, n + 1):
        t = i / n
        a, b, c, d = (1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t * t, t ** 3
        out.append((a * p[0][0] + b * p[1][0] + c * p[2][0] + d * p[3][0],
                    a * p[0][1] + b * p[1][1] + c * p[2][1] + d * p[3][1]))
    return out


def _arc(start: Point, rx: float, ry: float, phi_deg: float, large: bool, sweep: bool, end: Point) -> List[Point]:
    """Flatten an SVG elliptical arc (start excluded), endpoint parametrisation (SVG 1.1 F.6.5)."""
    if start == end:
        return []
    rx, ry = abs(rx), abs(ry)
    if rx == 0 or ry == 0:
        return [end]
    phi = math.radians(phi_deg)
    cos, sin = math.cos(phi), math.sin(phi)
    dx, dy = 0.5 * (start[0] - end[0]), 0.5 * (start[1] - end[1])
    x1, y1 = cos * dx + sin * dy, -sin * dx + cos * dy
    scale = (x1 / rx) ** 2 + (y1 / ry) ** 2
    if scale > 1:
        rx, ry = rx * math.sqrt(scale), ry * math.sqrt(scale)
    num = rx * rx * ry * ry - rx * rx * y1 * y1 - ry * ry * x1 * x1
    den = rx * rx * y1 * y1 + ry * ry * x1 * x1
    root = math.sqrt(max(0.0, num / den))
    if large == sweep:
        root = -root
    cx1, cy1 = root * rx * y1 / ry, -root * ry * x1 / rx
    cx = cos * cx1 - sin * cy1 + 0.5 * (start[0] + end[0])
    cy = sin * cx1 + cos * cy1 + 0.5 * (start[1] + end[1])
    theta = math.atan2((y1 - cy1) / ry, (x1 - cx1) / rx)
    delta = math.atan2((-y1 - cy1) / ry, (-x1 - cx1) / rx) - theta
    if sweep and delta < 0:
        delta += 2 * math.pi
    elif not sweep and delta > 0:
        delta -= 2 * math.pi
    n = _segments(delta, max(rx, ry))
    out = []
    for i in range(1, n):
        t = theta + delta * i / n
        ex, ey = rx * math.cos(t), ry * math.sin(t)
        out.append((cx + cos * ex - sin * ey, cy + sin * ex + cos * ey))
    return out + [end]


class _Reader:
    """A cursor over a path's ``d`` string."""

    def __init__(self, d: str):
        self.d, self.i = d, 0

    def _skip(self) -> None:
        self.i = _SEPARATOR.match(self.d, self.i).end()

    def done(self) -> bool:
        self._skip()
        return self.i >= len(self.d)

    def command(self) -> str | None:
        self._skip()
        if self.i < len(self.d) and self.d[self.i].isalpha():
            self.i += 1
            return self.d[self.i - 1]
        return None

    def number(self) -> float:
        self._skip()
        match = _NUMBER.match(self.d, self.i)
        if match is None:
            raise ValueError(f"expected a number at {self.d[self.i:self.i + 12]!r} in path {self.d!r}")
        self.i = match.end()
        return float(match.group())

    def flag(self) -> bool:
        self._skip()
        char = self.d[self.i:self.i + 1]
        if char not in ("0", "1"):
            raise ValueError(f"expected an arc flag at {self.d[self.i:self.i + 12]!r} in path {self.d!r}")
        self.i += 1
        return char == "1"

    def more(self) -> bool:
        """Whether another implicit parameter set follows the current command."""
        self._skip()
        return self.i < len(self.d) and not self.d[self.i].isalpha()


def _path(d: str) -> List[Tuple[List[Point], bool]]:
    """Strokes of a path's ``d`` string: the restricted command subset, absolute and relative."""
    reader = _Reader(d)
    strokes: List[Tuple[List[Point], bool]] = []
    current: List[Point] = []
    pen: Point = (0.0, 0.0)
    start: Point = (0.0, 0.0)
    control: Point | None = None  # the last cubic's second control point, for S
    previous = ""

    def flush(closed: bool = False) -> None:
        nonlocal current
        if len(current) > 1 or closed:
            strokes.append((current, closed))
        current = []

    command = reader.command()
    if command is None:
        raise ValueError(f"path {d!r} does not start with a command")
    while command is not None:
        rel = command.islower()
        op = command.upper()
        if op not in "MLHVCSQAZ":
            raise ValueError(f"unsupported path command {command!r} in {d!r}")
        first = True
        while True:
            def xy() -> Point:
                x, y = reader.number(), reader.number()
                return (pen[0] + x, pen[1] + y) if rel else (x, y)

            if op == "Z":
                if current:
                    flush(closed=True)
                pen = start
                control = None
            elif op == "M":
                if first:
                    flush()
                    pen = xy()
                    start = pen
                    current = [pen]
                else:  # implicit lineto after a moveto
                    pen = xy()
                    current.append(pen)
                control = None
            else:
                if not current:
                    current = [pen]
                if op == "L":
                    pen = xy()
                    current.append(pen)
                    control = None
                elif op == "H":
                    x = reader.number()
                    pen = (pen[0] + x if rel else x, pen[1])
                    current.append(pen)
                    control = None
                elif op == "V":
                    y = reader.number()
                    pen = (pen[0], pen[1] + y if rel else y)
                    current.append(pen)
                    control = None
                elif op in "CS":
                    if op == "C":
                        c1 = xy()
                    else:
                        c1 = (2 * pen[0] - control[0], 2 * pen[1] - control[1]) if control and previous in ("C", "S") else pen
                    c2, end = xy(), xy()
                    current += _bezier((pen, c1, c2, end))
                    pen, control = end, c2
                elif op == "Q":
                    c1, end = xy(), xy()
                    current += _bezier((pen, c1, end))
                    pen, control = end, None
                elif op == "A":
                    rx, ry, phi = reader.number(), reader.number(), reader.number()
                    large, sweep = reader.flag(), reader.flag()
                    end = xy()
                    current += _arc(pen, rx, ry, phi, large, sweep, end)
                    pen, control = end, None
            previous = op
            first = False
            if op == "Z" or not reader.more():
                break
        command = reader.command()
        if command is None and not reader.done():
            raise ValueError(f"unparsed text in path {d!r}")
    flush()
    return strokes


def _ellipse(cx: float, cy: float, rx: float, ry: float) -> List[Point]:
    n = max(8, _segments(2 * math.pi, max(rx, ry)))
    return [(cx + rx * math.cos(2 * math.pi * i / n), cy + ry * math.sin(2 * math.pi * i / n)) for i in range(n)]


def _rect(x: float, y: float, w: float, h: float, rx: float, ry: float) -> List[Point]:
    rx, ry = min(rx, w / 2), min(ry, h / 2)
    if rx <= 0 or ry <= 0:
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    corners = ((x + w - rx, y + ry, -0.5 * math.pi), (x + w - rx, y + h - ry, 0.0),
               (x + rx, y + h - ry, 0.5 * math.pi), (x + rx, y + ry, math.pi))
    n = _segments(0.5 * math.pi, max(rx, ry))
    return [(cx + rx * math.cos(a0 + 0.5 * math.pi * i / n), cy + ry * math.sin(a0 + 0.5 * math.pi * i / n))
            for cx, cy, a0 in corners for i in range(n + 1)]


def _strokes(markup: str) -> List[Tuple[List[Point], bool]]:
    """Strokes, in the 24-unit SVG grid (y down), of restricted-subset SVG child elements."""
    strokes: List[Tuple[List[Point], bool]] = []
    rest = _ELEMENT.sub("", markup).strip()
    if rest:
        raise ValueError(f"unsupported SVG content {rest[:40]!r}")
    for tag, raw in _ELEMENT.findall(markup):
        attrs = dict(_ATTRIBUTE.findall(raw))
        num = {k: float(v) for k, v in attrs.items() if k not in ("d", "points")}
        if tag == "path":
            strokes += _path(attrs["d"])
        elif tag == "line":
            strokes.append(([(num["x1"], num["y1"]), (num["x2"], num["y2"])], False))
        elif tag == "polyline":
            values = [float(v) for v in _NUMBER.findall(attrs["points"])]
            strokes.append((list(zip(values[::2], values[1::2])), False))
        elif tag == "circle":
            strokes.append((_ellipse(num["cx"], num["cy"], num["r"], num["r"]), True))
        elif tag == "ellipse":
            strokes.append((_ellipse(num["cx"], num["cy"], num["rx"], num["ry"]), True))
        elif tag == "rect":
            rx = num.get("rx", num.get("ry", 0.0))
            strokes.append((_rect(num.get("x", 0.0), num.get("y", 0.0), num["width"], num["height"],
                                  rx, num.get("ry", rx)), True))
        else:
            raise ValueError(f"unsupported SVG element <{tag}>")
    return strokes


def _normalise(strokes: List[Tuple[List[Point], bool]]) -> Tuple[Tuple[Tuple[Point, ...], bool], ...]:
    """Grid strokes into the unit box (y up); zero-length strokes become small rings."""
    out = []
    for points, closed in strokes:
        length = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
        if length < 0.1:
            points, closed = _ellipse(points[0][0], points[0][1], _DOT_RADIUS, _DOT_RADIUS), True
        unit = tuple((round(x / _GRID - 0.5, 6) + 0.0, round(0.5 - y / _GRID, 6) + 0.0) for x, y in points)
        out.append((unit, closed))
    return tuple(out)


# --------------------------------------------------------------------------
# the registry
# --------------------------------------------------------------------------

#: VAFT scientific symbols, in the same 24-unit grid as Lucide. Symbolic shapes only: the
#: contours of ``reconstruction`` or the D of ``tokamak`` are not any machine's geometry.
_VAFT = {
    # a poloidal cut of a torus: two D-shaped cross-sections about the symmetry axis
    "tokamak": '<path d="M12 2v20"/>'
               '<path d="M15 5C20.5 5 22 9 22 12S20.5 19 15 19C16.4 15 16.4 9 15 5Z"/>'
               '<path d="M9 5C3.5 5 2 9 2 12S3.5 19 9 19C7.6 15 7.6 9 9 5Z"/>',
    # a raw time trace: an irregular sampled waveform on its baseline
    "signal": '<path d="M2 13h3l2.5-8 3 14 3-11 2 7 1.5-2H22"/>',
    # nested, outward-shifted closed contours around an axis: a model-based reconstruction
    "reconstruction": '<ellipse cx="12" cy="12" rx="9" ry="10"/><ellipse cx="13" cy="12" rx="5.6" ry="6.6"/>'
                      '<ellipse cx="13.8" cy="12" rx="2.4" ry="3"/><path d="M14.2 12h.01"/>',
    # scattered data with a fitted curve through it: estimation from data via a model
    "inference": '<path d="M2 20C8 17 13 12 22 4"/><circle cx="4" cy="16" r="1.4"/><circle cx="8.5" cy="17.5" r="1.4"/>'
                 '<circle cx="11" cy="11.5" r="1.4"/><circle cx="15.5" cy="12" r="1.4"/>'
                 '<circle cx="18" cy="5.5" r="1.4"/>',
    # a trace whose second half is completed by assumption (dashes drawn as separate strokes)
    "synthetic": '<path d="M2 15l3-6 3 9 3-7"/><path d="M12.6 10.6l1.4-1.6"/><path d="M15.6 8.6l1.8.5"/>'
                 '<path d="M19 9.8l2.2 1"/>',
}

#: semantic name -> (meaning, ``"lucide:<icon>"`` or ``"vaft"``)
_REGISTRY = {
    # experiment and machine
    "experiment": ("an experiment or laboratory activity", "lucide:flask-conical"),
    "tokamak": ("a tokamak device or its machine description", "vaft"),
    "signal": ("a raw measured signal", "vaft"),
    # data
    "database": ("a database or data repository", "lucide:database"),
    "archive": ("a long-term archive", "lucide:archive"),
    "file": ("a data file or document", "lucide:file-text"),
    "input_file": ("an input file for a solver", "lucide:file-input"),
    "output_file": ("a solver's native output file", "lucide:file-output"),
    "standardized": ("a standardized, structured scientific representation (an IMAS-like tree)", "lucide:list-tree"),
    "history": ("a dated record or history", "lucide:history"),
    "library": ("a body of collected knowledge", "lucide:library"),
    # computation and processing
    "processing": ("data processing and qualification", "lucide:funnel"),
    "computation": ("a numerical computation, solver or simulation", "lucide:cpu"),
    "cluster": ("a compute server or HPC cluster", "lucide:server"),
    "laptop": ("a local workstation", "lucide:laptop"),
    "calculation": ("a deterministic calculation of derived quantities", "lucide:calculator"),
    "model": ("a mathematical or theoretical model", "lucide:sigma"),
    "reconstruction": ("a model-based reconstruction", "vaft"),
    "inference": ("an estimate inferred from data through a model", "vaft"),
    "synthetic": ("a quantity completed by assumptions", "vaft"),
    "parameters": ("chosen parameters, settings or prior assumptions", "lucide:sliders-horizontal"),
    "coordinates": ("a coordinate, sign or model convention", "lucide:axis-3d"),
    # software and interfaces
    "code": ("a programming interface", "lucide:code"),
    "source_code": ("an externally maintained source code", "lucide:file-code"),
    "terminal": ("a command-line interface", "lucide:terminal"),
    "monitor": ("a graphical interface", "lucide:monitor"),
    "repository": ("a code repository and its documentation", "lucide:book-open"),
    "integration": ("a protocol integration point", "lucide:plug"),
    "adapter": ("an adapter between two systems", "lucide:cable"),
    "interface": ("a standardized data interface", "lucide:network"),
    "package": ("an installable package or environment", "lucide:package"),
    # research and scientific process
    "researcher": ("a human researcher", "lucide:user"),
    "collaboration": ("a group of researchers", "lucide:users"),
    "ai_agent": ("an AI agent", "lucide:bot"),
    "workflow": ("a traceable workflow or pipeline", "lucide:workflow"),
    "validation": ("verification and validation", "lucide:shield-check"),
    "comparison": ("comparison and synthesis", "lucide:scale"),
    "analysis": ("analysis and visualization of results", "lucide:chart-line"),
    "discovery": ("a discovery or new question", "lucide:lightbulb"),
}


def _build() -> Dict[str, Symbol]:
    symbols = {}
    for name, (meaning, origin) in _REGISTRY.items():
        if origin == "vaft":
            markup, source = _VAFT[name], "vaft"
        else:
            icon = origin.split(":", 1)[1]
            markup, source = ICONS[icon][0], f"lucide:{icon}@{LUCIDE_VERSION}"
        symbols[name] = Symbol(name, meaning, source, _normalise(_strokes(markup)))
    return symbols


#: Every registered pictogram, by semantic name.
SYMBOLS: Dict[str, Symbol] = _build()


def names() -> Tuple[str, ...]:
    """The registered semantic names, in registry order."""
    return tuple(SYMBOLS)


def symbol(name: str) -> Symbol:
    """The registered :class:`Symbol` called ``name``; unknown names raise ``KeyError``."""
    try:
        return SYMBOLS[name]
    except KeyError:
        raise KeyError(f"unknown pictogram {name!r}; registered: {', '.join(SYMBOLS)}") from None


def pictogram(name: str, at: Point, size: float = 0.6, *, role: str, style: str = PICTOGRAM) -> Scene:
    """Pictogram ``name`` as scene polylines, centred on ``at`` and ``size`` cm across.

    ``role`` tags every item and must start with ``symbol:`` (e.g. ``"symbol:node:database"``);
    ``style`` is :data:`PICTOGRAM` or :data:`PICTOGRAM_MUTED`. The aspect ratio is kept and the
    stroke weight does not scale with ``size``.
    """
    if not role.startswith("symbol:"):
        raise ValueError(f"pictogram role must start with 'symbol:', not {role!r}")
    if not size > 0:
        raise ValueError(f"pictogram size must be positive, not {size!r}")
    x0, y0 = at
    return Scene(tuple(
        Polyline(tuple((x0 + size * x, y0 + size * y) for x, y in points), style, role, closed)
        for points, closed in symbol(name).strokes
    ))


def badge(name: str, at: Point, size: float = 0.6, *, role: str, style: str = PICTOGRAM) -> Scene:
    """Pictogram ``name`` on a white disc ``size`` cm across, centred on ``at``.

    For a tight box whose text fills its width: centre the badge on the box's top edge near a
    corner, so the disc straddles the outline instead of taking text space. A
    :data:`PICTOGRAM_MUTED` icon gets the grey dashed disc :data:`BADGE_MUTED`.
    """
    disc = Polyline(tuple((at[0] + 0.5 * size * math.cos(2 * math.pi * i / 48),
                           at[1] + 0.5 * size * math.sin(2 * math.pi * i / 48)) for i in range(48)),
                    BADGE_MUTED if style == PICTOGRAM_MUTED else BADGE, role, True)
    return Scene((disc,)) + pictogram(name, at, 0.62 * size, role=role, style=style)


def corner(box, size: float = 0.62) -> Point:
    """Where a :func:`badge` of ``size`` cm clips onto ``box``'s top-left corner (``box`` has ``x, y, width, height``).

    The disc dips 0.15 cm into the box -- enough to read as the box's, little enough to clear a first
    text line that spans the box -- and sits 0.19 cm in from the left edge, clear of the rounded corner.
    """
    return (box.x - 0.5 * box.width + 0.5 * size + 0.19, box.y + 0.5 * box.height + 0.5 * size - 0.15)


def pictogram_catalog(*, labels: bool = True):
    """Every registered pictogram with its semantic name: the symbol gallery (#2101).

    ``labels=False`` draws the symbols alone.
    """
    from ._render import Diagram

    if not isinstance(labels, bool):
        raise ValueError(f"labels must be True or False, not {labels!r}")
    columns, cell, size = 8, 2.4, 0.9
    items: list = []
    for i, name in enumerate(SYMBOLS):
        x, y = (i % columns) * cell, -(i // columns) * 1.9
        items += pictogram(name, (x, y), size, role=f"symbol:catalog:{name}").items
        if labels:
            items.append(Label((x, y - 0.75), r"\texttt{" + name.replace("_", r"\_") + "}", "concept annotation",
                               role=f"catalog:{name}"))
    if labels:
        items.append(Label((0.5 * (columns - 1) * cell, 1.0), r"\textbf{Semantic pictograms}", "concept group title",
                           anchor="south", role="title"))
    return Diagram("pictogram_catalog", Scene(tuple(items)), model={"symbols": dict(SYMBOLS)})
