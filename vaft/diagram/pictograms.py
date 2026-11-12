"""Semantic vector pictograms.

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
    "display_name",
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
_EXTENT = 20.0  # grid units: every symbol's larger side after optical normalisation (Lucide's live area)


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


def _optical(strokes: List[Tuple[List[Point], bool]]) -> List[Tuple[List[Point], bool]]:
    """Strokes scaled about their centre so the larger side of their extent is :data:`_EXTENT` grid units.

    Equal optical size: a narrow source icon (a terminal prompt, a flat trace) no longer reads
    smaller than a full-frame one beside it. Zero-length dot strokes do not count toward the extent.
    """
    drawn = [pt for points, _ in strokes if sum(math.dist(a, b) for a, b in zip(points, points[1:])) >= 0.1
             for pt in points]
    xs, ys = [x for x, _ in drawn], [y for _, y in drawn]
    cx, cy = 0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys))
    k = _EXTENT / max(max(xs) - min(xs), max(ys) - min(ys))
    return [([(0.5 * _GRID + k * (x - cx), 0.5 * _GRID + k * (y - cy)) for x, y in points], closed)
            for points, closed in strokes]


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
    # a full poloidal cut, symmetric about the machine axis: the central column, and on each side
    # the D-shaped vacuum vessel with the closed plasma inside it
    "tokamak": '<rect x="10.25" y="3" width="3.5" height="18" rx="1"/>'
               '<path d="M15.5 3C20 3 22.5 7 22.5 12S20 21 15.5 21Z"/>'
               '<path d="M17 7C20 7 21 9.5 21 12S20 17 17 17C16.6 14 16.6 10 17 7Z"/>'
               '<path d="M8.5 3C4 3 1.5 7 1.5 12S4 21 8.5 21Z"/>'
               '<path d="M7 7C4 7 3 9.5 3 12S4 17 7 17C7.4 14 7.4 10 7 7Z"/>',
    # a raw time trace: an irregular sampled waveform on its baseline
    "signal": '<path d="M2 12h2.5l2-8 3 16 3-12 2 8 1.5-4H22"/>',
    # nested, outward-shifted D-shaped flux surfaces recovered from the measurements (crosses) outside them
    "reconstruction": '<path d="M7 3.5C14 3.5 17.5 7.5 17.5 12S14 20.5 7 20.5C5.8 16 5.8 8 7 3.5Z"/>'
                      '<path d="M9.5 8C13 8 14.8 10 14.8 12S13 16 9.5 16C9 14 9 10 9.5 8Z"/><path d="M11.8 12h.01"/>'
                      '<path d="M19.8 4.8l2.4 2.4M22.2 4.8l-2.4 2.4"/><path d="M19.8 10.8l2.4 2.4M22.2 10.8l-2.4 2.4"/>'
                      '<path d="M19.8 16.8l2.4 2.4M22.2 16.8l-2.4 2.4"/>',
    # an estimate of an unknown: a distribution with its central value and uncertainty interval
    "inference": '<path d="M2 20C6.5 20 8 5 12 5S17.5 20 22 20"/><path d="M12 5v15"/>'
                 '<path d="M8.6 13h6.8M8.6 11.5v3M15.4 11.5v3"/>',
    # a model (the function box) generating a signal: synthetic data, not a measurement
    "synthetic": '<rect x="2" y="2" width="10" height="10" rx="1.5"/><path d="M5.5 10c1.2 0 1.6-.6 1.6-1.6V6.6c0-1.1.6-1.8 1.8-1.7"/>'
                 '<path d="M5.5 7.3h3.2"/><path d="M12 12l3 3M15 12.5V15h-2.5"/>'
                 '<path d="M10 19.5h2l1.5-4 2 7 2-6 1.5 3H22"/>',
    # a transformation, read left to right: a noisy signal enters the processing step and leaves clean
    "processing": '<path d="M1 12h1.2l1.2-4.5 1.4 9 1.2-6.5 1 2H8"/><rect x="8" y="4" width="8" height="16" rx="2"/>'
                  '<path d="M10 8l4 4-4 4"/><path d="M16 12c1.2-5 2.6-5 3.6 0s2.4 5 3.4 0"/>',
    # two systems exchanging data both ways
    "interface": '<rect x="2" y="5" width="6" height="14" rx="1.5"/><rect x="16" y="5" width="6" height="14" rx="1.5"/>'
                 '<path d="M9.5 9.5h5M12.5 7.5l2 2-2 2"/><path d="M14.5 14.5h-5M11.5 12.5l-2 2 2 2"/>',
    # measured points with error bars on the computed curve, and the agreement check
    "validation": '<path d="M2 21C8 20 14 14 21 3"/><circle cx="7" cy="19.3" r="1"/><circle cx="12" cy="16.2" r="1"/>'
                  '<circle cx="16.5" cy="10.4" r="1"/><path d="M7 16.5v5.6M12 13.4v5.6M16.5 7.6v5.6"/>'
                  '<path d="M3 7l2.5 2.5L10 5"/>',
    # two results set side by side on their own axes
    "comparison": '<path d="M2 5v14h9"/><path d="M13 5v14h9"/><path d="M4 16c1.5-6 3-7 5-3"/>'
                  '<path d="M15 16c1.5-6 3.5-7 5-2"/>',
}

#: semantic name -> (meaning, ``"lucide:<icon>"`` or ``"vaft"``)
_REGISTRY = {
    # experiment and machine
    "experiment": ("an experiment or laboratory activity", "lucide:flask-conical"),
    "tokamak": ("a tokamak device or its machine description", "vaft"),
    "signal": ("a raw measured signal", "vaft"),
    # data
    "database": ("structured data storage", "lucide:database"),
    "archive": ("a long-term archive", "lucide:archive"),
    "file": ("a data file or document", "lucide:file-text"),
    "input_file": ("an input file for a solver", "lucide:file-input"),
    "output_file": ("a solver's native output file", "lucide:file-output"),
    "standardized": ("a standardized, structured scientific representation (an IMAS-like tree)", "lucide:list-tree"),
    "history": ("a dated record or history", "lucide:history"),
    "library": ("a body of reusable knowledge", "lucide:library"),
    # computation and processing
    "processing": ("data processing: transforming raw data into qualified data", "vaft"),
    "computation": ("a numerical computation, solver or simulation", "lucide:cpu"),
    "cluster": ("a compute server or HPC cluster", "lucide:server"),
    "laptop": ("a local workstation", "lucide:laptop"),
    "calculation": ("a deterministic calculation of derived quantities", "lucide:calculator"),
    "model": ("a mathematical or theoretical model (a function, not a summation)", "lucide:square-function"),
    "reconstruction": ("a state (flux surfaces) reconstructed from measurements", "vaft"),
    "inference": ("an estimate of an unknown quantity, with its uncertainty", "vaft"),
    "synthetic": ("data generated by a model or completed by assumptions, never measured", "vaft"),
    "parameters": ("chosen parameters, settings or prior assumptions", "lucide:sliders-horizontal"),
    "coordinates": ("a coordinate, sign or model convention", "lucide:move-3d"),
    # software and interfaces
    "code": ("a programming interface", "lucide:code"),
    "source_code": ("an externally maintained source code", "lucide:file-code"),
    "terminal": ("a command-line interface", "lucide:terminal"),
    "monitor": ("a graphical interface", "lucide:monitor"),
    "repository": ("a version-controlled code repository and its documentation", "lucide:folder-git-2"),
    "integration": ("the combination of different systems or data structures", "lucide:puzzle"),
    "adapter": ("an adapter mediating compatibility between two systems", "lucide:cable"),
    "interface": ("a two-way data exchange between two systems", "vaft"),
    "package": ("an installable package or environment", "lucide:package"),
    # research and scientific process
    "researcher": ("a human researcher", "lucide:user"),
    "collaboration": ("a group of researchers", "lucide:users"),
    "ai_agent": ("an AI agent", "lucide:bot"),
    "workflow": ("a traceable workflow or pipeline", "lucide:workflow"),
    "validation": ("agreement of computed values with independent measurements", "vaft"),
    "comparison": ("results compared side by side", "vaft"),
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
        symbols[name] = Symbol(name, meaning, source, _normalise(_optical(_strokes(markup))))
    return symbols


#: Every registered pictogram, by semantic name.
SYMBOLS: Dict[str, Symbol] = _build()


#: display names that capitalisation alone would get wrong
_DISPLAY = {"ai_agent": "AI agent"}


def display_name(name: str) -> str:
    """The reader-facing name of semantic ``name``: ``"input_file"`` -> ``"Input file"``."""
    symbol(name)
    return _DISPLAY.get(name, name.replace("_", " ").capitalize())


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


def corner(box, size: float = 0.62, *, dip: float = 0.15) -> Point:
    """Where a :func:`badge` of ``size`` cm clips onto ``box``'s top-left corner (``box`` has ``x, y, width, height``).

    The disc dips ``dip`` cm into the box -- 0.15 cm reads as the box's while clearing a first text
    line that spans the box; ``dip=0`` rests it on the edge for a box its text fills -- and sits
    0.19 cm in from the left edge, clear of the rounded corner.
    """
    return (box.x - 0.5 * box.width + 0.5 * size + 0.19, box.y + 0.5 * box.height + 0.5 * size - dip)


def pictogram_catalog(*, labels: bool = True):
    """Every registered pictogram with its display name and identifier: the symbol gallery.

    ``labels=False`` draws the symbols alone.
    """
    from ._concept import escape_latex
    from ._render import Diagram

    if not isinstance(labels, bool):
        raise ValueError(f"labels must be True or False, not {labels!r}")
    columns, cell, size = 8, 2.4, 0.9
    items: list = []
    for i, name in enumerate(SYMBOLS):
        x, y = (i % columns) * cell, -(i // columns) * 2.3
        items += pictogram(name, (x, y), size, role=f"symbol:catalog:{name}").items
        if labels:
            # the reader-facing name, upright, and the identifier to pass to pictogram() beneath it
            items.append(Label((x, y - 0.7), escape_latex(display_name(name)), "concept plain", role=f"catalog:{name}"))
            items.append(Label((x, y - 1.12), r"\texttt{" + name.replace("_", r"\_") + "}", "concept reference",
                               role=f"catalog:{name}"))
    if labels:
        items.append(Label((0.5 * (columns - 1) * cell, 1.0), r"\textbf{Semantic pictograms}", "concept group title",
                           anchor="south", role="title"))
    return Diagram("pictogram_catalog", Scene(tuple(items)), model={"symbols": dict(SYMBOLS)})
