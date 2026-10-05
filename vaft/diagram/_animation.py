"""Animated diagrams: one scientific model over a trajectory of its states (issues #1049, #1053).

A diagram's static form stays the TikZ -> SVG path of #890.  Its dynamic form
draws the same :class:`~vaft.diagram._scene.Scene` -- built by the same
model for every state -- with Matplotlib, frame by frame, and hands the frames
to the animation result of :mod:`vaft.plot._animation`: the same
``save("x.mp4" | ".webm" | ".gif")``, ``frames()``, ``metadata`` and notebook
preview as ``plot_*(..., animation=True)``.  No new dependency: the frames are
Matplotlib, the encoder is the optional PyAV of ``vaft[video]`` (``.gif``
needs none).

The Matplotlib drawing is a *preview-grade* rendering of the scene: the
template's colours, line widths and dashes, with labels through mathtext.
It is not the canonical figure -- that remains the SVG -- and a frame is
never compared with it pixel for pixel.
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Mapping, Sequence

import numpy as np

__all__ = ["animate_states", "draw_scene"]

#: The template's colours (templates/standalone.tex ``\definecolor``), as RGB in [0, 1].
_COLOURS = {
    "islandblue": (28, 103, 210), "islanddark": (22, 55, 116), "qblue": (96, 164, 232),
    "meshgray": (150, 150, 150), "chartblue": (16, 24, 140), "driftred": (214, 48, 39),
    "abskinetic": (232, 140, 24), "absfluid": (44, 150, 84), "absequil": (120, 72, 164),
}
_COLOURS = {name: tuple(c / 255 for c in rgb) for name, rgb in _COLOURS.items()}


def _mix(name: str, percent: float, other: str = "white") -> tuple[float, float, float]:
    """TikZ ``name!percent`` (mixed with white) or ``name!percent!other``."""
    base = np.array(_COLOURS.get(name, (0.0, 0.0, 0.0)) if name != "black" else (0.0, 0.0, 0.0))
    if name == "gray":
        base = np.array((0.5, 0.5, 0.5))
    rest = np.array((1.0, 1.0, 1.0)) if other == "white" else np.array(_COLOURS.get(other, (0.0, 0.0, 0.0)))
    if other in ("black",):
        rest = np.array((0.0, 0.0, 0.0))
    return tuple(base * percent / 100 + rest * (1 - percent / 100))


#: Matplotlib equivalents of the template's styles (``\tikzset``).  TikZ widths
#: are points, as Matplotlib's are.  A style not listed draws as a thin black line.
_LINE = {
    "lcfs": {"color": "black", "lw": 0.95},
    "surface": {"color": _mix("gray", 55), "lw": 0.42},
    "rational": {"color": _mix("islandblue", 80, "black"), "lw": 0.65, "ls": "--"},
    "passing": {"color": _mix("islandblue", 45), "lw": 0.5},
    "island": {"color": _COLOURS["islandblue"], "lw": 0.52},
    "separatrix": {"color": _COLOURS["islandblue"], "lw": 0.95},
    "o locus": {"color": _COLOURS["islandblue"], "lw": 1.15},
    "x locus": {"color": _COLOURS["islanddark"], "lw": 0.95, "ls": (0, (3, 2))},
    "o locus hidden": {"color": _COLOURS["islandblue"], "lw": 0.7, "alpha": 0.3},
    "x locus hidden": {"color": _COLOURS["islanddark"], "lw": 0.6, "ls": (0, (3, 2)), "alpha": 0.3},
    "mesh": {"color": _COLOURS["meshgray"], "lw": 0.35},
    "mesh hidden": {"color": _COLOURS["meshgray"], "lw": 0.3, "alpha": 0.3},
    "machine": {"color": "black", "lw": 0.95},
    "cut": {"color": _mix("black", 70), "lw": 0.7, "ls": "-."},
    "axis": {"color": "black", "lw": 0.65},
    "leader": {"color": "black", "lw": 0.55},
    "width arrow": {"color": "black", "lw": 0.65},
}
_FILL = {
    "machine fill": {"facecolor": _mix("gray", 6), "edgecolor": "none"},
    "hole": {"facecolor": "white", "edgecolor": "none"},
}
_ARROW_BOTH = {"width arrow"}
#: Template styles that carry an arrow tip (``->`` / ``<->``): a polyline in
#: one of them ends in a head, as ``\draw[style]`` does in TikZ.
_ARROW_TIPS = {
    "axis": "->", "leader": "->", "width arrow": "<->", "chart axis": "->", "orbit tip": "->",
    "drift": "->", "drift ion": "->", "drift electron": "->", "exb": "->", "vector": "->",
    "field vector": "->", "field arrow": "->", "field line arrow": "->", "frame axis arrow": "->",
    "current": "->",
}
_TEXT = {
    "label": {"fontsize": 9},
    "title": {"fontsize": 14},
    "subtitle": {"fontsize": 8, "color": _mix("black", 75)},
    "note": {"fontsize": 8, "color": _mix("gray", 75, "black"), "multialignment": "center"},
}
_ANCHOR = {
    "center": ("center", "center"), "west": ("left", "center"), "east": ("right", "center"),
    "north": ("center", "top"), "south": ("center", "bottom"),
    "north west": ("left", "top"), "north east": ("right", "top"),
    "south west": ("left", "bottom"), "south east": ("right", "bottom"),
}
#: LaTeX the template's amsmath offers and mathtext does not.
_MATHTEXT = ((r"\tfrac", r"\frac"), (r"\dfrac", r"\frac"), (r"\text{", r"\mathrm{"))


def _mathtext(text: str) -> str:
    """A label mathtext can draw: amsmath spellings mapped, else the plain words.

    A label mathtext cannot parse is shown without its markup rather than
    failing the frame -- the preview is not the canonical figure.
    """
    from matplotlib.mathtext import MathTextParser

    for latex, mathtext in _MATHTEXT:
        text = text.replace(latex, mathtext)
    # \frac12 -> \frac{1}{2}: TeX takes single-token arguments, mathtext does not.
    text = re.sub(r"\\frac(\w)(\w)", r"\\frac{\1}{\2}", text)
    # A TikZ line break: a new line in plain text; mathtext lays out one line
    # only, so a label with math keeps its pieces on one line.
    pieces = text.split("\\\\")
    text = " ".join(pieces) if "$" in text else "\n".join(pieces)
    if "$" not in text:
        return text
    try:
        MathTextParser("path").parse(text)
    except ValueError:
        # Keep the words of the commands (\xi -> xi), drop the markup.
        return re.sub(r"\\([a-zA-Z]+)", r"\1", text).replace("{", "").replace("}", "").replace("$", "")
    return text


def _anchors(item: Any) -> list[tuple[float, float]]:
    """The points an item occupies or starts from, for the picture's extent."""
    for name in ("points",):
        if hasattr(item, name):
            return [tuple(p) for p in getattr(item, name)]
    found = []
    for name in ("at", "start", "end"):
        if hasattr(item, name):
            found.append(tuple(getattr(item, name)))
    return found


def draw_scene(scene: Any, *, ax: Any = None, figsize: tuple[float, float] = (9.6, 6.0)) -> tuple[Any, Any]:
    """Draw a diagram :class:`Scene` on Matplotlib axes; ``(figure, axes)``.

    Coordinates are the scene's own (TikZ centimetres), on equal axes with no
    frame: the diagram is a picture, not a plot.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Polygon

    from ._scene import Arrow, Image, Label, Marker, Polyline

    if ax is None:
        figure = plt.figure(figsize=figsize)
        ax = figure.add_axes((0.0, 0.03, 1.0, 0.95))
    else:
        figure = ax.figure
    for item in scene.items:
        if isinstance(item, Polyline):
            xy = np.asarray(item.points, dtype=float)
            if item.style in _FILL:
                ax.add_patch(Polygon(xy, closed=True, **_FILL[item.style], zorder=0))
                continue
            if item.closed:
                xy = np.vstack([xy, xy[:1]])
            line = _LINE.get(item.style, {"color": "black", "lw": 0.5})
            ax.plot(xy[:, 0], xy[:, 1], solid_capstyle="round", **line)
            tip = _ARROW_TIPS.get(item.style)
            if tip and len(xy) >= 2 and not item.closed:
                # The head sits on the last segment, pointing along the path
                # (the toroidal direction arc: which way the island turns).
                ax.add_patch(FancyArrowPatch(
                    tuple(xy[-2]), tuple(xy[-1]), arrowstyle="->", mutation_scale=9,
                    color=line.get("color", "black"), lw=line.get("lw", 0.5), zorder=3,
                ))
        elif isinstance(item, Marker):
            x, y = item.at
            if item.kind == "x" or item.style == "xpoint":
                ax.plot([x], [y], marker="x", ms=6, mew=0.9, color=_COLOURS["islanddark"], zorder=4)
            elif item.kind == ".":
                ax.plot([x], [y], marker=".", ms=3, color="black", zorder=4)
            else:
                ax.plot([x], [y], marker="o", ms=4, color=_COLOURS["islandblue"], mec="none", zorder=4)
        elif isinstance(item, Arrow):
            style = _LINE.get(item.style, {"color": "black", "lw": 0.6})
            arrow = "<->" if (item.both or item.style in _ARROW_BOTH) else "->"
            ax.add_patch(FancyArrowPatch(
                item.start, item.end, arrowstyle=arrow, mutation_scale=8,
                color=style.get("color", "black"), lw=style.get("lw", 0.6), zorder=3,
            ))
        elif isinstance(item, Label):
            ha, va = _ANCHOR.get(item.anchor, ("center", "center"))
            ax.text(*item.at, _mathtext(item.text), ha=ha, va=va, zorder=5, **_TEXT.get(item.style, _TEXT["label"]))
        elif isinstance(item, Image):
            continue  # packaged pictures are TikZ-only; the frame shows the geometry
    # The picture's extent is its geometry *and* where its labels start: a
    # leader's text sits outside the drawing, and a frame must not crop it.
    # Labels grow away from their anchor, so the side they grow to gets room.
    points = [p for item in scene.items for p in _anchors(item)]
    if points:
        xy = np.asarray(points, dtype=float)
        (x0, y0), (x1, y1) = xy.min(axis=0), xy.max(axis=0)
        span = max(x1 - x0, y1 - y0, 1e-9)
        ax.set_xlim(x0 - 0.55 * span, x1 + 0.55 * span)
        ax.set_ylim(y0 - 0.04 * span, y1 + 0.04 * span)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    return figure, ax


def animate_states(
    name: str,
    scene_at: Callable[[Any], Any],
    *,
    driver: str,
    values: Sequence[float],
    coordinate: str,
    unit: str | None,
    fps: float | None = None,
    duration: float | None = None,
    dpi: int | None = None,
    frame_label: bool = True,
    options: Mapping[str, Any] | None = None,
) -> Any:
    """The animation result of a diagram over ``values`` of its ``driver`` coordinate.

    ``scene_at(value)`` builds the diagram's scene for one state, through the
    same model as the static call.  The result behaves as
    ``plot_*(..., animation=True)`` does: lazy, one frame per state, in order.
    """
    from vaft.plot._animation import DEFAULT_DPI, Animation, Driver, _presentation

    values = tuple(float(v) for v in values)
    if not values:
        raise ValueError(f"{name}: the {driver} trajectory is empty")
    if len(values) < 2:
        raise ValueError(f"{name}: one {driver} is a static diagram; drop animation=True")
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"{name}: the {driver} trajectory holds a non-finite value")
    presentation = {k: v for k, v in (("fps", fps), ("duration", duration)) if v is not None}
    if dpi is not None:
        presentation["dpi"] = dpi
    rate, length, timing, resolution = _presentation(presentation)
    if length is not None:
        rate = len(values) / length
    record = Driver(
        name=driver, label=driver, coordinate=coordinate, unit=unit,
        indices=tuple(range(len(values))), values=values,
    )

    def build(chosen: Mapping[str, Any]) -> Any:
        return scene_at(values[int(chosen[driver])])

    def draw(scene: Any, *, ax: Any = None, show: bool = False) -> tuple[Any, Any]:
        return draw_scene(scene, ax=ax)

    return Animation(
        plot=name, label="", driver=record, selection={"kind": driver, "value": list(values)},
        fps=float(rate), timing=timing, dpi=int(resolution or DEFAULT_DPI), image=False,
        vmin=None, vmax=None, frame_label=bool(frame_label), options=dict(options or {}),
        style={}, build=build, draw=draw,
    )
