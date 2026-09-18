"""Graphical presentation of a canonical figure: geometry, format and theme (issue #689).

Three independent choices decide how a scientific view occupies a canvas,
none of them a scientific one (the unit, scale, notation, validity and
uncertainty of a quantity are the display policy's, issue #256; which series
go on which axes is the semantic layout's, issue #260):

``geometry policy``
    how a *kind* of plot occupies a canvas -- a time trace is a landscape
    strip, a profile near-square, an R-Z view takes its height from the
    machine it draws with ``1 unit of R = 1 unit of Z``, a composite divides
    one fixed width among its panels.  Owned by the view kind, never a public
    option.

``format``
    how large the final rendering is: the physical width, a height ceiling,
    the base font and the scales of everything measured in points.  Presets
    ``screen``, ``single_column`` and ``double_column`` generalise the
    recurring physical constraints of scientific journals; no publisher is
    named.

``theme``
    which visual grammar is used: font family, tick direction, grid and
    spines, the colour cycle and, for a monochrome figure, the linestyle and
    marker cycles that carry the distinction colour would.  Accessibility is
    a baseline of every theme, not a theme of its own: both colour cycles
    are colour-blind safe.

``format=None`` means :data:`DEFAULT_FORMAT` -- ``screen`` -- for a figure
the renderer creates on its own (issue #712, the presentation contract's
last phase); the default yields to a caller's ``ax=`` and to an explicit
``figsize=``, which decide the canvas themselves.  ``format="legacy"``
names the sizes the renderers had before this module existed (they were
never one width), so a workflow whose reference images predate it can pin
them.  ``theme=None`` stays Matplotlib's own look.

Nothing here mutates global Matplotlib state.  A :class:`Presentation` is
applied as a :func:`matplotlib.rc_context` around one whole render (axes
creation, lines, labels, legends, ticks), and ``rcParams`` are what they
were once the figure is returned.  A caller who owns the axes keeps the
canvas: ``format=`` with ``ax=`` is refused rather than ignored, and a
``theme=`` is applied to the caller's axes explicitly.
"""

from __future__ import annotations

import contextlib
import functools
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np

from .intent import (  # noqa: F401  re-exported: the colour vocabulary lives beside the themes
    DEFAULT_COLOURS,
    DEFAULT_PALETTE,
    active_theme,
    palette,
    resolve_color,
    resolve_style,
    themed,
)

__all__ = [
    "DEFAULT_FORMAT",
    "EQUILIBRIUM_ROLE",
    "LEGACY_FORMAT",
    "FORMATS",
    "FigureFormat",
    "GEOMETRY",
    "GeometryPolicy",
    "Presentation",
    "THEMES",
    "Theme",
    "apply_axes_theme",
    "presented",
    "resolve_color",
    "resolve_presentation",
    "resolve_style",
    "rz_extent",
]


# ---------------------------------------------------------------------------
# formats: how large
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FigureFormat:
    """The physical scale of one rendering: width, height ceiling, type sizes."""

    name: str
    width_in: float
    max_height_in: float
    base_font_pt: float
    label_scale: float = 1.0
    tick_scale: float = 0.9
    title_scale: float = 1.0
    legend_scale: float = 0.9
    line_scale: float = 1.0
    marker_scale: float = 1.0
    panel_gap_pt: float = 6.0
    outer_pad_pt: float = 3.0


#: What ``format=None`` means for a figure the renderer creates itself.
DEFAULT_FORMAT = "screen"

#: The spelling of "the renderers' own sizes from before the presentation
#: contract": no format resolution at all, the path a reference-image
#: workflow pins when it must not move.
LEGACY_FORMAT = "legacy"

#: The recurring physical constraints of scientific figures, without naming
#: a publisher: a screen figure, a single column (86 mm) and a double column
#: (178 mm), the latter two at the 8 pt final-size type most journals ask for.
FORMATS: Mapping[str, FigureFormat] = {
    "screen": FigureFormat(
        "screen", width_in=6.5, max_height_in=9.0, base_font_pt=10.0,
        label_scale=1.0, tick_scale=0.9, title_scale=1.1, legend_scale=0.9,
        line_scale=1.0, marker_scale=1.0, panel_gap_pt=8.0, outer_pad_pt=4.0,
    ),
    "single_column": FigureFormat(
        "single_column", width_in=3.375, max_height_in=9.0, base_font_pt=8.0,
        label_scale=1.0, tick_scale=0.9, title_scale=1.0, legend_scale=0.85,
        line_scale=0.75, marker_scale=0.75, panel_gap_pt=4.0, outer_pad_pt=2.0,
    ),
    "double_column": FigureFormat(
        "double_column", width_in=7.0, max_height_in=9.0, base_font_pt=8.0,
        label_scale=1.0, tick_scale=0.9, title_scale=1.0, legend_scale=0.85,
        line_scale=0.85, marker_scale=0.85, panel_gap_pt=5.0, outer_pad_pt=2.0,
    ),
}


# ---------------------------------------------------------------------------
# themes: which visual grammar
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Theme:
    """The visual grammar of a figure: type face, ticks, frame, cycles."""

    name: str
    font_family: tuple[str, ...]
    tick_direction: str
    grid: bool
    grid_alpha: float
    #: The spines left visible.
    spines: tuple[str, ...]
    colors: tuple[str, ...]
    #: Cycled beside the colours when a theme cannot rely on colour alone;
    #: ``None`` leaves Matplotlib's solid line and no marker.
    linestyles: tuple[str, ...] | None = None
    markers: tuple[str, ...] | None = None
    #: Fraction of the samples that carry a marker, so a dense waveform does
    #: not become a smear of symbols.
    markevery: float | None = None
    #: Baselines in points; a format scales them.
    line_pt: float = 1.2
    marker_pt: float = 4.0
    #: The distinguishing colours ``palette:<n>`` tokens take; ``None`` uses
    #: ``colors`` (issue #709).
    palette: tuple[str, ...] | None = None
    #: Colour intent tokens this theme overrides (``role:``, ``feature:``,
    #: ``state:``, ``emphasis:``): a colour, or a style patch when colour
    #: alone cannot carry the distinction.  Anything not named keeps the
    #: default of :data:`vaft.plot.intent.DEFAULT_COLOURS`.
    intents: Mapping[str, Any] = field(default_factory=dict, hash=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "intents", MappingProxyType(dict(self.intents)))

    def prop_cycle(self):
        """The ``axes.prop_cycle`` this theme sets, every sub-cycle equal in length."""
        from cycler import cycler

        cycle = cycler(color=list(self.colors))
        if self.linestyles is not None:
            cycle += cycler(linestyle=list(self.linestyles))
        if self.markers is not None:
            cycle += cycler(marker=list(self.markers))
            if self.markevery is not None:
                cycle += cycler(markevery=[self.markevery] * len(self.markers))
        return cycle


#: Okabe and Ito's eight colours, distinguishable under the common forms of
#: colour-vision deficiency; the first is black, as a technical figure's
#: principal trace usually is.
_OKABE_ITO = ("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

#: Paul Tol's "bright" scheme, likewise colour-blind safe, lighter in tone.
_TOL_BRIGHT = ("#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB")

#: Two greys alternating with six linestyle/marker pairs: every series is
#: told apart by dash and symbol, and the greys stay darker than the 0.65
#: an invalid channel is drawn in (vaft.plot.style.INVALID_COLOR).
_MONO_GREYS = ("0.0", "0.4", "0.0", "0.4", "0.0", "0.4")
_MONO_LINESTYLES = ("-", "--", "-.", ":", "-", "--")
_MONO_MARKERS = ("", "", "o", "s", "^", "D")

#: What each theme makes of the colour intents that are not palette slots.
#: Greys are left as they are: every theme's emphasis ladder is the default.
_TECHNICAL_INTENTS = {
    "role:measured": "#000000", "role:reconstructed": "#D55E00", "role:reference": "#000000",
    "feature:boundary": "#D55E00", "feature:coil": "#0072B2", "feature:limiter": "#000000",
    "feature:axis": "#000000",
    "state:enabled": "#000000", "state:disabled": "#E69F00", "state:missing": "#CC79A7",
    "emphasis:alert": "#D55E00",
}
_MINIMAL_INTENTS = {
    "role:reconstructed": "#EE6677", "role:reference": "0.2",
    "feature:boundary": "#EE6677", "feature:coil": "#4477AA", "feature:wall": "0.5",
    "feature:limiter": "0.2", "feature:axis": "0.2", "feature:passive": "0.7",
    "state:disabled": "#CCBB44", "state:missing": "#EE6677",
    "emphasis:alert": "#EE6677",
}
#: Where colour cannot carry a distinction, a patch adds a dash or a marker;
#: a patch never names markerfacecolor, so a hollow marker stays hollow.
_MONOCHROME_INTENTS = {
    "role:measured": "0.0",
    "role:reconstructed": {"color": "0.45", "marker": "^"},
    "role:reference": "0.0",
    "feature:boundary": {"color": "0.0", "linestyle": "--"},
    "feature:wall": "0.5", "feature:coil": "0.3", "feature:limiter": "0.0",
    "feature:axis": "0.0", "feature:passive": "0.6",
    "state:enabled": "0.0", "state:disabled": "0.45", "state:missing": "0.0",
    "emphasis:alert": "0.0",
}

THEMES: Mapping[str, Theme] = {
    "technical": Theme(
        "technical", font_family=("DejaVu Sans",), tick_direction="in",
        grid=True, grid_alpha=0.3, spines=("left", "right", "top", "bottom"),
        colors=_OKABE_ITO, line_pt=1.2, marker_pt=4.0, intents=_TECHNICAL_INTENTS,
    ),
    "minimal": Theme(
        "minimal", font_family=("Helvetica", "Arial", "DejaVu Sans"), tick_direction="out",
        grid=False, grid_alpha=0.0, spines=("left", "bottom"),
        colors=_TOL_BRIGHT, line_pt=1.5, marker_pt=4.0, intents=_MINIMAL_INTENTS,
    ),
    "monochrome": Theme(
        "monochrome", font_family=("DejaVu Sans",), tick_direction="in",
        grid=True, grid_alpha=0.2, spines=("left", "right", "top", "bottom"),
        colors=_MONO_GREYS, linestyles=_MONO_LINESTYLES, markers=_MONO_MARKERS,
        markevery=0.1, line_pt=1.2, marker_pt=4.0, intents=_MONOCHROME_INTENTS,
    ),
}


# ---------------------------------------------------------------------------
# geometry policy: how a kind of plot occupies a canvas
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GeometryPolicy:
    """``kind`` is ``aspect`` (height = aspect x width), ``extent`` (an R-Z view
    takes the ratio of what it draws), ``native`` (an image keeps its pixel
    ratio) or ``grid`` (a composite divides one width among its panels)."""

    kind: str
    aspect: float | None = None


#: Keyed by view-model class name so that this module imports no model.
GEOMETRY: Mapping[str, GeometryPolicy] = {
    "LineSeries": GeometryPolicy("aspect", 0.42),
    "Profile1D": GeometryPolicy("aspect", 0.70),
    "PowerSpectrum": GeometryPolicy("aspect", 0.62),
    "Spectrogram": GeometryPolicy("aspect", 0.45),
    "TextPanel": GeometryPolicy("aspect", 0.70),
    "Geometry3DLayers": GeometryPolicy("aspect", 1.0),
    "Field2D": GeometryPolicy("extent"),
    "GeometryLayers": GeometryPolicy("extent"),
    "Image2D": GeometryPolicy("native"),
    "ImageSequence": GeometryPolicy("native"),
    "Panels": GeometryPolicy("grid"),
}

#: An R-Z view with nothing but a plasma boundary to go on keeps the
#: portrait ratio the R-Z renderers always used, so a boundary alone never
#: decides the canvas (it moves between shots and times).
_RZ_FALLBACK_ASPECT = 7.0 / 6.0

#: The share of a canvas's height the axes gets once the title and the
#: x label have taken theirs.
_RZ_AXES_HEIGHT_FRACTION = 0.9

#: The share of the canvas width an R-Z axes actually gets once labels, and
#: for a field map its colorbar, have taken theirs.
_RZ_AXES_FRACTION = 0.85
_RZ_AXES_FRACTION_WITH_COLORBAR = 0.7

#: The height/width ratio of one cell of a composite grid, and the least a
#: row may be given, so a wide grid does not collapse its rows.
_PANEL_CELL_ASPECT = 0.4
_PANEL_MIN_ROW_IN = 0.9

#: The role a geometry layer carries when it belongs to the plasma rather
#: than the machine (``GeometryLayer.role``); such layers never size a canvas.
EQUILIBRIUM_ROLE = "equilibrium"


def rz_extent(model: Any) -> tuple[float, float] | None:
    """``(delta R, delta Z)`` of what an R-Z view displays, or ``None``.

    Priority follows issue #689 section 7: the machine geometry drawn over
    the view (wall, coils, sensors -- every layer that is not the plasma's
    and not a text annotation), then the field's own grid, and nothing
    otherwise -- a boundary alone returns ``None`` so the fallback ratio is
    used, because a canvas sized to the LCFS would change with every shot.
    """
    layers = tuple(getattr(model, "overlays", ()) or ()) or tuple(getattr(model, "layers", ()) or ())
    machine = [
        layer for layer in layers
        if getattr(layer, "kind", "") != "text"
        and getattr(layer, "role", "") != EQUILIBRIUM_ROLE
        and np.asarray(layer.r).size and np.asarray(layer.z).size
    ]
    if machine:
        r = np.concatenate([np.asarray(layer.r, dtype=float).ravel() for layer in machine])
        z = np.concatenate([np.asarray(layer.z, dtype=float).ravel() for layer in machine])
        return _span(r, z)
    r_axis, z_axis = getattr(model, "r", None), getattr(model, "z", None)
    if r_axis is not None and z_axis is not None and getattr(model, "values", None) is not None:
        return _span(np.asarray(r_axis, dtype=float), np.asarray(z_axis, dtype=float))
    return None


def _span(r: np.ndarray, z: np.ndarray) -> tuple[float, float] | None:
    r, z = r[np.isfinite(r)], z[np.isfinite(z)]
    if r.size < 2 or z.size < 2:
        return None
    dr, dz = float(r.max() - r.min()), float(z.max() - z.min())
    if dr <= 0.0 or dz <= 0.0:
        return None
    return dr, dz


# ---------------------------------------------------------------------------
# the resolved presentation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Presentation:
    """A format and/or a theme, resolved; what a renderer applies once."""

    format: FigureFormat | None
    theme: Theme | None

    @property
    def pad(self) -> float | None:
        """``tight_layout`` padding in font units, when a format sets one."""
        if self.format is None:
            return None
        return self.format.outer_pad_pt / self.format.base_font_pt

    def figsize(
        self, model: Any, fallback: tuple[float, float] | None, *, colorbar: bool | None = None,
    ) -> tuple[float, float] | None:
        """The canvas for ``model`` under this format; ``fallback`` when there is none.

        A theme alone never changes geometry.  Width is the format's; height
        follows the view kind's geometry policy and is held below the
        format's ceiling.  A view whose coordinates must keep their ratio --
        an R-Z machine, an image -- is the exception both ways: the format's
        width and ceiling are then maxima.  An image narrows with the
        ceiling here; an R-Z view keeps the width and is only started here:
        what its labels, title and colorbar really take is known once it is
        laid out, so :func:`presented` sizes and places it afterwards
        (:func:`_fit_canvas`).
        ``colorbar`` says whether a field map draws one, which takes part of
        the width the axes would have had.
        """
        if self.format is None:
            return fallback
        width = self.format.width_in
        ceiling = self.format.max_height_in
        policy = GEOMETRY.get(type(model).__name__, GeometryPolicy("aspect", 0.62))
        if policy.kind == "aspect":
            return (width, min(width * float(policy.aspect or 0.62), ceiling))
        if policy.kind == "extent":
            span = rz_extent(model)
            ratio = span[1] / span[0] if span else _RZ_FALLBACK_ASPECT
            # The axes gets only part of the width -- the colorbar of a field
            # map takes its strip, labels take their margin -- and equal
            # coordinate scaling then fixes the axes height from that width.
            draws_colorbar = _has_colorbar(model) if colorbar is None else bool(colorbar)
            usable = _RZ_AXES_FRACTION_WITH_COLORBAR if draws_colorbar else _RZ_AXES_FRACTION
            return _snug(width, ratio, usable, ceiling, narrow=False)
        if policy.kind == "native":
            return _snug(width, _native_ratio(model), _RZ_AXES_FRACTION, ceiling)
        # grid: one width whatever the column count; the rows add height,
        # each as tall as the members standing in it ask (issue #711).
        return (width, min(max(sum(self.row_heights(model)), 0.5 * width), ceiling))

    def row_heights(self, model: Any) -> list[float]:
        """The heights, in inches, a composite's rows need under this format.

        What :meth:`figsize` sums for a ``Panels`` model; a renderer takes
        them as ``row_heights`` so the grid divides the canvas the way the
        members asked rather than equally (issue #711).  Empty without a
        format.
        """
        if self.format is None:
            return []
        return _grid_rows(model, self.format.width_in, self.format.base_font_pt)

    def rc(self) -> dict[str, Any]:
        """The rcParams this presentation sets, theme baseline times format scale."""
        rc: dict[str, Any] = {}
        fmt, theme = self.format, self.theme
        if fmt is not None:
            base = fmt.base_font_pt
            rc.update({
                "font.size": base,
                "axes.labelsize": base * fmt.label_scale,
                "axes.titlesize": base * fmt.title_scale,
                "xtick.labelsize": base * fmt.tick_scale,
                "ytick.labelsize": base * fmt.tick_scale,
                "legend.fontsize": base * fmt.legend_scale,
                "figure.titlesize": base * fmt.title_scale,
                # Panel gaps relative to Matplotlib's own 0.2 at the screen
                # format's 8 pt, so a tighter format closes the gaps with it.
                "figure.subplot.wspace": 0.2 * fmt.panel_gap_pt / 8.0,
                "figure.subplot.hspace": 0.2 * fmt.panel_gap_pt / 8.0,
            })
        line_scale = fmt.line_scale if fmt is not None else 1.0
        marker_scale = fmt.marker_scale if fmt is not None else 1.0
        if theme is not None:
            rc.update({
                "font.family": list(theme.font_family),
                "xtick.direction": theme.tick_direction,
                "ytick.direction": theme.tick_direction,
                # The grid stays the renderer's decision (a contour map draws
                # none); a renderer that draws one takes the theme's alpha.
                "grid.alpha": theme.grid_alpha,
                "axes.spines.left": "left" in theme.spines,
                "axes.spines.right": "right" in theme.spines,
                "axes.spines.top": "top" in theme.spines,
                "axes.spines.bottom": "bottom" in theme.spines,
                "axes.prop_cycle": theme.prop_cycle(),
                "lines.linewidth": theme.line_pt * line_scale,
                "lines.markersize": theme.marker_pt * marker_scale,
            })
        elif fmt is not None:
            import matplotlib

            rc["lines.linewidth"] = float(matplotlib.rcParams["lines.linewidth"]) * line_scale
            rc["lines.markersize"] = float(matplotlib.rcParams["lines.markersize"]) * marker_scale
        return rc

    def context(self) -> contextlib.AbstractContextManager:
        """A context that applies :meth:`rc` for one render and restores after.

        It also makes the theme the one colour intents resolve against
        (:func:`vaft.plot.intent.themed`), for exactly as long.
        """
        import matplotlib

        rc = self.rc()
        if not rc and self.theme is None:
            return contextlib.nullcontext()
        stack = contextlib.ExitStack()
        if rc:
            stack.enter_context(matplotlib.rc_context(rc=rc))
        stack.enter_context(themed(self.theme))
        return stack


def _snug(
    width: float, ratio: float, usable: float, ceiling: float, *, narrow: bool = True,
) -> tuple[float, float]:
    """A canvas that fits axes of ``ratio`` (height/width), estimated.

    The axes takes ``usable`` of the width and ``_RZ_AXES_HEIGHT_FRACTION``
    of the height; when the height that implies exceeds the ceiling, the
    width comes down with it so the axes still fills the canvas.  A canvas
    is never narrower than a third of its height, so a very tall machine
    keeps room for its labels.

    ``narrow=False`` keeps the format's width when the ceiling binds: an
    R-Z view is then fitted from its laid-out figure (:func:`_fit_canvas`),
    because what labels, a title and a colorbar take is a cost in inches
    that a fixed share cannot estimate -- narrowing on the estimate left a
    tall machine's axes filling half its canvas beside a full-height
    colorbar.
    """
    height = width * usable * ratio / _RZ_AXES_HEIGHT_FRACTION
    if height > ceiling:
        height = ceiling
        if narrow:
            width = max(height * _RZ_AXES_HEIGHT_FRACTION / ratio / usable, height / 3.0)
    return (width, max(height, 0.3 * width))


#: A fit that moves no side by more than this (inches) is converged.
_FIT_TOLERANCE_IN = 0.01

#: The narrowest a fitted R-Z axes may be drawn; below it the fit is skipped.
_FIT_MIN_AXES_IN = 0.5


def _fit_canvas(
    figure: Any, axes: Any, pad: float | None, *, max_width: float, ceiling: float,
    passes: int = 3,
) -> None:
    """Size the canvas to an equal-scaled axes and place it, and its colorbar, snugly.

    ``tight_layout`` cannot do this: an equal-scaled axes is drawn smaller
    than the slot the layout gives it, the layout measures the labels
    against the drawn box, and the margin it leaves is then wrong on the
    side the axes did not fill (labels off the canvas, or a colorbar
    standing taller than the map).  So what surrounds the drawn axes is
    measured side by side -- its own tick labels, axis labels and title,
    and each axes standing to its right (a colorbar: its offset, strip and
    tick labels) -- the axes gets the largest size for which its own ratio
    plus that surround fits inside ``max_width`` x ``ceiling``, and every
    axes is placed explicitly on the canvas that implies, the map's slot
    equal to its drawn box.  A second pass re-measures, since tick labels
    can change with the axes size.  Anything else -- no equal-scaled axes,
    several, or another axes that is not beside it on the right -- is left
    as the layout drew it.
    """
    fixed = [axis for axis in _axes_of(axes) if axis.get_aspect() != "auto"]
    if len(fixed) != 1:
        return
    axis = fixed[0]
    pad_in = (pad or 0.0) * float(_rc_font_pt()) / 72.0
    for _ in range(passes):
        renderer = _renderer_of(figure)
        if renderer is None:
            return
        axis.apply_aspect()
        dpi = figure.dpi
        width, height = figure.get_size_inches()
        drawn = axis.get_window_extent(renderer)
        slot = axis.get_position(original=True)
        slot_x1 = slot.x1 * width
        slot_y0, slot_h = slot.y0 * height, slot.height * height
        tight = axis.get_tightbbox(renderer)
        if any(artist.get_visible() for artist in (*figure.texts, *figure.legends)):
            return  # a figure-level title or legend: not measured here
        x0, y0, x1, y1 = (drawn.x0 / dpi, drawn.y0 / dpi, drawn.x1 / dpi, drawn.y1 / dpi)
        if x1 - x0 <= 0.0 or y1 - y0 <= 0.0:
            return
        left = x0 - tight.x0 / dpi
        bottom = y0 - tight.y0 / dpi
        top = tight.y1 / dpi - y1
        right = tight.x1 / dpi - x1
        beside = []
        for other in figure.axes:
            if other is axis or not other.get_visible():
                continue
            box = other.get_position()
            ox0, ox1 = box.x0 * width, box.x1 * width
            if ox0 < slot_x1 - _FIT_TOLERANCE_IN:
                return  # not beside the map on the right: leave the layout alone
            oy0, oy1 = box.y0 * height, box.y1 * height
            otight = other.get_tightbbox(renderer)
            offset = ox0 - slot_x1
            # Relative to the slot the layout gave the map, which the fitted
            # map fills: a colorbar spanning the slot then spans the map.
            frac_y0 = (oy0 - slot_y0) / slot_h
            frac_h = (oy1 - oy0) / slot_h
            beside.append((other, offset, ox1 - ox0, frac_y0, frac_h))
            right = max(right, offset + (ox1 - ox0) + max(otight.x1 / dpi - ox1, 0.0))
            top = max(top, otight.y1 / dpi - oy1)
            bottom = max(bottom, oy0 - otight.y0 / dpi)
        left, bottom, top, right = (max(v, 0.0) + pad_in for v in (left, bottom, top, right))
        ratio = (y1 - y0) / (x1 - x0)
        axes_w = min(max_width - left - right, (ceiling - bottom - top) / ratio)
        if axes_w < _FIT_MIN_AXES_IN:
            return
        axes_h = axes_w * ratio
        fitted = (left + axes_w + right, bottom + axes_h + top)
        moved = max(abs(fitted[0] - width), abs(fitted[1] - height),
                    abs(axes_w - (x1 - x0)), abs(axes_h - (y1 - y0)))
        figure.set_size_inches(*fitted, forward=False)
        fw, fh = fitted
        axis.set_position([left / fw, bottom / fh, axes_w / fw, axes_h / fh])
        for other, offset, strip, frac_y0, frac_h in beside:
            other.set_position([
                (left + axes_w + offset) / fw, (bottom + frac_y0 * axes_h) / fh,
                strip / fw, frac_h * axes_h / fh,
            ])
        if moved < _FIT_TOLERANCE_IN:
            return


def _rc_font_pt() -> float:
    import matplotlib
    from matplotlib.font_manager import FontProperties

    return FontProperties(size=matplotlib.rcParams["font.size"]).get_size_in_points()


def _renderer_of(figure: Any) -> Any:
    get = getattr(figure.canvas, "get_renderer", None)
    if get is not None:
        try:
            return get()
        except Exception:  # pragma: no cover - a canvas without an Agg renderer
            pass
    private = getattr(figure, "_get_renderer", None)
    return private() if private is not None else None


def _has_colorbar(model: Any) -> bool:
    return bool(getattr(model, "values", None) is not None and getattr(model, "colorbar", True))


def _native_ratio(model: Any) -> float:
    values = getattr(model, "values", None)
    if values is None:
        frames = getattr(model, "frames", None) or ()
        values = frames[0] if len(frames) else None
    if values is None:
        return 1.0
    shape = np.asarray(values).shape
    if len(shape) < 2 or shape[1] == 0:
        return 1.0
    return float(shape[0]) / float(shape[1])


#: The tallest a member's cell gets relative to its width inside a grid: a
#: composite's width is the format's and cannot follow one member's ratio,
#: so a very tall machine keeps equal scaling inside its own panel instead.
_CELL_MAX_ASPECT = 2.5


def _grid_rows(model: Any, width: float, base_font_pt: float = 10.0) -> list[float]:
    """The height a composite's rows need under one fixed ``width``.

    Every member asks for the height its own geometry policy would give it
    on the width of the columns it spans -- a time trace its landscape
    strip, a field map its R-Z ratio less the colorbar's share, an image
    its pixel ratio -- and a row is as tall as the tallest request across
    it.  The rows start at the plain cell height the grid always used, so
    a grid of time traces is exactly as tall as before; only a member that
    needs more raises its rows, proportionally when it spans several.
    Whole-figure only: the slice navigator sizes its own canvas.
    """
    members = tuple(getattr(model, "models", ()) or ())
    ncols = max(1, int(getattr(model, "ncols", 1) or 1))
    nrows = max(1, int(getattr(model, "nrows", 1) or 1))
    spans = tuple(getattr(model, "spans", None) or ())
    if not spans:
        spans = tuple((i // ncols, i % ncols, 1, 1) for i in range(len(members)))
    column_width = width / ncols
    # A row is never shorter than an axes with its title and x label set at
    # the format's type: the plain floor plus three lines.
    floor = _PANEL_MIN_ROW_IN + 3.0 * _TEXT_LINE_PER_PT * base_font_pt
    cell = max(floor, _PANEL_CELL_ASPECT * column_width)
    # The plain grid's height, spread over the structural rows: a spans grid
    # counts rows as the deepest stack, not the LCM it is built on.
    rows = [_visual_rows(model) * cell / nrows] * nrows
    for member, (row, _col, rowspan, colspan) in zip(members, spans):
        need = _cell_need(member, colspan * column_width, base_font_pt)
        have = sum(rows[row:row + rowspan])
        if need is not None and need > have > 0.0:
            scale = need / have
            rows[row:row + rowspan] = [height * scale for height in rows[row:row + rowspan]]
    return [float(height) for height in rows]


#: The width an axes' y label and tick labels take, per point of base font
#: (0.6 in at 10 pt): what a cell loses before its map can start.
_LABEL_ALLOWANCE_PER_PT = 0.06

#: A text panel is set at Matplotlib's "small" -- 0.833 of the base font --
#: with 1.4 line spacing; this turns the base font into a line's inches.
_TEXT_LINE_PER_PT = 0.833 * 1.4 / 72.0


def _cell_need(member: Any, cell_width: float, base_font_pt: float = 10.0) -> float | None:
    """The height in inches a member asks of its cell, or ``None`` to accept it.

    Only a member whose shape is not its own to give asks: an R-Z map or an
    image must keep its coordinate ratio, a text block must fit its lines.
    A trace, a profile or a spectrum takes whatever height the grid gives
    its row, so a grid of them is exactly as tall as it always was.
    """
    policy = GEOMETRY.get(type(member).__name__, GeometryPolicy("aspect", 0.62))
    if policy.kind == "extent":
        span = rz_extent(member)
        ratio = span[1] / span[0] if span else _RZ_FALLBACK_ASPECT
        usable = _RZ_AXES_FRACTION_WITH_COLORBAR if _has_colorbar(member) else _RZ_AXES_FRACTION
        # Inside a grid the cell also pays for its own axis labels and ticks,
        # a cost in inches that grows with the type size, not with the cell;
        # the map is only as wide as what is left, and equal scaling then
        # fixes its height from that width.  Asking for more would give the
        # row height the map cannot use.
        drawn_width = max(cell_width * usable - _LABEL_ALLOWANCE_PER_PT * base_font_pt, 0.3 * cell_width)
        return min(ratio, _CELL_MAX_ASPECT) / _RZ_AXES_HEIGHT_FRACTION * drawn_width
    if policy.kind == "native":
        return min(_native_ratio(member), _CELL_MAX_ASPECT) * cell_width
    lines = getattr(member, "lines", None)
    if lines is not None and type(member).__name__ == "TextPanel":
        return (len(lines) + 2.0) * _TEXT_LINE_PER_PT * base_font_pt
    return None


def _visual_rows(model: Any) -> int:
    spans = getattr(model, "spans", None)
    if spans:
        from .renderers.panels import visual_rows

        return max(1, int(visual_rows(model)))
    return max(1, int(getattr(model, "nrows", 1) or 1))


def resolve_presentation(
    format: str | None,
    theme: str | None,
    *,
    ax: Any = None,
    figsize: tuple[float, float] | None = None,
) -> Presentation | None:
    """The presentation a renderer applies, or ``None`` for the legacy path.

    ``format=`` sets the canvas, so it is refused beside the two things that
    already decide one: a caller's ``ax=`` and an explicit ``figsize=``.
    Refusing rather than ignoring is the contract (issue #689 section 11);
    a ``theme=`` alone is fine with either, it changes artists, not canvases.
    """
    # A control spells "no theme" as the ``"none"`` sentinel every choice
    # control uses, and ``as_style`` must keep passing that word along since
    # it is also a real uncertainty mode -- so it is read as absence here.
    format = None if format in (None, "", "none", LEGACY_FORMAT) else format
    theme = None if theme in (None, "", "none") else theme
    if format is None and theme is None:
        return None
    fmt = None
    if format is not None:
        try:
            fmt = FORMATS[str(format)]
        except KeyError:
            raise ValueError(
                f"format must be one of {', '.join(FORMATS)} or {LEGACY_FORMAT!r}; got {format!r}"
            ) from None
        if ax is not None:
            raise TypeError(
                "format= sets the canvas; the caller's ax= already owns it -- "
                "pass theme= only, or drop ax="
            )
        if figsize is not None:
            raise ValueError(
                "figsize= and format= both set the canvas; pass one of them"
            )
    resolved_theme = None
    if theme is not None:
        try:
            resolved_theme = THEMES[str(theme)]
        except KeyError:
            raise ValueError(
                f"theme must be one of {', '.join(THEMES)}; got {theme!r}"
            ) from None
    return Presentation(format=fmt, theme=resolved_theme)


def apply_axes_theme(axes: Any, theme: Theme) -> None:
    """Apply a theme's axes-level tokens to axes that already exist.

    rcParams reach only axes created inside the presentation context; a
    caller-supplied ``ax=`` predates it, so the cycle, ticks, spines and
    grid are set on it directly.  The caller's figure and layout are not
    touched.
    """
    axes.set_prop_cycle(theme.prop_cycle())
    axes.tick_params(direction=theme.tick_direction)
    for name, spine in axes.spines.items():
        spine.set_visible(name in theme.spines)
    _set_grid(axes, theme)


def presented(default_figsize: tuple[float, float] | None = None) -> Callable:
    """Give a base renderer ``format=`` and ``theme=``, applied around it.

    The renderer keeps its body.  ``format=None`` on a canvas nobody else
    decides (no ``ax=``, no ``figsize=``) means :data:`DEFAULT_FORMAT`; a
    caller's ``ax=``, an explicit ``figsize=`` or ``format="legacy"`` take
    the untouched path, where the wrapper calls the renderer exactly as it
    was called before the contract existed.  With a preset the
    wrapper resolves it (refusing ``ax=`` or ``figsize=`` beside ``format=``),
    opens the presentation context for the whole render -- axes creation,
    artists, legend, ticks, the renderer's own ``finalize`` -- computes the
    canvas from the view kind's geometry policy in place of the renderer's
    default, applies the theme to a caller's pre-existing axes before
    anything is drawn, and afterwards re-styles a grid the renderer drew
    with its own constant and lays the figure out with the format's
    padding.  The renderer's signature must declare ``format`` and ``theme``
    (they are what the option schema reads), and may declare ``figsize``.
    """

    def decorate(render: Callable[..., Any]) -> Callable[..., Any]:
        import inspect

        takes_show = "show" in inspect.signature(render).parameters

        @functools.wraps(render)
        def wrapper(model: Any, *args: Any, ax: Any = None, figsize: Any = None,
                    format: str | None = None, theme: str | None = None, **kwargs: Any) -> Any:
            if format in (None, "", "none") and ax is None and figsize is None:
                # The canonical default, for a canvas nobody else decides;
                # the empty spellings a control or the CLI may pass mean it too.
                format = DEFAULT_FORMAT
            presentation = resolve_presentation(format, theme, ax=ax, figsize=figsize)
            if presentation is None:
                return render(model, *args, ax=ax, figsize=figsize, **kwargs)
            with presentation.context():
                size = presentation.figsize(
                    model, figsize or default_figsize, colorbar=kwargs.get("colorbar"),
                )
                if presentation.format is not None and type(model).__name__ == "Panels":
                    # The rows the format sized, for the grid to honour.
                    kwargs.setdefault("row_heights", presentation.row_heights(model))
                if ax is not None and presentation.theme is not None:
                    for axis in _axes_of(ax):
                        apply_axes_theme(axis, presentation.theme)
                # The figure is shown once it is finished: the renderer's own
                # finalize would call plt.show() -- which blocks -- before the
                # theme's grid and the format's padding below are applied, so
                # what the user saw was not what was returned or saved (cold
                # review plot G11).  The show is taken over and done last.
                show = bool(kwargs.pop("show", False)) if takes_show else False
                if takes_show:
                    kwargs["show"] = False
                result = render(model, *args, ax=ax, figsize=size, **kwargs)
                figure, axes = result[0], result[1]
                if presentation.theme is not None:
                    for axis in _axes_of(axes):
                        _restyle_grid(axis, presentation.theme)
                if ax is None and presentation.pad is not None:
                    from .style import finalize

                    finalize(figure, axes, show=False, tight_layout=True, pad=presentation.pad)
                    # Only an R-Z view the machine or the field grid sizes is
                    # fitted: a boundary alone keeps the fallback ratio, so the
                    # canvas does not follow the plasma from shot to shot, and
                    # an image keeps its estimated canvas (an animation is
                    # saved by the renderer, before anything here runs).
                    policy = GEOMETRY.get(type(model).__name__)
                    if policy is not None and policy.kind == "extent" and rz_extent(model) is not None:
                        _fit_canvas(
                            figure, axes, presentation.pad,
                            max_width=presentation.format.width_in,
                            ceiling=presentation.format.max_height_in,
                        )
                # An animation written to save_path= is saved instead of shown.
                if show and kwargs.get("save_path") is None:
                    import matplotlib.pyplot as plt

                    plt.show()
            return result

        return wrapper

    return decorate


def _axes_of(axes: Any) -> list[Any]:
    """The Matplotlib axes in a renderer's ``axes`` return, flattened."""
    from matplotlib.axes import Axes

    if isinstance(axes, Axes):
        return [axes]
    try:
        return [item for item in np.asarray(axes, dtype=object).ravel() if isinstance(item, Axes)]
    except Exception:  # pragma: no cover - a non-array return
        return []


def _restyle_grid(axes: Any, theme: Theme) -> None:
    """A grid the renderer drew with its own alpha takes the theme's; none is added."""
    drawn = any(line.get_visible() for line in axes.get_xgridlines()) or any(
        line.get_visible() for line in axes.get_ygridlines()
    )
    if drawn:
        _set_grid(axes, theme)


def _set_grid(axes: Any, theme: Theme) -> None:
    # ``grid(False, alpha=...)`` would switch the grid ON (Matplotlib warns
    # and honours the line property), so the flag and the alpha are separate.
    if theme.grid:
        axes.grid(True, alpha=theme.grid_alpha)
    else:
        axes.grid(False)
