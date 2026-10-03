"""Reproducible figure-level overrides on top of a format and a theme (issue #1421).

#689 settles how a canonical plot looks: the *format* says how large
everything is, the *theme* which graphical language it speaks.
:class:`FigureOptions` holds what a reader explicitly changes on top of
that -- a title, axis limits or scales, legend placement, ticks and grid,
type sizes and faces, the colour map and its range.  Every field is ``None``
until the reader sets it, and ``None`` means *inherit*: the format, theme or
plot keeps deciding.  So the options store intent, never a resolved default,
and :meth:`FigureOptions.to_dict` writes only what was set -- a figure
reproduced later picks up any improvement to the canonical defaults.

The boundary of #1421: a setting reused for another shot of the same plot
belongs here; a one-off nudge of one artist belongs to a finishing tool
(Pylustrator, #1202).  Nothing here edits arbitrary ``rcParams``.

How the options are applied:

* type sizes, faces and line/marker scales are rcParams, so they act while
  the figure is drawn: :func:`figure_options_scope` makes them visible to
  :func:`vaft.plot.presentation.presented`, which layers them over the
  format and theme of the plot it draws;
* everything else edits the finished figure: :meth:`FigureOptions.apply`
  for Matplotlib, :meth:`FigureOptions.apply_plotly` for Plotly.

The plot entry points take them as ``figure_options=`` (an instance or its
dict form), e.g. ``vaft.omas.plot_plasma_current_time(ods,
figure_options={"xlim": (0.30, 0.33), "legend": False})``.
"""

from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass, fields
from typing import Any, Iterator, Mapping

__all__ = [
    "AXIS_SCALES",
    "LEGEND_LOCATIONS",
    "FigureOptions",
    "as_figure_options",
    "figure_options_scope",
]

AXIS_SCALES = ("linear", "log", "symlog")
LEGEND_LOCATIONS = (
    "best", "upper right", "upper left", "lower left", "lower right", "right",
    "center left", "center right", "lower center", "upper center", "center",
)
TICK_DIRECTIONS = ("in", "out", "inout")
#: MathText font sets Matplotlib ships; no proprietary font is bundled.
MATH_FONTSETS = ("dejavusans", "dejavuserif", "cm", "stix", "stixsans", "custom")

#: The colorbar axes Matplotlib adds carry this label; axis edits are for data.
_COLORBAR = "<colorbar>"


def _pair(name: str, value: Any) -> tuple[float | None, float | None] | None:
    if value is None:
        return None
    try:
        low, high = value
    except (TypeError, ValueError):
        raise ValueError(f"{name} is (low, high), either may be None; got {value!r}") from None
    low = None if low is None else float(low)
    high = None if high is None else float(high)
    if low is not None and high is not None and low >= high:
        raise ValueError(f"{name}: low must be below high; got {value!r}")
    return (low, high)


def _choice(name: str, value: Any, allowed: tuple[str, ...]) -> Any:
    if value is not None and value not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(allowed)}; got {value!r}")
    return value


def _positive(name: str, value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive; got {value}")
    return value


@dataclass(frozen=True)
class FigureOptions:
    """Explicit figure-level overrides; every ``None`` inherits.

    Axes: ``title`` (``""`` hides it), ``xlabel``/``ylabel`` (replace a
    label the plot drew), ``xlim``/``ylim`` (``(low, high)``, either end
    ``None``), ``xscale``/``yscale`` (:data:`AXIS_SCALES`).  Legend:
    ``legend`` (show or hide), ``legend_loc`` (:data:`LEGEND_LOCATIONS`),
    ``legend_ncols``, ``legend_frame``.  Ticks and grid: ``grid``,
    ``minor_ticks``, ``tick_direction``.  Type: ``font_size`` (the base, in
    points; the format's label/tick/title/legend scales follow it unless the
    specific size is given), ``label_size``, ``tick_size``, ``title_size``,
    ``legend_size``, ``font_family``, ``math_fontset``.  Series:
    ``line_scale``, ``marker_scale``.  Scalar fields: ``cmap``, ``clim``,
    ``colorbar_label``.
    """

    title: str | None = None
    xlabel: str | None = None
    ylabel: str | None = None
    xlim: tuple[float | None, float | None] | None = None
    ylim: tuple[float | None, float | None] | None = None
    xscale: str | None = None
    yscale: str | None = None
    legend: bool | None = None
    legend_loc: str | None = None
    legend_ncols: int | None = None
    legend_frame: bool | None = None
    grid: bool | None = None
    minor_ticks: bool | None = None
    tick_direction: str | None = None
    font_size: float | None = None
    label_size: float | None = None
    tick_size: float | None = None
    title_size: float | None = None
    legend_size: float | None = None
    font_family: tuple[str, ...] | None = None
    math_fontset: str | None = None
    line_scale: float | None = None
    marker_scale: float | None = None
    cmap: str | None = None
    clim: tuple[float | None, float | None] | None = None
    colorbar_label: str | None = None

    def __post_init__(self) -> None:
        set_ = lambda name, value: object.__setattr__(self, name, value)  # noqa: E731
        for name in ("xlim", "ylim", "clim"):
            set_(name, _pair(name, getattr(self, name)))
        for name in ("xscale", "yscale"):
            _choice(name, getattr(self, name), AXIS_SCALES)
        _choice("legend_loc", self.legend_loc, LEGEND_LOCATIONS)
        _choice("tick_direction", self.tick_direction, TICK_DIRECTIONS)
        _choice("math_fontset", self.math_fontset, MATH_FONTSETS)
        if self.legend_ncols is not None:
            set_("legend_ncols", int(self.legend_ncols))
            if self.legend_ncols < 1:
                raise ValueError(f"legend_ncols must be at least 1; got {self.legend_ncols}")
        for name in ("font_size", "label_size", "tick_size", "title_size", "legend_size", "line_scale", "marker_scale"):
            set_(name, _positive(name, getattr(self, name)))
        if self.font_family is not None:
            family = (self.font_family,) if isinstance(self.font_family, str) else tuple(self.font_family)
            if not family or not all(isinstance(face, str) and face for face in family):
                raise ValueError(f"font_family names one or more faces; got {self.font_family!r}")
            set_("font_family", family)
        for name in ("legend", "legend_frame", "grid", "minor_ticks"):
            value = getattr(self, name)
            if value is not None:
                set_(name, bool(value))
        for scale, limits in (("xscale", "xlim"), ("yscale", "ylim")):
            if getattr(self, scale) == "log" and any(
                end is not None and end <= 0 for end in (getattr(self, limits) or ())
            ):
                raise ValueError(f"a log {scale[0]} axis needs positive {limits}; got {getattr(self, limits)!r}")

    # -- intent ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """The options that were set, as plain JSON data; inherited ones are left out."""
        data: dict[str, Any] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if value is not None:
                data[item.name] = list(value) if isinstance(value, tuple) else value
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FigureOptions":
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"FigureOptions takes no {', '.join(unknown)}; known: {', '.join(sorted(known))}")
        return cls(**dict(data))

    def __bool__(self) -> bool:
        return bool(self.to_dict())

    # -- while drawing -----------------------------------------------------------
    def rc(self, base: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """The rcParams these options set, over ``base`` (the rcParams in force).

        A base ``font_size`` rescales the other sizes by the ratio to the size
        in force, so a format's label/tick/title/legend proportions survive.
        """
        import matplotlib

        current = dict(base) if base is not None else matplotlib.rcParams
        rc: dict[str, Any] = {}
        if self.font_size is not None:
            ratio = self.font_size / float(matplotlib.font_manager.FontProperties(size=current["font.size"]).get_size_in_points())
            rc["font.size"] = self.font_size
            for key in ("axes.labelsize", "axes.titlesize", "xtick.labelsize", "ytick.labelsize",
                        "legend.fontsize", "figure.titlesize"):
                size = current[key]
                points = (
                    float(size) if isinstance(size, (int, float))
                    else matplotlib.font_manager.FontProperties(size=size).get_size_in_points()
                )
                rc[key] = points * ratio
        for key, value in (
            ("axes.labelsize", self.label_size), ("xtick.labelsize", self.tick_size),
            ("ytick.labelsize", self.tick_size), ("axes.titlesize", self.title_size),
            ("figure.titlesize", self.title_size), ("legend.fontsize", self.legend_size),
        ):
            if value is not None:
                rc[key] = value
        if self.font_family is not None:
            rc["font.family"] = list(self.font_family)
        if self.math_fontset is not None:
            rc["mathtext.fontset"] = self.math_fontset
        if self.tick_direction is not None:
            rc["xtick.direction"] = rc["ytick.direction"] = self.tick_direction
        if self.line_scale is not None:
            rc["lines.linewidth"] = float(current["lines.linewidth"]) * self.line_scale
        if self.marker_scale is not None:
            rc["lines.markersize"] = float(current["lines.markersize"]) * self.marker_scale
        return rc

    @contextlib.contextmanager
    def context(self) -> Iterator[None]:
        """These options' rcParams in force for the block (nothing global changes)."""
        import matplotlib

        rc = self.rc()
        if not rc:
            yield
            return
        with matplotlib.rc_context(rc):
            yield

    # -- the finished figure -----------------------------------------------------
    def apply(self, figure: Any, axes: Any = None) -> Any:
        """Edit a finished Matplotlib ``figure`` in place; returns it.

        ``axes`` limits the edits to the axes a plot drew -- a caller's own
        figure keeps its other axes as they were; ``None`` edits every data
        axes of a figure the plot made.  Axis settings apply to each of them,
        so on a multi-panel figure a ``ylim`` is every panel's ylim (per-cell
        settings belong to a composition's cells).
        """
        owned = axes is None
        candidates = figure.axes if owned else _flat_axes(axes)
        data_axes = [axis for axis in candidates if axis.get_label() != _COLORBAR and axis.get_visible()]
        # Text replaced after drawing keeps the size it was drawn at: the
        # format's rcParams are no longer in force here.
        if self.title is not None:
            current = figure._suptitle
            if len(data_axes) == 1 and (current is None or not owned):
                axis = data_axes[0]
                axis.set_title(self.title, fontsize=axis.title.get_fontsize())
            elif owned:
                if current is not None:
                    _retitle_in_place(figure, current, self.title)
                else:
                    size = data_axes[0].title.get_fontsize() if data_axes else None
                    figure.suptitle(self.title, fontsize=size)
        for axis in data_axes:
            self._edit_axes(axis)
        if self.colorbar_label is not None:
            for colorbar in _colorbars(data_axes):
                colorbar.set_label(self.colorbar_label, fontsize=colorbar.ax.yaxis.label.get_fontsize())
        return figure

    def _edit_axes(self, axis: Any) -> None:
        # A label is replaced where the plot drew one: a stacked panel that
        # leaves its time label to the panel below keeps it hidden.
        if self.xlabel is not None and axis.get_xlabel():
            axis.set_xlabel(self.xlabel, fontsize=axis.xaxis.label.get_fontsize())
        if self.ylabel is not None and axis.get_ylabel():
            axis.set_ylabel(self.ylabel, fontsize=axis.yaxis.label.get_fontsize())
        if self.xscale is not None:
            axis.set_xscale(self.xscale)
        if self.yscale is not None:
            axis.set_yscale(self.yscale)
        if self.xlim is not None:
            axis.set_xlim(left=self.xlim[0], right=self.xlim[1])
        if self.ylim is not None:
            axis.set_ylim(bottom=self.ylim[0], top=self.ylim[1])
        if self.grid is not None:
            axis.grid(self.grid)
        if self.minor_ticks is True:
            axis.minorticks_on()
        elif self.minor_ticks is False:
            axis.minorticks_off()
        if self.tick_direction is not None:
            axis.tick_params(direction=self.tick_direction, which="both")
        self._edit_legend(axis)
        if self.cmap is not None or self.clim is not None:
            # Only the scalar field a colorbar explains: overlays drawn in a
            # fixed colour (a grey flux contour set) keep their own map.
            for mappable in _scalar_fields(axis):
                if self.cmap is not None:
                    mappable.set_cmap(self.cmap)
                if self.clim is not None:
                    mappable.set_clim(*self.clim)

    def _edit_legend(self, axis: Any) -> None:
        legend = axis.get_legend()
        if self.legend is False:
            if legend is not None:
                legend.remove()
            return
        placed = self.legend_loc is not None or self.legend_ncols is not None or self.legend_frame is not None
        if legend is None and self.legend is not True:
            return
        if legend is not None and not placed and self.legend is not True:
            return
        if legend is not None:
            # Rebuilt from what the legend shows, proxy handles included.
            handles = list(legend.legend_handles)
            labels = [text.get_text() for text in legend.get_texts()]
        else:
            handles, labels = axis.get_legend_handles_labels()
        if not handles:
            return
        keywords: dict[str, Any] = {}
        if legend is not None and legend.get_title().get_text():
            keywords["title"] = legend.get_title().get_text()
        if self.legend_loc is not None:
            keywords["loc"] = self.legend_loc
        elif legend is not None:
            keywords["loc"] = legend._loc
        if self.legend_ncols is not None:
            keywords["ncols"] = self.legend_ncols
        if self.legend_frame is not None:
            keywords["frameon"] = self.legend_frame
        if legend is not None:
            texts = legend.get_texts()
            if texts:
                keywords.setdefault("fontsize", texts[0].get_fontsize())
            legend.remove()
        axis.legend(handles, labels, **keywords)

    def apply_plotly(self, figure: Any) -> Any:
        """Edit a finished Plotly ``figure`` in place; returns it."""
        import math

        layout: dict[str, Any] = {}
        if self.title is not None:
            layout["title"] = {"text": self.title}
        font: dict[str, Any] = {}
        if self.font_size is not None:
            font["size"] = self.font_size
        if self.font_family is not None:
            font["family"] = ", ".join(self.font_family)
        if font:
            layout["font"] = font
        if self.legend is not None:
            layout["showlegend"] = self.legend
        if layout:
            figure.update_layout(**layout)
        ignored = [name for name in ("legend_loc", "legend_ncols", "legend_frame", "label_size", "tick_size",
                                     "title_size", "legend_size", "math_fontset", "marker_scale")
                   if getattr(self, name) is not None]
        if "symlog" in (self.xscale, self.yscale):
            ignored.append("symlog scale")
        if ignored:
            import warnings

            warnings.warn(
                f"backend='plotly' does not apply {', '.join(ignored)}; they take effect with Matplotlib",
                UserWarning, stacklevel=3,
            )
        axis_names = list(figure.layout.to_plotly_json())
        for axis_name, label, limits, scale in (
            ("x", self.xlabel, self.xlim, self.xscale), ("y", self.ylabel, self.ylim, self.yscale),
        ):
            update = figure.update_xaxes if axis_name == "x" else figure.update_yaxes
            changes: dict[str, Any] = {}
            if scale is not None:
                changes["type"] = {"symlog": "linear"}.get(scale, scale)
            if limits is not None:
                # A log axis takes its range in decades, whether the options
                # or the plot made it log; set per axis.
                for name in [key for key in axis_names if key.startswith(f"{axis_name}axis")]:
                    log = (scale or figure.layout[name].type) == "log"
                    figure.layout[name].range = [None if v is None else (math.log10(v) if log else v) for v in limits]
                if not any(key.startswith(f"{axis_name}axis") for key in axis_names):
                    log = scale == "log"
                    changes["range"] = [None if v is None else (math.log10(v) if log else v) for v in limits]
            if self.grid is not None:
                changes["showgrid"] = self.grid
            if self.tick_direction is not None:
                changes["ticks"] = {"in": "inside", "out": "outside", "inout": "outside"}[self.tick_direction]
            if self.minor_ticks is not None:
                changes["minor"] = {"ticks": "inside" if self.tick_direction == "in" else "outside"} if self.minor_ticks else {"ticks": ""}
            if changes:
                update(**changes)
            if label is not None:
                # Only the axes that carry a title keep one, as in Matplotlib.
                for name in [key for key in figure.layout.to_plotly_json() if key.startswith(f"{axis_name}axis")]:
                    if figure.layout[name].title.text:
                        figure.layout[name].title.text = label
        if self.cmap is not None or self.clim is not None or self.colorbar_label is not None:
            from vaft.plot.plotly.fields import colorscale

            for trace in figure.data:
                if not hasattr(trace, "colorscale") or trace.colorscale is None or getattr(trace, "showscale", None) is False:
                    continue
                if self.cmap is not None:
                    scale, reverse = colorscale(self.cmap)
                    trace.colorscale = scale
                    trace.reversescale = reverse
                if self.clim is not None:
                    if hasattr(trace, "zmin"):
                        trace.zmin, trace.zmax = self.clim
                    elif hasattr(trace, "cmin"):
                        trace.cmin, trace.cmax = self.clim
                if self.colorbar_label is not None and hasattr(trace, "colorbar"):
                    trace.colorbar.title = {"text": self.colorbar_label}
        if self.line_scale is not None:
            for trace in figure.data:
                line = getattr(trace, "line", None)
                if line is not None and hasattr(line, "width"):
                    # Plotly's default trace width is 2 px.
                    line.width = (line.width if line.width is not None else 2.0) * self.line_scale
        return figure


def _retitle_in_place(figure: Any, suptitle: Any, text: str) -> None:
    """Replace the suptitle's text where the renderer hung it, keeping it off the panels.

    The suptitle hangs from the top edge in the band the layout reserved for
    the old text; a taller replacement (an extra line) would grow down onto
    the first row's titles, so the subplots are lowered by the difference.
    """
    renderer = figure.canvas.get_renderer()
    before = suptitle.get_window_extent(renderer).height
    suptitle.set_text(text)
    grown = (suptitle.get_window_extent(renderer).height - before) / figure.bbox.height
    if grown > 0 and suptitle.get_verticalalignment() == "top":
        figure.subplots_adjust(top=max(figure.subplotpars.bottom + 0.05, figure.subplotpars.top - grown))


def _flat_axes(axes: Any) -> list[Any]:
    from matplotlib.axes import Axes

    if isinstance(axes, Axes):
        return [axes]
    import numpy as np

    return [axis for axis in np.asarray(axes, dtype=object).ravel() if isinstance(axis, Axes)]


def _scalar_fields(axis: Any) -> list[Any]:
    """The mappables on ``axis`` a colorbar explains."""
    return [
        mappable for mappable in [*axis.collections, *axis.images]
        if getattr(mappable, "colorbar", None) is not None
    ]


def _colorbars(axes: Any) -> list[Any]:
    seen: list[Any] = []
    for axis in axes:
        for mappable in _scalar_fields(axis):
            if mappable.colorbar not in seen:
                seen.append(mappable.colorbar)
    return seen


def as_figure_options(value: Any) -> FigureOptions:
    """``value`` as :class:`FigureOptions`: one already, its dict form, or ``None`` (nothing set)."""
    if value is None:
        return FigureOptions()
    if isinstance(value, FigureOptions):
        return value
    if isinstance(value, Mapping):
        return FigureOptions.from_dict(value)
    raise TypeError(f"figure_options is a vaft.plot.FigureOptions or its dict form; got {type(value).__name__}")


#: The options of the figure being drawn, read by vaft.plot.presentation.presented.
_ACTIVE: contextvars.ContextVar[FigureOptions | None] = contextvars.ContextVar("vaft_figure_options", default=None)


@contextlib.contextmanager
def figure_options_scope(options: FigureOptions) -> Iterator[None]:
    """Make ``options`` the ones :func:`~vaft.plot.presentation.presented` layers
    over the format and theme of every renderer called inside the block."""
    token = _ACTIVE.set(options if options else None)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def active_figure_options() -> FigureOptions | None:
    """The options :func:`figure_options_scope` set, if any."""
    return _ACTIVE.get()


@contextlib.contextmanager
def figure_options_rc() -> Iterator[None]:
    """The active options' rcParams for one render, entered once.

    A renderer that draws others (the panels renderer and its members) would
    otherwise scale line widths twice; inside the block the options are no
    longer active, so a nested render inherits them instead of reapplying.
    """
    options = _ACTIVE.get()
    if options is None:
        yield
        return
    token = _ACTIVE.set(None)
    try:
        with options.context():
            yield
    finally:
        _ACTIVE.reset(token)
