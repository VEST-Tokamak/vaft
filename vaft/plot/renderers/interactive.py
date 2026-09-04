"""Slice-navigation figure: histories above, a slice summary below (issue #261).

The scientific contract -- one selected slice shared by every panel -- lives
in :class:`vaft.plot.navigation.SliceNavigator`.  This renderer owns the
Matplotlib side: it lays out the history axes with a shared time marker, the
2 x 2 slice grid, and an optional slider strip, and redraws the slice panels
from whatever :class:`~vaft.plot.models.Panels` the caller builds for the
selected slice.  Adapters supply models and a builder; they never touch
pyplot.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Any, Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from ..environment import default_interaction_backend, detect_environment
from ..models import LineSeries, Panels
from ..navigation import SliceNavigator
from .lines import render_line_series
from .panels import render_panels, slice_grid_axes, visual_rows

__all__ = ["BACKENDS", "SliceAxes", "render_slice_navigation", "resolve_backend"]


class SliceAxes:
    """The slice summary's axes, rebuilt when a slice draws a different set.

    ``axes`` are the panels in slot order and ``colorbar`` the flux panel's
    colorbar cell.  A slice with fewer (or more) members than the one drawn
    before gets fresh axes on the same figure; holding this object rather than
    the array keeps a caller's reference current.
    """

    def __init__(self, axes: np.ndarray, colorbar: Any) -> None:
        self.axes = axes
        self.colorbar = colorbar

    def __len__(self) -> int:
        return int(self.axes.size)

    def __getitem__(self, item: Any) -> Any:
        return self.axes[item]

    def __iter__(self):
        return iter(self.axes)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.axes.shape

    def ravel(self) -> np.ndarray:
        return self.axes.ravel()

#: ``auto`` picks by environment (:mod:`vaft.plot.environment`);
#: ``matplotlib`` is the slider on a live canvas; ``ipywidgets`` a notebook
#: slider that redraws a static figure into its own output; ``none`` draws
#: the figure once and leaves the navigator to code.
BACKENDS = ("auto", "matplotlib", "ipywidgets", "none")


def resolve_backend(backend: str) -> str:
    """Turn the public ``backend=`` into the concrete one, validating it."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {', '.join(BACKENDS)}; got {backend!r}")
    if backend == "auto":
        return default_interaction_backend()
    if backend == "matplotlib" and not detect_environment().live_figures:
        warnings.warn(
            "backend='matplotlib' draws a slider on a canvas that does not update in "
            "place under the current Matplotlib backend; the slider will be inert. "
            "backend='auto' picks a control that works here.",
            UserWarning,
            stacklevel=3,
        )
    return backend


def render_slice_navigation(
    navigator: SliceNavigator,
    histories: Sequence[LineSeries | str],
    build_slice: Callable[[int], Panels],
    *,
    backend: str = "auto",
    show: bool = False,
    figsize: tuple[float, float] | None = None,
) -> tuple[Any, np.ndarray, tuple[Any, ...], Any]:
    """Draw the navigation figure; returns ``(figure, slice_axes, history_axes, widget)``.

    ``slice_axes`` is a :class:`SliceAxes`: its ``axes`` are the summary's
    panels and are replaced when a slice draws a different set of them.

    ``histories`` are the time-history models drawn above the slice grid, or
    a text placeholder for one that is unavailable; ``build_slice(index)``
    returns the :class:`Panels` for a slice, whose own rows, columns and
    spans shape the grid.  The navigator's observers are wired here: a change
    moves the time marker on every history and redraws the slice panels.
    ``backend`` is one of :data:`BACKENDS`; ``"auto"`` resolves through
    :func:`vaft.plot.environment.default_interaction_backend`.
    """
    backend = resolve_backend(backend)
    # The slice summary decides its own shape (rows, columns, spans); the
    # navigator only puts the histories above it and a slider strip below.
    first = build_slice(navigator.selected)
    if figsize is None:
        figsize = (4.0 * first.ncols, 2.2 * len(histories) + 2.6 * visual_rows(first))
    if backend == "ipywidgets":
        # A static figure redrawn into a widget's output area: built outside
        # pyplot so the inline backend never displays it a second time.
        from matplotlib.figure import Figure

        figure = Figure(figsize=figsize)
    else:
        figure = plt.figure(figsize=figsize)
    slider_rows = 1 if backend == "matplotlib" else 0
    # The slice summary is one cell of the outer grid; its own rows and
    # columns live in a sub-grid, so a slice that draws a different set of
    # panels can be given fresh axes without touching the histories.
    grid = figure.add_gridspec(
        len(histories) + 1 + slider_rows, 1,
        height_ratios=[1] * len(histories) + [1.6 * visual_rows(first)] + ([0.25] * slider_rows),
    )
    history_axes = tuple(figure.add_subplot(grid[row, 0]) for row in range(len(histories)))
    top = len(histories)

    def slice_axes_for(model: Panels, cell: Any) -> tuple[np.ndarray, Any]:
        # The flux panel's colorbar gets a cell of its own beside the panel,
        # so a redraw touches no layout: nothing grows or shrinks.
        return slice_grid_axes(figure, cell, model, top=0, colorbar_slot=0)

    holder = SliceAxes(*slice_axes_for(first, grid[top, 0].subgridspec(first.nrows, first.ncols)))
    shape = (len(first.models), first.spans, first.nrows, first.ncols)

    markers = []
    for axis, model in zip(history_axes, histories):
        if isinstance(model, LineSeries):
            render_line_series(model, ax=axis, show=False)
            markers.append(axis.axvline(navigator.time, color="0.3", linestyle=":", linewidth=1.2))
        else:
            axis.set_axis_off()
            axis.text(0.5, 0.5, str(model), ha="center", va="center", color="0.4")
    for axis in history_axes[:-1]:
        axis.tick_params(labelbottom=False)
        axis.set_xlabel("")

    prebuilt = {navigator.selected: first}

    def draw_slice(nav: SliceNavigator) -> None:
        nonlocal shape
        model = prebuilt.pop(nav.selected, None) or build_slice(nav.selected)
        rebuilt = None
        if (len(model.models), model.spans, model.nrows, model.ncols) != shape:
            # This slice draws a different set of panels: fresh axes on a
            # gridspec of their own over the cell's extent, so the histories
            # above never move and only these axes are laid out again.
            from matplotlib.gridspec import GridSpec

            box = grid[top, 0].get_position(figure)
            # The colorbar goes first: tearing it down consults the axes its
            # mappable was drawn on, which must still exist (Matplotlib 3.11).
            for axis in (holder.colorbar, *holder.axes.ravel()):
                axis.remove()
            rebuilt = GridSpec(
                model.nrows, model.ncols, figure=figure,
                left=box.x0, right=box.x1, bottom=box.y0, top=box.y1,
            )
            holder.axes, holder.colorbar = slice_axes_for(model, rebuilt)
            shape = (len(model.models), model.spans, model.nrows, model.ncols)
        else:
            # A redraw leaves the figure as it found it: the same axes, cleared
            # and drawn again, with the flux panel's colorbar in its fixed cell.
            for axis in (*holder.axes.ravel(), holder.colorbar):
                axis.clear()
                axis.set_axis_on()
        styles = [dict(style) for style in (model.member_styles or ({},) * len(model.models))]
        if styles:
            styles[0]["colorbar_ax"] = holder.colorbar
        model = replace(model, member_styles=tuple(styles))
        render_panels(model, ax=holder.axes, show=False)
        if rebuilt is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rebuilt.tight_layout(figure, rect=(box.x0, box.y0, box.x1, box.y1))
        figure.suptitle(model.suptitle)
        for marker in markers:
            marker.set_xdata([nav.time, nav.time])
        figure.canvas.draw_idle()

    widget = None
    if backend == "matplotlib":
        widget = _matplotlib_slider(figure, grid[top + 1, 0], navigator)
    draw_slice(navigator)
    # Laid out once, after the first draw: later redraws reuse these axes.
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    navigator.subscribe(draw_slice)
    if backend == "matplotlib":
        _follow_slider(widget, navigator)
    elif backend == "ipywidgets":
        widget = _ipywidgets_slider(figure, navigator, live=detect_environment().live_figures)
    if show and backend != "ipywidgets":
        plt.show()
    return figure, holder, history_axes, widget


def _matplotlib_slider(figure: Any, spec: Any, navigator: SliceNavigator) -> Any:
    from matplotlib.widgets import Slider

    axis = figure.add_subplot(spec)
    count = len(navigator.usable)
    slider = Slider(axis, "slice", 0, max(count - 1, 0), valinit=navigator.position, valstep=1, valfmt="%d")

    def on_slider(value: float) -> None:
        # Snap the cursor's position to a real stored slice (section 16).
        navigator.select_position(int(round(value)))

    slider.on_changed(on_slider)
    return slider


def _follow_slider(slider: Any, navigator: SliceNavigator) -> None:
    def follow(nav: SliceNavigator) -> None:
        if int(round(slider.val)) != nav.position:
            slider.set_val(nav.position)

    navigator.subscribe(follow)


def _ipywidgets_slider(figure: Any, navigator: SliceNavigator, *, live: bool) -> Any:
    """An ipywidgets slider over the navigator, with the figure beneath it.

    Under a static figure backend (inline) the figure is a picture: every
    change redraws it into the widget's own output area, which is what makes
    the control work in Jupyter Notebook, JupyterLab and VS Code alike.  Under
    a live canvas (``ipympl``) the canvas updates itself and only the slider
    is shown.
    """
    import io

    from ipywidgets import IntSlider, Output, VBox
    from IPython.display import Image, clear_output, display

    slider = IntSlider(
        value=navigator.position, min=0, max=len(navigator.usable) - 1, step=1,
        description="slice", continuous_update=False,
    )
    output = Output()

    def refresh(_: SliceNavigator | None = None) -> None:
        # Rendered to PNG here rather than handed to the front end's figure
        # formatter, which is registered only for pyplot-managed figures.
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=figure.dpi)
        with output:
            clear_output(wait=True)
            display(Image(data=buffer.getvalue()))

    def on_change(change: dict) -> None:
        if change.get("name") == "value":
            navigator.select_position(int(change["new"]))

    slider.observe(on_change, names="value")

    def follow(nav: SliceNavigator) -> None:
        if slider.value != nav.position:
            slider.value = nav.position

    navigator.subscribe(follow)
    if live:
        display(slider)
    else:
        navigator.subscribe(refresh)
        refresh()
        display(VBox([slider, output]))
    slider.vaft_output = output
    return slider
