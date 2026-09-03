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
from .panels import render_panels, slice_grid_axes

__all__ = ["BACKENDS", "render_slice_navigation", "resolve_backend"]

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
        figsize = (4.0 * first.ncols, 2.2 * len(histories) + 2.6 * first.nrows)
    if backend == "ipywidgets":
        # A static figure redrawn into a widget's output area: built outside
        # pyplot so the inline backend never displays it a second time.
        from matplotlib.figure import Figure

        figure = Figure(figsize=figsize)
    else:
        figure = plt.figure(figsize=figsize)
    slider_rows = 1 if backend == "matplotlib" else 0
    grid = figure.add_gridspec(
        len(histories) + first.nrows + slider_rows, first.ncols,
        height_ratios=[1] * len(histories) + [1.6] * first.nrows + ([0.25] * slider_rows),
    )
    history_axes = tuple(figure.add_subplot(grid[row, :]) for row in range(len(histories)))
    top = len(histories)
    # The flux panel's colorbar gets a cell of its own beside the panel, made
    # once: a redraw then touches no layout, so nothing grows or shrinks.
    slice_axes, colorbar_axes = slice_grid_axes(figure, grid, first, top=top, colorbar_slot=0)

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
        # A redraw leaves the figure as it found it: the same axes, cleared and
        # drawn again, with the flux panel's colorbar in its own fixed cell.
        for axis in (*slice_axes.ravel(), colorbar_axes):
            axis.clear()
            axis.set_axis_on()
        model = prebuilt.pop(nav.selected, None) or build_slice(nav.selected)
        styles = [dict(style) for style in (model.member_styles or ({},) * len(model.models))]
        if styles:
            styles[0]["colorbar_ax"] = colorbar_axes
        model = replace(model, member_styles=tuple(styles))
        render_panels(model, ax=slice_axes, show=False)
        figure.suptitle(model.suptitle)
        for marker in markers:
            marker.set_xdata([nav.time, nav.time])
        figure.canvas.draw_idle()

    widget = None
    if backend == "matplotlib":
        widget = _matplotlib_slider(figure, grid[top + first.nrows, :], navigator)
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
    return figure, slice_axes, history_axes, widget


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
