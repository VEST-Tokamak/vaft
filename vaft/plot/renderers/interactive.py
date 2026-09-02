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

from typing import Any, Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from ..models import LineSeries, Panels
from ..navigation import SliceNavigator
from .lines import render_line_series
from .panels import render_panels

__all__ = ["BACKENDS", "render_slice_navigation"]

BACKENDS = ("matplotlib", "ipywidgets", "none")


def render_slice_navigation(
    navigator: SliceNavigator,
    histories: Sequence[LineSeries | str],
    build_slice: Callable[[int], Panels],
    *,
    backend: str = "matplotlib",
    show: bool = False,
    figsize: tuple[float, float] | None = None,
) -> tuple[Any, np.ndarray, tuple[Any, ...], Any]:
    """Draw the navigation figure; returns ``(figure, slice_axes, history_axes, widget)``.

    ``histories`` are the time-history models drawn above the slice grid, or
    a text placeholder for one that is unavailable; ``build_slice(index)``
    returns the :class:`Panels` for a slice (four panels, drawn into the
    2 x 2 grid).  The navigator's observers are wired here: a change moves the
    time marker on every history and redraws the slice panels.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {', '.join(BACKENDS)}; got {backend!r}")
    if figsize is None:
        figsize = (11.0, 9.5)
    figure = plt.figure(figsize=figsize)
    slider_rows = 1 if backend == "matplotlib" else 0
    grid = figure.add_gridspec(
        len(histories) + 2 + slider_rows, 2,
        height_ratios=[1] * len(histories) + [2.2, 2.2] + ([0.25] * slider_rows),
    )
    history_axes = tuple(figure.add_subplot(grid[row, :]) for row in range(len(histories)))
    top = len(histories)
    slice_axes = np.array(
        [[figure.add_subplot(grid[top, 0]), figure.add_subplot(grid[top, 1])],
         [figure.add_subplot(grid[top + 1, 0]), figure.add_subplot(grid[top + 1, 1])]],
        dtype=object,
    )

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

    def draw_slice(nav: SliceNavigator) -> None:
        for axis in slice_axes.ravel():
            axis.clear()
            axis.set_axis_on()
        model = build_slice(nav.selected)
        render_panels(model, ax=slice_axes, show=False)
        figure.suptitle(model.suptitle)
        for marker in markers:
            marker.set_xdata([nav.time, nav.time])
        figure.canvas.draw_idle()

    draw_slice(navigator)
    navigator.subscribe(draw_slice)

    widget = None
    if backend == "matplotlib":
        widget = _matplotlib_slider(figure, grid[top + 2, :], navigator)
    elif backend == "ipywidgets":
        widget = _ipywidgets_slider(navigator)
    if show:
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

    def follow(nav: SliceNavigator) -> None:
        if int(round(slider.val)) != nav.position:
            slider.set_val(nav.position)

    navigator.subscribe(follow)
    return slider


def _ipywidgets_slider(navigator: SliceNavigator) -> Any:
    from ipywidgets import IntSlider
    from IPython.display import display

    slider = IntSlider(
        value=navigator.position, min=0, max=len(navigator.usable) - 1, step=1,
        description="slice", continuous_update=False,
    )

    def on_change(change: dict) -> None:
        if change.get("name") == "value":
            navigator.select_position(int(change["new"]))

    slider.observe(on_change, names="value")

    def follow(nav: SliceNavigator) -> None:
        if slider.value != nav.position:
            slider.value = nav.position

    navigator.subscribe(follow)
    display(slider)
    return slider
