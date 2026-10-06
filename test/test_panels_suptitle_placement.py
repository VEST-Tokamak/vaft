"""A panel grid's suptitle stays on the canvas and above the first row's titles.

``render_panels`` used to re-place the suptitle with ``va="bottom"`` just
under the top edge, so the text grew *above* the figure: ``fig.savefig``
clipped it (notebooks and thumbnails hid it with ``bbox_inches="tight"``).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from vaft.plot.models import LineSeries, Panels, Series
from vaft.plot.renderers.panels import render_panels

SUPTITLES = {
    "one line": "Voltage consumption #39915",
    "two lines": "Voltage consumption #39915\nresistive and inductive parts",
}


def _panels(suptitle: str, count: int, ncols: int) -> Panels:
    x = np.linspace(0.0, 1.0, 20)
    members = tuple(
        LineSeries(series=(Series(x=x, y=x * (k + 1), label=f"s{k}"),), y_label="y", title=f"Panel {k}")
        for k in range(count)
    )
    return Panels(models=members, nrows=-(-count // ncols), ncols=ncols, suptitle=suptitle)


def _drawn_tick_labels(axis):
    """Tick labels inside the view limits: the ones outside are kept but never drawn."""
    labels = []
    for ticks, tick_labels, limits in (
        (axis.get_xticks(), axis.get_xticklabels(), axis.get_xlim()),
        (axis.get_yticks(), axis.get_yticklabels(), axis.get_ylim()),
    ):
        low, high = sorted(limits)
        labels += [label for tick, label in zip(ticks, tick_labels) if low <= tick <= high]
    return labels


def _assert_clear_of_panels(extent, drawn, renderer) -> None:
    """Above every panel's titles, and on top of none of its text.

    The layout bbox keeps the titles but not a y label taller than its short
    axes, which overflows at the far left, beside the suptitle; the overlap
    check covers that case in two dimensions.
    """
    assert extent.y0 >= max(axis.get_tightbbox(renderer, for_layout_only=True).y1 for axis in drawn) - 0.5
    for axis in drawn:
        texts = [axis.title, axis._left_title, axis._right_title, axis.xaxis.label, axis.yaxis.label]
        texts += _drawn_tick_labels(axis)
        for text in texts:
            if text.get_visible() and text.get_text():
                assert not extent.overlaps(text.get_window_extent(renderer)), text.get_text()


@pytest.mark.parametrize("suptitle", list(SUPTITLES.values()), ids=list(SUPTITLES))
@pytest.mark.parametrize(
    "count, ncols, figsize",
    [(3, 1, (6.5, 5.2)), (12, 3, (9.0, 14.0)), (2, 2, (8.0, 3.0))],
    ids=["stack", "tall-grid", "short-row"],
)
@pytest.mark.parametrize("format", [None, "screen", "single_column", "double_column", "slide", "poster"])
def test_the_suptitle_lies_inside_the_figure_and_above_the_panels(suptitle, count, ncols, figsize, format):
    kwargs = {"format": format} if format is not None else {"figsize": figsize}
    figure, axes = render_panels(_panels(suptitle, count, ncols), **kwargs)
    try:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        extent = figure._suptitle.get_window_extent(renderer)
        canvas = figure.bbox
        # On the canvas: nothing a plain savefig would clip.
        assert extent.y1 <= canvas.y1 + 0.5, (extent, canvas)
        assert extent.y0 >= canvas.y0 and extent.x0 >= canvas.x0 - 0.5 and extent.x1 <= canvas.x1 + 0.5
        # Above the topmost panel's own title, never on it.
        drawn = [axis for axis in np.asarray(axes).ravel() if axis.get_visible()]
        _assert_clear_of_panels(extent, drawn, renderer)
    finally:
        plt.close(figure)


def test_a_suptitle_the_caller_positioned_is_left_alone():
    from vaft.plot.style import finalize

    figure, axis = plt.subplots(figsize=(4.0, 3.0))
    figure.suptitle("placed", y=0.5)
    finalize(figure, axis)
    assert figure._suptitle.get_position()[1] == 0.5
    plt.close(figure)


@pytest.mark.parametrize("format", [None, "screen"])
def test_a_taller_title_set_after_layout_still_clears_the_panels(format):
    """FigureOptions replaces the suptitle's text after the layout: a second
    line must push the panels down, not land on their titles."""
    from vaft.plot.figure_options import FigureOptions

    kwargs = {"format": format} if format is not None else {"figsize": (8.0, 5.0)}
    figure, axes = render_panels(_panels("One line", 4, 2), **kwargs)
    try:
        FigureOptions(title="Two\nlines here").apply(figure)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        extent = figure._suptitle.get_window_extent(renderer)
        drawn = [axis for axis in np.asarray(axes).ravel() if axis.get_visible()]
        assert extent.y1 <= figure.bbox.y1 + 0.5
        _assert_clear_of_panels(extent, drawn, renderer)
    finally:
        plt.close(figure)
