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
        assert extent.y0 >= max(axis.get_tightbbox(renderer).y1 for axis in drawn) - 0.5
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
        assert extent.y0 >= max(axis.get_tightbbox(renderer).y1 for axis in drawn) - 0.5
    finally:
        plt.close(figure)
