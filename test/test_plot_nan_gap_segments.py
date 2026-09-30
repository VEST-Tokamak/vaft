"""A polyline broken by NaN is drawn as separate runs, whatever rewrites line data (#1314).

The camera overlays mark a field-line sample behind the camera or outside the
lens model as NaN.  A ``Line2D`` breaks at NaN only while the NaN is still in
its data; an IPython startup script that installs an ``xlim_changed`` hook
cropping every line to the view (GPEC's ``pypec.modplot`` does this) drops the
NaN and joins the runs with straight segments that are no part of the line.
The renderer draws a gapped polyline as explicit segments instead.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection

from vaft.plot.models import GeometryLayer, GeometryLayers, Image2D
from vaft.plot.renderers.geometry import finite_runs, render_geometry_layers
from vaft.plot.renderers.images import render_image_2d


def _gapped_line() -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 4.0 * np.pi, 121)
    r = 50.0 + 40.0 * np.cos(t)
    z = 50.0 + 40.0 * np.sin(t)
    for start, stop in ((20, 31), (60, 62), (90, 100)):
        r[start:stop] = np.nan
        z[start:stop] = np.nan
    z[45] = np.nan  # one coordinate alone also breaks the line
    return r, z


def _allowed_pairs(r: np.ndarray, z: np.ndarray) -> set[tuple[float, float, float, float]]:
    """Every pair of consecutive samples that are both finite: the only segments that exist."""
    finite = np.isfinite(r) & np.isfinite(z)
    return {
        (r[i], z[i], r[i + 1], z[i + 1])
        for i in range(r.size - 1)
        if finite[i] and finite[i + 1]
    }


def _drawn_pairs(ax) -> list[tuple[float, float, float, float]]:
    """Every straight segment the axes will stroke, from lines and line collections."""
    pairs: list[tuple[float, float, float, float]] = []
    polylines: list[np.ndarray] = []
    for line in ax.lines:
        if line.get_linestyle() in ("None", "none", "", " "):
            continue
        xy = np.asarray(line.get_xydata(), dtype=float)
        polylines.extend(finite_runs(xy[:, 0], xy[:, 1]) or [xy])
    for collection in ax.collections:
        if isinstance(collection, LineCollection):
            polylines.extend(np.asarray(segment, dtype=float) for segment in collection.get_segments())
    for xy in polylines:
        for a, b in zip(xy[:-1], xy[1:]):
            pairs.append((a[0], a[1], b[0], b[1]))
    return pairs


def _crop_lines_on_xlim_change(ax) -> None:
    """What pypec.modplot's downsampler does: crop each line's data to the view."""

    def crop(axes):
        lo, hi = sorted(axes.get_xlim())
        for line in axes.lines:
            x = np.asarray(line.get_xdata(), dtype=float)
            y = np.asarray(line.get_ydata(), dtype=float)
            window = (x >= lo) & (x <= hi)  # NaN compares False: the gap is gone
            line.set_data(x[window], y[window])

    ax.callbacks.connect("xlim_changed", crop)


def test_finite_runs_splits_at_every_non_finite_sample():
    r, z = _gapped_line()
    runs = finite_runs(r, z)
    assert runs is not None and len(runs) == 5
    assert all(np.isfinite(run).all() and len(run) >= 2 for run in runs)
    assert finite_runs(np.arange(3.0), np.arange(3.0)) is None
    # a lone finite sample between gaps is no segment
    assert finite_runs(np.array([np.nan, 1.0, np.nan]), np.array([0.0, 1.0, 0.0])) == []


@pytest.mark.parametrize("hooked", [False, True], ids=["plain", "xlim-hook"])
def test_camera_overlay_never_draws_a_segment_across_a_nan_gap(hooked):
    r, z = _gapped_line()
    layer = GeometryLayer(r=r, z=z, kind="polyline", label="Field line", style={"color": "red"})
    fig, ax = plt.subplots()
    if hooked:
        _crop_lines_on_xlim_change(ax)
    render_image_2d(Image2D(values=np.zeros((100, 100)), overlays=(layer,)), ax=ax, show=False)
    ax.set_xlim(0.0, 60.0)  # a view change after drawing, as a notebook's layout pass makes
    fig.canvas.draw()

    drawn = _drawn_pairs(ax)
    allowed = _allowed_pairs(r, z)
    assert drawn, "the overlay drew nothing"
    assert set(drawn) <= allowed, "a drawn segment bridges a NaN gap"
    assert len(set(drawn)) == len(allowed)  # and every real segment is still there
    handles, labels = ax.get_legend_handles_labels()
    assert "Field line" in labels
    plt.close(fig)


def test_a_gapped_polyline_keeps_its_style_and_autoscales():
    r, z = _gapped_line()
    layer = GeometryLayer(r=r, z=z, kind="polyline", style={"color": "red", "linewidth": 2.5, "linestyle": "--"})
    fig, ax = render_geometry_layers(GeometryLayers((layer,)), show=False)
    (collection,) = [c for c in ax.collections if isinstance(c, LineCollection)]
    assert matplotlib.colors.same_color(collection.get_color()[0], "red")
    assert collection.get_linewidth()[0] == pytest.approx(2.5)
    assert collection.get_linestyle()[0][1] is not None  # dashed
    lo, hi = ax.get_xlim()
    assert lo <= np.nanmin(r) and hi >= np.nanmax(r)
    plt.close(fig)


def test_a_finite_polyline_is_still_one_line():
    layer = GeometryLayer(r=np.arange(5.0), z=np.arange(5.0) ** 2, kind="polyline", label="LCFS")
    fig, ax = render_geometry_layers(GeometryLayers((layer,)), show=False)
    assert [line.get_label() for line in ax.lines] == ["LCFS"]
    assert not [c for c in ax.collections if isinstance(c, LineCollection)]
    plt.close(fig)
