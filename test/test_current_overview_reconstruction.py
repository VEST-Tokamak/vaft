"""current_overview_reconstruction: I_p measured and per slice, PF coil currents, eddy currents.

The packaged 39915 carries no eddy solution and no convergence evidence, so the
input is the sample with its eddy currents solved (the same input the recipe
read-recording uses) and, where a test needs verdicts, a run flag per slice --
the one piece of evidence ``verify_convergence`` grades on its own.
"""

from __future__ import annotations

import contextlib
import io
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import vaft
import vaft.omas
import vaft.plot
from vaft.omas.entries import normalize_entries
from vaft.plot.backend.recipes import build_model
from vaft.plot.models import LineSeries, Panels

from _sample_fixtures import sample_ods
from _synthetic_inputs import make_eddy_solved

NAME = "current_overview_reconstruction"


@pytest.fixture(scope="module")
def solved():
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return make_eddy_solved(sample_ods(39915))


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _with_flags(ods, flags):
    """A copy of ``ods`` whose slices carry ``code.output_flag`` = ``flags``."""
    flagged = ods.copy()
    flagged["equilibrium.code.output_flag"] = np.asarray(flags, dtype=int)
    return flagged


def _build(ods, **options) -> Panels:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_model(NAME, normalize_entries(ods), **options)


#: A window wider than any record: the whole of every trace, for tests that
#: compare against the stored arrays (the default window is the discharge).
EVERYTHING = (-1.0e9, 1.0e9)


def _by_label(panel: LineSeries, prefix: str):
    return [s for s in panel.series if s.label.startswith(prefix)]


def test_three_stacked_panels_in_kiloampere(solved):
    model = _build(solved, time_range=EVERYTHING)
    assert isinstance(model, Panels) and model.ncols == 1 and model.share_x
    assert [panel.y_unit for panel in model.models] == ["kA", "kA", "kA"]
    assert all(isinstance(panel, LineSeries) for panel in model.models)
    measured = model.models[0].series[0]
    stored = np.asarray(solved["magnetics.ip.0.data"], dtype=float)
    assert np.allclose(np.abs(measured.y), np.abs(stored) * 1e-3)


def test_slices_without_evidence_are_unknown_not_converged(solved):
    ip = _build(solved).models[0]
    count = len(solved["equilibrium.time_slice"])
    unknown = _by_label(ip, "EFIT (convergence unknown)")
    assert len(unknown) == 1 and unknown[0].x.size == count
    assert unknown[0].label.endswith(f"(n={count})")
    assert not _by_label(ip, "EFIT converged") and not _by_label(ip, "EFIT not converged")


def test_marker_split_follows_the_convergence_verdict(solved):
    count = len(solved["equilibrium.time_slice"])
    flags = np.zeros(count, dtype=int)
    flags[2] = -1
    ip = _build(_with_flags(solved, flags)).models[0]
    (converged,) = _by_label(ip, "EFIT converged")
    (failed,) = _by_label(ip, "EFIT not converged")
    assert converged.x.size == count - 1 and failed.x.size == 1
    assert converged.label == f"EFIT converged (n={count - 1})"
    assert failed.label == "EFIT not converged (n=1)"
    assert failed.style["marker"] == "x" and converged.style["marker"] == "o"
    # Matched by the slice's own time and value, not by position in the bucket.
    times = np.asarray(solved["equilibrium.time"], dtype=float)
    value = float(solved["equilibrium.time_slice.2.global_quantities.ip"])
    assert failed.x[0] == pytest.approx(times[2])
    assert abs(failed.y[0]) == pytest.approx(abs(value) * 1e-3)


def test_every_eddy_loop_is_faint_on_the_right_axis_and_sums_to_the_total(solved):
    eddy = _build(solved, time_range=EVERYTHING).models[2]
    loops = [s for s in eddy.series if s.secondary]
    (total,) = _by_label(eddy, "Total")
    assert not total.secondary
    assert len(loops) == len(solved["pf_passive.loop"])
    assert total.label == f"Total ({len(loops)} loops)"
    # One loop names them all; the rest stay out of the legend.
    assert [s.label for s in loops if s.label] == ["Each loop (right axis)"]
    assert all(s.style["alpha"] < 1 for s in loops)
    assert eddy.secondary_y_unit == "kA"
    assert np.allclose(np.sum([s.y for s in loops], axis=0), total.y)
    raw = np.sum([np.asarray(solved[f"pf_passive.loop.{i}.current"]) for i in range(len(loops))], axis=0)
    assert np.allclose(total.y, raw * 1e-3)


def test_pf_panel_is_the_stored_coil_current_in_kiloampere(solved):
    pf = _build(solved, time_range=EVERYTHING).models[1]
    assert pf.series
    for trace in pf.series:
        current = np.asarray(solved[f"pf_active.coil.{trace.index}.current.data"], dtype=float)
        assert np.allclose(trace.y, current * 1e-3)


def test_orientation_flips_measured_and_reconstructed_together(solved):
    canonical = _build(solved, orientation="canonical", time_range=EVERYTHING).models[0]
    intuitive = _build(solved, orientation="intuitive", time_range=EVERYTHING).models[0]
    stored = np.asarray(solved["magnetics.ip.0.data"], dtype=float)
    assert np.allclose(canonical.series[0].y, stored * 1e-3)
    sign = np.sign(np.sum(intuitive.series[0].y * canonical.series[0].y))
    assert sign != 0
    for a, b in zip(canonical.series, intuitive.series):
        assert np.allclose(b.y, sign * a.y)
    # The default is the intuitive sign, as for plasma_current_time.
    assert np.allclose(_build(solved, time_range=EVERYTHING).models[0].series[0].y, intuitive.series[0].y)


def test_time_range_windows_every_panel(solved):
    times = np.asarray(solved["equilibrium.time"], dtype=float)
    window = (float(times[1]), float(times[-2]))
    model = _build(solved, time_range=window)
    for panel in model.models:
        assert panel.x_limits == window
        for trace in panel.series:
            assert trace.x.min() >= window[0] and trace.x.max() <= window[1]


def test_renders_through_the_public_function(solved):
    fig, axes = vaft.omas.plot_current_overview_reconstruction(solved)
    axes = np.ravel(axes)
    assert len(axes) == 3
    assert "[kA]" in axes[1].get_ylabel() and "turns" not in axes[1].get_ylabel()
    # The loops are drawn on a right-hand twin, keyed in the panel's one legend.
    legend = axes[2].get_legend()
    assert legend is not None
    assert [t.get_text() for t in legend.get_texts()] == [
        "Each loop (right axis)", f"Total ({len(solved['pf_passive.loop'])} loops)"]
    (twin,) = [a for a in fig.axes if a is not axes[2] and a.get_shared_x_axes().joined(a, axes[2])
               and a.yaxis.get_label_position() == "right"]
    assert len(twin.lines) == len(solved["pf_passive.loop"])
    assert "per loop" in twin.get_ylabel()


def test_secondary_axis_puts_both_zeros_level():
    from vaft.plot.models import Series
    from vaft.plot.renderers.lines import render_line_series

    t = np.linspace(0.0, 1.0, 50)
    model = LineSeries(
        series=(Series(x=t, y=100.0 * np.sin(6 * t) + 30.0, label="primary"),
                Series(x=t, y=np.cos(6 * t), label="secondary", secondary=True)),
        secondary_y_label="per loop", secondary_y_unit="kA",
    )
    fig, ax = render_line_series(model)
    (twin,) = [a for a in fig.axes if a is not ax]
    low, high = ax.get_ylim()
    s_low, s_high = twin.get_ylim()
    assert -low / (high - low) == pytest.approx(-s_low / (s_high - s_low))
    # Widened, never cut: the secondary trace stays in view.
    assert s_low <= -1.0 and s_high >= 1.0
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["primary", "secondary"]


def test_secondary_axis_in_plotly_overlays_the_cell():
    pytest.importorskip("plotly")
    from vaft.plot.models import Series
    from vaft.plot.plotly import renderer_for_model

    t = np.linspace(0.0, 1.0, 20)
    panel = LineSeries(series=(Series(x=t, y=t, label="a"),
                               Series(x=t, y=-t, label="b", secondary=True)),
                       secondary_y_label="per loop", secondary_y_unit="kA")
    model = Panels(models=(panel, panel), ncols=1, share_x=True)
    figure = renderer_for_model(Panels).render(model)
    on_secondary = [trace for trace in figure.data if trace.yaxis not in (None, "y", "y2")]
    assert len(on_secondary) == 2
    # Each panel's overlay sits over its own cell, not the first one.
    overlaid = sorted(figure.layout[t.yaxis.replace("y", "yaxis")].overlaying for t in on_secondary)
    assert overlaid == ["y", "y2"]
    for trace in on_secondary:
        assert figure.layout[trace.yaxis.replace("y", "yaxis")].side == "right"


def test_default_window_is_the_discharge_not_the_record(solved):
    model = _build(solved)
    limits = model.models[0].x_limits
    ip = model.models[0].series[0]
    record = np.asarray(solved["magnetics.ip.0.time"], dtype=float)
    assert limits is not None
    # Narrower than the record, and holding every sample where I_p is driven.
    assert limits[1] - limits[0] < record.max() - record.min()
    driven = ip.x[np.abs(ip.y) > 0.1 * np.nanmax(np.abs(ip.y))]
    assert limits[0] <= driven.min() and driven.max() <= limits[1]
    for panel in model.models:
        assert panel.x_limits == limits


def test_default_figure_is_narrow_with_legends_inside(solved):
    fig, axes = vaft.omas.plot_current_overview_reconstruction(solved, format="slide")
    width, _ = fig.get_size_inches()
    assert width == pytest.approx(6.5)
    for axis in np.ravel(axes):
        legend = axis.get_legend()
        if legend is None:
            continue
        box = legend.get_window_extent(fig.canvas.get_renderer())
        frame = axis.get_window_extent(fig.canvas.get_renderer())
        assert frame.x0 <= box.x0 and box.x1 <= frame.x1
    # Title over the time axis, not the canvas.
    first = np.ravel(axes)[0].get_position()
    assert fig._suptitle.get_position()[0] == pytest.approx(first.x0 + first.width / 2)
    # An explicit canvas is the caller's.
    fig2, _ = vaft.omas.plot_current_overview_reconstruction(solved, figsize=(10.0, 6.0))
    assert tuple(fig2.get_size_inches()) == pytest.approx((10.0, 6.0))


def test_marker_counts_are_of_the_slices_drawn(solved):
    times = np.asarray(solved["equilibrium.time"], dtype=float)
    window = (float(times[1]), float(times[-2]))
    ip = _build(solved, time_range=window).models[0]
    markers = [trace for trace in ip.series if trace.label.startswith("EFIT")]
    assert markers
    for trace in markers:
        assert trace.label.endswith(f"(n={trace.x.size})")
    assert sum(trace.x.size for trace in markers) == times.size - 2


def test_without_an_equilibrium_the_top_panel_is_the_measurement(solved):
    bare = solved.copy()
    del bare["equilibrium"]
    ip = _build(bare).models[0]
    assert [trace.label for trace in ip.series] == ["Measured"]


def test_one_missing_loop_sample_does_not_blank_the_total(solved):
    gappy = solved.copy()
    current = np.asarray(gappy["pf_passive.loop.0.current"], dtype=float).copy()
    current[10] = np.nan
    gappy["pf_passive.loop.0.current"] = current
    eddy = _build(gappy, time_range=EVERYTHING).models[2]
    (total,) = _by_label(eddy, "Total")
    assert np.isfinite(total.y[10])


def test_legends_take_the_theme_and_the_format_layout(solved):
    fig, axes = vaft.omas.plot_current_overview_reconstruction(solved, format="slide", theme="minimal")
    flat = np.ravel(axes)
    label_family = flat[0].yaxis.label.get_fontfamily()
    for axis in flat:
        legend = axis.get_legend()
        if legend is not None:
            assert legend.get_texts()[0].get_fontfamily() == label_family
    # The suptitle stays clear of the first panel under the legacy layout too.
    fig2, axes2 = vaft.omas.plot_current_overview_reconstruction(solved, format="legacy")
    renderer = fig2.canvas.get_renderer()
    title = fig2._suptitle.get_window_extent(renderer)
    top = np.ravel(axes2)[0].get_window_extent(renderer)
    assert title.y0 >= top.y1 - 1.0


def test_a_line_series_without_secondary_traces_draws_as_before():
    from vaft.plot.models import Series
    from vaft.plot.renderers.lines import render_line_series

    t = np.linspace(0.0, 1.0, 10)
    fig, ax = render_line_series(LineSeries(series=(Series(x=t, y=t, label="a"), Series(x=t, y=-t, label="b"))))
    assert fig.axes == [ax]
    assert ax.patch.get_visible()
    assert ax.get_zorder() == 0
