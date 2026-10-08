"""current_overview_reconstruction: I_p measured and per slice, PF ampere-turns, eddy currents.

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


def _by_label(panel: LineSeries, prefix: str):
    return [s for s in panel.series if s.label.startswith(prefix)]


def test_three_stacked_panels_in_kiloampere(solved):
    model = _build(solved)
    assert isinstance(model, Panels) and model.ncols == 1 and model.share_x
    assert [panel.y_unit for panel in model.models] == ["kA", "kA-turns", "kA"]
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


def test_every_eddy_loop_is_faint_unlabelled_and_sums_to_the_total(solved):
    eddy = _build(solved).models[2]
    loops = [s for s in eddy.series if not s.label]
    (total,) = _by_label(eddy, "Total")
    assert len(loops) == len(solved["pf_passive.loop"])
    assert total.label == f"Total ({len(loops)} loops)"
    assert all(s.style["alpha"] < 1 for s in loops)
    assert np.allclose(np.sum([s.y for s in loops], axis=0), total.y)
    raw = np.sum([np.asarray(solved[f"pf_passive.loop.{i}.current"]) for i in range(len(loops))], axis=0)
    assert np.allclose(total.y, raw * 1e-3)


def test_pf_panel_weights_by_turns(solved):
    pf = _build(solved).models[1]
    for trace in pf.series:
        i = trace.index
        turns = np.sum(np.abs(np.asarray(solved[f"pf_active.coil.{i}.element.:.turns_with_sign"], dtype=float)))
        current = np.asarray(solved[f"pf_active.coil.{i}.current.data"], dtype=float)
        assert np.allclose(trace.y, current * turns * 1e-3)


def test_orientation_flips_measured_and_reconstructed_together(solved):
    canonical = _build(solved, orientation="canonical").models[0]
    intuitive = _build(solved, orientation="intuitive").models[0]
    stored = np.asarray(solved["magnetics.ip.0.data"], dtype=float)
    assert np.allclose(canonical.series[0].y, stored * 1e-3)
    sign = np.sign(np.sum(intuitive.series[0].y * canonical.series[0].y))
    assert sign != 0
    for a, b in zip(canonical.series, intuitive.series):
        assert np.allclose(b.y, sign * a.y)
    # The default is the intuitive sign, as for plasma_current_time.
    assert np.allclose(_build(solved).models[0].series[0].y, intuitive.series[0].y)


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
    assert "kA-turns" in axes[1].get_ylabel()
    # The total is named even though it is the eddy panel's only labelled trace.
    legend = axes[2].get_legend()
    assert legend is not None
    assert [t.get_text() for t in legend.get_texts()] == [f"Total ({len(solved['pf_passive.loop'])} loops)"]
