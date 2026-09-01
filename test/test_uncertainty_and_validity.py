"""Uncertainty and validity rendering contract (issue #256, phase C).

Plotting renders what the source stored and never invents it: it does not
estimate uncertainty, does not compute validity, and does not label a band as a
confidence interval without metadata saying so.  Four states stay distinct --
valid, invalid interval, invalid channel, missing -- and the default never
hides the invalid ones.  Policy:
``notebooks/plotting_sample_using_vaft_plot_module.ipynb``.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import omas
import pytest

import vaft.omas
from vaft.plot.models import LineSeries, Series
from vaft.plot.renderers.lines import render_line_series
from vaft.plot.style import INVALID_COLOR, draw_series


def _line(**kwargs):
    return Series(x=np.linspace(0.0, 1.0, 5), y=np.ones(5), **kwargs)


# ---------------------------------------------------------------------------
# The four states
# ---------------------------------------------------------------------------

def test_a_valid_trace_is_drawn_plainly():
    figure, axes = plt.subplots()
    draw_series(axes, _line(), label="ok")
    line = axes.lines[0]
    assert line.get_linestyle() == "-"
    assert line.get_color() != INVALID_COLOR
    plt.close(figure)


def test_an_invalid_channel_is_demoted_not_hidden():
    figure, axes = plt.subplots()
    labelled = draw_series(axes, _line(validity=-1), label="MP07")
    assert labelled
    line = axes.lines[0]
    # Visible, but visibly untrustworthy, and the legend says why.
    assert line.get_linestyle() == "--"
    assert line.get_color() == INVALID_COLOR
    assert line.get_label() == "MP07 (invalid)"
    plt.close(figure)


def test_an_invalid_interval_shades_only_that_span():
    figure, axes = plt.subplots()
    draw_series(axes, _line(valid_mask=[True, False, False, True, True]))
    spans = axes.patches
    assert len(spans) == 1, "one contiguous dropout should shade one span"
    assert axes.lines, "the trace itself is still drawn"
    plt.close(figure)


def test_two_separate_dropouts_shade_separately():
    figure, axes = plt.subplots()
    draw_series(axes, _line(valid_mask=[False, True, False, True, True]))
    assert len(axes.patches) == 2
    plt.close(figure)


def test_a_valid_channel_flag_is_not_an_invalid_one():
    """IMAS spells "valid" as 0; only a negative code means invalid."""
    assert not _line(validity=0).is_invalid_channel
    assert _line(validity=-1).is_invalid_channel
    assert _line(validity=-2).is_invalid_channel
    assert not _line().is_invalid_channel


# ---------------------------------------------------------------------------
# The validity modes
# ---------------------------------------------------------------------------

def test_mask_removes_what_show_demotes():
    figure, axes = plt.subplots()
    assert draw_series(axes, _line(validity=-1), validity="mask", label="x") is False
    assert not axes.lines
    plt.close(figure)


def test_mask_drops_only_the_invalid_samples_of_a_partial_trace():
    figure, axes = plt.subplots()
    draw_series(axes, _line(valid_mask=[True, False, False, True, True]), validity="mask")
    assert len(axes.lines[0].get_xdata()) == 3
    assert not axes.patches, "masked data needs no dropout shading"
    plt.close(figure)


def test_ignore_renders_as_though_there_were_no_metadata():
    figure, axes = plt.subplots()
    draw_series(axes, _line(validity=-1), validity="ignore", label="x")
    line = axes.lines[0]
    assert line.get_linestyle() == "-"
    assert line.get_label() == "x"
    plt.close(figure)


def test_unknown_modes_are_refused():
    figure, axes = plt.subplots()
    with pytest.raises(ValueError, match="validity must be one of"):
        draw_series(axes, _line(), validity="hide")
    with pytest.raises(ValueError, match="uncertainty must be one of"):
        draw_series(axes, _line(), uncertainty="sigma")
    plt.close(figure)


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------

def test_a_continuous_trace_gets_a_band_and_a_scatter_gets_error_bars():
    figure, axes = plt.subplots()
    draw_series(axes, _line(yerr=np.full(5, 0.1)))
    assert axes.collections, "a continuous trace shades a band"
    plt.close(figure)

    figure, axes = plt.subplots()
    draw_series(
        axes, _line(yerr=np.full(5, 0.1), style={"marker": "o", "linestyle": "none"})
    )
    assert axes.containers, "a scatter-like trace gets error bars"
    plt.close(figure)


def test_the_band_mode_is_forced_independently_of_the_trace_shape():
    figure, axes = plt.subplots()
    draw_series(
        axes,
        _line(yerr=np.full(5, 0.1), style={"marker": "o", "linestyle": "none"}),
        uncertainty="band",
    )
    assert axes.collections and not axes.containers
    plt.close(figure)


def test_asymmetric_uncertainty_uses_both_bounds():
    lower, upper = np.full(5, 0.1), np.full(5, 0.4)
    figure, axes = plt.subplots()
    draw_series(axes, _line(yerr=np.vstack([lower, upper])), uncertainty="band")
    band = axes.collections[0].get_paths()[0].vertices[:, 1]
    assert band.min() == pytest.approx(0.9)
    assert band.max() == pytest.approx(1.4)
    plt.close(figure)


def test_uncertainty_none_draws_the_trace_alone():
    figure, axes = plt.subplots()
    draw_series(axes, _line(yerr=np.full(5, 0.1)), uncertainty="none")
    assert axes.lines and not axes.collections and not axes.containers
    plt.close(figure)


def test_plotting_never_invents_a_spread():
    """No stored uncertainty means no band -- not a default or estimated one."""
    figure, axes = plt.subplots()
    draw_series(axes, _line())
    assert not axes.collections
    plt.close(figure)


# ---------------------------------------------------------------------------
# Through the renderer and the adapters
# ---------------------------------------------------------------------------

def test_the_renderer_passes_the_modes_through():
    model = LineSeries(series=(_line(validity=-1),), y_label="V")
    figure, axes = render_line_series(model, validity="mask")
    assert not axes.lines
    plt.close(figure)


def test_the_adapter_reads_imas_validity_from_the_ods():
    ods = omas.ODS()
    ods["magnetics.time"] = np.linspace(0.0, 0.1, 4)
    for index, code in enumerate((0, -1)):
        ods[f"magnetics.b_field_pol_probe.{index}.voltage.data"] = np.ones(4) * (index + 1)
        ods[f"magnetics.b_field_pol_probe.{index}.voltage.validity"] = code

    figure, axes = vaft.omas.plot_mirnov_time_voltage(ods)
    styles = sorted(line.get_linestyle() for line in axes.lines)
    assert styles == ["-", "--"], "the flagged channel is demoted, the other is not"
    plt.close(figure)

    figure, axes = vaft.omas.plot_mirnov_time_voltage(ods, validity="mask")
    assert len(axes.lines) == 1
    plt.close(figure)


def test_the_packaged_sample_has_channels_its_source_flagged(sample_ods=None):
    """Regression guard: these were previously drawn as if trustworthy."""
    from vaft.data import sample

    ods = vaft.omas.load(sample(39915, "omas"))
    figure, axes = vaft.omas.plot_mirnov_time_voltage(ods)
    demoted = [line for line in axes.lines if line.get_linestyle() == "--"]
    assert demoted, "the sample carries channels flagged invalid at the source"
    assert all("(invalid)" in line.get_label() for line in demoted)
    plt.close(figure)
