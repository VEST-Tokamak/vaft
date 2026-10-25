"""One owner for the SXR constant-offset baselines (#2162 B3.11).

Before the consolidation the soft-X-ray chain removed a constant per-channel
offset in three places, each with its own window rule:

* ``vaft.machine_mapping.soft_x_rays.vfit_soft_x_rays_dynamic`` -- the mean of
  an **index** range ``signal[start:stop]`` (half-open, NaN-propagating); an
  empty range raises;
* ``vaft.process.soft_x_rays.sxr_baseline_correction`` -- the mean of the
  quiet **tail** ``time >= baseline_start``; no sample there falls back to the
  last 1000 samples;
* ``vaft.plot.soft_x_rays._baseline_correct`` -- the NaN-skipping mean of a
  closed **time** range ``[start, stop]``; an empty range is a no-op.

The three are the same operation -- subtract the mean of a selection along
the sample axis -- and differ only in how the selection is made, whether NaN
is skipped, and what an empty selection does.  They now all call
``vaft.process.soft_x_rays.sxr_offset_baseline``, which takes those three
choices explicitly.  The frozen copies below are the three as they stood on
develop f7472664d; each is held to its replacement bit for bit.

The fourth variant the audit lists, the centred rolling trend in
``sxr_electron_temperature``, is not a baseline: it is the trend a Te
fluctuation is measured against (a moving mean of Te, not an offset of the
detector signal), so it stays where it is.  A test below pins that it is
untouched.
"""

from __future__ import annotations

import numpy as np
import pytest
from omas import ODS

# ---------------------------------------------------------------------------
# Frozen copies (develop f7472664d)
# ---------------------------------------------------------------------------


def _frozen_mm_offset(signal, baseline_range):
    signal = np.asarray(signal, dtype=float).copy()
    start, stop = baseline_range
    baseline_values = signal[slice(start, stop)]
    if baseline_values.size == 0:
        raise ValueError("baseline_range selects no SXR samples.")
    baseline = np.mean(baseline_values)
    return signal - baseline


def _frozen_tail_correction(time, data, baseline_start):
    time = np.asarray(time, dtype=float)
    values = np.asarray(data, dtype=float)
    if values.ndim == 1:
        values = values[np.newaxis, :]
    mask = time >= float(baseline_start)
    if np.any(mask):
        baseline = values[:, mask].mean(axis=1, keepdims=True)
    else:
        baseline = values[:, -1000:].mean(axis=1, keepdims=True)
    return values - baseline


def _frozen_plot_offset(time, signal, baseline_range):
    corrected = np.asarray(signal, dtype=float).copy()
    if baseline_range is None:
        return corrected
    start, stop = baseline_range
    mask = (time >= float(start)) & (time <= float(stop))
    if np.any(mask):
        corrected -= float(np.nanmean(corrected[mask]))
    return corrected


# ---------------------------------------------------------------------------
# Synthetic records
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(2162_11)
_TIME = np.arange(0.0, 0.05, 1.0 / 125_000.0)  # 6250 samples, a VEST SXR-like rate


def _traces(n_channels=4, *, nan_at=()):
    dark = _RNG.uniform(-0.3, 0.3, size=(n_channels, 1))
    pulse = np.exp(-(((_TIME - 0.02) / 0.004) ** 2))
    noise = 1e-3 * _RNG.standard_normal((n_channels, _TIME.size))
    values = dark + 0.8 * pulse + 2e-4 * _TIME + noise
    for channel, sample in nan_at:
        values[channel, sample] = np.nan
    return values


_INDEX_RANGES = [(0, 500), (None, 300), (6000, None), (-200, None), (100, 101), (0, 10_000)]
_TAIL_STARTS = [0.04, 0.0, 0.049, 1.0]  # 1.0: no sample -> last-1000 fallback
_TIME_RANGES = [(0.0, 0.004), (0.045, 0.05), (0.03, 0.03), (0.06, 0.07), (-1.0, 1.0)]


def _owner():
    from vaft.process.soft_x_rays import sxr_offset_baseline

    return sxr_offset_baseline


# ---------------------------------------------------------------------------
# Each caller against its frozen copy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("baseline_range", _INDEX_RANGES)
@pytest.mark.parametrize("nan_at", [(), ((0, 50),)])
def test_mapper_offset_is_the_frozen_index_range_mean(baseline_range, nan_at):
    from vaft.machine_mapping.soft_x_rays import vfit_soft_x_rays_dynamic

    values = _traces(nan_at=nan_at)
    ods = ODS()
    channel_map = [{"source_column": c, "identifier": f"SXR{c:02d}"} for c in range(values.shape[0])]
    vfit_soft_x_rays_dynamic(
        ods, data=values.T, time=_TIME, channel_map=channel_map, baseline_range=baseline_range,
        polarity=-1.0, brightness_scale=2.5,
    )
    for c in range(values.shape[0]):
        expected = -1.0 * 2.5 * _frozen_mm_offset(values[c], baseline_range)
        got = np.asarray(ods[f"soft_x_rays.channel.{c}.brightness.data"]).reshape(-1)
        np.testing.assert_array_equal(got, expected)


def test_mapper_empty_index_range_still_raises_the_same_error():
    from vaft.machine_mapping.soft_x_rays import vfit_soft_x_rays_dynamic

    values = _traces(1)
    with pytest.raises(ValueError, match="^baseline_range selects no SXR samples.$"):
        vfit_soft_x_rays_dynamic(
            ODS(), data=values.T, time=_TIME, channel_map=[{"source_column": 0, "identifier": "SXR00"}],
            baseline_range=(500, 500),
        )


@pytest.mark.parametrize("baseline_start", _TAIL_STARTS)
@pytest.mark.parametrize("shape", ["matrix", "trace"])
@pytest.mark.parametrize("nan_at", [(), ((1, 6200),)])
def test_tail_correction_is_the_frozen_tail_mean(baseline_start, shape, nan_at):
    from vaft.process.soft_x_rays import sxr_baseline_correction

    values = _traces(nan_at=nan_at)
    data = values if shape == "matrix" else values[1]
    np.testing.assert_array_equal(
        sxr_baseline_correction(_TIME, data, baseline_start),
        _frozen_tail_correction(_TIME, data, baseline_start),
    )


@pytest.mark.parametrize("baseline_range", _TIME_RANGES + [None])
@pytest.mark.parametrize("nan_at", [(), ((0, 100),)])
def test_plot_offset_is_the_frozen_nan_skipping_range_mean(baseline_range, nan_at):
    from vaft.plot.soft_x_rays import _baseline_correct

    signal = _traces(1, nan_at=nan_at)[0]
    np.testing.assert_array_equal(
        _baseline_correct(_TIME, signal, baseline_range),
        _frozen_plot_offset(_TIME, signal, baseline_range),
    )


# ---------------------------------------------------------------------------
# The owner's own contract
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=True, raises=ImportError, reason="B3.11 owner lands in the next commit")
def test_owner_removes_exactly_the_selection_mean_per_channel():
    values = _traces()
    selection = _TIME < 0.004
    out = _owner()(values, selection)
    np.testing.assert_allclose(out[:, selection].mean(axis=1), 0.0, atol=1e-15)
    assert out.shape == values.shape


@pytest.mark.xfail(strict=True, raises=ImportError, reason="B3.11 owner lands in the next commit")
def test_owner_states_what_an_empty_selection_does():
    values = _traces(2)
    empty = np.zeros(_TIME.size, dtype=bool)
    with pytest.raises(ValueError, match="selects no sample"):
        _owner()(values, empty, on_empty="raise")
    np.testing.assert_array_equal(_owner()(values, empty, on_empty="keep"), values)
    np.testing.assert_array_equal(
        _owner()(values, empty, on_empty="tail", tail_samples=1000),
        values - values[:, -1000:].mean(axis=1, keepdims=True),
    )


@pytest.mark.xfail(strict=True, raises=ImportError, reason="B3.11 owner lands in the next commit")
def test_owner_nan_policy():
    values = _traces(1, nan_at=((0, 10),))
    selection = slice(0, 100)
    assert np.isnan(_owner()(values, selection)).all()
    skipped = _owner()(values, selection, skip_nan=True)
    assert np.isnan(skipped[0, 10]) and np.isfinite(np.delete(skipped, 10, axis=1)).all()


@pytest.mark.xfail(strict=True, raises=ImportError, reason="B3.11 owner lands in the next commit")
def test_owner_refuses_unknown_options():
    with pytest.raises(ValueError, match="on_empty"):
        _owner()(_traces(1), slice(0, 10), on_empty="zero")
    with pytest.raises(ValueError, match="tail_samples"):
        _owner()(_traces(1), slice(0, 0), on_empty="tail")


# ---------------------------------------------------------------------------
# The fourth variant is not a baseline and is left alone
# ---------------------------------------------------------------------------


def test_the_rolling_detrend_is_not_routed_through_the_offset_owner():
    import ast
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "vaft/process/soft_x_rays.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "sxr_electron_temperature"
    )
    body = ast.get_source_segment(source, function)
    assert ".rolling(window=int(detrend_window), center=True, min_periods=1)" in body
    assert "sxr_offset_baseline" not in body
