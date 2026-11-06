"""One owner for the index-based linear baseline.

Before the consolidation three helpers fitted a straight line through selected
samples with ``np.polyfit`` and evaluated it with ``np.polyval``:

* ``vaft.process.magnetics._linear_baseline`` -- the probe and flux-loop
  chains, with its guards (out-of-range indices dropped, non-finite
  samples dropped, fewer than two valid samples raises
  ``DegenerateBaselineWindowError`` or, when allowed, falls back to zero);
* ``vaft.machine_mapping.magnetics._linear_baseline_subtract`` -- the plasma
  current, with no guard at all: a NaN in the window poisons the fit and the
  whole trace comes back NaN;
* ``vaft.machine_mapping.magnetics._polyfit_baseline`` -- no caller.

The frozen copies below are those helpers verbatim, as they stood on develop
at 75b154e31.  They are the reference the canonical
implementation is held to, so the comparison does not depend on the code it
checks.  Agreement is asserted bit for bit: the canonical path performs the
same ``np.polyfit`` / ``np.polyval`` calls on the same samples.

Scope of the identity for the plasma-current helper: it holds for in-range
integer indices with at least two samples, which is all its only caller
(``_plasma_current_baseline_indices``) can produce.  Outside that the frozen
helper indexed raw (negative indices wrapped, past-the-end raised
``IndexError``, a one-sample window raised ``LinAlgError``); the delegating one
applies the guarded behaviour instead (drops out-of-range indices, raises
``DegenerateBaselineWindowError``).  A boolean mask now selects samples under
``fitting_opt="polyfit"`` rather than being cast to the indices 0 and 1.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from vaft.machine_mapping import magnetics as mm_magnetics
from vaft.process import magnetics as process_magnetics
from vaft.process.magnetics import DegenerateBaselineWindowError


# ---------------------------------------------------------------------------
# Frozen references (develop 75b154e31), verbatim apart from the names
# ---------------------------------------------------------------------------

def _frozen_process_linear_baseline(
    time_axis, values, indices, *, allow_zero_fallback=False, shot=None, channel=None, window_desc=None
):
    indices = np.asarray(indices, dtype=int)
    in_bounds = indices[(indices >= 0) & (indices < values.size)]
    valid = in_bounds[np.isfinite(values[in_bounds]) & np.isfinite(time_axis[in_bounds])]
    if valid.size < 2:
        context_parts = []
        if shot is not None:
            context_parts.append(f"shot {shot}")
        if channel is not None:
            context_parts.append(f"channel {channel!r}")
        context_str = f" [{', '.join(context_parts)}]" if context_parts else ""

        if allow_zero_fallback:
            warnings.warn(
                f"Degenerate baseline window{context_str}: {valid.size} valid sample(s) available "
                f"(requires >= 2). Falling back to zero baseline because allow_zero_fallback=True.",
                UserWarning,
                stacklevel=2,
            )
            return np.zeros(values.size, dtype=float)

        prefix = f"for {', '.join(context_parts)}: " if context_parts else ""
        time_span = (
            f"[{time_axis[0]:.6g}, {time_axis[-1]:.6g}] s"
            if time_axis.size > 0
            else "[]"
        )
        req_desc = f" ({window_desc})" if window_desc else ""
        if indices.size > 0:
            index_range = f"[{indices.min()}, {indices.max()}] (count: {indices.size})"
        else:
            index_range = "[] (empty)"

        raise DegenerateBaselineWindowError(
            f"Degenerate baseline window {prefix}found {valid.size} valid sample(s) (requires >= 2). "
            f"Requested indices{req_desc}: {index_range}. "
            f"Signal sample count: {values.size}, time span: {time_span}."
        )

    return np.polyval(np.polyfit(time_axis[valid], values[valid], 1), time_axis)


def _frozen_mm_linear_baseline_subtract(time, values, x_base):
    return values - np.polyval(np.polyfit(time[x_base], values[x_base], 1), time)


# ---------------------------------------------------------------------------
# Synthetic cases
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(2162)
_TIME = np.linspace(0.0, 0.4, 10_001)
#: An integrator drift (offset + slope) plus a pulse and noise, on a 1e-3 scale
#: like an integrated probe signal.
_DRIFTING = (
    2.0e-4
    + 3.0e-3 * _TIME
    + 5.0e-3 * np.exp(-(((_TIME - 0.3) / 0.01) ** 2))
    + 1.0e-6 * _RNG.standard_normal(_TIME.size)
)
_WINDOWS = {
    "leading": np.arange(0, 2_000),
    "two_regions": np.concatenate((np.arange(0, 1_500), np.arange(9_000, 10_001))),
    "out_of_range": np.arange(-50, 40),
    "past_the_end": np.arange(9_990, 10_050),
}


def _with_nan(values, positions):
    out = values.copy()
    out[list(positions)] = np.nan
    return out


@pytest.mark.parametrize("window", sorted(_WINDOWS))
def test_process_linear_baseline_matches_frozen_reference(window):
    indices = _WINDOWS[window]
    expected = _frozen_process_linear_baseline(_TIME, _DRIFTING, indices)
    actual = process_magnetics._linear_baseline(_TIME, _DRIFTING, indices)
    np.testing.assert_array_equal(actual, expected)


def test_process_linear_baseline_drops_nonfinite_samples_like_the_reference():
    values = _with_nan(_DRIFTING, (3, 17, 1_200))
    time = _TIME.copy()
    time[25] = np.nan  # a non-finite *time* sample is dropped too
    indices = _WINDOWS["leading"]
    expected = _frozen_process_linear_baseline(time, values, indices)
    actual = process_magnetics._linear_baseline(time, values, indices)
    np.testing.assert_array_equal(actual, expected)
    assert np.isfinite(actual[np.isfinite(time)]).all()


@pytest.mark.parametrize(
    ("indices", "kwargs"),
    [
        (np.array([5]), {"shot": 41445, "channel": "probe_test"}),
        (np.array([], dtype=int), {"shot": 43685, "channel": "flux_loop_3", "window_desc": "time window [0.5, 0.6] s"}),
        (np.array([-3, 20_000]), {}),
        (np.array([1, 2]), {"channel": 7, "window_desc": "samples 1:3"}),
    ],
)
def test_process_linear_baseline_degenerate_error_message_is_unchanged(indices, kwargs):
    values = _with_nan(_DRIFTING, (2,)) if indices.size == 2 else _DRIFTING
    with pytest.raises(DegenerateBaselineWindowError) as expected:
        _frozen_process_linear_baseline(_TIME, values, indices, **kwargs)
    with pytest.raises(DegenerateBaselineWindowError) as actual:
        process_magnetics._linear_baseline(_TIME, values, indices, **kwargs)
    assert str(actual.value) == str(expected.value)


def test_process_linear_baseline_zero_fallback_warning_is_unchanged():
    kwargs = {"allow_zero_fallback": True, "shot": 41445, "channel": "probe_test"}
    with pytest.warns(UserWarning) as expected_record:
        expected = _frozen_process_linear_baseline(_TIME, _DRIFTING, np.array([4]), **kwargs)
    with pytest.warns(UserWarning) as actual_record:
        actual = process_magnetics._linear_baseline(_TIME, _DRIFTING, np.array([4]), **kwargs)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype
    assert [str(w.message) for w in actual_record] == [str(w.message) for w in expected_record]
    # stacklevel still names the caller of _linear_baseline, i.e. this file.
    assert actual_record[0].filename == __file__


@pytest.mark.parametrize("window", ["leading", "two_regions"])
def test_mm_linear_baseline_subtract_matches_frozen_reference(window):
    indices = _WINDOWS[window]
    expected = _frozen_mm_linear_baseline_subtract(_TIME, _DRIFTING, indices)
    actual = mm_magnetics._linear_baseline_subtract(_TIME, _DRIFTING, indices)
    np.testing.assert_array_equal(actual, expected)


def test_mm_linear_baseline_subtract_still_propagates_a_nan_in_the_window():
    """The plasma-current helper had no finite guard: a NaN in the baseline
    window makes the whole corrected trace NaN.  Consolidation keeps that --
    turning an all-NaN Ip into finite numbers would be a behaviour change, not
    a refactor."""
    values = _with_nan(_DRIFTING, (10,))
    indices = _WINDOWS["leading"]
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = _frozen_mm_linear_baseline_subtract(_TIME, values, indices)
        actual = mm_magnetics._linear_baseline_subtract(_TIME, values, indices)
    assert np.isnan(expected).all()
    np.testing.assert_array_equal(actual, expected)


def test_the_canonical_line_is_the_least_squares_line():
    """Independent of either implementation: an exact line through the window
    is recovered to round-off, so the subtraction leaves only the pulse."""
    line = 2.0e-4 + 3.0e-3 * _TIME
    pulse = 5.0e-3 * np.exp(-(((_TIME - 0.3) / 0.01) ** 2))
    indices = _WINDOWS["leading"]
    baseline = process_magnetics._linear_baseline(_TIME, line + pulse, indices)
    np.testing.assert_allclose(baseline, line, rtol=0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# The canonical owner: subtract_baseline(fitting_opt="polyfit")
# ---------------------------------------------------------------------------

from vaft.process import signal_processing  # noqa: E402


@pytest.mark.parametrize("window", sorted(_WINDOWS))
def test_subtract_baseline_polyfit_is_the_frozen_guarded_line(window):
    indices = _WINDOWS[window]
    inside = indices[(indices >= 0) & (indices < _TIME.size)]
    values = _with_nan(_DRIFTING, (int(inside[len(inside) // 2]),))  # a NaN the window really sees
    expected = _frozen_process_linear_baseline(_TIME, values, indices)
    corrected, baseline = signal_processing.subtract_baseline(
        _TIME, values, indices, fitting_opt="polyfit"
    )
    np.testing.assert_array_equal(baseline, expected)
    np.testing.assert_array_equal(corrected, values - expected)


def test_subtract_baseline_polyfit_propagate_is_the_frozen_plasma_current_line():
    indices = _WINDOWS["two_regions"]
    corrected, _ = signal_processing.subtract_baseline(
        _TIME, _DRIFTING, indices, fitting_opt="polyfit", nan_policy="propagate"
    )
    np.testing.assert_array_equal(corrected, _frozen_mm_linear_baseline_subtract(_TIME, _DRIFTING, indices))


def test_subtract_baseline_polyfit_degenerate_window_raises_or_falls_back():
    with pytest.raises(DegenerateBaselineWindowError, match=r"for shot 1, channel 'x': found 1 valid"):
        signal_processing.subtract_baseline(
            _TIME, _DRIFTING, np.array([3]), fitting_opt="polyfit", context="shot 1, channel 'x'"
        )
    with pytest.warns(UserWarning, match=r"\[ctx\]: 0 valid sample") as record:
        corrected, baseline = signal_processing.subtract_baseline(
            _TIME, _DRIFTING, np.array([], dtype=int), fitting_opt="polyfit",
            allow_zero_fallback=True, context="ctx",
        )
    assert record[0].filename == __file__
    np.testing.assert_array_equal(baseline, np.zeros_like(_DRIFTING))
    np.testing.assert_array_equal(corrected, _DRIFTING)


def test_subtract_baseline_rejects_an_unknown_nan_policy():
    with pytest.raises(ValueError, match="nan_policy"):
        signal_processing.subtract_baseline(
            _TIME, _DRIFTING, _WINDOWS["leading"], fitting_opt="polyfit", nan_policy="drop"
        )


def test_subtract_baseline_linear_curve_fit_path_is_left_as_it_was():
    """``"linear"`` (curve_fit) is not rerouted to polyfit: the two agree only
    to round-off-plus-tolerance, and the diamagnetic chain's stored values were
    built with curve_fit."""
    from scipy.optimize import curve_fit

    indices = _WINDOWS["two_regions"]
    popt, _ = curve_fit(lambda x, a, b: a * x + b, _TIME[indices], _DRIFTING[indices])
    _, baseline = signal_processing.subtract_baseline(_TIME, _DRIFTING, indices, fitting_opt="linear")
    np.testing.assert_array_equal(baseline, popt[0] * _TIME + popt[1])


def test_one_error_class_and_no_dead_helper():
    assert process_magnetics.DegenerateBaselineWindowError is signal_processing.DegenerateBaselineWindowError
    import vaft.process

    assert vaft.process.DegenerateBaselineWindowError is signal_processing.DegenerateBaselineWindowError
    assert not hasattr(mm_magnetics, "_polyfit_baseline")


def test_the_plasma_current_caller_only_produces_the_identity_domain():
    """The precondition the identity above rests on: in-range integer indices."""
    time = np.linspace(-0.01, 0.4, 5_000)
    for start, end, lookback in ((100, 200, 50), (10, 20, 50), (4_990, 5_100, 30), (6_000, 6_100, 10)):
        idx = np.asarray(mm_magnetics._plasma_current_baseline_indices(
            time, {"analysis_start": start, "analysis_end": end, "lookback": lookback}
        ))
        assert idx.dtype.kind == "i"
        assert ((idx >= 0) & (idx < time.size)).all()


def test_polyfit_accepts_a_boolean_mask_as_a_selection():
    mask = np.zeros(_TIME.size, dtype=bool)
    mask[:400] = True
    from vaft.process.signal_processing import subtract_baseline
    by_mask = subtract_baseline(_TIME, _DRIFTING, mask, fitting_opt="polyfit")
    by_index = subtract_baseline(_TIME, _DRIFTING, np.flatnonzero(mask), fitting_opt="polyfit")
    np.testing.assert_array_equal(by_mask, by_index)


def test_an_unknown_nan_policy_is_refused_for_every_fitting_option():
    from vaft.process.signal_processing import subtract_baseline
    with pytest.raises(ValueError, match="nan_policy"):
        subtract_baseline(_TIME, _DRIFTING, np.arange(10), fitting_opt="linear", nan_policy="drop")
