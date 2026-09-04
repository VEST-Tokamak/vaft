"""EFIT constraint times from the shared plasma range and the detected window (issue #409).

The script used to take a fixed 0.28-0.38 s and keep the samples above 20 kA;
it now intersects the configured ``plasma_analysis`` range with the plasma
window ``vaft.omas.plasma_timing`` finds, and says so in the product.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from _plasma_timing_fixtures import DT, current, grid, light, pickup_only, synthetic_ods
from vaft.machine_mapping.utils import resolve_plasma_timing_policy

SCRIPT = (
    Path(__file__).parents[1]
    / "workflow/automatic_pipeline_1_routine_data_processing/generate_constraints_ods.py"
)
SPEC = importlib.util.spec_from_file_location("generate_constraints_ods_time_selection", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

TSTEP = 1e-3


def _snap(value: float, tstep: float = TSTEP) -> float:
    return round(value / tstep) * tstep


def test_auto_mode_cuts_the_slices_from_the_detected_plasma_window():
    t = grid()
    ods = synthetic_ods(slow=light(t), ip=current(t))

    times, window = MODULE._select_times(ods, "auto", TSTEP, None, None)

    assert window.source == "h_alpha_primary"
    assert not window.fallback
    assert window.agreement == "consistent"
    assert window.start == pytest.approx(0.306, abs=3e-4)
    assert window.end == pytest.approx(0.331, abs=5e-4)
    assert times[0] == pytest.approx(_snap(window.start))
    assert times[-1] == pytest.approx(_snap(window.end))  # the end is included
    np.testing.assert_allclose(np.diff(times), TSTEP)
    assert "analysis-range fallback" not in MODULE._window_comment(window, times)
    assert "from h_alpha_primary" in MODULE._window_comment(window, times)


def test_no_plasma_uses_the_whole_range_and_says_so():
    t = grid()
    ods = synthetic_ods(slow=light(t, amplitude=0.0), ip=pickup_only(t))
    policy = resolve_plasma_timing_policy()

    times, window = MODULE._select_times(ods, "auto", TSTEP, None, None)

    assert window.fallback
    assert window.source == "analysis_range"
    assert window.start == pytest.approx(policy.window.tstart)
    assert window.end == pytest.approx(float(t[-1]))  # the grid ends before tend
    assert "no_plasma_timing" in window.flags
    assert window.fallback_reason and "ip_principal" in window.fallback_reason
    assert times[0] == pytest.approx(0.28)
    comment = MODULE._window_comment(window, times)
    assert "analysis-range fallback" in comment and "from analysis_range" in comment


def test_the_range_is_the_configured_window_not_the_old_literals():
    """0.28-0.38 s with an Ip > 20 kA gate is gone: the upper bound is the shared 0.36 s."""
    t = grid(0.26, 0.40)
    ods = synthetic_ods(slow=light(t, onset=0.29, offset=0.39), ip=current(t, onset=0.29, offset=0.39), t=t)

    _, window = MODULE._select_times(ods, "auto", TSTEP, None, None)

    assert window.end == pytest.approx(0.36)
    assert window.start == pytest.approx(0.29, abs=3e-4)


def test_manual_mode_is_unchanged_and_carries_no_window():
    t = grid()
    ods = synthetic_ods(slow=light(t), ip=current(t))

    times, window = MODULE._select_times(ods, "manual", TSTEP, 0.30, 0.34)

    assert window is None
    np.testing.assert_allclose(times, np.arange(0.30, 0.34, TSTEP))
    assert MODULE._window_comment(window, times).endswith(": manual")
    with pytest.raises(ValueError, match="manual timeset requires"):
        MODULE._select_times(ods, "manual", TSTEP, None, None)


def test_bounds_clamp_the_window_and_snap_to_the_step():
    t = grid()
    ods = synthetic_ods(slow=light(t), ip=current(t))

    times, window = MODULE._select_times(ods, "auto", 2.5e-3, 0.31, 0.32)

    assert times[0] == pytest.approx(_snap(0.31, 2.5e-3))
    assert times[-1] == pytest.approx(_snap(0.32, 2.5e-3))
    assert window.start < 0.31 < window.end  # the clamp narrowed a real window


def test_a_product_without_the_filterscope_is_timed_from_the_current():
    t = grid()
    ods = synthetic_ods(ip=current(t))

    times, window = MODULE._select_times(ods, "auto", TSTEP, None, None)

    assert window.source == "ip_principal"
    assert not window.fallback
    assert window.start == pytest.approx(0.3068, abs=1e-3)
    assert "h_alpha_primary: present" in window.fallback_reason


def test_an_empty_current_axis_is_still_an_error():
    from omas import ODS

    ods = ODS(consistency_check=False)
    ods["magnetics.ip.0.time"] = np.zeros(0)
    ods["magnetics.ip.0.data"] = np.zeros(0)
    with pytest.raises(ValueError, match="magnetics.ip.0.time is empty"):
        MODULE._select_times(ods, "auto", TSTEP, None, None)
