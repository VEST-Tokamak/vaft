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


def test_a_current_record_outside_the_range_is_an_error():
    """Review finding: an empty intersection used to yield one slice at 0.28 s."""
    t = grid(0.10, 0.20)
    ods = synthetic_ods(ip=current(t, onset=0.12, offset=0.18), t=t)
    with pytest.raises(ValueError, match="does not overlap"):
        MODULE._select_times(ods, "auto", TSTEP, None, None)


def test_a_product_without_plasma_current_fails_at_selection():
    from vaft.omas.plasma_timing import PlasmaTimingError

    t = grid()
    ods = synthetic_ods(slow=light(t))
    ods["magnetics.time"] = t  # a time axis but no current
    with pytest.raises(PlasmaTimingError, match="carries no plasma current"):
        MODULE._select_times(ods, "auto", TSTEP, None, None)


def test_the_provenance_comment_survives_the_constraint_builder():
    """Review finding: legacy.vfit_equilibrium_form_constraints overwrote the comment."""
    from omas import ODS

    from vaft.code.efit.legacy import annotate_constraint_equilibrium

    t = grid()
    ods = synthetic_ods(slow=light(t), ip=current(t))
    times, window = MODULE._select_times(ods, "auto", TSTEP, None, None)
    comment = MODULE._window_comment(window, times)

    eq = ODS(consistency_check=False)
    eq["ids_properties.comment"] = comment
    annotate_constraint_equilibrium(eq, times)
    assert eq["ids_properties.comment"].startswith(comment)
    assert eq["ids_properties.comment"].endswith("constraint equilibrium")
    assert eq["ids_properties.homogeneous_time"] == 1

    bare = ODS(consistency_check=False)
    annotate_constraint_equilibrium(bare, times)
    assert bare["ids_properties.comment"] == "constraint equilibrium"


# --- the vacuum cut (#76) ---------------------------------------------------


def _ip_ods(values, dt: float = DT):
    """A product carrying only the plasma current the cut is judged on."""
    from omas import ODS

    ods = ODS(consistency_check=False)
    clock = np.arange(len(values), dtype=float) * dt + 0.300
    ods["magnetics.ip.0.time"] = clock
    ods["magnetics.ip.0.data"] = np.asarray(values, dtype=float)
    return ods, clock


def test_instants_below_cutip_are_dropped_before_efit_sees_them():
    """EFIT answers a sub-CUTIP slice with a vacuum solution, not a failure.

    It writes an a-file whose `limloc` is `VAC` and a g-file to go with it, so
    a yield counted from produced files counts vacuum as success. Measured on
    39915: four of the seventeen instants that produced g-files were below the
    cut, so its plasma yield is thirteen of twenty-two, not seventeen of
    twenty-six.
    """
    ods, clock = _ip_ods([1_000.0] * 5 + [80_000.0] * 5 + [500.0] * 5)
    times = clock[::5] + 2 * DT  # one instant inside each plateau

    kept, dropped = MODULE._drop_vacuum_times(ods, times, average_window=DT)

    assert kept.size == 1
    assert [round(current / 1e3) for _, current in dropped] == [1, 0]


def test_the_cut_uses_the_same_box_average_the_constraint_builder_uses():
    """Judging on an instantaneous value would disagree at the boundary.

    Measured on 39915's 330 ms instant: 15.4 kA instantaneous, 14.9 kA
    box-averaged. The k-file carries the averaged number as `PLASMA`, so that
    is the one `CUTIP` will be compared against and the one this must use.
    """
    from vaft.code.efit.legacy import box_average

    # A spike above the cut for one sample and below it either side, plus a
    # plateau that is genuinely above it so something survives to keep.
    values = [8_000.0, 8_000.0, 40_000.0, 8_000.0, 8_000.0] + [80_000.0] * 5
    ods, clock = _ip_ods(values)
    spike, plateau = float(clock[2]), float(clock[7])

    instantaneous = values[2]
    averaged = box_average(clock, np.asarray(values), spike, 2 * DT, what="ip")
    assert instantaneous > 15_000.0 > averaged, "the fixture must straddle the cut"

    kept, dropped = MODULE._drop_vacuum_times(
        ods, np.asarray([spike, plateau]), average_window=2 * DT
    )
    # The spike goes, on the averaged value; the plateau stays.
    assert [round(value, 6) for value in kept] == [round(plateau, 6)]
    assert len(dropped) == 1 and dropped[0][1] == pytest.approx(averaged)


def test_a_window_where_nothing_reaches_cutip_is_reported_not_emptied():
    """Silently producing no slices would look like a clean run with no data."""
    ods, clock = _ip_ods([1_000.0] * 10)

    with pytest.raises(ValueError, match="would return a vacuum solution for all"):
        MODULE._drop_vacuum_times(ods, clock[::3], average_window=DT)


def test_a_product_without_a_plasma_current_keeps_every_instant():
    """Unknown is not a reason to drop: nothing was measured to drop it by."""
    from omas import ODS

    times = np.asarray([0.300, 0.301, 0.302])
    kept, dropped = MODULE._drop_vacuum_times(
        ODS(consistency_check=False), times, average_window=DT
    )
    np.testing.assert_allclose(kept, times)
    assert dropped == []


def test_the_product_says_how_many_instants_the_cut_removed():
    """A reader cannot otherwise tell a dropped instant from a failed one."""
    comment = MODULE._window_comment(
        None, np.asarray([0.306, 0.331]), [(0.306, 1_300.0), (0.330, 14_900.0)]
    )
    assert "2 instant(s) below CUTIP dropped before EFIT" in comment
    assert "0.3060s" in comment and "0.3300s" in comment

    assert "CUTIP" not in MODULE._window_comment(None, np.asarray([0.306, 0.331]))
