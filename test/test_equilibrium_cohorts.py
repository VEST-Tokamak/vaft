"""What failed, in which operating regime (#76).

The join of #820's regime labels and #845's termination evidence. These pin
the join rule -- by time, never by position -- and the three ways it is
allowed to be incomplete.
"""

from __future__ import annotations

import numpy as np
import pytest
from omas import ODS

from vaft.code.efit import parse_slices
from vaft.validation import (
    FAILURE_MODES,
    join_regime_and_termination,
    summarize_by_cohort,
)


def _reconstruction(currents, *, start=0.310, step=1e-3, limloc=None):
    ods = ODS(consistency_check=False)
    ods["equilibrium.time"] = np.arange(len(currents)) * step + start
    for index, value in enumerate(currents):
        ods[f"equilibrium.time_slice.{index}.constraints.ip.measured"] = float(value)
        if limloc is not None:
            ods[
                f"equilibrium.code.parameters.time_slice.{index}.aeqdsk.limloc"
            ] = limloc[index]
    return ods


def _converged(t):
    return [
        f" r=  0 t=   {t} it=  1 chi2=8.7E+01 zm= 0.0 err=7.6E+01 dz= 0.0",
        f" r=  0 t=   {t} it=  2 chi2=4.1E+00 zm= 1.2E-02 err=2.2E+00 dz= 0.0",
        f" WARNING in fit at r=  0, t=   {t}: iconvr=2 satisfied, exiting",
    ]


def _lost_boundary(t):
    return [
        f" r=  0 t=   {t} it=  1 chi2=8.7E+01 zm= 0.0 err=7.6E+01 dz= 0.0",
        f" WARNING in findax at r=  0, t=   {t}: 2nd separatrix point is off grid",
        f"ERROR in bound at r=  0, t=   {t}: First and last contour points are too far apart",
    ]


# --- the join rule ----------------------------------------------------------


def test_the_join_is_keyed_on_time_and_not_on_position():
    """This repository's most repeated defect, in the one place it is invisible.

    EFIT's log names slices by whole milliseconds and the reconstruction names
    them in seconds, and the two lists agree in length whenever every slice
    reached the solver. Zipping them works until one does not -- and then it
    silently attributes every later failure to the wrong instant.
    """
    ods = _reconstruction([100_000, 100_000, 100_000])
    # The log is missing the FIRST slice, so position and time disagree by one.
    text = "\n".join(_converged(311) + _lost_boundary(312))

    join = join_regime_and_termination(ods, parse_slices(text))

    assert [item.time for item in join] == pytest.approx([0.310, 0.311, 0.312])
    assert [item.mode for item in join] == ["unknown", "accepted", "boundary_failure"]


def test_a_reconstruction_slice_the_log_has_nothing_for_is_unknown():
    """Not `accepted`. Nothing ran, or nothing was recorded; either way we do
    not know, and a cohort that counts it as a success is wrong in the
    direction that flatters the reconstruction."""
    join = join_regime_and_termination(_reconstruction([100_000]), [])

    assert len(join) == 1
    assert join.outcomes[0].mode == "unknown"
    assert join.outcomes[0].termination is None
    assert "unknown" in FAILURE_MODES


def test_a_log_slice_the_reconstruction_has_nothing_for_is_reported():
    """Dropping it silently would hide a run against a different time base."""
    ods = _reconstruction([100_000, 100_000])
    text = "\n".join(_converged(310) + _converged(311) + _converged(999))

    join = join_regime_and_termination(ods, parse_slices(text))

    assert len(join) == 2
    assert join.unmatched_log_times == (999,)


def test_two_slices_on_one_millisecond_are_refused():
    """EFIT's log has no more resolution than a millisecond.

    Keeping one of them would attribute a failure to the wrong slice, and the
    reconstruction cadence is the caller's to choose, so this is reported
    rather than resolved.
    """
    ods = _reconstruction([100_000, 100_000], step=1e-4)  # 0.1 ms apart

    with pytest.raises(ValueError, match="same millisecond"):
        join_regime_and_termination(ods, parse_slices("\n".join(_converged(310))))


# --- the mode ---------------------------------------------------------------


def test_a_collapse_is_named_ahead_of_the_solver_error_it_causes():
    """A slice that falls to a null solution and then fails in `bound` failed
    because it collapsed. Calling it a boundary failure sends a reader after
    the boundary tracer, which was working."""
    text = "\n".join(
        [
            " r=  0 t=   310 it=  1 chi2=6.9E-08 zm= 0.0E+00 err=3.3E-01 dz= 0.0",
            "ERROR in bound at r=  0, t=   310: Less than 3 contour points found",
        ]
    )
    join = join_regime_and_termination(_reconstruction([100_000]), parse_slices(text))

    assert join.outcomes[0].mode == "collapsed"
    assert join.outcomes[0].solver_routines == ("bound",)


def test_bound_and_findax_get_their_own_names():
    """The distinction a recovery strategy turns on, so it is not lumped."""
    ods = _reconstruction([100_000, 100_000])
    text = "\n".join(
        _lost_boundary(310)
        + [
            " r=  0 t=   311 it=  1 chi2=8.7E+01 zm= 0.0 err=7.6E+01 dz= 0.0",
            "ERROR in findax at r=  0, t=   311: Iterative method reached max iterations",
        ]
    )
    modes = [item.mode for item in join_regime_and_termination(ods, parse_slices(text))]
    assert modes == ["boundary_failure", "axis_failure"]


def test_every_mode_is_one_of_the_declared_ones():
    ods = _reconstruction([100_000, 60_000, 5_000])
    text = "\n".join(_converged(310) + _lost_boundary(311))
    for item in join_regime_and_termination(ods, parse_slices(text)):
        assert item.mode in FAILURE_MODES


# --- the summary ------------------------------------------------------------


def test_the_cohorts_say_whether_failures_sort_by_phase():
    """The question #76 exists to ask, on the shape 46742 actually showed."""
    currents = [20_000, 60_000, 100_000, 98_000, 95_000, 60_000, 30_000, 5_000]
    limloc = ["IN" if value >= 90_000 else "" for value in currents]
    ods = _reconstruction(currents, limloc=limloc)

    lines = []
    for index, value in enumerate(currents):
        if value < 15_000:
            continue
        lines += _converged(310 + index) if value >= 90_000 else _lost_boundary(310 + index)

    join = join_regime_and_termination(ods, parse_slices("\n".join(lines)))
    summary = summarize_by_cohort(join, group_by=("phase",))

    assert summary["slices"] == 8
    assert summary["cohorts"]["flat"]["modes"] == {"accepted": 3}
    assert summary["cohorts"]["ramp_up"]["modes"] == {"boundary_failure": 2}
    assert summary["cohorts"]["ramp_down"]["modes"] == {"boundary_failure": 2}
    # The sub-CUTIP slice has no phase to blame and no log line; it is neither
    # a success nor a failure of the solver.
    assert summary["cohorts"]["vacuum"]["modes"] == {"unknown": 1}


def test_the_warning_tally_is_reported_apart_from_the_mode():
    """They answer different questions and on VEST they disagree.

    The mode says which routine gave up -- `bound`. The warnings say what the
    solver had been complaining about on the way -- `findax`, unable to use
    the second separatrix. A cohort dominated by one can have its modes name
    the other, which is the case on 46742 and is worth being able to see.
    """
    ods = _reconstruction([60_000, 60_000])
    join = join_regime_and_termination(
        ods, parse_slices("\n".join(_lost_boundary(310) + _lost_boundary(311)))
    )
    cohort = summarize_by_cohort(join, group_by=("phase",))["cohorts"]

    only = next(iter(cohort.values()))
    assert only["solver_routines"] == {"bound": 2}
    assert only["warning_routines"] == {"findax": 2}


def test_cohorts_can_be_cut_on_more_than_one_field():
    currents = [100_000, 100_000]
    ods = _reconstruction(currents, limloc=["IN", ""])
    join = join_regime_and_termination(
        ods, parse_slices("\n".join(_converged(310) + _converged(311)))
    )
    summary = summarize_by_cohort(join, group_by=("phase", "topology"))

    assert summary["group_by"] == ["phase", "topology"]
    assert set(summary["cohorts"]) == {"flat|inboard_limited", "flat|unknown"}


def test_an_unknown_grouping_field_is_refused():
    join = join_regime_and_termination(_reconstruction([100_000]), [])
    with pytest.raises(ValueError, match="no 'ip_bin' to group by"):
        summarize_by_cohort(join, group_by=("ip_bin",))
    with pytest.raises(ValueError, match="at least one field"):
        summarize_by_cohort(join, group_by=())


def test_the_record_is_json_safe_and_carries_both_halves():
    import json

    join = join_regime_and_termination(
        _reconstruction([100_000], limloc=["IN"]),
        parse_slices("\n".join(_converged(310))),
    )
    record = join.outcomes[0].as_dict()
    json.dumps(record)

    assert record["mode"] == "accepted"
    assert record["regime"]["phase"] == "flat"
    assert record["regime"]["topology"] == "inboard_limited"
    assert record["exit_path"] == "iconvr=2"
    assert record["iterations_n"] == 2
