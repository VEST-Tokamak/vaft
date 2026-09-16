"""EFIT's own log as a readable record (#76, #171).

The reader used to live in `workflow/efit_numerics/baseline_termination.py`
and was reachable only by loading that script by path. These pin the move and
the one thing the move adds.
"""

from __future__ import annotations

import pytest

from vaft.code.efit import parse_slices
from vaft.code.efit.termination import EFIT_LOG_PATTERNS

# A slice that warns and is still accepted. The `iconvr` line is EFIT's own
# spelling -- it announces the chi-square exit as a warning -- and is read as
# the exit path, so the findax line is the only actual warning here.
CONVERGED = """
 r=  0 t=   319 it=  1 chi2=8.7E+01 zm= 3.2E-08 err=7.6E+01 dz=-5.0E-02
 WARNING in findax at r=  0, t=   319: 2nd separatrix point is off grid
 r=  0 t=   319 it=  2 chi2=4.1E+00 zm= 1.2E-02 err=2.2E+00 dz= 9.1E-03
 WARNING in fit at r=  0, t=   319: iconvr=2 satisfied, exiting
"""

PRE_ITERATION_COLLAPSE = """
 r=  0 t=   315 it=  1 chi2=6.9E-08 zm= 0.0E+00 err=3.3E-01 dz= 0.0E+00
 WARNING in findax at r=  0, t=   315: 2nd separatrix point is off grid
ERROR in bound at r=  0, t=   315: First and last contour points are too far apart
ERROR in bound at r=  0, t=   316: Less than 3 contour points found
ERROR in bound at r=  0, t=   317: First and last contour points are too far apart
"""


def test_the_reader_is_importable_from_the_package():
    """The point of the move: three studies had to load a script by path."""
    import vaft.code.efit as package

    assert package.parse_slices is parse_slices
    assert set(EFIT_LOG_PATTERNS) == {
        "iteration", "iconvr", "failed", "failure", "solver_error", "solver_warning",
    }


def test_the_workflow_name_still_resolves_to_the_same_function():
    """Three studies and their stored tables are written against it."""
    import importlib.util
    import sys
    from pathlib import Path

    script = (
        Path(__file__).resolve().parents[1]
        / "workflow" / "efit_numerics" / "baseline_termination.py"
    )
    spec = importlib.util.spec_from_file_location("baseline_termination_reexport", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules["baseline_termination_reexport"] = module
    spec.loader.exec_module(module)

    assert module.parse_slices is parse_slices


def test_a_solver_error_opens_a_slice_but_a_warning_does_not():
    """The distinction the move had to preserve, and the one it adds.

    A slice that fails in `bound` before its first Picard iteration prints no
    iteration line, so an error has to open a slice or that slice is lost and
    its errors are handed to its predecessor.

    A warning must not: EFIT prints them and carries on, so a warning for a
    time no slice has started yet belongs to the slice about to begin.
    Opening one would invent a slice that never ran.
    """
    slices = parse_slices(PRE_ITERATION_COLLAPSE)

    assert [item["time_ms"] for item in slices] == [315, 316, 317]
    assert [len(item["solver_errors"]) for item in slices] == [1, 1, 1]
    # The warning stays with 315, where it was printed, and does not make a
    # fourth slice.
    assert [len(item["warnings"]) for item in slices] == [1, 0, 0]
    assert slices[0]["warnings"][0]["routine"] == "findax"


def test_findax_warnings_are_kept_because_they_are_the_evidence():
    """What the workflow original discarded.

    It matched `ERROR in <routine>` only. But `findax` reports a second
    separatrix it cannot use as a *warning* -- off grid, or outside the vessel
    -- and on shot 46742 those outnumbered the errors two to one on the slices
    that failed. They say why a boundary could not be formed, which is exactly
    what a failure-mode cohort needs.
    """
    text = """
 r=  0 t=   320 it=  1 chi2=1.0E+00 zm= 0.0 err=1.0E+00 dz= 0.0
 WARNING in findax at r=  0, t=   320: 2nd separatrix point is off grid
 WARNING in findax at r=  0, t=   320: 2nd separatrix point is not inside vessel, zeross.le.0.1
"""
    slice_ = parse_slices(text)[0]
    assert [item["routine"] for item in slice_["warnings"]] == ["findax", "findax"]
    assert "off grid" in slice_["warnings"][0]["detail"]
    assert "not inside vessel" in slice_["warnings"][1]["detail"]


def test_a_warning_does_not_make_a_slice_unaccepted():
    """EFIT warns and carries on; a warned slice can still be accepted.

    The `iconvr` line is not counted among them: EFIT announces the
    chi-square exit as `WARNING in fit ... iconvr=2 satisfied`, and reading
    that as a warning would both lose the exit path and make every converged
    slice look warned.
    """
    slice_ = parse_slices(CONVERGED)[0]

    assert [item["routine"] for item in slice_["warnings"]] == ["findax"]
    assert slice_["solver_errors"] == []
    assert slice_["failures"] == []
    assert slice_["accepted"] is True
    assert slice_["exit_path"] == "iconvr=2" and slice_["iconvr"] == 2


def test_both_ends_of_the_log_chi_square_are_kept_and_neither_is_called_the_answer():
    """On a collapsed slice the log's chi2 falls to ~1e-7 while the a-file says 200.

    So the reader reports the first and the last and names neither "the"
    chi-square; EFIT's own answer is the a-file's.
    """
    slice_ = parse_slices(CONVERGED)[0]
    assert slice_["chi2_initial"] == pytest.approx(87.0)
    assert slice_["chi2_final"] == pytest.approx(4.1)
    assert slice_["iterations_n"] == 2
    assert "iterations" not in slice_, "the per-iteration list is not part of the record"


def test_a_null_solution_is_named_rather_than_counted_as_a_fit():
    """The residual vanishes while the Grad-Shafranov error does not."""
    collapsed = parse_slices(
        " r=  0 t=   315 it=  1 chi2=6.9E-08 zm= 0.0E+00 err=3.3E-01 dz= 0.0E+00\n"
    )[0]
    assert collapsed["collapsed"] is True

    genuine = parse_slices(CONVERGED)[0]
    assert genuine["collapsed"] is False


def test_an_empty_log_reads_as_no_slices_rather_than_raising():
    assert parse_slices("") == []
    assert parse_slices("nothing EFIT would ever print\n") == []
