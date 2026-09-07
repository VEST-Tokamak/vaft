"""The #171 baseline's log reading, on synthetic logs with known answers.

The study itself needs EFIT and a shot; what is pinned here is the parsing it
rests on, because every conclusion #171 draws is a claim about what the log
said. The two readings that mislead if taken at face value each get a test.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "workflow" / "efit_numerics" / "baseline_termination.py"


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location("baseline_termination", SCRIPT)
    loaded = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = loaded
    try:
        spec.loader.exec_module(loaded)
    except Exception:
        del sys.modules[spec.name]
        raise
    yield loaded
    sys.modules.pop(spec.name, None)


CONVERGED = """
 r=  0 t=   319 it=  1 chi2=2.07E+02 zm= 5.96E-08 err=4.273E+00 dz=-4.418E-03 chigam= 0.00E+00
 r=  0 t=   319 it=  2 chi2=1.41E+02 zm= 1.66E-07 err=6.815E-01 dz= 1.062E-07 chigam= 0.00E+00
 r=  0 t=   319 it=  3 chi2=7.44E+01 zm= 1.70E-07 err=8.523E-03 dz= 1.062E-07 chigam= 0.00E+00
WARNING in fit at r=  0, t=   319: iconvr=2 satisfied, exiting
  Failed to reach fit/convergence criteria, shot   39915   319. msec. all eqdsks are still written
     Failure #20, Bp+li/2 not consistent, error =  0.586E-01
     Failure #21, Bp not consistent, error =  0.145E+01
"""

COLLAPSED = """
 r=  0 t=   307 it=  1 chi2=9.45E+01 zm= 1.40E-08 err=1.893E+01 dz= 1.403E-08 chigam= 0.00E+00
 r=  0 t=   307 it=  2 chi2=9.19E-08 zm= 3.57E-08 err=5.014E-01 dz= 2.164E-08 chigam= 0.00E+00
 r=  0 t=   307 it=  3 chi2=9.18E-08 zm= 2.49E-07 err=6.820E-01 dz= 1.052E-07 chigam= 0.00E+00
"""

BOUNDARY_ERROR = """
 r=  0 t=   323 it=  1 chi2=1.06E+01 zm= 0.00E+00 err=9.333E-03 dz= 0.00E+00 chigam= 0.00E+00
ERROR in findax at r=  0, t=   323: 1st separatrix point is off grid
"""


def test_slices_are_delimited_by_the_iteration_counter_not_the_time(module):
    """EFIT prints whole milliseconds, so two slices can share a printed time."""
    text = CONVERGED + COLLAPSED + BOUNDARY_ERROR
    slices = module.parse_slices(text)
    assert [item["time_ms"] for item in slices] == [319, 307, 323]
    assert [item["iterations_n"] for item in slices] == [3, 3, 1]


def test_the_exit_path_and_the_failures_are_read_by_number(module):
    slice_ = module.parse_slices(CONVERGED)[0]
    assert slice_["exit_path"] == "iconvr=2" and slice_["iconvr"] == 2
    assert slice_["accepted"] is False
    assert [failure["code"] for failure in slice_["failures"]] == [20, 21]
    assert slice_["failures"][1]["criterion"].startswith("Bp not consistent")
    assert slice_["failures"][1]["value"] == pytest.approx(1.45)


def test_the_log_chi_square_is_not_reported_as_the_fit_chi_square(module):
    """After the first step it collapses on slices whose a-file reports 200.

    Keeping both ends under their own names is what stops a summary claiming
    convergence where there is none.
    """
    slice_ = module.parse_slices(CONVERGED)[0]
    assert slice_["chi2_initial"] == pytest.approx(207.0)
    assert slice_["chi2_final"] == pytest.approx(74.4)
    assert "chi2" not in slice_


def test_a_null_solution_is_recognised_and_not_counted_as_a_fit(module):
    """Residual below 1e-5 while the Grad-Shafranov error stays above 0.1."""
    collapsed = module.parse_slices(COLLAPSED)[0]
    assert collapsed["collapsed"] is True
    assert collapsed["chi2_final"] < 1e-5 and collapsed["gs_error"] > 0.1

    fitted = module.parse_slices(CONVERGED)[0]
    assert fitted["collapsed"] is False


def test_a_solver_error_is_its_own_exit_path(module):
    slice_ = module.parse_slices(BOUNDARY_ERROR)[0]
    assert slice_["exit_path"] == "solver_error"
    assert slice_["solver_errors"][0]["routine"] == "findax"
    assert "separatrix" in slice_["solver_errors"][0]["detail"]
    assert slice_["accepted"] is False


def test_an_empty_log_yields_no_slices_rather_than_a_crash(module):
    assert module.parse_slices("") == []
    assert module.parse_slices("nothing to see here\n") == []
