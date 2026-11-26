"""DCON stopped at a zero crossing is an ideal-instability verdict, not a broken run.

With ``termbycross_flag = t`` (the packaged ``dcon.in``) DCON stops its ODE
integration at the first zero crossing of the energy determinant, prints
``PROGRAM STOP => Terminated by zero crossing.`` and exits 0 -- having written
no ``dcon_output_n<n>.nc`` and possibly part of ``euler.bin``.  The suite used
to report that as a failed DCON and, for ideal GPEC, as an "invalid DCON
result" (a raise under ``strict``).  It is a physics result: the equilibrium
is ideally unstable to that n and there is no stable response to compute.
"""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from vaft.code import gpec
from vaft.code.gpec._solvers import dcon_zero_crossing, zero_crossing_reason
from vaft.compat import IS_WINDOWS

GFILE_TEXT = "  EFITD   01/01/2024   #  39915  325ms        3  65  65\n 1.0 2.0 3.0\n"

#: The tail of a real DCON v1.5.7 log for an ideally unstable n=1 case.
ZERO_CROSSING_LOG = """ DCON START => v1.5.7-611-ge68d7ac2
 Evaluating Mercier criterion
   q0 =  1.015E+00, qmin =  1.015E+00, qmax =  8.035E+00, q95 =  5.701E+00
 Starting integration of ODE's
   psi = 2.395E-02, q =  1.020
   psi = 5.836E-01, q =  2.000
   psi = 7.365E-01, q =  3.000
 Total cpu time =  2 seconds
 PROGRAM STOP => Terminated by zero crossing.
"""

NORMAL_LOG = """ Starting integration of ODE's
   psi = 5.836E-01, q =  2.000
 PROGRAM STOP => Normal termination.
"""


@pytest.fixture()
def case(tmp_path):
    geqdsk = tmp_path / "g039915.00325"
    geqdsk.write_text(GFILE_TEXT, encoding="utf-8")
    return gpec.GPECCaseInputs(shot=39915, time_ms=325, geqdsk=geqdsk, workdir=tmp_path / "run")


def _script(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _unstable_dcon(home: Path) -> Path:
    """A DCON that does what the real one does on an unstable equilibrium."""
    return _script(
        home / "bin" / "dcon",
        "cat <<'LOG'\n" + ZERO_CROSSING_LOG + "LOG\n"
        "printf partial > euler.bin\nprintf psi > psi_in.bin\nexit 0\n",
    )


# --- reading DCON's verdict --------------------------------------------------


def test_the_stop_and_the_last_surface_are_read_from_the_log(tmp_path):
    (tmp_path / "dcon.log").write_text(ZERO_CROSSING_LOG, encoding="utf-8")
    crossing = dcon_zero_crossing(tmp_path)
    assert crossing == {"log": "dcon.log", "psi": pytest.approx(0.7365), "q": pytest.approx(3.0)}
    reason = zero_crossing_reason(crossing)
    assert "zero crossing" in reason and "psi_n = 0.7365" in reason and "q = 3" in reason


def test_dcon_out_is_read_when_there_is_no_log(tmp_path):
    (tmp_path / "dcon.out").write_text(" PROGRAM STOP => Terminated by zero crossing.\n", encoding="utf-8")
    assert dcon_zero_crossing(tmp_path) == {"log": "dcon.out", "psi": None, "q": None}
    assert "ideally unstable" in zero_crossing_reason(dcon_zero_crossing(tmp_path))


def test_a_normal_termination_is_not_a_crossing(tmp_path):
    (tmp_path / "dcon.log").write_text(NORMAL_LOG, encoding="utf-8")
    assert dcon_zero_crossing(tmp_path) is None
    assert dcon_zero_crossing(tmp_path / "missing") is None


# --- how the suite classifies it ----------------------------------------------


@pytest.mark.skipif(IS_WINDOWS, reason="the stub is a POSIX shell script")
@pytest.mark.parametrize("verify_outputs", [True, False])
def test_dcon_is_unstable_and_gpec_is_skipped_even_under_strict(case, monkeypatch, tmp_path, verify_outputs):
    home = tmp_path / "gpec_home"
    _unstable_dcon(home)
    # The real chain runs match after the stop too, and it terminates normally.
    _script(home / "bin" / "match", "echo 'PROGRAM STOP => Normal termination.'\nexit 0\n")
    # Never launched: the skip comes before GPEC is started.
    _script(home / "bin" / "gpec", "echo launched > launched\nexit 0\n")
    monkeypatch.setenv(gpec.GPEC_HOME_ENV, str(home))

    config = gpec.GPECSuiteConfig(
        modules=("dcon", "gpec"), modes=(1,), run_mode="strict", verify_outputs=verify_outputs
    )
    result = gpec.run_gpec_suite_case(case, config)

    dcon_record, gpec_record = result.records
    assert dcon_record.status == "unstable", dcon_record.reason
    assert dcon_record.returncode == 0
    assert "zero crossing" in dcon_record.reason and "q = 3" in dcon_record.reason
    assert dcon_record.outputs == ()
    assert not dcon_record.ok  # no usable products
    assert gpec_record.status == "skipped"
    assert "ideally unstable" in gpec_record.reason
    assert result.returncode == 0  # a verdict, not a failure
    gpec_dir = gpec._module_dir(case.workdir, case.time_ms, "gpec", 1, geqdsk=case.geqdsk)
    assert not (gpec_dir / "launched").exists()
    assert not (gpec_dir / "euler.bin").exists()


@pytest.mark.skipif(IS_WINDOWS, reason="the stub is a POSIX shell script")
def test_a_crashed_dcon_that_mentions_nothing_is_still_a_failure(case, monkeypatch, tmp_path):
    home = tmp_path / "gpec_home"
    _script(home / "bin" / "dcon", "echo 'PROGRAM STOP => Normal termination.'\nexit 0\n")
    monkeypatch.setenv(gpec.GPEC_HOME_ENV, str(home))

    config = gpec.GPECSuiteConfig(modules=("dcon",), modes=(1,), run_mode="auto", verify_outputs=True)
    (record,) = gpec.run_gpec_suite_case(case, config).records
    assert record.status == "failed"
    assert "dcon_output_n1.nc" in record.reason


@pytest.mark.skipif(IS_WINDOWS, reason="the stub is a POSIX shell script")
def test_a_nonzero_exit_is_a_failure_whatever_the_log_says(case, monkeypatch, tmp_path):
    home = tmp_path / "gpec_home"
    _script(home / "bin" / "dcon", "cat <<'LOG'\n" + ZERO_CROSSING_LOG + "LOG\nexit 3\n")
    monkeypatch.setenv(gpec.GPEC_HOME_ENV, str(home))

    config = gpec.GPECSuiteConfig(modules=("dcon",), modes=(1,), run_mode="auto")
    (record,) = gpec.run_gpec_suite_case(case, config).records
    assert record.status == "failed"
