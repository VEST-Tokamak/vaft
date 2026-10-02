"""Unit tests for TES parameter-scan argument validation.

``scan_tes`` feeds ``param`` straight into ``dataclasses.replace``, so a typo
used to surface as an opaque ``TypeError`` from deep inside the stdlib. It is
validated against ``TESConfig`` up front instead.
"""

import os
import time

import pytest

from vaft.code.tes import TESConfig, scan_tes


def test_scan_rejects_unknown_param():
    with pytest.raises(ValueError, match="Unknown TESConfig field"):
        scan_tes(
            ods=None,
            base_config=TESConfig(),
            values=[100.0],
            param="ip0_ka",  # correct spelling is ip0_kA
        )


def test_scan_error_lists_valid_fields():
    with pytest.raises(ValueError) as excinfo:
        scan_tes(ods=None, base_config=TESConfig(), values=[100.0], param="nonsense")

    message = str(excinfo.value)
    assert "ip0_kA" in message
    assert "betap" in message


def test_scan_validates_before_touching_the_ods():
    # ods=None would fail loudly downstream; the ValueError proves validation
    # happens before any solve is attempted.
    with pytest.raises(ValueError):
        scan_tes(ods=None, base_config=TESConfig(), values=[], param="bogus")


def test_run_tes_handles_timeout_gracefully(monkeypatch, tmp_path):
    import subprocess
    import sys
    from vaft.code.tes.config import TESInputs
    from vaft.code.tes.runner import run_tes

    def mock_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["rtes"], timeout=10.0, stderr="divergence detected")

    monkeypatch.setattr(subprocess, "run", mock_run)

    dummy_cinput = tmp_path / "rtes.in"
    dummy_cinput.write_text("DUMMY")
    inputs = TESInputs(workdir=tmp_path, cinput=dummy_cinput)
    cfg = TESConfig(executable=sys.executable, timeout=10.0)

    result = run_tes(inputs, cfg)
    assert not result.ok
    assert (result.status, result.runtime_status, result.returncode) == ("failed", "timeout", None)
    # The patched stop is instant, well short of the 10 s limit, so the reason
    # reports the time actually run rather than claiming the limit.
    assert "rtes timed out after " in result.stderr and "10 s" not in result.stderr
    assert "divergence detected" in result.stderr


def _tes_case(tmp_path):
    from vaft.code.tes.config import TESInputs

    cinput = tmp_path / "rtes.in"
    cinput.write_text("DUMMY")
    return TESInputs(workdir=tmp_path, cinput=cinput)


def test_run_tes_builds_its_command_for_the_configured_backend(tmp_path):
    import sys

    from external_code_stubs import RecordingBackend
    from vaft.code.execution import ExecutionResult
    from vaft.code.tes.runner import run_tes

    backend = RecordingBackend(ExecutionResult(returncode=0, stdout="done", stderr=""))
    config = TESConfig(
        executable=sys.executable,
        niter=5,
        restart="restart.dat",
        env={"TES_FLAG": "1"},
        timeout=30.0,
        backend=backend,
    )
    gfile = tmp_path / "g039915.00325"
    record = backend.run

    def run_and_write(request):   # a converged rtes leaves a g-file behind
        gfile.write_text("stub")
        return record(request)

    backend.run = run_and_write
    result = run_tes(_tes_case(tmp_path), config)

    (request,) = backend.requests
    assert list(request.command[1:]) == ["-f5", "rtes.in", "-rrestart.dat"]
    assert request.workdir == tmp_path
    assert request.env == {"TES_FLAG": "1"}
    assert request.timeout == 30.0
    assert (result.returncode, result.stdout) == (0, "done")


def test_run_tes_without_a_gfile_is_a_failed_result(tmp_path):
    # rtes exits 0 when its Picard loop diverges and writes no equilibrium
    import sys

    from external_code_stubs import RecordingBackend
    from vaft.code.execution import ExecutionResult
    from vaft.code.tes.runner import run_tes

    backend = RecordingBackend(ExecutionResult(returncode=0, stdout="[Warn] out of range", stderr=""))
    result = run_tes(_tes_case(tmp_path), TESConfig(executable=sys.executable, backend=backend))

    assert not result.ok
    assert result.returncode == 1
    assert "wrote no g-file" in result.stderr


def test_run_tes_ignores_a_gfile_left_by_an_earlier_run(tmp_path):
    # rtes writes no g-file when it fails; the previous run's file in a reused
    # workdir must not be reported as this run's equilibrium
    import sys

    from external_code_stubs import RecordingBackend
    from vaft.code.execution import ExecutionResult
    from vaft.code.tes.runner import run_tes

    (tmp_path / "g039915.00325").write_text("stub")   # from an earlier run
    backend = RecordingBackend(ExecutionResult(returncode=0, stdout="", stderr=""))

    result = run_tes(_tes_case(tmp_path), TESConfig(executable=sys.executable, backend=backend))

    assert result.gfile is None
    assert not result.ok
    assert "wrote no g-file" in result.stderr


def test_run_tes_finds_its_gfile_among_older_ones(tmp_path):
    # a reused workdir: g...00320 from an earlier run sorts before this run's
    # g...00325 and must not shadow it
    import sys

    from external_code_stubs import RecordingBackend
    from vaft.code.execution import ExecutionResult
    from vaft.code.tes.runner import run_tes

    (tmp_path / "g039915.00320").write_text("older")
    backend = RecordingBackend(ExecutionResult(returncode=0, stdout="", stderr=""))
    record = backend.run

    def run_and_write(request):
        (tmp_path / "g039915.00325").write_text("this run")
        return record(request)

    backend.run = run_and_write
    result = run_tes(_tes_case(tmp_path), TESConfig(executable=sys.executable, backend=backend))

    assert result.ok
    assert result.gfile.name == "g039915.00325"


def test_run_tes_returns_a_backend_timeout_as_a_failed_result(tmp_path):
    import sys

    from external_code_stubs import RecordingBackend
    from vaft.code.execution import ExecutionResult
    from vaft.code.tes.runner import run_tes

    backend = RecordingBackend(
        ExecutionResult(returncode=None, stdout="", stderr="", timed_out=True, elapsed_s=2.0)
    )
    config = TESConfig(executable=sys.executable, timeout=2.0, backend=backend)
    result = run_tes(_tes_case(tmp_path), config)
    assert (result.status, result.runtime_status, result.returncode) == ("failed", "timeout", None)
    assert result.elapsed_s == 2.0
    assert result.stderr == "rtes timed out after 2 s of running"
