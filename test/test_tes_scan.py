"""Unit tests for TES parameter-scan argument validation.

``scan_tes`` feeds ``param`` straight into ``dataclasses.replace``, so a typo
used to surface as an opaque ``TypeError`` from deep inside the stdlib. It is
validated against ``TESConfig`` up front instead.
"""

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
    assert result.returncode == 124
    assert "timed out after 10.0 seconds" in result.stderr
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
    result = run_tes(_tes_case(tmp_path), config)

    (request,) = backend.requests
    assert list(request.command[1:]) == ["-f5", "rtes.in", "-rrestart.dat"]
    assert request.workdir == tmp_path
    assert request.env == {"TES_FLAG": "1"}
    assert request.timeout == 30.0
    assert (result.returncode, result.stdout) == (0, "done")


def test_run_tes_maps_a_backend_timeout_to_124(tmp_path):
    import sys

    from external_code_stubs import RecordingBackend
    from vaft.code.execution import ExecutionResult
    from vaft.code.tes.runner import run_tes

    backend = RecordingBackend(
        ExecutionResult(returncode=None, stdout="", stderr="", timed_out=True)
    )
    config = TESConfig(executable=sys.executable, timeout=2.0, backend=backend)
    result = run_tes(_tes_case(tmp_path), config)
    assert result.returncode == 124
    assert result.stderr == "rtes timed out after 2.0 seconds"
