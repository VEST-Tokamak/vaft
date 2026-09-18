"""The shared execution backend every external-code adapter launches through.

Stub programs only (``#!/bin/sh`` or ``.cmd``, see ``external_code_stubs``):
these tests pin the launch contract -- environment overlay, captured versus
logged output, stdin, thread declaration, timeout and launch failure -- without
any physics code installed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from external_code_stubs import (
    RecordingBackend,
    write_launchable_stub,
    write_unlaunchable_file,
)
from vaft.code import (
    ExecutableNotLaunchable,
    ExecutionBackend,
    ExecutionRequest,
    ExecutionResult,
    LocalBackend,
    ResourceRequest,
    resolve_backend,
)
from vaft.code.execution import THREAD_ENV_VARIABLES, execution_environment


def _python(tmp_path: Path, code: str, **request) -> ExecutionResult:
    return LocalBackend().run(
        ExecutionRequest(command=(sys.executable, "-c", code), workdir=tmp_path, **request)
    )


def test_captures_stdout_stderr_and_exit_code(tmp_path):
    result = _python(
        tmp_path,
        "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)",
    )
    assert result.returncode == 3
    assert result.stdout.strip() == "out"
    assert result.stderr.strip() == "err"
    assert not result.timed_out
    assert result.launcher[0] == sys.executable
    assert result.log_path is None


def test_runs_in_the_requested_workdir(tmp_path):
    result = _python(tmp_path, "import os; print(os.getcwd())")
    assert Path(result.stdout.strip()).resolve() == tmp_path.resolve()


def test_env_is_an_overlay_on_the_inherited_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("VAFT_INHERITED", "kept")
    result = _python(
        tmp_path,
        "import os; print(os.environ['VAFT_INHERITED'], os.environ['VAFT_OVERLAY'])",
        env={"VAFT_OVERLAY": "added"},
    )
    assert result.stdout.split() == ["kept", "added"]


def test_stdin_is_fed_to_the_program(tmp_path):
    result = _python(tmp_path, "import sys; print(sys.stdin.read().upper())", stdin="2\n1\n")
    assert result.stdout.split() == ["2", "1"]


def test_log_path_merges_both_streams_into_the_file(tmp_path):
    log = tmp_path / "logs" / "run.log"
    result = _python(
        tmp_path,
        "import sys; print('out', flush=True); print('err', file=sys.stderr)",
        log_path=log,
    )
    assert result.returncode == 0
    assert result.log_path == log
    assert (result.stdout, result.stderr) == ("", "")
    assert log.read_text(encoding="utf-8").split() == ["out", "err"]


def test_timeout_is_returned_with_partial_output(tmp_path):
    result = _python(
        tmp_path,
        "import sys, time; print('started', flush=True); time.sleep(30)",
        timeout=5.0,
    )
    assert result.timed_out
    assert result.returncode is None
    assert "started" in result.stdout


def test_timeout_decodes_bytes_the_stdlib_hands_back(tmp_path, monkeypatch):
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=args[0], timeout=1.0, output=b"partial \xff", stderr=b"diverged"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = _python(tmp_path, "pass", timeout=1.0)
    assert result.timed_out
    assert result.stdout == "partial �"
    assert result.stderr == "diverged"


def test_launchable_stub_exit_code_is_reported(tmp_path):
    program = write_launchable_stub(tmp_path / "solver", exit_code=7)
    result = LocalBackend().run(ExecutionRequest(command=(str(program),), workdir=tmp_path))
    assert result.returncode == 7


def test_unlaunchable_program_raises_the_shared_error(tmp_path):
    program = write_unlaunchable_file(tmp_path / "solver")
    with pytest.raises(ExecutableNotLaunchable) as raised:
        LocalBackend().run(ExecutionRequest(command=(str(program),), workdir=tmp_path))
    assert isinstance(raised.value.__cause__, OSError)


def test_bad_log_path_is_not_reported_as_unlaunchable(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("", encoding="utf-8")
    with pytest.raises(OSError) as raised:
        _python(tmp_path, "pass", log_path=blocker / "run.log")
    assert not isinstance(raised.value, ExecutableNotLaunchable)


def test_threads_fill_only_variables_the_environment_leaves_unset(tmp_path, monkeypatch):
    for variable in THREAD_ENV_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    request = ExecutionRequest(
        command=("true",),
        workdir=tmp_path,
        env={"MKL_NUM_THREADS": "4"},
        resources=ResourceRequest(threads_per_task=1),
    )
    environment = execution_environment(request)
    assert environment["OMP_NUM_THREADS"] == "8"
    assert environment["MKL_NUM_THREADS"] == "4"
    assert environment["OPENBLAS_NUM_THREADS"] == "1"


def test_undeclared_threads_leave_the_environment_alone(tmp_path, monkeypatch):
    for variable in THREAD_ENV_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    environment = execution_environment(ExecutionRequest(command=("true",), workdir=tmp_path))
    assert not set(THREAD_ENV_VARIABLES) & set(environment)


@pytest.mark.parametrize(
    "kwargs",
    [{"ntasks": 0}, {"threads_per_task": 0}, {"memory_mb": 0}],
)
def test_resource_request_rejects_non_positive_counts(kwargs):
    with pytest.raises(ValueError):
        ResourceRequest(**kwargs)


def test_resolve_backend_defaults_to_local_and_honours_the_config():
    assert isinstance(resolve_backend(None), LocalBackend)
    assert isinstance(resolve_backend(object()), LocalBackend)
    recording = RecordingBackend()

    class Config:
        backend = recording

    assert resolve_backend(Config()) is recording
    assert isinstance(recording, ExecutionBackend)
    assert isinstance(LocalBackend(), ExecutionBackend)


def test_resolve_backend_rejects_something_that_cannot_run():
    class Config:
        backend = "slurm"

    with pytest.raises(TypeError, match="ExecutionBackend"):
        resolve_backend(Config())


def test_missing_workdir_stays_a_configuration_error(tmp_path):
    with pytest.raises(FileNotFoundError) as raised:
        _python(tmp_path / "absent", "pass")
    assert not isinstance(raised.value, ExecutableNotLaunchable)
