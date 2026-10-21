"""The shared execution backend every external-code adapter launches through.

Stub programs only (``#!/bin/sh`` or ``.cmd``, see ``external_code_stubs``):
these tests pin the launch contract -- environment overlay, captured versus
logged output, stdin, thread declaration, timeout and launch failure -- without
any physics code installed.
"""

from __future__ import annotations

import subprocess
import sys
import time
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


def test_resolve_backend_rejects_a_backend_class():
    class Config:
        backend = LocalBackend

    with pytest.raises(TypeError, match="ExecutionBackend"):
        resolve_backend(Config())


def test_log_path_may_be_a_string(tmp_path):
    log = tmp_path / "run.log"
    result = _python(tmp_path, "print('ok')", log_path=str(log))
    assert result.returncode == 0
    assert result.log_path == log
    assert log.read_text(encoding="utf-8").strip() == "ok"


def test_execution_submodule_is_an_attribute_of_vaft_code():
    import vaft.code

    assert vaft.code.execution.LocalBackend is LocalBackend


def test_missing_workdir_stays_a_configuration_error(tmp_path):
    with pytest.raises(FileNotFoundError) as raised:
        _python(tmp_path / "absent", "pass")
    assert not isinstance(raised.value, ExecutableNotLaunchable)


def test_stack_limit_wraps_only_where_ulimit_can_act(monkeypatch):
    from vaft import compat
    from vaft.code._launch import with_stack_limit

    monkeypatch.setattr(compat, "IS_WINDOWS", False)
    wrapped = with_stack_limit(["solver", 129], 32768, runner_name="solver-runner")
    assert wrapped[:2] == ["bash", "-lc"]
    assert "ulimit -s 32768" in wrapped[2] and "ulimit -Hs" in wrapped[2]
    assert wrapped[3:] == ["solver-runner", "solver", "129"]
    assert "ulimit -s $(ulimit -Hs)" in with_stack_limit(["s"], "hard", runner_name="r")[2]
    assert with_stack_limit(["s", 1], None, runner_name="r") == ["s", "1"]

    monkeypatch.setattr(compat, "IS_WINDOWS", True)
    assert with_stack_limit(["s"], 32768, runner_name="r") == ["s"]


def test_runtime_status_is_derived_from_timed_out_and_kept_consistent():
    """#1016: every backend result names how the run ended."""
    from vaft.code.execution import timeout_reason

    assert ExecutionResult(returncode=0).runtime_status == "completed"
    stopped = ExecutionResult(returncode=None, timed_out=True, elapsed_s=12.4)
    assert stopped.runtime_status == "timeout"
    queued = ExecutionResult(
        returncode=None, timed_out=True, elapsed_s=12.4, runtime_status="queue_timeout"
    )
    assert timeout_reason("X", stopped, 10.0) == "X timed out after 10 s of running"
    assert timeout_reason("X", stopped, None) == "X timed out after 12.4 s of running"
    assert timeout_reason("X", queued, 10.0) == (
        "X was cancelled after waiting 12 s in the scheduler queue (it never started)"
    )
    # A scheduler's max_wait stopping a running job at 2 s of a 600 s walltime.
    early = ExecutionResult(returncode=None, timed_out=True, elapsed_s=2.0)
    assert timeout_reason("X", early, 600.0) == "X timed out after 2 s of running"
    # A batch job's elapsed_s counts from submission; the running time is
    # run_s when the backend measured it (cold review 0.8.0 execution-backend F2).
    queued_then_ran = ExecutionResult(
        returncode=None, timed_out=True, elapsed_s=1000.0, run_s=1.4, job_id="7"
    )
    assert timeout_reason("X", queued_then_ran, 600.0) == "X timed out after 1.4 s of running"
    ran_out = ExecutionResult(returncode=None, timed_out=True, elapsed_s=1000.0, run_s=598.0)
    assert timeout_reason("X", ran_out, 600.0) == "X timed out after 600 s of running"
    with pytest.raises(ValueError, match="disagrees"):
        ExecutionResult(returncode=0, runtime_status="timeout")
    with pytest.raises(ValueError, match="must be one of"):
        ExecutionResult(returncode=None, timed_out=True, runtime_status="killed")


def test_a_local_timeout_reports_runtime_status(tmp_path):
    result = _python(tmp_path, "import time; time.sleep(30)", timeout=1.0)
    assert (result.timed_out, result.runtime_status) == (True, "timeout")


# -- submit / wait / cancel (#1018) ----------------------------------------------


def _request(tmp_path: Path, code: str, **request) -> ExecutionRequest:
    return ExecutionRequest(command=(sys.executable, "-c", code), workdir=tmp_path, **request)


def test_submit_without_max_concurrent_is_eager(tmp_path):
    handle = LocalBackend().submit(_request(tmp_path, "print('done')"))
    assert handle.done()
    assert handle.result().stdout.strip() == "done"
    assert not handle.cancel()  # finished: nothing to stop


def test_max_concurrent_bounds_the_live_programs(tmp_path):
    import json

    from vaft.code import wait_handles

    code = (
        "import json, os, sys, time; start = time.time(); time.sleep(0.6);"
        "print(json.dumps([start, time.time()]))"
    )
    backend = LocalBackend(max_concurrent=2)
    handles = [backend.submit(_request(tmp_path, code)) for _ in range(5)]
    assert not all(handle.done() for handle in handles)  # submit did not wait
    done, pending = wait_handles(handles)
    assert (len(done), pending) == (5, [])
    spans = [json.loads(handle.result().stdout) for handle in handles]
    for start, _ in spans:
        live = sum(1 for other_start, other_end in spans if other_start <= start < other_end)
        assert live <= 2
    # Started in submission order: the first two did not wait for a slot.
    assert sorted(spans)[:2] == sorted(spans[:2])


@pytest.mark.skipif(sys.platform == "win32", reason="probes the grandchild with os.kill(pid, 0)")
def test_cancel_stops_a_running_tree_and_a_queued_request(tmp_path):
    import os
    from concurrent.futures import CancelledError

    from vaft.code import cancel_handles

    pid_file = tmp_path / "grandchild.pid"
    code = (
        "import subprocess, sys, time;"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']);"
        f"open({str(pid_file)!r}, 'w').write(str(child.pid)); time.sleep(60)"
    )
    marker = tmp_path / "queued-ran"
    backend = LocalBackend(max_concurrent=1)
    running = backend.submit(_request(tmp_path, code))
    queued = backend.submit(_request(tmp_path, f"open({str(marker)!r}, 'w')"))
    deadline = time.monotonic() + 20
    while not pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    grandchild = int(pid_file.read_text())
    started = time.monotonic()
    cancel_handles([running, queued])
    assert time.monotonic() - started < 15
    for handle in (running, queued):
        assert handle.cancelled and handle.done()
        with pytest.raises(CancelledError):
            handle.result()
    assert not marker.exists()
    with pytest.raises(ProcessLookupError):
        os.kill(grandchild, 0)


def test_a_launch_error_comes_out_of_result_and_a_missing_workdir_out_of_submit(tmp_path):
    backend = LocalBackend(max_concurrent=1)
    with pytest.raises(FileNotFoundError):
        backend.submit(_request(tmp_path / "missing", "pass"))
    program = write_unlaunchable_file(tmp_path / "not-a-program")
    handle = backend.submit(ExecutionRequest(command=(str(program),), workdir=tmp_path))
    with pytest.raises(ExecutableNotLaunchable):
        handle.result()


def test_submit_request_runs_a_run_only_backend_eagerly(tmp_path):
    from vaft.code import submit_request

    backend = RecordingBackend()
    handle = submit_request(backend, _request(tmp_path, "pass"))
    assert handle.done() and handle.result().returncode == 0
    assert len(backend.requests) == 1

    class Failing:
        def run(self, request):
            raise RuntimeError("refused")

    handle = submit_request(Failing(), _request(tmp_path, "pass"))
    with pytest.raises(RuntimeError, match="refused"):
        handle.result()


def test_wait_handles_first_completed(tmp_path):
    from vaft.code import cancel_handles, wait_handles
    from vaft.code.execution import FIRST_COMPLETED

    backend = LocalBackend(max_concurrent=2)
    slow = backend.submit(_request(tmp_path, "import time; time.sleep(30)"))
    fast = backend.submit(_request(tmp_path, "pass"))
    done, pending = wait_handles([slow, fast], return_when=FIRST_COMPLETED, timeout=20)
    assert (done, pending) == ([fast], [slow])
    cancel_handles([slow])
    with pytest.raises(ValueError):
        wait_handles([fast], return_when="ANY")
    with pytest.raises(ValueError):
        LocalBackend(max_concurrent=0)


def test_memory_admission_stops_waiting_once_cancelled(tmp_path, monkeypatch):
    """Cold review #1018: a cancelled request does not wait out admission_wait_s."""
    from vaft.code._memory_gate import MemoryLedger

    ledger = MemoryLedger(tmp_path, floor_mb=0.0)
    monkeypatch.setattr(MemoryLedger, "fits", lambda self, reserve_mb: True)
    monkeypatch.setattr(MemoryLedger, "try_admit", lambda self, reserve_mb: None)
    flag = []
    started = time.monotonic()
    assert ledger.admit(10.0, wait_s=3600, poll_s=0.01, cancelled=lambda: bool(flag.append(1)) or len(flag) > 3) is None
    assert time.monotonic() - started < 5
