"""Memory admission and the resident-size limit of LocalBackend (#1460).

Real child processes only. The memory-stop cases allocate a few hundred MiB
and set a limit far below that, so they do not depend on the host's memory.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from vaft.code import ExecutionRequest, LocalBackend, ResourceRequest
from vaft.code._memory_gate import MemoryLedger, tree_rss_mb
from vaft.code.execution import RUNTIME_MEMORY_LIMIT, RUNTIME_QUEUE_TIMEOUT, timeout_reason

POSIX = pytest.mark.skipif(sys.platform == "win32", reason="tree RSS needs /proc or ps")

#: Allocates and touches ~300 MiB, then sleeps long enough to be sampled.
HOG = "b = bytearray(300 * 1024 * 1024)\nb[::4096] = b'x' * len(b[::4096])\nimport time; time.sleep(30)"


def _run(backend: LocalBackend, tmp_path: Path, code: str, **request):
    return backend.run(ExecutionRequest(command=(sys.executable, "-c", code), workdir=tmp_path, **request))


@POSIX
def test_a_tree_past_the_limit_is_stopped_and_returned(tmp_path):
    backend = LocalBackend(memory_limit_mb=100, poll_interval_s=0.2, ledger_dir=tmp_path / "ledger")
    log = tmp_path / "program.log"
    result = _run(backend, tmp_path, HOG, timeout=60, log_path=log)
    assert result.runtime_status == RUNTIME_MEMORY_LIMIT
    assert result.timed_out and result.returncode is None
    assert result.peak_rss_mb is not None and result.peak_rss_mb > 100
    assert result.elapsed_s < 30
    assert "memory_limit_mb=100" in log.read_text()
    assert "stopped by the memory limit" in timeout_reason("X", result, 60)


@POSIX
def test_a_grandchild_counts_toward_the_limit(tmp_path):
    spawn = f"import subprocess, sys; subprocess.run([sys.executable, '-c', {HOG!r}])"
    backend = LocalBackend(memory_limit_mb=100, poll_interval_s=0.2, ledger_dir=tmp_path / "ledger")
    result = _run(backend, tmp_path, spawn, timeout=60)
    assert result.runtime_status == RUNTIME_MEMORY_LIMIT and result.elapsed_s < 30


@POSIX
def test_a_program_under_the_limit_completes_with_its_peak(tmp_path):
    backend = LocalBackend(memory_limit_mb=4096, poll_interval_s=0.1, ledger_dir=tmp_path / "ledger")
    result = _run(backend, tmp_path, "import time; time.sleep(0.5); print('done')", timeout=30)
    assert result.runtime_status == "completed" and result.returncode == 0
    assert result.stdout.strip() == "done"
    assert result.peak_rss_mb is not None and 0 < result.peak_rss_mb < 4096


def test_the_time_limit_still_applies_while_memory_is_watched(tmp_path):
    backend = LocalBackend(memory_limit_mb=1e6, poll_interval_s=0.2, ledger_dir=tmp_path / "ledger")
    result = _run(backend, tmp_path, "import time; time.sleep(30)", timeout=0.6)
    assert result.runtime_status == "timeout" and result.elapsed_s < 20


@pytest.mark.skipif(sys.platform == "win32", reason="admission needs fcntl")
def test_the_ledger_counts_what_admitted_jobs_have_not_yet_used(tmp_path):
    ledger = MemoryLedger(tmp_path, floor_mb=100, available=lambda: 1000.0)
    first = ledger.try_admit(500)
    assert first is not None
    # MemAvailable is unchanged (the first job has not grown), but its
    # reservation is still outstanding: 1000 - 500 < 500 + 100.
    assert ledger.try_admit(500) is None
    first.release()
    assert ledger.try_admit(500) is not None


@pytest.mark.skipif(sys.platform == "win32", reason="admission needs fcntl")
def test_a_record_left_by_a_dead_owner_is_dropped(tmp_path):
    ledger = MemoryLedger(tmp_path, floor_mb=0, available=lambda: 1000.0)
    (tmp_path / "stale.json").write_text(json.dumps({"owner": 2**22 + 12345, "reserve_mb": 900, "root": None}))
    assert ledger.try_admit(500) is not None
    assert not (tmp_path / "stale.json").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="admission needs fcntl")
def test_a_launch_never_admitted_is_returned_not_raised(tmp_path):
    backend = LocalBackend(reserve_mb=1e12, admission_wait_s=0, ledger_dir=tmp_path / "ledger")
    log = tmp_path / "program.log"
    result = _run(backend, tmp_path, "print('never')", log_path=log)
    assert result.runtime_status == RUNTIME_QUEUE_TIMEOUT and result.waited_for == "memory"
    assert result.returncode is None and "not started" in log.read_text()
    assert "for memory to become available" in timeout_reason("X", result, None)
    assert list((tmp_path / "ledger").glob("*.json")) == []


@pytest.mark.skipif(sys.platform == "win32", reason="admission needs fcntl")
def test_the_reservation_is_released_after_the_run(tmp_path):
    ledger_dir = tmp_path / "ledger"
    backend = LocalBackend(reserve_mb=1, ledger_dir=ledger_dir)
    result = _run(backend, tmp_path, "print('ok')", resources=ResourceRequest(memory_mb=2))
    assert result.returncode == 0
    assert list(ledger_dir.glob("*.json")) == []


def test_a_default_backend_ignores_memory_declarations(tmp_path, monkeypatch):
    monkeypatch.setenv("VAFT_MEMORY_LEDGER_DIR", str(tmp_path / "ledger"))
    result = _run(LocalBackend(), tmp_path, "print('ok')", resources=ResourceRequest(memory_mb=10**9))
    assert result.returncode == 0 and result.peak_rss_mb is None
    assert not (tmp_path / "ledger").exists()


@POSIX
def test_tree_rss_of_this_process_is_positive_and_gone_pids_count_zero():
    assert tree_rss_mb([os.getpid()]) > 0
    assert tree_rss_mb([2**22 + 12345]) == 0
    assert tree_rss_mb([]) == 0
