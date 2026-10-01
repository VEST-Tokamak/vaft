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


def test_a_record_left_by_a_dead_owner_is_dropped(tmp_path):
    # The lock is a no-op without fcntl; the probe itself must work everywhere.
    ledger = MemoryLedger(tmp_path, floor_mb=0, available=lambda: 1000.0)
    (tmp_path / "stale.json").write_text(json.dumps({"owner": 2**22 + 12345, "reserve_mb": 900, "root": None}))
    assert ledger.try_admit(500) is not None
    assert not (tmp_path / "stale.json").exists()


def test_the_owner_probe_never_signals_on_windows(monkeypatch):
    """``os.kill(pid, 0)`` is TerminateProcess on Windows: the ledger must not use it there.

    Runs on every platform by simulating ``os.name == "nt"`` with a psutil stub,
    so the POSIX CI legs guard the Windows branch (cold review 0.8.0 delta-absorb-5 F1).
    """
    import types

    from vaft.code import _memory_gate as mg

    dead = 2**22 + 12345
    kills: list[tuple[int, int]] = []
    real_kill = os.kill

    def recording_kill(pid, sig):
        kills.append((int(pid), int(sig)))
        return real_kill(pid, sig)

    monkeypatch.setattr(os, "kill", recording_kill)

    # POSIX branch: a probe, and a dead owner is "gone", not an exception.
    if os.name != "nt":
        assert mg._alive(os.getpid()) is True
        assert mg._alive(dead) is False
        assert (os.getpid(), 0) in kills and (dead, 0) in kills

    # Simulated Windows: psutil answers, os.kill is never reached.
    kills.clear()
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setitem(sys.modules, "psutil", types.SimpleNamespace(pid_exists=lambda pid: pid == os.getpid()))
    assert mg._alive(os.getpid()) is True
    assert mg._alive(dead) is False
    assert mg._alive(0) is False
    assert kills == []


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
    # floor 0 and a short wait: on a loaded host MemAvailable may be below the default floor.
    backend = LocalBackend(reserve_mb=1, floor_mb=0, admission_wait_s=5, ledger_dir=ledger_dir)
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


@pytest.mark.skipif(sys.platform == "win32", reason="admission needs fcntl")
def test_unreadable_ledger_records_are_dropped_not_fatal(tmp_path):
    ledger = MemoryLedger(tmp_path, floor_mb=0, available=lambda: 1000.0)
    ledger.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    (tmp_path / "list.json").write_text("[1, 2]")
    (tmp_path / "no_owner.json").write_text(json.dumps({"reserve_mb": 900}))
    (tmp_path / "partial.json").write_text('{"owner": ')
    assert ledger.try_admit(500) is not None
    assert sorted(p.name for p in tmp_path.glob("*.json") if p.name in ("list.json", "no_owner.json", "partial.json")) == []


@pytest.mark.skipif(sys.platform == "win32", reason="admission needs fcntl")
def test_a_reservation_larger_than_the_host_is_refused_at_once(tmp_path):
    import time

    ledger = MemoryLedger(tmp_path, floor_mb=100, available=lambda: 1000.0, total=lambda: 1000.0)
    started = time.monotonic()
    assert ledger.admit(950, wait_s=60, poll_s=1) is None
    assert time.monotonic() - started < 5


def test_admission_is_measured_against_the_process_budget_not_only_the_host(tmp_path):
    """A cgroup/Slurm/env limit (memory_budget, #1433) caps the ledger's room too.

    The host reports 10 GB free, the budget allows 1000 MiB; the launched
    tree lands in the same cgroup as its launcher, so the budget is what
    binds (cold review 0.8.0 delta-absorb-5 F2).
    """
    from vaft.code.resources import MemoryBudgetInfo

    # A cgroup budget: usage is the cgroup's non-reclaimable memory, read from memory.stat.
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    (cgroup / "memory.stat").write_text(f"anon {100 * 1024 * 1024}\nshmem 0\nfile {5 * 1024 * 1024}\n")
    info = MemoryBudgetInfo(limit_mb=1000.0, source="cgroup_v2", candidates={"cgroup_v2": 1000.0},
                            usage_cgroup=str(cgroup), usage_kind="v2")
    ledger = MemoryLedger(tmp_path / "ledger", floor_mb=100, available=lambda: 10000.0,
                          total=lambda: 10000.0, budget=lambda: info)
    assert ledger.available_mb() == 900.0 and ledger.total_mb() == 1000.0
    first = ledger.try_admit(450)
    assert first is not None
    assert ledger.try_admit(450) is None  # 900 - 450 outstanding < 450 + 100
    first.release()
    assert ledger.admit(950, wait_s=60, poll_s=1) is None  # never fits under the budget

    # An env/Slurm budget: usage is this process's RSS; a huge budget keeps the host reading.
    loose = MemoryBudgetInfo(limit_mb=10**7, source="env", candidates={"env": 10**7})
    host = MemoryLedger(tmp_path / "ledger", floor_mb=100, available=lambda: 10000.0,
                        total=lambda: 10000.0, budget=lambda: loose)
    assert host.available_mb() == 10000.0 and host.total_mb() == 10000.0
    # No budget at all: the host readings alone.
    none = MemoryBudgetInfo(limit_mb=None, source=None, candidates={})
    bare = MemoryLedger(tmp_path / "ledger", floor_mb=0, available=lambda: None, total=lambda: None,
                        budget=lambda: none)
    assert bare.available_mb() is None and bare.total_mb() is None


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX ownership")
def test_the_ledger_directory_is_private(tmp_path):
    ledger = MemoryLedger(tmp_path / "ledger", floor_mb=0, available=lambda: 1000.0)
    assert ledger.try_admit(1) is not None
    assert (tmp_path / "ledger").stat().st_mode & 0o777 == 0o700


def test_memory_settings_are_validated():
    with pytest.raises(ValueError):
        LocalBackend(poll_interval_s=0)
    with pytest.raises(ValueError):
        LocalBackend(floor_mb=-1)
    with pytest.raises(ValueError):
        LocalBackend(memory_limit_mb=0)


def test_a_memory_stop_counts_as_a_limit_stop_for_adapters():
    from vaft.code.base import CodeResult

    stopped = CodeResult(returncode=None, workdir=Path("."), runtime_status="memory_limit")
    assert stopped.timed_out and stopped.status == "failed"
