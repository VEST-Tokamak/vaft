"""The in-process memory guard (#1146): RSS, the effective budget, clean stop.

Budget resolution is tested against fake ``SLURM_*`` environments and fake
cgroup trees written under ``tmp_path``, never against the host's own limits.
The guard tests allocate at most ~128 MiB, and the RSS tests only check units
and direction, so the module stays in the develop gate.
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

import pytest

import vaft.code as code
from vaft.code import resources
from vaft.code.resources import (
    MEMORY_ENV_VARIABLES,
    MemoryBudget,
    MemoryBudgetExceeded,
    MemoryBudgetInfo,
    memory_budget,
    peak_rss_mb,
    rss_mb,
)

GIB = 1024**3
MIB = 1024**2


@pytest.fixture
def no_host(monkeypatch, tmp_path):
    """Keyword arguments that hide every host source from ``memory_budget``."""
    monkeypatch.setattr(resources, "_rlimit_as_bytes", lambda: None)
    return {
        "cgroup_root": tmp_path / "no-cgroup",
        "proc_cgroup": tmp_path / "no-proc-cgroup",
        "meminfo": tmp_path / "no-meminfo",
    }


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _fake_cgroup_v2(root: Path, job_bytes: int) -> Path:
    """A Slurm-shaped v2 tree: the limit sits on the job, the process in a step."""
    _write(root / "memory.max", "max\n")
    _write(root / "system.slice" / "slurmstepd.scope" / "job_7" / "memory.max", f"{job_bytes}\n")
    _write(
        root / "system.slice" / "slurmstepd.scope" / "job_7" / "step_0" / "memory.max",
        "max\n",
    )
    proc = root.parent / "proc-self-cgroup"
    _write(proc, "0::/system.slice/slurmstepd.scope/job_7/step_0\n")
    return proc


def _fake_cgroup_v1(root: Path, job_bytes: int) -> Path:
    memory = root / "memory"
    _write(memory / "memory.limit_in_bytes", "9223372036854771712\n")  # unlimited
    _write(memory / "slurm" / "uid_1" / "job_7" / "memory.limit_in_bytes", f"{job_bytes}\n")
    _write(
        memory / "slurm" / "uid_1" / "job_7" / "step_0" / "memory.limit_in_bytes",
        "9223372036854771712\n",
    )
    proc = root.parent / "proc-self-cgroup"
    _write(
        proc,
        "12:cpuset:/slurm/uid_1/job_7/step_0\n"
        "4:memory:/slurm/uid_1/job_7/step_0\n"
        "1:name=systemd:/user.slice\n",
    )
    return proc


# ------------------------------------------------------------ budget sources


def test_no_source_means_no_limit(no_host):
    info = memory_budget({}, **no_host)
    assert info == MemoryBudgetInfo(limit_mb=None, source=None, candidates={})


def test_slurm_mem_per_node(no_host):
    info = memory_budget({"SLURM_MEM_PER_NODE": "8000"}, **no_host)
    assert (info.limit_mb, info.source) == (8000.0, "slurm_mem_per_node")


def test_slurm_mem_per_cpu_is_multiplied_by_the_cpus_on_the_node(no_host):
    info = memory_budget({"SLURM_MEM_PER_CPU": "2000", "SLURM_CPUS_ON_NODE": "4"}, **no_host)
    assert (info.limit_mb, info.source) == (8000.0, "slurm_mem_per_cpu")


def test_slurm_mem_per_cpu_without_a_cpu_count_is_not_a_limit(no_host):
    assert memory_budget({"SLURM_MEM_PER_CPU": "2000"}, **no_host).limit_mb is None


def test_sizes_accept_slurm_unit_suffixes(no_host):
    assert memory_budget({"SLURM_MEM_PER_NODE": "4G"}, **no_host).limit_mb == 4096.0
    assert memory_budget({"VAFT_MEMORY_BUDGET_MB": "512M"}, **no_host).limit_mb == 512.0
    assert memory_budget({"VAFT_MEMORY_BUDGET_MB": "junk"}, **no_host).limit_mb is None


def test_cgroup_v2_limit_is_found_above_the_step_cgroup(no_host, tmp_path):
    root = tmp_path / "cgroup"
    proc = _fake_cgroup_v2(root, 3 * GIB)
    info = memory_budget({}, **{**no_host, "cgroup_root": root, "proc_cgroup": proc})
    assert (info.limit_mb, info.source) == (3072.0, "cgroup_v2")


def test_cgroup_v2_max_everywhere_is_no_limit(no_host, tmp_path):
    root = tmp_path / "cgroup"
    _write(root / "memory.max", "max\n")
    proc = tmp_path / "proc-self-cgroup"
    _write(proc, "0::/\n")
    info = memory_budget({}, **{**no_host, "cgroup_root": root, "proc_cgroup": proc})
    assert info.limit_mb is None


def test_cgroup_v1_limit_ignores_the_unlimited_sentinel(no_host, tmp_path):
    root = tmp_path / "cgroup"
    proc = _fake_cgroup_v1(root, 5 * GIB)
    info = memory_budget({}, **{**no_host, "cgroup_root": root, "proc_cgroup": proc})
    assert (info.limit_mb, info.source) == (5120.0, "cgroup_v1")


def test_rlimit_as_is_a_candidate(no_host, monkeypatch):
    monkeypatch.setattr(resources, "_rlimit_as_bytes", lambda: 2.0 * GIB)
    info = memory_budget({}, **no_host)
    assert (info.limit_mb, info.source) == (2048.0, "rlimit_as")


def test_mem_available_is_read_from_meminfo(no_host, tmp_path):
    meminfo = tmp_path / "meminfo"
    _write(meminfo, "MemTotal:       16384000 kB\nMemAvailable:    1048576 kB\n")
    info = memory_budget({}, **{**no_host, "meminfo": meminfo})
    assert (info.limit_mb, info.source) == (1024.0, "mem_available")


def test_the_budget_is_the_minimum_of_every_source(no_host, tmp_path, monkeypatch):
    root = tmp_path / "cgroup"
    proc = _fake_cgroup_v2(root, 6 * GIB)
    meminfo = tmp_path / "meminfo"
    _write(meminfo, "MemAvailable:   20971520 kB\n")  # 20 GiB
    monkeypatch.setattr(resources, "_rlimit_as_bytes", lambda: 64.0 * GIB)
    environ = {
        "SLURM_MEM_PER_NODE": "8000",
        "SLURM_MEM_PER_CPU": "1500", "SLURM_CPUS_ON_NODE": "4",  # 6000
        "VAFT_MEMORY_BUDGET_MB": "7000",
    }
    info = memory_budget(environ, cgroup_root=root, proc_cgroup=proc, meminfo=meminfo)
    assert info.candidates == {
        "slurm_mem_per_node": 8000.0,
        "slurm_mem_per_cpu": 6000.0,
        "cgroup_v2": 6144.0,
        "rlimit_as": 65536.0,
        "env": 7000.0,
        "mem_available": 20480.0,
    }
    assert (info.limit_mb, info.source) == (6000.0, "slurm_mem_per_cpu")

    # The explicit budget can lower the limit, never raise it.
    lower = memory_budget(
        {**environ, "VAFT_MEMORY_BUDGET_MB": "1000"},
        cgroup_root=root, proc_cgroup=proc, meminfo=meminfo,
    )
    assert (lower.limit_mb, lower.source) == (1000.0, "env")
    assert lower.as_dict()["candidates"]["env"] == 1000.0


def test_the_default_environment_is_os_environ(no_host, monkeypatch):
    monkeypatch.setenv("VAFT_MEMORY_BUDGET_MB", "321")
    info = memory_budget(**no_host)
    assert (info.limit_mb, info.source) == (321.0, "env")


def test_the_host_budget_resolves_to_a_positive_limit():
    info = memory_budget()
    if info.limit_mb is None:
        pytest.skip("this host reports no memory limit at all")
    assert info.limit_mb > 0
    assert info.source in info.candidates


# ----------------------------------------------------------------------- RSS


def test_rss_units_are_mebibytes():
    current, peak = rss_mb(), peak_rss_mb()
    assert current is not None and peak is not None
    # A running CPython with pytest loaded is tens to hundreds of MiB. A KiB or
    # byte mix-up is off by 1024 either way and lands far outside this band.
    assert 5.0 < current < 64 * 1024
    assert 5.0 < peak < 64 * 1024
    assert peak >= 0.9 * current


def test_rss_follows_an_allocation():
    before = rss_mb()
    block = b"\x01" * (64 * MIB)  # written, so every page is resident
    after = rss_mb()
    peak = peak_rss_mb()
    assert after - before > 48.0, (before, after)
    assert peak >= after - 1.0
    del block


# --------------------------------------------------------------------- guard


def test_check_raises_above_the_threshold():
    guard = MemoryBudget(limit_mb=1.0, fraction=0.5)
    with guard:
        with pytest.raises(MemoryBudgetExceeded) as caught:
            guard.check(label="slice 3")
    error = caught.value
    assert error.label == "slice 3"
    assert error.limit_mb == 1.0 and error.threshold_mb == 0.5
    assert error.source == "explicit"
    assert error.rss_mb > 0.5 and error.peak_mb >= error.rss_mb
    assert guard.exceeded is error
    assert "slice 3" in str(error)


def test_an_allocation_past_a_small_limit_is_caught_with_its_peak():
    start = rss_mb()
    guard = MemoryBudget(limit_mb=start + 64.0, fraction=1.0)
    with guard:
        assert guard.check("before") is not None  # under the limit: no raise
        block = b"\x01" * (128 * MIB)
        with pytest.raises(MemoryBudgetExceeded) as caught:
            guard.check("after")
        del block
    assert caught.value.rss_mb > start + 64.0
    assert guard.peak_mb >= caught.value.rss_mb
    assert guard.as_dict()["exceeded"] is True


def test_the_sampler_catches_a_crossing_between_checks():
    start = rss_mb()
    guard = MemoryBudget(limit_mb=start + 64.0, fraction=1.0, interval_s=0.01)
    with guard:
        block = b"\x01" * (128 * MIB)
        deadline = time.monotonic() + 5.0
        while guard._tripped is None and time.monotonic() < deadline:
            time.sleep(0.01)
        del block  # the flag stands even after RSS falls back
        with pytest.raises(MemoryBudgetExceeded) as caught:
            guard.check("next slice")
    assert caught.value.label == "next slice"
    assert guard.peak_mb > start + 64.0


def test_leaving_the_block_never_raises_but_records_the_crossing():
    guard = MemoryBudget(limit_mb=1.0, fraction=1.0, interval_s=0.01)
    with guard:
        deadline = time.monotonic() + 5.0
        while guard._tripped is None and time.monotonic() < deadline:
            time.sleep(0.01)
    assert isinstance(guard.exceeded, MemoryBudgetExceeded)
    assert guard._thread is None


def test_warn_mode_warns_once_and_records():
    guard = MemoryBudget(limit_mb=1.0, on_exceed="warn")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with guard:
            guard.check()
            guard.check()
    assert len([w for w in caught if issubclass(w.category, ResourceWarning)]) == 1
    assert guard.exceeded is not None


def test_a_resolved_budget_carries_its_source():
    guard = MemoryBudget(environ={"VAFT_MEMORY_BUDGET_MB": "1"})
    with guard, pytest.raises(MemoryBudgetExceeded) as caught:
        guard.check()
    assert guard.limit_mb <= 1.0
    assert caught.value.source == guard.source


def test_without_any_limit_the_guard_only_records(monkeypatch):
    monkeypatch.setattr(
        resources, "memory_budget", lambda environ=None: MemoryBudgetInfo(None, None)
    )
    with MemoryBudget() as guard:
        assert guard.check() is not None
        assert guard.threshold_mb is None
    assert guard.peak_mb > 0 and guard.exceeded is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fraction": 0.0},
        {"fraction": 1.5},
        {"limit_mb": -1.0},
        {"limit_mb": float("inf")},
        {"interval_s": 0.0},
        {"on_exceed": "kill"},
    ],
)
def test_invalid_arguments_are_rejected(kwargs):
    with pytest.raises(ValueError):
        MemoryBudget(**kwargs)


# ----------------------------------------------------------- suite contracts


def test_the_suite_clears_every_memory_variable():
    conftest = (Path(__file__).parent / "conftest.py").read_text()
    for name in MEMORY_ENV_VARIABLES:
        assert f'"{name}"' in conftest, name
        assert name not in os.environ, name


def test_vaft_code_exports_the_guard():
    for name in (
        "MemoryBudget", "MemoryBudgetExceeded", "MemoryBudgetInfo",
        "memory_budget", "peak_rss_mb", "rss_mb",
    ):
        assert name in code.__all__
        assert getattr(code, name) is getattr(resources, name)
    assert code.resources is resources


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS fallback paths")
def test_macos_native_and_ps_readings_agree():
    native = resources._darwin_resident_bytes() / MIB
    ps = resources._ps_rss_kib() / 1024.0
    assert abs(native - ps) < 0.1 * max(native, ps) + 16.0


def test_the_budget_stop_is_a_memory_error_not_a_runtime_error():
    # Loops catch RuntimeError around one item as "that item failed"; a budget
    # stop must not be retried as if it were one.
    assert issubclass(MemoryBudgetExceeded, MemoryError)
    assert not issubclass(MemoryBudgetExceeded, RuntimeError)


def test_an_interval_peak_belongs_to_one_item(monkeypatch):
    # Scripted readings: whether freed memory leaves the RSS promptly is the
    # allocator's business (macOS keeps it), not the guard's.
    readings = iter([100.0, 100.0, 500.0, 120.0, 130.0, 125.0, 125.0])
    monkeypatch.setattr(resources, "rss_mb", lambda: next(readings))
    with MemoryBudget(limit_mb=1e6) as guard:     # enter: 100
        guard.start_interval()                     # 100
        guard.check("big item")                    # 500
        big = guard.start_interval()               # ends at 500, opens at 120
        guard.check("small item")                  # 130
        small = guard.start_interval()             # ends at 130, opens at 125
    assert (big, small) == (500.0, 130.0)
    assert guard.peak_mb == 500.0
