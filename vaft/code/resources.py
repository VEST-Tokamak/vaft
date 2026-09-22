"""In-process memory guard: RSS, the effective budget, and a clean stop before OOM.

:class:`~vaft.code.execution.ResourceRequest` only *asks* a scheduler for memory
(``SlurmBackend`` turns ``memory_mb`` into ``--mem``). Nothing tells a
long-running Python loop that it is about to be killed by the kernel's or
Slurm's OOM killer, which ends the process without a traceback, a flushed file
or a status record. This module gives such a loop three things (#1146):

* :func:`rss_mb` and :func:`peak_rss_mb`, the current and peak resident set
  size of this process;
* :func:`memory_budget`, the *effective* limit the process runs under, which is
  the smallest of every limit that applies to it, together with the source that
  set it;
* :class:`MemoryBudget`, a context manager whose :meth:`~MemoryBudget.check`
  raises :class:`MemoryBudgetExceeded` above ``fraction x limit``, so the caller
  can stop at a point of its own choosing and record why.

Units: every ``*_mb`` value is a mebibyte (2**20 bytes), which is what Slurm's
``--mem`` and ``SLURM_MEM_PER_*`` mean by "MB".

Platforms: Linux reads ``/proc/self/status`` (``VmRSS``/``VmHWM``). macOS reads
the current RSS through ``task_info`` and the peak through ``getrusage``, whose
``ru_maxrss`` is in bytes there and in KiB on Linux. Windows reads
``GetProcessMemoryInfo``. ``psutil`` is used when it is installed and a native
reading is not available; it is never required.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import sys
import threading
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Union

__all__ = [
    "MEMORY_BUDGET_ENV",
    "MEMORY_ENV_VARIABLES",
    "MemoryBudget",
    "MemoryBudgetExceeded",
    "MemoryBudgetInfo",
    "memory_budget",
    "peak_rss_mb",
    "rss_mb",
]

#: Explicit budget, in MiB. The smallest limit still wins: this can lower the
#: effective budget but never raise it above what Slurm or a cgroup enforces.
MEMORY_BUDGET_ENV = "VAFT_MEMORY_BUDGET_MB"

#: Every environment variable :func:`memory_budget` reads.
MEMORY_ENV_VARIABLES: tuple[str, ...] = (
    MEMORY_BUDGET_ENV,
    "SLURM_MEM_PER_NODE",
    "SLURM_MEM_PER_CPU",
    "SLURM_CPUS_ON_NODE",
)

_MIB = 1024 * 1024
#: cgroup v1 writes "unlimited" as a page-rounded 2**63 - 1; anything this large
#: is no limit.
_UNLIMITED_BYTES = 1 << 60
_DEFAULT_CGROUP_ROOT = Path("/sys/fs/cgroup")
_DEFAULT_PROC_CGROUP = Path("/proc/self/cgroup")
_DEFAULT_MEMINFO = Path("/proc/meminfo")

PathLike = Union[str, "os.PathLike[str]"]


# --------------------------------------------------------------------------- RSS


def _proc_status_kib(key: str) -> Optional[float]:
    try:
        text = Path("/proc/self/status").read_text()
    except OSError:
        return None
    match = re.search(rf"^{key}:\s*(\d+)\s*kB", text, re.MULTILINE)
    return float(match.group(1)) if match else None


def _windows_memory_counters() -> Optional[tuple[float, float]]:
    """(working set, peak working set) in bytes via ``GetProcessMemoryInfo``."""
    try:
        import ctypes
        from ctypes import wintypes
    except ImportError:  # pragma: no cover - ctypes always ships with CPython
        return None

    class _Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    try:
        counters = _Counters()
        counters.cb = ctypes.sizeof(_Counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(_Counters), wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        ok = psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
    except (OSError, AttributeError):
        return None
    if not ok:
        return None
    return float(counters.WorkingSetSize), float(counters.PeakWorkingSetSize)


def _darwin_resident_bytes() -> Optional[float]:
    """Current resident size from ``task_info(MACH_TASK_BASIC_INFO)``."""
    try:
        import ctypes
        import ctypes.util

        class _TimeValue(ctypes.Structure):
            _fields_ = [("seconds", ctypes.c_int), ("microseconds", ctypes.c_int)]

        class _BasicInfo(ctypes.Structure):
            _fields_ = [
                ("virtual_size", ctypes.c_uint64),
                ("resident_size", ctypes.c_uint64),
                ("resident_size_max", ctypes.c_uint64),
                ("user_time", _TimeValue),
                ("system_time", _TimeValue),
                ("policy", ctypes.c_int),
                ("suspend_count", ctypes.c_int),
            ]

        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.dylib")
        task = ctypes.c_uint.in_dll(libc, "mach_task_self_")
        info = _BasicInfo()
        count = ctypes.c_uint(ctypes.sizeof(_BasicInfo) // ctypes.sizeof(ctypes.c_uint))
        libc.task_info.argtypes = [
            ctypes.c_uint, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint),
        ]
        libc.task_info.restype = ctypes.c_int
        mach_task_basic_info = 20
        status = libc.task_info(
            task, mach_task_basic_info, ctypes.byref(info), ctypes.byref(count)
        )
    except (OSError, AttributeError, ValueError):
        return None
    return float(info.resident_size) if status == 0 else None


def _psutil_memory_info() -> Optional[Any]:
    try:
        import psutil  # optional
    except ImportError:
        return None
    try:
        return psutil.Process().memory_info()
    except Exception:  # pragma: no cover - psutil platform failure
        return None


def _ps_rss_kib() -> Optional[float]:
    try:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            capture_output=True, text=True, timeout=5, check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        return float(out.split()[0])
    except (IndexError, ValueError):
        return None


def _getrusage_peak_bytes() -> Optional[float]:
    try:
        import resource
    except ImportError:
        return None
    maxrss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux and the BSDs report KiB.
    return maxrss if sys.platform == "darwin" else maxrss * 1024.0


def rss_mb() -> Optional[float]:
    """Current resident set size of this process, in MiB.

    ``None`` only when no reading is available on this platform at all.
    """
    if sys.platform.startswith("linux"):
        kib = _proc_status_kib("VmRSS")
        if kib is not None:
            return kib / 1024.0
    elif sys.platform == "win32":
        counters = _windows_memory_counters()
        if counters is not None:
            return counters[0] / _MIB
    elif sys.platform == "darwin":
        resident = _darwin_resident_bytes()
        if resident is not None:
            return resident / _MIB
    info = _psutil_memory_info()
    if info is not None:
        return float(info.rss) / _MIB
    kib = _ps_rss_kib()
    if kib is not None:
        return kib / 1024.0
    return None


def peak_rss_mb() -> Optional[float]:
    """Peak resident set size of this process since it started, in MiB."""
    if sys.platform.startswith("linux"):
        kib = _proc_status_kib("VmHWM")
        if kib is not None:
            return kib / 1024.0
    elif sys.platform == "win32":
        counters = _windows_memory_counters()
        if counters is not None:
            return counters[1] / _MIB
    peak = _getrusage_peak_bytes()
    if peak is not None:
        return peak / _MIB
    info = _psutil_memory_info()
    if info is not None:
        peak_wset = getattr(info, "peak_wset", None)
        return float(peak_wset if peak_wset is not None else info.rss) / _MIB
    return None


# ------------------------------------------------------------------ the budget


@dataclass(frozen=True)
class MemoryBudgetInfo:
    """The effective memory limit of this process and where it came from.

    ``limit_mb`` is the minimum over ``candidates``; ``source`` names the
    candidate that set it. Both are ``None`` when no source reported a limit.
    ``candidates`` keeps every limit that was found, keyed by source.
    """

    limit_mb: Optional[float]
    source: Optional[str]
    candidates: Mapping[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form, for status records."""
        return {
            "limit_mb": self.limit_mb,
            "source": self.source,
            "candidates": dict(self.candidates),
        }


_SIZE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([KMGT]?)B?\s*$", re.IGNORECASE)
_SCALE_TO_MIB = {"": 1.0, "K": 1.0 / 1024, "M": 1.0, "G": 1024.0, "T": 1024.0 * 1024}


def _parse_mib(value: Optional[str]) -> Optional[float]:
    """Slurm-style size (plain number = MiB, optional K/M/G/T suffix)."""
    if value is None:
        return None
    match = _SIZE.match(str(value))
    if not match:
        return None
    number = float(match.group(1)) * _SCALE_TO_MIB[match.group(2).upper()]
    return number if number > 0 else None


def _slurm_limits(environ: Mapping[str, str]) -> dict[str, float]:
    limits: dict[str, float] = {}
    per_node = _parse_mib(environ.get("SLURM_MEM_PER_NODE"))
    if per_node is not None:
        limits["slurm_mem_per_node"] = per_node
    per_cpu = _parse_mib(environ.get("SLURM_MEM_PER_CPU"))
    cpus = environ.get("SLURM_CPUS_ON_NODE")
    if per_cpu is not None and cpus is not None:
        try:
            n_cpus = int(str(cpus).strip())
        except ValueError:
            n_cpus = 0
        if n_cpus > 0:
            limits["slurm_mem_per_cpu"] = per_cpu * n_cpus
    return limits


def _proc_cgroup_paths(proc_cgroup: Path) -> tuple[Optional[str], Optional[str]]:
    """(cgroup v2 path, cgroup v1 memory-controller path) of this process."""
    try:
        lines = proc_cgroup.read_text().splitlines()
    except OSError:
        return None, None
    v2 = v1 = None
    for line in lines:
        parts = line.strip().split(":", 2)
        if len(parts) != 3:
            continue
        hierarchy, controllers, path = parts
        if hierarchy == "0" and controllers == "":
            v2 = path
        elif "memory" in controllers.split(","):
            v1 = path
    return v2, v1


def _read_limit_bytes(path: Path) -> Optional[float]:
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    if not text or text == "max":
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if value <= 0 or value >= _UNLIMITED_BYTES:
        return None
    return value


def _walk_limit(base: Path, relative: Optional[str], filename: str) -> Optional[float]:
    """Smallest ``filename`` limit from the process's cgroup up to ``base``.

    A Slurm job's limit sits on its job cgroup while the process runs in a
    step or task cgroup below it, so the leaf alone is not enough.
    """
    leaf = base / relative.strip("/") if relative else base
    directories = [leaf, *leaf.parents] if leaf != base else [base]
    best: Optional[float] = None
    for directory in directories:
        value = _read_limit_bytes(directory / filename)
        if value is not None and (best is None or value < best):
            best = value
        if directory == base:
            break
    return best


def _cgroup_limits(cgroup_root: Path, proc_cgroup: Path) -> dict[str, float]:
    limits: dict[str, float] = {}
    v2_path, v1_path = _proc_cgroup_paths(proc_cgroup)
    v2 = _walk_limit(cgroup_root, v2_path, "memory.max")
    if v2 is not None:
        limits["cgroup_v2"] = v2 / _MIB
    v1_base = cgroup_root / "memory"
    if not v1_base.is_dir():
        v1_base = cgroup_root
    v1 = _walk_limit(v1_base, v1_path, "memory.limit_in_bytes")
    if v1 is not None:
        limits["cgroup_v1"] = v1 / _MIB
    return limits


def _rlimit_as_bytes() -> Optional[float]:
    try:
        import resource
    except ImportError:
        return None
    soft, _hard = resource.getrlimit(resource.RLIMIT_AS)
    if soft == resource.RLIM_INFINITY or soft <= 0 or soft >= _UNLIMITED_BYTES:
        return None
    return float(soft)


def _meminfo_available_mib(meminfo: Path) -> Optional[float]:
    try:
        text = meminfo.read_text()
    except OSError:
        return None
    match = re.search(r"^MemAvailable:\s*(\d+)\s*kB", text, re.MULTILINE)
    return float(match.group(1)) / 1024.0 if match else None


def _vm_stat_available_mib() -> Optional[float]:
    """macOS without psutil: free + inactive + speculative + purgeable pages."""
    try:
        out = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, timeout=5, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    page = re.search(r"page size of (\d+) bytes", out)
    if not page:
        return None
    pages = 0
    for key in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
        match = re.search(rf"^{key}:\s*(\d+)", out, re.MULTILINE)
        if match:
            pages += int(match.group(1))
    return pages * int(page.group(1)) / _MIB if pages else None


def _available_mib(meminfo: Optional[Path]) -> Optional[float]:
    if meminfo is not None:
        return _meminfo_available_mib(meminfo)
    if sys.platform.startswith("linux"):
        return _meminfo_available_mib(_DEFAULT_MEMINFO)
    try:
        import psutil  # optional
    except ImportError:
        psutil = None
    if psutil is not None:
        try:
            return float(psutil.virtual_memory().available) / _MIB
        except Exception:  # pragma: no cover - psutil platform failure
            pass
    if sys.platform == "darwin":
        return _vm_stat_available_mib()
    return None


def memory_budget(
    environ: Optional[Mapping[str, str]] = None,
    *,
    cgroup_root: Optional[PathLike] = None,
    proc_cgroup: Optional[PathLike] = None,
    meminfo: Optional[PathLike] = None,
) -> MemoryBudgetInfo:
    """The effective memory limit of this process, in MiB, and its source.

    The limit is the **minimum** of every source that reports one:

    ============================ ==================================================
    source                       value
    ============================ ==================================================
    ``slurm_mem_per_node``       ``SLURM_MEM_PER_NODE``
    ``slurm_mem_per_cpu``        ``SLURM_MEM_PER_CPU`` x ``SLURM_CPUS_ON_NODE``
    ``cgroup_v2``                ``memory.max``, smallest from the process's cgroup
                                 up to the root
    ``cgroup_v1``                ``memory/.../memory.limit_in_bytes``, likewise
    ``rlimit_as``                the soft ``RLIMIT_AS`` (address space, so it is
                                 an upper bound on RSS rather than an RSS limit)
    ``env``                      ``VAFT_MEMORY_BUDGET_MB``
    ``mem_available``            ``MemAvailable`` now (psutil or ``vm_stat`` off
                                 Linux); memory other processes may also take
    ============================ ==================================================

    ``environ`` defaults to ``os.environ``. ``cgroup_root`` (default
    ``/sys/fs/cgroup``), ``proc_cgroup`` (default ``/proc/self/cgroup``) and
    ``meminfo`` (default ``/proc/meminfo``) are injectable for tests; an
    explicit ``meminfo`` that does not exist means "no MemAvailable".
    """
    env = os.environ if environ is None else environ
    candidates: dict[str, float] = {}
    candidates.update(_slurm_limits(env))
    candidates.update(
        _cgroup_limits(
            Path(cgroup_root) if cgroup_root is not None else _DEFAULT_CGROUP_ROOT,
            Path(proc_cgroup) if proc_cgroup is not None else _DEFAULT_PROC_CGROUP,
        )
    )
    rlimit = _rlimit_as_bytes()
    if rlimit is not None:
        candidates["rlimit_as"] = rlimit / _MIB
    explicit = _parse_mib(env.get(MEMORY_BUDGET_ENV))
    if explicit is not None:
        candidates["env"] = explicit
    available = _available_mib(Path(meminfo) if meminfo is not None else None)
    if available is not None and available > 0:
        candidates["mem_available"] = available

    if not candidates:
        return MemoryBudgetInfo(limit_mb=None, source=None, candidates={})
    source = min(candidates, key=lambda name: candidates[name])
    return MemoryBudgetInfo(
        limit_mb=candidates[source], source=source, candidates=candidates
    )


# ------------------------------------------------------------------- the guard


class MemoryBudgetExceeded(MemoryError):
    """RSS crossed ``fraction x limit`` inside a :class:`MemoryBudget`.

    A :class:`MemoryError`, not a :class:`RuntimeError`: analysis loops
    routinely catch ``RuntimeError``/``ValueError`` around one item as "this
    item failed, try the next", and a budget stop must not be retried that
    way. Code that already handles ``MemoryError`` (numpy's allocation
    failure) is exactly the code that should see it.
    """

    def __init__(
        self,
        rss_mb: float,
        threshold_mb: float,
        limit_mb: float,
        *,
        label: Any = None,
        source: Optional[str] = None,
        peak_mb: Optional[float] = None,
    ) -> None:
        self.rss_mb = rss_mb
        self.threshold_mb = threshold_mb
        self.limit_mb = limit_mb
        self.label = label
        self.source = source
        self.peak_mb = peak_mb
        where = f" at {label!r}" if label is not None else ""
        origin = f" ({source})" if source else ""
        super().__init__(
            f"RSS {rss_mb:.0f} MiB{where} is above the memory threshold "
            f"{threshold_mb:.0f} MiB of a {limit_mb:.0f} MiB budget{origin}"
        )


class MemoryBudget:
    """Stop cleanly before the process runs out of memory.

    Parameters
    ----------
    limit_mb:
        The budget in MiB. ``None`` resolves it with :func:`memory_budget` when
        the block is entered. If nothing reports a limit either, the guard only
        records ``peak_mb`` and never raises.
    fraction:
        The threshold is ``fraction x limit_mb``; ``0 < fraction <= 1``.
    on_exceed:
        ``"raise"`` (default) raises :class:`MemoryBudgetExceeded` from
        :meth:`check`; ``"warn"`` issues a ``ResourceWarning`` once and records
        the event in ``exceeded`` instead.
    interval_s:
        When set, a daemon thread samples RSS every ``interval_s`` seconds while
        the block is active. A sample above the threshold sets a flag, and the
        next :meth:`check` raises even if RSS has dropped since, so a tight
        numpy call between two checks is still caught. The thread never raises
        into the main thread by itself.
    source:
        Label recorded with an explicit ``limit_mb``; resolved budgets carry the
        source :func:`memory_budget` found.

    ``peak_mb`` is the largest RSS the guard has seen (every :meth:`check`,
    every sampler tick, entry and exit). It is a lower bound on the true peak
    inside the block; :func:`peak_rss_mb` is the process-lifetime peak.
    ``interval_peak_mb`` is the same maximum since the last
    :meth:`start_interval`, so a loop can attribute a peak to one item (a
    time slice, a scan case) even though the process peak only ever grows.
    Leaving the block never raises: a crossing the sampler saw but no
    :meth:`check` reported is left in ``exceeded``.
    """

    def __init__(
        self,
        limit_mb: Optional[float] = None,
        fraction: float = 0.9,
        *,
        on_exceed: str = "raise",
        interval_s: Optional[float] = None,
        source: Optional[str] = None,
        environ: Optional[Mapping[str, str]] = None,
    ) -> None:
        if not 0.0 < float(fraction) <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        if on_exceed not in ("raise", "warn"):
            raise ValueError(f"on_exceed must be 'raise' or 'warn', got {on_exceed!r}")
        if limit_mb is not None and not (float(limit_mb) > 0 and math.isfinite(limit_mb)):
            raise ValueError(f"limit_mb must be a positive number or None, got {limit_mb}")
        if interval_s is not None and not float(interval_s) > 0:
            raise ValueError(f"interval_s must be > 0 or None, got {interval_s}")
        self.fraction = float(fraction)
        self.on_exceed = on_exceed
        self.interval_s = None if interval_s is None else float(interval_s)
        self._environ = environ
        self.limit_mb: Optional[float] = None if limit_mb is None else float(limit_mb)
        self.source: Optional[str] = (
            source if source is not None else ("explicit" if limit_mb is not None else None)
        )
        self.peak_mb: Optional[float] = None
        self.interval_peak_mb: Optional[float] = None
        self.exceeded: Optional[MemoryBudgetExceeded] = None
        self._tripped: Optional[tuple[float, Any]] = None
        self._warned = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._resolved = limit_mb is not None

    # -- limit ---------------------------------------------------------------

    def _resolve(self) -> None:
        if self._resolved:
            return
        info = memory_budget(self._environ)
        self.limit_mb, self.source = info.limit_mb, info.source
        self._resolved = True

    @property
    def threshold_mb(self) -> Optional[float]:
        """``fraction x limit_mb``, or ``None`` without a limit."""
        self._resolve()
        return None if self.limit_mb is None else self.fraction * self.limit_mb

    # -- sampling ------------------------------------------------------------

    def sample(self) -> Optional[float]:
        """Read RSS now and fold it into ``peak_mb``; never raises."""
        current = rss_mb()
        if current is not None:
            with self._lock:
                if self.peak_mb is None or current > self.peak_mb:
                    self.peak_mb = current
                if self.interval_peak_mb is None or current > self.interval_peak_mb:
                    self.interval_peak_mb = current
        return current

    def start_interval(self) -> Optional[float]:
        """Start a new ``interval_peak_mb`` window; returns the one that ended.

        The new window opens at the current RSS, so an interval's peak is never
        below what it started with.
        """
        with self._lock:
            ended, self.interval_peak_mb = self.interval_peak_mb, None
        self.sample()
        return ended

    def _sampler(self) -> None:
        assert self.interval_s is not None
        while not self._stop.wait(self.interval_s):
            current = self.sample()
            threshold = self.threshold_mb
            if current is not None and threshold is not None and current > threshold:
                with self._lock:
                    if self._tripped is None:
                        self._tripped = (current, "sampler")

    def check(self, label: Any = None) -> Optional[float]:
        """Raise :class:`MemoryBudgetExceeded` if RSS is above the threshold.

        Also raises when the sampler thread saw a crossing since the last
        check. Returns the current RSS in MiB (``None`` when unreadable).
        """
        current = self.sample()
        threshold = self.threshold_mb
        if threshold is None:
            return current
        with self._lock:
            tripped, self._tripped = self._tripped, None
        observed = None
        if current is not None and current > threshold:
            observed = current
        elif tripped is not None:
            observed = tripped[0]
        if observed is None:
            return current
        error = MemoryBudgetExceeded(
            observed, threshold, float(self.limit_mb),
            label=label, source=self.source, peak_mb=self.peak_mb,
        )
        self.exceeded = error
        if self.on_exceed == "raise":
            raise error
        if not self._warned:
            warnings.warn(str(error), ResourceWarning, stacklevel=2)
            self._warned = True
        return current

    # -- context -------------------------------------------------------------

    def __enter__(self) -> "MemoryBudget":
        self._resolve()
        self.sample()
        if self.interval_s is not None and self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._sampler, name="vaft-memory-budget", daemon=True
            )
            self._thread.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 2 * (self.interval_s or 0.0)))
            self._thread = None
        self.sample()
        with self._lock:
            tripped = self._tripped
        if tripped is not None and self.exceeded is None and self.limit_mb is not None:
            self.exceeded = MemoryBudgetExceeded(
                tripped[0], float(self.threshold_mb), self.limit_mb,
                label=tripped[1], source=self.source, peak_mb=self.peak_mb,
            )

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready summary for status records."""
        return {
            "limit_mb": self.limit_mb,
            "source": self.source,
            "fraction": self.fraction,
            "threshold_mb": self.threshold_mb,
            "peak_mb": self.peak_mb,
            "exceeded": self.exceeded is not None,
        }
