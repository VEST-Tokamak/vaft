"""What a launch actually used: CPU time, utilization and peak memory.

A scheduler's own view of usage is not enough on the clusters VAFT runs on.
``pestat``'s CPUload is a load average, and ``sacct``'s ``TotalCPU`` is lost when
a job is cancelled (every cancelled job of a 2026-11 regeneration reported 0 s). The job's
cgroup is the one record that is always there: ``cpu.stat`` counts the CPU time
of every process the job ever ran, and ``memory.peak`` its largest footprint.
That footprint includes the page cache the job's file reads filled: on a node
that does not constrain RAM (Slurm ``ConstrainRAMSpace=no``) it can exceed the
booked ``--mem`` many times over while the processes' resident memory stays
far below it (one regeneration lane: 160 GB ``memory.peak``, 20 GB sacct ``MaxRSS``, 64 GB
booked). Compare ``peak_mem_mb`` with a memory request only together with RSS.
This module reads it, and gives batch job scripts the few shell lines that
write it next to their exit record, where a cancel cannot erase it.

``ResourceUsage.utilization`` is CPU-seconds over wall-seconds times the CPUs
booked: what an administrator means by "the job used its allocation".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Union

PathLike = Union[str, Path]

__all__ = [
    "NODE_FAIL_STATES",
    "OUT_OF_MEMORY_STATES",
    "ResourceUsage",
    "USAGE_RECORD",
    "UsageMeter",
    "cgroup_cpu_seconds",
    "cgroup_memory_peak_mb",
    "job_cgroup_dir",
    "parse_usage_record",
    "usage_record_lines",
]

_MIB = 1024.0 * 1024.0

#: The scratch file a batch job script writes its cgroup samples into.
USAGE_RECORD = "usage"

#: Slurm states that mean the job's memory limit stopped the program.
OUT_OF_MEMORY_STATES = frozenset({"OUT_OF_MEMORY"})
#: Slurm states that mean the node or the scheduler, not the program, ended it.
NODE_FAIL_STATES = frozenset({"NODE_FAIL", "PREEMPTED", "BOOT_FAIL"})


@dataclass(frozen=True)
class ResourceUsage:
    """CPU and memory one launch used, and where the numbers came from.

    ``source`` is ``"cgroup"`` (the job's cgroup, read by the job itself),
    ``"sacct"`` (scheduler accounting) or ``"rusage"`` (the waited-for children
    of this process, :func:`resource.getrusage`). Any number the source did not provide is ``None``.
    """

    wall_s: Optional[float]
    cpu_s: Optional[float]
    booked_cpus: Optional[int]
    source: str
    user_s: Optional[float] = None
    system_s: Optional[float] = None
    peak_mem_mb: Optional[float] = None
    #: Resident memory of the largest single process (``getrusage``), MiB.
    max_process_rss_mb: Optional[float] = None
    start_epoch: Optional[float] = None
    end_epoch: Optional[float] = None

    @property
    def utilization(self) -> Optional[float]:
        """CPU-seconds / (wall-seconds x booked CPUs); ``None`` when undefined."""
        if self.cpu_s is None or not self.wall_s or not self.booked_cpus:
            return None
        return self.cpu_s / (self.wall_s * self.booked_cpus)

    def as_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["utilization"] = self.utilization
        return record


def job_cgroup_dir(
    proc_cgroup: PathLike = "/proc/self/cgroup", cgroup_root: PathLike = "/sys/fs/cgroup"
) -> Optional[Path]:
    """The Slurm job cgroup (``.../job_<id>``) above this process, or ``None``.

    cgroup v2 only. The process runs in a step/task cgroup below the job's, and
    the job's is the one that counts every step.
    """
    from .resources import _proc_cgroup_paths

    v2, _ = _proc_cgroup_paths(Path(proc_cgroup))
    if not v2:
        return None
    directory = Path(cgroup_root) / v2.strip("/")
    for candidate in (directory, *directory.parents):
        if candidate.name.startswith("job_"):
            return candidate
        if candidate == Path(cgroup_root):
            break
    return None


def cgroup_cpu_seconds(directory: PathLike) -> Optional[tuple[float, float, float]]:
    """``(usage, user, system)`` CPU seconds from a cgroup's ``cpu.stat``."""
    try:
        text = (Path(directory) / "cpu.stat").read_text()
    except OSError:
        return None
    values = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].isdigit():
            values[parts[0]] = int(parts[1]) / 1e6
    if "usage_usec" not in values:
        return None
    return values["usage_usec"], values.get("user_usec", 0.0), values.get("system_usec", 0.0)


def cgroup_memory_peak_mb(directory: PathLike) -> Optional[float]:
    """``memory.peak`` of a cgroup in MiB (kernel >= 5.19), else ``None``.

    Page cache included: an upper bound on resident memory, not a measure of it.
    """
    try:
        text = (Path(directory) / "memory.peak").read_text().strip()
    except OSError:
        return None
    return int(text) / _MIB if text.isdigit() else None


def usage_record_lines(record: str) -> list[str]:
    """Shell lines that define ``vaft_usage_sample``, appending one sample to ``record``.

    A sample is ``<epoch> <usage_usec> <user_usec> <system_usec> <memory.peak>``
    of the job cgroup, with ``-`` where a value is unavailable. Outside a job
    cgroup the function writes nothing, so a script on a host without cgroup v2
    keeps working.
    """
    return [
        "vaft_job_cg=",
        'if [ -r /proc/self/cgroup ]; then vaft_cg=/sys/fs/cgroup$(sed -n "s/^0:://p" /proc/self/cgroup)',
        '  while [ -n "$vaft_cg" ] && [ "$vaft_cg" != /sys/fs/cgroup ]; do',
        '    case "${vaft_cg##*/}" in job_*) vaft_job_cg=$vaft_cg; break ;; esac; vaft_cg=${vaft_cg%/*}',
        "  done; fi",
        "vaft_usage_sample() {",
        '  [ -n "$vaft_job_cg" ] && [ -r "$vaft_job_cg/cpu.stat" ] || return 0',
        '  awk -v t="$(date +%s)" -v peak="$(cat "$vaft_job_cg/memory.peak" 2>/dev/null || echo -)" \\',
        "    '$1 == \"usage_usec\" {u = $2} $1 == \"user_usec\" {s = $2} $1 == \"system_usec\" {y = $2}",
        "     END {print t, (u == \"\" ? \"-\" : u), (s == \"\" ? \"-\" : s), (y == \"\" ? \"-\" : y), peak}' \\",
        f'    "$vaft_job_cg/cpu.stat" >> {record}',
        "}",
    ]


def _number(text: str) -> Optional[float]:
    return None if text == "-" else float(text)


def parse_usage_record(text: Optional[str], booked_cpus: Optional[int]) -> Optional[ResourceUsage]:
    """A :class:`ResourceUsage` from the first and last samples of a usage record.

    The record is what :func:`usage_record_lines` wrote; ``None`` without two
    readable samples. The job cgroup outlives the program, so the difference
    between the samples is the program's own CPU time (the script's ``date``
    and ``awk`` are a rounding error).
    """
    samples = []
    for line in (text or "").splitlines():
        fields = line.split()
        if len(fields) != 5:
            continue
        try:
            samples.append([float(fields[0])] + [_number(value) for value in fields[1:]])
        except ValueError:
            continue
    if len(samples) < 2:
        return None
    first, last = samples[0], samples[-1]

    def delta(index: int) -> Optional[float]:
        if first[index] is None or last[index] is None:
            return None
        return (last[index] - first[index]) / 1e6

    peaks = [sample[4] for sample in samples if sample[4] is not None]
    return ResourceUsage(
        wall_s=max(last[0] - first[0], 0.0),
        cpu_s=delta(1),
        user_s=delta(2),
        system_s=delta(3),
        booked_cpus=booked_cpus,
        peak_mem_mb=max(peaks) / _MIB if peaks else None,
        source="cgroup",
        start_epoch=first[0],
        end_epoch=last[0],
    )


class UsageMeter:
    """What the commands run between :meth:`start` and :meth:`stop` used, anywhere.

    Inside a Slurm job (a cgroup v2 ``job_*`` directory above this process) CPU
    time is the job cgroup's ``cpu.stat`` difference, and memory the largest
    ``memory.current`` sampled in between (page cache included, see the module
    docstring). Anywhere else -- a workstation, a server without a scheduler --
    CPU time is :func:`resource.getrusage` of the waited-for children, which
    covers a whole process tree once each parent has reaped its own. Either way
    ``max_process_rss_mb`` is the largest resident peak of a child reaped in the
    interval -- known only when it raised the process's lifetime child peak
    (``getrusage`` keeps no per-interval peak); otherwise ``None``. On Linux a
    forked child counts its parent's pages until it execs, so the figure is never
    below the launching process's own size.

    The meter measures *this* process's children: start it before launching the
    command and stop it after waiting for it. ``booked_cpus`` is what the run
    reserved (the scheduler allocation, or the cores it was told to use).
    """

    def __init__(self, booked_cpus: Optional[int], *, sample_s: float = 30.0, cgroup: Any = "auto") -> None:
        self.booked_cpus = booked_cpus
        self.sample_s = sample_s
        self.cgroup = job_cgroup_dir() if cgroup == "auto" else (Path(cgroup) if cgroup else None)
        self._start: Optional[tuple[float, Optional[tuple[float, float, float]], tuple[float, float]]] = None
        self._mem_max: Optional[float] = None
        self._stop_sampling = None
        self._sampler = None

    @staticmethod
    def _rusage() -> tuple[float, float]:
        import resource

        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        return usage.ru_utime, usage.ru_stime

    @staticmethod
    def _children_max_rss_mb() -> Optional[float]:
        import resource
        import sys

        rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        if not rss:
            return None
        return rss / _MIB if sys.platform == "darwin" else rss / 1024.0  # bytes on macOS, KiB on Linux

    def _interval_max_rss_mb(self) -> Optional[float]:
        after = self._children_max_rss_mb()
        before = getattr(self, "_rss_before", None)
        if after is None or (before is not None and after <= before):
            return None
        return after

    def _sample_memory(self) -> None:
        try:
            text = (self.cgroup / "memory.current").read_text().strip()
        except (OSError, TypeError):
            return
        if text.isdigit():
            value = int(text) / _MIB
            self._mem_max = value if self._mem_max is None else max(self._mem_max, value)

    def start(self) -> "UsageMeter":
        import threading
        import time

        cpu = cgroup_cpu_seconds(self.cgroup) if self.cgroup else None
        self._start = (time.time(), cpu, self._rusage())
        self._rss_before = self._children_max_rss_mb()
        self._mem_max = None
        if cpu is not None and self.sample_s > 0:
            self._sample_memory()
            self._stop_sampling = threading.Event()

            def loop() -> None:
                while not self._stop_sampling.wait(self.sample_s):
                    self._sample_memory()

            self._sampler = threading.Thread(target=loop, name="vaft-usage-meter", daemon=True)
            self._sampler.start()
        return self

    def stop(self) -> ResourceUsage:
        import time

        if self._start is None:
            raise RuntimeError("UsageMeter.stop() before start()")
        if self._stop_sampling is not None:
            self._stop_sampling.set()
            self._sampler.join()
            self._stop_sampling = self._sampler = None
        t0, cpu0, rusage0 = self._start
        now = time.time()
        cpu1 = cgroup_cpu_seconds(self.cgroup) if cpu0 is not None else None
        if cpu0 is not None and cpu1 is not None:
            self._sample_memory()
            usage, user, system = (b - a for a, b in zip(cpu0, cpu1))
            source, peak = "cgroup", self._mem_max
        else:
            user, system = (b - a for a, b in zip(rusage0, self._rusage()))
            usage, source, peak = user + system, "rusage", None
        return ResourceUsage(
            wall_s=now - t0,
            cpu_s=usage,
            user_s=user,
            system_s=system,
            booked_cpus=self.booked_cpus,
            peak_mem_mb=peak,
            max_process_rss_mb=self._interval_max_rss_mb(),
            source=source,
            start_epoch=t0,
            end_epoch=now,
        )

    def __enter__(self) -> "UsageMeter":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        if self._start is not None and self._stop_sampling is not None:
            self._stop_sampling.set()
