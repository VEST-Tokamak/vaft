"""Snakemake helpers: a rule-script logger, and what a run did, read back from its log.

:func:`init_snakemake_logger` is for ``script:`` files. The log reader is for
whoever runs Snakemake (the routine worker, a campaign lane): Snakemake prints a
timestamp line before every job start and every finish, so one log gives each
job's rule, wildcards, threads and the interval it held a slot. From that: how
busy the run kept its cores (slot occupancy), what each rule cost in
job-seconds, and how long run-scoped checkpoints held the whole run while every
other core idled.

Slot occupancy is Snakemake's view: a slot is counted from the job's start line
to its "Finished" line, whether or not the job's process was on a CPU. The CPU
the processes actually used comes from :mod:`vaft.code.accounting`; the two
together separate an idle scheduler from waiting jobs.
"""

import collections
import logging
import pathlib
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Union

class _StreamToLogger:
    """File-like object that forwards writes to a logger."""
    def __init__(self, logger: logging.Logger, level: int):
        self._logger, self._level = logger, level
    def write(self, message: str):
        msg = message.rstrip()
        if msg:
            self._logger.log(self._level, msg)
    def flush(self):   # 필요 인터페이스
        pass

def init_snakemake_logger(snakemake, *,
                          name: str = "vaft",
                          std_opt: bool = True) -> logging.Logger:
    """
    Create (or reuse) a logger that writes to both console and the rule's
    log file.  
    Parameters
    ----------
    snakemake : object
        Snakemake-injected object available inside `script:` files.
    name : str
        Logger name (default "vaft").
    std_opt : bool
        If True  → redirect sys.stdout → logger at INFO level,
                   redirect sys.stderr → logger at ERROR level.
        If False → leave stdout/stderr unchanged.
    """
    # ---- 1) build / reuse logger -----------------------------------------
    log_path = pathlib.Path(snakemake.log[0])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:         # idempotent: add handlers only once
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        sh = logging.StreamHandler(sys.stdout)  # console
        sh.setFormatter(fmt)
        logger.addHandler(sh)

        fh = logging.FileHandler(log_path)      # rule log file
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    # ---- 2) optional stdout/stderr redirect ------------------------------
    if std_opt:
        sys.stdout = _StreamToLogger(logger, logging.INFO)
        sys.stderr = _StreamToLogger(logger, logging.ERROR)

    return logger


__all__ = [
    "SnakemakeJob",
    "init_snakemake_logger",
    "occupancy",
    "parse_log",
    "rule_costs",
    "run_span",
]

_TIMESTAMP = re.compile(r"^\[(\w{3} \w{3} +\d+ \d\d:\d\d:\d\d \d{4})\]$")
_START = re.compile(r"^(?:local)?(?:rule|checkpoint) (\w+):$")
_JOBID = re.compile(r"^\s+jobid: (\d+)$")
_WILDCARDS = re.compile(r"^\s+wildcards: (.*)$")
_THREADS = re.compile(r"^\s+threads: (\d+)$")
_FINISHED = re.compile(r"^Finished job(?:id:)? (\d+)")  # Snakemake 9 / 8
_ERROR = re.compile(r"^Error in rule (\w+):")


@dataclass(frozen=True)
class SnakemakeJob:
    """One job of a run: its rule, wildcards and the interval it held a slot."""

    jobid: int
    rule: str
    wildcards: str
    threads: int
    start: float
    end: Optional[float]
    failed: bool = False

    @property
    def seconds(self) -> Optional[float]:
        return None if self.end is None else max(self.end - self.start, 0.0)


def parse_log(source: Union[str, Path, Iterable[str]]) -> list[SnakemakeJob]:
    """Every job one Snakemake log started, finished or failed, in start order.

    Job ids restart in every run appended to the same file, so a job is closed
    by the next "Finished"/"Error" for its id after it started. A job still open
    at the end of the log (a killed run) has ``end=None``.
    """
    if isinstance(source, Path) or (isinstance(source, str) and "\n" not in source):
        lines = Path(source).read_text(encoding="utf-8", errors="replace").splitlines()
    elif isinstance(source, str):
        lines = source.splitlines()
    else:
        lines = list(source)
    now: Optional[float] = None
    pending: Optional[dict] = None
    pending_id: Optional[int] = None
    running: dict[int, dict] = {}
    jobs: list[SnakemakeJob] = []
    for index, line in enumerate(lines):
        stamp = _TIMESTAMP.match(line)
        if stamp:
            now = datetime.strptime(stamp.group(1), "%a %b %d %H:%M:%S %Y").timestamp()
            continue
        started = _START.match(line)
        if started and now is not None:
            pending = {"rule": started.group(1), "start": now, "wildcards": "", "threads": 1}
            pending_id = None
            continue
        if pending is not None:
            # wildcards/threads may come before or after jobid: they fill the
            # pending header, which running[...] refers to once the id is known.
            jobid = _JOBID.match(line)
            if jobid and pending_id is None:
                pending_id = int(jobid.group(1))
                if pending_id in running:
                    # An appended run reused the id of a job the last one never
                    # finished (it was killed): that job ends open, not lost.
                    jobs.append(_close(pending_id, running.pop(pending_id), None, failed=False))
                running[pending_id] = pending
                continue
            wildcards = _WILDCARDS.match(line)
            if wildcards:
                pending["wildcards"] = wildcards.group(1)
                continue
            threads = _THREADS.match(line)
            if threads:
                pending["threads"] = int(threads.group(1))
                continue
        finished = _FINISHED.match(line)
        if finished and now is not None:
            jobid = int(finished.group(1))
            if jobid in running:
                jobs.append(_close(jobid, running.pop(jobid), now, failed=False))
            pending = None
            continue
        error = _ERROR.match(line)
        if error and now is not None:
            # The error block runs until the next timestamp or blank line; its
            # message may span several lines before the jobid.
            for follow in lines[index + 1:]:
                if _TIMESTAMP.match(follow) or not follow.strip():
                    break
                found = _JOBID.match(follow)
                if found and int(found.group(1)) in running:
                    jobid = int(found.group(1))
                    jobs.append(_close(jobid, running.pop(jobid), now, failed=True))
                    break
            pending = None
    jobs.extend(_close(jobid, job, None, failed=False) for jobid, job in running.items())
    return sorted(jobs, key=lambda job: (job.start, job.jobid))


def _close(jobid: int, job: dict, end: Optional[float], *, failed: bool) -> SnakemakeJob:
    return SnakemakeJob(
        jobid=jobid, rule=job["rule"], wildcards=job["wildcards"], threads=job["threads"],
        start=job["start"], end=end, failed=failed,
    )


def run_span(jobs: Iterable[SnakemakeJob]) -> Optional[tuple[float, float]]:
    """``(first start, last end)`` of a run's jobs, or ``None`` without a finished job."""
    jobs = [job for job in jobs if job.end is not None]
    if not jobs:
        return None
    return min(job.start for job in jobs), max(job.end for job in jobs)


def occupancy(jobs: Iterable[SnakemakeJob], cores: int) -> dict[str, float]:
    """How busy a run kept its ``cores`` slots, as fractions of its span.

    ``slot_occupancy`` is the mean of (threads held / cores); ``below_quarter``
    the fraction of the span with at most a quarter of the slots held (the
    serial checkpoints and the chunk tail).
    """
    jobs = [job for job in jobs if job.end is not None]
    span = run_span(jobs)
    if span is None or span[1] <= span[0] or cores <= 0:
        return {"slot_occupancy": 0.0, "below_quarter": 1.0, "span_s": 0.0}
    events = sorted([(job.start, min(job.threads, cores)) for job in jobs]
                    + [(job.end, -min(job.threads, cores)) for job in jobs])
    held, last, busy, sparse = 0, span[0], 0.0, 0.0
    for time, delta in events:
        busy += held * (time - last)
        if held <= cores / 4:
            sparse += time - last
        held += delta
        last = time
    width = span[1] - span[0]
    return {"slot_occupancy": busy / (width * cores), "below_quarter": sparse / width, "span_s": width}


def rule_costs(jobs: Iterable[SnakemakeJob]) -> dict[str, dict[str, float]]:
    """Per rule: job count, job-seconds (slot-seconds x threads), median and max seconds."""
    by_rule: dict[str, list[SnakemakeJob]] = collections.defaultdict(list)
    for job in jobs:
        if job.end is not None:
            by_rule[job.rule].append(job)
    costs = {}
    for rule, members in sorted(by_rule.items()):
        seconds = sorted(job.seconds for job in members)
        costs[rule] = {
            "jobs": len(members),
            "job_seconds": sum(job.seconds * job.threads for job in members),
            "median_s": seconds[len(seconds) // 2],
            "max_s": seconds[-1],
            "failed": sum(job.failed for job in members),
        }
    return costs
