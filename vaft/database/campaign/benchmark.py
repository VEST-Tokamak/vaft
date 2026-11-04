"""One report of what a lane's chunks cost: CPU, wall, memory, throughput, per rule.

Read from the files a lane already writes -- ``lane.log`` (the per-chunk line
``run_lane.sh`` logs, with the job cgroup's ``cpu_s``/``cpu_util``/memory),
``logs/<chunk>.cgroup.tsv`` and ``logs/<chunk>.snakemake.log`` -- so it costs no
extra instrumentation and works on a lane that has finished, failed or is still
running.

The numbers an administrator asks about are ``cpu_util`` (CPU-seconds over
wall-seconds times booked cores, from the cgroup), ``booked_cpu_h_per_shot`` and
``shots_per_hour``. ``slot_occupancy`` (Snakemake's own view) and the per-rule
job-seconds say where the gap between booked and used went: serial checkpoints,
a chunk's tail, or jobs that held a slot while waiting.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional, Union

from .smklog import occupancy, parse_log, rule_costs

__all__ = ["compare_reports", "lane_report"]

PathLike = Union[str, Path]

_START = re.compile(r"^(\S+ \S+) (shots\.\d{3}) start \((\d+) shots\)")
_EXIT = re.compile(r"^(\S+ \S+) (shots\.\d{3}) exit=(-?\d+) wall=(\d+)s(.*?) log=")
_LANE_START = re.compile(r"lane start .* cores=(\d+) chunk=(\d+)")
_FIELD = re.compile(r"(\w+)=([-0-9.]+)")

#: Rules that only render validation PNGs (``validation_plots``).
PLOT_RULE_PREFIX = "plot_"
#: The run-scoped checkpoints every shot waits for.
CHECKPOINT_RULES = ("preflight_raw_dumps", "select_impa_shots")


def _lane_log(lane: Path) -> tuple[Optional[int], list[dict[str, Any]]]:
    cores: Optional[int] = None
    chunks: dict[str, dict[str, Any]] = {}
    try:
        lines = (lane / "lane.log").read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return None, []
    for line in lines:
        started = _LANE_START.search(line)
        if started:
            cores = int(started.group(1))
        start = _START.match(line)
        if start:
            chunks[start.group(2)] = {"chunk": start.group(2), "shots": int(start.group(3)), "cores": cores}
            continue
        end = _EXIT.match(line)
        if end and end.group(2) in chunks:
            entry = chunks[end.group(2)]
            entry.update(exit=int(end.group(3)), wall_s=float(end.group(4)))
            entry.update({key: float(value) for key, value in _FIELD.findall(end.group(5))})
    return cores, list(chunks.values())


def lane_report(lane: PathLike) -> dict[str, Any]:
    """Per-chunk and total cost of one lane directory (finished chunks only)."""
    lane = Path(lane)
    cores, chunks = _lane_log(lane)
    rules: dict[str, dict[str, float]] = {}
    for chunk in chunks:
        log = lane / "logs" / f"{chunk['chunk']}.snakemake.log"
        if not log.exists() or "wall_s" not in chunk:
            continue
        jobs = parse_log(log)
        chunk_cores = int(chunk.get("cores") or cores or 1)
        chunk.update(occupancy(jobs, chunk_cores))
        costs = rule_costs(jobs)
        chunk["job_seconds"] = sum(cost["job_seconds"] for cost in costs.values())
        chunk["plot_job_seconds"] = sum(
            cost["job_seconds"] for rule, cost in costs.items() if rule.startswith(PLOT_RULE_PREFIX)
        )
        # Wall time the checkpoints held the run (not job-seconds: the parallel
        # preflight books every core).
        chunk["checkpoint_s"] = sum(job.seconds for job in jobs if job.rule in CHECKPOINT_RULES and job.end is not None)
        for rule, cost in costs.items():
            total = rules.setdefault(rule, {"jobs": 0, "job_seconds": 0.0, "max_s": 0.0, "failed": 0})
            total["jobs"] += cost["jobs"]
            total["job_seconds"] += cost["job_seconds"]
            total["max_s"] = max(total["max_s"], cost["max_s"])
            total["failed"] += cost["failed"]
    done = [chunk for chunk in chunks if "wall_s" in chunk]
    shots = sum(chunk["shots"] for chunk in done)
    wall = sum(chunk["wall_s"] for chunk in done)
    booked = sum(chunk["wall_s"] * int(chunk.get("cores") or cores or 0) for chunk in done)
    measured = [chunk for chunk in done if "cpu_s" in chunk]
    cpu = sum(chunk["cpu_s"] for chunk in measured)
    measured_booked = sum(chunk["wall_s"] * int(chunk.get("cores") or cores or 0) for chunk in measured)
    job_seconds = sum(chunk.get("job_seconds", 0.0) for chunk in done)
    totals = {
        "chunks": len(done),
        "shots": shots,
        "wall_h": wall / 3600,
        "shots_per_hour": shots / (wall / 3600) if wall else None,
        "booked_cpu_h": booked / 3600,
        "booked_cpu_h_per_shot": booked / 3600 / shots if shots else None,
        "cpu_h": cpu / 3600 if measured else None,
        "cpu_util": cpu / measured_booked if measured_booked else None,
        "cpu_h_per_shot": cpu / 3600 / sum(c["shots"] for c in measured) if measured else None,
        "mem_max_gb": max((chunk.get("chunk_mem_max_gb", 0.0) for chunk in done), default=None),
        "slot_occupancy": (
            sum(chunk.get("slot_occupancy", 0.0) * chunk["wall_s"] for chunk in done) / wall if wall else None
        ),
        "plot_share": sum(c.get("plot_job_seconds", 0.0) for c in done) / job_seconds if job_seconds else None,
        "checkpoint_share": sum(c.get("checkpoint_s", 0.0) for c in done) / wall if wall else None,
    }
    return {"lane": str(lane), "cores": cores, "totals": totals, "chunks": chunks, "rules": rules}


def compare_reports(reports: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """One row of totals per labelled report, for a side-by-side table."""
    keys = ("shots", "wall_h", "shots_per_hour", "cpu_util", "booked_cpu_h_per_shot", "cpu_h_per_shot",
            "mem_max_gb", "slot_occupancy", "plot_share", "checkpoint_share")
    return [{"label": label, **{key: report["totals"].get(key) for key in keys}} for label, report in reports.items()]
