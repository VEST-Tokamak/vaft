"""One report of what a lane's chunks cost: CPU, wall, memory, throughput, per rule.

Read from what :func:`~.run.run_lane` records -- ``lane.jsonl`` (per chunk:
shots, exit, :class:`~vaft.code.accounting.ResourceUsage`) and
``logs/<chunk>.snakemake.log`` -- so it works on a lane that finished, failed or
is still running, wherever it ran.

The numbers an administrator asks about are ``cpu_util`` (CPU-seconds over
wall-seconds times booked cores), ``booked_cpu_h_per_shot`` and
``shots_per_hour``. ``slot_occupancy`` (Snakemake's own view) and the per-rule
job-seconds say where the gap between booked and used went: run-scoped
checkpoints, a chunk's tail, or jobs that held a slot while waiting. Which rules
are checkpoints or only draw figures is the workflow's to say
(:class:`~.spec.WorkflowSpec`); without one those shares are not reported.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

from vaft.code.snakemake import occupancy, parse_log, rule_costs

from .spec import WorkflowSpec

__all__ = ["compare_reports", "lane_report"]

PathLike = Union[str, Path]


def _records(lane: Path) -> list[dict[str, Any]]:
    try:
        lines = (lane / "lane.jsonl").read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    records = []
    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict) and "chunk" in entry and "usage" in entry:
            records.append(entry)
    return records


def lane_report(lane: PathLike, spec: Optional[WorkflowSpec] = None) -> dict[str, Any]:
    """Per-chunk and total cost of one lane directory."""
    lane = Path(lane)
    checkpoints = set(spec.checkpoint_rules) if spec else set()
    plot_prefix = spec.plot_rule_prefix if spec else None
    rules: dict[str, dict[str, float]] = {}
    chunks: list[dict[str, Any]] = []
    for record in _records(lane):
        usage = record.get("usage") or {}
        chunk: dict[str, Any] = {
            "chunk": record["chunk"], "shots": int(record.get("ran") or 0), "exit": record.get("returncode"),
            "done": record.get("done"), "cores": int(record.get("cores") or usage.get("booked_cpus") or 0),
            "wall_s": float(usage.get("wall_s") or 0.0), "cpu_s": usage.get("cpu_s"),
            "cpu_util": usage.get("utilization"), "mem_max_mb": usage.get("peak_mem_mb"),
            "max_rss_mb": usage.get("max_process_rss_mb"), "source": usage.get("source"),
        }
        log = lane / "logs" / str(record.get("log") or "")
        if record.get("log") and log.is_file():
            jobs = parse_log(log)
            chunk.update(occupancy(jobs, chunk["cores"] or 1))
            costs = rule_costs(jobs)
            chunk["job_seconds"] = sum(cost["job_seconds"] for cost in costs.values())
            if plot_prefix:
                chunk["plot_job_seconds"] = sum(
                    cost["job_seconds"] for rule, cost in costs.items() if rule.startswith(plot_prefix)
                )
            if checkpoints:
                # Wall time the checkpoints held the run, not job-seconds.
                chunk["checkpoint_s"] = sum(
                    job.seconds for job in jobs if job.rule in checkpoints and job.end is not None
                )
            for rule, cost in costs.items():
                total = rules.setdefault(rule, {"jobs": 0, "job_seconds": 0.0, "max_s": 0.0, "failed": 0})
                total["jobs"] += cost["jobs"]
                total["job_seconds"] += cost["job_seconds"]
                total["max_s"] = max(total["max_s"], cost["max_s"])
                total["failed"] += cost["failed"]
        chunks.append(chunk)
    shots = sum(chunk["shots"] for chunk in chunks)
    wall = sum(chunk["wall_s"] for chunk in chunks)
    booked = sum(chunk["wall_s"] * chunk["cores"] for chunk in chunks)
    measured = [chunk for chunk in chunks if chunk["cpu_s"] is not None]
    cpu = sum(chunk["cpu_s"] for chunk in measured)
    measured_booked = sum(chunk["wall_s"] * chunk["cores"] for chunk in measured)
    job_seconds = sum(chunk.get("job_seconds", 0.0) for chunk in chunks)
    totals = {
        "chunks": len(chunks),
        "shots": shots,
        "wall_h": wall / 3600,
        "shots_per_hour": shots / (wall / 3600) if wall else None,
        "booked_cpu_h": booked / 3600,
        "booked_cpu_h_per_shot": booked / 3600 / shots if shots else None,
        "cpu_h": cpu / 3600 if measured else None,
        "cpu_util": cpu / measured_booked if measured_booked else None,
        "cpu_h_per_shot": cpu / 3600 / shots if measured and shots else None,
        "mem_max_gb": max((c["mem_max_mb"] / 1024 for c in chunks if c["mem_max_mb"]), default=None),
        "max_rss_gb": max((c["max_rss_mb"] / 1024 for c in chunks if c["max_rss_mb"]), default=None),
        "slot_occupancy": (
            sum(chunk.get("slot_occupancy", 0.0) * chunk["wall_s"] for chunk in chunks) / wall if wall else None
        ),
        "plot_share": (sum(c.get("plot_job_seconds", 0.0) for c in chunks) / job_seconds
                       if plot_prefix and job_seconds else None),
        "checkpoint_share": (sum(c.get("checkpoint_s", 0.0) for c in chunks) / wall
                             if checkpoints and wall else None),
    }
    return {"lane": str(lane), "totals": totals, "chunks": chunks, "rules": rules}


def compare_reports(reports: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """One row of totals per labelled report, for a side-by-side table."""
    keys = ("shots", "wall_h", "shots_per_hour", "cpu_util", "booked_cpu_h_per_shot", "cpu_h_per_shot",
            "mem_max_gb", "max_rss_gb", "slot_occupancy", "plot_share", "checkpoint_share")
    return [{"label": label, **{key: report["totals"].get(key) for key in keys}} for label, report in reports.items()]
