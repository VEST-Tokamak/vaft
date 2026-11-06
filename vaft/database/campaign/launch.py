"""Start a run's lanes through a :mod:`vaft.code` backend: local processes or Slurm jobs.

Each lane becomes one :class:`~vaft.code.execution.ExecutionRequest` running
``python -m vaft.database.campaign run-lane <lane> ...`` with the lane's cores
and memory, submitted with :func:`~vaft.code.execution.submit_request`. On
:class:`~vaft.code.slurm.SlurmBackend` that is one batch job per lane, checked
by the execution policy (:mod:`vaft.code.policy`) before ``sbatch``, measured by
the job script; on :class:`~vaft.code.execution.LocalBackend` a child process per
lane. Where the jobs go -- partition, account, QOS, cluster, approval -- is the
caller's (command line or ``VAFT_SLURM_*``), never this package's.

A site that must prepare the lane process first (a pinned environment, a mount
namespace) passes ``lane_command``: the job then runs ``<lane_command> <lane>``,
a site script that ends in ``run-lane``. The job id of each submission is kept in
``<lane>/job``; a lane whose recorded job the backend still reports queued or
running is not submitted again.

Inside a lane, Snakemake hands each rule its own thread count, so the BLAS and
OpenMP thread variables default to 1 for the lane process (an explicit value in
the caller's ``env`` wins).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

from vaft.code.execution import THREAD_ENV_VARIABLES, ExecutionRequest, JobHandle, ResourceRequest, submit_request

from .lanes import LaneState

__all__ = ["lane_dirs", "lane_request", "launch_lanes"]

PathLike = Union[str, Path]


def lane_dirs(run_dir: PathLike) -> list[Path]:
    """The lane directories of a run (``lane0``, ``lane1``, ...), in order."""
    found = [path for path in Path(run_dir).glob("lane*") if path.is_dir() and path.name[4:].isdigit()]
    return sorted(found, key=lambda path: int(path.name[4:]))


def lane_request(
    lane: PathLike,
    run_lane_args: Sequence[str],
    *,
    cores: int,
    memory_mb: Optional[int] = None,
    timeout: Optional[float] = None,
    env: Optional[Mapping[str, str]] = None,
    python: Optional[str] = None,
    lane_command: Sequence[str] = (),
) -> ExecutionRequest:
    lane = Path(lane).resolve()
    overlay = {variable: "1" for variable in THREAD_ENV_VARIABLES}
    overlay.update(env or {})
    if lane_command:
        command = [*lane_command, str(lane)]
    else:
        command = [python or sys.executable, "-m", "vaft.database.campaign", "run-lane", str(lane),
                   "--cores", str(cores), *run_lane_args]
    return ExecutionRequest(
        command=command,
        workdir=lane,
        env=overlay,
        timeout=timeout,
        log_path=lane / "job.log",
        resources=ResourceRequest(ntasks=1, threads_per_task=cores, memory_mb=memory_mb),
        label=f"campaign-{lane.parent.name}-{lane.name}",
    )


def launch_lanes(
    run_dir: PathLike,
    backend: object,
    run_lane_args: Sequence[str],
    *,
    cores: int,
    memory_mb: Optional[int] = None,
    timeout: Optional[float] = None,
    env: Optional[Mapping[str, str]] = None,
    lanes: Optional[Sequence[str]] = None,
    lane_command: Sequence[str] = (),
) -> dict[str, JobHandle]:
    """Submit every lane of ``run_dir`` with pending chunks (or the named ``lanes``).

    A lane is skipped when it is finished, holds a ``STOP`` file, or its recorded
    job is still queued or running -- or the backend cannot tell (a duplicate
    runner would only wait on the lane lock, but it would hold an allocation).
    """
    handles: dict[str, JobHandle] = {}
    for lane in lane_dirs(run_dir):
        if lanes is not None and lane.name not in lanes:
            continue
        state = LaneState(lane)
        if (state.chunks() and not state.pending()) or state.stop_requested or _still_active(backend, lane):
            continue
        request = lane_request(lane, run_lane_args, cores=cores, memory_mb=memory_mb, timeout=timeout,
                               env=env, lane_command=lane_command)
        handle = submit_request(backend, request)
        if handle.job_id:
            (lane / "job").write_text(f"{handle.job_id}\n", encoding="utf-8")
        handles[lane.name] = handle
    return handles


def _still_active(backend: object, lane: Path) -> bool:
    try:
        job = (lane / "job").read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return False
    count = getattr(backend, "_active_jobs", None)
    if not job or count is None:
        return False
    active = count([job])
    return active is None or active > 0
