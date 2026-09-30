"""Run the routine Snakemake pipeline for a batch of shots (issue #58).

One run is one ``snakemake`` process in the workflow directory, so the
worker and a manual ``make run`` share Snakemake's directory lock: whichever
starts second is refused, and the worker treats that refusal as *busy* --
retry next cycle, attempt not counted -- never as a failure and never as a
reason to ``--unlock`` someone else's run.

Flags the worker always passes, and why:

``--scheduler greedy``
    snakemake 7.32's ILP scheduler crashes whenever pulp is installed
    (DEPLOYMENT.md, section 4).
``--keep-going``
    one shot's failed stage must not stop the other shots in the batch.
``--rerun-incomplete``
    a run the worker had to terminate leaves incomplete outputs; the next run
    rebuilds exactly those.
``--resources hsds=1``
    replications of one shot that overlap in time lose ``master.h5`` links
    (#913), and a worker batch of a few new shots is precisely the
    few-pending-targets case where they overlap.  Remove when #913 is fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
from typing import Any, Callable, Mapping, Sequence

import yaml

from .config import WorkerConfig


#: Serial HSDS replication until #913 is fixed.
HSDS_RESOURCE = "hsds=1"

_LOCK_MARKERS = ("LockException", "Directory cannot be locked")
_TERMINATE_GRACE_SECONDS = 120.0


@dataclass(frozen=True)
class RunPlan:
    run_id: int
    #: Shots requested through ``rule all`` -- the whole routine pipeline.
    full_shots: tuple[int, ...]
    #: Explicit file targets, e.g. the raw dump of a raw-only shot.
    file_targets: tuple[str, ...] = ()

    @property
    def shots_label(self) -> str:
        return ", ".join(str(shot) for shot in self.full_shots) or "-"


@dataclass(frozen=True)
class RunResult:
    exit_code: int | None
    log_path: str
    busy: bool = False
    timed_out: bool = False
    interrupted: bool = False


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class SnakemakeRunner:
    """Builds and runs the Snakemake command for a :class:`RunPlan`."""

    def __init__(self, config: WorkerConfig, pipeline_config: Mapping[str, Any]):
        self.config = config
        self.pipeline_config = dict(pipeline_config)
        self._process: subprocess.Popen | None = None
        self.terminate_grace = _TERMINATE_GRACE_SECONDS
        self._stop = threading.Event()

    # -- command -----------------------------------------------------------
    def run_dir(self, run_id: int) -> Path:
        return self.config.log_dir / "runs" / f"{run_id:06d}"

    def write_configfile(self, plan: RunPlan) -> Path:
        """The pipeline config with ``shots`` replaced by this run's batch.

        Kept beside the run log, so every run's exact configuration survives it.
        """
        data = dict(self.pipeline_config)
        data["shots"] = [int(shot) for shot in plan.full_shots]
        data["conda"] = None
        path = self.run_dir(plan.run_id) / "config.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return path

    def base_command(self, configfile: Path, targets: Sequence[str] = ()) -> list[str]:
        # Targets come straight after the executable: `--resources` (and any
        # other nargs="*" option in `extra_args`) would swallow a trailing one.
        return [
            *self.config.snakemake_cmd,
            *targets,
            "--snakefile",
            str(self.config.snakefile),
            "--directory",
            str(self.config.workflow_dir),
            "--configfile",
            str(configfile),
        ]

    def command(self, plan: RunPlan, configfile: Path) -> list[str]:
        targets: list[str] = []
        if plan.full_shots:
            targets.append("all")
        targets.extend(plan.file_targets)
        return [
            *self.base_command(configfile, targets),
            "--cores",
            str(self.config.cores),
            "--scheduler",
            "greedy",
            "--keep-going",
            "--rerun-incomplete",
            "--resources",
            HSDS_RESOURCE,
            *self.config.extra_args,
        ]

    # -- execution ---------------------------------------------------------
    def run(
        self, plan: RunPlan, *, on_start: Callable[[list[str], str, int], None] | None = None
    ) -> RunResult:
        configfile = self.write_configfile(plan)
        command = self.command(plan, configfile)
        log_path = self.run_dir(plan.run_id) / "snakemake.log"
        with log_path.open("w", encoding="utf-8") as log:
            log.write("# " + json.dumps(command) + "\n")
            log.flush()
            process = subprocess.Popen(
                command,
                cwd=str(self.config.workflow_dir),
                env=self.config.environment(),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                # Its own process group, so a timeout reaches every job
                # Snakemake started and not the worker itself.
                start_new_session=os.name != "nt",
            )
            self._process = process
            if on_start is not None:
                on_start(command, str(log_path), process.pid)
            timed_out = False
            try:
                process.wait(timeout=self.config.run_timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                self._terminate(process)
            finally:
                self._process = None
        busy = process.returncode != 0 and _log_mentions_lock(log_path)
        return RunResult(
            exit_code=process.returncode,
            log_path=str(log_path),
            busy=busy,
            timed_out=timed_out,
            interrupted=self._stop.is_set(),
        )

    def stop(self) -> None:
        """Ask a running Snakemake to stop gracefully (called from a signal handler)."""
        self._stop.set()
        process = self._process
        if process is not None and process.poll() is None:
            self._signal(process, signal.SIGTERM)

    def _terminate(self, process: subprocess.Popen) -> None:
        # SIGTERM lets Snakemake mark its running jobs incomplete and release
        # the directory lock; SIGKILL only if it will not.
        self._signal(process, signal.SIGTERM)
        try:
            process.wait(timeout=self.terminate_grace)
        except subprocess.TimeoutExpired:
            self._signal(process, getattr(signal, "SIGKILL", signal.SIGTERM))
            process.wait()
            # A killed Snakemake cannot release its directory lock, and every
            # later run would read as "busy" forever.  The lock is provably
            # this worker's own: its holder was just killed.
            self._unlock()

    @staticmethod
    def _signal(process: subprocess.Popen, signum: int) -> None:
        try:
            if os.name != "nt":
                os.killpg(process.pid, signum)
            else:
                process.send_signal(signum)
        except ProcessLookupError:
            pass

    def unlock_after_crash(self, stale_pid: int | None) -> bool:
        """``--unlock`` the workflow directory after the worker's own run died.

        Only when the recorded Snakemake process is gone: a live one is still
        working, and a lock that belongs to a manual run is not the worker's
        to clear.
        """
        if stale_pid is not None and _pid_alive(stale_pid):
            return False
        self._unlock()
        return True

    def _unlock(self) -> None:
        configfile = self.write_configfile(RunPlan(run_id=0, full_shots=()))
        subprocess.run(
            [*self.base_command(configfile), "--unlock"],
            cwd=str(self.config.workflow_dir),
            env=self.config.environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def _log_mentions_lock(log_path: Path) -> bool:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return any(marker in text for marker in _LOCK_MARKERS)


__all__ = ["HSDS_RESOURCE", "RunPlan", "RunResult", "SnakemakeRunner"]
