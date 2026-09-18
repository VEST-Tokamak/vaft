"""Run external codes as Slurm jobs through the shared execution contract (#1017).

:class:`SlurmBackend` is a drop-in :class:`~vaft.code.execution.ExecutionBackend`:
``EFITConfig(backend=SlurmBackend(partition="short"))`` runs EFIT as a cluster
job and returns the same result a local run would.

Two modes, picked automatically unless ``mode`` says otherwise:

* **step** (``srun``) when the caller is already inside an allocation
  (``SLURM_JOB_ID`` is set), so a Snakemake job running under Slurm never
  submits a nested ``sbatch``. Output, stdin and the environment pass through
  as for a local child. The timeout is both ``srun --time`` and a local
  deadline, after which ``srun`` is interrupted twice so that it cancels the
  step; a client that still does not exit is killed and its step cancelled by
  name. The local deadline starts at launch, so waiting for step resources
  counts against it.
* **batch** (``sbatch``) otherwise. A job script is written under
  ``<workdir>/.vaft-slurm/``, submitted with ``sbatch --parsable --no-requeue``,
  and its state polled with ``squeue`` until it is no longer live. The script
  runs the program in the background and traps ``TERM``, so it records the
  program's own exit status -- or that the job was terminated around it --
  without needing accounting; ``sacct`` refines the job state when present.

Both modes request one task on one node holding every CPU the program needs:
``--nodes=1 --ntasks=1 --cpus-per-task=<ntasks x threads>``.
``ResourceRequest.ntasks`` counts ranks a self-launching program (GACODE and
FLARE ``-n``) starts itself; those ranks must fit inside that task, which for
an MPI launcher depends on its own Slurm integration and has to be smoke-tested
per code.

Contract details specific to Slurm:

* ``ExecutionRequest.timeout`` is the job's walltime (``--time``, whole minutes,
  rounded up, at least one; the cluster's ``OverTimeLimit`` may add more).
  ``timeout=None`` sends no ``--time``, so the partition default applies. A job
  Slurm ends as ``TIMEOUT``/``DEADLINE`` comes back ``timed_out=True``. Without
  accounting this is inferred on the node: from ``SLURM_JOB_END_TIME`` where
  Slurm sets it, else from the runtime against the requested walltime, less a
  minute for the prolog.
* ``max_wait`` bounds the total blocking time of a batch job, queue included;
  past it the job is cancelled and reported timed out, with the reason in
  ``stderr`` (or the log).
* ``returncode`` is ``None`` only on a timeout. A job ended around the program
  (cancelled, node failure, preemption, out of memory) gets a non-zero code,
  and its Slurm state is appended to ``stderr`` or to the log file.
* A submission Slurm rejects, or a missing ``sbatch``/``srun``, raises
  :class:`~vaft.code._executables.ExecutableNotLaunchable`. An executable that
  exists on the submitting host but not on the compute node shows up as exit
  status 127.
* ``KeyboardInterrupt`` while waiting cancels the job and re-raises.
* The working directory, and the executables, must be on a filesystem the
  compute nodes share. This is not checked.
* Only the part of ``ExecutionRequest.env`` that differs from the submitting
  process is written into the job script (mode 0700); ``--export=ALL`` carries
  the rest. ``SLURM_*``/``SBATCH_*``/``SRUN_*`` keys and names that are not
  shell identifiers are never exported, and a submission made inside another
  allocation does not inherit that allocation's ``SLURM_*`` variables.
"""

from __future__ import annotations

import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import IO, Any, Mapping, Optional, Sequence

from ._executables import ExecutableNotLaunchable
from .execution import (
    THREAD_ENV_VARIABLES,
    ExecutionRequest,
    ExecutionResult,
    _text,
    execution_environment,
)

#: Slurm states a job ends in because of a time limit.
TIMEOUT_STATES = frozenset({"TIMEOUT", "DEADLINE"})

#: States in which a job is still queued, running or being torn down.
LIVE_STATES = frozenset(
    {
        "PENDING", "CONFIGURING", "RUNNING", "COMPLETING", "SUSPENDED",
        "REQUEUED", "REQUEUE_FED", "REQUEUE_HOLD", "RESIZING", "STOPPED",
        "SIGNALING", "STAGE_OUT", "RESV_DEL_HOLD", "SPECIAL_EXIT",
    }
)

#: Scratch directory (under the request's working directory) for job scripts.
SCRATCH_DIRECTORY = ".vaft-slurm"

_MODES = ("auto", "step", "batch")
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SCHEDULER_PREFIXES = ("SLURM_", "SBATCH_", "SRUN_")
#: An sbatch error that means the controller was never reached, so the job
#: cannot have been accepted and resubmitting cannot duplicate it.
_UNREACHED = ("Unable to contact slurm controller",)
#: Transient accounting errors worth retrying within ``accounting_grace``.
_ACCOUNTING_TRANSIENT = ("Socket timed out", "Unable to contact", "slurmdbd", "Connection refused")
_JOB_ID = re.compile(r"^(\d+)(?:;(\S+))?$")
#: Consecutive unanswered squeue polls before accounting is asked instead.
_UNKNOWN_POLLS = 30
_TERMINATED = "terminated"


def walltime(seconds: float) -> str:
    """``--time`` value for ``seconds``: whole minutes, rounded up, at least 1."""
    return str(max(1, math.ceil(float(seconds) / 60.0)))


def exported_environment(overlay: Mapping[str, str], parent: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    """The part of ``overlay`` a job script must export itself.

    Keys whose value already matches the submitting environment travel with
    ``--export=ALL``; scheduler variables and non-identifiers never travel.
    """
    parent = os.environ if parent is None else parent
    return {
        str(key): str(value)
        for key, value in overlay.items()
        if _IDENTIFIER.match(str(key))
        and not str(key).startswith(_SCHEDULER_PREFIXES)
        and parent.get(str(key)) != str(value)
    }


def _submission_environment() -> dict[str, str]:
    # A job submitted from inside another allocation must not inherit that
    # allocation's geometry: SLURM_MEM_PER_CPU next to the new job's
    # SLURM_MEM_PER_NODE makes an inner srun refuse to start, and sbatch
    # (22.05+) exports SRUN_CPUS_PER_TASK into every job it starts.
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(_SCHEDULER_PREFIXES) or key == "SLURM_CONF"
    }


#: Variables srun reads as option defaults that describe the *enclosing* step
#: or job rather than the step being launched. Inherited through
#: ``--export=ALL`` from a caller that is itself a step (``srun python ...``),
#: the outer CPU binding makes the new step fail with "CPU binding outside of
#: job step allocation" (measured on Slurm 22.05), and an outer per-CPU memory
#: default conflicts with ``--mem``.
_STEP_INHERITED = (
    "SLURM_CPU_BIND", "SLURM_MEM_BIND", "SLURM_DISTRIBUTION", "SLURM_CPUS_PER_TASK",
    "SLURM_CPUS_PER_GPU", "SLURM_GPUS_PER_TASK", "SLURM_TRES_PER_TASK", "SRUN_CPUS_PER_TASK",
)
#: Dropped only when the step asks for ``--mem``, which they would contradict;
#: otherwise the step keeps the job's per-CPU/per-node memory default.
_STEP_MEMORY = ("SLURM_MEM_PER_CPU", "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_GPU")


def _step_environment(request: ExecutionRequest) -> dict[str, str]:
    """The environment for ``srun``: inherited step defaults dropped, then the
    request's own overlay and thread defaults applied as for a local launch.

    Only *inherited* values are filtered, so a caller who sets, say,
    ``SLURM_CPU_BIND`` in ``request.env`` to something new (or passes
    ``--cpu-bind`` through ``extra_args``) still gets it. An overlay entry that
    merely repeats the inherited value counts as inherited: GACODE, GPEC and
    EFIT pass a full ``os.environ`` snapshot as their overlay, and it would
    otherwise carry the enclosing step's binding straight back in (measured on
    Slurm 22.05 with NEO under ``srun python driver.py``).
    """
    dropped = _STEP_INHERITED + (_STEP_MEMORY if request.resources.memory_mb is not None else ())
    inherited = {key: value for key, value in os.environ.items() if not key.startswith(dropped)}
    overlay = {
        key: value
        for key, value in request.env.items()
        if not str(key).startswith(dropped) or os.environ.get(str(key)) != str(value)
    }
    return execution_environment(replace(request, env=overlay), base=inherited)


def _cpus(request: ExecutionRequest) -> int:
    resources = request.resources
    return int(resources.ntasks) * int(resources.threads_per_task or 1)


def _slurm_path(path: Path) -> str:
    # Slurm expands %j, %u, ... in --output/--error file names.
    return str(path).replace("%", "%%")


def _read(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return None


class SlurmBackend:
    """Run each request as a Slurm job step (``srun``) or batch job (``sbatch``).

    Parameters
    ----------
    partition, account, qos : str, optional
        Placement options for ``sbatch``; a step runs inside the caller's
        allocation and takes its placement from it [n/a].
    extra_args : sequence of str
        Further ``sbatch``/``srun`` options, passed through verbatim. Job
        arrays and heterogeneous jobs are refused: one request is one status
        [n/a].
    mode : {"auto", "step", "batch"}
        ``auto`` uses ``srun`` inside an allocation and ``sbatch`` outside [n/a].
    max_wait : float, optional
        Upper bound on the time ``run`` blocks for a batch job, queue included;
        the job is cancelled past it [s].
    poll_interval : float
        Seconds between ``squeue`` polls of a batch job [s].
    step_grace : float
        After a step's local deadline, how long ``srun`` gets to cancel the
        step before it is killed [s].
    keep_scratch : bool
        Keep the job script and captured streams even after a clean exit [n/a].
    """

    #: How long a finished job's status file and accounting may lag behind
    #: ``squeue`` on a shared filesystem / slurmdbd [s].
    status_grace: float = 10.0
    accounting_grace: float = 30.0
    #: After cancelling, how long to wait for the job to leave the queue [s].
    cancel_grace: float = 60.0

    def __init__(
        self,
        *,
        partition: Optional[str] = None,
        account: Optional[str] = None,
        qos: Optional[str] = None,
        extra_args: Sequence[str] = (),
        mode: str = "auto",
        max_wait: Optional[float] = None,
        poll_interval: float = 10.0,
        step_grace: float = 30.0,
        keep_scratch: bool = False,
    ) -> None:
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        if max_wait is not None and max_wait <= 0:
            raise ValueError(f"max_wait must be > 0 or None, got {max_wait}")
        if poll_interval <= 0 or step_grace < 0:
            raise ValueError("poll_interval must be > 0 and step_grace >= 0")
        extra = tuple(str(arg) for arg in extra_args)
        refused = [arg for arg in extra if arg.startswith(("--array", "-a", "--het", ":")) or arg == ":"]
        if refused:
            raise ValueError(f"job arrays and heterogeneous jobs are not supported: {refused}")
        self.partition = partition
        self.account = account
        self.qos = qos
        self.extra_args = extra
        self.mode = mode
        self.max_wait = max_wait
        self.poll_interval = float(poll_interval)
        self.step_grace = float(step_grace)
        self.keep_scratch = keep_scratch

    @classmethod
    def from_environment(cls, environ: Optional[Mapping[str, str]] = None) -> "SlurmBackend":
        """Build a backend from ``VAFT_SLURM_*`` variables.

        ``VAFT_SLURM_PARTITION``, ``VAFT_SLURM_ACCOUNT``, ``VAFT_SLURM_QOS``,
        ``VAFT_SLURM_MODE`` and ``VAFT_SLURM_MAX_WAIT`` (seconds); unset ones
        keep the constructor defaults.
        """
        env = os.environ if environ is None else environ
        max_wait = env.get("VAFT_SLURM_MAX_WAIT")
        return cls(
            partition=env.get("VAFT_SLURM_PARTITION") or None,
            account=env.get("VAFT_SLURM_ACCOUNT") or None,
            qos=env.get("VAFT_SLURM_QOS") or None,
            mode=env.get("VAFT_SLURM_MODE") or "auto",
            max_wait=float(max_wait) if max_wait else None,
        )

    def __repr__(self) -> str:
        return (
            f"SlurmBackend(partition={self.partition!r}, account={self.account!r}, "
            f"qos={self.qos!r}, mode={self.mode!r})"
        )

    # -- dispatch ---------------------------------------------------------

    def resolved_mode(self) -> str:
        """``"step"`` or ``"batch"``: what ``run`` would do right now."""
        if self.mode != "auto":
            return self.mode
        return "step" if os.environ.get("SLURM_JOB_ID") else "batch"

    def run(self, request: ExecutionRequest) -> ExecutionResult:
        if not Path(request.workdir).is_dir():
            raise FileNotFoundError(f"working directory does not exist: {request.workdir}")
        if self.resolved_mode() == "step":
            return self._run_step(request)
        return self._run_batch(request)

    # -- shared -----------------------------------------------------------

    def _sizing(self, request: ExecutionRequest) -> list[str]:
        options = ["--nodes=1", "--ntasks=1", f"--cpus-per-task={_cpus(request)}"]
        if request.resources.memory_mb is not None:
            options.append(f"--mem={int(request.resources.memory_mb)}M")
        if request.timeout is not None:
            options.append(f"--time={walltime(request.timeout)}")
        return options

    def _placement(self) -> list[str]:
        return [
            f"--{name}={value}"
            for name, value in (("partition", self.partition), ("account", self.account), ("qos", self.qos))
            if value
        ]

    @staticmethod
    def _job_name(request: ExecutionRequest) -> str:
        return f"vaft-{request.label}" if request.label else "vaft"

    @staticmethod
    def _tool(name: str) -> str:
        found = shutil.which(name)
        if found is None:
            raise ExecutableNotLaunchable(f"cannot launch through Slurm: {name!r} is not on PATH")
        return found

    # -- step mode --------------------------------------------------------

    def _run_step(self, request: ExecutionRequest) -> ExecutionResult:
        workdir = Path(request.workdir).resolve()
        argv = [
            self._tool("srun"),
            *self._sizing(request),
            # Unique, so a step whose client had to be killed can be found.
            f"--job-name={self._job_name(request)}-{uuid.uuid4().hex[:8]}",
            f"--chdir={workdir}",
            "--export=ALL",
        ]
        if os.environ.get("SLURM_STEP_ID") is not None:
            # The caller is itself a step holding the allocation's CPUs; a new
            # step would otherwise wait for them forever.
            argv.append("--overlap")
        argv += [*self.extra_args, "--", *(str(part) for part in request.command)]

        environment = _step_environment(request)
        log_path = Path(request.log_path) if request.log_path is not None else None
        started = time.monotonic()
        if log_path is None:
            return self._step(argv, request, environment, None, started)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            return self._step(argv, request, environment, log, started)

    def _step(
        self,
        argv: list[str],
        request: ExecutionRequest,
        environment: dict[str, str],
        log: Optional[IO[str]],
        started: float,
    ) -> ExecutionResult:
        try:
            process = subprocess.Popen(
                argv,
                cwd=str(request.workdir),
                env=environment,
                stdin=subprocess.PIPE if request.stdin is not None else subprocess.DEVNULL,
                stdout=log if log is not None else subprocess.PIPE,
                stderr=subprocess.STDOUT if log is not None else subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except OSError as error:
            raise ExecutableNotLaunchable(f"cannot launch {argv[0]}: {error}") from error
        timed_out = False
        try:
            stdout, stderr = process.communicate(request.stdin, timeout=request.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            stdout, stderr = self._interrupt(process)
        except BaseException:
            self._interrupt(process)
            raise
        return ExecutionResult(
            returncode=None if timed_out else int(process.returncode),
            stdout=_text(stdout) if log is None else "",
            stderr=_text(stderr) if log is None else "",
            timed_out=timed_out,
            elapsed_s=time.monotonic() - started,
            launcher=tuple(argv),
            log_path=Path(request.log_path) if request.log_path is not None else None,
            job_id=os.environ.get("SLURM_JOB_ID"),
        )

    def _interrupt(self, process: "subprocess.Popen[str]") -> tuple[Any, Any]:
        # Two SIGINTs within a second make srun cancel the step; SIGKILL would
        # only kill the client and leave the step running.
        for _ in range(2):
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                time.sleep(0.2)
        try:
            return process.communicate(timeout=self.step_grace)
        except subprocess.TimeoutExpired:
            process.kill()
            outputs = process.communicate()
            self._cancel_step(process.args)
            return outputs

    def _cancel_step(self, argv: Sequence[str]) -> None:
        """Cancel a step whose srun client had to be killed, by its unique name."""
        job_id = os.environ.get("SLURM_JOB_ID")
        name = next((a.split("=", 1)[1] for a in argv if str(a).startswith("--job-name=")), None)
        squeue = shutil.which("squeue")
        if not (job_id and name and squeue):
            return
        steps = self._slurm([squeue, "-h", "-s", "-j", job_id, "-o", "%i %j"])
        for line in steps.stdout.splitlines():
            step_id, _, step_name = line.strip().partition(" ")
            if step_name == name:
                self._cancel(step_id, [])

    # -- batch mode -------------------------------------------------------

    @staticmethod
    def _script(request: ExecutionRequest, workdir: Path, stdin: Optional[Path], scratch: Path) -> str:
        quote = shlex.quote
        status, started = scratch / "returncode", scratch / "started"
        lines = [
            "#!/bin/bash",
            f"cd {quote(str(workdir))} || exit 1",
            f"rm -f {quote(str(status))}",
        ]
        for key, value in sorted(exported_environment(request.env).items()):
            lines.append(f"export {key}={quote(value)}")
        threads = request.resources.threads_per_task
        if threads is not None:
            # setdefault semantics, as for a local launch.
            for variable in THREAD_ENV_VARIABLES:
                lines.append(f'export {variable}="${{{variable}:-{int(threads)}}}"')
        command = " ".join(quote(str(part)) for part in request.command)
        source = quote(str(stdin)) if stdin is not None else "/dev/null"
        lines += [
            f"date +%s > {quote(str(started))}",
            # The termination record carries the node's own clock and, where
            # Slurm provides it, the job's end time, so a walltime kill can be
            # told from a cancel without accounting and without comparing
            # clocks across hosts.
            f'terminated() {{ echo "{_TERMINATED} $(date +%s) ${{SLURM_JOB_END_TIME:-}}" > {quote(str(status))}; }}',
            # Backgrounded so a TERM at the time limit or on scancel runs the
            # trap now rather than after the program exits.
            "trap 'terminated; kill -TERM $child 2>/dev/null; wait $child; exit 143' TERM",
            # A non-interactive shell starts background jobs with INT and QUIT
            # ignored; restore them so scancel --signal=INT reaches the program.
            f"( trap - INT QUIT; exec {command} ) < {source} &",
            "child=$!",
            "wait $child",
            "rc=$?",
            # The program may die of the TERM before this shell handles its own.
            f'if [ "$rc" -eq 143 ]; then terminated; else echo "$rc" > {quote(str(status))}; fi',
            "exit $rc",
            "",
        ]
        return "\n".join(lines)

    def _run_batch(self, request: ExecutionRequest) -> ExecutionResult:
        sbatch = self._tool("sbatch")
        workdir = Path(request.workdir).resolve()
        scratch = workdir / SCRATCH_DIRECTORY / f"{request.label or 'job'}-{uuid.uuid4().hex[:8]}"
        scratch.mkdir(parents=True)
        stdin = None
        if request.stdin is not None:
            stdin = scratch / "stdin"
            stdin.write_text(request.stdin, encoding="utf-8")
        script = scratch / "job.sh"
        script.write_text(self._script(request, workdir, stdin, scratch), encoding="utf-8")
        script.chmod(0o700)

        log_path = Path(request.log_path).resolve() if request.log_path is not None else None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            streams = [f"--output={_slurm_path(log_path)}"]  # stderr joins it without --error
        else:
            streams = [
                f"--output={_slurm_path(scratch / 'stdout')}",
                f"--error={_slurm_path(scratch / 'stderr')}",
            ]
        argv = [
            sbatch, "--parsable", "--no-requeue", "--export=ALL",
            f"--job-name={self._job_name(request)}", f"--chdir={workdir}", *streams,
            *self._sizing(request), *self._placement(), *self.extra_args, str(script),
        ]

        started = time.monotonic()
        job_id, cluster = self._submit(argv)
        clusters = ["-M", cluster] if cluster else []

        try:
            cancelled = self._wait(job_id, clusters, started)
        except BaseException:
            self._cancel(job_id, clusters)
            raise
        recorded = self._recorded(scratch / "returncode")
        state, exit_code = self._accounting(job_id, clusters)

        terminated = recorded is not None and recorded.startswith(_TERMINATED)
        exited = recorded is not None and not terminated
        if cancelled and exited:
            cancelled = False  # the program exited before the cancel landed
        timed_out = cancelled or state in TIMEOUT_STATES
        if not timed_out and state is None and terminated:
            timed_out = self._walltime_kill(recorded, scratch, request)

        if timed_out:
            returncode: Optional[int] = None
        elif exited:
            returncode = int(recorded)
        elif state == "COMPLETED":
            returncode = exit_code or 0
        else:
            returncode = exit_code or (143 if terminated else 1)

        note = ""
        if cancelled:
            note = f"slurm job {job_id} cancelled after max_wait={self.max_wait} s"
        elif state not in (None, "COMPLETED", "FAILED"):
            note = f"slurm job {job_id} ended {state}"
        elif state is None and not exited and not timed_out:
            note = f"slurm job {job_id} ended without an exit status of its own (state unknown)"

        if log_path is not None:
            stdout = stderr = ""
            if note:
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(f"\n{note}\n")
        else:
            stdout = _read(scratch / "stdout") or ""
            stderr = _read(scratch / "stderr") or ""
            if note:
                stderr = f"{stderr}\n{note}" if stderr else note

        if returncode == 0 and not self.keep_scratch:
            shutil.rmtree(scratch, ignore_errors=True)
            try:
                scratch.parent.rmdir()
            except OSError:
                pass
        return ExecutionResult(
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            elapsed_s=time.monotonic() - started,
            launcher=tuple(argv),
            log_path=log_path,
            job_id=job_id,
        )

    def _slurm(self, argv: list[str], *, environment: Optional[dict[str, str]] = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            argv, capture_output=True, text=True, encoding="utf-8", errors="replace",
            env=environment, check=False,
        )

    def _submit(self, argv: list[str]) -> tuple[str, str]:
        for attempt in range(3):
            submitted = self._slurm(argv, environment=_submission_environment())
            if submitted.returncode == 0:
                # Site wrappers may print banners first; the id is the last line.
                lines = [line.strip() for line in submitted.stdout.splitlines() if line.strip()]
                parsed = _JOB_ID.match(lines[-1]) if lines else None
                if parsed is None:
                    raise ExecutableNotLaunchable(
                        f"sbatch accepted {argv[-1]} but printed no job id: {submitted.stdout!r}"
                    )
                return parsed.group(1), parsed.group(2) or ""
            message = (submitted.stderr or submitted.stdout).strip()
            if attempt == 2 or not any(marker in message for marker in _UNREACHED):
                raise ExecutableNotLaunchable(f"sbatch rejected the job for {argv[-1]}: {message}")
            time.sleep(self.poll_interval)
        raise AssertionError("unreachable")  # pragma: no cover

    def _live(self, job_id: str, clusters: list[str]) -> Optional[bool]:
        """True while the job is live, False once it is not, None if unknown."""
        listed = self._slurm([self._tool("squeue"), *clusters, "-h", "-j", job_id, "-o", "%T"])
        self._last_squeue_error = listed.stderr.strip()
        if listed.returncode != 0:
            # Only a purged job is gone; a controller that did not answer is not.
            return False if "Invalid job id" in listed.stderr else None
        states = {line.strip().split()[0] for line in listed.stdout.splitlines() if line.strip()}
        return bool(states & LIVE_STATES)

    def _wait(self, job_id: str, clusters: list[str], started: float) -> bool:
        """Block until the job is no longer live; True if ``max_wait`` cancelled it."""
        cancelled_at: Optional[float] = None
        unknown = 0
        while True:
            live = self._live(job_id, clusters)
            if live is False:
                return cancelled_at is not None
            unknown = unknown + 1 if live is None else 0
            if unknown >= _UNKNOWN_POLLS:
                # squeue has stopped answering; accounting may still know.
                state, _ = self._accounting(job_id, clusters)
                if state is not None:
                    return cancelled_at is not None
                raise RuntimeError(
                    f"slurm job {job_id} could not be followed: squeue failed "
                    f"{unknown} times in a row ({getattr(self, '_last_squeue_error', '')}) "
                    "and accounting has no final state; the job may still be running"
                )
            now = time.monotonic()
            if cancelled_at is None and self.max_wait is not None and now - started > self.max_wait:
                self._cancel(job_id, clusters)
                cancelled_at = now
            elif cancelled_at is not None and now - cancelled_at > self.cancel_grace:
                return True
            time.sleep(self.poll_interval)

    def _cancel(self, job_id: str, clusters: list[str]) -> None:
        scancel = shutil.which("scancel")
        if scancel is None:
            return
        for _ in range(3):
            if self._slurm([scancel, *clusters, job_id]).returncode == 0:
                return
            time.sleep(1.0)

    def _recorded(self, status: Path) -> Optional[str]:
        """The script's own record: an exit status, ``terminated``, or None."""
        deadline = time.monotonic() + self.status_grace
        while True:
            text = _read(status)
            if text is not None and text.strip():
                value = text.strip()
                if value.startswith(_TERMINATED) or value.lstrip("-").isdigit():
                    return value
            if time.monotonic() >= deadline:
                return None
            time.sleep(min(1.0, self.status_grace))

    @staticmethod
    def _walltime_kill(recorded: str, scratch: Path, request: ExecutionRequest) -> bool:
        """Whether a ``terminated`` record is the walltime rather than a cancel.

        Decided from the node's own clock: against ``SLURM_JOB_END_TIME`` when
        the job had one, else against the walltime actually requested (whole
        minutes), less a minute for the prolog that runs before ``started``.
        """
        fields = recorded.split()
        if len(fields) < 2 or not fields[1].isdigit():
            return False
        ended = int(fields[1])
        if len(fields) >= 3 and fields[2].isdigit():
            return ended >= int(fields[2]) - 10
        begin = (_read(scratch / "started") or "").strip()
        if request.timeout is None or not begin.isdigit():
            return False
        limit = int(walltime(request.timeout)) * 60
        return ended - int(begin) >= limit - 60

    def _accounting(self, job_id: str, clusters: list[str]) -> tuple[Optional[str], Optional[int]]:
        """``(state, exit code)`` from ``sacct``, or ``(None, None)`` without it."""
        sacct = shutil.which("sacct")
        if sacct is None:
            return None, None
        deadline = time.monotonic() + self.accounting_grace
        while True:
            report = self._slurm([sacct, *clusters, "-j", job_id, "-X", "-n", "-P", "-o", "State,ExitCode"])
            line = next((row for row in report.stdout.splitlines() if row.strip()), "")
            if report.returncode == 0 and "|" in line:
                state_text, code_text = line.split("|", 1)
                state = state_text.split()[0] if state_text.strip() else None
                # slurmdbd lags the controller; wait for the final state.
                if state is not None and state not in LIVE_STATES:
                    return state, self._exit_code(code_text)
            elif report.returncode != 0 and not any(
                marker in report.stderr for marker in _ACCOUNTING_TRANSIENT
            ):
                return None, None  # accounting is not configured here
            if time.monotonic() >= deadline:
                return None, None
            time.sleep(min(1.0, self.accounting_grace))

    @staticmethod
    def _exit_code(text: str) -> Optional[int]:
        code, _, signal_number = text.strip().partition(":")
        try:
            number = int(code)
            if number == 0 and signal_number and int(signal_number):
                number = 128 + int(signal_number)
            return number
        except ValueError:
            return None


__all__ = [
    "LIVE_STATES",
    "SCRATCH_DIRECTORY",
    "TIMEOUT_STATES",
    "SlurmBackend",
    "exported_environment",
    "walltime",
]
