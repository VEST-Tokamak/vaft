"""Execution backends: how an adapter's prepared command is actually run.

An adapter's ``run_*`` owns the science: it resolves the executable, builds the
command line and turns the solver's exit into a result. *How* that command is
launched -- as a local child process today, through Slurm later -- is the
backend's job, so the launch behaviour (environment merge, timeout, output
capture, launch failure) is written once instead of once per adapter.

The contract every backend keeps:

* ``run`` blocks until the program exits or times out.
* A timeout is **returned**, not raised: ``timed_out=True``, ``returncode=None``
  and whatever output was captured before the kill. The kill stops the whole
  process tree, not just the direct child (see :mod:`vaft.code._process_tree`).
  ``runtime_status`` says which limit ended it: ``"timeout"`` for a program
  that ran too long, ``"queue_timeout"`` for a scheduler job cancelled before
  it ever started. Every adapter turns either into a result with
  ``status="failed"``, the same ``runtime_status``, ``returncode=None`` and
  ``elapsed_s`` (#1016), so one timed-out case never stops a scan or a batch.
* ``KeyboardInterrupt``, ``SIGTERM`` or ``SIGHUP`` during the wait stops the
  tree the same way and is then raised (or re-delivered) unchanged; it is never
  turned into a result.
* A program the operating system will not start raises
  :class:`~vaft.code._executables.ExecutableNotLaunchable`, chained to the
  ``OSError``.
* ``ResourceRequest`` is a declaration. A default :class:`LocalBackend` honours
  only ``threads_per_task``; ``ntasks`` and ``memory_mb`` are for scheduler
  backends. Codes that start their own MPI ranks (GACODE, FLARE) keep passing
  ``-n`` on their command line.
* A :class:`LocalBackend` built with memory settings (#1460) also admits a
  launch only when the host has room for its reservation (``memory_mb``, or the
  backend's ``reserve_mb``), and stops a tree whose resident size passes
  ``memory_limit_mb``. Both are returned, not raised: a launch never admitted
  within ``admission_wait_s`` is ``runtime_status="queue_timeout"`` (it never
  started), and a stopped tree is ``"memory_limit"`` with ``peak_rss_mb``.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable

from . import _process_tree
from ._executables import ExecutableNotLaunchable

#: Thread-count variables the common Fortran/BLAS runtimes read.
THREAD_ENV_VARIABLES: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

# The stdlib's own ``subprocess.run``. Several adapter tests replace it to see or
# refuse a launch without starting a process; while it is replaced, a local
# launch still goes through it (see ``LocalBackend._run``).
_STDLIB_RUN = subprocess.run


@dataclass(frozen=True)
class ResourceRequest:
    """Compute resources one execution asks for.

    ``threads_per_task=None`` leaves the thread variables of the inherited
    environment untouched; a number sets each of :data:`THREAD_ENV_VARIABLES`
    that the caller's environment does not already set.
    """

    ntasks: int = 1
    threads_per_task: Optional[int] = None
    memory_mb: Optional[int] = None

    def __post_init__(self) -> None:
        if self.ntasks < 1:
            raise ValueError(f"ntasks must be >= 1, got {self.ntasks}")
        if self.threads_per_task is not None and self.threads_per_task < 1:
            raise ValueError(
                f"threads_per_task must be >= 1 or None, got {self.threads_per_task}"
            )
        if self.memory_mb is not None and self.memory_mb < 1:
            raise ValueError(f"memory_mb must be >= 1 or None, got {self.memory_mb}")


@dataclass(frozen=True)
class ExecutionRequest:
    """One program launch, fully described.

    ``env`` is an overlay on the launching process's environment, not a
    replacement for it. With ``log_path`` set, stdout and stderr are merged into
    that file and the result's ``stdout``/``stderr`` are empty; without it both
    streams are captured as text.
    """

    command: Sequence[str]
    workdir: Path
    env: Mapping[str, str] = field(default_factory=dict)
    stdin: Optional[str] = None
    timeout: Optional[float] = None
    log_path: Optional[Path] = None
    resources: ResourceRequest = field(default_factory=ResourceRequest)
    label: str = ""


@dataclass
class ExecutionResult:
    """What a backend observed about one launch."""

    returncode: Optional[int]
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    elapsed_s: float = 0.0
    launcher: tuple[str, ...] = ()
    log_path: Optional[Path] = None
    job_id: Optional[str] = None
    #: ``"completed"`` (the program exited; see ``returncode``), ``"timeout"``
    #: ``"queue_timeout"`` or ``"memory_limit"``. Left empty, it is derived from
    #: ``timed_out``, which is true for every limit stop.
    runtime_status: str = ""
    #: How long the program itself ran [s], when the backend can tell that
    #: apart from time spent queued: a scheduler batch job's own start and
    #: termination stamps. ``None`` when ``elapsed_s`` is the only measure
    #: (a local launch, or a job that left no termination record).
    run_s: Optional[float] = None
    #: Largest resident size of the program's process tree that the backend
    #: sampled [MiB]; only a backend that watches memory sets it.
    peak_rss_mb: Optional[float] = None
    #: What a ``queue_timeout`` waited for: ``"memory"`` for local memory
    #: admission (#1460); empty otherwise, including a scheduler queue.
    waited_for: str = ""

    def __post_init__(self) -> None:
        if not self.runtime_status:
            self.runtime_status = RUNTIME_TIMEOUT if self.timed_out else RUNTIME_COMPLETED
        if self.runtime_status not in RUNTIME_STATUSES:
            raise ValueError(
                f"runtime_status must be one of {RUNTIME_STATUSES}, got {self.runtime_status!r}"
            )
        if (self.runtime_status != RUNTIME_COMPLETED) != self.timed_out:
            raise ValueError(
                f"runtime_status={self.runtime_status!r} disagrees with timed_out={self.timed_out}"
            )


#: The program exited on its own (with any return code).
RUNTIME_COMPLETED = "completed"
#: The program ran past its time limit and was stopped.
RUNTIME_TIMEOUT = "timeout"
#: A scheduler job was cancelled while still queued, or a local launch was never
#: admitted for memory; either way the program never started.
RUNTIME_QUEUE_TIMEOUT = "queue_timeout"
#: The program's process tree grew past the backend's memory limit and was stopped.
RUNTIME_MEMORY_LIMIT = "memory_limit"
RUNTIME_STATUSES: tuple[str, ...] = (
    RUNTIME_COMPLETED,
    RUNTIME_TIMEOUT,
    RUNTIME_QUEUE_TIMEOUT,
    RUNTIME_MEMORY_LIMIT,
)


def timeout_reason(program: str, execution: ExecutionResult, timeout: Optional[float]) -> str:
    """The one-line reason an adapter gives for a timed-out execution.

    A job that never left the scheduler queue did not "time out after N
    seconds" of running, so the two limits are worded apart. The running
    time is ``run_s`` when the backend measured it (a batch job's own
    stamps; ``elapsed_s`` there counts from submission, queue wait included),
    else ``elapsed_s``.
    """
    if execution.runtime_status == RUNTIME_MEMORY_LIMIT:
        peak = "" if execution.peak_rss_mb is None else f" at {execution.peak_rss_mb:.0f} MiB resident"
        return f"{program} was stopped by the memory limit{peak} after {execution.elapsed_s:.3g} s"
    if execution.runtime_status == RUNTIME_QUEUE_TIMEOUT:
        where = "for memory to become available" if execution.waited_for == "memory" else "in the scheduler queue"
        return f"{program} was cancelled after waiting {execution.elapsed_s:.0f} s {where} (it never started)"
    ran = execution.elapsed_s if execution.run_s is None else execution.run_s
    # A stop well short of ``timeout`` (a scheduler's max_wait cancelling a
    # running job) did not run for ``timeout`` seconds; say how long it did.
    if timeout is None or ran < 0.9 * timeout:
        return f"{program} timed out after {ran:.3g} s of running"
    return f"{program} timed out after {timeout:g} s of running"


@runtime_checkable
class ExecutionBackend(Protocol):
    """Anything that can run an :class:`ExecutionRequest` to completion."""

    def run(self, request: ExecutionRequest) -> ExecutionResult:
        """Run the request and report how it ended."""
        ...


def _text(stream: Any) -> str:
    # A foreign program's bytes: TimeoutExpired carries raw bytes even when the
    # run asked for text, so decode leniently rather than fail on the report.
    if isinstance(stream, bytes):
        return stream.decode("utf-8", "replace")
    return stream or ""


def execution_environment(
    request: ExecutionRequest, base: Optional[Mapping[str, str]] = None
) -> dict[str, str]:
    """The full environment a local launch of ``request`` receives.

    ``base`` replaces the inherited ``os.environ`` (a backend that must filter
    what it inherits passes the filtered copy).
    """
    environment = dict(os.environ if base is None else base)
    environment.update({str(key): str(value) for key, value in request.env.items()})
    threads = request.resources.threads_per_task
    if threads is not None:
        for variable in THREAD_ENV_VARIABLES:
            environment.setdefault(variable, str(threads))
    return environment


class LocalBackend:
    """Run the command as a child process of the current Python process.

    With no arguments nothing about memory is checked. The memory settings
    (#1460) are for hosts shared by heavy solvers:

    ``reserve_mb``
        Default reservation for a launch whose ``resources.memory_mb`` is
        unset. A launch with a reservation waits until the host-wide
        :class:`~vaft.code._memory_gate.MemoryLedger` admits it.
    ``memory_limit_mb``
        Resident size of the launched tree above which it is stopped.
    ``admission_wait_s``
        How long a launch may wait for admission (``None``: forever).
    ``floor_mb``
        Memory left free for everything else on the host.
    ``poll_interval_s``
        How often a running tree's resident size is sampled.
    ``ledger_dir``
        The reservation ledger; default ``$VAFT_MEMORY_LEDGER_DIR`` or a
        per-user temp directory. Only processes sharing one directory see each
        other's reservations, so give workers with different ``TMPDIR`` the
        same ``VAFT_MEMORY_LEDGER_DIR``.
    """

    def __init__(
        self,
        *,
        reserve_mb: Optional[float] = None,
        memory_limit_mb: Optional[float] = None,
        admission_wait_s: Optional[float] = 3600.0,
        floor_mb: float = 4096.0,
        poll_interval_s: float = 2.0,
        ledger_dir: Optional[Path] = None,
    ) -> None:
        for name, value in (("reserve_mb", reserve_mb), ("memory_limit_mb", memory_limit_mb)):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be > 0 or None, got {value}")
        if poll_interval_s <= 0:
            raise ValueError(f"poll_interval_s must be > 0, got {poll_interval_s}")
        if floor_mb < 0:
            raise ValueError(f"floor_mb must be >= 0, got {floor_mb}")
        self.reserve_mb = reserve_mb
        self.memory_limit_mb = memory_limit_mb
        self.admission_wait_s = admission_wait_s
        self.floor_mb = floor_mb
        self.poll_interval_s = poll_interval_s
        self.ledger_dir = ledger_dir

    def _watches_memory(self) -> bool:
        return self.reserve_mb is not None or self.memory_limit_mb is not None

    def run(self, request: ExecutionRequest) -> ExecutionResult:
        command = tuple(str(part) for part in request.command)
        # A missing working directory is a configuration error, not a program
        # the OS refused to start; keep it a FileNotFoundError.
        if not Path(request.workdir).is_dir():
            raise FileNotFoundError(f"working directory does not exist: {request.workdir}")
        kwargs: dict[str, Any] = dict(
            cwd=str(request.workdir),
            env=execution_environment(request),
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=request.timeout,
            check=False,
        )
        if request.stdin is not None:
            kwargs["input"] = request.stdin
        reserve = request.resources.memory_mb if self._watches_memory() else None
        if reserve is None:
            reserve = self.reserve_mb
        reservation = None
        try:
            # Admission sits inside the try, so an interrupt right after it
            # still releases the record.
            if reserve is not None:
                from ._memory_gate import MemoryLedger

                waited = time.monotonic()
                reservation = MemoryLedger(self.ledger_dir, floor_mb=self.floor_mb).admit(
                    float(reserve), wait_s=self.admission_wait_s, poll_s=max(self.poll_interval_s, 1.0)
                )
                if reservation is None:
                    return self._not_admitted(command, request, float(reserve), time.monotonic() - waited)
            if request.log_path is None:
                return self._run(command, request, kwargs, reservation)
            # Opened outside the launch so a bad log path stays its own error
            # rather than being reported as an unlaunchable program.
            log_path = Path(request.log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as log:
                kwargs.update(stdout=log, stderr=subprocess.STDOUT)
                return self._run(command, request, kwargs, reservation)
        finally:
            if reservation is not None:
                reservation.release()

    def _not_admitted(
        self, command: tuple[str, ...], request: ExecutionRequest, reserve: float, waited: float
    ) -> ExecutionResult:
        log_path = None if request.log_path is None else Path(request.log_path)
        message = (
            f"[vaft LocalBackend] not started: waited {waited:.0f} s for {reserve:.0f} MiB "
            f"(+{self.floor_mb:.0f} MiB floor) to become available\n"
        )
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(message, encoding="utf-8")
        return ExecutionResult(
            returncode=None,
            stderr="" if log_path is not None else message,
            timed_out=True,
            elapsed_s=waited,
            launcher=command,
            log_path=log_path,
            runtime_status=RUNTIME_QUEUE_TIMEOUT,
            waited_for="memory",
        )

    def _run(
        self,
        command: tuple[str, ...],
        request: ExecutionRequest,
        kwargs: dict[str, Any],
        reservation: Any = None,
    ) -> ExecutionResult:
        if subprocess.run is _STDLIB_RUN:
            return self._run_tree(command, request, kwargs, reservation)
        # A test replaced ``subprocess.run`` to intercept the launch: honour it,
        # so no real program starts. New tests use real programs or replace
        # ``vaft.code._process_tree.ProcessTree.start`` instead.
        capture = request.log_path is None
        log_path = None if capture else Path(request.log_path)
        started = time.monotonic()
        try:
            # Looked up on the module at call time so tests that patch
            # ``subprocess.run`` keep intercepting every adapter.
            completed = subprocess.run(list(command), capture_output=capture, **kwargs)
        except subprocess.TimeoutExpired as expired:
            return ExecutionResult(
                returncode=None,
                stdout=_text(expired.stdout) if capture else "",
                stderr=_text(expired.stderr) if capture else "",
                timed_out=True,
                elapsed_s=time.monotonic() - started,
                launcher=command,
                log_path=log_path,
            )
        except OSError as error:
            raise ExecutableNotLaunchable(f"cannot launch {command[0]}: {error}") from error
        return ExecutionResult(
            returncode=int(completed.returncode),
            stdout=_text(completed.stdout) if capture else "",
            stderr=_text(completed.stderr) if capture else "",
            elapsed_s=time.monotonic() - started,
            launcher=command,
            log_path=log_path,
        )


    def _run_tree(
        self,
        command: tuple[str, ...],
        request: ExecutionRequest,
        kwargs: dict[str, Any],
        reservation: Any = None,
    ) -> ExecutionResult:
        """Launch as a :class:`~vaft.code._process_tree.ProcessTree` (#1016).

        Same result as the ``subprocess.run`` path, but a timeout, a
        ``KeyboardInterrupt``, a ``SIGTERM`` or a ``SIGHUP`` stops the whole
        tree. The program stays in the caller's process group, so signals sent
        to the group (Ctrl-C, Ctrl-Z, a supervisor's ``SIGSTOP``) reach it as
        before.
        """
        capture = request.log_path is None
        log_path = None if capture else Path(request.log_path)
        popen_kwargs = dict(kwargs)
        timeout = popen_kwargs.pop("timeout", None)
        stdin = popen_kwargs.pop("input", None)
        popen_kwargs.pop("check", None)
        if stdin is not None:
            popen_kwargs["stdin"] = subprocess.PIPE
        if capture:
            popen_kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        started = time.monotonic()
        tree: Optional[_process_tree.ProcessTree] = None
        timed_out = False
        memory_stop = False
        peak: Optional[float] = None
        try:
            # The guard spans the stop as well, so a second signal during it
            # escalates to SIGKILL instead of ending Python with a frozen tree.
            with _process_tree.forward_termination():
                try:
                    tree = _process_tree.ProcessTree.start(command, **popen_kwargs)
                except OSError as error:
                    raise ExecutableNotLaunchable(
                        f"cannot launch {command[0]}: {error}"
                    ) from error
                try:
                    if reservation is not None:
                        reservation.attach(tree.process.pid)
                    if not self._watches_memory():
                        try:
                            stdout, stderr = tree.process.communicate(stdin, timeout=timeout)
                        except subprocess.TimeoutExpired:
                            timed_out = True
                            tree.terminate()
                            stdout, stderr = tree.collect()
                    else:
                        stdout, stderr, timed_out, memory_stop, peak = self._watch(tree, stdin, timeout, started)
                except BaseException:
                    # KeyboardInterrupt, a forwarded signal, or one landing
                    # during the timeout's own stop. ``terminate`` is idempotent.
                    tree.terminate()
                    raise
                finally:
                    tree.close()
        except _process_tree.Terminated as stop:
            # Our handler is gone again; the default disposition decides.
            stop.redeliver()
            raise
        stopped = timed_out or memory_stop
        if memory_stop and not capture:
            # GPEC-style adapters report any limit stop as a timeout (#1460
            # follow-up); the solver's own log keeps the real reason.
            with open(log_path, "a", encoding="utf-8") as log:
                log.write(
                    f"\n[vaft LocalBackend] stopped: resident memory {peak:.0f} MiB passed "
                    f"memory_limit_mb={self.memory_limit_mb:.0f}\n"
                )
        return ExecutionResult(
            returncode=None if stopped else int(tree.process.returncode),
            stdout=_text(stdout) if capture else "",
            stderr=_text(stderr) if capture else "",
            timed_out=stopped,
            elapsed_s=time.monotonic() - started,
            launcher=command,
            log_path=log_path,
            runtime_status=RUNTIME_MEMORY_LIMIT if memory_stop else "",
            peak_rss_mb=peak,
        )

    def _watch(
        self, tree: "_process_tree.ProcessTree", stdin: Optional[str], timeout: Optional[float], started: float
    ) -> tuple[Any, Any, bool, bool, Optional[float]]:
        """Wait for the tree while sampling its resident size; stop it at a limit.

        ``communicate`` may be called again after ``TimeoutExpired`` without
        losing output; input is passed only on the first call.
        """
        from ._memory_gate import tree_rss_mb

        # One sample right away, so a program that exits before the first
        # poll still reports a peak (what it had resident at launch) rather
        # than the ``None`` an unwatched run carries.
        peak: Optional[float] = tree_rss_mb(tree.members())
        first = True
        while True:
            wait = self.poll_interval_s
            if timeout is not None:
                wait = max(0.0, min(wait, timeout - (time.monotonic() - started)))
            try:
                stdout, stderr = tree.process.communicate(stdin if first else None, timeout=wait)
                return stdout, stderr, False, False, peak
            except subprocess.TimeoutExpired:
                first = False
            rss = tree_rss_mb(tree.members())
            peak = rss if peak is None else max(peak, rss)
            over_memory = self.memory_limit_mb is not None and rss > self.memory_limit_mb
            over_time = timeout is not None and time.monotonic() - started >= timeout
            if over_memory or over_time:
                tree.terminate()
                stdout, stderr = tree.collect()
                return stdout, stderr, not over_memory, over_memory, peak


#: Environment variable that picks the backend for configs that name none.
BACKEND_ENV = "VAFT_EXECUTION_BACKEND"


def default_backend() -> ExecutionBackend:
    """The backend ``$VAFT_EXECUTION_BACKEND`` selects: ``local`` (default) or ``slurm``.

    ``slurm`` builds :meth:`vaft.code.slurm.SlurmBackend.from_environment`, so
    a whole session or pipeline can move onto a cluster without touching any
    adapter configuration.
    """
    name = os.environ.get(BACKEND_ENV, "").strip().lower()
    if name in ("", "local"):
        return LocalBackend()
    if name == "slurm":
        from .slurm import SlurmBackend

        return SlurmBackend.from_environment()
    raise ValueError(f"{BACKEND_ENV} must be 'local' or 'slurm', got {name!r}")


def resolve_backend(config: Any = None) -> ExecutionBackend:
    """The backend ``config`` names, else :func:`default_backend`."""
    backend = getattr(config, "backend", None)
    if backend is None:
        return default_backend()
    # A class has a ``run`` attribute too; only an instance can run a request.
    if isinstance(backend, type) or not isinstance(backend, ExecutionBackend):
        raise TypeError(
            f"backend must implement ExecutionBackend.run, got {type(backend).__name__}"
        )
    return backend


__all__ = [
    "BACKEND_ENV",
    "RUNTIME_COMPLETED",
    "RUNTIME_MEMORY_LIMIT",
    "RUNTIME_QUEUE_TIMEOUT",
    "RUNTIME_STATUSES",
    "RUNTIME_TIMEOUT",
    "THREAD_ENV_VARIABLES",
    "ExecutionBackend",
    "ExecutionRequest",
    "ExecutionResult",
    "LocalBackend",
    "ResourceRequest",
    "default_backend",
    "execution_environment",
    "resolve_backend",
    "timeout_reason",
]
