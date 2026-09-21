"""Execution backends: how an adapter's prepared command is actually run.

An adapter's ``run_*`` owns the science: it resolves the executable, builds the
command line and turns the solver's exit into a result. *How* that command is
launched -- as a local child process today, through Slurm later -- is the
backend's job, so the launch behaviour (environment merge, timeout, output
capture, launch failure) is written once instead of once per adapter.

The contract every backend keeps:

* ``run`` blocks until the program exits or times out.
* A timeout is **returned**, not raised: ``timed_out=True``, ``returncode=None``
  and whatever output was captured before the kill. Each adapter maps that to
  its own documented timeout result (TES returns 124, EFIT marks the slice
  ``"timeout"``, ...), so moving an adapter onto a backend changes nothing its
  callers see.
* A program the operating system will not start raises
  :class:`~vaft.code._executables.ExecutableNotLaunchable`, chained to the
  ``OSError``.
* ``ResourceRequest`` is a declaration. :class:`LocalBackend` honours only
  ``threads_per_task``; ``ntasks`` and ``memory_mb`` are for scheduler backends.
  Codes that start their own MPI ranks (GACODE, FLARE) keep passing ``-n`` on
  their command line.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable

from ._executables import ExecutableNotLaunchable

#: Thread-count variables the common Fortran/BLAS runtimes read.
THREAD_ENV_VARIABLES: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


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
    """Run the command as a child process of the current Python process."""

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
        if request.log_path is None:
            return self._run(command, request, kwargs)
        # Opened outside the launch so a bad log path stays its own error
        # rather than being reported as an unlaunchable program.
        log_path = Path(request.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            kwargs.update(stdout=log, stderr=subprocess.STDOUT)
            return self._run(command, request, kwargs)

    @staticmethod
    def _run(
        command: tuple[str, ...], request: ExecutionRequest, kwargs: dict[str, Any]
    ) -> ExecutionResult:
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
    "THREAD_ENV_VARIABLES",
    "ExecutionBackend",
    "ExecutionRequest",
    "ExecutionResult",
    "LocalBackend",
    "ResourceRequest",
    "default_backend",
    "execution_environment",
    "resolve_backend",
]
