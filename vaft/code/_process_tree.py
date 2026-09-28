"""Stop a program together with everything it started.

``subprocess.run(timeout=...)`` kills only the direct child. A launcher that
does not ``exec`` (GACODE's scripts, FLARE's ``-n`` MPI driver, a ``.cmd``
wrapper) leaves its own children running after the kill, and on Windows the
wait can then hang until they exit. :class:`ProcessTree` stops the whole tree:

* POSIX: the child stays in the caller's process group, so every signal sent
  to that group -- a terminal's Ctrl-C and Ctrl-Z, a supervisor's ``SIGSTOP``
  or ``SIGKILL``, Snakemake stopping a job -- still reaches the solver exactly
  as it did before (#1016). :meth:`ProcessTree.terminate` finds the tree by
  parent id and -- on Linux -- by the :data:`TREE_ENV` token every descendant
  inherits, which also finds one whose parent already exited (a launcher
  script that died of Ctrl-C while its background job ignored it). It freezes
  the tree with ``SIGSTOP`` so nothing forks while it is being walked, sends ``SIGTERM`` (and ``SIGCONT`` so it can act on it), waits
  :data:`TERMINATE_GRACE_S` seconds, then sends ``SIGKILL`` to what is left.
* Windows: the child is assigned to a Job Object created with
  ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``; terminating the job stops every
  process in it. If the job cannot be set up, ``taskkill /T /F`` stops the tree
  by parent id instead.

A signal sent to the caller alone (``kill <pid>``) does not reach the tree, so
:func:`forward_termination` turns ``SIGTERM`` and ``SIGHUP`` into
:class:`Terminated` while a program runs; the caller stops the tree and calls
:meth:`Terminated.redeliver`. Ctrl-C arrives as ``KeyboardInterrupt``.

Not covered:

* macOS: a descendant whose parent had already exited is no longer found
  (the token cannot be read there). On Linux, one that cleared its
  environment (``env -i``) is found only by parent id.
* A process that makes itself a new session leader or daemonizes.
* Windows: a grandchild started before the job assignment completes escapes
  the job, and when an escaped process keeps the pipes open the output read
  before the stop is lost (``communicate`` there reports none on timeout).
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import threading
import time
import uuid
from typing import Any, Iterable, Iterator, Optional, Sequence

#: Seconds a tree gets between the polite stop and the forced one.
TERMINATE_GRACE_S = 10.0

#: Environment variable carrying a per-launch token that every descendant
#: inherits, so a stop finds processes that were reparented away (Linux).
TREE_ENV = "VAFT_PROCESS_TREE"

#: Signals :func:`forward_termination` turns into :class:`Terminated` (POSIX).
FORWARDED_SIGNALS: tuple[str, ...] = ("SIGTERM", "SIGHUP")

_POLL_S = 0.1
_FREEZE_PASSES = 10
_WINDOWS = os.name == "nt"
_STDLIB_RUN = subprocess.run


class Terminated(BaseException):
    """A forwarded signal arrived while a program was running under :func:`forward_termination`.

    A ``BaseException`` like ``KeyboardInterrupt``, so ``except Exception``
    blocks between the wait and the caller do not swallow it.
    """

    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum

    def redeliver(self) -> None:
        """Deliver the signal again, now that the default disposition is back.

        The process then ends by the signal, as it would have without VAFT in
        the way.
        """
        signal.raise_signal(self.signum)


def _forwarded() -> list[int]:
    return [getattr(signal, name) for name in FORWARDED_SIGNALS if hasattr(signal, name)]


@contextlib.contextmanager
def forward_termination() -> Iterator[None]:
    """Raise :class:`Terminated` on each of :data:`FORWARDED_SIGNALS` in the block.

    A handler is installed only on POSIX, only from the main thread (Python
    delivers signals there alone) and only over the **default** disposition:
    an ignored signal stays ignored (``nohup``), and a caller's own handler --
    one that drains gracefully, say -- keeps its choice. The signals are
    blocked while handlers are swapped, so one that arrives meanwhile is
    delivered after the swap rather than lost or left half-installed.
    """
    if _WINDOWS or threading.current_thread() is not threading.main_thread():
        yield
        return
    signums = _forwarded()
    replaced: list[int] = []

    def _stop(signum: int, frame: Any) -> None:
        raise Terminated(signum)

    try:
        with _blocked(signums):
            for signum in signums:
                if signal.getsignal(signum) is not signal.SIG_DFL:
                    continue
                try:
                    signal.signal(signum, _stop)
                except ValueError:  # not the main interpreter
                    break
                replaced.append(signum)
        yield
    finally:
        with _blocked(signums):
            for signum in replaced:
                signal.signal(signum, signal.SIG_DFL)


@contextlib.contextmanager
def _blocked(signums: Sequence[int]) -> Iterator[None]:
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, signums)
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


class ProcessTree:
    """A started program together with everything it starts."""

    def __init__(
        self, process: "subprocess.Popen[Any]", job: Any = None, token: Optional[str] = None
    ) -> None:
        self.process = process
        self._job = job
        self._token = token

    @classmethod
    def start(cls, argv: Sequence[str], **popen_kwargs: Any) -> "ProcessTree":
        """``Popen(argv, **popen_kwargs)``; on Windows, inside a kill-on-close job.

        ``OSError`` from the launch propagates unchanged.
        """
        token = uuid.uuid4().hex
        environment = dict(os.environ if popen_kwargs.get("env") is None else popen_kwargs["env"])
        environment[TREE_ENV] = token
        popen_kwargs["env"] = environment
        process = subprocess.Popen(list(argv), **popen_kwargs)
        return cls(process, _windows_job_for(process) if _WINDOWS else None, token)

    def terminate(self, grace: Optional[float] = None) -> None:
        """Stop the whole tree; return once its leader has been reaped.

        Safe to call again: a tree that is already stopped is left alone.
        """
        grace = TERMINATE_GRACE_S if grace is None else float(grace)
        if _WINDOWS:
            self._terminate_windows()
        else:
            self._terminate_posix(grace)

    def collect(self, grace: Optional[float] = None) -> tuple[Any, Any]:
        """Finish reading a stopped tree's output, waiting at most ``grace`` s.

        A process that escaped the tree may still hold the pipes open; the
        output read so far is returned rather than waiting on it.
        """
        grace = TERMINATE_GRACE_S if grace is None else float(grace)
        try:
            return self.process.communicate(timeout=grace)
        except subprocess.TimeoutExpired as expired:
            self.process.kill()
            _reap(self.process)
            return expired.output, expired.stderr

    def close(self) -> None:
        """Release the job handle without stopping what is still in it.

        A program that exited on its own keeps the behaviour it had before
        the job existed: whatever it left running is not killed here.
        """
        if self._job is not None:
            _windows_release_job(self._job)
            self._job = None

    # -- POSIX ------------------------------------------------------------

    def _terminate_posix(self, grace: float) -> None:
        # Filled while freezing, so an interrupt mid-walk still kills what was
        # already stopped instead of leaving it frozen.
        tree: set[int] = set()
        try:
            self._freeze(tree)
            _send(tree, signal.SIGTERM)
            _send(tree, signal.SIGCONT)
            deadline = time.monotonic() + grace
            while time.monotonic() < deadline:
                self.process.poll()  # reap the leader; a zombie is not "alive"
                table = _process_table()
                if not any(_alive(pid, table) for pid in tree):
                    return
                time.sleep(_POLL_S)
        except BaseException:
            # A second interrupt skips to the forced stop rather than
            # abandoning a frozen tree.
            _send(tree, signal.SIGKILL)
            _reap(self.process)
            raise
        _send(tree, signal.SIGKILL)
        _reap(self.process)

    def _freeze(self, frozen: set[int]) -> None:
        """``SIGSTOP`` the leader and its descendants into ``frozen`` until no new one appears."""
        for _ in range(_FREEZE_PASSES):
            table = _process_table()
            # A reaped leader's pid may already belong to someone else.
            roots = {self.process.pid} if self.process.poll() is None else set()
            roots |= _tagged(self._token, table)
            found: set[int] = set()
            for root in roots:
                found |= _descendants(root, table) | {root}
            found = {pid for pid in found if _alive(pid, table) and pid != os.getpid()}
            fresh = found - frozen
            if not fresh:
                break
            frozen |= fresh
            _send(fresh, signal.SIGSTOP)

    # -- Windows ----------------------------------------------------------

    def _terminate_windows(self) -> None:
        if self.process.poll() is not None and self._job is None:
            return
        job, self._job = self._job, None
        if job is None or not _windows_terminate_job(job):
            _STDLIB_RUN(
                ["taskkill", "/T", "/F", "/PID", str(self.process.pid)],
                capture_output=True,
                check=False,
            )
        _reap(self.process)


def _reap(process: "subprocess.Popen[Any]") -> None:
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=2.0)


def _send(pids: Iterable[int], signum: int) -> None:
    for pid in pids:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signum)


# -- POSIX process table --------------------------------------------------


def _process_table() -> dict[int, tuple[int, str]]:
    """``{pid: (ppid, state)}`` for every process this user can see."""
    if os.path.isdir("/proc/self/task"):
        table: dict[int, tuple[int, str]] = {}
        for entry in os.scandir("/proc"):
            if not entry.name.isdigit():
                continue
            try:
                with open(f"/proc/{entry.name}/stat", "rb") as stat:
                    # comm may contain spaces and parentheses; it ends at the last ')'.
                    fields = stat.read().rsplit(b")", 1)[1].split()
            except (OSError, IndexError):
                continue
            table[int(entry.name)] = (int(fields[1]), fields[0].decode("ascii", "replace"))
        return table
    listing = _STDLIB_RUN(
        ["ps", "-A", "-o", "pid=,ppid=,stat="], capture_output=True, text=True, check=False
    )
    table = {}
    for line in listing.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit():
            table[int(parts[0])] = (int(parts[1]), parts[2])
    return table


def _tagged(token: Optional[str], table: dict[int, tuple[int, str]]) -> set[int]:
    """Processes whose environment carries ``TREE_ENV=token`` (Linux only)."""
    if token is None or not os.path.isdir("/proc/self/task"):
        return set()
    needle = f"{TREE_ENV}={token}".encode("ascii")
    tagged: set[int] = set()
    for pid in table:
        try:
            with open(f"/proc/{pid}/environ", "rb") as environ:
                if needle in environ.read().split(b"\0"):
                    tagged.add(pid)
        except OSError:  # gone, or another user's process
            continue
    return tagged


def _descendants(root: int, table: dict[int, tuple[int, str]]) -> set[int]:
    children: dict[int, list[int]] = {}
    for pid, (ppid, _) in table.items():
        children.setdefault(ppid, []).append(pid)
    found: set[int] = set()
    pending = [root]
    while pending:
        for child in children.get(pending.pop(), ()):
            if child not in found:
                found.add(child)
                pending.append(child)
    return found


def _alive(pid: int, table: dict[int, tuple[int, str]]) -> bool:
    entry = table.get(pid)
    return entry is not None and not entry[1].startswith(("Z", "X"))


# -- Windows Job Object ---------------------------------------------------

_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000


def _kernel32() -> Any:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    )
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
    kernel32.TerminateJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    return kernel32


def _limit_information(flags: int) -> Any:
    import ctypes
    from ctypes import wintypes

    class _Basic(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in (
            "ReadOperationCount",
            "WriteOperationCount",
            "OtherOperationCount",
            "ReadTransferCount",
            "WriteTransferCount",
            "OtherTransferCount",
        )]

    class _Extended(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _Basic),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    info = _Extended()
    info.BasicLimitInformation.LimitFlags = flags
    return info


def _set_limits(kernel32: Any, job: Any, flags: int) -> bool:
    import ctypes

    info = _limit_information(flags)
    return bool(
        kernel32.SetInformationJobObject(
            job, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(info), ctypes.sizeof(info)
        )
    )


def _windows_job_for(process: "subprocess.Popen[Any]") -> Any:
    """A kill-on-close job holding ``process``, or ``None`` if one cannot be made."""
    try:
        kernel32 = _kernel32()
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None
        if _set_limits(kernel32, job, _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE) and (
            kernel32.AssignProcessToJobObject(job, int(process._handle))  # type: ignore[attr-defined]
        ):
            return job
        kernel32.CloseHandle(job)
    except Exception:  # ctypes.ArgumentError is not an OSError
        pass
    return None


def _windows_terminate_job(job: Any) -> bool:
    try:
        kernel32 = _kernel32()
        stopped = bool(kernel32.TerminateJobObject(job, 1))
        kernel32.CloseHandle(job)
        return stopped
    except Exception:  # ctypes.ArgumentError is not an OSError
        return False


def _windows_release_job(job: Any) -> None:
    try:
        kernel32 = _kernel32()
        # Drop kill-on-close first so closing the handle leaves survivors alone.
        _set_limits(kernel32, job, 0)
        kernel32.CloseHandle(job)
    except Exception:  # ctypes.ArgumentError is not an OSError
        pass


__all__ = [
    "FORWARDED_SIGNALS",
    "TERMINATE_GRACE_S",
    "TREE_ENV",
    "ProcessTree",
    "Terminated",
    "forward_termination",
]
