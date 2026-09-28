"""Start a program so that it and everything it starts can be stopped together.

``subprocess.run(timeout=...)`` kills only the direct child. A launcher that
does not ``exec`` (GACODE's scripts, FLARE's ``-n`` MPI driver, a ``.cmd``
wrapper) leaves its own children running after the kill, and on Windows the
wait can then hang until they exit. :class:`ProcessTree` puts the whole tree in
one unit the operating system can stop:

* POSIX: the child starts its own session (``start_new_session=True``), so it
  leads a process group that :meth:`ProcessTree.terminate` signals as a whole --
  ``SIGTERM``, :data:`TERMINATE_GRACE_S` seconds to exit, then ``SIGKILL``.
* Windows: the child is assigned to a Job Object created with
  ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``; terminating the job stops every
  process in it. If the job cannot be set up, ``taskkill /T /F`` stops the tree
  by parent id instead. A grandchild started before the assignment completes
  escapes the job; ``taskkill`` is not attempted for it.

A new session no longer receives the terminal's Ctrl-C or a signal sent to the
caller's process group (which is how Snakemake stops a job), so the caller must
forward them: :func:`forward_termination` turns ``SIGTERM`` into
:class:`Terminated` while a program runs, and the caller stops the tree and
calls :meth:`Terminated.redeliver`.

A process that makes itself a new session leader (``setsid``) or daemonizes has
left the tree on purpose and is not stopped.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import threading
import time
from typing import Any, Iterator, Optional, Sequence

#: Seconds a tree gets between the polite stop and the forced one.
TERMINATE_GRACE_S = 10.0

_POLL_S = 0.05
_WINDOWS = os.name == "nt"


class Terminated(BaseException):
    """``SIGTERM`` arrived while a program was running under :func:`forward_termination`.

    A ``BaseException`` like ``KeyboardInterrupt``, so ``except Exception``
    blocks between the wait and the caller do not swallow it.
    """

    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum

    def redeliver(self) -> None:
        """Deliver the signal again, now that the caller's own handler is back.

        Under the default disposition the process ends here, by the signal, as
        it would have without VAFT in the way. A handler that returns leaves
        the caller to re-raise this exception.
        """
        signal.raise_signal(self.signum)


@contextlib.contextmanager
def forward_termination() -> Iterator[None]:
    """Raise :class:`Terminated` on ``SIGTERM`` for the duration of the block.

    The handler is installed only on POSIX and only from the main thread
    (Python delivers signals there alone), and only when the current
    disposition can be put back afterwards: an ignored ``SIGTERM`` stays
    ignored, and a handler installed outside Python is left alone. The
    previous handler is restored on every exit from the block.
    """
    if _WINDOWS or threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGTERM)
    if previous is None or previous is signal.SIG_IGN:
        yield
        return

    def _stop(signum: int, frame: Any) -> None:
        raise Terminated(signum)

    try:
        signal.signal(signal.SIGTERM, _stop)
    except ValueError:  # not the main interpreter
        yield
        return
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


class ProcessTree:
    """A started program together with everything it starts."""

    def __init__(self, process: "subprocess.Popen[Any]", job: Any = None) -> None:
        self.process = process
        self._job = job

    @classmethod
    def start(cls, argv: Sequence[str], **popen_kwargs: Any) -> "ProcessTree":
        """``Popen(argv, **popen_kwargs)`` in its own process group or job.

        ``OSError`` from the launch propagates unchanged.
        """
        if not _WINDOWS:
            return cls(subprocess.Popen(list(argv), start_new_session=True, **popen_kwargs))
        process = subprocess.Popen(list(argv), **popen_kwargs)
        return cls(process, _windows_job_for(process))

    def terminate(self, grace: Optional[float] = None) -> None:
        """Stop the whole tree; return once its leader has been reaped."""
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
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.process.wait(timeout=grace)
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
        # The child leads its own session, so its process group id is its pid.
        group = self.process.pid
        if not _signal_group(group, signal.SIGTERM):
            self.process.poll()
            return
        try:
            deadline = time.monotonic() + grace
            while time.monotonic() < deadline:
                # Reap the leader: a zombie still counts as a group member.
                self.process.poll()
                if not _group_alive(group):
                    return
                time.sleep(_POLL_S)
        except BaseException:
            # A second interrupt during the grace period skips straight to
            # the forced stop rather than abandoning the tree.
            _signal_group(group, signal.SIGKILL)
            raise
        _signal_group(group, signal.SIGKILL)
        with contextlib.suppress(subprocess.TimeoutExpired):
            self.process.wait(timeout=grace)

    # -- Windows ----------------------------------------------------------

    def _terminate_windows(self) -> None:
        job, self._job = self._job, None
        if job is not None and _windows_terminate_job(job):
            return
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(self.process.pid)],
            capture_output=True,
            check=False,
        )


def _signal_group(group: int, signum: int) -> bool:
    try:
        os.killpg(group, signum)
    except ProcessLookupError:
        return False
    except PermissionError:
        # A member changed its credentials; it is out of reach either way.
        return False
    return True


def _group_alive(group: int) -> bool:
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


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
    except (OSError, AttributeError, ValueError):
        pass
    return None


def _windows_terminate_job(job: Any) -> bool:
    try:
        kernel32 = _kernel32()
        stopped = bool(kernel32.TerminateJobObject(job, 1))
        kernel32.CloseHandle(job)
        return stopped
    except (OSError, AttributeError, ValueError):
        return False


def _windows_release_job(job: Any) -> None:
    try:
        kernel32 = _kernel32()
        # Drop kill-on-close first so closing the handle leaves survivors alone.
        _set_limits(kernel32, job, 0)
        kernel32.CloseHandle(job)
    except (OSError, AttributeError, ValueError):
        pass


__all__ = ["TERMINATE_GRACE_S", "ProcessTree", "Terminated", "forward_termination"]
