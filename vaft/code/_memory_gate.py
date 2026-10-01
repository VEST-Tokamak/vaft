"""Memory admission and resident-size readings for child process trees (#1460).

:mod:`vaft.code.resources` guards the *current* Python process. A solver that
an adapter launches is a separate process tree, and several of them, started
by separate Python workers or by separate pipelines, share one host. Two
things are needed for that, and both live here:

* :func:`tree_rss_mb`, the summed resident size of a set of processes;
* :class:`MemoryLedger`, a host-wide ledger of memory *reservations*. A launch
  is admitted only when ``available - outstanding >= request + floor``, where
  ``outstanding`` is what already-admitted jobs have reserved but not yet
  touched (reservation minus their tree's current RSS). Counting only
  ``MemAvailable`` would admit every waiting job at once, because a job that
  was just started has not grown yet.

The ledger is a directory of one JSON file per admitted job plus a lock file
(``flock``), so it works across unrelated processes of the same user. A
record whose owner process is gone is dropped on the next admission. On a
host without ``fcntl`` (Windows) admission is not enforced.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

from .resources import _available_mib

try:  # POSIX only
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

_STDLIB_RUN = subprocess.run

#: Default ledger location: per user, on the local temp filesystem.
LEDGER_ENV = "VAFT_MEMORY_LEDGER_DIR"


def default_ledger_dir() -> Path:
    configured = os.environ.get(LEDGER_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path(tempfile.gettempdir()) / f"vaft-memory-ledger-{os.getuid() if hasattr(os, 'getuid') else 'user'}"


def available_mb() -> Optional[float]:
    """Memory the kernel could hand out now, in MiB (``MemAvailable``); ``None`` if unknown."""
    return _available_mib(None)


def tree_rss_mb(pids: Iterable[int]) -> float:
    """Summed resident set size of ``pids`` in MiB; processes that are gone count 0."""
    pids = [int(pid) for pid in pids]
    if not pids:
        return 0.0
    if os.path.isdir("/proc/self/task"):
        total_kib = 0.0
        for pid in pids:
            try:
                with open(f"/proc/{pid}/status", "rb") as status:
                    for line in status:
                        if line.startswith(b"VmRSS:"):
                            total_kib += float(line.split()[1])
                            break
            except (OSError, ValueError, IndexError):
                continue
        return total_kib / 1024.0
    try:
        listing = _STDLIB_RUN(
            ["ps", "-o", "rss=", "-p", ",".join(str(pid) for pid in pids)],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return 0.0
    return sum(float(part) for part in listing.stdout.split() if part.strip().isdigit()) / 1024.0


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class Reservation:
    """One admitted job's entry in the ledger; release it when the job ends."""

    def __init__(self, ledger: "MemoryLedger", path: Path, reserve_mb: float) -> None:
        self.ledger = ledger
        self.path = path
        self.reserve_mb = reserve_mb

    def attach(self, root_pid: int) -> None:
        """Record the launched program's pid so its growth reduces what it still claims."""
        record = {"owner": os.getpid(), "reserve_mb": self.reserve_mb, "root": int(root_pid)}
        with contextlib.suppress(OSError):
            self.path.write_text(json.dumps(record))

    def release(self) -> None:
        with contextlib.suppress(OSError):
            self.path.unlink()


class MemoryLedger:
    """Host-wide admission control for memory reservations (see module docstring).

    ``members`` maps a recorded root pid to the pids of its tree; it defaults
    to the root alone and is replaced by :class:`LocalBackend` with the
    process-tree walk, so descendants count too.
    """

    def __init__(
        self,
        directory: Optional[Path] = None,
        *,
        floor_mb: float = 4096.0,
        available: Callable[[], Optional[float]] = available_mb,
        members: Callable[[int], Iterable[int]] = lambda root: (root,),
    ) -> None:
        self.directory = Path(directory) if directory is not None else default_ledger_dir()
        self.floor_mb = float(floor_mb)
        self._available = available
        self._members = members

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        self.directory.mkdir(parents=True, exist_ok=True)
        with open(self.directory / ".lock", "a+") as lock:
            if fcntl is not None:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def outstanding_mb(self) -> float:
        """Reserved but not yet resident memory of live admitted jobs; drops dead records."""
        total = 0.0
        for path in self.directory.glob("*.json"):
            try:
                record = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if not _alive(int(record.get("owner", -1))):
                with contextlib.suppress(OSError):
                    path.unlink()
                continue
            root = record.get("root")
            used = 0.0 if root is None else tree_rss_mb(self._members(int(root)))
            total += max(0.0, float(record["reserve_mb"]) - used)
        return total

    def try_admit(self, reserve_mb: float) -> Optional[Reservation]:
        """Admit now if the host has room, else ``None``. Unknown availability admits."""
        with self._locked():
            available = self._available()
            if available is not None and available - self.outstanding_mb() < reserve_mb + self.floor_mb:
                return None
            path = self.directory / f"{uuid.uuid4().hex}.json"
            path.write_text(json.dumps({"owner": os.getpid(), "reserve_mb": float(reserve_mb), "root": None}))
            return Reservation(self, path, float(reserve_mb))

    def admit(self, reserve_mb: float, *, wait_s: Optional[float], poll_s: float = 5.0) -> Optional[Reservation]:
        """Wait up to ``wait_s`` seconds (``None``: forever) for admission."""
        started = time.monotonic()
        while True:
            reservation = self.try_admit(reserve_mb)
            if reservation is not None:
                return reservation
            if wait_s is not None and time.monotonic() - started >= wait_s:
                return None
            time.sleep(poll_s)


__all__ = ["LEDGER_ENV", "MemoryLedger", "Reservation", "available_mb", "default_ledger_dir", "tree_rss_mb"]
