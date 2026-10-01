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

The ledger is shared only by processes that use the same directory. The
default is per user under the temp directory, so workers with different
``TMPDIR`` (Slurm jobs, for instance) must set :data:`LEDGER_ENV` to one path.
The directory is created ``0700`` and refused if another user owns it.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import tempfile
import time
import uuid
import warnings
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

from . import _process_tree
from .resources import _DEFAULT_MEMINFO, _available_mib

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


def total_mb() -> Optional[float]:
    """Physical memory of the host in MiB; ``None`` if unknown."""
    try:
        for line in _DEFAULT_MEMINFO.read_text().splitlines():
            if line.startswith("MemTotal:"):
                return float(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    try:
        import psutil  # optional
    except ImportError:
        return None
    try:
        return float(psutil.virtual_memory().total) / (1024.0 * 1024.0)
    except Exception:  # pragma: no cover - psutil platform failure
        return None


def rss_by_pid(pids: Iterable[int]) -> dict[int, float]:
    """Resident size [MiB] of each live pid in ``pids``; one ``ps`` call off Linux."""
    pids = sorted({int(pid) for pid in pids})
    if not pids:
        return {}
    found: dict[int, float] = {}
    if os.path.isdir("/proc/self/task"):
        for pid in pids:
            try:
                with open(f"/proc/{pid}/status", "rb") as status:
                    for line in status:
                        if line.startswith(b"VmRSS:"):
                            found[pid] = float(line.split()[1]) / 1024.0
                            break
            except (OSError, ValueError, IndexError):
                continue
        return found
    try:
        listing = _STDLIB_RUN(
            ["ps", "-o", "pid=,rss=", "-p", ",".join(str(pid) for pid in pids)],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return found
    for line in listing.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            found[int(parts[0])] = float(parts[1]) / 1024.0
    return found


def tree_rss_mb(pids: Iterable[int]) -> float:
    """Summed resident set size of ``pids`` in MiB; processes that are gone count 0."""
    return sum(rss_by_pid(pids).values())


def _alive(pid: int) -> bool:
    """Whether ``pid`` is a live process of this user.

    The ledger directory is private to one user, so an owner pid that now
    belongs to someone else (``PermissionError``) was reused: the record is
    stale.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _write_json_atomic(path: Path, record: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(record))
    os.replace(temporary, path)


def _private_directory(directory: Path) -> Path:
    """Create ``directory`` 0700, or refuse one that is not this user's own."""
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if directory.is_symlink():
        raise PermissionError(f"memory ledger {directory} is a symlink; refusing it")
    if hasattr(os, "getuid"):
        info = directory.stat()
        if info.st_uid != os.getuid():
            raise PermissionError(f"memory ledger {directory} belongs to uid {info.st_uid}; set {LEDGER_ENV}")
    return directory


_UNKNOWN_AVAILABILITY_WARNED = False


class Reservation:
    """One admitted job's entry in the ledger; release it when the job ends."""

    def __init__(self, ledger: "MemoryLedger", path: Path, reserve_mb: float) -> None:
        self.ledger = ledger
        self.path = path
        self.reserve_mb = reserve_mb

    def attach(self, root_pid: int) -> None:
        """Record the launched program's pid so its growth reduces what it still claims."""
        record = {"owner": os.getpid(), "reserve_mb": self.reserve_mb, "root": int(root_pid)}
        with contextlib.suppress(OSError), self.ledger._locked():
            _write_json_atomic(self.path, record)

    def release(self) -> None:
        with contextlib.suppress(OSError):
            self.path.unlink()


class MemoryLedger:
    """Host-wide admission control for memory reservations (see module docstring).

    A recorded root pid counts together with its descendants, found from one
    process table per admission. ``available`` and ``total`` are injectable
    for tests.
    """

    def __init__(
        self,
        directory: Optional[Path] = None,
        *,
        floor_mb: float = 4096.0,
        available: Callable[[], Optional[float]] = available_mb,
        total: Callable[[], Optional[float]] = total_mb,
    ) -> None:
        if floor_mb < 0:
            raise ValueError(f"floor_mb must be >= 0, got {floor_mb}")
        self.directory = Path(directory) if directory is not None else default_ledger_dir()
        self.floor_mb = float(floor_mb)
        self._available = available
        self._total = total

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        _private_directory(self.directory)
        with open(self.directory / ".lock", "a+") as lock:
            if fcntl is not None:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def outstanding_mb(self) -> float:
        """Reserved but not yet resident memory of live admitted jobs.

        Records of dead owners and records that cannot be read as
        ``{"owner": int, "reserve_mb": number, "root": int | null}`` are removed.
        """
        records = []
        for path in self.directory.glob("*.json"):
            try:
                record = json.loads(path.read_text())
                owner = int(record["owner"])
                reserve = float(record["reserve_mb"])
                root = None if record.get("root") is None else int(record["root"])
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                record = None
            if record is None or not _alive(owner):
                with contextlib.suppress(OSError):
                    path.unlink()
                continue
            records.append((reserve, root))
        roots = [root for _, root in records if root is not None]
        table = _process_tree._process_table() if roots else {}
        trees = {root: {root} | _process_tree._descendants(root, table) for root in roots}
        rss = rss_by_pid(pid for tree in trees.values() for pid in tree)
        total = 0.0
        for reserve, root in records:
            used = 0.0 if root is None else sum(rss.get(pid, 0.0) for pid in trees[root])
            total += max(0.0, reserve - used)
        return total

    def fits(self, reserve_mb: float) -> bool:
        """Whether ``reserve_mb`` + floor could ever be admitted on this host."""
        total = self._total()
        return total is None or reserve_mb + self.floor_mb <= total

    def try_admit(self, reserve_mb: float) -> Optional[Reservation]:
        """Admit now if the host has room, else ``None``. Unknown availability admits."""
        global _UNKNOWN_AVAILABILITY_WARNED
        with self._locked():
            available = self._available()
            if available is None and not _UNKNOWN_AVAILABILITY_WARNED:
                _UNKNOWN_AVAILABILITY_WARNED = True
                warnings.warn(
                    "available memory cannot be read on this host; memory admission is not enforced",
                    RuntimeWarning,
                    stacklevel=3,
                )
            if available is not None and available - self.outstanding_mb() < reserve_mb + self.floor_mb:
                return None
            path = self.directory / f"{uuid.uuid4().hex}.json"
            _write_json_atomic(path, {"owner": os.getpid(), "reserve_mb": float(reserve_mb), "root": None})
            return Reservation(self, path, float(reserve_mb))

    def admit(self, reserve_mb: float, *, wait_s: Optional[float], poll_s: float = 5.0) -> Optional[Reservation]:
        """Wait up to ``wait_s`` seconds (``None``: forever) for admission.

        A reservation that cannot fit even on an idle host returns ``None`` at once.
        """
        if not self.fits(reserve_mb):
            return None
        started = time.monotonic()
        while True:
            reservation = self.try_admit(reserve_mb)
            if reservation is not None:
                return reservation
            if wait_s is not None and time.monotonic() - started >= wait_s:
                return None
            time.sleep(poll_s)


__all__ = [
    "LEDGER_ENV",
    "MemoryLedger",
    "Reservation",
    "available_mb",
    "default_ledger_dir",
    "rss_by_pid",
    "total_mb",
    "tree_rss_mb",
]
