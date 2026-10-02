"""Serialize the read-merge-replace of one shot's ``master.h5`` (issue #913).

A shot's ``master.h5`` names every IDS file beside it, and a write replaces it
with "what was there, plus this write's IDS". Two writes of one shot that
overlap both read the old master, and whichever replaces it last drops the
other's links: the files stay on HSDS, unreachable. On 2026-09-17 that hid
six IDS of shots 39241 and 39620 behind records that said ``passed: true``.

:func:`shot_master_lock` holds an exclusive advisory lock per ``(source, shot)``
from the master read to the master replace. What it covers, and what not:

* **Every writer on one host** that goes through VAFT -- the routine pipeline,
  the corrective updaters, the new-shot worker, maintenance repairs -- because
  the lock lives in one host-wide directory: ``$VAFT_HSDS_LOCK_DIR``, or
  ``/tmp/vaft-hsds-locks``. It is deliberately not ``tempfile.gettempdir()``:
  that follows ``$TMPDIR``, which Snakemake jobs and other tools may set
  differently, and two writers with two lock directories are not serialized.
  When that directory cannot be written by this account (made by another one
  without the sticky world-writable mode), the lock moves to a per-user
  directory under the temp root (:func:`fallback_lock_directory`), with one
  warning naming both: a lock that serializes this account's own writers is
  weaker than the host-wide one, but refusing to write at all would turn a
  directory mode into a failed replication.
* **Not writers on another host**, nor tools that bypass VAFT (``hsload`` by
  hand). The write path also re-reads the remote master immediately before
  replacing it, which narrows that window but cannot close it.
* **Not Windows**, where ``fcntl`` is absent; the lock is a no-op there.

Different shots never wait for each other. The lock is re-entrant within a
thread, so a caller that holds it -- replication, around its whole attempt --
can call the write path, which takes it again.
"""

from __future__ import annotations

from contextlib import contextmanager
import logging
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from typing import Iterator
import warnings

logger = logging.getLogger(__name__)

LOCK_DIR_ENV = "VAFT_HSDS_LOCK_DIR"
DEFAULT_LOCK_DIR = "/tmp/vaft-hsds-locks"
#: Seconds to wait for another writer of the same shot.  An upload holding it
#: takes a minute or two; far longer means a hung writer, and the replication
#: attempt that times out is retried like any other transient failure.
DEFAULT_TIMEOUT = 1800.0

_held: dict[tuple[str, int], int] = {}
_registry = threading.Lock()
#: Lock directories already reported as unwritable; the fallback is said once.
_fallback_warned: set[str] = set()


class MasterLockTimeout(TimeoutError):
    """Another writer held the shot's master lock for longer than allowed."""


def lock_directory() -> Path:
    return Path(os.environ.get(LOCK_DIR_ENV) or DEFAULT_LOCK_DIR)


def fallback_lock_directory() -> Path:
    """Where the lock goes when :func:`lock_directory` cannot be written.

    Per user under the temp root, so it is always creatable; shared only with
    this account's other writers that fall back the same way.
    """
    try:
        owner = str(os.getuid())
    except AttributeError:  # pragma: no cover - no uid on Windows
        owner = os.environ.get("USERNAME") or os.environ.get("USER") or "user"
    return Path(tempfile.gettempdir()) / f"vaft-hsds-locks-{owner}"


def lock_path(source: str, shot: int) -> Path:
    """The lock file for one ``(source, shot)``; a nested source is flattened."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "__", str(source).strip("/"))
    return lock_directory() / f"{safe}.{int(shot)}.lock"


def _open_in(path: Path) -> int:
    """Open one lock file, creating its directory; raise if neither can be made."""
    directory = path.parent
    if not directory.is_dir():
        directory.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(directory, 0o1777)  # sticky and shared, like /tmp itself
        except OSError:
            pass
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    except PermissionError:
        # Either the file exists read-only (fine) or the directory refuses the
        # create, which the read-only open reports as FileNotFoundError.
        return os.open(path, os.O_RDONLY)
    try:
        os.fchmod(fd, 0o666)
    except OSError:  # not the owner: the owner made it shareable already, or not
        pass
    return fd


def _open_shared(path: Path) -> int:
    """Open a lock file every account on the host can lock.

    Created world-writable (the umask would otherwise leave it 0644, and a
    second account could not open it for writing). ``flock`` needs no write
    access, so a file some other account created read-only is opened read-only
    rather than refused -- one operator's audit must not lock the worker out.

    A directory this account cannot create files in -- or cannot create at
    all -- sends the lock to :func:`fallback_lock_directory` with a warning,
    once per directory, instead of failing the write that asked for it.
    """
    try:
        return _open_in(path)
    except OSError as error:
        fallback = fallback_lock_directory()
        if path.parent == fallback:
            raise
        message = (
            f"the HSDS master lock directory {path.parent} cannot be written by this "
            f"account ({error}); locking in {fallback} instead, which serializes only "
            f"this account's writers. Make it shared (chmod 1777 {path.parent}) or set "
            f"${LOCK_DIR_ENV} to a directory every writer on this host can write."
        )
        if str(path.parent) not in _fallback_warned:
            _fallback_warned.add(str(path.parent))
            warnings.warn(message, RuntimeWarning, stacklevel=3)
            logger.warning(message)
        return _open_in(fallback / path.name)


@contextmanager
def shot_master_lock(source: str, shot: int, *, timeout: float | None = DEFAULT_TIMEOUT) -> Iterator[None]:
    """Hold the exclusive master lock of ``(source, shot)``; see the module docstring."""
    try:
        import fcntl
    except ImportError:  # pragma: no cover - Windows
        yield
        return

    key = (f"{source}:{int(shot)}", threading.get_ident())
    with _registry:
        depth = _held.get(key, 0)
        if depth:
            _held[key] = depth + 1
    if depth:
        try:
            yield
        finally:
            with _registry:
                _held[key] -= 1
        return

    path = lock_path(source, shot)
    fd = _open_shared(path)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            logger.info("waiting for another writer of %s shot %s (%s)", source, shot, path)
            deadline = None if timeout is None else time.monotonic() + timeout
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if deadline is not None and time.monotonic() >= deadline:
                        raise MasterLockTimeout(
                            f"another writer held the master of {source} shot {shot} for over "
                            f"{timeout:g} s ({path})"
                        ) from None
                    time.sleep(0.2)
        with _registry:
            _held[key] = 1
        try:
            yield
        finally:
            with _registry:
                _held.pop(key, None)
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


__all__ = [
    "DEFAULT_LOCK_DIR",
    "LOCK_DIR_ENV",
    "MasterLockTimeout",
    "fallback_lock_directory",
    "lock_path",
    "shot_master_lock",
]
