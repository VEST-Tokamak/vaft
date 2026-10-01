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
import threading
import time
from typing import Iterator

logger = logging.getLogger(__name__)

LOCK_DIR_ENV = "VAFT_HSDS_LOCK_DIR"
DEFAULT_LOCK_DIR = "/tmp/vaft-hsds-locks"
#: Seconds to wait for another writer of the same shot.  An upload holding it
#: takes a minute or two; far longer means a hung writer, and the replication
#: attempt that times out is retried like any other transient failure.
DEFAULT_TIMEOUT = 1800.0

_held: dict[tuple[str, int], int] = {}
_registry = threading.Lock()


class MasterLockTimeout(TimeoutError):
    """Another writer held the shot's master lock for longer than allowed."""


def lock_directory() -> Path:
    return Path(os.environ.get(LOCK_DIR_ENV) or DEFAULT_LOCK_DIR)


def lock_path(source: str, shot: int) -> Path:
    """The lock file for one ``(source, shot)``; a nested source is flattened."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "__", str(source).strip("/"))
    return lock_directory() / f"{safe}.{int(shot)}.lock"


def _open_shared(path: Path) -> int:
    """Open a lock file every account on the host can lock.

    Created world-writable (the umask would otherwise leave it 0644, and a
    second account could not open it for writing). ``flock`` needs no write
    access, so a file some other account created read-only is opened read-only
    rather than refused -- one operator's audit must not lock the worker out.
    """
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
        return os.open(path, os.O_RDONLY)
    try:
        os.fchmod(fd, 0o666)
    except OSError:  # not the owner: the owner made it shareable already, or not
        pass
    return fd


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


__all__ = ["DEFAULT_LOCK_DIR", "LOCK_DIR_ENV", "MasterLockTimeout", "lock_path", "shot_master_lock"]
