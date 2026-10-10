"""Opt-in I/O instrumentation for HSDS access (#1869).

Nothing here runs unless a caller enters :func:`record_io`. Inside it, every
HTTP request made through :class:`requests.Session` -- which is how h5pyd talks
to HSDS -- is recorded, together with the hsget/hsload subprocesses
:mod:`vaft.database.transport` starts. Outside it, no function is patched and
behaviour is unchanged.

What is observed and what is not:

* ``requests``: one record per request -- method, URL path, status, request body
  size, and the response ``Content-Length`` when the server sends one
  (``response_bytes_observed``; otherwise ``None``). ``seconds`` is the time to
  the response **headers**: h5pyd reads with ``stream=True``, so the body is
  downloaded after ``Session.send`` returns and is not in this latency. Wall
  time of a whole operation is the caller's measurement.
* ``subprocess``: one record per hsget/hsload -- wall seconds and the size the
  transport passed (``hsstat`` total for downloads, the local file for
  uploads). Their HTTP traffic happens in another process and is not counted
  as requests.

Each request also carries an ``X-VAFT-Run-ID`` header while recording, so HSDS
access logs can be correlated with a benchmark run.
"""

from __future__ import annotations

import contextlib
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterator
from urllib.parse import urlsplit

import numpy as np

__all__ = ["IORecorder", "RUN_ID_HEADER", "active_recorder", "record_io"]

#: Header added to every request while a recorder is active.
RUN_ID_HEADER = "X-VAFT-Run-ID"

_lock = threading.Lock()
_stack: list["IORecorder"] = []
_original_send = None


@dataclass
class IORecorder:
    """The requests and subprocess transfers seen while it was active."""

    run_id: str
    requests: list[dict[str, Any]] = field(default_factory=list)
    subprocesses: list[dict[str, Any]] = field(default_factory=list)
    started_wall: float = field(default_factory=time.time)
    ended_wall: float | None = None

    def _add(self, kind: str, record: dict[str, Any]) -> None:
        with _lock:
            (self.requests if kind == "request" else self.subprocesses).append(record)

    def summary(self) -> dict[str, Any]:
        """Counts, observed bytes and header-latency percentiles; ``None`` where nothing was measured."""
        with _lock:
            requests, subprocesses = list(self.requests), list(self.subprocesses)
        by_method: dict[str, int] = {}
        for record in requests:
            by_method[record["method"]] = by_method.get(record["method"], 0) + 1
        latencies = np.asarray([r["seconds"] for r in requests], dtype=float)
        observed = [r["response_bytes_observed"] for r in requests if r["response_bytes_observed"] is not None]
        return {
            "run_id": self.run_id,
            "started_utc": _utc(self.started_wall),
            "ended_utc": _utc(self.ended_wall) if self.ended_wall is not None else None,
            "request_count": len(requests),
            "requests_by_method": by_method,
            "error_count": sum(1 for r in requests if r["status"] is None or r["status"] >= 400),
            "request_body_bytes": int(sum(r["request_bytes"] for r in requests)),
            "response_bytes_observed": int(sum(observed)),
            "responses_without_length": len(requests) - len(observed),
            "header_latency_p50_s": float(np.percentile(latencies, 50)) if latencies.size else None,
            "header_latency_p95_s": float(np.percentile(latencies, 95)) if latencies.size else None,
            "subprocess_count": len(subprocesses),
            "subprocess_seconds": float(sum(r["seconds"] for r in subprocesses)),
            "subprocess_bytes": int(sum(r["bytes"] or 0 for r in subprocesses)),
        }


def _utc(seconds: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(seconds)) + f".{int(seconds % 1 * 1000):03d}Z"


def active_recorder() -> IORecorder | None:
    """The innermost recorder, or ``None`` when nothing is recording."""
    with _lock:
        return _stack[-1] if _stack else None


def _recording_send(session, request, **kwargs):
    recorder = active_recorder()
    if recorder is None:  # a thread that outlived the context
        return _original_send(session, request, **kwargs)
    request.headers.setdefault(RUN_ID_HEADER, recorder.run_id)
    body = request.body
    request_bytes = len(body) if isinstance(body, (bytes, bytearray, str)) else 0
    started = time.perf_counter()
    status, length = None, None
    try:
        response = _original_send(session, request, **kwargs)
        status = response.status_code
        header = response.headers.get("Content-Length")
        length = int(header) if header is not None and header.isdigit() else None
        return response
    finally:
        recorder._add("request", {
            "method": request.method,
            "path": urlsplit(request.url).path,
            "status": status,
            "request_bytes": request_bytes,
            "response_bytes_observed": length,
            "seconds": time.perf_counter() - started,
            "at": time.time(),
        })


def record_subprocess(command: str, remote_uri: str, seconds: float, size: int | None) -> None:
    """Called by :mod:`vaft.database.transport` after an hsget/hsload; a no-op unless recording."""
    recorder = active_recorder()
    if recorder is not None:
        recorder._add("subprocess", {"command": command, "remote": remote_uri, "seconds": seconds,
                                     "bytes": size, "at": time.time()})


@contextlib.contextmanager
def record_io(run_id: str | None = None) -> Iterator[IORecorder]:
    """Record HSDS I/O inside the block; patches ``requests.Session.send`` only while active."""
    global _original_send
    import requests

    recorder = IORecorder(run_id=run_id or uuid.uuid4().hex[:12])
    with _lock:
        if not _stack:
            _original_send = requests.Session.send
            requests.Session.send = _recording_send
        _stack.append(recorder)
    try:
        yield recorder
    finally:
        recorder.ended_wall = time.time()
        with _lock:
            _stack.remove(recorder)
            if not _stack:
                requests.Session.send = _original_send
                _original_send = None
