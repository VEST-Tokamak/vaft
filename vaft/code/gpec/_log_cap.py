"""Run a program with its output capped: head and tail kept, a stall stopped.

``python -I .../vaft/code/gpec/_log_cap.py --max-bytes N --stall-lines K
--stall-pattern REGEX [--ignore-case] -- program [args...]`` runs *program* with its stdout
and stderr merged and writes them to this process's stdout -- which the
execution backend points at the run's log -- with two limits:

* **Size.**  The first ``N/2`` bytes are written as they arrive; after that the
  last ``N/2`` bytes are held in memory and written when the program ends,
  behind one line saying how much was left out.  A PENTRC whose energy LSODE
  stalls writes the same line millions of times -- 1.8 GB before a one-hour
  timeout in the PENTRC NTV coil-response scan of #1887 -- and none of the
  middle of that says anything the head and the tail do not.
* **Stall.**  When ``K`` lines have matched ``REGEX`` the program is stopped
  and this process exits :data:`STALL_EXIT_CODE` after a final
  :data:`STALL_MARKER` line, so the caller can say *stalled* rather than
  *timed out* -- and say it within seconds of the stall rather than at the
  timeout.

Terminated itself (the caller's timeout or memory stop ends the process
tree), it still writes the held tail before it exits -- on POSIX, where that
is a signal it can catch.  The held tail is up to ``N/2`` bytes of memory,
which a memory limit on the run counts with PENTRC's own.
"""

from __future__ import annotations

import argparse
from collections import deque
import re
import signal
import subprocess
import sys
from typing import BinaryIO, Sequence

#: Exit status when the stall limit stopped the program.  Outside what a
#: Fortran ``STOP`` or a signal gives (0-255 with 128+ for signals).
STALL_EXIT_CODE = 86

#: The last line of a log this filter stopped; the caller looks for it.
STALL_MARKER = "VAFT-LOG-CAP: stalled"

#: The line standing in for the elided middle of a capped log.
ELISION_MARKER = "VAFT-LOG-CAP: elided"


class _CappedWriter:
    """Head written through, tail held, middle counted."""

    def __init__(self, sink: BinaryIO, max_bytes: int | None) -> None:
        self.sink = sink
        self.head_budget = None if max_bytes is None else max_bytes // 2
        self.tail_budget = None if max_bytes is None else max_bytes - max_bytes // 2
        self.head_written = 0
        #: Once a line has gone to the tail, every later one does: a shorter
        #: line that would still fit the head must not jump the queue.
        self.head_full = False
        self.tail: deque[bytes] = deque()
        self.tail_bytes = 0
        self.elided_bytes = 0
        self.elided_lines = 0
        self.closed = False

    def write(self, line: bytes) -> None:
        if self.head_budget is None:
            self.sink.write(line)
            self.sink.flush()
            return
        if not self.head_full and self.head_written + len(line) <= self.head_budget:
            self.sink.write(line)
            self.sink.flush()
            self.head_written += len(line)
            return
        self.head_full = True
        if len(line) > self.tail_budget:
            # One line longer than the whole tail: keep its end, which is
            # where a program says how it stopped.
            self.elided_bytes += len(line) - self.tail_budget
            line = line[-self.tail_budget:]
        self.tail.append(line)
        self.tail_bytes += len(line)
        while self.tail and self.tail_bytes > self.tail_budget:
            dropped = self.tail.popleft()
            self.tail_bytes -= len(dropped)
            self.elided_bytes += len(dropped)
            self.elided_lines += 1

    def close(self, trailer: str | None = None) -> None:
        """Write the held tail once, then *trailer*; idempotent."""
        if not self.closed:
            self.closed = True
            if self.elided_bytes:
                self.sink.write(
                    f"{ELISION_MARKER} {self.elided_bytes} bytes in {self.elided_lines} lines "
                    f"(head and tail of the log kept)\n".encode()
                )
            while self.tail:
                self.sink.write(self.tail.popleft())
        if trailer:
            self.sink.write(trailer.encode())
        self.sink.flush()


def _exit_status(returncode: int) -> int:
    """A shell's status for a child that died of a signal: ``128 + N``, not ``256 - N``."""
    return 128 - returncode if returncode < 0 else returncode


def run(
    command: Sequence[str],
    *,
    max_bytes: int | None,
    stall_lines: int | None,
    stall_pattern: str | None,
    ignore_case: bool = False,
    sink: BinaryIO | None = None,
) -> int:
    """Run *command* through the cap; return its exit status or :data:`STALL_EXIT_CODE`."""
    sink = sink if sink is not None else sys.stdout.buffer
    flags = re.IGNORECASE if ignore_case else 0
    pattern = re.compile(stall_pattern.encode(), flags) if stall_pattern and stall_lines else None
    writer = _CappedWriter(sink, max_bytes)
    child = subprocess.Popen(list(command), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def _stop(signum, _frame):
        # The caller's timeout or memory stop: end the program, keep the tail.
        if child.poll() is None:
            child.kill()
        writer.close()
        sys.exit(128 + signum)

    # POSIX only in practice: on Windows the backend ends the process tree
    # through a job object, which no handler sees, so only the head survives.
    for name in ("SIGTERM", "SIGINT", "SIGHUP"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), _stop)

    matched = 0
    last_match = b""
    stalled = False
    assert child.stdout is not None
    for line in iter(child.stdout.readline, b""):
        writer.write(line)
        if pattern is not None and pattern.search(line):
            matched += 1
            last_match = line
            if matched >= stall_lines:
                stalled = True
                child.kill()
                break
    child.stdout.close()
    returncode = child.wait()
    if stalled:
        writer.close(
            f"{STALL_MARKER} after {matched} lines matching {stall_pattern!r}; "
            f"last: {last_match.decode(errors='replace').strip()}\n"
        )
        return STALL_EXIT_CODE
    writer.close()
    return _exit_status(returncode)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-bytes", type=int, default=None)
    parser.add_argument("--stall-lines", type=int, default=None)
    parser.add_argument("--stall-pattern", default=None)
    parser.add_argument("--ignore-case", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("no program to run")
    return run(
        command,
        max_bytes=args.max_bytes,
        stall_lines=args.stall_lines,
        stall_pattern=args.stall_pattern,
        ignore_case=args.ignore_case,
    )


if __name__ == "__main__":
    raise SystemExit(main())
