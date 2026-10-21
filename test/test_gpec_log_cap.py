"""The PENTRC log cap: head and tail kept, a stall stopped (#1929)."""

from __future__ import annotations

import io
import sys

import pytest

from vaft.code.gpec import PENTRCLogLimits
from vaft.code.gpec._log_cap import ELISION_MARKER, STALL_EXIT_CODE, STALL_MARKER, run


def _python(source: str) -> list[str]:
    return [sys.executable, "-c", source]


def test_a_long_log_keeps_its_head_and_tail_within_the_cap():
    sink = io.BytesIO()
    code = run(
        _python("for i in range(20000): print(f'line {i:05d} ' + 'x' * 40)"),
        max_bytes=10_000, stall_lines=None, stall_pattern=None, sink=sink,
    )
    text = sink.getvalue().decode()
    assert code == 0
    assert text.startswith("line 00000 ")
    assert text.rstrip().endswith("line 19999 " + "x" * 40)
    assert ELISION_MARKER in text
    assert len(sink.getvalue()) < 10_000 + 200


def test_a_short_log_is_written_unchanged_and_the_status_passes_through():
    sink = io.BytesIO()
    code = run(
        _python("import sys; print('hello'); sys.exit(3)"),
        max_bytes=10_000, stall_lines=10, stall_pattern="corrector", sink=sink,
    )
    assert code == 3
    assert sink.getvalue() == b"hello\n"


def test_a_stalled_program_is_stopped_and_says_so():
    sink = io.BytesIO()
    code = run(
        # Would never end on its own.
        _python("print('start')\nwhile True: print(' LSODE--  Corrector convergence failed repeatedly')"),
        max_bytes=4_000, stall_lines=500, stall_pattern="corrector convergence failed",
        ignore_case=True, sink=sink,
    )
    text = sink.getvalue().decode()
    assert code == STALL_EXIT_CODE
    assert text.startswith("start")
    assert text.splitlines()[-1].startswith(STALL_MARKER)
    assert "after 500 lines" in text
    assert len(sink.getvalue()) < 4_000 + 400


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_the_limits_refuse_nonsense(bad):
    with pytest.raises(ValueError):
        PENTRCLogLimits(max_bytes=bad)
    with pytest.raises(ValueError):
        PENTRCLogLimits(stall_lines=bad)


def test_both_limits_off_is_inactive():
    assert not PENTRCLogLimits(max_bytes=None, stall_lines=None).active
    assert PENTRCLogLimits().active


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_a_terminated_cap_still_writes_the_held_tail(tmp_path):
    """The caller's timeout terminates the process tree; the tail is the evidence."""
    import signal
    import subprocess
    import time
    from pathlib import Path

    import vaft.code.gpec._log_cap as cap

    log = tmp_path / "pentrc.log"
    with log.open("wb") as handle:
        process = subprocess.Popen(
            [sys.executable, "-I", str(Path(cap.__file__)), "--max-bytes", "2000", "--",
             sys.executable, "-c",
             "import itertools\nfor i in itertools.count(): print(f'line {i}', flush=True)"],
            stdout=handle,
        )
        deadline = time.monotonic() + 20
        while log.stat().st_size < 900 and time.monotonic() < deadline:
            time.sleep(0.05)
        time.sleep(0.3)
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=20) == 128 + signal.SIGTERM
    text = log.read_text(encoding="utf-8")
    assert text.startswith("line 0\n")
    assert ELISION_MARKER in text
    assert text.rstrip().splitlines()[-1].startswith("line ")


def test_the_head_does_not_take_a_line_after_the_tail_started():
    """Order is kept at the boundary: a short line after a long one stays behind it."""
    sink = io.BytesIO()
    run(
        _python("print('A' * 39); print('B' * 19); print('C'); [print('D' * 30) for _ in range(5)]"),
        max_bytes=100, stall_lines=None, stall_pattern=None, sink=sink,
    )
    lines = sink.getvalue().decode().splitlines()
    assert lines[0] == "A" * 39
    assert lines[1].startswith(ELISION_MARKER)
    assert "C" not in lines[:1]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_a_program_killed_by_a_signal_reports_128_plus_the_signal():
    sink = io.BytesIO()
    code = run(
        _python("import os, signal; os.kill(os.getpid(), signal.SIGSEGV)"),
        max_bytes=1000, stall_lines=None, stall_pattern=None, sink=sink,
    )
    assert code == 128 + 11
