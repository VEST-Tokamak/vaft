"""Launchable stand-ins for the external Fortran solvers, per platform.

Every adapter test that asserts "VAFT actually ran the solver" needs a program
the operating system will start. On POSIX that is a ``#!/bin/sh`` script with
the execute bit set. Windows has no shebang: ``CreateProcess`` refuses such a
file with ``WinError 193``, which is why those cases used to be skipped there
wholesale.

A ``.cmd`` batch file is the Windows equivalent -- ``subprocess.run`` starts one
directly, with ``shell=False`` -- and ``.cmd`` rather than ``.bat`` because
``.cmd`` does not reset ``ERRORLEVEL`` on an internal command, so ``exit /b``
reliably carries the code the test asked for.

``.cmd`` is one of :func:`vaft.compat.executable_suffixes`, so a stub written
here is found by exactly the resolution code production uses. The fixture is not
a special case bolted onto the adapters; it travels the same path a real
``dcon.exe`` does.

Both writers return the path they actually created, which is *not* the path they
were given on Windows. Call sites must use the return value when they assert on
recorded commands.
"""

from __future__ import annotations

from pathlib import Path
import stat

from vaft.compat import IS_WINDOWS


def write_launchable_stub(path: str | Path, *, exit_code: int = 0) -> Path:
    """Write a program at ``path`` that exits with ``exit_code``, and return it."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if IS_WINDOWS:
        created = target.with_name(target.name + ".cmd")
        created.write_text(f"@echo off\r\nexit /b {int(exit_code)}\r\n", encoding="ascii")
        return created
    target.write_text(f"#!/bin/sh\nexit {int(exit_code)}\n", encoding="utf-8")
    target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return target


def write_unlaunchable_file(path: str | Path) -> Path:
    """Write a file at ``path`` that this platform will refuse to start.

    On POSIX that means clearing the execute bits. On Windows it means a name
    with no launchable suffix and no ``MZ`` header -- the two things
    :func:`vaft.compat.is_executable` looks at -- which a plain text file at the
    documented, extension-less name already satisfies.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("not a program\n", encoding="utf-8")
    if not IS_WINDOWS:
        target.chmod(0o644)
    return target
