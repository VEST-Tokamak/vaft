"""Private helpers for deterministic external-code executable lookup."""

from __future__ import annotations

import os
from pathlib import Path, PurePath

from ..compat import executable_suffixes, is_executable, resolve_executable


class ExecutableNotLaunchable(RuntimeError):
    """The operating system refused to start an external-code executable.

    Distinct from ``FileNotFoundError`` and ``PermissionError``, which mean the
    file was absent or is not a program at all. This means the file passed every
    check VAFT can make and the process still would not start -- typically a
    POSIX build dropped into a Windows installation, or a native build whose
    runtime libraries cannot be resolved from the launching process.

    A named type rather than a bare ``OSError``: ``FileNotFoundError`` *is* an
    ``OSError``, and the adapters raise it deliberately for configuration
    problems that must keep propagating.
    """


def executable_from_home(
    home: str | os.PathLike[str] | None,
    *,
    home_variable: str,
    relative_path: str | os.PathLike[str],
    code_name: str,
) -> Path | None:
    """Resolve and validate one executable beneath an external-code root."""

    if home is None or not str(home).strip():
        return None

    root = Path(home).expanduser()
    expected = root / relative_path
    # The documented layout is POSIX -- `bin/dcon` -- because that is what the
    # upstream projects and every VAFT surface say. A native Windows build of
    # the same code is `bin\dcon.exe`, so the documented name has to find it.
    executable = resolve_executable(expected)
    if executable is None:
        raise FileNotFoundError(
            f"{code_name} executable is missing for ${home_variable}={root}: "
            f"expected {expected}. Compile or install {code_name} so that "
            "the executable exists at the documented location."
        )
    if not is_executable(executable):
        raise PermissionError(
            f"{code_name} executable is not executable for "
            f"${home_variable}={root}: expected {executable}. Compile or "
            f"install {code_name} correctly and ensure the file is executable."
        )
    return executable


def missing_home_message(
    *,
    home_variable: str,
    relative_path: str | os.PathLike[str],
    code_name: str,
    compatibility_variables: tuple[str, ...] = (),
) -> str:
    """Return an actionable message for an unconfigured external code."""

    # Rendered POSIX-style on every platform. Callers pass a mix of Path
    # constants and brace-glob strings, so without this the same guidance reads
    # "bin/{dcon,...}" for GPEC but "bin\efit" for EFIT on Windows. The message
    # already spells the variable "$EFITHOME" rather than "%EFITHOME%", so one
    # documentation-style separator is the consistent choice.
    shown = (
        relative_path.as_posix()
        if isinstance(relative_path, PurePath)
        else str(relative_path)
    )
    message = (
        f"{code_name} installation is not configured: set ${home_variable} "
        f"to its installation root containing {shown}."
    )
    if compatibility_variables:
        names = ", ".join(f"${name}" for name in compatibility_variables)
        message += f" Existing installations may continue to use {names}."
    # Appended only where it is true, so every POSIX message is unchanged.
    suffixes = executable_suffixes()
    if suffixes:
        spelled = ", ".join(suffixes[:-1]) + f" or {suffixes[-1]}"
        message += f" A native Windows build ending in {spelled} is also accepted."
    return message
