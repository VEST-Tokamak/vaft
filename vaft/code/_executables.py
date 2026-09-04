"""Private helpers for deterministic external-code executable lookup."""

from __future__ import annotations

import os
from pathlib import Path, PurePath


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
    executable = root / relative_path
    if not executable.is_file():
        raise FileNotFoundError(
            f"{code_name} executable is missing for ${home_variable}={root}: "
            f"expected {executable}. Compile or install {code_name} so that "
            "the executable exists at the documented location."
        )
    if not os.access(executable, os.X_OK):
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
    return message
