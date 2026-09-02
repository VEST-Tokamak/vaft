"""Private helpers for deterministic external-code executable lookup."""

from __future__ import annotations

import os
from pathlib import Path


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

    message = (
        f"{code_name} installation is not configured: set ${home_variable} "
        f"to its installation root containing {relative_path}."
    )
    if compatibility_variables:
        names = ", ".join(f"${name}" for name in compatibility_variables)
        message += f" Existing installations may continue to use {names}."
    return message
