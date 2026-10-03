"""Confining caller-supplied file names to one directory the server's user chose.

The MCP tools read local files only from a directory named by an environment
variable of the server process (``VAFT_ATLAS_DIR``, ``VAFT_ARTIFACT_DIR``).  A
caller passes a path relative to it.  :func:`contained_file` refuses absolute,
drive-qualified and UNC paths, ``~``, ``:``, NUL bytes and ``..`` before any
filesystem call, then resolves the path (symlinks included) and refuses it
unless it is a file inside the directory.  Every refusal of one kind of path
gives the same message, so a refusal reveals nothing about the filesystem.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath, PureWindowsPath


class PathRefused(ValueError):
    """A path outside the configured directory, or a directory that is not configured."""


def configured_root(env: str, what: str) -> Path:
    """The directory named by ``env``, resolved; :class:`PathRefused` when unset or not a directory."""
    configured = os.environ.get(env, "").strip()
    if not configured:
        raise PathRefused(f"{env} is not set: point it at the directory holding the {what} before starting the server")
    root = Path(configured).expanduser()
    if not root.is_dir():
        raise PathRefused(f"{env} does not name a directory")
    return root.resolve(strict=True)


def lexical_parts(path: str, refused: PathRefused) -> tuple[str, ...]:
    """The parts of a relative ``path``, checked without touching the filesystem."""
    text = str(path)
    if not text or "\x00" in text or text.startswith(("/", "\\", "~")) or ":" in text:
        raise refused
    windows = PureWindowsPath(text)
    if windows.is_absolute() or windows.drive or windows.root or PurePosixPath(text).is_absolute():
        raise refused
    parts = tuple(part for part in text.replace("\\", "/").split("/") if part not in ("", "."))
    if not parts or any(part == ".." for part in parts):
        raise refused
    return parts


def contained(root: Path, candidate: Path, *, directory: bool = False) -> Path | None:
    """``candidate`` resolved, when it exists inside ``root`` as a file (or a directory)."""
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError, RuntimeError):
        return None
    if directory:
        return resolved if resolved.is_dir() or resolved.is_file() else None
    return resolved if resolved.is_file() else None


def contained_file(root: Path, path: str, refused: PathRefused, *, suffixes=None, directory: bool = False) -> Path:
    """The file (or, with ``directory``, file or directory) ``path`` names inside ``root``."""
    parts = lexical_parts(path, refused)
    if suffixes is not None and PurePosixPath(parts[-1]).suffix.lower() not in suffixes:
        raise refused
    resolved = contained(root, root.joinpath(*parts), directory=directory)
    if resolved is None:
        raise refused
    return resolved
