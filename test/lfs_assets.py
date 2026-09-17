"""Packaged assets that live in Git LFS, and what to do when they are pointers.

`vaft/data/efit/*.ddd` -- the EFIT Green tables, ~164 MB -- are LFS objects.
A checkout without LFS leaves a ~130-byte text pointer at each path, and the
readers then fail deep inside a Fortran record parser with "not a sequential
unformatted file with 4-byte record markers": a true statement about the file
on disk, and a completely misleading one about the repository.

CI fetches them (`lfs: true` on the full-suite checkout). A developer who
cloned without LFS gets a skip that names the command, rather than a failure
they have to decompile.
"""

from __future__ import annotations

from pathlib import Path

import pytest

#: The first line of the pointer file the LFS spec defines.
_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def is_lfs_pointer(path: str | Path) -> bool:
    """True when ``path`` holds an LFS pointer rather than the object itself."""
    candidate = Path(path)
    try:
        with candidate.open("rb") as handle:
            return handle.read(len(_POINTER_PREFIX)) == _POINTER_PREFIX
    except OSError:
        return False


def skip_unless_materialized(*paths: str | Path) -> None:
    """Skip the calling test when any of ``paths`` is still an LFS pointer."""
    pointers = [str(Path(path)) for path in paths if is_lfs_pointer(path)]
    if pointers:
        pytest.skip(
            "Git LFS objects are not checked out: "
            + ", ".join(pointers)
            + ". Run `git lfs install && git lfs pull` to fetch them."
        )
