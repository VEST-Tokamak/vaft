"""Software identity and content hashing for ML artifacts (#669 section 7)."""

from __future__ import annotations

import hashlib
import platform
import subprocess
from pathlib import Path

#: The checkout this package was imported from, when it is one.
_PACKAGE_ROOT = Path(__file__).resolve().parents[3]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            # --no-optional-locks: `status` otherwise refreshes the index under
            # index.lock, and a status killed by the timeout below would leave
            # that lock behind and block the user's next commit.
            ["git", "--no-optional-locks", *args],
            cwd=_PACKAGE_ROOT,
            text=True,
            capture_output=True,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = completed.stdout.strip()
    return output if completed.returncode == 0 else None


def vaft_revision() -> tuple[str | None, bool | None]:
    """The commit and dirty flag of the VAFT checkout, or ``(None, None)``.

    Only when ``vaft`` is imported from a git checkout of VAFT itself: an
    installed wheel inside some *other* repository (a venv in a project tree)
    would otherwise report that repository's HEAD as VAFT's revision.
    """
    toplevel = _git("rev-parse", "--show-toplevel")
    if toplevel is None or Path(toplevel).resolve() != _PACKAGE_ROOT:
        return None, None
    if not (_PACKAGE_ROOT / "vaft" / "version.py").is_file():
        return None, None
    revision = _git("rev-parse", "HEAD")
    if not revision:
        return None, None
    status = _git("status", "--porcelain", "--untracked-files=normal", "--", "vaft")
    return revision, (None if status is None else bool(status))


def software_provenance(backend_versions: dict[str, str] | None = None) -> dict:
    """VAFT version, revision and the runtimes a training run depended on."""
    import numpy

    from vaft.version import __version__

    revision, dirty = vaft_revision()
    runtimes = {"python": platform.python_version(), "numpy": numpy.__version__}
    runtimes.update(backend_versions or {})
    return {
        "vaft_version": __version__,
        "vaft_revision": revision,
        "vaft_revision_dirty": dirty,
        "runtimes": runtimes,
    }
