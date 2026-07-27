"""Access packaged VAFT sample and fixture data."""

from __future__ import annotations

from importlib import resources
from pathlib import Path


def data_path(name: str = "") -> Path:
    """Return an absolute path inside the packaged ``vaft/data`` directory."""
    root = resources.files("vaft.data")
    target = root.joinpath(name) if name else root
    return Path(str(target))


def require_repository_sample(path: Path) -> Path:
    """Return an archived sample path or explain how to obtain it.

    Large reference samples are deliberately kept in the Git repository but
    excluded from PyPI distributions.
    """
    if not path.exists():
        raise FileNotFoundError(
            "This sample dataset is not included in the PyPI distribution. "
            "Clone the VAFT GitHub repository to access the archived samples."
        )
    return path


def sample_geqdsk(name: str = "efit/g039915.00319"):
    """Load one packaged GEQDSK sample as a :class:`vaft.data.eqdsk.GEQDSK`."""
    from .eqdsk import read_geqdsk

    return read_geqdsk(require_repository_sample(data_path(name)))
