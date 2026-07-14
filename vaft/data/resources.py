"""Access packaged VAFT sample and fixture data."""

from __future__ import annotations

from importlib import resources
from pathlib import Path


def data_path(name: str = "") -> Path:
    """Return an absolute path inside the packaged ``vaft/data`` directory."""
    root = resources.files("vaft.data")
    target = root.joinpath(name) if name else root
    return Path(str(target))


def sample_geqdsk(name: str = "efit/g039915.00319"):
    """Load one packaged GEQDSK sample as a :class:`vaft.data.eqdsk.GEQDSK`."""
    from .eqdsk import read_geqdsk

    return read_geqdsk(data_path(name))
