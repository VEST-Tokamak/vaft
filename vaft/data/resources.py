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


def sample_camera_visible_frame_paths(shot: int = 39915) -> list[tuple[float, Path]]:
    """Return ``(time_s, path)`` pairs for packaged FAST-camera sample frames.

    Frames are archived PNGs under ``data/legacy/{shot}_{time_ms}_ms.png`` --
    already-arranged, time-labeled camera frames restricted to the valid
    (non-dark) interval (see
    :func:`vaft.machine_mapping.camera_visible.find_valid_frame_interval`,
    which was applied once at packaging time). Like the large legacy
    digitizer/`.mat` samples, these are Git-repository-only and excluded from
    the PyPI distribution -- see :func:`require_repository_sample`.
    """
    import re

    legacy_dir = require_repository_sample(data_path("legacy"))
    pattern = re.compile(rf"^{int(shot)}_(\d+\.?\d*)_ms\.png$")
    frames: list[tuple[float, Path]] = []
    for path in legacy_dir.glob(f"{int(shot)}_*_ms.png"):
        match = pattern.match(path.name)
        if match:
            frames.append((float(match.group(1)) / 1000.0, path))
    if not frames:
        raise FileNotFoundError(
            f"No packaged camera_visible sample frames found for shot {shot} in "
            f"{legacy_dir}. Clone the VAFT GitHub repository to access archived samples."
        )

    return sorted(frames, key=lambda item: item[0])
