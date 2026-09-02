"""Access packaged VAFT sample and fixture data."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import yaml


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


_SAMPLE_REPRESENTATIONS = frozenset({"omas", "imas"})


def sample_manifest(shot: int) -> dict:
    """Return the validated manifest for one registered reference sample."""
    try:
        shot_number = int(shot)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Sample shot must be an integer, got {shot!r}") from exc

    manifest_path = data_path(f"samples/{shot_number}/manifest.yaml")
    if not manifest_path.is_file():
        available = ", ".join(str(value) for value in available_samples()) or "none"
        raise ValueError(
            f"Unknown VAFT sample shot {shot_number}; available shots: {available}"
        )
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle) or {}
    if int(manifest.get("schema_version", 0)) != 1:
        raise ValueError(f"Invalid sample manifest schema: {manifest_path}")
    if int(manifest.get("shot", -1)) != shot_number:
        raise ValueError(
            f"Sample manifest shot does not match its directory: {manifest_path}"
        )
    representations = manifest.get("representations")
    if not isinstance(representations, dict) or not representations:
        raise ValueError(f"Sample manifest has no representations: {manifest_path}")
    for name, record in representations.items():
        if name not in _SAMPLE_REPRESENTATIONS or not isinstance(record, dict):
            raise ValueError(f"Invalid sample representation {name!r}: {manifest_path}")
        if not record.get("path") or not record.get("sha256"):
            raise ValueError(
                f"Sample representation {name!r} requires path and sha256: {manifest_path}"
            )
        adapters = record.get("compatible_adapters", [name])
        if not isinstance(adapters, list) or not adapters:
            raise ValueError(
                f"Sample representation {name!r} requires compatible_adapters: "
                f"{manifest_path}"
            )
        unsupported = sorted(set(adapters) - _SAMPLE_REPRESENTATIONS)
        if unsupported:
            raise ValueError(
                f"Sample representation {name!r} declares unsupported adapters "
                f"{unsupported}: {manifest_path}"
            )
    return manifest


def available_samples() -> tuple[int, ...]:
    """Return registered sample shot numbers in ascending order."""
    root = data_path("samples")
    if not root.is_dir():
        return ()
    shots = []
    for manifest in root.glob("*/manifest.yaml"):
        try:
            shots.append(int(manifest.parent.name))
        except ValueError:
            continue
    return tuple(sorted(shots))


def sample(shot: int, representation: str = "omas") -> Path:
    """Return an artifact path compatible with the requested data adapter.

    The returned path can be passed to either :func:`vaft.omas.load` or
    :func:`vaft.imas.load`. An exact stored representation is preferred; when
    it is absent, a manifest-declared compatible representation is returned.
    """
    representation_name = str(representation).lower()
    if representation_name not in _SAMPLE_REPRESENTATIONS:
        choices = ", ".join(sorted(_SAMPLE_REPRESENTATIONS))
        raise ValueError(
            f"Unsupported sample representation {representation!r}; expected one of: {choices}"
        )
    manifest = sample_manifest(shot)
    representations = manifest["representations"]
    storage_name = (
        representation_name if representation_name in representations else None
    )
    if storage_name is None:
        storage_name = next(
            (
                name
                for name, record in sorted(representations.items())
                if representation_name in record.get("compatible_adapters", [name])
            ),
            None,
        )
    if storage_name is None:
        choices = ", ".join(
            sorted(
                {
                    adapter
                    for name, record in representations.items()
                    for adapter in record.get("compatible_adapters", [name])
                }
            )
        )
        raise ValueError(
            f"Sample shot {int(shot)} is not compatible with the "
            f"{representation_name!r} adapter; compatible adapters: {choices}"
        )
    record = representations[storage_name]
    path = data_path(f"samples/{int(shot)}/{record['path']}")
    if not path.is_file():
        if record.get("package") == "repository-only":
            raise FileNotFoundError(
                f"VAFT sample shot {int(shot)} is a repository-only {storage_name} "
                "artifact. Clone the VAFT GitHub repository to access it."
            )
        raise FileNotFoundError(f"Registered VAFT sample artifact is missing: {path}")
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
