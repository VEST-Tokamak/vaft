"""Byte-exact derived HDF5 images for eager HSDS materialization."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import omas

try:
    import h5pyd
except ImportError:  # pragma: no cover - guarded by database public APIs
    h5pyd = None

from .utils import _require_h5pyd


H5IMAGE_SCHEMA_VERSION = 1
H5IMAGE_CHUNK_BYTES = 4 * 1024 * 1024


class H5ImageUnavailableError(RuntimeError):
    """A requested derived HDF5 image is missing, stale, or corrupt."""


def derived_filename(filename: str) -> str:
    if not filename.endswith(".h5") or Path(filename).name != filename:
        raise ValueError(f"Expected a simple HDF5 filename, got {filename!r}")
    return filename[:-3] + ".h5image.h5"


def is_derived_filename(filename: str) -> bool:
    return filename.endswith(".h5image.h5")


def _manifest_array(manifest: dict[str, Any]) -> np.ndarray:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return np.frombuffer(payload, dtype=np.uint8)


def read_manifest(domain) -> dict[str, Any] | None:
    try:
        payload = np.asarray(domain["manifest"][:], dtype=np.uint8).tobytes()
        value = json.loads(payload.decode("utf-8"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def canonical_info(directory: str, shot: int, filename: str) -> dict[str, str | None]:
    _require_h5pyd()
    uri = f"hdf5://{directory}/{shot}/{filename}"
    with h5pyd.File(uri, "r") as domain:
        modified = getattr(domain, "modified", None)
        backend = domain.attrs.get("HDF5_BACKEND_VERSION")
    if isinstance(backend, bytes):
        backend = backend.decode("utf-8", errors="replace")
    return {
        "remote_uri": uri,
        "last_modified": None if modified is None else str(modified),
        "backend_version": None if backend is None else str(backend),
    }


def publish_image(
    local_path: Path,
    directory: str,
    shot: int,
    *,
    imas_version: str | None,
    source_info: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    """Publish one local canonical HDF5 file as a compressed opaque image."""
    _require_h5pyd()
    from ..version import __version__

    path = Path(local_path)
    filename = path.name
    info = source_info or canonical_info(directory, shot, filename)
    if info.get("last_modified") is None:
        raise H5ImageUnavailableError(
            f"Canonical domain {info.get('remote_uri')} has no revision metadata"
        )
    size = path.stat().st_size
    checksum = hashlib.sha256()
    uri = f"hdf5://{directory}/{shot}/{derived_filename(filename)}"
    with h5pyd.File(uri, "w") as domain:
        chunk = max(1, min(size, H5IMAGE_CHUNK_BYTES))
        image = domain.create_dataset(
            "h5image",
            shape=(size,),
            dtype=np.uint8,
            chunks=(chunk,),
            compression="gzip",
            compression_opts=1,
        )
        offset = 0
        with path.open("rb") as source:
            while block := source.read(H5IMAGE_CHUNK_BYTES):
                checksum.update(block)
                image[offset : offset + len(block)] = np.frombuffer(
                    block, dtype=np.uint8
                )
                offset += len(block)
        manifest = {
            "schema_version": H5IMAGE_SCHEMA_VERSION,
            "kind": "imas-hdf5-image",
            "source_uri": info["remote_uri"],
            "source_last_modified": info["last_modified"],
            "source_backend_version": info.get("backend_version"),
            "source_size": size,
            "source_sha256": checksum.hexdigest(),
            "imas_version": imas_version,
            "omas_version": getattr(omas, "__version__", None),
            "vaft_version": __version__,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        manifest_payload = _manifest_array(manifest)
        stored = domain.create_dataset(
            "manifest", shape=manifest_payload.shape, dtype=np.uint8
        )
        stored[:] = manifest_payload
    return {"uri": uri, "manifest": manifest}


def materialize_image(
    directory: str,
    shot: int,
    filename: str,
    target: Path,
    *,
    source_info: dict[str, str | None],
) -> dict[str, Any]:
    """Restore and validate one derived image into an ordinary HDF5 file."""
    _require_h5pyd()
    uri = f"hdf5://{directory}/{shot}/{derived_filename(filename)}"
    target = Path(target)
    checksum = hashlib.sha256()
    try:
        with h5pyd.File(uri, "r") as domain:
            manifest = read_manifest(domain)
            expected = {
                "kind": "imas-hdf5-image",
                "source_uri": source_info.get("remote_uri"),
                "source_last_modified": source_info.get("last_modified"),
                "source_backend_version": source_info.get("backend_version"),
            }
            if manifest is None or any(
                manifest.get(key) != value for key, value in expected.items()
            ):
                raise H5ImageUnavailableError(
                    f"Derived image {uri} is missing or stale"
                )
            image = domain["h5image"]
            if tuple(image.shape) != (int(manifest.get("source_size", -1)),):
                raise H5ImageUnavailableError(
                    f"Derived image {uri} has an invalid payload size"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as output:
                for offset in range(0, image.shape[0], H5IMAGE_CHUNK_BYTES):
                    block = np.asarray(
                        image[
                            offset : min(offset + H5IMAGE_CHUNK_BYTES, image.shape[0])
                        ],
                        dtype=np.uint8,
                    ).tobytes()
                    checksum.update(block)
                    output.write(block)
            if checksum.hexdigest() != manifest.get("source_sha256"):
                raise H5ImageUnavailableError(
                    f"Derived image {uri} failed checksum validation"
                )
        with h5py.File(target, "r"):
            pass
        return {
            "transport": "h5image",
            "remote_uri": uri,
            "logical_bytes": target.stat().st_size,
        }
    except H5ImageUnavailableError:
        target.unlink(missing_ok=True)
        raise
    except Exception as exc:
        target.unlink(missing_ok=True)
        raise H5ImageUnavailableError(
            f"Could not materialize derived image {uri}: {exc}"
        ) from exc
