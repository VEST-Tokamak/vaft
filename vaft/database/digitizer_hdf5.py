"""Lossless HDF5 container for VEST soft X-ray digitizer records.

``sample_v3.py``/``sample_v4.py`` wrote each SXR acquisition as a CSV: one row
per digitizer channel, every sample printed as Python's shortest round-trip
``repr`` of a float64, rows joined by ``,`` and ended by ``\\r\\n``. That spends
~20 bytes of text on an 8-byte value, and the values are already-processed
floats (every sample in a row is distinct), so there is no integer width to
reclaim the way camera frames are narrowed. Stored as float64 with the HDF5
byte-shuffle filter and gzip, a record shrinks to 27--30 % of the CSV and
loads 20--40x faster, without changing a single bit.

The container is *byte-exact reversible*: :func:`rebuild_csv_bytes` regenerates
the original CSV file, and :func:`pack_digitizer_csv` refuses to write a
container unless that regeneration reproduces the source's sha256. A caller may
therefore delete the CSV once :func:`pack_digitizer_csv` has returned, and a
CSV that does not follow the writer's exact text format is left alone rather
than packed approximately.

Layout, chosen so the same file can later be published unchanged to an HSDS
raw domain and read one channel at a time:

``/data``
    float64, ``(channels, samples)`` -- the CSV's own orientation. One chunk per
    channel (split along samples above :data:`MAX_CHUNK_SAMPLES`), shuffle +
    gzip level :data:`GZIP_LEVEL`.
root attributes
    ``schema``/``schema_version``, ``shot``, ``daq_label``, ``layout``, and the
    source record: ``source_name``, ``source_size``, ``source_sha256``,
    ``line_terminator``, ``float_format``, plus ``created_at`` and
    ``vaft_version``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import os
from pathlib import Path
import re
from typing import Any

import numpy as np

from vaft.compat import IS_WINDOWS

SCHEMA = "vaft.soft_x_rays.digitizer"
SCHEMA_VERSION = 1
SUFFIX = ".h5"
LAYOUT = "channels_as_rows"
FLOAT_FORMAT = "repr"
GZIP_LEVEL = 4
#: Chunks longer than this are split along the sample axis. A 48,828-sample
#: channel (the routine record) is a single 390 kB chunk; the multi-second
#: calibration records run to millions of samples per channel.
MAX_CHUNK_SAMPLES = 1 << 20

_CSV_NAME = re.compile(r"^digitizer_(?P<daq>\d+)_(?P<shot>\d+)\.csv$")


class DigitizerPackError(ValueError):
    """A CSV could not be packed without loss; the CSV was left untouched."""


@dataclass(frozen=True)
class PackResult:
    source: Path
    target: Path
    shape: tuple[int, int]
    source_size: int
    target_size: int
    source_sha256: str

    @property
    def ratio(self) -> float:
        return self.target_size / self.source_size if self.source_size else float("nan")


def container_path(csv_path: str | Path) -> Path:
    """Where the container for ``digitizer_{daq}_{shot}.csv`` lives."""
    path = Path(csv_path)
    return path.with_suffix(SUFFIX)


def fsync_path(path: str | Path) -> None:
    """Force a file (or directory) to stable storage before anything depends on it.

    ``verify_container`` re-reads through the page cache, so it proves what the
    kernel holds, not what the disk holds; a CSV must not be unlinked until the
    container that replaces it is durable. macOS needs ``F_FULLFSYNC`` for that.

    Windows can neither open a directory as a descriptor nor ``fsync`` a
    read-only one: files are opened read-write there, and directories are
    skipped because ``os.replace`` already made the name visible.
    """
    if IS_WINDOWS:
        if Path(path).is_dir():
            return
        fd = os.open(path, os.O_RDWR)
    else:
        fd = os.open(path, os.O_RDONLY)
    try:
        try:
            import fcntl

            fcntl.fcntl(fd, getattr(fcntl, "F_FULLFSYNC"))
        except (ImportError, AttributeError, OSError):
            os.fsync(fd)
    finally:
        os.close(fd)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _detect_terminator(payload: bytes) -> str:
    if payload.endswith(b"\r\n"):
        return "\r\n"
    if payload.endswith(b"\n"):
        return "\n"
    raise DigitizerPackError("CSV does not end with a line terminator")


def _parse(payload: bytes) -> np.ndarray:
    import pandas as pd

    try:
        # round_trip is the only pandas parser guaranteed to invert repr(); the
        # default fast parser can land one ulp away, which the byte check below
        # would (correctly) reject as a lossy pack.
        frame = pd.read_csv(
            io.BytesIO(payload), header=None, delimiter=",", float_precision="round_trip"
        )
    except Exception as exc:  # pandas raises several parser error types
        raise DigitizerPackError(f"CSV does not parse as a rectangular table: {exc}") from exc
    values = frame.to_numpy()
    if values.dtype != np.float64:
        raise DigitizerPackError(f"CSV holds non-float cells (dtype {values.dtype})")
    if values.ndim != 2 or not values.size:
        raise DigitizerPackError("CSV is empty")
    return values


def rebuild_csv_bytes(data: np.ndarray, line_terminator: str = "\r\n") -> bytes:
    """Regenerate the CSV text ``sample_v3.py`` wrote for ``data``."""
    rows = np.asarray(data, dtype=np.float64).tolist()
    return (
        line_terminator.join(",".join(map(repr, row)) for row in rows) + line_terminator
    ).encode("ascii")


def _chunks(shape: tuple[int, int]) -> tuple[int, int]:
    return (1, max(1, min(shape[1], MAX_CHUNK_SAMPLES)))


def pack_digitizer_csv(
    csv_path: str | Path,
    target: str | Path | None = None,
    *,
    shot: int | None = None,
    daq_label: str | None = None,
    overwrite: bool = False,
) -> PackResult:
    """Write the lossless container for one digitizer CSV and verify it.

    The CSV is never modified. The container is assembled under a temporary
    name, re-read, regenerated to CSV text and compared against the source's
    sha256 before it is renamed into place, so a container at ``target`` is
    always a verified one.
    """
    import h5py

    from vaft.version import __version__

    source = Path(csv_path)
    target_path = Path(target) if target is not None else container_path(source)
    if target_path.exists() and not overwrite:
        raise FileExistsError(target_path)
    match = _CSV_NAME.match(source.name)
    if shot is None or daq_label is None:
        if match is None:
            raise DigitizerPackError(
                f"{source.name} is not digitizer_{{daq}}_{{shot}}.csv; pass shot= and daq_label="
            )
        shot = int(match["shot"]) if shot is None else shot
        daq_label = match["daq"] if daq_label is None else daq_label

    payload = source.read_bytes()
    digest = _sha256_bytes(payload)
    terminator = _detect_terminator(payload)
    data = _parse(payload)
    if _sha256_bytes(rebuild_csv_bytes(data, terminator)) != digest:
        raise DigitizerPackError(
            "CSV text is not the sample_v3 repr format, so a container could not "
            "regenerate it byte for byte"
        )

    partial = target_path.with_name(f".{target_path.name}.partial")
    partial.unlink(missing_ok=True)
    try:
        with h5py.File(partial, "w") as handle:
            handle.create_dataset(
                "data",
                data=data,
                dtype="<f8",
                chunks=_chunks(data.shape),
                shuffle=True,
                compression="gzip",
                compression_opts=GZIP_LEVEL,
            )
            handle.attrs.update(
                {
                    "schema": SCHEMA,
                    "schema_version": SCHEMA_VERSION,
                    "shot": int(shot),
                    "daq_label": str(daq_label),
                    "layout": LAYOUT,
                    "source_name": source.name,
                    "source_size": len(payload),
                    "source_sha256": digest,
                    "line_terminator": terminator,
                    "float_format": FLOAT_FORMAT,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "vaft_version": __version__,
                }
            )
        verify_container(partial, expected_sha256=digest)
        fsync_path(partial)
        os.replace(partial, target_path)
        fsync_path(target_path.parent)
    finally:
        partial.unlink(missing_ok=True)
    return PackResult(
        source=source,
        target=target_path,
        shape=(int(data.shape[0]), int(data.shape[1])),
        source_size=len(payload),
        target_size=target_path.stat().st_size,
        source_sha256=digest,
    )


def read_attributes(path: str | Path) -> dict[str, Any]:
    import h5py

    with h5py.File(path, "r") as handle:
        return {
            key: (value.decode() if isinstance(value, bytes) else value)
            for key, value in handle.attrs.items()
        }


def _check_schema(attrs: dict[str, Any], path: Path) -> None:
    if attrs.get("schema") != SCHEMA:
        raise ValueError(f"{path} is not a {SCHEMA} container")
    if int(attrs.get("schema_version", -1)) > SCHEMA_VERSION:
        raise ValueError(
            f"{path} uses schema_version {attrs['schema_version']}; this VAFT reads "
            f"up to {SCHEMA_VERSION}"
        )


def load_digitizer_hdf5(path: str | Path, *, nrows: int | None = None) -> np.ndarray:
    """Return the ``(channels, samples)`` array a container holds.

    ``nrows`` mirrors ``pandas.read_csv(nrows=)`` on the CSV: the first
    ``nrows`` channel rows.
    """
    import h5py

    path = Path(path)
    with h5py.File(path, "r") as handle:
        _check_schema(dict(handle.attrs), path)
        dataset = handle["data"]
        return np.asarray(dataset[:nrows] if nrows is not None else dataset[()], dtype=np.float64)


def verify_container(path: str | Path, *, expected_sha256: str | None = None) -> str:
    """Regenerate the source CSV from a container and check its sha256.

    Returns the digest. Raises :class:`DigitizerPackError` on any mismatch,
    including against ``expected_sha256`` when given (e.g. a CSV still on disk).
    """
    path = Path(path)
    attrs = read_attributes(path)
    _check_schema(attrs, path)
    data = load_digitizer_hdf5(path)
    rebuilt = rebuild_csv_bytes(data, str(attrs["line_terminator"]))
    digest = _sha256_bytes(rebuilt)
    if len(rebuilt) != int(attrs["source_size"]) or digest != attrs["source_sha256"]:
        raise DigitizerPackError(f"{path} does not regenerate its recorded source CSV")
    if expected_sha256 is not None and digest != expected_sha256:
        raise DigitizerPackError(f"{path} does not regenerate the CSV it is checked against")
    return digest


def restore_csv(path: str | Path, target: str | Path | None = None) -> Path:
    """Write the original CSV back out of a verified container."""
    path = Path(path)
    attrs = read_attributes(path)
    verify_container(path)
    out = Path(target) if target is not None else path.with_name(str(attrs["source_name"]))
    out.write_bytes(rebuild_csv_bytes(load_digitizer_hdf5(path), str(attrs["line_terminator"])))
    return out


__all__ = [
    "DigitizerPackError",
    "GZIP_LEVEL",
    "PackResult",
    "SCHEMA",
    "SCHEMA_VERSION",
    "SUFFIX",
    "container_path",
    "fsync_path",
    "load_digitizer_hdf5",
    "pack_digitizer_csv",
    "read_attributes",
    "rebuild_csv_bytes",
    "restore_csv",
    "verify_container",
]
