"""Where the MCP adapter gets data from: the private source seam (#1423, #188 Phase 2).

The MCP protocol names a dataset semantically: a shot number, or a local
artifact by its path relative to a directory the server's user chose.  How
that dataset is opened is decided here and nowhere else, so a later
DD/DDView-backed loader (#1127, #1132, #1135) replaces this module without
changing any tool schema.  Three kinds of dataset resolve:

* a packaged reference shot (:func:`vaft.omas.sample.sample_ods`), offline;
* a local artifact under ``VAFT_ARTIFACT_DIR`` -- a g-file, OMAS JSON, IMAS
  netCDF or IMAS HDF5 -- through :func:`vaft.omas.load`;
* a database shot through :func:`vaft.database.load`, only when the server was
  started with ``VAFT_MCP_DATABASE=1``.  Credentials come from the user's own
  HSDS configuration; nothing here reads, logs or returns them, and a failure
  is reported by its exception class only.

Every load returns a fresh object: VAFT's updaters and extraction may
materialise paths on the loaded data, and a cached object would carry that
into the next answer.  Nothing here writes to the data's home: database shots
are read with the download cache off, and the only files created are the
temporary copies VAFT's own loaders stage (a decompressed ``.json.gz``, a
partial IMAS entry), which they remove.  Artifacts are size-capped so that
staging stays small.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Dataset", "known_shots", "load_dataset", "load_reference_shot"]

ARTIFACT_ENV = "VAFT_ARTIFACT_DIR"
#: Largest artifact (a file, or all files of a directory) and largest compressed one.
MAX_ARTIFACT_BYTES = 512 * 1024 * 1024
MAX_COMPRESSED_BYTES = 32 * 1024 * 1024
DATABASE_ENV = "VAFT_MCP_DATABASE"


@dataclass
class Dataset:
    """A loaded dataset and the provenance a tool reports with it."""

    data: Any
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def shot(self) -> int:
        return int(self.provenance.get("shot") or 0)


def known_shots() -> tuple[int, ...]:
    """The packaged reference shots this installation declares."""
    from vaft.data import available_samples

    return tuple(int(shot) for shot in available_samples())


def database_enabled() -> bool:
    return os.environ.get(DATABASE_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def load_reference_shot(shot: int) -> Any:
    """Open packaged reference shot ``shot`` for :func:`vaft.plot.extract`."""
    shot = int(shot)
    shots = known_shots()
    if shot not in shots:
        raise ValueError(f"no packaged reference shot {shot}; available: {list(shots)}")
    from vaft.omas.sample import sample_ods

    try:
        return sample_ods(shot)
    except FileNotFoundError as error:
        raise ValueError(
            f"reference shot {shot} is declared but its data is not installed here "
            f"(repository-only sample): {type(error).__name__}"
        ) from None


def _data_dictionary_version(ods) -> str | None:
    from vaft.ods_access import path_value

    for path in ("dataset_description.ids_properties.version_put.data_dictionary",
                 "equilibrium.ids_properties.version_put.data_dictionary"):
        value = path_value(ods, path, None)
        if isinstance(value, str) and value:
            return value
    return None


def _vaft_version() -> str:
    from vaft.version import __version__

    return str(__version__)


def _check_artifact(root, resolved, relative: str) -> None:
    """Refuse a directory artifact with a member outside ``root`` (a symlink out), and oversized input."""
    from ._paths import contained

    if resolved.is_dir():
        members = [m for m in resolved.rglob("*") if not m.is_dir()]
        if len(members) > 10_000:
            raise ValueError(f"artifact {relative!r} holds {len(members)} files; the limit is 10000")
        if any(m.is_symlink() or contained(root, m) is None for m in members):
            raise ValueError(f"artifact {relative!r} links outside {ARTIFACT_ENV}; copy the files in instead")
        size = sum(m.stat().st_size for m in members)
    else:
        size = resolved.stat().st_size
        if resolved.suffix.lower() == ".gz" and size > MAX_COMPRESSED_BYTES:
            raise ValueError(f"compressed artifact {relative!r} is {size} bytes; the limit is {MAX_COMPRESSED_BYTES}")
    if size > MAX_ARTIFACT_BYTES:
        raise ValueError(f"artifact {relative!r} is {size} bytes; the limit is {MAX_ARTIFACT_BYTES}")


def _artifact(path: str) -> Dataset:
    from ._paths import PathRefused, configured_root, contained_file

    refused = PathRefused(
        f"no readable artifact {str(path)[:200]!r}: pass a path relative to {ARTIFACT_ENV} "
        f"(a g-file, OMAS JSON, IMAS netCDF or IMAS HDF5 file or directory)"
    )
    root = configured_root(ARTIFACT_ENV, "local artifacts (g-files, ODS JSON, IMAS netCDF/HDF5)")
    resolved = contained_file(root, path, refused, directory=True)
    relative = resolved.relative_to(root).as_posix()
    _check_artifact(root, resolved, relative)
    from vaft.omas import load

    try:
        ods = load(resolved)
    except Exception as error:  # noqa: BLE001 - loader messages carry absolute paths
        raise ValueError(f"artifact {relative!r} could not be read: {type(error).__name__}") from None
    from vaft.ods_access import path_value

    shot = path_value(ods, "dataset_description.data_entry.pulse", None)
    return Dataset(ods, {
        "kind": "artifact",
        "artifact": relative,
        "shot": int(shot) if isinstance(shot, (int, float)) and shot == shot else None,
        "loader": "vaft.omas.load",
        "data_dictionary": _data_dictionary_version(ods),
        "vaft_version": _vaft_version(),
    })


def _database(shot: int, source: str | None) -> Dataset:
    if not database_enabled():
        raise ValueError(
            f"shot {shot} is not a packaged sample, and database access is off: start the server "
            f"with {DATABASE_ENV}=1 to read database shots (read-only, your own HSDS configuration)"
        )
    import vaft.database as database

    try:
        ods = database.load(int(shot), source=source, cache="off")
    except Exception as error:  # noqa: BLE001 - never echo server text: it may carry endpoints or accounts
        raise ValueError(f"database shot {shot} (source {source or 'default'}) is unavailable: "
                         f"{type(error).__name__}") from None
    return Dataset(ods, {
        "kind": "database",
        "shot": int(shot),
        "database_source": source or "default",
        "loader": "vaft.database.load",
        "data_dictionary": _data_dictionary_version(ods),
        "vaft_version": _vaft_version(),
    })


def load_dataset(shot: int | None = None, artifact: str | None = None,
                 database_source: str | None = None) -> Dataset:
    """Resolve one dataset: exactly one of ``shot`` and ``artifact``.

    A shot is the packaged sample when one ships and no ``database_source`` is
    named; otherwise it is read from the database (when enabled).
    """
    if (shot is None) == (artifact is None):
        raise ValueError("name exactly one dataset: shot=<number> or artifact=<relative path>")
    if artifact is not None:
        if database_source is not None:
            raise ValueError("database_source applies to shots, not artifacts")
        return _artifact(str(artifact))
    try:
        shot = int(shot)
    except (TypeError, ValueError):
        raise ValueError(f"shot must be a shot number, got {shot!r}") from None
    if database_source is None and shot in known_shots():
        return Dataset(load_reference_shot(shot), {
            "kind": "sample",
            "shot": shot,
            "loader": "vaft.omas.sample.sample_ods",
            "data_dictionary": None,
            "vaft_version": _vaft_version(),
        })
    return _database(shot, None if database_source is None else str(database_source))
