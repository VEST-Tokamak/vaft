"""Export one HSDS shot as portable local files (issue #450).

A shot on HSDS is a folder -- ``master.h5`` plus the IDS images it links to --
so ``hsget`` cannot fetch it whole. :func:`export` stages the canonical IMAS
HDF5 entry once, through the same cache and transport rules as
:func:`vaft.database.load`, and writes every requested backend from that one
local copy.

The IMAS and OMAS families are different formats even where they share an
extension: ``imas-hdf5`` is a native Data Entry directory, ``omas-hdf5`` one
OMAS file; ``imas-nc`` follows the IMAS netCDF convention, ``omas-nc`` is
OMAS's flat netCDF serialization.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
from typing import Any, Iterator, Literal
import uuid


#: backend -> (artifact name for a shot, whether it is a semantic conversion)
BACKENDS: dict[str, tuple[Callable[[int], str], bool]] = {
    "imas-hdf5": (lambda shot: f"imas_{shot}_hdf5", False),
    "imas-nc": (lambda shot: f"imas_{shot}.nc", True),
    "omas-json": (lambda shot: f"omas_{shot}.json", True),
    "omas-hdf5": (lambda shot: f"omas_{shot}.h5", True),
    "omas-nc": (lambda shot: f"omas_{shot}.nc", True),
    "geqdsk": (lambda shot: f"geqdsk_{shot}", True),
}

_NETCDF_BACKENDS = frozenset({"imas-nc", "omas-nc"})
_ODS_BACKENDS = frozenset({"omas-json", "omas-hdf5", "omas-nc", "geqdsk"})


def _backends(backend: str | Iterable[str]) -> list[str]:
    names = [backend] if isinstance(backend, str) else list(backend)
    names = list(dict.fromkeys(names))
    if not names:
        raise ValueError(f"backend must name at least one of: {', '.join(BACKENDS)}")
    unknown = [name for name in names if name not in BACKENDS]
    if unknown:
        raise ValueError(
            f"Unknown export backend(s) {', '.join(map(repr, unknown))}; "
            f"choose from: {', '.join(BACKENDS)}"
        )
    return names


def _require_netcdf(names: list[str]) -> None:
    requested = [name for name in names if name in _NETCDF_BACKENDS]
    if not requested:
        return
    try:
        import netCDF4  # noqa: F401
    except ImportError as error:
        raise ImportError(
            f"Export backend(s) {', '.join(requested)} need the netCDF4 package; "
            "install it with `pip install netCDF4`."
        ) from error


@contextmanager
def _atomic(target: Path, *, overwrite: bool) -> Iterator[Path]:
    """Yield a hidden sibling path; move it onto ``target`` only on success.

    A failed backend leaves neither a partial artifact nor a damaged previous
    one. The partial keeps the target's name as its suffix, because the
    writers choose their format from the extension.
    """
    partial = target.with_name(f".{uuid.uuid4().hex[:12]}.partial.{target.name}")
    try:
        yield partial
        if not partial.exists():
            raise RuntimeError(f"export backend wrote nothing for {target.name}")
        if target.exists() or target.is_symlink():
            if not overwrite:
                raise FileExistsError(f"{target} already exists; pass overwrite=True")
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink()
        os.replace(partial, target)
    finally:
        if partial.is_dir() and not partial.is_symlink():
            shutil.rmtree(partial, ignore_errors=True)
        elif partial.exists() or partial.is_symlink():
            partial.unlink()


def _ids_names(entry: Path, version: str) -> list[str]:
    """IDS names linked from ``master.h5``.

    Occurrences other than 0 appear there as ``<ids>_<n>`` links; they are
    found through ``list_all_occurrences`` instead, so only names the data
    dictionary knows are returned.
    """
    import h5py
    import imas

    factory = imas.IDSFactory(version)
    with h5py.File(entry / "master.h5", "r") as master:
        return sorted(name for name in master.keys() if factory.exists(name))


def _require_single_occurrence(entry: Path, names: list[str], version: str) -> None:
    """Refuse a converted export when any IDS holds an occurrence other than 0."""
    import imas

    extra: dict[str, list[int]] = {}
    with imas.DBEntry("imas:hdf5?path=" + str(entry), "r", dd_version=version) as db:
        for name in names:
            others = [int(value) for value in db.list_all_occurrences(name) if int(value) != 0]
            if others:
                extra[name] = others
    if extra:
        detail = "; ".join(f"{name}: {values}" for name, values in sorted(extra.items()))
        raise ValueError(
            "This shot stores IDS occurrences other than 0, which the converted "
            f"backends do not carry ({detail}). Export 'imas-hdf5' for a lossless copy."
        )


def _write_geqdsk(ods: Any, shot: int, directory: Path) -> None:
    from vaft.data.eqdsk import from_omas, geqdsk_filenames, write_geqdsk

    count = len(ods["equilibrium.time_slice"]) if "equilibrium.time_slice" in ods else 0
    if not count:
        raise ValueError(f"Shot {shot} has no equilibrium time slices to write as GEQDSK")
    times = []
    for index in range(count):
        key = f"equilibrium.time_slice.{index}.time"
        times.append(float(ods[key]) if key in ods else float(ods["equilibrium.time"][index]))
    directory.mkdir()
    for index, name in enumerate(geqdsk_filenames(shot, times)):
        write_geqdsk(from_omas(ods, index), directory / name)


def export(
    shot: int,
    source: str | None = None,
    *,
    backend: str | Iterable[str],
    output: str | Path | None = None,
    overwrite: bool = False,
    occurrence: int = 0,
    imas_version: str | None = None,
    cache: str | Path = "auto",
    transport: Literal["auto", "canonical", "h5image"] = "auto",
) -> dict[str, Path]:
    """Implementation of :func:`vaft.database.export`."""
    from vaft.compat import temporary_directory

    from . import staging
    from ._local import _imas_version_hdf5
    from .sources import resolve

    names = _backends(backend)
    semantic = [name for name in names if BACKENDS[name][1]]
    if occurrence != 0 and semantic:
        raise ValueError(
            f"occurrence={occurrence!r} is not supported for {', '.join(semantic)}; "
            "these backends export occurrence 0 only (use 'imas-hdf5' for a lossless copy)"
        )
    directory = resolve(source)
    shot = int(shot)
    root = Path(Path.cwd() if output is None else output).expanduser()
    targets = {name: root / BACKENDS[name][0](shot) for name in names}
    existing = [str(path) for path in targets.values() if path.exists() or path.is_symlink()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Export target(s) already exist: {', '.join(existing)}; pass overwrite=True"
        )
    _require_netcdf(names)
    root.mkdir(parents=True, exist_ok=True)

    with temporary_directory(prefix="vaft_export_") as scratch:
        entry = scratch / str(shot)
        entry.mkdir()
        staging.stage_imas_shot(
            directory, shot, entry, requested_ids=None, cache=cache, transport=transport
        )
        version = imas_version or _imas_version_hdf5(entry / "master.h5")
        if version is None:
            from vaft.imas import IMAS_DD_VERSION_CONVERSION

            version = IMAS_DD_VERSION_CONVERSION
        ids_names = _ids_names(entry, version)
        if semantic:
            _require_single_occurrence(entry, ids_names, version)

        ods = None
        if any(name in _ODS_BACKENDS for name in names):
            from vaft.omas import load as load_ods

            ods = load_ods(entry, imas_version=version)

        for name in names:
            with _atomic(targets[name], overwrite=overwrite) as partial:
                if name == "imas-hdf5":
                    shutil.copytree(entry, partial)
                elif name == "imas-nc":
                    from vaft.imas import _copy_entry

                    _copy_entry("imas:hdf5?path=" + str(entry), partial, ids_names, version, occurrence=0)
                elif name == "geqdsk":
                    _write_geqdsk(ods, shot, partial)
                else:
                    from vaft.omas import save as save_ods

                    save_ods(ods, partial)
    return dict(targets)


__all__ = ["BACKENDS", "export"]
