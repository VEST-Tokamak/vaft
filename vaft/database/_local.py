"""Private local-source detection shared by :mod:`vaft.imas` and
:mod:`vaft.omas`.

The public modules deliberately expose different result types, but share one
content-based detector so callers never have to pre-classify a local artifact.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
import gzip
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Sequence
import warnings

import h5py

from ..compat import reopenable_temporary_file


_FALLBACK_DD = "3.41.0"


@dataclass(frozen=True)
class SourceInfo:
    """Provenance recorded for an automatically detected local source."""

    format: str
    source_paths: tuple[Path, ...]
    imas_version: str
    version_fallback: bool
    converted: bool = False
    available_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Descriptor:
    format: str
    paths: tuple[Path, ...]
    entry_path: Path | None
    imas_version: str | None


def _as_paths(source: str | Path | Sequence[str | Path]) -> tuple[Path, ...]:
    if isinstance(source, (str, Path)):
        values = (source,)
    else:
        values = tuple(source)
    if not values:
        raise ValueError("source must contain at least one file or directory")
    paths = tuple(Path(value).expanduser().resolve() for value in values)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Local source does not exist: " + ", ".join(missing))
    return paths


def _decode(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _imas_version_hdf5(path: Path) -> str | None:
    try:
        with h5py.File(path, "r") as handle:
            version = _decode(handle.attrs.get("data_dictionary_version"))
            if version:
                return version
            roots = [handle.get("dataset_description"), handle]
            for root in roots:
                if root is None:
                    continue
                for name in (
                    "imas_version",
                    "ids_properties&version_put&data_dictionary",
                ):
                    if name in root:
                        return _decode(root[name][()])
    except OSError:
        return None
    return None


def _is_geqdsk(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() in {
        ".h5",
        ".hdf5",
        ".nc",
        ".json",
        ".gz",
    }:
        return False
    try:
        from ..data.eqdsk import read_geqdsk

        read_geqdsk(path)
        return True
    except Exception:
        return False


def _native_image(path: Path) -> bool:
    try:
        with h5py.File(path, "r") as handle:
            if "h5image" in handle:
                return False
            if any(
                isinstance(handle.get(name, getlink=True), h5py.ExternalLink)
                for name in handle
            ):
                return True

            def flat_names(group: h5py.Group) -> bool:
                return any("&" in name or name.endswith("_SHAPE") for name in group)

            return flat_names(handle) or any(
                isinstance(item, h5py.Group) and flat_names(item)
                for item in handle.values()
            )
    except OSError:
        return False


def _make_partial_entry(
    images: Iterable[Path],
) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    """Create an isolated IMAS HDF5 entry for IDS image files without master."""
    temp = tempfile.TemporaryDirectory(prefix="vaft-local-imas-")
    root = Path(temp.name)
    copied: list[Path] = []
    for image in images:
        if image.name == "master.h5" or image.suffix != ".h5":
            continue
        destination = root / image.name
        shutil.copy2(image, destination)
        copied.append(destination)
    if not copied:
        temp.cleanup()
        raise ValueError(
            "No IMAS IDS HDF5 images were available to build a partial entry"
        )
    with h5py.File(root / "master.h5", "w") as master:
        # IMAS AL5 identifies an HDF5 entry through these root attributes.
        # Copy them from an image before adding the synthetic external links.
        with h5py.File(copied[0], "r") as first_image:
            for name, value in first_image.attrs.items():
                master.attrs[name] = value
        for image in copied:
            master[image.stem] = h5py.ExternalLink(image.name, f"/{image.stem}")
    return temp, root


def _detect(source: str | Path | Sequence[str | Path]) -> _Descriptor:
    paths = _as_paths(source)
    if len(paths) > 1:
        if all(_is_geqdsk(path) for path in paths):
            return _Descriptor("geqdsk_collection", paths, None, None)
        raise ValueError(
            "Multiple local sources are supported only for a GEQDSK collection"
        )

    path = paths[0]
    if path.is_dir():
        master = path / "master.h5"
        if master.is_file():
            return _Descriptor("imas_hdf5", (path,), path, _imas_version_hdf5(master))
        gfiles = tuple(sorted(item for item in path.iterdir() if _is_geqdsk(item)))
        if gfiles:
            return _Descriptor("geqdsk_collection", gfiles, None, None)
        images = tuple(
            sorted(item for item in path.glob("*.h5") if _native_image(item))
        )
        if images:
            return _Descriptor(
                "imas_images", images, None, _imas_version_hdf5(images[0])
            )
        raise ValueError(
            f"Could not identify a supported local source in directory: {path}"
        )

    if _is_geqdsk(path):
        return _Descriptor("geqdsk", (path,), None, None)
    if path.suffix.lower() in {".json", ".gz"}:
        return _Descriptor("omas_json", (path,), None, None)
    if path.suffix.lower() == ".nc":
        version = _imas_version_hdf5(path)
        return _Descriptor("imas_netcdf", (path,), path, version)
    if path.suffix.lower() in {".h5", ".hdf5"}:
        if path.name == "master.h5":
            return _Descriptor(
                "imas_hdf5", (path.parent,), path.parent, _imas_version_hdf5(path)
            )
        if _native_image(path):
            return _Descriptor("imas_images", (path,), None, _imas_version_hdf5(path))
        return _Descriptor("omas_hdf5", (path,), None, None)
    raise ValueError(f"Unsupported local source: {path}")


def _resolved_version(
    descriptor: _Descriptor, requested: str | None
) -> tuple[str, bool]:
    if requested is not None:
        return requested, False
    if descriptor.imas_version:
        return descriptor.imas_version, False
    warnings.warn(
        f"Could not infer an IMAS DD version for {descriptor.format}; using {_FALLBACK_DD}",
        RuntimeWarning,
        stacklevel=3,
    )
    return _FALLBACK_DD, True


def _merge_geqdsk(paths: tuple[Path, ...]):
    from omas import ODS
    from ..data.eqdsk import read_geqdsk, to_omas

    records = [read_geqdsk(path) for path in paths]
    records.sort(key=lambda item: (float(item.get("TIME", 0.0)), str(item.source)))
    ods = ODS(consistency_check=False)
    for index, record in enumerate(records):
        to_omas(record, ods=ods, time_index=index)
    return ods


def load_ods(
    source: str | Path | Sequence[str | Path], *, imas_version: str | None = None
):
    """Load any supported local source and normalize it to an OMAS ODS."""
    descriptor = _detect(source)
    version, fallback = _resolved_version(descriptor, imas_version)
    if descriptor.format in {"geqdsk", "geqdsk_collection"}:
        return _merge_geqdsk(descriptor.paths), SourceInfo(
            descriptor.format, descriptor.paths, version, fallback, converted=True
        )
    if descriptor.format in {"omas_json", "omas_hdf5"}:
        from omas import ODS

        ods = ODS(consistency_check=False)
        source_path = descriptor.paths[0]
        if source_path.suffixes[-2:] == [".json", ".gz"]:
            # `ODS.load` opens the path itself, which a still-open
            # NamedTemporaryFile forbids on Windows -- and this is the path
            # `sample_ods()` takes, so it breaks the packaged sample and every
            # tutorial that starts from it.
            with reopenable_temporary_file(suffix=".json") as plain_path:
                with gzip.open(source_path, "rb") as compressed:
                    with plain_path.open("wb") as plain:
                        shutil.copyfileobj(compressed, plain)
                ods.load(str(plain_path), consistency_check=False)
        else:
            ods.load(str(source_path), consistency_check=False)
        return ods, SourceInfo(descriptor.format, descriptor.paths, version, fallback)

    temporary: tempfile.TemporaryDirectory[str] | None = None
    try:
        if descriptor.format == "imas_images":
            temporary, entry_root = _make_partial_entry(descriptor.paths)
            uri = "imas:hdf5?path=" + str(entry_root)
        elif descriptor.format == "imas_hdf5":
            assert descriptor.entry_path is not None
            uri = "imas:hdf5?path=" + str(descriptor.entry_path)
        else:  # imas_netcdf
            assert descriptor.entry_path is not None
            uri = str(descriptor.entry_path)
        from ..imas.omas_imas import load_omas_imas

        ods = load_omas_imas(
            uri=uri, imas_version=version, occurrence={}, verbose=False
        )
        return ods, SourceInfo(descriptor.format, descriptor.paths, version, fallback)
    finally:
        if temporary is not None:
            temporary.cleanup()


class IMASHandle(AbstractContextManager["IMASHandle"]):
    """A native IMAS entry with ownership of any temporary conversion store."""

    def __init__(
        self,
        source: str | Path | Sequence[str | Path],
        *,
        imas_version: str | None = None,
    ):
        self._descriptor = _detect(source)
        self._version, fallback = _resolved_version(self._descriptor, imas_version)
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self._entry: Any = None
        self._ids: tuple[str, ...] = ()
        self.info = SourceInfo(
            self._descriptor.format,
            self._descriptor.paths,
            self._version,
            fallback,
            converted=self._descriptor.format not in {"imas_hdf5", "imas_netcdf"},
        )

    @property
    def ids(self) -> tuple[str, ...]:
        return self._ids

    def open(self) -> "IMASHandle":
        if self._entry is not None:
            return self
        descriptor = self._descriptor
        ids_hint: tuple[str, ...] = ()
        if descriptor.format == "imas_hdf5":
            assert descriptor.entry_path is not None
            uri = "imas:hdf5?path=" + str(descriptor.entry_path)
            master = descriptor.entry_path / "master.h5"
            with h5py.File(master, "r") as handle:
                ids_hint = tuple(sorted(handle.keys()))
        elif descriptor.format == "imas_netcdf":
            assert descriptor.entry_path is not None
            uri = str(descriptor.entry_path)
            with h5py.File(uri, "r") as handle:
                ids_hint = tuple(sorted(handle.keys()))
        else:
            ods, _ = load_ods(descriptor.paths, imas_version=self._version)
            ids_hint = tuple(sorted(ods.keys()))
            self._temporary = tempfile.TemporaryDirectory(prefix="vaft-imas-handle-")
            root = Path(self._temporary.name)
            from ..imas.omas_imas import save_omas_imas

            save_omas_imas(
                ods,
                occurrence={},
                imas_version=self._version,
                new=True,
                verbose=False,
                uri="imas:hdf5?path=" + str(root),
            )
            uri = "imas:hdf5?path=" + str(root)
        import imas

        self._entry = imas.DBEntry(uri, "r", dd_version=self._version)
        self._entry.__enter__()
        self._ids = ids_hint
        self.info = replace(self.info, available_ids=ids_hint)
        return self

    def get(self, ids_name: str, occurrence: int | None = None):
        self.open()
        assert self._entry is not None
        return self._entry.get(ids_name, 0 if occurrence is None else int(occurrence))

    def to_omas(self):
        self.open()
        assert self._entry is not None
        # Reuse the detector-backed reader for a stable, complete conversion.
        if self._temporary is not None:
            source: Path = Path(self._temporary.name)
        elif self._descriptor.entry_path is not None:
            source = self._descriptor.entry_path
        else:  # pragma: no cover - defensive
            raise RuntimeError("IMAS handle has no readable source")
        ods, _ = load_ods(source, imas_version=self._version)
        return ods

    def close(self) -> None:
        if self._entry is not None:
            self._entry.__exit__(None, None, None)
            self._entry = None
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None

    def __enter__(self) -> "IMASHandle":
        return self.open()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def open_imas(
    source: str | Path | Sequence[str | Path], *, imas_version: str | None = None
) -> IMASHandle:
    return IMASHandle(source, imas_version=imas_version)
