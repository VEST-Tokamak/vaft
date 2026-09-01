"""Direct-HSDS, read-only lazy native IMAS IDS access.

This is deliberately an adapter rather than an IMAS-Core backend: HSDS stores
IMAS HDF5 images as domains, so the adapter translates IMAS-Python lazy-node
access into ``h5pyd`` dataset selections without materializing an HDF5 file.
It relies on IMAS-Python's lazy node protocol and is consequently pinned to
the supported IMAS-Python release by VAFT's optional dependency set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

try:  # pragma: no cover - guarded by the public constructor
    import h5pyd
except ImportError:  # pragma: no cover
    h5pyd = None

from .sources import resolve as resolve_source
from .lazy_common import decode_hdf5_value, discover_hsds_ids, normalize_ids
from .utils import _require_h5pyd


_AOS = object()


class LazyIMASClosedError(RuntimeError):
    """Raised when an uncached native IDS leaf is read after ``close()``."""


@dataclass(frozen=True)
class _Record:
    template: tuple[Any, ...]
    dataset: Any


class _IDSIndex:
    """Metadata-only index for one HSDS IMAS HDF5 image domain."""

    def __init__(self, handle: "HSDSIMASHandle", ids_name: str, domain: Any) -> None:
        self.handle = handle
        self.ids_name = ids_name
        self.records: dict[tuple[Any, ...], _Record] = {}
        self.shapes: dict[tuple[Any, ...], _Record] = {}
        self.aos_shapes: dict[tuple[Any, ...], _Record] = {}
        root = domain[ids_name] if ids_name in domain else domain

        def walk(group: Any, prefix: tuple[Any, ...] = ()) -> None:
            for name, node in group.items():
                if hasattr(node, "items"):
                    walk(node, prefix + (name,))
                    continue
                components = list(prefix)
                for component in name.split("&"):
                    if component.endswith("[]"):
                        components.extend((component[:-2], _AOS))
                    else:
                        components.append(component)
                template = tuple(components)
                if components and components[-1] == "AOS_SHAPE":
                    base = template[:-1]
                    self.aos_shapes[self._without_aos(base)] = _Record(base, node)
                elif components and str(components[-1]).endswith("_SHAPE"):
                    base = (*template[:-1], str(template[-1])[:-6])
                    self.shapes[self._without_aos(base)] = _Record(base, node)
                else:
                    self.records[self._without_aos(template)] = _Record(template, node)

        walk(root)
        handle._metrics["metadata_dataset_count"] += (
            len(self.records) + len(self.shapes) + len(self.aos_shapes)
        )

    @staticmethod
    def _without_aos(template: tuple[Any, ...]) -> tuple[Any, ...]:
        return tuple(component for component in template if component is not _AOS)

    @staticmethod
    def _path_key(path: tuple[Any, ...]) -> tuple[Any, ...]:
        """Metadata keys omit concrete AOS indexes; records retain them only in selections."""
        return tuple(component for component in path if not isinstance(component, int))

    @staticmethod
    def _selection(template: tuple[Any, ...], path: tuple[Any, ...]) -> tuple[Any, ...]:
        selected: list[Any] = []
        for component, value in zip(template, path):
            if component is _AOS:
                selected.append(int(value))
        return tuple(selected)

    def aos_length(self, path: tuple[Any, ...]) -> int:
        record = self.aos_shapes.get(self._path_key(path))
        if record is None:
            return 0
        value = np.asarray(record.dataset[self._selection(record.template, path)])
        self.handle._metrics["payload_selection_count"] += 1
        return int(value.reshape(-1)[0]) if value.size else 0

    def value(self, path: tuple[Any, ...]) -> Any | None:
        record = self.records.get(self._path_key(path))
        if record is None:
            return None
        cache_key = (self.ids_name, path)
        if cache_key in self.handle._leaf_cache:
            self.handle._metrics["leaf_cache_hits"] += 1
            return self.handle._leaf_cache[cache_key]
        selection = self._selection(record.template, path)
        value = np.asarray(record.dataset[selection])
        shape_record = self.shapes.get(self._path_key(path))
        if shape_record is not None:
            logical_shape = np.asarray(
                shape_record.dataset[self._selection(shape_record.template, path)]
            ).reshape(-1)
            if logical_shape.size:
                value = value[tuple(slice(0, int(size)) for size in logical_shape)]
        value = decode_hdf5_value(value)
        self.handle._metrics["payload_selection_count"] += 1
        self.handle._metrics["returned_logical_bytes"] += int(np.asarray(value).nbytes)
        if isinstance(value, np.ndarray):
            value = value.view()
            value.flags.writeable = False
        self.handle._leaf_cache[cache_key] = value
        return value


class _HSDSLazyContext:
    """Duck-typed replacement for IMAS-Core's private LazyALContext."""

    def __init__(self, handle: "HSDSIMASHandle", index: _IDSIndex, path: tuple[Any, ...], time_mode: int) -> None:
        self.handle = handle
        self.index = index
        self.path = path
        self.time_mode = time_mode

    def _check(self) -> None:
        if self.handle.closed:
            raise LazyIMASClosedError(
                "Cannot fetch an uncached lazy IDS leaf after HSDSIMASHandle.close()"
            )

    def get_context(self) -> "_HSDSLazyContext":
        self._check()
        return self

    def get_child(self, child: Any) -> None:
        self._check()
        from imas.ids_data_type import IDSDataType

        name = child.metadata.name
        path = (*self.path, name)
        if child.metadata.data_type is IDSDataType.STRUCT_ARRAY:
            child._set_lazy_context(_HSDSAOSContext(self.handle, self.index, path, self.time_mode))
        elif child.metadata.data_type is IDSDataType.STRUCTURE:
            child._set_lazy_context(_HSDSLazyContext(self.handle, self.index, path, self.time_mode))
        else:
            value = self.index.value(path)
            if value is not None:
                child._IDSPrimitive__value = value

    def read_data(self, *_: Any) -> Any:
        # IMAS' helper invokes this only for a primitive represented by the
        # current child context; VAFT handles primitives in ``get_child``.
        raise RuntimeError("HSDS lazy IMAS adapter does not expose raw AL read_data")


class _HSDSAOSContext(_HSDSLazyContext):
    @property
    def size(self) -> int:
        return self.index.aos_length(self.path)

    def iterate_to_index(self, index: int) -> _HSDSLazyContext:
        self._check()
        return _HSDSLazyContext(
            self.handle, self.index, (*self.path, int(index)), self.time_mode
        )


class HSDSIMASHandle:
    """Read-only context manager returning native lazy ``IDSToplevel`` objects."""

    def __init__(
        self,
        shot: int,
        source: str | None = None,
        *,
        directory: str | None = None,
        ids: str | Iterable[str] | None = None,
        imas_version: str,
        h5pyd_module: Any = None,
    ) -> None:
        if h5pyd_module is None:
            _require_h5pyd()
            h5pyd_module = h5pyd
        self.shot = int(shot)
        self.source = resolve_source(source, directory=directory)
        self.imas_version = imas_version
        self._h5pyd = h5pyd_module
        self._requested_ids = normalize_ids(ids)
        self._available_ids: tuple[str, ...] | None = self._requested_ids
        self._domains: dict[str, Any] = {}
        self._indexes: dict[str, _IDSIndex] = {}
        self._objects: dict[str, Any] = {}
        self._leaf_cache: dict[tuple[str, tuple[Any, ...]], Any] = {}
        self.closed = False
        self._metrics = {
            "ids_domain_open_count": 0,
            "metadata_dataset_count": 0,
            "payload_selection_count": 0,
            "leaf_cache_hits": 0,
            "returned_logical_bytes": 0,
        }

    @property
    def ids(self) -> tuple[str, ...]:
        if self._available_ids is None:
            if self.closed:
                raise LazyIMASClosedError("Cannot discover IDS after close()")
            self._available_ids = discover_hsds_ids(
                self._h5pyd, self.source, self.shot
            )
        return self._available_ids

    @property
    def metrics(self) -> dict[str, int]:
        return dict(self._metrics)

    def open(self) -> "HSDSIMASHandle":
        if self.closed:
            raise LazyIMASClosedError("A closed HSDSIMASHandle cannot be reopened")
        return self

    def _index(self, ids_name: str) -> _IDSIndex:
        if ids_name in self._indexes:
            return self._indexes[ids_name]
        if self.closed:
            raise LazyIMASClosedError(f"Cannot open IDS {ids_name!r} after close()")
        if ids_name not in self.ids:
            raise KeyError(f"IDS {ids_name!r} is not available for shot {self.shot}")
        domain = self._h5pyd.File(
            f"hdf5://{self.source}/{self.shot}/{ids_name}.h5", "r"
        )
        self._domains[ids_name] = domain
        self._metrics["ids_domain_open_count"] += 1
        self._indexes[ids_name] = _IDSIndex(self, ids_name, domain)
        return self._indexes[ids_name]

    def get(self, ids_name: str | None = None, occurrence: int | None = None):
        """Return a native, read-only, lazy ``IDSToplevel`` for an IDS."""
        if occurrence not in (None, 0):
            raise NotImplementedError("HSDS native lazy IDS supports occurrence 0 only")
        if ids_name is None:
            if len(self.ids) != 1:
                raise ValueError("ids_name is required when more than one IDS is selected")
            ids_name = self.ids[0]
        if ids_name in self._objects:
            return self._objects[ids_name]
        index = self._index(ids_name)
        import imas
        from imas.ids_defs import IDS_TIME_MODE_HOMOGENEOUS

        factory = imas.IDSFactory(self.imas_version)
        ids = factory.new(ids_name, _lazy=True)
        time_mode = IDS_TIME_MODE_HOMOGENEOUS
        mode_path = ("ids_properties", "homogeneous_time")
        mode = index.value(mode_path)
        if mode is not None:
            time_mode = int(mode)
        ids._set_lazy_context(_HSDSLazyContext(self, index, (), time_mode))
        self._objects[ids_name] = ids
        return ids

    def close(self) -> None:
        if self.closed:
            return
        for domain in self._domains.values():
            domain.close()
        self._domains.clear()
        self.closed = True

    def __enter__(self) -> "HSDSIMASHandle":
        return self.open()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def open_imas(
    shot: int,
    source: str | None = None,
    *,
    directory: str | None = None,
    ids: str | Iterable[str] | None = None,
    imas_version: str,
) -> HSDSIMASHandle:
    return HSDSIMASHandle(
        shot, source, directory=directory, ids=ids, imas_version=imas_version
    )


__all__ = ["HSDSIMASHandle", "LazyIMASClosedError", "open_imas"]
