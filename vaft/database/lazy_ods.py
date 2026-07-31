"""Lazy, selection-based OMAS access to IMAS HDF5 domains in HSDS.

This module deliberately does not use ``hsget``.  It indexes dataset metadata
for an IDS when that IDS is first visited, then asks h5pyd for only the dataset
selection corresponding to the requested OMAS leaf.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np
import omas
from omas.omas_core import dynamic_ODS
from omas.omas_utils import p2l

try:
    import h5pyd
except ImportError:  # pragma: no cover - exercised through the public guard
    h5pyd = None

from .lazy_common import decode_hdf5_value, discover_hsds_ids, normalize_ids
from .utils import _require_h5pyd


_AOS = object()


class LazyODSClosedError(RuntimeError):
    """Raised when an uncached leaf is requested after closing a lazy ODS."""


@dataclass(frozen=True)
class _DatasetRecord:
    template: tuple[Any, ...]
    dataset: Any


def _path_parts(path: Any) -> list[Any]:
    """Normalize every OMAS-supported path syntax to strings and integers."""
    return list(p2l(path))


class HSDSStore(dynamic_ODS):
    """OMAS dynamic backend for occurrence-zero IMAS HDF5 HSDS domains."""

    def __init__(
        self,
        shot: int,
        directory: str = "public",
        *,
        ids: str | Iterable[str] | None = None,
        h5pyd_module: Any = None,
    ) -> None:
        if h5pyd_module is None:
            _require_h5pyd()
            h5pyd_module = h5pyd

        self.shot = int(shot)
        self.directory = directory.strip("/")
        self._h5pyd = h5pyd_module
        self._requested_ids = normalize_ids(ids)
        self._available_ids_cache = self._requested_ids
        self._handles: dict[str, Any] = {}
        self._records: dict[str, list[_DatasetRecord]] = {}
        self._record_by_template: dict[str, dict[tuple[Any, ...], _DatasetRecord]] = {}
        self._children: dict[str, dict[tuple[Any, ...], set[Any]]] = {}
        self._shape_datasets: dict[str, dict[tuple[Any, ...], Any]] = {}
        self._aos_shape_datasets: dict[str, dict[tuple[Any, ...], Any]] = {}
        self._aos_length_cache: dict[tuple[Any, ...], int] = {}
        self._leaf_cache: dict[str, Any] = {}
        self._metrics = {
            "ids_domain_open_count": 0,
            "metadata_dataset_count": 0,
            "payload_selection_count": 0,
            "leaf_cache_hits": 0,
            "returned_logical_bytes": 0,
        }
        self.closed = False
        self.active = True
        self.kw = {
            "shot": self.shot,
            "directory": self.directory,
            "ids": list(self._requested_ids) if self._requested_ids is not None else None,
        }

    @property
    def opened_ids(self) -> tuple[str, ...]:
        """IDS domains currently opened by this store (primarily diagnostic)."""
        return tuple(self._handles)

    @property
    def metrics(self) -> dict[str, int]:
        """Snapshot lightweight client-side lazy-read instrumentation.

        Values describe local h5pyd operations, not compressed wire bytes or
        reverse-proxy requests (those are intentionally outside this client API).
        """
        return dict(self._metrics)

    def open(self) -> "HSDSStore":
        if self.closed:
            raise LazyODSClosedError("A closed HSDSStore cannot be reopened")
        self.active = True
        return self

    def close(self) -> None:
        if self.closed:
            return
        for handle in tuple(self._handles.values()):
            handle.close()
        self._handles.clear()
        self.closed = True
        # Keep the dynamic adapter active so OMAS routes an uncached access here
        # and receives a useful closed-store exception. Cached OMAS leaves never
        # call the adapter again.
        self.active = True

    def _available_ids(self) -> tuple[str, ...]:
        if self._available_ids_cache is not None:
            return self._available_ids_cache
        if self.closed:
            raise LazyODSClosedError("Cannot discover IDS domains after the lazy ODS is closed")
        self._available_ids_cache = discover_hsds_ids(
            self._h5pyd, self.directory, self.shot
        )
        return self._available_ids_cache

    def _ensure_index(self, ids_name: str) -> None:
        if ids_name in self._records:
            return
        if self.closed:
            raise LazyODSClosedError(
                f"Cannot open uncached IDS {ids_name!r} after the lazy ODS is closed"
            )
        if ids_name not in self._available_ids():
            raise KeyError(f"IDS {ids_name!r} is not available for shot {self.shot}")

        uri = f"hdf5://{self.directory}/{self.shot}/{ids_name}.h5"
        handle = self._h5pyd.File(uri, "r")
        self._metrics["ids_domain_open_count"] += 1
        self._handles[ids_name] = handle
        root = handle[ids_name] if ids_name in handle else handle
        records: list[_DatasetRecord] = []
        shapes: dict[tuple[Any, ...], Any] = {}
        aos_shapes: dict[tuple[Any, ...], Any] = {}

        def walk(group: Any, prefix: tuple[Any, ...] = ()) -> None:
            for name, node in group.items():
                if hasattr(node, "items"):
                    walk(node, prefix + (name,))
                    continue
                components = list(prefix)
                for flat_name in name.split("&"):
                    if flat_name.endswith("[]"):
                        components.extend((flat_name[:-2], _AOS))
                    else:
                        components.append(flat_name)
                template = (ids_name, *components)
                if components and components[-1] == "AOS_SHAPE":
                    aos_shapes[tuple(template[:-1])] = node
                elif components and str(components[-1]).endswith("_SHAPE"):
                    data_template = (*template[:-1], str(template[-1])[:-6])
                    shapes[tuple(data_template)] = node
                else:
                    records.append(_DatasetRecord(tuple(template), node))

        walk(root)
        self._records[ids_name] = records
        self._record_by_template[ids_name] = {
            record.template: record for record in records
        }
        self._shape_datasets[ids_name] = shapes
        self._aos_shape_datasets[ids_name] = aos_shapes
        self._metrics["metadata_dataset_count"] += len(records) + len(shapes) + len(aos_shapes)
        children: dict[tuple[Any, ...], set[Any]] = {}
        for template in [*(record.template for record in records), *aos_shapes]:
            for offset, child in enumerate(template):
                children.setdefault(template[:offset], set()).add(child)
        self._children[ids_name] = children

    @staticmethod
    def _prefix_matches(template: tuple[Any, ...], location: list[Any]) -> bool:
        if len(location) > len(template):
            return False
        for expected, actual in zip(template, location):
            if expected is _AOS:
                if not isinstance(actual, int):
                    return False
            elif expected != actual:
                return False
        return True

    @staticmethod
    def _indices_for(template: tuple[Any, ...], location: list[Any]) -> tuple[int, ...]:
        return tuple(
            int(actual)
            for expected, actual in zip(template, location)
            if expected is _AOS
        )

    def _aos_length(
        self,
        ids_name: str,
        aos_template: tuple[Any, ...],
        location: list[Any],
    ) -> int:
        parent_indices = self._indices_for(aos_template[:-1], location)
        cache_key = (ids_name, aos_template, parent_indices)
        if cache_key in self._aos_length_cache:
            return self._aos_length_cache[cache_key]

        shape_dataset = self._aos_shape_datasets[ids_name].get(aos_template)
        if shape_dataset is not None:
            selection = parent_indices if parent_indices else (...,)
            raw = np.asarray(shape_dataset[selection]).reshape(-1)
            length = int(raw[0]) if raw.size else 0
        else:
            # A conservative fallback for older files lacking AOS_SHAPE.
            axis = sum(part is _AOS for part in aos_template) - 1
            candidates = [
                record.dataset.shape[axis]
                for record in self._records[ids_name]
                if record.template[: len(aos_template)] == aos_template
                and len(record.dataset.shape) > axis
            ]
            length = max(candidates, default=0)
        self._aos_length_cache[cache_key] = length
        return length

    def keys(self, location: Any) -> list[Any]:
        parts = _path_parts(location)
        if not parts:
            return list(self._available_ids())
        ids_name = str(parts[0])
        self._ensure_index(ids_name)
        template_prefix = tuple(_AOS if isinstance(part, int) else part for part in parts)
        children: set[Any] = set()
        for child in self._children[ids_name].get(template_prefix, ()):
            if child is _AOS:
                aos_template = (*template_prefix, _AOS)
                children.update(range(self._aos_length(ids_name, aos_template, parts)))
            else:
                children.add(child)
        return sorted(children, key=lambda value: (isinstance(value, int), str(value)))

    def __contains__(self, location: Any) -> bool:
        parts = _path_parts(location)
        if not parts:
            return False
        if str(parts[0]) not in self._available_ids():
            return False
        self._ensure_index(str(parts[0]))
        # OMAS asks __contains__ to decide whether a location can be fetched as
        # a value. Returning True for a structural node would make OMAS call
        # __getitem__ for that subtree instead of constructing a lazy ODS node.
        return self._find_record(parts) is not None

    def _find_record(self, parts: list[Any]) -> Optional[_DatasetRecord]:
        ids_name = str(parts[0])
        template = tuple(_AOS if isinstance(part, int) else part for part in parts)
        return self._record_by_template[ids_name].get(template)

    def __getitem__(self, location: Any) -> Any:
        parts = _path_parts(location)
        cache_key = ".".join(map(str, parts))
        if cache_key in self._leaf_cache:
            self._metrics["leaf_cache_hits"] += 1
            return self._leaf_cache[cache_key]
        if self.closed:
            raise LazyODSClosedError(
                f"Cannot fetch uncached path {cache_key!r} after the lazy ODS is closed"
            )
        if not parts:
            raise KeyError(location)
        ids_name = str(parts[0])
        self._ensure_index(ids_name)
        record = self._find_record(parts)
        if record is None:
            raise KeyError(f"No HSDS dataset maps to OMAS path {cache_key!r}")

        indices = self._indices_for(record.template, parts)
        shape_dataset = self._shape_datasets[ids_name].get(record.template)
        if shape_dataset is not None:
            selection = indices if indices else (...,)
            extents = np.asarray(shape_dataset[selection]).reshape(-1)
            trailing = tuple(slice(0, int(size)) for size in extents)
        else:
            trailing = (slice(None),) * max(0, len(record.dataset.shape) - len(indices))
        selection = indices + trailing
        self._metrics["payload_selection_count"] += 1
        value = decode_hdf5_value(record.dataset[selection])
        self._metrics["returned_logical_bytes"] += int(np.asarray(value).nbytes)
        self._leaf_cache[cache_key] = value
        return value


class HSDSODS(omas.ODS):
    """An :class:`omas.ODS` whose missing leaves are fetched from HSDS."""

    def __init__(
        self,
        *,
        store: HSDSStore | None = None,
        dynamic: HSDSStore | None = None,
        imas_version: str | None = None,
        consistency_check: bool = True,
        **kwargs: Any,
    ) -> None:
        backend = store or dynamic
        if backend is None:
            raise TypeError("HSDSODS requires an HSDSStore")
        init_kwargs = dict(kwargs)
        init_kwargs.update(consistency_check=consistency_check, dynamic=backend)
        if imas_version is not None:
            init_kwargs["imas_version"] = imas_version
        super().__init__(**init_kwargs)
        self.store = backend

    def keys(self, dynamic: bool = True) -> list[Any]:
        # With consistency_check=False OMAS cannot infer that a schema node is
        # an AOS, leaving its internal data as None. The HSDS trie still knows
        # the concrete children, so expose them directly in that case.
        if dynamic and self.active_dynamic and self.omas_data is None and self.location:
            return self.store.keys(self.location)
        return super().keys(dynamic=dynamic)

    def __getitem__(self, key: Any, cocos_and_coords: bool | None = True) -> Any:
        requested = _path_parts(key)
        location = _path_parts(self.location)
        full_path = location + requested
        # Fetch exact leaves in one backend operation. Besides saving several
        # metadata round trips, this lets lazy access work when callers disable
        # OMAS schema consistency (OMAS otherwise cannot infer AOS containers).
        if full_path and self.store.__contains__(full_path):
            value = self.store.__getitem__(full_path)
            # OMAS does not permit assigning index N to an empty AOS when N>0.
            # Materialize only the lightweight intermediate ODS nodes needed to
            # cache this selection; no additional leaf is fetched.
            for offset, part in enumerate(requested):
                if not isinstance(part, int) or part <= 0:
                    continue
                container = super().__getitem__(requested[:offset], False)
                while len(container.keys(dynamic=0)) <= part:
                    container.setraw(len(container.keys(dynamic=0)), container.same_init_ods())
            self.__setitem__(requested, value)
        return super().__getitem__(key, cocos_and_coords)

    def __enter__(self) -> "HSDSODS":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        self.store.close()


def open_ods(
    shot: int,
    directory: str = "public",
    *,
    ids: str | list[str] | None = None,
    imas_version: str | None = None,
    consistency_check: bool = True,
) -> HSDSODS:
    """Open an occurrence-zero ODS that fetches only requested HSDS leaves.

    Unlike :func:`vaft.database.load`, this function does not create a staging
    directory and never invokes ``hsget``. ``ids`` can restrict discovery to a
    known set of IDS domains and avoids even listing the remote shot folder.
    """
    store = HSDSStore(shot, directory, ids=ids)
    return HSDSODS(
        store=store,
        imas_version=imas_version,
        consistency_check=consistency_check,
    )


__all__ = ["HSDSODS", "HSDSStore", "LazyODSClosedError", "open_ods"]
