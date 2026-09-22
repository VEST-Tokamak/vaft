"""In-memory EntryStore (#1132 §6).

For unit tests, notebooks, and actor/simulation products that exist before
they are persisted. Keyed by physical ``(IDS, occurrence)`` like every store;
semantic names live in the stored IDS, not in the key.
"""

from __future__ import annotations

import copy
from typing import Any


class MemoryStore:
    """A writable in-memory Data Entry.

    ``put`` and ``get`` copy, so the store behaves like persistence (a DBEntry
    ``put`` does not keep a reference to the caller's object either): the live
    shared objects of a DD belong to the DD's cache, not to the store.
    """

    def __init__(self, dd_version: str | None = None):
        import imas

        self._factory = imas.IDSFactory(dd_version)
        self._data: dict[tuple[str, int], Any] = {}
        self._closed = False

    @property
    def dd_version(self) -> str:
        return self._factory.dd_version

    @property
    def factory(self):
        """IDS factory at this store's DD version, for building content."""
        return self._factory

    @property
    def writable(self) -> bool:
        return not self._closed

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("this MemoryStore has been closed")

    def available_ids(self) -> tuple[str, ...]:
        self._check_open()
        return tuple(sorted({ids for ids, _ in self._data}))

    def has(self, ids_name: str, occurrence: int) -> bool:
        self._check_open()
        return (ids_name, int(occurrence)) in self._data

    def occurrences(self, ids_name: str) -> tuple[int, ...]:
        self._check_open()
        return tuple(sorted(occ for ids, occ in self._data if ids == ids_name))

    def occurrence_names(self, ids_name: str) -> dict[int, str]:
        self._check_open()
        return {
            occ: str(ids.ids_properties.name)
            for (name, occ), ids in sorted(self._data.items())
            if name == ids_name
        }

    def get(self, ids_name: str, occurrence: int, *, lazy: bool = True) -> Any:
        # ``lazy`` is accepted for protocol parity; memory has nothing to defer.
        self._check_open()
        try:
            stored = self._data[(ids_name, int(occurrence))]
        except KeyError:
            raise KeyError(
                f"{ids_name} occurrence {occurrence} is not stored; stored "
                f"occurrences: {list(self.occurrences(ids_name))}"
            ) from None
        return copy.deepcopy(stored)

    def put(self, ids: Any, occurrence: int = 0) -> None:
        self._check_open()
        occurrence = int(occurrence)
        if occurrence < 0:
            raise ValueError(f"occurrence must be >= 0, got {occurrence}")
        if ids._dd_version != self.dd_version:
            import imas

            # Conversion happens at the storage boundary (#1132 §13).
            ids = imas.convert_ids(ids, self.dd_version, deepcopy=True)
        else:
            ids = copy.deepcopy(ids)
        self._data[(ids.metadata.name, occurrence)] = ids

    def delete(self, ids_name: str, occurrence: int) -> None:
        self._check_open()
        self._data.pop((ids_name, int(occurrence)), None)

    def close(self) -> None:
        self._closed = True
        self._data.clear()

    def __repr__(self) -> str:
        return f"<MemoryStore DD {self.dd_version}: {len(self._data)} occurrences>"


__all__ = ["MemoryStore"]
