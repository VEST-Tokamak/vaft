"""Native IMAS EntryStore (#1132 §4-§5).

Read access reuses :class:`vaft.imas.IMASHandle` -- one native access stack,
with its format detection, DD-version resolution and temporary-conversion
ownership -- rather than a second one. Writable access opens a native IMAS
HDF5/netCDF entry directly, since IMASHandle is read-only by design.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


def _native_uri(target: str | Path) -> str:
    text = str(target)
    if text.startswith("imas:") or text.endswith(".nc"):
        return text
    return "imas:hdf5?path=" + str(Path(text).expanduser().resolve())


class IMASStore:
    """An :class:`EntryStore` over one native IMAS Data Entry.

    ``IMASStore(source)`` opens any local artifact :func:`vaft.imas.load`
    accepts, read-only. :meth:`writable_entry` opens (or creates) a native IMAS
    HDF5 directory or netCDF file for writing.
    """

    def __init__(
        self,
        source: str | Path | Sequence[str | Path] | None = None,
        *,
        imas_version: str | None = None,
        handle: Any = None,
    ):
        from ...database._local import IMASHandle

        if (source is None) == (handle is None):
            raise TypeError("give exactly one of source= or handle=")
        self._handle = handle if handle is not None else IMASHandle(
            source, imas_version=imas_version
        )
        self._handle.open()
        self._entry: Any = None
        self._mode = "r"
        self._closed = False

    @classmethod
    def writable_entry(
        cls,
        target: str | Path,
        *,
        mode: str = "a",
        dd_version: str | None = None,
    ) -> "IMASStore":
        """Open a native IMAS entry for writing.

        ``mode="a"`` opens an existing entry (or creates it) and keeps every
        occurrence already there; ``"w"`` starts empty. ``dd_version`` is
        required to create a new entry and must match an existing one.
        """
        import imas

        if mode not in {"a", "w"}:
            raise ValueError(f"mode must be 'a' or 'w', got {mode!r}")
        store = cls.__new__(cls)
        store._handle = None
        store._mode = mode
        store._closed = False
        store._entry = imas.DBEntry(_native_uri(target), mode, dd_version=dd_version)
        store._entry.__enter__()
        return store

    # -- EntryStore ---------------------------------------------------------

    @property
    def dd_version(self) -> str:
        if self._handle is not None:
            return self._handle.info.imas_version
        return self._entry.dd_version

    @property
    def writable(self) -> bool:
        return self._entry is not None and not self._closed

    @property
    def handle(self):
        """The underlying read-only :class:`IMASHandle`, if any."""
        return self._handle

    def _check_open(self) -> None:
        # IMASHandle reopens itself on demand; a closed store must not, or a
        # stray call would leave a DBEntry (or a fresh conversion) nobody closes.
        if self._closed:
            raise RuntimeError("this IMASStore has been closed")

    def _dbentry(self) -> Any:
        self._check_open()
        return self._entry

    def available_ids(self) -> tuple[str, ...]:
        self._check_open()
        if self._handle is not None:
            return self._handle.ids
        import imas

        entry = self._dbentry()
        return tuple(
            name
            for name in imas.IDSFactory(entry.dd_version).ids_names()
            if entry.list_all_occurrences(name)
        )

    def occurrences(self, ids_name: str) -> tuple[int, ...]:
        self._check_open()
        if self._handle is not None:
            return self._handle.occurrences(ids_name)
        return tuple(sorted(int(v) for v in self._dbentry().list_all_occurrences(ids_name)))

    def occurrence_names(self, ids_name: str) -> dict[int, str]:
        self._check_open()
        if self._handle is not None:
            return self._handle.occurrence_names(ids_name)
        found, values = self._dbentry().list_all_occurrences(
            ids_name, "ids_properties/name"
        )
        return {int(occ): str(value) for occ, value in zip(found, values)}

    def has(self, ids_name: str, occurrence: int) -> bool:
        return int(occurrence) in self.occurrences(ids_name)

    def get(self, ids_name: str, occurrence: int, *, lazy: bool = True) -> Any:
        self._check_open()
        if self._handle is not None:
            return self._handle.get(ids_name, occurrence, lazy=lazy)
        return self._dbentry().get(ids_name, int(occurrence), lazy=lazy)

    def put(self, ids: Any, occurrence: int = 0) -> None:
        if not self.writable:
            raise PermissionError(
                "this IMASStore is read-only; open it with IMASStore.writable_entry()"
            )
        self._dbentry().put(ids, int(occurrence))

    def delete(self, ids_name: str, occurrence: int) -> None:
        if not self.writable:
            raise PermissionError("this IMASStore is read-only")
        self._dbentry().delete_data(ids_name, int(occurrence))

    def close(self) -> None:
        self._closed = True
        if self._handle is not None:
            self._handle.close()
        if self._entry is not None:
            self._entry.__exit__(None, None, None)
            self._entry = None

    def __enter__(self) -> "IMASStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        kind = "writable" if self.writable else "read-only"
        return f"<IMASStore {kind} DD {self.dd_version}>"


__all__ = ["IMASStore"]
