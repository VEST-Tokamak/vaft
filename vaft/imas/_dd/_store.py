"""The physical storage protocol behind a DD (#1132 §3, §7).

An :class:`EntryStore` is one physical IMAS Data Entry and knows only
``(IDS, occurrence)``. It returns what is stored -- including the string in
``ids_properties.name`` -- but never interprets a semantic name: resolving
``"kinetic-efit"`` to an occurrence is the DD layer's job, so the same store
works for a VAFT-managed entry and for an arbitrary third-party one.
"""

from __future__ import annotations

from typing import Any, Iterator, Mapping, Protocol, runtime_checkable


@runtime_checkable
class EntryStore(Protocol):
    """Read access to one physical IMAS Data Entry."""

    @property
    def dd_version(self) -> str: ...

    @property
    def writable(self) -> bool: ...

    def available_ids(self) -> tuple[str, ...]:
        """IDS names with at least one stored occurrence.

        Not in the #1132 sketch, which has no way to ask a store what it holds
        short of probing every IDS in the Data Dictionary.
        """
        ...

    def has(self, ids_name: str, occurrence: int) -> bool: ...

    def get(self, ids_name: str, occurrence: int, *, lazy: bool = True) -> Any: ...

    def occurrences(self, ids_name: str) -> tuple[int, ...]: ...

    def close(self) -> None: ...


@runtime_checkable
class MutableEntryStore(EntryStore, Protocol):
    """An EntryStore that can also write single occurrences.

    ``put`` replaces exactly one occurrence and leaves its siblings alone.
    Remote publication (retry, master-last commit, sibling merge) is not a
    store concern: #1128/#163 own it.
    """

    def put(self, ids: Any, occurrence: int) -> None: ...

    def delete(self, ids_name: str, occurrence: int) -> None: ...


def stored_names(store: EntryStore, ids_name: str) -> dict[int, str]:
    """``{occurrence: ids_properties.name}`` for every stored occurrence.

    Uses the store's own ``occurrence_names`` when it has one (one listing call
    on a native entry); otherwise reads each occurrence lazily. ``""`` means
    unnamed. The strings are returned verbatim -- nothing here decides what a
    name means.
    """
    names = getattr(store, "occurrence_names", None)
    if callable(names):
        return {int(occ): str(value) for occ, value in names(ids_name).items()}
    return {
        occ: str(store.get(ids_name, occ, lazy=True).ids_properties.name)
        for occ in store.occurrences(ids_name)
    }


class StoreSet(Mapping[str, EntryStore]):
    """Named physical stores owned by one DD.

    During the source-separated migration one logical Data Entry spans several
    physical entries (``main``, ``main/chease``, ``kinetic-efit``, ...); after
    #1128 consolidation it is normally just ``main``. Responsibilities are
    lookup, lifetime ownership and close -- nothing semantic.
    """

    def __init__(self, stores: Mapping[str, EntryStore] | None = None, **named: EntryStore):
        merged = dict(stores or {})
        overlap = set(merged) & set(named)
        if overlap:
            raise ValueError(f"store names given twice: {sorted(overlap)}")
        merged.update(named)
        if not merged:
            raise ValueError("a StoreSet needs at least one store")
        for name, store in merged.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"store names must be non-empty strings, got {name!r}")
            if not isinstance(store, EntryStore):
                raise TypeError(
                    f"store {name!r} does not implement the EntryStore protocol: "
                    f"{type(store).__name__}"
                )
        self._stores = merged
        self._closed = False

    @classmethod
    def of(cls, stores: "StoreSet | EntryStore | Mapping[str, EntryStore]") -> "StoreSet":
        """Coerce one store (named ``"main"``), a mapping, or a StoreSet."""
        if isinstance(stores, StoreSet):
            return stores
        if isinstance(stores, Mapping):
            return cls(stores)
        return cls({"main": stores})

    def __getitem__(self, name: str) -> EntryStore:
        if self._closed:
            raise RuntimeError("this StoreSet has been closed")
        try:
            return self._stores[name]
        except KeyError:
            raise KeyError(
                f"no store named {name!r}; stores: {sorted(self._stores)}"
            ) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._stores)

    def __len__(self) -> int:
        return len(self._stores)

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """Close every store; the first failure is re-raised after all run."""
        if self._closed:
            return
        self._closed = True
        failure: BaseException | None = None
        for store in self._stores.values():
            try:
                store.close()
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                failure = failure or exc
        if failure is not None:
            raise failure

    def __repr__(self) -> str:
        state = " closed" if self._closed else ""
        return f"<StoreSet{state} {sorted(self._stores)}>"


__all__ = ["EntryStore", "MutableEntryStore", "StoreSet", "stored_names"]
