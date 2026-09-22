"""Migration-era storage bindings (#1132 §2).

Storage that predates self-describing occurrences keeps each variant in its own
physical entry, every one at occurrence 0 and unnamed -- ``main/chease`` holds
the CHEASE-refined equilibrium, but nothing inside it says so. A binding states
that fact explicitly, ``InstanceKey -> PhysicalLocator``, so a DD can present
the variant under its semantic name until the data carries the name itself.

The payload always wins: a binding may name an unnamed occurrence, but a stored
``ids_properties.name`` that disagrees with the binding is an error. Bindings
are a compatibility device, to be emptied by the single-``main`` consolidation
(#1128 Phase 4).
"""

from __future__ import annotations

from typing import Iterable, Mapping

from ._types import InstanceKey, PhysicalLocator


class StorageBindings:
    """A one-to-one map between semantic keys and physical locators."""

    def __init__(
        self,
        bindings: Mapping[InstanceKey, PhysicalLocator]
        | Iterable[tuple[InstanceKey, PhysicalLocator]]
        | None = None,
    ):
        items = bindings.items() if isinstance(bindings, Mapping) else (bindings or ())
        by_key: dict[InstanceKey, PhysicalLocator] = {}
        by_locator: dict[PhysicalLocator, InstanceKey] = {}
        for key, locator in items:
            if not isinstance(key, InstanceKey) or not isinstance(locator, PhysicalLocator):
                raise TypeError("bindings map InstanceKey -> PhysicalLocator")
            if key.ids != locator.ids:
                raise ValueError(f"binding {key} -> {locator} crosses IDS")
            if key in by_key and by_key[key] != locator:
                raise ValueError(f"{key} is bound twice ({by_key[key]}, {locator})")
            if locator in by_locator and by_locator[locator] != key:
                raise ValueError(f"{locator} is bound twice ({by_locator[locator]}, {key})")
            by_key[key] = locator
            by_locator[locator] = key
        self._by_key = by_key
        self._by_locator = by_locator

    @classmethod
    def single_entry(cls) -> "StorageBindings":
        """No bindings: names come from the stored occurrences alone."""
        return cls()

    def __bool__(self) -> bool:
        return bool(self._by_key)

    def __len__(self) -> int:
        return len(self._by_key)

    def locator_for(self, key: InstanceKey) -> PhysicalLocator | None:
        return self._by_key.get(key)

    def name_for(self, locator: PhysicalLocator) -> str | None:
        key = self._by_locator.get(locator)
        return None if key is None else key.name

    def keys(self, ids: str | None = None) -> tuple[InstanceKey, ...]:
        return tuple(sorted(k for k in self._by_key if ids is None or k.ids == ids))

    def __repr__(self) -> str:
        return f"<StorageBindings {len(self)} bindings>"


__all__ = ["StorageBindings"]
