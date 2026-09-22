"""Identity types and errors of the DD layer (#1127, #1128, #1132).

Two identities are kept apart on purpose:

``InstanceKey(ids, name)``
    what an IDS occurrence *is* -- the semantic name stored in its own
    ``ids_properties.name`` (#1128);
``PhysicalLocator(store, ids, occurrence)``
    where it currently *lives* -- which physical Data Entry, which occurrence.

The same scientific instance may move between locators (a source-separated
HSDS layout today, one occurrence-native ``main`` entry later) without the key
changing, and an unnamed occurrence has a locator but no key.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class InstanceKey:
    """Semantic identity of one IDS instance: ``(ids, name)``."""

    ids: str
    name: str

    def __str__(self) -> str:
        return f"{self.ids}/{self.name}"


@dataclass(frozen=True, order=True)
class PhysicalLocator:
    """Physical location of one IDS occurrence: ``(store, ids, occurrence)``.

    ``store`` names an entry of the DD's :class:`~vaft.imas.StoreSet`; it is a
    deployment detail and never a scientific identity.
    """

    store: str
    ids: str
    occurrence: int

    def __str__(self) -> str:
        return f"{self.store}:{self.ids}:{self.occurrence}"


@dataclass(frozen=True)
class InstanceInfo:
    """One discovered IDS occurrence, as :meth:`DD.instances` reports it.

    ``name`` is the stored ``ids_properties.name``; ``None`` for an unnamed
    occurrence, which stays addressable by its number and is never given an
    invented name. ``name_source`` says where the name came from: ``"payload"``
    (the IDS itself, canonical) or ``"binding"`` (a migration-era
    :class:`~vaft.imas.StorageBindings` entry for storage that predates
    self-describing occurrences).
    """

    ids: str
    locator: PhysicalLocator
    name: str | None
    name_source: str | None = None

    @property
    def occurrence(self) -> int:
        return self.locator.occurrence

    @property
    def store(self) -> str:
        return self.locator.store

    @property
    def key(self) -> InstanceKey | None:
        return None if self.name is None else InstanceKey(self.ids, self.name)

    def __str__(self) -> str:
        label = self.name if self.name is not None else "<unnamed>"
        return f"{self.ids}[{self.occurrence}] {label} ({self.store})"


class InstanceLookupError(LookupError):
    """Base class: an IDS instance could not be resolved.

    Every subclass states what was requested and what exists instead; none of
    them is ever answered by substituting another instance.
    """


class UnknownInstanceError(InstanceLookupError):
    """No stored occurrence has this name, and no catalog registers it."""


class InstanceUnavailableError(InstanceLookupError):
    """The instance is registered (catalog or default policy) but absent here."""


class AmbiguousInstanceError(InstanceLookupError):
    """More than one stored occurrence answers to the request."""


class NoDefaultError(InstanceLookupError):
    """The IDS has no default-selection policy, so bare access cannot choose."""


class CatalogError(ValueError):
    """A deployment catalog is malformed or internally inconsistent."""


class CatalogMismatchError(ValueError):
    """Stored occurrence metadata disagrees with a catalog or storage binding."""


class CoherenceError(ValueError):
    """A state that must be coherent (the default view) is not."""


def describe_instances(infos) -> str:
    """One-line summary of discovered instances for error messages."""
    if not infos:
        return "none stored"
    return ", ".join(
        f"{info.name!r} (occurrence {info.occurrence}, {info.store})"
        if info.name is not None
        else f"unnamed occurrence {info.occurrence} ({info.store})"
        for info in infos
    )


__all__ = [
    "AmbiguousInstanceError",
    "CatalogError",
    "CatalogMismatchError",
    "CoherenceError",
    "InstanceInfo",
    "InstanceKey",
    "InstanceLookupError",
    "InstanceUnavailableError",
    "NoDefaultError",
    "PhysicalLocator",
    "UnknownInstanceError",
    "describe_instances",
]
