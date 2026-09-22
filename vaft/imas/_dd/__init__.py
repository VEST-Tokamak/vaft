"""The DD layer: ``DD``, ``DDView`` and ``DDCollection`` over native IMAS (#1127).

Public names are re-exported from :mod:`vaft.imas`; this package is private so
its module layout can change. Nothing here imports OMAS: ODS projection
(#1131) is a separate adapter, and the core must run on native IMAS alone.
"""

from ._bindings import StorageBindings
from ._catalog import CatalogEntry, CatalogUpdate, Finding, InstanceCatalog
from ._collection import DDCollection
from ._dd import DD, CoherenceReport, DDView, Dependency, ValidationReport
from ._imas_store import IMASStore
from ._memory_store import MemoryStore
from ._provenance import format_reference, parse_reference, read_references, write_references
from ._store import EntryStore, MutableEntryStore, StoreSet
from ._types import (
    AmbiguousInstanceError,
    CatalogError,
    CatalogMismatchError,
    CoherenceError,
    InstanceInfo,
    InstanceKey,
    InstanceLookupError,
    InstanceUnavailableError,
    NoDefaultError,
    PhysicalLocator,
    UnknownInstanceError,
)

__all__ = [
    "AmbiguousInstanceError",
    "CatalogEntry",
    "CatalogError",
    "CatalogMismatchError",
    "CatalogUpdate",
    "CoherenceError",
    "CoherenceReport",
    "DD",
    "DDCollection",
    "DDView",
    "Dependency",
    "EntryStore",
    "Finding",
    "IMASStore",
    "InstanceCatalog",
    "InstanceInfo",
    "InstanceKey",
    "InstanceLookupError",
    "InstanceUnavailableError",
    "MemoryStore",
    "MutableEntryStore",
    "NoDefaultError",
    "PhysicalLocator",
    "StorageBindings",
    "StoreSet",
    "UnknownInstanceError",
    "ValidationReport",
    "format_reference",
    "parse_reference",
    "read_references",
    "write_references",
]
