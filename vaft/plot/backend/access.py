"""Path access for the extraction layer, dispatched on the data object.

Recipes read every value through :func:`get`, :func:`count`, :func:`has` and
:func:`array`, never through a data model's own API, so one recipe serves
every model.  The dispatch itself lives in :mod:`vaft.ods_access` -- the
matplotlib-free core every ``vaft.omas`` helper already reads through -- so a
helper and a recipe see the same accessor for the same object: OMAS ``ODS``
(and its lazy subclasses) and plain mappings through the ODS readers, and a
namespace that reads another model through the accessor it registered with
:func:`register_accessor` (``vaft.imas`` does so for its ``IDSEntry``).  A
raw native IMAS object is refused by name rather than read as empty.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vaft import ods_access as _core
from vaft.ods_access import ODS_ACCESSOR, PathAccessor, register_accessor

__all__ = [
    "ODS_ACCESSOR",
    "PathAccessor",
    "accessor_for",
    "array",
    "count",
    "get",
    "has",
    "register_accessor",
]


def accessor_for(obj: Any) -> PathAccessor:
    """The accessor that reads ``obj`` (:func:`vaft.ods_access.accessor_for`)."""
    return _core.accessor_for(obj)


def get(obj: Any, path: str, default: Any = None) -> Any:
    """The value at ``path`` in ``obj``, or ``default``; never creates anything."""
    return _core.accessor_for(obj).get(obj, path, default)


def count(obj: Any, path: str) -> int:
    """Length of the array of structures (or array) at ``path``; 0 when absent."""
    return _core.accessor_for(obj).count(obj, path)


def has(obj: Any, path: str) -> bool:
    """Whether ``path`` holds a value in ``obj``."""
    return _core.accessor_for(obj).has(obj, path)


def array(obj: Any, path: str) -> np.ndarray | None:
    """The value at ``path`` as a float array, or ``None`` when absent or empty."""
    value = get(obj, path)
    if value is None:
        return None
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    return result if result.size else None
