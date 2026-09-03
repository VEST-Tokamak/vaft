"""Path access for the extraction layer, dispatched on the data object.

Recipes read every value through :func:`get`, :func:`count`, :func:`has` and
:func:`array`, never through a data model's own API, so one recipe serves
every model.  The accessor for an object is chosen here: OMAS ``ODS`` (and
its lazy subclasses) and plain mappings go through :mod:`vaft.ods_access`;
a namespace that reads another model registers its accessor with
:func:`register_accessor` (``vaft.imas`` does so for its ``IDSEntry``).  A
raw native IMAS object is refused by name rather than read as empty: the
IMAS-DD path grammar looks alike, but an ``IDSToplevel`` does not answer it.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

import numpy as np

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


class PathAccessor(Protocol):
    """How one data model answers IMAS-DD dotted paths, without mutation."""

    def get(self, obj: Any, path: str, default: Any = None) -> Any: ...

    def count(self, obj: Any, path: str) -> int: ...

    def has(self, obj: Any, path: str) -> bool: ...


class _ODSAccessor:
    """OMAS ``ODS``/``ODC``/mapping access through :mod:`vaft.ods_access`."""

    def get(self, obj: Any, path: str, default: Any = None) -> Any:
        from vaft.ods_access import path_value

        return path_value(obj, path, default)

    def count(self, obj: Any, path: str) -> int:
        from vaft.ods_access import path_count

        return path_count(obj, path)

    def has(self, obj: Any, path: str) -> bool:
        from vaft.ods_access import path_exists

        return path_exists(obj, path)


ODS_ACCESSOR: PathAccessor = _ODSAccessor()

_REGISTERED: list[tuple[Callable[[Any], bool], PathAccessor]] = []


def register_accessor(predicate: Callable[[Any], bool], accessor: PathAccessor) -> None:
    """Route objects for which ``predicate`` holds to ``accessor``.

    Registered accessors are consulted in registration order before the
    default OMAS accessor; registering the same pair twice is a no-op.
    """
    if (predicate, accessor) not in _REGISTERED:
        _REGISTERED.append((predicate, accessor))


def _module_root(obj: Any) -> str:
    return type(obj).__module__.partition(".")[0]


def accessor_for(obj: Any) -> PathAccessor:
    """The accessor that reads ``obj``."""
    for predicate, accessor in _REGISTERED:
        if predicate(obj):
            return accessor
    if _module_root(obj) == "imas":
        raise TypeError(
            f"a native IMAS {type(obj).__name__} is read by vaft.imas, not through "
            "the OMAS accessor: plot it with vaft.imas.plot_* (or wrap it in "
            "vaft.imas.IDSEntry)"
        )
    return ODS_ACCESSOR


def get(obj: Any, path: str, default: Any = None) -> Any:
    """The value at ``path`` in ``obj``, or ``default``; never creates anything."""
    return accessor_for(obj).get(obj, path, default)


def count(obj: Any, path: str) -> int:
    """Length of the array of structures (or array) at ``path``; 0 when absent."""
    return accessor_for(obj).count(obj, path)


def has(obj: Any, path: str) -> bool:
    """Whether ``path`` holds a value in ``obj``."""
    return accessor_for(obj).has(obj, path)


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
