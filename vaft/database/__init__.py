"""Lazy database namespace for raw and ODS access.

Notebook and workflow code primarily use ``vaft.database.load`` for IMAS/ODS
shot loading, while raw DAQ access is reached explicitly via
``vaft.database.raw``. Resolve flat attribute lookups in that order so the
public database namespace matches long-standing usage.
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "raw",
    "ods",
    "ids",
    "utils",
    "load",
    "save",
    "load_ids",
    "save_ids",
    "load_ods",
    "open_ods",
    "save_ods",
]


def load(shot, directory="public", **kwargs):
    """Load a shot using the historical ODS API, or native IDSs with ids_name.

    ``vaft.database.load(39915)`` and ``vaft.database.load(39915, "public")``
    have long meant "return an OMAS ODS".  Native IMAS IDS loading requires an
    explicit ``ids_name`` keyword to avoid confusing the second positional
    argument with the legacy HSDS directory.

    ``dd_version`` is an IDS-only concept (the ODS loader takes ``imas_version``
    instead), so passing it also routes to the native IDS loader. Because the
    IDS loader must know *which* IDS to return, ``dd_version`` without
    ``ids_name`` raises rather than silently falling back to the ODS path.
    """
    ids_name = kwargs.pop("ids_name", None)
    if ids_name is not None or kwargs.get("dd_version") is not None:
        if ids_name is None:
            raise ValueError(
                "dd_version is only meaningful for native IDS loading; "
                "pass ids_name=... to choose which IDS to load."
            )
        from .ids import load as _load_ids

        return _load_ids(shot, ids_name, directory=directory, **kwargs)

    from .ods import load_ods

    return load_ods(shot, directory=directory, **kwargs)


def save(obj, shot, *args, **kwargs):
    """Save an OMAS ODS or native IMAS IDS by dispatching on ``obj`` type.

    Native IMAS IDS toplevels (e.g. ``imas.IDSFactory(...).equilibrium()``) are
    routed to :func:`ids.save`, which accepts ``dd_version``; everything else is
    treated as an OMAS ODS and routed to :func:`ods.save_ods`. This lets a
    single ``save`` entry point handle both representations instead of raising a
    ``TypeError`` when an IDS is passed with IDS-only keywords like
    ``dd_version``.
    """
    if _is_imas_ids(obj):
        from .ids import save as _save_ids

        return _save_ids(obj, shot, *args, **kwargs)

    from .ods import save_ods

    return save_ods(obj, shot, *args, **kwargs)


def _is_imas_ids(obj) -> bool:
    """Return True if ``obj`` is a native IMAS IDS toplevel."""
    try:
        from imas.ids_toplevel import IDSToplevel
    except Exception:
        return False

    return isinstance(obj, IDSToplevel)


def load_ids(shot, ids_name, *args, **kwargs):
    """Load native IMAS IDS object(s)."""
    from .ids import load as _load_ids

    return _load_ids(shot, ids_name, *args, **kwargs)


def save_ids(ids_obj, shot, *args, **kwargs):
    """Save a native IMAS IDS object."""
    from .ids import save as _save_ids

    return _save_ids(ids_obj, shot, *args, **kwargs)


def load_ods(*args, **kwargs):
    """Load OMAS ODS data from IMAS-backed storage."""
    from .ods import load_ods as _load_ods

    return _load_ods(*args, **kwargs)


def open_ods(*args, **kwargs):
    """Open an ODS backed by direct, lazy HSDS dataset selections."""
    from .lazy_ods import open_ods as _open_ods

    return _open_ods(*args, **kwargs)


def save_ods(*args, **kwargs):
    """Save OMAS ODS data to IMAS-backed storage."""
    from .ods import save_ods as _save_ods

    return _save_ods(*args, **kwargs)


def __getattr__(name: str):
    if name in __all__:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module

    for module_name in ("ods", "lazy_ods", "ids", "raw", "utils"):
        try:
            module = import_module(f".{module_name}", __name__)
        except Exception:
            continue
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
