"""Lazy database namespace for raw and ODS access.

Notebook and workflow code primarily use ``vaft.database.load`` for IMAS/ODS
shot loading, while raw DAQ access is reached explicitly via
``vaft.database.raw``. Resolve flat attribute lookups in that order so the
public database namespace matches long-standing usage.
"""

from __future__ import annotations

from importlib import import_module

__all__ = ["raw", "ods", "ids", "utils"]


def __getattr__(name: str):
    if name in __all__:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module

    for module_name in ("ids", "ods", "raw", "utils"):
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
