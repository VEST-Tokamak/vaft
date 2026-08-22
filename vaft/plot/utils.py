"""Deprecated helpers kept for the ``vaft.plot`` compatibility window.

Both functions here interpret OMAS objects, which is adapter work rather than
plotting work, so they now live in :mod:`vaft.omas`.  This module re-exports them
until the removal release; see :func:`vaft.plot.migration_table`.
"""

from __future__ import annotations

from typing import Any

__all__ = ["extract_labels_from_odc", "get_from_path"]


def get_from_path(obj: Any, path: str) -> Any:
    """Read a dotted ``a.b.c`` path out of a dict-like or attribute-like object.

    Deprecated: index the ODS/IDS directly instead.
    """
    for key in path.split("."):
        if isinstance(obj, dict):
            obj = obj.get(key)
        else:
            obj = getattr(obj, key, None)
        if obj is None:
            return None
    return obj


def extract_labels_from_odc(odc: Any, opt: str = "shot") -> list[str]:
    """Return one label per ODC entry.

    Deprecated alias of :func:`vaft.omas.extract_labels_from_odc`.
    """
    from vaft.omas._plot_recipes import extract_labels_from_odc as _implementation

    return _implementation(odc, opt)
