"""IDS-aware plot discovery: native IMAS inputs in, the shared catalog out."""

from __future__ import annotations

from typing import Any

from vaft.plot.backend.discovery import describe_entries
from vaft.plot.discovery import PlotCatalog

from .entries import normalize_entries

__all__ = ["describe"]


def describe(
    source: Any = None,
    *,
    query: str | None = None,
    detail: bool = False,
    available_only: bool | None = None,
    **filters: Any,
) -> PlotCatalog:
    """The catalog of plots, evaluated against a native IMAS ``source`` when given.

    Same contract as :func:`vaft.omas.available_plots`: without ``source`` the
    registry plus what the recipes declare; with an IDS, a DBEntry, a handle
    or a list of them, the plots the input can draw (``available_only=False``
    keeps the rest with their reasons).
    """
    if source is None:
        return describe_entries(None, query=query, detail=detail, **filters)
    entries = normalize_entries(source, label="key")
    shots = [label for label, _ in normalize_entries(source, label="shot")]
    return describe_entries(
        entries, shots, query=query, detail=detail, available_only=available_only, **filters
    )
