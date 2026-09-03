"""ODS-aware plot discovery: OMAS inputs in, the shared catalog out (issue #262).

The instance-level answers live in :mod:`vaft.plot.backend.discovery`; this
module only turns an ``ODS``/``ODC``/list into entries and labels.
"""

from __future__ import annotations

from typing import Any

from vaft.plot.backend.discovery import (  # noqa: F401  (re-exported for callers)
    ANALYSIS_METHODS,
    INTERACTION,
    INTERACTION_ENTRY_POINTS,
    OVERVIEW_CONTENTS,
    describe_entries,
)
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
    """The catalog of plots, evaluated against ``source`` when one is given.

    Without ``source`` the records carry the registry's identity plus what the
    recipes declare (units, layouts, methods, overview members).  With an
    ``ODS``, ``ODC`` or list, each record also says whether the input can draw
    it and, for multi-channel plots, what the selection policy finds there.

    ``available_only`` defaults to ``True`` with a source -- the catalog then
    holds exactly the plots the input can draw, which is what
    ``vaft.omas.available_plots(ods)`` has always meant -- and pass ``False``
    to keep the unavailable ones with their reasons.  It is meaningless
    without a source.
    """
    if source is None:
        return describe_entries(None, query=query, detail=detail, **filters)
    entries = normalize_entries(source, label="key")
    shots = [label for label, _ in normalize_entries(source, label="shot")]
    return describe_entries(
        entries, shots, query=query, detail=detail, available_only=available_only, **filters
    )
