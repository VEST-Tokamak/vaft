"""How the plot explorer presents a discovery catalog (#1172).

Discovery decides what a plot *is* and whether it can be drawn here
(:class:`vaft.plot.PlotCapability`: identity, controls, availability and its
reason); this module only decides how that is shown.  It keeps no list of
plots, IDS or controls of its own and imports no widget toolkit:

* :func:`plot_label` -- the reader-facing label, ``view / quantity``, under
  the subject it belongs to; the canonical plot name stays the value.
* :func:`group_options` -- the selector's groups, the subject heading (with
  its aliases) over its plots, unavailable ones marked when they are shown.
* :func:`matching_names` -- the plots a search names: the registry's own
  query (so ``ip`` finds the plasma current) plus a plain text match on
  each record's name, label and subject.
* :func:`describe` -- the record as a short Markdown card.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

__all__ = ["UNAVAILABLE", "describe", "group_options", "matching_names", "plot_label"]

#: Appended to the label of a plot this source cannot draw.
UNAVAILABLE = " (unavailable)"


def plot_label(record: Any) -> str:
    """``view / quantity`` (``view`` alone when there is no quantity)."""
    view = getattr(record, "view", "") or ""
    quantity = getattr(record, "quantity", "") or ""
    label = " / ".join(part for part in (view, quantity) if part)
    return label or str(record.name)


def _heading(record: Any) -> str:
    heading = getattr(record, "heading", None)
    return str(heading) if heading else str(getattr(record, "subject", "") or "other")


def group_options(
    records: Iterable[Any], *, unavailable: bool = False, names: set[str] | None = None,
) -> dict[str, dict[str, str]]:
    """``{subject heading: {label: plot name}}`` in catalog order.

    ``unavailable`` keeps the plots this source cannot draw, marked
    :data:`UNAVAILABLE`; ``names`` keeps only those plots (a search).  Two
    plots sharing a label under one subject are told apart by their name.
    """
    groups: dict[str, dict[str, str]] = {}
    for record in records:
        if names is not None and record.name not in names:
            continue
        missing = getattr(record, "available", True) is False
        if missing and not unavailable:
            continue
        options = groups.setdefault(_heading(record), {})
        suffix = UNAVAILABLE if missing else ""
        label = plot_label(record) + suffix
        if label in options:
            label = f"{plot_label(record)} ({record.name}){suffix}"
        options[label] = record.name
    return groups


def matching_names(query: str | None, records: Iterable[Any] = ()) -> set[str] | None:
    """The plots a search names; ``None`` for no search.

    The registry's own query (``vaft.plot.available_plots(query=...)``:
    names, subjects, aliases such as ``ip``) plus every record whose name,
    label or subject contains the text, ignoring case.
    """
    query = (query or "").strip()
    if not query:
        return None
    from vaft.plot import available_plots

    names = set(available_plots(query=query, status=None).names())
    needle = query.lower()
    for record in records:
        text = " ".join((str(record.name), plot_label(record), _heading(record))).lower()
        if needle in text:
            names.add(record.name)
    return names


def _listing(values: Any) -> str:
    values = [str(value) for value in (values or ()) if str(value)]
    return ", ".join(f"`{value}`" for value in values) if values else "none"


def describe(record: Any) -> str:
    """The discovery record as Markdown: identity, availability, data, controls."""
    if record is None:
        return ""
    lines = [f"**{plot_label(record)}** -- {getattr(record, 'subject', '')}"]
    description = " ".join(str(getattr(record, "description", "") or "").split())
    if description:
        lines.append(description)
    if getattr(record, "available", True) is False:
        reason = getattr(record, "reason", "") or "the source lacks the data it needs"
        lines.append(f"**Unavailable here:** {reason}")
    facts = [
        f"Name: `{record.name}`",
        f"Function: `{getattr(record, 'function', '') or record.name}`",
        f"IDS: {_listing(getattr(record, 'ids', ()))}",
        f"Backends: {_listing(getattr(record, 'backends', ()))}",
        f"Controls: {_listing(getattr(record, 'controls', ()))}",
    ]
    interaction = getattr(record, "interaction", ())
    if interaction:
        facts.append(f"Interaction: {_listing(interaction)}")
    aliases = getattr(record, "aliases", ())
    if aliases:
        facts.append(f"Also found as: {_listing(aliases)}")
    lines.append("  \n".join(facts))
    return "\n\n".join(lines)
