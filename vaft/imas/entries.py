"""Turn native IMAS inputs into the ``(label, entry)`` pairs the recipes read.

Accepts an ``IDSToplevel``, an ``imas.DBEntry``, a :class:`~vaft.imas.IMASHandle`,
an HSDS lazy IMAS handle, a mapping of IDS name to toplevel, an
:class:`~vaft.imas.access.IDSEntry`, or a list/tuple of any of these; the
caller's order is the legend order.  Labels follow the shared rule in
:func:`vaft.plot.backend.entries.label_entries`: the data entry's pulse for
``"shot"``/``"pulse"`` -- which a bare toplevel does not carry, so a bare
toplevel is labelled by its position unless labels are given explicitly.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from vaft.plot.backend.entries import label_entries

from .access import IDSEntry, is_native_entry, is_native_ids

__all__ = ["normalize_entries"]


def _as_entry(item: Any) -> IDSEntry:
    if isinstance(item, IDSEntry):
        return item
    module = type(item).__module__.partition(".")[0]
    if module == "omas" or type(item).__name__ in ("ODS", "ODC") or any(
        cls.__name__ in ("ODS", "ODC") for cls in type(item).__mro__
    ):
        # An ODS is a Mapping too; it must be named before the mapping branch.
        raise TypeError(
            f"expected native IMAS input; got an OMAS {type(item).__name__}, which is "
            "plotted by vaft.omas.plot_*"
        )
    if is_native_ids(item) or is_native_entry(item) or isinstance(item, Mapping):
        return IDSEntry(item)
    if hasattr(item, "get") and hasattr(item, "ids"):
        return IDSEntry(item)
    raise TypeError(
        "expected an imas IDSToplevel, a DBEntry, a vaft.imas.IMASHandle, an HSDS lazy "
        "IMAS handle, a mapping of IDS name to IDSToplevel, or a list of them; got "
        f"{type(item).__name__}"
    )


def normalize_entries(
    source: Any, *, label: str | Sequence[str] = "shot"
) -> tuple[tuple[str, IDSEntry], ...]:
    """Return deterministic ``(label, entry)`` pairs for any supported input."""
    if isinstance(source, (list, tuple)):
        entries = [(str(position), _as_entry(item)) for position, item in enumerate(source)]
    else:
        entries = [("0", _as_entry(source))]
    return label_entries(entries, label)
