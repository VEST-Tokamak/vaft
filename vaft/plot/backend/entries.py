"""Labelling of ``(key, object)`` entries, shared by every namespace.

A namespace decides what its inputs are (an ODS, a DBEntry, a shot) and in
what order; the rule for naming each entry in a legend is the same for all of
them and lives here: the data entry's pulse (``"shot"``/``"pulse"``) or run
(``"run"``), the caller's key (``"key"``), or an explicit sequence of labels.
"""

from __future__ import annotations

from typing import Any, Sequence

from .access import get

__all__ = ["LABEL_OPTIONS", "label_entries"]

LABEL_OPTIONS = ("shot", "pulse", "run", "key")


def label_entries(
    entries: Sequence[tuple[str, Any]], label: str | Sequence[str] = "shot"
) -> tuple[tuple[str, Any], ...]:
    """Return ``(label, object)`` pairs for keyed ``entries``.

    ``label`` is one of :data:`LABEL_OPTIONS` or an explicit sequence, which
    must match the number of entries.  An entry whose data entry lacks the
    requested field keeps its key, so the order and count never change.
    """
    entries = [(str(key), obj) for key, obj in entries]
    if isinstance(label, (list, tuple)):
        supplied = [str(item) for item in label]
        if len(supplied) != len(entries):
            raise ValueError(
                f"received {len(supplied)} labels for {len(entries)} entries"
            )
        return tuple(zip(supplied, (obj for _, obj in entries)))
    if label == "key" or not entries:
        return tuple(entries)
    field_name = "run" if label == "run" else "pulse"
    labelled = []
    for key, obj in entries:
        try:
            value = get(obj, f"dataset_description.data_entry.{field_name}")
        except Exception:
            value = None
        labelled.append((key if value is None else str(value), obj))
    return tuple(labelled)
