"""Grid facts of a :class:`~vaft.plot.models.Panels`, shared by both backends (#1467).

Where each panel sits, which x-linked panels have another linked panel
below them, and how panels are marked.  Kept free of pyplot so the Plotly
path never imports it.
"""

from __future__ import annotations

from typing import Any


def panel_slots(model: Any) -> list[tuple[int, int, int, int]]:
    """``(row, col, rowspan, colspan)`` of every panel, spans or not."""
    if model.spans is not None:
        return [tuple(span) for span in model.spans]
    return [(slot // model.ncols, slot % model.ncols, 1, 1) for slot in range(len(model.models))]


def columns(slot: tuple[int, int, int, int]) -> set[int]:
    """The grid columns a panel covers."""
    return set(range(slot[1], slot[1] + slot[3]))


def linked_above(model: Any) -> set[int]:
    """Slots on an x link with another linked panel below them in a column they cover.

    Those keep no x tick labels or x title: the stacked-time-trace
    convention, applied to every column a spanning panel covers.
    """
    slots = panel_slots(model)
    hidden: set[int] = set()
    for axis_name, members in model.links:
        if axis_name != "x":
            continue
        for slot in members:
            row, _, rowspan, _ = slots[slot]
            if any(
                other != slot and slots[other][0] >= row + rowspan and columns(slots[other]) & columns(slots[slot])
                for other in members
            ):
                hidden.add(slot)
    return hidden


def panel_label(index: int) -> str:
    """``(a)``, ``(b)``, ... ``(z)``, then ``(27)``, ``(28)``, ...: slot ``index``'s mark."""
    return f"({chr(ord('a') + index)})" if index < 26 else f"({index + 1})"
