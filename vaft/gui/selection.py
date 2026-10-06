"""What the reader has selected, shared by every workspace (#1174).

The application shell hosts several workspaces; each keeps its own widgets,
but the scientific context -- which database namespace, which entries are
open, which time the reader is looking at -- is one value they all read and
any of them may change.  :class:`SelectionState` is that value with
observers, the same contract as :class:`vaft.plot.navigation.ControlState`:
it imports no widget toolkit, validates what it is given, and notifies only
on a change.

Every change names its ``origin`` (the workspace that made it), so a
workspace can tell its own echo from a selection made elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable

from .state import Source

__all__ = ["Selection", "SelectionState"]


@dataclass(frozen=True)
class Selection:
    """One state of the shared context.

    ``namespace`` is the database source a database entry is read from
    (``None``: the default source); ``sources`` the entries open now (sample
    shots, files or database shots, as :class:`~vaft.gui.state.Source`);
    ``time`` the selected time as the plot labels it (``None`` when the plot
    on screen has no time axis to select on).
    """

    namespace: str | None = None
    sources: tuple[Source, ...] = field(default_factory=tuple)
    time: str | None = None

    def __post_init__(self) -> None:
        sources = tuple(self.sources)
        if any(not isinstance(source, Source) for source in sources):
            raise TypeError("sources must be vaft.gui.Source values")
        object.__setattr__(self, "sources", sources)
        if self.namespace is not None:
            object.__setattr__(self, "namespace", str(self.namespace).strip() or None)

    @property
    def label(self) -> str:
        """The open entries in words, for a status line."""
        if not self.sources:
            return "nothing open"
        return ", ".join(source.label for source in self.sources)


Observer = Callable[[Selection, Any], Any]


class SelectionState:
    """The current :class:`Selection`, with observers.

    ``update(origin=..., **changes)`` replaces fields and calls every observer
    with ``(selection, origin)`` -- once, and only when something changed.
    """

    def __init__(self, selection: Selection | None = None) -> None:
        self._value = selection or Selection()
        self._observers: list[Observer] = []

    @property
    def value(self) -> Selection:
        return self._value

    def update(self, *, origin: Any = None, **changes: Any) -> bool:
        """Change fields of the selection; returns whether it changed."""
        unknown = set(changes) - {"namespace", "sources", "time"}
        if unknown:
            raise TypeError(f"a selection has no field {', '.join(sorted(unknown))}")
        new = replace(self._value, **changes)
        if new == self._value:
            return False
        self._value = new
        for callback in list(self._observers):
            callback(new, origin)
        return True

    def subscribe(self, callback: Observer) -> Callable[[], None]:
        """Call ``callback(selection, origin)`` after every change; returns an unsubscribe."""
        self._observers.append(callback)

        def unsubscribe() -> None:
            if callback in self._observers:
                self._observers.remove(callback)

        return unsubscribe

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"SelectionState({self._value!r})"
