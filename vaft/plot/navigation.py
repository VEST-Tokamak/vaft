"""Slice navigation: the scientific interaction contract (issue #261 §14-17).

Interactive exploration of an equilibrium means one thing scientifically:
there is one *selected slice*, every panel shows that slice, and moving the
selection moves them all together.  :class:`SliceNavigator` is that contract
and nothing else -- it knows the stored slice times, which of them are
usable, which one is selected, and whom to tell when that changes.  It never
imports a widget toolkit.

Backends -- a Matplotlib slider, an ipywidgets slider, a test that calls
:meth:`SliceNavigator.select` directly -- are adapters over this object.  A
requested time always snaps to the nearest *stored* slice: an equilibrium
exists only where the solver wrote one, so nothing is ever interpolated and
presented as a reconstruction (§13, §16).
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

__all__ = ["SliceNavigator"]


class SliceNavigator:
    """One selected slice among stored ones, with observers.

    ``times`` are the stored slice times in slice order; ``usable`` the
    indices a summary may stand on (default: all).  ``selected`` starts at
    ``initial`` and only ever holds a usable index.
    """

    def __init__(
        self,
        times: Sequence[float] | np.ndarray,
        *,
        usable: Sequence[int] | None = None,
        initial: int | None = None,
    ) -> None:
        self._times = np.asarray(times, dtype=float).ravel()
        indices = list(range(self._times.size)) if usable is None else sorted(set(int(i) for i in usable))
        indices = [i for i in indices if 0 <= i < self._times.size and np.isfinite(self._times[i])]
        if not indices:
            raise ValueError("a navigator needs at least one usable slice with a finite time")
        self._usable = tuple(indices)
        self._observers: list[Callable[["SliceNavigator"], Any]] = []
        self._selected = self._usable[len(self._usable) // 2]
        if initial is not None:
            self.select_index(initial)

    # -- state ----------------------------------------------------------------
    @property
    def times(self) -> np.ndarray:
        return self._times

    @property
    def usable(self) -> tuple[int, ...]:
        return self._usable

    @property
    def selected(self) -> int:
        """Index of the selected slice, always one of :attr:`usable`."""
        return self._selected

    @property
    def time(self) -> float:
        """Stored time of the selected slice."""
        return float(self._times[self._selected])

    @property
    def position(self) -> int:
        """Rank of the selected slice among the usable ones (0-based)."""
        return self._usable.index(self._selected)

    # -- selection ------------------------------------------------------------
    def nearest(self, time: float) -> int:
        """The usable slice nearest ``time``; ties go to the earlier slice."""
        candidates = np.asarray(self._usable)
        distance = np.abs(self._times[candidates] - float(time))
        return int(candidates[int(np.argmin(distance))])

    def select(self, time: float) -> tuple[int, float]:
        """Snap to the stored slice nearest ``time``; returns ``(index, time)``."""
        return self.select_index(self.nearest(time))

    def select_index(self, index: int) -> tuple[int, float]:
        """Select a stored slice by index; it must be usable."""
        index = int(index)
        if index not in self._usable:
            raise ValueError(
                f"slice {index} is not usable; usable slices are {list(self._usable)}"
            )
        changed = index != self._selected
        self._selected = index
        if changed:
            self._notify()
        return index, self.time

    def select_position(self, position: int) -> tuple[int, float]:
        """Select by rank among the usable slices (what a slider drives)."""
        position = int(np.clip(position, 0, len(self._usable) - 1))
        return self.select_index(self._usable[position])

    def step(self, delta: int) -> tuple[int, float]:
        """Move ``delta`` usable slices forward (or back), clamped at the ends."""
        return self.select_position(self.position + int(delta))

    # -- observers ------------------------------------------------------------
    def subscribe(self, callback: Callable[["SliceNavigator"], Any]) -> Callable[[], None]:
        """Call ``callback(navigator)`` after every change; returns an unsubscribe."""
        self._observers.append(callback)

        def unsubscribe() -> None:
            if callback in self._observers:
                self._observers.remove(callback)

        return unsubscribe

    def _notify(self) -> None:
        for callback in list(self._observers):
            callback(self)

    def refresh(self) -> None:
        """Re-run every observer without changing the selection."""
        self._notify()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"SliceNavigator(selected={self._selected}, t={self.time:.4f}, "
            f"usable={len(self._usable)}/{self._times.size})"
        )
