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

__all__ = ["ControlState", "SliceNavigator"]


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
        if not np.isfinite(float(time)):
            raise ValueError(f"time must be a finite number of seconds; got {time!r}")
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


class ControlState:
    """The current value of every control of one plot, with observers (issue #480).

    ``controls`` are the :class:`~vaft.plot.controls.ControlSpec` entries a
    record supports; ``values`` start at their defaults.  :meth:`set`
    validates against the spec and notifies only on a change, the same
    contract as :class:`SliceNavigator`; :meth:`as_options` is what a
    builder receives.  No widget toolkit is imported here either.
    """

    def __init__(self, controls: Sequence[Any], values: dict[str, Any] | None = None) -> None:
        self._controls = tuple(controls)
        self._by_name = {control.name: control for control in self._controls}
        self._values: dict[str, Any] = {control.name: control.default for control in self._controls}
        self._observers: list[Callable[["ControlState"], Any]] = []
        for name, value in (values or {}).items():
            self.set(name, value, notify=False)

    @property
    def controls(self) -> tuple[Any, ...]:
        return self._controls

    @property
    def values(self) -> dict[str, Any]:
        return dict(self._values)

    def __getitem__(self, name: str) -> Any:
        return self._values[name]

    def spec(self, name: str) -> Any:
        try:
            return self._by_name[name]
        except KeyError:
            raise KeyError(
                f"no control named {name!r}; controls: {', '.join(self._by_name) or 'none'}"
            ) from None

    def set(self, name: str, value: Any, *, notify: bool = True) -> bool:
        """Set one control; returns whether the value changed."""
        control = self.spec(name)
        value = control.validate(value)
        changed = value != self._values.get(name)
        self._values[name] = value
        if changed and notify:
            self._notify()
        return changed

    def update(self, **values: Any) -> bool:
        """Set several controls; observers run once if anything changed."""
        changed = False
        for name, value in values.items():
            changed = self.set(name, value, notify=False) or changed
        if changed:
            self._notify()
        return changed

    def as_options(self) -> dict[str, Any]:
        """The builder's keyword arguments for the current values.

        ``None`` and the ``"none"`` choice mean "leave the option out"; an
        explicit ``channels`` selection replaces the ``selection`` preset.
        Renderer-side controls (group ``"style"``) are left to :meth:`as_style`.
        """
        options: dict[str, Any] = {}
        for control in self._controls:
            value = self._values.get(control.name)
            if control.group == "style" or value is None or value == "none" or value == ():
                continue
            if not self._applies(control):
                continue
            options[control.name] = value
        if options.get("channels"):
            options["selection"] = list(options.pop("channels"))
        return options

    def as_style(self) -> dict[str, Any]:
        """The renderer's keyword arguments: the ``"style"`` group's values."""
        return {
            control.name: self._values[control.name]
            for control in self._controls
            if control.group == "style"
            and self._values.get(control.name) is not None
            and self._applies(control)
        }

    def _applies(self, control: Any) -> bool:
        """Whether ``control`` bears on what the other controls currently say.

        A control may declare ``applies_to={"field": ("psi",)}``: while the
        field control reads something else, its value is not sent to the
        builder at all, rather than being refused there (issue #483).
        """
        for name, accepted in (getattr(control, "applies_to", None) or {}).items():
            if name in self._values and self._values[name] not in accepted:
                return False
        return True

    def subscribe(self, callback: Callable[["ControlState"], Any]) -> Callable[[], None]:
        """Call ``callback(state)`` after every change; returns an unsubscribe."""
        self._observers.append(callback)

        def unsubscribe() -> None:
            if callback in self._observers:
                self._observers.remove(callback)

        return unsubscribe

    def _notify(self) -> None:
        for callback in list(self._observers):
            callback(self)

    def refresh(self) -> None:
        """Re-run every observer without changing anything."""
        self._notify()

    def bind_navigator(self, navigator: SliceNavigator, name: str = "time_slice") -> None:
        """Keep a slice control and a :class:`SliceNavigator` in step, both ways."""
        control = self.spec(name)

        def from_state(state: "ControlState") -> None:
            index = int(state[name])
            if index != navigator.selected:
                navigator.select_index(index)

        def from_navigator(nav: SliceNavigator) -> None:
            if control.validate(nav.selected) != self._values.get(name):
                self.set(name, nav.selected)

        self.subscribe(from_state)
        navigator.subscribe(from_navigator)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ControlState({self._values!r})"
