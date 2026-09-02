"""Interactive equilibrium exploration (issue #261 §14-17).

:func:`plot_equilibrium_interactive` is a public entry point, not a view: it
puts the static slice summary of :func:`vaft.omas.plot_equilibrium_overview`
under the equilibrium time histories, with one selected slice shared by all
of them.  Moving the selection moves the time marker on every history,
redraws the 2-D flux, the profiles and the global quantities for the new
slice, and snaps to a stored slice on the way (§15, §16).

The scientific contract lives in :class:`vaft.plot.navigation.SliceNavigator`
and the Matplotlib side in :func:`vaft.plot.renderers.interactive.
render_slice_navigation`; this module only builds models from the ODS.
Backends (``"matplotlib"`` slider, the default; ``"ipywidgets"``; ``"none"``
to drive the navigator yourself) are chosen here and imported only by the
renderer, only when asked for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from vaft.plot.navigation import SliceNavigator
from vaft.plot.renderers.interactive import BACKENDS, render_slice_navigation

from ._plot_recipes import (
    _count,
    _get,
    _usable_slices,
    build_model,
    normalize_entries,
    resolve_time_slice,
)

__all__ = ["InteractiveEquilibrium", "plot_equilibrium_interactive", "BACKENDS"]

#: The time histories drawn above the slice summary, with a shared time marker.
_HISTORIES = ("equilibrium_time_plasma_current", "equilibrium_time_q95")


@dataclass
class InteractiveEquilibrium:
    """What :func:`plot_equilibrium_interactive` returns.

    ``figure`` and ``axes`` follow the renderer contract (``axes`` is the 2 x 2
    grid of the slice summary); ``navigator`` is the shared selection;
    ``history_axes`` carry the time marker; ``widget`` is the backend's
    control, or ``None`` for ``backend="none"``.
    """

    figure: Any
    axes: np.ndarray
    navigator: SliceNavigator
    history_axes: tuple[Any, ...]
    widget: Any = None

    def __iter__(self):
        # Unpacks like a renderer result plus the navigator.
        yield self.figure
        yield self.axes
        yield self.navigator


def plot_equilibrium_interactive(
    source: Any,
    *,
    time: float | None = None,
    time_slice: int | None = None,
    backend: str = "matplotlib",
    show: bool = False,
    figsize: tuple[float, float] | None = None,
    **options: Any,
) -> InteractiveEquilibrium:
    """Explore one shot's equilibrium slices with a shared selected time.

    Same slice policy as the static overview: the representative slice unless
    ``time=`` (snapped to a stored slice) or ``time_slice=`` says otherwise.
    ``options`` reach the slice summary (``title=`` and the like).  A single
    ODS only: a navigator has one set of slices.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {', '.join(BACKENDS)}; got {backend!r}")
    entries = normalize_entries(source, label="shot")
    if len(entries) != 1:
        raise ValueError(
            "plot_equilibrium_interactive explores one shot at a time; "
            f"got {len(entries)} entries"
        )
    _, ods = entries[0]
    total = _count(ods, "equilibrium.time_slice")
    times = np.array(
        [_scalar(_get(ods, f"equilibrium.time_slice.{i}.time")) for i in range(total)], dtype=float
    )
    initial, _, _ = resolve_time_slice(ods, time=time, time_slice=time_slice)
    navigator = SliceNavigator(times, usable=_usable_slices(ods), initial=initial)

    histories = []
    for name in _HISTORIES:
        try:
            histories.append(build_model(name, entries, _panel_member=True))
        except ValueError:
            histories.append(f"{name}\nnot available in this input")

    def build_slice(index: int):
        return build_model(
            "equilibrium_overview", entries, time_slice=index, _slice_reason="selected", **options
        )

    figure, slice_axes, history_axes, widget = render_slice_navigation(
        navigator, histories, build_slice, backend=backend, show=show, figsize=figsize
    )
    return InteractiveEquilibrium(
        figure=figure, axes=slice_axes, navigator=navigator, history_axes=history_axes, widget=widget
    )


def _scalar(raw: Any) -> float:
    try:
        return float(np.asarray(raw, dtype=float).ravel()[0])
    except (IndexError, TypeError, ValueError):
        return float("nan")
