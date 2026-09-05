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
Backends: ``"auto"`` (the default) picks the control that works where the
figure is shown -- a Matplotlib slider on a live canvas (a GUI window,
``ipympl``), an ipywidgets slider that redraws a static figure under Jupyter
or VS Code, and none at all in a script; ``"matplotlib"``, ``"ipywidgets"``
and ``"none"`` force one.  See :mod:`vaft.plot.environment`.
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

#: The time history drawn above the slice summary, with a shared time marker:
#: the measured plasma-current waveform with the reconstruction's prediction
#: at each slice (the G·1 overlay, section 15's "synthetic diagnostic values"),
#: falling back to the equilibrium's own Ip when no magnetics is stored.  The
#: stored equilibrium slices are marked on it, so the reader sees where a
#: reconstruction exists before moving the selection.  Each entry is (name,
#: options, fallback name or None).
_HISTORIES = (
    ("plasma_current_time", {"synthetic": "equilibrium"}, "equilibrium_time_plasma_current"),
)


@dataclass
class InteractiveEquilibrium:
    """What :func:`plot_equilibrium_interactive` returns.

    ``figure`` and ``axes`` follow the renderer contract (``axes`` are the
    slice summary's panels in slot order, current for the selected slice: a
    slice that draws fewer panels gets fresh axes); ``navigator`` is the
    shared selection; ``history_axes`` carry the time marker; ``widget`` is
    the backend's control, or ``None`` for ``backend="none"``.
    """

    figure: Any
    slice_axes: Any
    navigator: SliceNavigator
    history_axes: tuple[Any, ...]
    widget: Any = None

    @property
    def axes(self) -> np.ndarray:
        return self.slice_axes.axes

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
    backend: str = "auto",
    show: bool = False,
    figsize: tuple[float, float] | None = None,
    **options: Any,
) -> InteractiveEquilibrium:
    """Explore one shot's equilibrium slices with a shared selected time.

    Same slice policy as the static overview: the representative slice unless
    ``time=`` (snapped to a stored slice) or ``time_slice=`` says otherwise.
    ``options`` reach the slice summary (``title=``, ``style=`` for the flux
    map, and the like).  A single ODS only: a navigator has one set of
    slices.  ``backend`` is one of :data:`BACKENDS`; the default ``"auto"``
    chooses by where the figure is shown.
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
    for name, history_options, fallback in _HISTORIES:
        for candidate in (name, fallback):
            if candidate is None:
                continue
            try:
                model = build_model(candidate, entries, _panel_member=True, **(history_options if candidate == name else {}))
                histories.append(_with_slice_markers(model, navigator))
                break
            except ValueError:
                continue
        else:
            histories.append(f"{name}\nnot available in this input")

    def build_slice(index: int):
        return build_model(
            "equilibrium_overview", entries, time_slice=index, _slice_reason="selected", **options
        )

    figure, slice_axes, history_axes, widget = render_slice_navigation(
        navigator, histories, build_slice, backend=backend, show=show, figsize=figsize
    )
    return InteractiveEquilibrium(
        figure=figure, slice_axes=slice_axes, navigator=navigator, history_axes=history_axes, widget=widget
    )


def _with_slice_markers(model: Any, navigator: SliceNavigator) -> Any:
    """Mark the usable equilibrium slices on a history's first measured trace.

    The marker sits on the waveform at each stored slice time (the value is
    read off the waveform, never invented), carries the ``slices`` role so
    it is styled as an overlay rather than a channel, and is drawn only when
    the history has a measured trace to sit on.
    """
    import dataclasses

    from vaft.plot.models import Series

    measured = next((trace for trace in model.series if not trace.role), None)
    if measured is None or measured.x.size < 2:
        return model
    times = navigator.times[list(navigator.usable)]
    order = np.argsort(measured.x)  # np.interp needs a monotonic abscissa
    marker = Series(
        x=times,
        y=np.interp(times, measured.x[order], measured.y[order]),
        label="equilibrium slices",
        style={"marker": "o", "linestyle": "none", "markerfacecolor": "none",
               "markeredgecolor": "0.25", "markersize": 5},
        entry=measured.entry,
        role="slices",
    )
    return dataclasses.replace(model, series=(*model.series, marker))


def _scalar(raw: Any) -> float:
    try:
        return float(np.asarray(raw, dtype=float).ravel()[0])
    except (IndexError, TypeError, ValueError):
        return float("nan")
