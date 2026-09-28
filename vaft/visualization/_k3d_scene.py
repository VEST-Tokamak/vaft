"""K3D rendering of 3-D scenes and the coil-phasing explorer (issue #1087).

K3D is the Jupyter-native interactive frontend: it draws points and lines
that VAFT computed and updates them in place when a control changes.  It
integrates nothing, solves nothing and filters nothing -- the explorer's
physics is :func:`vaft.process.coils_non_axisymmetric.phased_sector_currents`
and :func:`~vaft.process.coils_non_axisymmetric.biot_savart_filaments`, and
its controls are the :class:`~vaft.plot.controls.ControlSpec` /
:class:`~vaft.plot.navigation.ControlState` pair every interactive
:mod:`vaft.plot` view uses, drawn by the same ipywidgets layer.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vaft.plot.models import Geometry3DLayer

from ._optional import require_k3d
from ._scene import as_layers, finite_runs, layer_rgb

__all__ = ["coil_phase_explorer", "to_k3d"]

#: Point size of a ``kind="points"`` layer [m].
_POINT_SIZE = 0.02


def _hex(rgb: tuple[float, float, float]) -> int:
    red, green, blue = (int(round(255 * min(max(channel, 0.0), 1.0))) for channel in rgb)
    return (red << 16) | (green << 8) | blue


def _layer_object(k3d: Any, layer: Geometry3DLayer, *, point_size: float, width: float) -> Any | None:
    """One K3D object for a layer, or ``None`` when it has nothing finite to draw."""
    runs = finite_runs(layer)
    if not runs:
        return None
    kept = np.concatenate(runs)
    vertices = np.column_stack([layer.x, layer.y, layer.z])[kept].astype(np.float32)
    name = layer.group or layer.label or layer.kind
    color = _hex(layer_rgb(layer))
    if layer.kind == "points":
        return k3d.points(vertices, point_size=point_size, color=color, shader="3d", name=name)
    # Segment indices keep each finite run separate: a NaN breaks the line.
    segments, offset = [], 0
    for run in runs:
        segments += [(offset + i, offset + i + 1) for i in range(run.size - 1)]
        offset += run.size
    if not segments:
        return None
    return k3d.line(
        vertices, indices=np.asarray(segments, dtype=np.uint32), indices_type="segment",
        color=color, width=width, shader="simple", name=name,
    )


def _add_layers(plot: Any, k3d: Any, scene: Any, *, point_size: float, width: float) -> list[tuple[Geometry3DLayer, Any]]:
    drawn = []
    for layer in scene.layers:
        obj = _layer_object(k3d, layer, point_size=point_size, width=width)
        if obj is not None:
            plot += obj
            drawn.append((layer, obj))
    return drawn


def to_k3d(model: Any, plot: Any = None, *, point_size: float = _POINT_SIZE, width: float = 0.004) -> Any:
    """A 3-D scene as a K3D plot, one named object per layer; needs ``vaft[jupyter3d]``.

    ``model`` is a :class:`~vaft.plot.models.Geometry3DLayers` (e.g.
    ``vaft.plot.extract("machine_geometry3d", ods)``); ``plot`` is an existing
    ``k3d.Plot`` to draw into.  Objects are named by the layer's ``group``, so
    K3D's object panel lists subsystems.  Display it with ``plot.display()``.
    """
    k3d = require_k3d()
    scene = as_layers(model)
    if plot is None:
        plot = k3d.plot(name=scene.title or "vaft", grid_visible=False)
    plot.axes = [scene.x_label, scene.y_label, scene.z_label]
    _add_layers(plot, k3d, scene, point_size=point_size, width=width)
    return plot


def _diverging(values: Any, scale: float) -> np.ndarray:
    """``values/scale`` on a blue-white-red map, as packed ``0xRRGGBB`` colours."""
    from matplotlib import colormaps

    cmap = colormaps["RdBu_r"]
    normalised = 0.5 + 0.5 * np.clip(np.asarray(values, dtype=float) / (scale or 1.0), -1.0, 1.0)
    return np.array([_hex(cmap(float(v))[:3]) for v in np.atleast_1d(normalised)], dtype=np.uint32)


def _coil_turns(source: Any) -> dict[str, float]:
    """Coil name -> turns, as ``coils_non_axisymmetric`` stores them (1 when absent)."""
    from vaft.plot.backend.access import count, get

    turns = {}
    for index in range(count(source, "coils_non_axisymmetric.coil")):
        name = get(source, f"coils_non_axisymmetric.coil.{index}.name")
        value = get(source, f"coils_non_axisymmetric.coil.{index}.turns")
        if name is not None:
            turns[str(name).replace("/", "-")] = float(value) if value is not None else 1.0
    return turns


_FIELD_COMPONENTS = ("B_R", "B_phi", "B_Z", "|B|")


def _closed(layer: Geometry3DLayer) -> bool:
    ends = np.array([[layer.x[0], layer.y[0], layer.z[0]], [layer.x[-1], layer.y[-1], layer.z[-1]]])
    return layer.x.size > 1 and np.allclose(ends[0], ends[1])


def coil_phase_explorer(
    source: Any,
    *,
    coil_set: str | None = None,
    n: int = 1,
    phase_deg: int = 0,
    amplitude_a: float = 1.0e3,
    probe_r: float | None = None,
    probe_z: float = 0.0,
    probes: int = 90,
    show: bool = True,
) -> Any:
    """Explore the phasing of one non-axisymmetric coil set in K3D.

    Draws every coil of ``source``'s ``coils_non_axisymmetric`` and a ring of
    field probes at ``(probe_r, probe_z)``.  The controls -- toroidal mode
    number ``n`` (1-3), phase (0-345 degrees) and the field component --
    drive :func:`~vaft.process.coils_non_axisymmetric.phased_sector_currents`
    (``I_k = A cos(n phi_k + delta)`` per turn, ``phi_k`` the toroidal angle
    of each sector's centroid) and
    :func:`~vaft.process.coils_non_axisymmetric.biot_savart_filaments` with
    the stored ``turns``; the coils are coloured by their current and the
    probes by the field, updated in place.  Other sets carry no current.

    ``coil_set`` defaults to the first set.  ``probe_r`` defaults to 85 % of
    the set's smallest centroid radius -- inside the coil row, not a stated
    plasma edge.  Returns :class:`vaft.plot.renderers.interactive.Interactive`
    with the K3D plot as ``figure`` and the last evaluation in ``computed``;
    ``show=False`` builds without displaying (no widgets), for scripts and
    tests that drive ``result.state``.
    """
    k3d = require_k3d()
    import vaft.plot as vplot
    from vaft.plot.controls import ControlSpec
    from vaft.plot.navigation import ControlState
    from vaft.plot.renderers.interactive import Interactive, _ipywidgets_controls
    from vaft.process.coils_non_axisymmetric import biot_savart_filaments, phased_sector_currents

    scene = vplot.extract("coil_3d_geometry3d", source)
    sets: dict[str, list[Geometry3DLayer]] = {}
    for layer in scene.layers:
        parts = layer.group.split("/")
        sets.setdefault(parts[1] if len(parts) > 1 else layer.group, []).append(layer)
    if coil_set is None:
        coil_set = next(iter(sets))
    if coil_set not in sets:
        raise ValueError(f"coil_set must be one of {', '.join(sets)}; got {coil_set!r}")
    driven = sets[coil_set]
    turns_by_name = _coil_turns(source)
    turns = np.array([turns_by_name.get(layer.group.split("/")[-1], 1.0) for layer in driven])
    # The closing vertex repeats the first; counting it twice would bias the angle.
    centroids = np.array([np.mean((layer.x + 1j * layer.y)[:-1] if _closed(layer) else layer.x + 1j * layer.y)
                          for layer in driven])
    phi_sector = np.angle(centroids)
    filaments = [np.column_stack([layer.x, layer.y, layer.z]) for layer in driven]
    if probe_r is None:
        probe_r = 0.85 * float(np.min(np.abs(centroids)))
    angle = np.linspace(0.0, 2.0 * np.pi, int(probes), endpoint=False)
    probe_xyz = np.column_stack([probe_r * np.cos(angle), probe_r * np.sin(angle), np.full(angle.size, probe_z)])

    plot = k3d.plot(name=f"{coil_set} coil phasing", grid_visible=False)
    plot.axes = [scene.x_label, scene.y_label, scene.z_label]
    drawn = {id(layer): obj for layer, obj in _add_layers(plot, k3d, scene, point_size=_POINT_SIZE, width=0.004)}
    driven_ids = {id(layer) for layer in driven}
    for layer in scene.layers:
        if id(layer) not in driven_ids and id(layer) in drawn:
            drawn[id(layer)].color = 0xBBBBBB
            drawn[id(layer)].opacity = 0.35
    probe_points = k3d.points(probe_xyz.astype(np.float32), point_size=0.03, shader="3d", name="field probes",
                              colors=np.zeros(angle.size, dtype=np.uint32))
    plot += probe_points
    caption = k3d.text2d("", position=(0.02, 0.02), size=0.9, is_html=True, name="state")
    plot += caption

    controls = (
        ControlSpec("n", "range", "n", default=int(n), options=(1, 3, 1)),
        ControlSpec("phase_deg", "range", "phase [deg]", default=int(phase_deg), options=(0, 345, 15)),
        ControlSpec("field", "choice", "probe field", default="B_R", options=_FIELD_COMPONENTS),
    )
    state = ControlState(controls)
    computed: dict[str, Any] = {}

    def update(current: Any = state) -> None:
        mode, phase = current["n"], np.deg2rad(current["phase_deg"])
        currents = phased_sector_currents(phi_sector, mode, phase, amplitude_a)
        field = biot_savart_filaments(filaments, currents * turns, probe_xyz)
        cos, sin = np.cos(angle), np.sin(angle)
        components = {
            "B_R": field[:, 0] * cos + field[:, 1] * sin,
            "B_phi": -field[:, 0] * sin + field[:, 1] * cos,
            "B_Z": field[:, 2],
            "|B|": np.linalg.norm(field, axis=1),
        }
        shown = components[current["field"]]
        scale = float(np.max(np.abs(shown))) or 1.0
        for layer, current_a in zip(driven, currents):
            if id(layer) in drawn:
                drawn[id(layer)].color = int(_diverging([current_a], amplitude_a)[0])
        probe_points.colors = _diverging(shown, scale)
        magnitude = current["field"] if current["field"] == "|B|" else f"|{current['field']}|"
        caption.text = (
            f"{coil_set}: n={mode}, phase={current['phase_deg']}&deg;, {amplitude_a:g} A/turn; "
            f"max {magnitude} = {scale * 1e4:.3g} G at R = {probe_r:.3f} m"
        )
        computed.update(currents=currents, field=field, component=shown, probe_xyz=probe_xyz, phi_sector=phi_sector)

    state.subscribe(update)
    update()
    widget = None
    if show:
        widget = _ipywidgets_controls(state, live=True)
        plot.display()
    result = Interactive(plot, None, state, controls, widget)
    result.computed = computed
    return result
