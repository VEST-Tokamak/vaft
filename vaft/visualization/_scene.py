"""Backend-free helpers shared by the PyVista and K3D adapters."""

from __future__ import annotations

from typing import Any

import numpy as np

from vaft.plot.models import Geometry3DLayer, Geometry3DLayers

__all__ = ["as_layers", "finite_runs", "layer_rgb"]


def as_layers(model: Any) -> Geometry3DLayers:
    """A :class:`Geometry3DLayers` from itself or from a single layer; anything else is refused."""
    if isinstance(model, Geometry3DLayer):
        return Geometry3DLayers((model,))
    if isinstance(model, Geometry3DLayers):
        return model
    raise TypeError(
        f"expected a vaft.plot.models.Geometry3DLayers or Geometry3DLayer; got {type(model).__name__}. "
        "Build one from data with vaft.plot.extract(<3-D plot name>, source), "
        "e.g. vaft.plot.extract('machine_geometry3d', ods)."
    )


def finite_runs(layer: Geometry3DLayer) -> list[np.ndarray]:
    """Indices of each run of finite vertices: a NaN breaks a polyline, as Matplotlib draws it."""
    finite = np.isfinite(layer.x) & np.isfinite(layer.y) & np.isfinite(layer.z)
    runs: list[np.ndarray] = []
    start = None
    for index, ok in enumerate(np.append(finite, False)):
        if ok and start is None:
            start = index
        elif not ok and start is not None:
            runs.append(np.arange(start, index))
            start = None
    return runs


def layer_rgb(layer: Geometry3DLayer, fallback: str = "C0") -> tuple[float, float, float]:
    """The layer's colour as RGB in ``[0, 1]``; colour intents resolve without a theme."""
    from matplotlib.colors import to_rgb

    from vaft.plot.intent import resolve_color

    value = dict(layer.style).get("color", fallback)
    try:
        return tuple(float(channel) for channel in to_rgb(resolve_color(value, theme=None)))
    except ValueError:
        return tuple(float(channel) for channel in to_rgb(fallback))
