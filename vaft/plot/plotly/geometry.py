"""Plotly drawing of :class:`~vaft.plot.models.Geometry3DLayers` (issue #1087).

Each layer is one ``Scatter3d`` trace, a polyline or a point cloud.  A
labelled layer opens a legend group and the unlabelled layers after it join
that group -- the "label the first member of a set" convention the recipes
and the Matplotlib legend follow -- so one legend click hides the whole set.
The layer's ``group`` path rides along in the hover text and ``meta``.
The scene spans a common cube around the layers, as the Matplotlib
renderer's equal limits do, so toroidal placement and coil size read true.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..models import Geometry3DLayers
from . import require_plotly
from ._style import plain_text, translate_style

__all__ = ["add_geometry_3d_layers", "render_geometry_3d_layers"]

#: The marker symbols ``Scatter3d`` knows; anything else falls back to a circle.
_SYMBOLS_3D = frozenset({"circle", "circle-open", "cross", "diamond", "diamond-open", "square", "square-open", "x"})


def _props_3d(style: Any, *, points: bool) -> dict[str, Any]:
    props = translate_style(style, has_line=not points)
    marker = props.get("marker")
    if marker is not None:
        if marker.get("symbol") not in _SYMBOLS_3D:
            marker["symbol"] = "circle"
        marker.setdefault("size", 4.0)
    return props


def _scene_ranges(model: Geometry3DLayers) -> dict[str, Any]:
    if not model.layers:
        return {}
    stacked = np.concatenate([np.column_stack([layer.x, layer.y, layer.z]) for layer in model.layers])
    stacked = stacked[np.all(np.isfinite(stacked), axis=1)]
    if not stacked.size:
        return {}
    centre = (stacked.max(axis=0) + stacked.min(axis=0)) / 2.0
    half_span = float((stacked.max(axis=0) - stacked.min(axis=0)).max()) / 2.0 or 1.0
    return {
        axis: {"range": [float(centre[index] - half_span), float(centre[index] + half_span)]}
        for index, axis in enumerate(("xaxis", "yaxis", "zaxis"))
    }


def add_geometry_3d_layers(
    figure: Any,
    model: Geometry3DLayers,
    *,
    row: int | None = None,
    col: int | None = None,
    legend: bool | None = None,
    x_title: bool = True,
    **style: Any,
) -> None:
    """Draw every layer into ``figure`` (or its 3-D cell) and fit the scene."""
    go = require_plotly()
    cell = {"row": row, "col": col} if row is not None else {}
    show_legend = legend is not False
    legend_group = None
    for index, layer in enumerate(model.layers):
        if layer.label:
            legend_group = f"layer{index}:{layer.label}"
        points = layer.kind == "points"
        props = _props_3d({**style, **dict(layer.style)}, points=points)
        figure.add_trace(
            go.Scatter3d(
                x=layer.x, y=layer.y, z=layer.z,
                name=plain_text(layer.label) or layer.group or None,
                showlegend=bool(layer.label) and show_legend,
                legendgroup=legend_group,
                meta={"vaft": "trace", "kind": layer.kind, "group": layer.group, "label": layer.label},
                hovertemplate=f"{layer.group or layer.label}<br>x=%{{x:.3f}} m<br>y=%{{y:.3f}} m<br>z=%{{z:.3f}} m<extra></extra>",
                **props,
            ),
            **cell,
        )
    scene = {
        "aspectmode": "cube",
        **_scene_ranges(model),
    }
    for axis, title in (("xaxis", model.x_label), ("yaxis", model.y_label), ("zaxis", model.z_label)):
        scene.setdefault(axis, {})["title"] = {"text": plain_text(title)}
    figure.update_scenes(**scene, **cell)


def render_geometry_3d_layers(model: Geometry3DLayers, *, show: bool = False, **style: Any) -> Any:
    """A Plotly figure of one 3-D machine-coordinate scene."""
    go = require_plotly()
    if not isinstance(model, Geometry3DLayers):
        raise TypeError(f"expected a vaft.plot.models.Geometry3DLayers; got {type(model).__name__}.")
    figure = go.Figure()
    add_geometry_3d_layers(figure, model, **style)
    figure.update_layout(
        title={"text": plain_text(model.title)} if model.title else None,
        template="plotly_white", height=640,
    )
    if show:
        figure.show()
    return figure
