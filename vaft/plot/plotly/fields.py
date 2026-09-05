"""Plotly drawing of :class:`~vaft.plot.models.Field2D`.

A filled or line contour map of the field with its overlays.  Plotly's
contour levels are uniform (``start``/``end``/``size``), so an explicit list
of levels is drawn at the uniform spacing between its first and last entry
-- exact for the flux-surface levels, an approximation for an irregular
list, and said so in the notebook.  The region mask, the secondary levels,
the colorbar title and the equal aspect all carry over.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..models import Field2D, GeometryLayer
from . import require_plotly
from ._style import color, plain_axis_label, plain_text, translate_style

__all__ = ["add_field_2d", "add_geometry_layer", "render_field_2d"]

_COLORSCALES = {"viridis": "Viridis", "hot_r": ("Hot", True), "hot": "Hot", "plasma": "Plasma",
                "inferno": "Inferno", "magma": "Magma", "rdbu": "RdBu", "coolwarm": "RdBu", "gray": "Greys"}


def colorscale(cmap: str) -> tuple[str, bool]:
    entry = _COLORSCALES.get(str(cmap).lower(), "Viridis")
    return (entry, False) if isinstance(entry, str) else entry


def _contours(levels: Any) -> dict[str, Any]:
    if levels is None:
        return {}
    if isinstance(levels, int):
        return {"ncontours": int(levels)}
    values = np.asarray(levels, dtype=float)
    if values.size < 2:
        return {}
    return {"contours": {"start": float(values.min()), "end": float(values.max()),
                         "size": float((values.max() - values.min()) / (values.size - 1))}}


def add_geometry_layer(figure: Any, layer: GeometryLayer, *, row: int | None = None, col: int | None = None) -> None:
    go = require_plotly()
    cell = {"row": row, "col": col} if row is not None else {}
    if layer.kind == "text":
        figure.add_annotation(x=float(layer.r[0]), y=float(layer.z[0]), text=layer.label, showarrow=False,
                              font={"size": 9, "color": color(layer.style.get("color", "black"))}, **cell)
        return
    # Overlays name what they are on hover; as in the Matplotlib map, they
    # are not legend entries.
    r, z = np.asarray(layer.r, dtype=float), np.asarray(layer.z, dtype=float)
    if layer.kind == "polygon" and r.size and (r[0] != r[-1] or z[0] != z[-1]):
        r, z = np.append(r, r[0]), np.append(z, z[0])
    props = translate_style(layer.style, has_line=layer.kind != "points")
    if layer.kind == "points":
        props["mode"] = "markers"
    figure.add_trace(go.Scatter(x=r, y=z, name=layer.label or None, showlegend=False, hoverinfo="name",
                                meta={"vaft": "overlay", "label": layer.label}, **props), **cell)


def add_field_2d(
    figure: Any,
    model: Field2D,
    *,
    row: int | None = None,
    col: int | None = None,
    colorbar: bool = True,
    cmap: str = "viridis",
    x_title: bool = True,
    **style: Any,
) -> None:
    go = require_plotly()
    cell = {"row": row, "col": col} if row is not None else {}
    scale, reverse = colorscale(cmap)
    values = np.asarray(model.values, dtype=float)
    if model.secondary_levels:
        secondary = np.asarray(model.secondary_levels, dtype=float)
        figure.add_trace(go.Contour(
            x=model.r, y=model.z, z=values, showscale=False, hoverinfo="skip",
            contours={"coloring": "none", "start": float(secondary.min()), "end": float(secondary.max()),
                      "size": float((secondary.max() - secondary.min()) / max(secondary.size - 1, 1))},
            line={"color": "rgb(170,170,170)", "width": 0.7}, meta={"vaft": "secondary"},
        ), **cell)
    if model.region is not None:
        values = np.where(model.region, values, np.nan)
    bar: dict[str, Any] = {"title": {"text": plain_axis_label(model.value_label), "side": "right"}}
    if cell:
        # The colorbar sits beside its own cell, not beside the whole figure.
        subplot = figure.get_subplot(row, col)
        x0, x1 = subplot.xaxis.domain
        y0, y1 = subplot.yaxis.domain
        bar.update(x=x1 + 0.01, xanchor="left", y=(y0 + y1) / 2, len=y1 - y0, thickness=12)
    contour = go.Contour(
        x=model.r, y=model.z, z=values, colorscale=scale, reversescale=reverse,
        showscale=bool(colorbar and model.colorbar), colorbar=bar,
        meta={"vaft": "field"}, **_contours(model.contour_levels),
    )
    contour.contours.coloring = "fill" if model.filled else "lines"
    if not model.filled:
        contour.line.width = 1.5
    figure.add_trace(contour, **cell)
    for layer in model.overlays:
        add_geometry_layer(figure, layer, **cell)
    figure.update_xaxes(title_text=plain_text(model.x_label) if x_title else None, **cell)
    yaxis: dict[str, Any] = {"title_text": plain_text(model.y_label)}
    if model.aspect_equal:
        anchor = "x"
        if cell:
            subplot = figure.get_subplot(row, col)
            anchor = subplot.yaxis.anchor or "x"
        yaxis.update(scaleanchor=anchor, scaleratio=1)
    figure.update_yaxes(**yaxis, **cell)


def render_field_2d(model: Field2D, *, show: bool = False, **style: Any) -> Any:
    go = require_plotly()
    if not isinstance(model, Field2D):
        raise TypeError(f"expected a vaft.plot.models.Field2D; got {type(model).__name__}.")
    figure = go.Figure()
    add_field_2d(figure, model, **style)
    figure.update_layout(title={"text": plain_text(model.title)} if model.title else None, template="plotly_white")
    if show:
        figure.show()
    return figure
