"""Plotly drawing of :class:`~vaft.plot.models.Spectrogram` as a heatmap."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..models import Spectrogram
from . import require_plotly
from ._style import plain_axis_label
from .fields import colorscale

__all__ = ["add_spectrogram", "render_spectrogram"]


def _bar(figure: Any, model: Spectrogram, row: int | None, col: int | None) -> dict[str, Any]:
    bar: dict[str, Any] = {"title": {"text": plain_axis_label(model.value_label), "side": "right"}}
    if row is not None:
        subplot = figure.get_subplot(row, col)
        x0, x1 = subplot.xaxis.domain
        y0, y1 = subplot.yaxis.domain
        bar.update(x=x1 + 0.01, xanchor="left", y=(y0 + y1) / 2, len=y1 - y0, thickness=12)
    return bar


def add_spectrogram(
    figure: Any,
    model: Spectrogram,
    *,
    row: int | None = None,
    col: int | None = None,
    colorbar: bool = True,
    x_title: bool = True,
    **style: Any,
) -> None:
    go = require_plotly()
    cell = {"row": row, "col": col} if row is not None else {}
    scale, reverse = colorscale(style.get("cmap", model.cmap))
    figure.add_trace(go.Heatmap(
        x=np.asarray(model.time), y=np.asarray(model.frequency), z=np.asarray(model.magnitude),
        colorscale=scale, reversescale=reverse, showscale=colorbar,
        colorbar=_bar(figure, model, row, col), meta={"vaft": "spectrogram"},
    ), **cell)
    figure.update_xaxes(title_text=model.x_label if x_title else None, **cell)
    figure.update_yaxes(title_text=model.y_label,
                        range=[0.0, float(model.max_frequency)] if model.max_frequency is not None else None, **cell)


def render_spectrogram(model: Spectrogram, *, show: bool = False, **style: Any) -> Any:
    go = require_plotly()
    if not isinstance(model, Spectrogram):
        raise TypeError(f"expected a vaft.plot.models.Spectrogram; got {type(model).__name__}.")
    figure = go.Figure()
    add_spectrogram(figure, model, **style)
    figure.update_layout(title={"text": model.title} if model.title else None, template="plotly_white")
    if show:
        figure.show()
    return figure
