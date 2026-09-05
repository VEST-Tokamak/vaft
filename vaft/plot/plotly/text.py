"""Plotly drawing of :class:`~vaft.plot.models.TextPanel`: lines of monospace text."""

from __future__ import annotations

from typing import Any

from ..models import TextPanel
from . import require_plotly
from ._style import cell_refs

__all__ = ["add_text_panel", "render_text_panel"]


def add_text_panel(figure: Any, model: TextPanel, *, row: int | None = None, col: int | None = None, **_: Any) -> None:
    go = require_plotly()
    cell = {"row": row, "col": col} if row is not None else {}
    # Plotly keeps an axes pair only while a trace uses it, and an annotation
    # placed against the domain of dropped axes lands on the first ones; an
    # invisible point keeps the cell's axes alive.
    figure.add_trace(go.Scatter(x=[0.0], y=[0.0], mode="markers", marker={"opacity": 0}, showlegend=False,
                                hoverinfo="skip", meta={"vaft": "anchor"}), **cell)
    text = "<br>".join(line.replace(" ", "&nbsp;") for line in model.lines)
    figure.add_annotation(text=text, **cell_refs(figure, row, col),
                          x=0.0, y=1.0, xanchor="left", yanchor="top", showarrow=False, align="left",
                          font={"family": "monospace", "size": 11})
    # Blank, not hidden: a hidden axis loses the domain the annotation is
    # placed against.
    hidden = {"showgrid": False, "zeroline": False, "showticklabels": False, "showline": False, "ticks": ""}
    figure.update_xaxes(**hidden, **cell)
    figure.update_yaxes(**hidden, **cell)


def render_text_panel(model: TextPanel, *, show: bool = False, **style: Any) -> Any:
    go = require_plotly()
    if not isinstance(model, TextPanel):
        raise TypeError(f"expected a vaft.plot.models.TextPanel; got {type(model).__name__}.")
    figure = go.Figure()
    add_text_panel(figure, model, **style)
    figure.update_layout(title={"text": model.title} if model.title else None, template="plotly_white")
    if show:
        figure.show()
    return figure
