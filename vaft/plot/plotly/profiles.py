"""Plotly drawing of :class:`~vaft.plot.models.Profile1D`."""

from __future__ import annotations

from typing import Any

from ..models import Profile1D
from ..style import trace_labels
from . import require_plotly
from ._style import color, plain_axis_label, plain_text
from .lines import _apply_legend_policy, add_series

__all__ = ["add_profile_1d", "render_profile_1d"]


def add_profile_1d(
    figure: Any,
    model: Profile1D,
    *,
    row: int | None = None,
    col: int | None = None,
    legend: bool | None = None,
    uncertainty: str = "auto",
    validity: str = "show",
    x_title: bool = True,
    **style: Any,
) -> None:
    cell = {"row": row, "col": col} if row is not None else {}
    labels, legend_title = trace_labels(model.series, panel_title=model.title)
    labelled = judged = 0
    for series, label in zip(model.series, labels):
        if add_series(figure, series, name=label, uncertainty=uncertainty, validity=validity,
                      legend=legend is not False, **cell, **style):
            labelled += 1
            if not series.role:
                judged += 1
    _apply_legend_policy(figure, judged, labelled, legend, legend_title, cell)
    for line in model.reference_lines:
        figure.add_vline(
            x=line.x,
            line={
                "dash": "dash" if line.style.get("linestyle") == "--" else "dot",
                "color": color(line.style.get("color", "#666666")),
                "width": line.style.get("linewidth", 1.0),
            },
            annotation_text=plain_text(line.label) if line.label else None,
            annotation_position="top",
            **cell,
        )
    figure.update_xaxes(title_text=plain_text(model.coordinate_label) if x_title else None,
                        range=list(model.x_limits) if model.x_limits else None, **cell)
    yaxis: dict[str, Any] = {"title_text": plain_axis_label(model.y_label, model.y_unit)}
    if model.display is not None and model.display.notation == "scientific":
        yaxis["exponentformat"] = "e"
    figure.update_yaxes(**yaxis, **cell)


def render_profile_1d(model: Profile1D, *, show: bool = False, **style: Any) -> Any:
    go = require_plotly()
    if not isinstance(model, Profile1D):
        raise TypeError(f"expected a vaft.plot.models.Profile1D; got {type(model).__name__}.")
    figure = go.Figure()
    add_profile_1d(figure, model, **style)
    figure.update_layout(title={"text": plain_text(model.title)} if model.title else None, template="plotly_white")
    if show:
        figure.show()
    return figure
