"""Plotly drawing of :class:`~vaft.plot.models.LineSeries`.

Reproduces :func:`vaft.plot.style.draw_series` and the legend policy: a
flagged channel is drawn grey and dashed with ``(invalid)`` in its name, a
flagged interval is shaded, uncertainty is error bars on a scatter-like
trace and a band otherwise, a lone sample gets a marker, and past
:data:`~vaft.plot.style.LEGEND_MAX_ENTRIES` traces the legend gives way to
a count note.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..models import LineSeries, Series
from ..style import LEGEND_MAX_ENTRIES, UNCERTAINTY_MODES, VALIDITY_MODES, _invalid_runs, _run_extent, trace_labels
from . import require_plotly
from ._style import INVALID_COLOR, cell_refs, plain_axis_label, plain_text, translate_style

__all__ = ["add_line_series", "add_series", "render_line_series"]


def _scatter_like(series: Series, props: dict[str, Any]) -> bool:
    return props.get("mode") == "markers"


def add_series(
    figure: Any,
    series: Series,
    *,
    name: str,
    row: int | None = None,
    col: int | None = None,
    uncertainty: str = "auto",
    validity: str = "show",
    legend: bool = True,
    **style: Any,
) -> bool:
    """Add one trace (plus its band or shading); returns whether it is legend-worthy."""
    go = require_plotly()
    if uncertainty not in UNCERTAINTY_MODES:
        raise ValueError(f"uncertainty must be one of {', '.join(UNCERTAINTY_MODES)}; got {uncertainty!r}")
    if validity not in VALIDITY_MODES:
        raise ValueError(f"validity must be one of {', '.join(VALIDITY_MODES)}; got {validity!r}")
    cell = {"row": row, "col": col} if row is not None else {}
    x, y, yerr = series.x, series.y, series.yerr
    invalid_channel = series.is_invalid_channel and validity != "ignore"
    mask = None if validity == "ignore" else series.valid_mask
    if validity == "mask":
        if invalid_channel:
            return False
        if mask is not None:
            if not mask.any():
                return False
            x, y = x[mask], y[mask]
            if yerr is not None:
                yerr = np.asarray(yerr)[..., mask]
            mask = None
    props = translate_style({**style, **dict(series.style)})
    if x.size == 1 and props.get("mode") == "lines":
        props["mode"] = "markers"
    if invalid_channel:
        props.setdefault("line", {})
        props["line"].setdefault("color", INVALID_COLOR)
        props["line"]["dash"] = "dash"
        name = f"{name} (invalid)" if name else name
    role = series.role or "measured"
    meta = {"vaft": "trace", "role": role, "entry": series.entry, "channel": series.channel}
    trace = go.Scatter(x=np.asarray(x), y=np.asarray(y), name=name or None, showlegend=bool(name) and legend, meta=meta, **props)
    draw_errorbars = yerr is not None and uncertainty in ("errorbar", "auto") and (
        uncertainty == "errorbar" or _scatter_like(series, props)
    )
    if yerr is not None and uncertainty != "none":
        spread = np.asarray(yerr, dtype=float)
        low, high = (spread[0], spread[1]) if spread.ndim == 2 else (spread, spread)
        if draw_errorbars:
            trace.update(error_y={"type": "data", "array": high, "arrayminus": low, "visible": True})
        elif uncertainty in ("band", "auto"):
            colour = props.get("line", {}).get("color")
            figure.add_trace(go.Scatter(x=np.asarray(x), y=np.asarray(y) - low, mode="lines",
                                        line={"width": 0}, showlegend=False, hoverinfo="skip",
                                        meta={"vaft": "band", "role": role}), **cell)
            figure.add_trace(go.Scatter(x=np.asarray(x), y=np.asarray(y) + high, mode="lines",
                                        line={"width": 0}, fill="tonexty", opacity=0.2,
                                        fillcolor=colour, showlegend=False, hoverinfo="skip",
                                        meta={"vaft": "band", "role": role}), **cell)
    figure.add_trace(trace, **cell)
    if mask is not None and not mask.all() and validity == "show":
        for start, end in _invalid_runs(mask):
            span = _run_extent(x, start, end)
            if span is not None:
                figure.add_vrect(x0=span[0], x1=span[1], fillcolor=INVALID_COLOR, opacity=0.25,
                                 line_width=0, layer="below", **cell)
    return bool(name)


def _apply_legend_policy(figure: Any, judged: int, labelled: int, legend: bool | None, title: str | None, cell: dict) -> None:
    """The Matplotlib legend policy, on a Plotly figure."""
    if legend is False or labelled == 0:
        figure.update_layout(showlegend=False)
        return
    if legend is True:
        figure.update_layout(showlegend=True, legend={"title": {"text": title or ""}})
        return
    if labelled <= 1:
        figure.update_layout(showlegend=False)
        return
    if judged > LEGEND_MAX_ENTRIES:
        figure.update_layout(showlegend=False)
        figure.add_annotation(text=f"{labelled} traces", **cell_refs(figure, cell.get("row"), cell.get("col")),
                              x=0.99, y=0.97, showarrow=False, xanchor="right", yanchor="top",
                              opacity=0.7, font={"size": 11})
        return
    figure.update_layout(showlegend=True, legend={"title": {"text": title or ""}})


def add_line_series(
    figure: Any,
    model: LineSeries,
    *,
    row: int | None = None,
    col: int | None = None,
    legend: bool | None = None,
    uncertainty: str = "auto",
    validity: str = "show",
    x_title: bool = True,
    **style: Any,
) -> None:
    """Draw a :class:`LineSeries` into a cell of ``figure`` (or the whole figure)."""
    cell = {"row": row, "col": col} if row is not None else {}
    labels, legend_title = trace_labels(model.series, panel_title=model.title)
    labelled = 0
    judged = 0
    for series, label in zip(model.series, labels):
        if add_series(figure, series, name=label, uncertainty=uncertainty, validity=validity,
                      legend=legend is not False, **cell, **style):
            labelled += 1
            if not series.role:
                judged += 1
    _apply_legend_policy(figure, judged, labelled, legend, legend_title, cell)
    figure.update_xaxes(title_text=plain_axis_label(model.x_label, model.x_unit) if x_title else None,
                        range=list(model.x_limits) if model.x_limits else None, **cell)
    yaxis: dict[str, Any] = {"title_text": plain_axis_label(model.y_label, model.y_unit)}
    if model.log_y:
        yaxis["type"] = "log"
    if model.y_limits is not None:
        yaxis["range"] = list(model.y_limits)
    if model.display is not None and model.display.notation == "scientific":
        yaxis["exponentformat"] = "e"
        yaxis["showexponent"] = "all"
    figure.update_yaxes(**yaxis, **cell)


def render_line_series(model: LineSeries, *, show: bool = False, **style: Any) -> Any:
    """A :class:`LineSeries` as a Plotly figure."""
    go = require_plotly()
    if not isinstance(model, LineSeries):
        raise TypeError(f"expected a vaft.plot.models.LineSeries; got {type(model).__name__}.")
    figure = go.Figure()
    add_line_series(figure, model, **style)
    figure.update_layout(title={"text": plain_text(model.title)} if model.title else None, template="plotly_white")
    if show:
        figure.show()
    return figure
