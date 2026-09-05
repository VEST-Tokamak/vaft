"""Translate the Matplotlib-flavoured style of the models into Plotly's.

The view models carry ``style`` mappings in Matplotlib's vocabulary
(``color``, ``linestyle``, ``marker``, ``lw``, ``alpha``) because that is the
canonical renderer's; here they are read into Plotly ``line``/``marker``
properties.  Unknown keys are dropped rather than forwarded: a Plotly trace
rejects what it does not know.
"""

from __future__ import annotations

import re
from typing import Any

from ..display import unit_markup

__all__ = ["INVALID_COLOR", "cell_refs", "plain_axis_label", "translate_style"]


def cell_refs(figure: Any, row: int | None, col: int | None) -> dict[str, str]:
    """``xref``/``yref`` naming a subplot's own domain (or the paper without a cell)."""
    if row is None:
        return {"xref": "paper", "yref": "paper"}
    subplot = figure.get_subplot(row, col)
    x_name = subplot.yaxis.anchor or "x"
    y_name = subplot.xaxis.anchor or "y"
    return {"xref": f"{x_name} domain", "yref": f"{y_name} domain"}

#: Matplotlib's default cycle (tab10), for ``"C0"``-style colours.
_TAB10 = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
)
_TAB = {
    "tab:blue": "#1f77b4", "tab:orange": "#ff7f0e", "tab:green": "#2ca02c", "tab:red": "#d62728",
    "tab:purple": "#9467bd", "tab:brown": "#8c564b", "tab:pink": "#e377c2", "tab:gray": "#7f7f7f",
    "tab:grey": "#7f7f7f", "tab:olive": "#bcbd22", "tab:cyan": "#17becf",
    "k": "black", "w": "white", "r": "red", "g": "green", "b": "blue", "c": "cyan", "m": "magenta", "y": "yellow",
}
_DASH = {
    "-": "solid", "solid": "solid", "--": "dash", "dashed": "dash", ":": "dot", "dotted": "dot",
    "-.": "dashdot", "dashdot": "dashdot",
}
_SYMBOL = {
    "o": "circle", ".": "circle", "s": "square", "x": "x", "+": "cross", "^": "triangle-up",
    "v": "triangle-down", "<": "triangle-left", ">": "triangle-right", "d": "diamond", "D": "diamond",
    "*": "star", "p": "pentagon", "h": "hexagon", "|": "line-ns", "_": "line-ew",
}
#: The grey a flagged channel is drawn in (the Matplotlib renderers' "0.65").
INVALID_COLOR = "rgb(166,166,166)"


def color(value: Any) -> Any:
    """A Matplotlib colour spelling as a Plotly one."""
    if value is None:
        return None
    text = str(value)
    if re.fullmatch(r"C\d+", text):
        return _TAB10[int(text[1:]) % 10]
    if text in _TAB:
        return _TAB[text]
    try:
        grey = float(text)
    except ValueError:
        return text
    level = int(round(255 * min(max(grey, 0.0), 1.0)))
    return f"rgb({level},{level},{level})"


def translate_style(style: Any, *, has_line: bool = True) -> dict[str, Any]:
    """Plotly scatter properties for a model's style mapping.

    Returns ``line``, ``marker``, ``opacity`` and ``mode`` keys as needed.
    A ``linestyle`` of ``"none"`` draws markers only; a marker beside a line
    draws both.
    """
    style = dict(style or {})
    line: dict[str, Any] = {}
    marker: dict[str, Any] = {}
    out: dict[str, Any] = {}
    colour = color(style.get("color"))
    if colour is not None:
        line["color"] = colour
        marker["color"] = colour
    width = style.get("lw", style.get("linewidth"))
    if width is not None:
        line["width"] = float(width)
    linestyle = str(style.get("linestyle", style.get("ls", "-")))
    markers_only = linestyle.lower() in ("none", "") or not has_line
    if not markers_only and linestyle in _DASH:
        line["dash"] = _DASH[linestyle]
    symbol = style.get("marker")
    if symbol not in (None, "", "None", "none"):
        marker["symbol"] = _SYMBOL.get(str(symbol), "circle")
        size = style.get("markersize", style.get("ms"))
        if size is not None:
            marker["size"] = float(size) * 1.6
        if style.get("markerfacecolor") in ("none", "None"):
            marker["color"] = "rgba(0,0,0,0)"
            marker["line"] = {"color": color(style.get("markeredgecolor", colour)), "width": 1}
    if markers_only:
        out["mode"] = "markers"
        marker.setdefault("symbol", "circle")
    elif "symbol" in marker:
        out["mode"] = "lines+markers"
    else:
        out["mode"] = "lines"
    if style.get("alpha") is not None:
        out["opacity"] = float(style["alpha"])
    if line:
        out["line"] = line
    if marker:
        out["marker"] = marker
    return out


def plain_axis_label(label: str, unit: str = "") -> str:
    """``"Label [unit]"`` with exponents as HTML superscripts, for Plotly."""
    label = str(label or "")
    unit = unit_markup(unit, flavor="html")
    if not unit:
        return label
    if not label:
        return f"[{unit}]"
    return f"{label} [{unit}]"
