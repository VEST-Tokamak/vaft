"""Operational-space population plots with registered-boundary overlays (#944, #1425).

A population of plasma states -- one row per equilibrium slice, discharge or
database entry -- is drawn on a canonical projection of
:mod:`vaft.diagram` (``"hugill"``, ``"troyon"``, ``"beta_n_li"``, ...). The
boundaries come from :mod:`vaft.formula.boundaries` through the projection's
overlay plan; no limit is restated here.

The renderer takes a table, not an ODS or a database handle. Columns are named
by quantity identity (``murakami_parameter``, ``inverse_cylindrical_q``,
``normalized_beta``, ``internal_inductance_li3``, ...) and their units are
declared in ``table.attrs["units"]`` or ``units=``. A boundary is drawn only
when both plotted columns *are* the projection's quantities in its units; a
``q95`` column on a ``q_psi`` axis, or an undeclared unit, leaves the data on
the plot and the boundary off, with the reason in a warning and in
``ax.vaft_overlay``. It follows the renderer contract of :mod:`vaft.plot`
(``ax=None``, ``show=False``, return ``(Figure, Axes)``).
"""

from __future__ import annotations

import re
import warnings
from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from vaft.diagram._op_space import OperationalProjection, OverlayPlan, get_projection, overlay_plan
from vaft.formula import boundaries as _b

__all__ = ["operational_space_population", "population_overlay"]

#: Categorical slots in fixed order (the validated palette of vaft.plot.population).
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7")
MARKERS = ("o", "s", "^", "D", "v", "P", "X")
BOUNDARY_COLORS = ("#1a1a19", "#a3442b", "#2f6f4f", "#5b4a9e")


def _projection(projection) -> OperationalProjection:
    return get_projection(projection) if isinstance(projection, str) else projection


def _declared_units(table: pd.DataFrame, units: Optional[Mapping[str, str]]) -> dict:
    declared = dict(getattr(table, "attrs", {}).get("units", {}) or {})
    declared.update(units or {})
    return declared


def _axis_mismatch(column: str, quantity: _b.BoundaryQuantity, declared: Mapping[str, str]) -> Optional[str]:
    """Why this column cannot stand for the axis quantity, or None when it can."""
    if column != quantity.name:
        return f"column {column!r} is not the projection's {quantity.name!r}"
    unit = declared.get(column, "-" if quantity.unit == "-" else None)
    if unit is None:
        return f"unit of column {column!r} is not declared (expected {quantity.unit!r})"
    if unit != quantity.unit:
        return f"column {column!r} is in {unit!r}, the boundary axis is in {quantity.unit!r}"
    return None


def _finite(table: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(table[column], errors="coerce").to_numpy(float)


def population_overlay(table: pd.DataFrame, projection, *, x: Optional[str] = None, y: Optional[str] = None,
                       boundaries: Union[str, bool, Sequence[str]] = "default",
                       boundary_inputs: Optional[Mapping[str, float]] = None,
                       units: Optional[Mapping[str, str]] = None,
                       x_range: Optional[Tuple[float, float]] = None,
                       y_range: Optional[Tuple[float, float]] = None) -> OverlayPlan:
    """The boundaries a table's projection can carry, without drawing anything.

    Parameters
    ----------
    table : pandas.DataFrame
        One row per plasma state; columns named by quantity identity.
    projection : str or OperationalProjection
        Canonical projection, for example ``"hugill"``.
    x, y : str, optional
        Columns to plot; default to the projection's quantity names.
    boundaries : "default", False or sequence of str
        As in :func:`vaft.diagram._op_space.overlay_plan`.
    boundary_inputs : mapping, optional
        Values of boundary inputs that are not on an axis. A missing input is
        taken as the table median of the column of that name when the column
        exists and its unit is declared and matches; otherwise the boundary is
        omitted.
    units : mapping, optional
        Column units, added to ``table.attrs["units"]``.
    x_range, y_range : (float, float), optional
        Sampling ranges; default to the finite data span padded by 10 %.

    Returns
    -------
    OverlayPlan
        Curves to draw, and ``(key, reason)`` for each requested boundary left off.
    """
    proj = _projection(projection)
    x = x or proj.x.name
    y = y or proj.y.name
    declared = _declared_units(table, units)
    reasons = [r for r in (_axis_mismatch(x, proj.x, declared), _axis_mismatch(y, proj.y, declared)) if r]
    requested = (proj.default_boundaries if boundaries == "default"
                 else () if boundaries is False or boundaries is None else tuple(boundaries))
    if reasons:
        return OverlayPlan(proj.key, (), tuple((key, "; ".join(reasons)) for key in requested))

    def span(column, given):
        if given is not None:
            return given
        values = _finite(table, column)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return (0.0, 1.0)
        lo, hi = float(values.min()), float(values.max())
        pad = 0.1 * (hi - lo) if hi > lo else 0.1 * max(abs(hi), 1.0)
        return (min(0.0, lo - pad) if lo >= 0 else lo - pad, hi + pad)

    fixed = dict(boundary_inputs or {})
    for key in requested:
        entry = _b.get_boundary(key)
        for q in getattr(entry, "inputs", ()):
            if q.name in fixed or q.name not in table.columns or q.name in (x, y):
                continue
            if _axis_mismatch(q.name, q, declared) is None:
                values = _finite(table, q.name)
                if np.isfinite(values).any():
                    fixed[q.name] = float(np.nanmedian(values))
    return overlay_plan(proj, boundaries, x_range=span(x, x_range), y_range=span(y, y_range), fixed=fixed)


def _label(quantity: _b.BoundaryQuantity, column: str) -> str:
    if column != quantity.name:
        return column
    symbol = re.sub(r"(?<![\\A-Za-z])(beta|kappa|delta|epsilon|psi)(?![A-Za-z])", r"\\\1", quantity.symbol)
    symbol = re.sub(r"_([A-Za-z0-9]+)", r"_{\1}", symbol)
    unit = "" if quantity.unit == "-" else f" [{quantity.unit}]"
    return f"${symbol}${unit}"


def _boundary_label(curve: _b.BoundaryCurve) -> str:
    """Key and fixed inputs; the full citation stays on the registered entry."""
    fixed = ", ".join(f"{k} = {v:.3g}" for k, v in curve.fixed.items())
    return curve.key + (f" ({fixed})" if fixed else "")


def _shade_forbidden(ax, curve: _b.BoundaryCurve, color: str) -> None:
    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    x, y = curve.x, curve.y
    if curve.allowed_side == "below":
        ax.fill_between(x, y, yhi, color=color, alpha=0.06, linewidth=0, zorder=0)
    elif curve.allowed_side == "above":
        ax.fill_between(x, ylo, y, color=color, alpha=0.06, linewidth=0, zorder=0)
    elif curve.allowed_side == "left":
        ax.fill_betweenx(y, x, xhi, color=color, alpha=0.06, linewidth=0, zorder=0)
    elif curve.allowed_side == "right":
        ax.fill_betweenx(y, xlo, x, color=color, alpha=0.06, linewidth=0, zorder=0)


def operational_space_population(table: pd.DataFrame, projection, *, x: Optional[str] = None,
                                 y: Optional[str] = None, color: Optional[str] = None,
                                 marker: Optional[str] = None, hollow: Sequence = (),
                                 boundaries: Union[str, bool, Sequence[str]] = "default",
                                 boundary_inputs: Optional[Mapping[str, float]] = None,
                                 units: Optional[Mapping[str, str]] = None, cmap: str = "viridis",
                                 title: Optional[str] = None, ax=None, show: bool = False):
    """Scatter a population on a canonical operational-space projection.

    Parameters
    ----------
    table : pandas.DataFrame
        One row per state. Columns are quantity identities; units in
        ``table.attrs["units"]`` or ``units=``.
    projection : str or OperationalProjection
        Canonical projection key, see ``vaft.diagram._op_space.list_projections()``.
    x, y : str, optional
        Columns; default to the projection's quantities. Another column is
        plotted as data but carries no boundary.
    color : str, optional
        Column for colour: numeric columns use ``cmap`` with a colour bar,
        others the categorical palette.
    marker : str, optional
        Categorical column for marker shape (for example an EFIT quality label).
    hollow : sequence
        Values of ``marker`` drawn as open markers.
    boundaries : "default", False or sequence of str
        Registered boundaries to overlay.
    boundary_inputs : mapping, optional
        Fixed values for boundary inputs that are not on an axis.
    units : mapping, optional
        Column units.
    cmap : str
        Colormap for a numeric ``color`` column.
    title : str, optional
        Axes title; defaults to the projection's title.
    ax : matplotlib.axes.Axes, optional
        Target axes.
    show : bool
        Call ``plt.show()``.

    Returns
    -------
    (Figure, Axes)
        ``ax.vaft_overlay`` holds the :class:`OverlayPlan` that was drawn.
    """
    import matplotlib.pyplot as plt

    proj = _projection(projection)
    x = x or proj.x.name
    y = y or proj.y.name
    missing = [c for c in (x, y, color, marker) if c and c not in table.columns]
    if missing:
        raise KeyError(f"table has no column(s) {missing}; columns are {list(table.columns)}")
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
    else:
        fig = ax.figure

    xs, ys = _finite(table, x), _finite(table, y)
    ok = np.isfinite(xs) & np.isfinite(ys)
    rows = table.loc[ok]
    xs, ys = xs[ok], ys[ok]

    groups = [(None, np.ones(len(rows), bool))]
    if marker:
        labels = rows[marker].astype(object).where(rows[marker].notna(), "unknown").astype(str)
        groups = [(name, (labels == name).to_numpy()) for name in pd.unique(labels)]
    hollow = {str(h) for h in hollow}

    numeric_color = color is not None and pd.api.types.is_numeric_dtype(rows[color])
    norm = None
    if numeric_color:
        cvals = pd.to_numeric(rows[color], errors="coerce").to_numpy(float)
        finite_c = cvals[np.isfinite(cvals)]
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=float(finite_c.min()) if finite_c.size else 0.0,
                         vmax=float(finite_c.max()) if finite_c.size else 1.0)
        colormap = plt.get_cmap(cmap)
    elif color is not None:
        cats = rows[color].astype(object).where(rows[color].notna(), "unknown").astype(str)
        palette = {name: CATEGORICAL[i % len(CATEGORICAL)] for i, name in enumerate(pd.unique(cats))}

    mappable = None
    for gi, (name, mask) in enumerate(groups):
        if not mask.any():
            continue
        shape = MARKERS[gi % len(MARKERS)]
        open_marker = name is not None and name in hollow
        if numeric_color:
            col = colormap(norm(cvals[mask]))
        elif color is not None:
            col = [palette[c] for c in cats[mask]]
        else:
            col = CATEGORICAL[0]
        kw = dict(marker=shape, s=34, linewidths=1.1, zorder=3)
        if open_marker:
            kw.update(facecolors="none", edgecolors=col)
        else:
            kw.update(c=col if isinstance(col, str) else np.asarray(col), edgecolors="white", linewidths=0.4)
        sc = ax.scatter(xs[mask], ys[mask], label=(f"{marker}={name}" if name is not None else None), **kw)
        if numeric_color and not open_marker:
            mappable = sc
    if numeric_color:
        from matplotlib.cm import ScalarMappable
        sm = mappable if mappable is not None else ScalarMappable(norm=norm, cmap=colormap)
        if mappable is not None:
            mappable.set_cmap(colormap)
            mappable.set_norm(norm)
        fig.colorbar(sm, ax=ax, label=color)
    if color is not None and not numeric_color:
        for name, c in palette.items():
            ax.scatter([], [], color=c, marker="o", s=30, label=f"{color}={name}")

    plan = population_overlay(table, proj, x=x, y=y, boundaries=boundaries, boundary_inputs=boundary_inputs,
                              units=units, x_range=ax.get_xlim() if len(xs) else None,
                              y_range=ax.get_ylim() if len(ys) else None)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    for i, curve in enumerate(plan.curves):
        c = BOUNDARY_COLORS[i % len(BOUNDARY_COLORS)]
        ax.plot(curve.x, curve.y, color=c, linewidth=1.6, label=_boundary_label(curve), zorder=2)
        _shade_forbidden(ax, curve, c)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    for key, reason in plan.omitted:
        warnings.warn(f"boundary {key!r} not drawn on {proj.key!r}: {reason}", stacklevel=2)

    ax.set_xlabel(_label(proj.x, x))
    ax.set_ylabel(_label(proj.y, y))
    ax.set_title(title if title is not None else proj.title)
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize="x-small", frameon=False, loc="best")
    ax.vaft_overlay = plan
    if show:
        plt.show()
    return fig, ax
