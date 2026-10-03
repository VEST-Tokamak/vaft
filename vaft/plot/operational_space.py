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
import textwrap
import warnings
from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from vaft.diagram._op_space import (
    OperationalProjection,
    OverlayPlan,
    get_projection,
    overlay_plan,
    requested_boundaries,
)
from vaft.formula import boundaries as _b
from vaft.plot.presentation import presented

__all__ = ["operational_space_population", "population_overlay"]

#: Categorical slots in fixed order (the validated palette of vaft.plot.population).
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7")
MARKERS = ("o", "s", "^", "D", "v", "P", "X")
MISSING_COLOR = "#b8b7ae"
#: categorical values that mean "no data", drawn in MISSING_COLOR
MISSING_LABELS = frozenset({"unknown", "not available"})
BOUNDARY_COLORS = ("#1a1a19", "#a3442b", "#2f6f4f", "#5b4a9e")
#: Edge colours of representative-discharge trajectories, in order.
TRAJECTORY_COLORS = ("#1a1a19", "#5b2a9e", "#a3442b")
#: Display names for ``boundary_style="inline"``; a boundary not listed shows its key.
BOUNDARY_NAMES = {
    "freidberg_2008_kink_qstar": "External kink limit",
    "freidberg_2008_kink_current": "Freidberg kink current limit",
    "troyon": "Troyon limit",
    "wesson_1989_jet_li_qpsi_lower": "Kink / double-tearing limit",
    "wesson_1989_jet_li_qpsi_upper": "Density-limit disruptions",
    "cheng_1987_li_qa_lower": "Ideal external kink",
    "cheng_1987_li_qa_upper": "Resistive kink",
    "cheng_1987_qa_min": "q(a) = 2",
    "low_q": "Low-q limit",
    "greenwald_hugill": "Greenwald/Hugill limit",
    "murakami_hugill": "Murakami limit",
    "greenwald_fraction_unity": "Greenwald limit",
    "martin_2008_lh": "L-H threshold (Martin 2008)",
    "takizuka_2004_lh": "L-H threshold (Takizuka 2004)",
    "menard_2004_qstar_min": "Menard current limit",
    "menard_2004_qstar_current": "Menard current limit (q* = 1)",
    "iter_1991_q95_min": "ITER q95 guideline",
    "iter_1991_q95_estimate_min": "ITER q95 guideline",
    "iter_1991_q95_current": "ITER q95 = 2.1",
    "akers_2000_q95_estimate_min": "q95 guideline (START estimate)",
    "akers_2000_q95_current": "START q95 = 2.1",
}


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
                       y_range: Optional[Tuple[float, float]] = None,
                       samples: int = 1601) -> OverlayPlan:
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
    requested = requested_boundaries(proj, boundaries)
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

    plotted = np.isfinite(_finite(table, x)) & np.isfinite(_finite(table, y))
    fixed = dict(boundary_inputs or {})
    for key in requested:
        entry = _b.get_boundary(key)
        for q in getattr(entry, "inputs", ()):
            if q.name in fixed or q.name not in table.columns or q.name in (x, y):
                continue
            if _axis_mismatch(q.name, q, declared) is None:
                values = _finite(table, q.name)[plotted]
                if np.isfinite(values).any():
                    fixed[q.name] = float(np.nanmedian(values))
    return overlay_plan(proj, requested, x_range=span(x, x_range), y_range=span(y, y_range), fixed=fixed,
                        samples=samples)


_POWER_OF_TEN = re.compile(r"\b1e(-?\d+)\b")


def _label(quantity: _b.BoundaryQuantity, column: str) -> str:
    if column != quantity.name:
        return column
    symbol = re.sub(r"(?<![\\A-Za-z])(beta|kappa|delta|epsilon|psi)(?![A-Za-z])", r"\\\1", quantity.symbol)
    symbol = re.sub(r"_([A-Za-z0-9]+)", r"_{\1}", symbol)
    from vaft.plot.display import unit_markup

    # "1e19 m^-2 T^-1" reads as 10^19 m^-2 T^-1, typeset like every other vaft.plot axis
    # computed outside the f-string: a backslash inside one is a SyntaxError before Python 3.12
    unit_text = _POWER_OF_TEN.sub(r"10^\1", quantity.unit)
    unit = "" if quantity.unit == "-" else f" [{unit_markup(unit_text)}]"
    return f"${symbol}${unit}"


def _boundary_label(curve: _b.BoundaryCurve) -> str:
    """Key and fixed inputs by their symbols, wrapped; the full citation stays on the registered entry."""
    entry = _b.get_boundary(curve.key)

    def symbol(name):
        try:
            return entry.input(name).symbol
        except (KeyError, AttributeError):
            return name

    fixed = ", ".join(f"{symbol(k)}={v:.3g}" for k, v in curve.fixed.items())
    return textwrap.fill(curve.key + (f" ({fixed})" if fixed else ""), width=44, subsequent_indent="  ")


def _basis_word(entry) -> str:
    """Empirical, Analytical or Numerical, from the registered ``basis``."""
    basis = entry.basis
    if basis.endswith("numerical"):
        return "Numerical"
    if "empirical" in basis:
        return "Empirical"
    return "Analytical"


def _side_word(entry) -> str:
    """What lies on the forbidden side: a regime threshold names its source regime, a limit is "Unstable"."""
    if entry.target_regime:
        return (entry.source_regime or "below threshold").replace("_", "-")
    return "Unstable"


def _allowed_word(entry) -> str:
    if entry.target_regime:
        return entry.target_regime.replace("_", "-") + " accessible"
    return "Stable"


def _math(symbol: str) -> str:
    """A registry symbol (``kappa_a``, ``B_T``, ``|I_p|``) as mathtext."""
    text = re.sub(r"(?<![\\A-Za-z])(beta|kappa|delta|epsilon|psi|rho|mu)(?![A-Za-z])", r"\\\1", symbol)
    text = re.sub(r"_([A-Za-z0-9]+)", r"_{\1}", text)
    return f"${text}$"


def _inline_legend_label(curve: _b.BoundaryCurve) -> str:
    """``{name} ({fixed inputs}) {Unstable} ({Empirical|Analytical|Numerical})``."""
    entry = _b.get_boundary(curve.key)

    def symbol(name):
        if name == entry.target.name:
            return _math(entry.target.symbol)
        try:
            return _math(entry.input(name).symbol)
        except (KeyError, AttributeError):
            return name

    fixed = ", ".join(f"{symbol(k)}={v:.3g}" for k, v in curve.fixed.items())
    text = BOUNDARY_NAMES.get(curve.key, curve.key) + (f" ({fixed})" if fixed else "")
    return textwrap.fill(f"{text} {_side_word(entry)} ({_basis_word(entry)})", width=46, subsequent_indent="  ")


def _unstable_direction(side: str) -> Tuple[float, float]:
    return {"below": (0.0, 1.0), "above": (0.0, -1.0), "left": (1.0, 0.0), "right": (-1.0, 0.0)}[side]


def _label_along(ax, curve: _b.BoundaryCurve, text: str, color: str, where: float = 0.6):
    """The boundary's name written along it, on its forbidden side; None when the curve is not in view."""
    xlo, xhi = sorted(ax.get_xlim())
    ylo, yhi = sorted(ax.get_ylim())
    x, y = np.asarray(curve.x, float), np.asarray(curve.y, float)
    inside = np.isfinite(x) & np.isfinite(y) & (x >= xlo) & (x <= xhi) & (y >= ylo) & (y <= yhi)
    idx = np.flatnonzero(inside)
    if idx.size < 2:
        return None
    pts = ax.transData.transform(np.c_[x[idx], y[idx]])
    box = ax.get_window_extent()
    u = (pts[:, 0] - box.x0) / box.width
    v = (pts[:, 1] - box.y0) / box.height
    if np.ptp(u) > 0.05 and (v.max() < 0.07 or v.min() > 0.93) or np.ptp(v) > 0.05 and (u.max() < 0.05 or u.min() > 0.95):
        return None   # it runs along an edge: a name there would sit on the tick labels
    arc = np.r_[0.0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    if arc[-1] <= 0:
        return None
    seg = np.diff(pts, axis=0)
    steep = np.abs(np.degrees(np.arctan2(np.abs(seg[:, 1]), np.abs(seg[:, 0])))) > 60.0
    target = int(np.clip(np.searchsorted(arc, where * arc[-1]), 1, idx.size - 1))
    usable = np.flatnonzero(~steep & (np.hypot(*seg.T) > 0)) + 1   # segment k ends at point k + 1
    # a saw-tooth's vertical edges are poor places for a name: use the nearest gentler segment if there is one
    j = int(usable[np.argmin(np.abs(arc[usable] - arc[target]))]) if usable.size and steep.mean() < 0.9 else target
    d_disp = pts[j] - pts[j - 1]
    d_data = np.array([x[idx[j]] - x[idx[j - 1]], y[idx[j]] - y[idx[j - 1]]])
    anchor = (0.5 * (x[idx[j]] + x[idx[j - 1]]), 0.5 * (y[idx[j]] + y[idx[j - 1]]))
    moving = np.hypot(*seg.T) > 0.5   # ignore sub-pixel steps
    rises = np.sign(seg[moving, 1])
    rises = rises[rises != 0]
    if rises.size and np.count_nonzero(np.diff(rises)) > 2:   # it turns back and forth more than twice
        # a zig-zag (a saw-tooth boundary): follow its overall direction rather than one short tooth, and sit
        # beyond its extreme on the forbidden side, clear of the teeth
        d_disp = pts[-1] - pts[0]
        d_data = np.array([x[idx[-1]] - x[idx[0]], y[idx[-1]] - y[idx[0]]])
        side = curve.allowed_side
        anchor = (float(np.median(x[idx])), float(np.min(y[idx]) if side == "above" else np.max(y[idx])) if side in (
            "above", "below") else float(np.median(y[idx])))
        if side in ("left", "right"):
            anchor = (float(np.max(x[idx]) if side == "left" else np.min(x[idx])), anchor[1])
    if d_disp[0] < 0 or (d_disp[0] == 0 and d_disp[1] < 0):   # keep the text upright
        d_disp, d_data = -d_disp, -d_data
    angle_disp = np.degrees(np.arctan2(d_disp[1], d_disp[0]))
    up = np.array([-np.sin(np.radians(angle_disp)), np.cos(np.radians(angle_disp))])
    va = "bottom" if float(np.dot(up, _unstable_direction(curve.allowed_side))) >= 0 else "top"
    # the angle is given in data coordinates and turned with the axes, so a later resize keeps it on the line
    return ax.text(anchor[0], anchor[1], " " + text + " ",
                   rotation=float(np.degrees(np.arctan2(d_data[1], d_data[0]))), transform_rotates_text=True,
                   rotation_mode="anchor", ha="center", va=va, color=color, fontsize="small", zorder=4,
                   clip_on=True, bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none",
                                           alpha=0.75))


def _allowed_mask(curve: _b.BoundaryCurve, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Which points lie on the boundary's allowed side; False where the curve does not reach."""
    x, y = np.asarray(curve.x, float), np.asarray(curve.y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 2:
        return np.zeros(px.shape, bool)
    if curve.allowed_side in ("below", "above"):
        order = np.argsort(x)
        level = np.interp(px, x[order], y[order], left=np.nan, right=np.nan)
        if np.ptp(x) == 0:
            return np.zeros(px.shape, bool)
        return (py < level) if curve.allowed_side == "below" else (py > level)
    order = np.argsort(y)
    level = np.interp(py, y[order], x[order], left=np.nan, right=np.nan)
    if np.ptp(y) == 0:
        return np.zeros(px.shape, bool)
    return (px < level) if curve.allowed_side == "left" else (px > level)


def _label_allowed_zone(ax, curves, xs: np.ndarray, ys: np.ndarray, text: str):
    """One label in the zone every drawn boundary allows, where it is farthest from the data."""
    if not curves:
        return None
    (xlo, xhi), (ylo, yhi) = ax.get_xlim(), ax.get_ylim()
    u, v = np.meshgrid(np.linspace(0.2, 0.8, 33), np.linspace(0.08, 0.92, 33))   # the label's width stays inside
    px, py = xlo + u.ravel() * (xhi - xlo), ylo + v.ravel() * (yhi - ylo)
    allowed = np.ones(px.shape, bool)
    for curve in curves:
        allowed &= _allowed_mask(curve, px, py)
    if not allowed.any():
        return None
    cand = np.c_[u.ravel(), v.ravel()][allowed]
    others = [np.c_[(xs - xlo) / (xhi - xlo), (ys - ylo) / (yhi - ylo)]] if len(xs) else []
    for curve in curves:
        cx, cy = np.asarray(curve.x, float), np.asarray(curve.y, float)
        ok = np.isfinite(cx) & np.isfinite(cy)
        others.append(np.c_[(cx[ok] - xlo) / (xhi - xlo), (cy[ok] - ylo) / (yhi - ylo)])
    others = np.vstack(others) if others else np.empty((0, 2))
    if len(others):
        # an anisotropic distance: the label is wider than tall
        d = np.hypot((cand[:, None, 0] - others[None, :, 0]) / 0.18, (cand[:, None, 1] - others[None, :, 1]) / 0.06)
        best = cand[int(np.argmax(d.min(axis=1)))]
    else:
        best = cand[len(cand) // 2]
    return ax.text(best[0], best[1], text, transform=ax.transAxes, ha="center", va="center", fontsize="medium",
                   style="italic", color="#2f6f4f", zorder=4)


def _applicability_warnings(curve: _b.BoundaryCurve, rows: pd.DataFrame, axes: Tuple[str, str]):
    """Where the plotted states or the fixed inputs leave the boundary's declared fitted ranges."""
    entry = _b.get_boundary(curve.key)
    ranges = getattr(entry.applicability, "ranges", {}) or {}
    for name, (lo, hi) in ranges.items():
        if name in curve.fixed:
            values = np.array([curve.fixed[name]])
        elif name in axes:
            values = _finite(rows, name)
            values = values[np.isfinite(values)]
        else:
            continue
        outside = (values < lo) | (values > hi)
        if outside.any():
            yield (f"boundary {curve.key!r} is extrapolated: {int(outside.sum())} of {values.size} value(s) of "
                   f"{name} lie outside its fitted range {lo:g}..{hi:g}")
    machine = getattr(entry.applicability, "machine_class", "")
    if machine and machine != "tokamak":
        yield f"boundary {curve.key!r} applies to: {machine}"


def _shade_forbidden(ax, curve: _b.BoundaryCurve, color: str, hatch: Optional[str] = None) -> None:
    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    x, y = curve.x, curve.y
    if hatch:
        from matplotlib.colors import to_rgba
        kw = dict(facecolor=to_rgba(color, 0.05), edgecolor=to_rgba(color, 0.28), hatch=hatch, linewidth=0, zorder=0)
    else:
        kw = dict(color=color, alpha=0.06, linewidth=0, zorder=0)
    if curve.allowed_side == "below":
        ax.fill_between(x, y, yhi, **kw)
    elif curve.allowed_side == "above":
        ax.fill_between(x, ylo, y, **kw)
    elif curve.allowed_side == "left":
        ax.fill_betweenx(y, x, xhi, **kw)
    elif curve.allowed_side == "right":
        ax.fill_betweenx(y, xlo, x, **kw)


def _scales() -> Tuple[float, float]:
    """Line-width and marker-area factors of the active format/theme, relative to Matplotlib's defaults."""
    import matplotlib

    return (float(matplotlib.rcParams["lines.linewidth"]) / 1.5,
            (float(matplotlib.rcParams["lines.markersize"]) / 6.0) ** 2)


def _draw_trajectories(ax, trajectories, x: str, y: str, time: str, colour_of, size: float) -> None:
    """Each discharge's states joined in time order, one marker size throughout; an arrow ends the path."""
    for k, (label, sub) in enumerate(trajectories.items()):
        if time not in sub.columns:
            raise KeyError(f"trajectory {label!r} has no time column {time!r}")
        t = sub.assign(_x=_finite(sub, x), _y=_finite(sub, y)).sort_values(time)
        t = t[np.isfinite(t["_x"]) & np.isfinite(t["_y"])]
        if t.empty:
            continue
        edge = TRAJECTORY_COLORS[k % len(TRAJECTORY_COLORS)]
        lw, _ = _scales()
        ax.plot(t["_x"], t["_y"], color=edge, linewidth=1.0 * lw, zorder=5)
        ax.scatter(t["_x"], t["_y"], s=np.full(len(t), size), c=[colour_of(r) for _, r in t.iterrows()],
                   edgecolors=edge, linewidths=1.4 * lw, zorder=6, label=str(label))
        if len(t) > 1:
            ax.annotate("", xy=(t["_x"].iloc[-1], t["_y"].iloc[-1]), xytext=(t["_x"].iloc[-2], t["_y"].iloc[-2]),
                        arrowprops=dict(arrowstyle="-|>", color=edge, linewidth=1.2 * lw, shrinkA=0,
                                        shrinkB=0.5 * np.sqrt(size) + 2, mutation_scale=12 * lw),
                        zorder=7)


@presented()
def operational_space_population(table: pd.DataFrame, projection, *, x: Optional[str] = None,
                                 y: Optional[str] = None, color: Optional[str] = None,
                                 marker: Optional[str] = None, hollow: Sequence = (),
                                 boundaries: Union[str, bool, Sequence[str]] = "default",
                                 boundary_inputs: Optional[Mapping[str, float]] = None,
                                 units: Optional[Mapping[str, str]] = None, cmap: str = "viridis",
                                 color_limits: Optional[Tuple[float, float]] = None,
                                 x_range: Optional[Tuple[float, float]] = None,
                                 y_range: Optional[Tuple[float, float]] = None,
                                 title: Optional[str] = None, boundary_style: str = "shade",
                                 trajectories: Optional[Mapping[str, pd.DataFrame]] = None,
                                 time_column: str = "time_efit_s", legend_keys: bool = True,
                                 marker_size: float = 34.0, category_colors: Optional[Mapping[str, str]] = None,
                                 ax=None, figsize: Optional[Tuple[float, float]] = None,
                                 format: Optional[str] = None, theme: Optional[str] = None, show: bool = False):
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
    color_limits : (float, float), optional
        Colour-scale limits for a numeric ``color``; values outside are drawn
        at the ends (the colour bar says so). Default: the finite data span.
    x_range, y_range : (float, float), optional
        Axis limits, for example the span of a reference diagram so its whole
        boundary is visible. Default: the data span.
    title : str, optional
        Axes title; defaults to the projection's title.
    boundary_style : {"shade", "inline"}
        ``"shade"`` tints each forbidden side and lists the boundary keys in
        the legend. ``"inline"`` hatches the forbidden side, writes each
        boundary's name along it, puts one "Stable" label in the zone every
        boundary allows, and gives each boundary a legend patch
        ``{name} ({fixed inputs}) Unstable ({Empirical|Analytical|Numerical})``;
        a regime threshold names its regimes instead.
    trajectories : mapping of str to pandas.DataFrame, optional
        Representative discharges, legend label to their states. Each is drawn
        in ``time_column`` order at one marker size, joined by a line, with an
        arrow on the last step; markers take the ``color`` column's colours.
    time_column : str
        Time column of the trajectories.
    legend_keys : bool
        Legend entries as ``column=value``; False shows the value alone.
    marker_size : float
        Population marker area [pt^2] at Matplotlib's default marker size;
        a format or theme scales it, and trajectory markers are drawn at
        twice it, one size for every state of a discharge.
    category_colors : mapping of str to colour, optional
        Colours for named categories of a categorical ``color`` column,
        replacing their palette slots (for example stable in green).
    ax : matplotlib.axes.Axes, optional
        Target axes.
    figsize : (float, float), optional
        Canvas size; beside ``format=`` it is refused.
    format : str, optional
        Presentation format of :mod:`vaft.plot.presentation` (``screen``,
        ``single_column``, ``double_column``, ...); ``None`` on a canvas the
        renderer creates means ``screen``. Type, lines and markers scale with it.
    theme : str, optional
        Presentation theme (``technical``, ``minimal``, ``monochrome``).
    show : bool
        Call ``plt.show()``.

    Returns
    -------
    (Figure, Axes)
        ``ax.vaft_overlay`` holds the :class:`OverlayPlan` that was drawn.
    """
    import matplotlib.pyplot as plt

    if boundary_style not in ("shade", "inline"):
        raise ValueError(f"boundary_style must be 'shade' or 'inline', not {boundary_style!r}")
    proj = _projection(projection)
    x = x or proj.x.name
    y = y or proj.y.name
    missing = [c for c in (x, y, color, marker) if c and c not in table.columns]
    if missing:
        raise KeyError(f"table has no column(s) {missing}; columns are {list(table.columns)}")
    if ax is None:
        # a format sizes the canvas and lays it out afterwards; the legacy path keeps its own layout
        fig, ax = (plt.subplots(figsize=figsize) if figsize is not None
                   else plt.subplots(figsize=(5.2, 4.2), constrained_layout=True))
    else:
        fig = ax.figure
    line_scale, area_scale = _scales()
    marker_size = marker_size * area_scale

    xs, ys = _finite(table, x), _finite(table, y)
    ok = np.isfinite(xs) & np.isfinite(ys)
    rows = table.loc[ok]
    xs, ys = xs[ok], ys[ok]

    groups = [(None, np.ones(len(rows), bool))]
    if marker:
        labels = rows[marker].astype(object).where(rows[marker].notna(), "unknown").astype(str)
        # like the colours, shapes follow the whole table's order, so a group keeps its shape in every panel
        whole_m = table[marker].astype(object).where(table[marker].notna(), "unknown").astype(str)
        order_m = ([str(c) for c in table[marker].cat.categories]
                   if isinstance(table[marker].dtype, pd.CategoricalDtype) else list(pd.unique(whole_m)))
        order_m += [c for c in pd.unique(labels) if c not in order_m]
        groups = [(name, (labels == name).to_numpy()) for name in order_m]
    hollow = {str(h) for h in hollow}

    numeric_color = (color is not None and pd.api.types.is_numeric_dtype(rows[color])
                     and not pd.api.types.is_bool_dtype(rows[color]))
    norm = None
    if numeric_color:
        cvals = pd.to_numeric(rows[color], errors="coerce").to_numpy(float)
        finite_c = cvals[np.isfinite(cvals)]
        from matplotlib.colors import Normalize
        lo_c, hi_c = color_limits if color_limits is not None else (
            (float(finite_c.min()), float(finite_c.max())) if finite_c.size else (0.0, 1.0))
        norm = Normalize(vmin=lo_c, vmax=hi_c, clip=False)
        colormap = plt.get_cmap(cmap).with_extremes(bad=MISSING_COLOR)  # a missing colour value stays visible
    elif color is not None:
        cats = rows[color].astype(object).where(rows[color].notna(), "unknown").astype(str)
        # a fixed order, so one category keeps its colour across figures: a pandas Categorical's own
        # categories, otherwise sorted; categories absent from these rows still hold their slot
        # the order comes from the whole table, not the rows this panel can plot, so a category keeps its
        # slot in every panel of a figure
        whole = table[color].astype(object).where(table[color].notna(), "unknown").astype(str)
        order = ([str(c) for c in table[color].cat.categories] if isinstance(table[color].dtype, pd.CategoricalDtype)
                 else sorted(pd.unique(whole)))
        order += [c for c in sorted(pd.unique(cats)) if c not in order]
        palette = {name: CATEGORICAL[i % len(CATEGORICAL)] for i, name in enumerate(order) if name in set(cats)}
        slots = {name: i for i, name in enumerate(order)}
        palette = {name: CATEGORICAL[slots[name] % len(CATEGORICAL)] for name in palette}
        for name in palette:   # missing data is grey in every figure, never a palette colour
            if name in MISSING_LABELS:
                palette[name] = MISSING_COLOR
            elif category_colors and name in category_colors:
                palette[name] = category_colors[name]

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
        kw = dict(marker=shape, s=marker_size, linewidths=1.1, zorder=3)
        if open_marker:
            kw.update(facecolors="none", edgecolors=col)
        else:
            kw.update(c=col if isinstance(col, str) else np.asarray(col), edgecolors="white", linewidths=0.4)
        key = (lambda column, value: f"{column}={value}") if legend_keys else (lambda column, value: str(value))
        sc = ax.scatter(xs[mask], ys[mask], label=(key(marker, name) if name is not None and color is None else None),
                        **kw)
        if name is not None and color is not None:  # shape legend in neutral grey; colour has its own entries
            ax.scatter([], [], marker=shape, s=30 * area_scale, label=key(marker, name),
                       **({"facecolors": "none", "edgecolors": "#6b6b66"} if open_marker else {"c": "#6b6b66"}))
        if numeric_color and not open_marker:
            mappable = sc
    if numeric_color:
        from matplotlib.cm import ScalarMappable
        sm = mappable if mappable is not None else ScalarMappable(norm=norm, cmap=colormap)
        if mappable is not None:
            mappable.set_cmap(colormap)
            mappable.set_norm(norm)
        extend = "neither"
        if color_limits is not None and np.isfinite(cvals).any():
            below, above = np.nanmin(cvals) < color_limits[0], np.nanmax(cvals) > color_limits[1]
            extend = {(True, True): "both", (True, False): "min", (False, True): "max"}.get((below, above), "neither")
        fig.colorbar(sm, ax=ax, label=color, extend=extend)
    if color is not None and not numeric_color:
        for name, c in palette.items():
            ax.scatter([], [], color=c, marker="o", s=30 * area_scale,
                       label=f"{color}={name}" if legend_keys else str(name))

    if x_range is not None:
        ax.set_xlim(x_range)
    if y_range is not None:
        ax.set_ylim(y_range)
    plan = population_overlay(table, proj, x=x, y=y, boundaries=boundaries, boundary_inputs=boundary_inputs,
                              units=units, x_range=ax.get_xlim() if len(xs) else None,
                              y_range=ax.get_ylim() if len(ys) else None)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    # a threshold just outside the data span is still part of the picture: widen to show it. Inline, the name
    # is written on the far side of the line, so leave room for it there too (also when the line is in view
    # but close to the edge)
    margin = 0.12 if boundary_style == "inline" else 0.05
    for curve in plan.curves:
        for values, lim, axis in ((curve.x, xlim, "x"), (curve.y, ylim, "y")):
            if values.size and np.ptp(values) == 0:
                level = float(values[0])
                span = lim[1] - lim[0]
                if lim[1] - margin * (level - lim[0]) < level <= lim[1] + 3.0 * span and level > lim[0]:
                    lim = (lim[0], max(lim[1], level + margin * (level - lim[0])))
                elif lim[0] - 3.0 * span <= level < lim[0] + margin * (lim[1] - level) and level < lim[1]:
                    lim = (min(lim[0], level - margin * (lim[1] - level)), lim[1])
                elif not lim[0] <= level <= lim[1]:
                    warnings.warn(f"boundary {curve.key!r} at {axis} = {level:.3g} lies outside the plotted "
                                  f"range {lim[0]:.3g}..{lim[1]:.3g} and is not in view", stacklevel=2)
                xlim, ylim = (lim, ylim) if axis == "x" else (xlim, lim)
    inline = boundary_style == "inline"
    patches = []
    # the forbidden side is shaded to the final limits: shading to the limits from before a threshold widened
    # them filled the wrong side of it
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    for i, curve in enumerate(plan.curves):
        c = BOUNDARY_COLORS[i % len(BOUNDARY_COLORS)]
        ax.plot(curve.x, curve.y, color=c, linewidth=1.6 * line_scale, label=None if inline else _boundary_label(curve),
                zorder=2)
        _shade_forbidden(ax, curve, c, hatch="////" if inline else None)
        if inline:
            from matplotlib.colors import to_rgba
            from matplotlib.patches import Patch
            patches.append(Patch(facecolor=to_rgba(c, 0.05), edgecolor=c, hatch="////", linewidth=1.0,
                                 label=_inline_legend_label(curve)))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if inline and plan.curves:
        for i, curve in enumerate(plan.curves):
            _label_along(ax, curve, BOUNDARY_NAMES.get(curve.key, curve.key), BOUNDARY_COLORS[i % len(BOUNDARY_COLORS)])
        words = {_allowed_word(_b.get_boundary(curve.key)) for curve in plan.curves}
        _label_allowed_zone(ax, plan.curves, xs, ys, " / ".join(sorted(words)))
    if trajectories:
        def colour_of(row):
            if color is None:
                return CATEGORICAL[0]
            value = row[color]
            if numeric_color:
                value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
                return colormap(norm(value)) if np.isfinite(value) else MISSING_COLOR
            name = "unknown" if pd.isna(value) else str(value)
            return palette.get(name, MISSING_COLOR)

        _draw_trajectories(ax, trajectories, x, y, time_column, colour_of, 2.0 * marker_size)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    for curve in plan.curves:
        for message in _applicability_warnings(curve, rows, (x, y)):
            warnings.warn(message, stacklevel=2)
    for key, reason in plan.omitted:
        warnings.warn(f"boundary {key!r} not drawn on {proj.key!r}: {reason}", stacklevel=2)

    ax.set_xlabel(_label(proj.x, x))
    ax.set_ylabel(_label(proj.y, y))
    # an inline figure's legend sits beside the axes, so a long title starts at the axes' left edge
    ax.set_title(title if title is not None else proj.title, loc="left" if inline else "center")
    handles, labels_ = ax.get_legend_handles_labels()
    handles, labels_ = handles + patches, labels_ + [p.get_label() for p in patches]
    if handles and inline:
        ax.legend(handles, labels_, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0,
                  fontsize="small")
    elif handles:
        ax.legend(handles, labels_, fontsize="x-small", frameon=False, loc="best")
    ax.vaft_overlay = plan
    if show:
        plt.show()
    return fig, ax
