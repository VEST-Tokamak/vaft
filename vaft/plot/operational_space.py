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

__all__ = ["operational_space_population", "population_overlay", "applicability_status", "APPLICABILITY_STATUSES",
           "li_qa_pair"]

#: How a drawn boundary relates to the plotted population (#1628, plotting side): inside its declared calibration
#: (SUPPORTED), outside it (OUTSIDE), not assessable from what is declared (UNASSESSED), or not drawable on this
#: projection (NOT_APPLICABLE). There is no MARGINAL: no cutoff is invented (#1639).
APPLICABILITY_STATUSES = ("SUPPORTED", "OUTSIDE", "UNASSESSED", "NOT_APPLICABLE")

#: Aspect ratio R/a below which a tokamak is spherical (Peng and Strickler, Nucl. Fusion 26 (1986) 769:
#: low aspect ratio A <~ 2). A boundary declared for "conventional aspect ratio" is outside its domain there.
SPHERICAL_ASPECT_RATIO_MAX = 2.0

#: Categorical slots in fixed order (the validated palette of vaft.plot.population).
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7")
MARKERS = ("o", "s", "^", "D", "v", "P", "X")
MISSING_COLOR = "#b8b7ae"
#: categorical values that mean "no data", drawn in MISSING_COLOR
MISSING_LABELS = frozenset({"unknown", "not available"})
BOUNDARY_COLORS = ("#1a1a19", "#a3442b", "#2f6f4f", "#5b4a9e")
#: Edge colours of representative-discharge trajectories, in order.
TRAJECTORY_COLORS = ("#1a1a19", "#5b2a9e", "#a3442b")
#: Boundaries shown in the inline style as a dashed reference line, never as a limit: no hatched forbidden side,
#: no part in the "Stable" zone. Murakami is a historical conventional-tokamak reference, not a limit for a
#: spherical tokamak (#1602).
REFERENCE_ONLY = frozenset({"murakami_hugill"})
#: Boundary pairs drawn in the inline style as one theoretical region between them (a filled permissible domain,
#: with each bound as a line), not as two hatched limits: the Cheng-Furth-Boozer domain (#1603).
REGIONS = {
    ("cheng_1987_li_qa_lower", "cheng_1987_li_qa_upper"):
        "Cheng-Furth-Boozer 1987 stable domain (theory: q(0) = 1.01, pressureless straight cylinder, no shell)",
}
#: Where a source figure ends: annotated at the registered end of the boundary's x range (its
#: ``Applicability.ranges``), and only when that end is inside the axes, so that the end of a literature
#: reference is not read as a limit and the data beyond it stay on the plot (#1603). ``{end}`` is that end;
#: the side ("right" of the end point, or "above" it) keeps the note clear of the curve's along-line label.
SOURCE_ENDS = {
    "wesson_1989_jet_li_qpsi_upper": ("JET Fig. 6 ends\nat $q_\\psi$ = {end}", "right"),
    "cheng_1987_li_qa_upper": ("Fig. 4 ends at $q(a)$ = {end}:\nno upper-$q$ limit implied", "above"),
}
#: Fill of a theoretical permissible domain (REGIONS), shared by the plot and its legend patch.
REGION_FILL = ("#1baf7a", 0.16)
#: Projections on which a REFERENCE_ONLY boundary is a reference in every style, the default shaded one too:
#: on the spherical-tokamak Hugill diagram Murakami must never read as a limit (#1602). The same holds for any
#: population whose ``table.attrs["machine_class"]`` is a spherical tokamak.
REFERENCE_PROJECTIONS = frozenset({"hugill_st"})


def _is_spherical(machine_class: Optional[str]) -> bool:
    text = (machine_class or "").lower().replace("_", " ")
    return "spherical" in text or text.split()[:1] == ["st"]


#: Shorter names written along a line in the inline style, where the legend's full name would not fit the curve.
ALONG_LINE_NAMES = {"murakami_hugill": "Murakami (reference)", "sykes_2000_st_hugill": "Hugill (Sykes 2000)",
                    "greenwald_hugill_st": "Greenwald (ST)", "cheng_1987_li_qa_upper": "Resistive kink",
                    "cheng_1987_li_qa_lower": "Ideal external kink"}
#: Display names for ``boundary_style="inline"``; a boundary not listed shows its key.
BOUNDARY_NAMES = {
    "freidberg_2008_kink_qstar": "External kink limit",
    "freidberg_2008_kink_current": "Freidberg kink current limit",
    "troyon": "Troyon (ideal MHD, no wall)",
    "strait_1988_diiid_beta_n_envelope": "DIII-D",
    "taylor_1995_diiid_beta_n_record": "DIII-D, wall-stabilised",
    "garstka_2002_st_beta_n_reference": "ST (START, PEGASUS)",
    "sabbagh_2006_nstx_beta_n_record": "NSTX, wall-stabilised",
    "wesson_1989_jet_li_qpsi_lower": "Kink / double-tearing limit",
    "wesson_1989_jet_li_qpsi_upper": "Density-limit disruptions",
    "cheng_1987_li_qa_lower": "Lower: ideal external kink",
    "cheng_1987_li_qa_upper": "Upper: low-order resistive kink (mainly 2/1, 3/2)",
    "cheng_1987_qa_min": "q(a) = 2",
    "low_q": "Low-q limit",
    "greenwald_hugill": "Greenwald/Hugill limit",
    "greenwald_hugill_st": "Greenwald limit (ST coordinates)",
    "sykes_2000_st_hugill": "Hugill limit (Sykes 2000, MAST)",
    "murakami_hugill": "Murakami (historical conventional-tokamak reference)",
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


#: How a relation that is not a limit is drawn (#1691): no forbidden side and no "Stable" zone, a line style and a
#: legend word per ``Boundary.kind``. A threshold reference also names its registered value in the legend.
REFERENCE_KINDS = {
    "stability_reference": ("-", "stability reference"),
    "experimental_envelope": ("--", "envelope"),
    "experimental_achievement": (":", "record"),
    "reduced_comparator": ("-.", "reduced comparator"),
}
#: Short names written along a reference line, followed by its registered value (the legend has the full name).
REFERENCE_SHORT_NAMES = {
    "troyon": "Troyon",
    "strait_1988_diiid_beta_n_envelope": "DIII-D",
    "taylor_1995_diiid_beta_n_record": "DIII-D record",
    "garstka_2002_st_beta_n_reference": "ST",
    "sabbagh_2006_nstx_beta_n_record": "NSTX",
}


def _kind(key: str) -> str:
    return getattr(_b.get_boundary(key), "kind", "limit")


def _reference_label(curve: _b.BoundaryCurve, suffix: str = "") -> str:
    """``{name}, {target} = {value}: {kind word}``: the value from the registry, not the renderer; the basis only
    when it is not empirical (an experimental level is empirical by its kind)."""
    entry = _b.get_boundary(curve.key)
    value = f", {_math(entry.target.symbol)} = {entry.coefficient:.3g}" if entry.form == "threshold" else ""
    basis = "" if _basis_word(entry) == "Empirical" else f" ({_basis_word(entry)})"
    evidence = entry.applicability.evidence_ranges if entry.applicability is not None else {}
    span = next(iter(evidence.values())) if evidence else None
    shown = "" if span is None else " (◆ its discharge)" if span[0] == span[1] else " (heavy: source data)"
    if entry.kind in ("experimental_envelope", "experimental_achievement"):
        # another machine's operating level is not calibrated on this population by construction, so its
        # applicability status says nothing new; it stays in ax.vaft_applicability, out of the legend
        suffix = ""
    return textwrap.fill(f"{BOUNDARY_NAMES.get(curve.key, curve.key)}{value}: {REFERENCE_KINDS[entry.kind][1]}"
                         f"{basis}{shown}{suffix}", width=46, subsequent_indent="  ")


def _draw_reference(ax, curve: _b.BoundaryCurve, x_name: str, color: str, linewidth: float, label=None) -> None:
    """A reference line: whole where the source gives no evidence range on this x axis; otherwise light (an
    extrapolated reference slope) with the evidence range heavy, or a marker where the evidence is one point."""
    entry = _b.get_boundary(curve.key)
    style = REFERENCE_KINDS[entry.kind][0]
    evidence = (entry.applicability.evidence_ranges if entry.applicability is not None else {}).get(x_name)
    cx, cy = np.asarray(curve.x, dtype=float), np.asarray(curve.y, dtype=float)
    if evidence is None:
        ax.plot(cx, cy, color=color, linewidth=linewidth, linestyle=style, label=label, zorder=2)
        return
    ax.plot(cx, cy, color=color, linewidth=0.6 * linewidth, linestyle=style, alpha=0.45, label=label, zorder=2)
    low, high = (-np.inf if evidence[0] is None else evidence[0]), (np.inf if evidence[1] is None else evidence[1])
    ok = np.isfinite(cx) & np.isfinite(cy)
    if low == high:
        order = np.argsort(cx[ok])
        ax.plot([low], [np.interp(low, cx[ok][order], cy[ok][order])], marker="D", color=color,
                markersize=2.6 * linewidth, markeredgecolor="black", markeredgewidth=0.5, linestyle="none", zorder=4)
        return
    inside = ok & (cx >= low) & (cx <= high)
    ax.plot(np.where(inside, cx, np.nan), np.where(inside, cy, np.nan), color=color, linewidth=1.4 * linewidth,
            linestyle=style, zorder=3)


def _basis_word(entry) -> str:
    """Derived (a registered relation re-expressed by VAFT), else Empirical, Analytical or Numerical from ``basis``."""
    if entry.origin == "derived":
        return "Derived"
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


def _inline_legend_label(curve: _b.BoundaryCurve, suffix: str = "") -> str:
    """``{name} ({fixed inputs}) {Unstable} ({Empirical|Analytical|Numerical|Derived})``."""
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
    return textwrap.fill(f"{text} {_side_word(entry)} ({_basis_word(entry)}){suffix}", width=46, subsequent_indent="  ")


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
    label = ax.text(anchor[0], anchor[1], " " + text + " ",
                    rotation=float(np.degrees(np.arctan2(d_data[1], d_data[0]))), transform_rotates_text=True,
                    rotation_mode="anchor", ha="center", va=va, color=color, fontsize="small", zorder=4,
                    clip_on=True, bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none",
                                            alpha=0.75))
    return _keep_inside(ax, label)


#: Where along a boundary its name is tried, in order, until it covers no name already written.
LABEL_POSITIONS = (0.6, 0.35, 0.8, 0.2, 0.9, 0.5, 0.7, 0.1, 0.45, 0.25, 0.95)


def _label_is_fixed(ax, curve: _b.BoundaryCurve) -> bool:
    """Whether the name's place along the line does not depend on the requested position (a saw-tooth)."""
    probes = []
    for where in (0.2, 0.8):
        label = _label_along(ax, curve, "x", "black", where=where)
        if label is None:
            return False
        probes.append(label.get_position())
        label.remove()
    return np.allclose(probes[0], probes[1])


def _footprint(text, renderer):
    """The label's outline in display coordinates: its rotated background box when it has one (a rotated name's
    axis-aligned extent covers far more than its words), else its extent."""
    from matplotlib.path import Path

    from matplotlib.text import Annotation

    if isinstance(text, Annotation):   # an annotation places its offset text only when it is laid out
        text.update_positions(renderer)
    patch = text.get_bbox_patch()
    if patch is not None:
        text.update_bbox_position_size(renderer)
        return patch.get_transform().transform_path(patch.get_path())
    e = text.get_window_extent(renderer)
    return Path([(e.x0, e.y0), (e.x1, e.y0), (e.x1, e.y1), (e.x0, e.y1), (e.x0, e.y0)])


def _overlap(path, others) -> float:
    """How much an outline covers the others: the fraction of a sample grid inside it that falls in any of them."""
    v = path.vertices
    xs = np.linspace(v[:, 0].min(), v[:, 0].max(), 12)
    ys = np.linspace(v[:, 1].min(), v[:, 1].max(), 6)
    grid = np.array([(a, b) for a in xs for b in ys])
    own = grid[path.contains_points(grid)]
    if not len(own):
        return 0.0
    hit = np.zeros(len(own), bool)
    for other in others:
        hit |= other.contains_points(own)
    return float(hit.mean())


def _place_label(ax, curve: _b.BoundaryCurve, text: str, color: str, placed: list):
    """``_label_along`` at the first position along the line where the name covers none in ``placed`` (outlines
    in display coordinates); when every position covers one, the position with the least overlap (a crowded plot
    keeps its name rather than losing it)."""
    try:
        renderer = ax.figure.canvas.get_renderer()
    except Exception:  # noqa: BLE001 - no renderer: the default position
        return _label_along(ax, curve, text, color)
    best = None
    for where in LABEL_POSITIONS:
        label = _label_along(ax, curve, text, color, where=where)
        if label is None:
            return None
        _shift_inside(ax, label, renderer)   # where it will be drawn, before it is compared
        area = _overlap(_footprint(label, renderer), placed)
        label.remove()
        if best is None or area < best[0]:
            best = (area, where)
        if area == 0:
            break
    label = _label_along(ax, curve, text, color, where=best[1])
    if label is not None:
        _shift_inside(ax, label, renderer)
        placed.append(_footprint(label, renderer))
    return label


def _shift_inside(ax, text, renderer) -> None:
    """Move a label that pokes out of the axes back inside; a label larger than the axes stays where it is."""
    try:
        bb, box = text.get_window_extent(renderer), ax.get_window_extent(renderer)
    except Exception:  # noqa: BLE001 - a backend without extents: keep the placement
        return
    if bb.width >= box.width or bb.height >= box.height:
        return
    dx = max(0.0, box.x0 - bb.x0) - max(0.0, bb.x1 - box.x1)
    dy = max(0.0, box.y0 - bb.y0) - max(0.0, bb.y1 - box.y1)
    if not (dx or dy):
        return
    from matplotlib.text import Annotation

    if isinstance(text, Annotation) and text.anncoords == "offset points":   # its text sits at an offset in points
        to_points = 72.0 / ax.figure.dpi
        ox, oy = text.xyann
        text.xyann = (ox + dx * to_points, oy + dy * to_points)
    else:
        px, py = ax.transData.transform(text.get_position())
        text.set_position(tuple(ax.transData.inverted().transform((px + dx, py + dy))))


def _keep_inside(ax, text):
    """Keep a label inside its axes: at every draw, once the layout is final, a label that pokes out is shifted
    back in so clipping does not cut its words; a label larger than the axes is left where it is."""
    draw = text.draw

    def draw_inside(renderer):
        _shift_inside(ax, text, renderer)
        return draw(renderer)

    text.draw = draw_inside
    return text


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


def _label_allowed_zone(ax, curves, xs: np.ndarray, ys: np.ndarray, text: str, avoid=()):
    """One label in the zone every drawn limit allows, farthest from the data and from every drawn line.

    ``avoid`` holds further curves (reference lines) the label must keep clear of without bounding the zone.
    """
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
    for curve in tuple(curves) + tuple(avoid):
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


def _machine_coverage(declared: str, plotted: Optional[str]) -> Optional[bool]:
    """Whether a boundary's declared machine class covers the plotted machine class; None when undecidable.

    Only what the registry states is used. For a spherical-tokamak population, a class that names spherical
    tokamaks covers it and one calibrated on a conventional machine (conventional aspect ratio, JET) excludes it.
    For a conventional population, a class limited to spherical tokamaks excludes it. A generic "tokamak", or a
    descriptive phrase, never counts as coverage: that is undecided.
    """
    if not plotted or not declared:
        return None
    declared, plotted = declared.lower(), plotted.lower().replace("_", " ")
    spherical = "spherical" in plotted or plotted.split()[:1] == ["st"]
    if spherical:
        if "spherical" in declared:
            return True
        if "conventional" in declared or "jet" in declared:
            return False
        return None
    if "conventional" in plotted and "spherical" in declared and "including" not in declared:
        return False
    return None


def applicability_status(curve: _b.BoundaryCurve, rows: pd.DataFrame, axes: Tuple[str, str],
                         machine_class: Optional[str] = None) -> Tuple[str, Tuple[str, ...]]:
    """The applicability status of a drawn boundary for the plotted states, with its reasons.

    Parameters
    ----------
    curve : BoundaryCurve
        A boundary as drawn on the projection.
    rows : pandas.DataFrame
        The plotted states.
    axes : (str, str)
        The plotted x and y columns.
    machine_class : str, optional
        The plotted population's machine class, e.g. ``"spherical_tokamak"``
        (``table.attrs["machine_class"]``).

    Returns
    -------
    (str, tuple of str)
        One of :data:`APPLICABILITY_STATUSES` and the reasons behind it.

    Notes
    -----
    OUTSIDE when a plotted value or a fixed input leaves a declared fitted
    range, or the declared machine class excludes the plotted one, or the
    class is declared for conventional aspect ratio and the fixed inputs give
    a spherical one (``R/a`` below :data:`SPHERICAL_ASPECT_RATIO_MAX`);
    SUPPORTED only on evidence: the machine class is covered (by name, or by
    a conventional-aspect-ratio declaration met by the fixed ``R/a``) *and*
    at least one declared calibration range was tested on the plotted
    values; UNASSESSED otherwise (no machine class given, a class that does
    not decide it, or no range to test). A boundary the projection omits is
    UNASSESSED when inputs are missing and NOT_APPLICABLE when its quantities
    are not this plane's.
    """
    entry = _b.get_boundary(curve.key)
    reasons = []
    ranges = getattr(entry.applicability, "ranges", {}) or {}
    outside = False
    checked = 0   # declared ranges actually tested on values
    for name, (lo, hi) in ranges.items():
        if name in curve.fixed:
            values = np.array([curve.fixed[name]])
        elif name in axes:
            values = _finite(rows, name)
            values = values[np.isfinite(values)]
        else:
            reasons.append(f"range of {name} not checked (neither plotted nor fixed)")
            continue
        if values.size == 0:
            continue
        checked += 1
        beyond = (values < lo) | (values > hi)
        if beyond.any():
            outside = True
            quantity = next((q for q in (entry.target, *entry.inputs) if q.name == name), None)
            label = _math(quantity.symbol) if quantity is not None else name
            reasons.append(f"{label} {int(beyond.sum())}/{values.size} outside {lo:g}-{hi:g}")
    declared = getattr(entry.applicability, "machine_class", "") or ""
    covered = _machine_coverage(declared, machine_class)
    aspect = _fixed_aspect_ratio(curve.fixed)
    if "conventional" in declared.lower() and aspect is not None:
        # The precondition the registry states, tested on the plotted shape rather than on a class label.
        if aspect < SPHERICAL_ASPECT_RATIO_MAX:
            outside = True
            covered = False
            reasons.append(f"A = R/a = {aspect:.2f} is spherical (< {SPHERICAL_ASPECT_RATIO_MAX:g}); "
                           "declared for conventional aspect ratio")
        elif covered is None:
            covered = True
            reasons.append(f"A = R/a = {aspect:.2f} meets the declared conventional aspect ratio")
    elif covered is False:
        outside = True
        # the class's first phrase ("JET (conventional ...), 1985-88" -> "JET"); the rest stays on the registry entry
        reasons.append("calibrated on " + declared.split("(")[0].split(",")[0].strip())
    if outside:
        return "OUTSIDE", tuple(r for r in reasons if "not checked" not in r)
    if covered and checked:
        # supported only on evidence: the class covers the population and a declared range was tested
        return "SUPPORTED", tuple(reasons) or (f"inside {checked} declared calibration range(s)",)
    if covered:
        reasons.append("its class covers the plotted machines, but no calibration range was declared or tested")
    else:
        reasons.append("no machine class given for the plotted states" if not machine_class
                       else f"declared class '{declared or 'none'}' does not decide {machine_class}")
    return "UNASSESSED", tuple(reasons)


def _fixed_aspect_ratio(fixed: Mapping[str, float]) -> Optional[float]:
    """R/a from a curve's fixed inputs (``aspect_ratio``, or ``major_radius`` over ``minor_radius``); None if unknown."""
    if "aspect_ratio" in fixed:
        aspect = float(fixed["aspect_ratio"])
    elif "major_radius" in fixed and "minor_radius" in fixed and float(fixed["minor_radius"]) > 0.0:
        aspect = float(fixed["major_radius"]) / float(fixed["minor_radius"])
    else:
        return None
    return aspect if np.isfinite(aspect) else None


#: Omission reasons that mean missing information (UNASSESSED), not a quantity that cannot be drawn here.
_MISSING_INPUT_MARKERS = ("needs fixed input", "is not declared", "are not fixed")


def _omission_status(reason: str) -> Tuple[str, Tuple[str, ...]]:
    missing = any(marker in reason for marker in _MISSING_INPUT_MARKERS)
    return ("UNASSESSED" if missing else "NOT_APPLICABLE"), (reason,)


def _status_suffix(status: str, reasons: Sequence[str]) -> str:
    if status == "OUTSIDE":
        return " [outside calibration: " + "; ".join(reasons) + "]"
    if status == "UNASSESSED":
        return " [applicability unassessed]"
    return ""


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
    """Each discharge's states joined in time order, one marker size throughout; an arrowhead on every step, so the
    direction reads along the whole equilibrium sequence, not only at its end."""
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
        for i in range(len(t) - 1):
            ax.annotate("", xy=(t["_x"].iloc[i + 1], t["_y"].iloc[i + 1]), xytext=(t["_x"].iloc[i], t["_y"].iloc[i]),
                        arrowprops=dict(arrowstyle="-|>", color=edge, linewidth=1.2 * lw,
                                        shrinkA=0.5 * np.sqrt(size), shrinkB=0.5 * np.sqrt(size) + 2,
                                        mutation_scale=12 * lw),
                        zorder=7)


def li_qa_pair(table: pd.DataFrame, *, x_range: Optional[Tuple[float, float]] = None,
               y_range: Optional[Tuple[float, float]] = None, format: Optional[str] = None,
               theme: Optional[str] = None, show: bool = False, **kwargs):
    """The empirical (Wesson 1989, JET) and theoretical (Cheng-Furth-Boozer 1987) l_i-q references side by side.

    The two references use different quantities -- Wesson the equilibrium edge
    $q_\\psi$ and $l_i(3)$, Cheng et al. a straight cylinder's $q(a)$ and $l_i$ --
    so they are never drawn on one pair of axes (#1603, #1620): the left panel
    places the population on Wesson's plane, the right panel draws the CFB
    stable domain on its own plane, with the population only where the table
    has the cylinder quantities.

    Parameters
    ----------
    table : pandas.DataFrame
        Population, columns named by quantity identity.
    x_range, y_range : (float, float), optional
        Limits of the Wesson panel. By default each panel spans its reference
        (Wesson: q_psi 0-18, l_i 0-2; CFB: q(a) 1-9, l_i 0.2-2.9) widened to
        every finite data point, so the data are never cut to a reference.
    format, theme : str, optional
        Presentation format and theme of :mod:`vaft.plot.presentation`.
    show : bool
        Call ``plt.show()``.
    **kwargs
        Passed to :func:`operational_space_population` for both panels
        (``color``, ``marker``, ``category_colors``, ...). A table without the
        Wesson columns leaves the left panel as a reference only.

    Returns
    -------
    (Figure, ndarray of Axes)
        The two panels.
    """
    import matplotlib.pyplot as plt

    from vaft.plot.presentation import DEFAULT_FORMAT, resolve_presentation

    pres = resolve_presentation(format or DEFAULT_FORMAT, theme)
    with pres.context():
        width = pres.format.width_in
        fig, axs = plt.subplots(1, 2, figsize=(width, min(0.6 * width, pres.format.max_height_in)))
        panels = (("li_qa_wesson", ("edge_safety_factor", "internal_inductance_li3"), (0.0, 18.0), (0.0, 2.0),
                   "Wesson 1989 (JET, empirical)", "no $q_\\psi$, $l_i(3)$ for these states:\nreference only"),
                  ("li_qa_cheng", ("cylinder_edge_safety_factor", "internal_inductance_cylinder"), (1.0, 9.0),
                   (0.2, 2.9), "Cheng-Furth-Boozer 1987 (theory)",
                   "no cylinder $q(a)$, $l_i$\nfor these states: reference only"))
        for ax, (proj, cols, x_ref, y_ref, title, empty_note) in zip(axs, panels):
            values = [pd.to_numeric(table[c], errors="coerce") if c in table.columns else None for c in cols]
            finite = (values[0].notna() & values[1].notna()
                      & np.isfinite(values[0]) & np.isfinite(values[1])) if all(v is not None for v in values) else None
            has_data = finite is not None and bool(finite.any())
            data = table if has_data else pd.DataFrame({c: pd.Series(dtype=float) for c in cols})
            if has_data and proj == "li_qa_wesson" and (x_range is not None or y_range is not None):
                xr, yr = x_range or x_ref, y_range or y_ref   # explicit limits are the caller's
            else:
                xr, yr = x_ref, y_ref
                if has_data:   # widen the reference span to every finite data point: the data are never cut
                    xs_, ys_ = values[0][finite], values[1][finite]
                    xr = (min(xr[0], float(xs_.min())), max(xr[1], 1.05 * float(xs_.max())))
                    yr = (min(yr[0], float(ys_.min())), max(yr[1], 1.05 * float(ys_.max())))
            operational_space_population(data, proj, boundary_style="inline", x_range=xr, y_range=yr, ax=ax,
                                         legend_keys=False, title=title, **(kwargs if has_data else {}))
            if not has_data:
                ax.text(0.97, 0.03, empty_note, transform=ax.transAxes, ha="right", va="bottom", fontsize="x-small",
                        style="italic", color="0.35")
        for ax in axs:   # the legends go under their panels, so the two panels keep their width
            legend = ax.get_legend()
            if legend is not None:
                handles, labels = legend.legend_handles, [t.get_text() for t in legend.get_texts()]
                ax.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16),
                          fontsize="xx-small", ncol=1)
        fig.tight_layout(pad=pres.pad if pres.pad is not None else 1.08)
        if show:
            plt.show()
    return fig, axs


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
        ``ax.vaft_overlay`` holds the :class:`OverlayPlan` that was drawn and
        ``ax.vaft_applicability`` each requested boundary's
        :func:`applicability_status`, judged against the population's
        ``table.attrs["machine_class"]``; the inline legend names OUTSIDE and
        UNASSESSED boundaries.
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
                              units=units, x_range=ax.get_xlim() if (len(xs) or x_range is not None) else None,
                              y_range=ax.get_ylim() if (len(ys) or y_range is not None) else None)
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
    machine_class = (getattr(table, "attrs", {}) or {}).get("machine_class")
    applicability = {key: _omission_status(reason) for key, reason in plan.omitted}
    drawn = {curve.key: curve for curve in plan.curves}
    regions = {pair: text for pair, text in REGIONS.items() if inline and all(k in drawn for k in pair)}
    in_region = {k for pair in regions for k in pair}
    for i, curve in enumerate(plan.curves):
        c = BOUNDARY_COLORS[i % len(BOUNDARY_COLORS)]
        applicability[curve.key] = applicability_status(curve, rows, (x, y), machine_class)
        suffix = _status_suffix(*applicability[curve.key])
        if _kind(curve.key) in REFERENCE_KINDS:   # a reference, not a limit: no forbidden side (#1691)
            _draw_reference(ax, curve, proj.x.name, c, 1.6 * line_scale,
                            label=None if inline else _reference_label(curve, suffix))
            if inline:
                from matplotlib.lines import Line2D
                patches.append(Line2D([], [], color=c, linestyle=REFERENCE_KINDS[_kind(curve.key)][0],
                                      linewidth=1.6 * line_scale, label=_reference_label(curve, suffix)))
            continue
        reference = curve.key in REFERENCE_ONLY and (
            inline or proj.key in REFERENCE_PROJECTIONS or _is_spherical(machine_class))
        ax.plot(curve.x, curve.y, color=c, linewidth=1.6 * line_scale, linestyle="--" if reference else "-",
                label=None if inline else (_boundary_label(curve) + (" (reference only)" if reference else "")),
                zorder=2)
        if reference and not inline:
            continue   # a dashed line without a forbidden side
        if reference or curve.key in in_region:
            from matplotlib.lines import Line2D
            entry = _b.get_boundary(curve.key)
            patches.append(Line2D([], [], color=c, linestyle="--" if reference else "-", linewidth=1.6 * line_scale,
                                  label=textwrap.fill(f"{BOUNDARY_NAMES.get(curve.key, curve.key)} "
                                                      f"({_basis_word(entry)}){suffix}", width=46, subsequent_indent="  ")))
            continue
        _shade_forbidden(ax, curve, c, hatch="////" if inline else None)
        if inline:
            from matplotlib.colors import to_rgba
            from matplotlib.patches import Patch
            patches.append(Patch(facecolor=to_rgba(c, 0.05), edgecolor=c, hatch="////", linewidth=1.0,
                                 label=_inline_legend_label(curve, suffix)))
    for (lower_key, upper_key), text in regions.items():
        from matplotlib.colors import to_rgba
        from matplotlib.patches import Patch
        lower, upper = drawn[lower_key], drawn[upper_key]
        ok = np.isfinite(lower.x) & np.isfinite(lower.y)
        lx, ly = np.asarray(lower.x)[ok], np.asarray(lower.y)[ok]
        uok = np.isfinite(upper.x) & np.isfinite(upper.y)
        if not ok.any() or not uok.any():
            continue
        order = np.argsort(np.asarray(upper.x)[uok])
        uy = np.interp(lx, np.asarray(upper.x)[uok][order], np.asarray(upper.y)[uok][order], left=np.nan, right=np.nan)
        span = np.isfinite(uy) & (uy >= ly)
        # only where the source draws both bounds: no fill beyond the figure's end
        ax.fill_between(lx, ly, uy, where=span, color=REGION_FILL[0], alpha=REGION_FILL[1], linewidth=0, zorder=0)
        patches.append(Patch(facecolor=to_rgba(*REGION_FILL), edgecolor="none",
                             label=textwrap.fill(text + " (Theoretical)", width=46, subsequent_indent="  ")))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    placed = []   # extents of the notes and names already written, so a later one does not cover them
    if inline:
        for curve in plan.curves:
            note, side = SOURCE_ENDS.get(curve.key, (None, None))
            ok = np.isfinite(curve.x) & np.isfinite(curve.y)
            entry = _b.get_boundary(curve.key)
            ranges = dict(entry.applicability.ranges) if entry.applicability is not None else {}
            if note is None or not ok.any() or proj.x.name not in ranges:
                continue
            # the registered end of the source, not the last sampled point (which is the view edge when the
            # axes stop short of the source)
            x_end = float(ranges[proj.x.name][1])
            cx, cy = np.asarray(curve.x, dtype=float)[ok], np.asarray(curve.y, dtype=float)[ok]
            order = np.argsort(cx)
            cx, cy = cx[order], cy[order]
            (x0, x1), (y0, y1) = sorted(ax.get_xlim()), sorted(ax.get_ylim())
            if not (x0 <= x_end <= x1) or x_end > cx[-1] + 0.01 * (x1 - x0):
                continue   # the source end is off the axes, or the curve was not sampled up to it
            y_end = float(np.interp(x_end, cx, cy))
            note = note.format(end=f"{x_end:g}")
            if y0 <= y_end <= y1:
                offset, ha, va = ((4, 0), "left", "center") if side == "right" else ((-2, 4), "right", "bottom")
                written = _keep_inside(ax, ax.annotate(note, xy=(x_end, y_end), xytext=offset,
                                                       textcoords="offset points", ha=ha, va=va, fontsize="x-small",
                                                       style="italic", color="0.35", zorder=5, annotation_clip=True))   # above the names' boxes
                try:
                    renderer = ax.figure.canvas.get_renderer()
                    _shift_inside(ax, written, renderer)
                    placed.append(_footprint(written, renderer))
                except Exception:  # noqa: BLE001 - no renderer: the note is not counted
                    pass
    if inline and plan.curves:
        # names that cannot move along their line (a saw-tooth sits beyond its extreme) go first, so the movable
        # ones find room around them
        order = sorted(range(len(plan.curves)), key=lambda k: not _label_is_fixed(ax, plan.curves[k]))
        for i in order:
            curve = plan.curves[i]
            name = ALONG_LINE_NAMES.get(curve.key, BOUNDARY_NAMES.get(curve.key, curve.key))
            entry = _b.get_boundary(curve.key)
            if entry.kind in REFERENCE_KINDS and entry.form == "threshold":   # short: the legend has the full name
                name = f"{REFERENCE_SHORT_NAMES.get(curve.key, entry.target.symbol)} {entry.coefficient:.3g}"
            _place_label(ax, curve, name, BOUNDARY_COLORS[i % len(BOUNDARY_COLORS)], placed)
        limits = [curve for curve in plan.curves
                  if curve.key not in REFERENCE_ONLY and _kind(curve.key) not in REFERENCE_KINDS]
        words = {_allowed_word(_b.get_boundary(curve.key)) for curve in limits}
        if limits:
            _label_allowed_zone(ax, limits, xs, ys, " / ".join(sorted(words)),
                                avoid=[curve for curve in plan.curves if curve not in limits])
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
    ax.vaft_applicability = applicability
    if show:
        plt.show()
    return fig, ax
