"""Dimensionless-similarity population plots with literature reference points (#1624).

A population is drawn on one of the global similarity projections of
:mod:`vaft.diagram._similarity_space` (``"rho_star_nu_star"``,
``"rho_star_beta_n"``, ``"rho_star_omega_ci_tau_e"``) through
:func:`vaft.plot.operational_space.operational_space_population`, with log axes,
the reference design points a source states in its text, and an explicit count
of the states that could not be placed.

The table's columns must *be* the projection's quantities: a $\\nu_*$ in the
ITPA-database convention is the column ``nu_star_verdoolaege_2021``, and a
column holding Sauter's local $\\nu_*$ under another name is refused rather
than drawn on that axis. A row whose axis value is missing or not positive is
reported as unassessed, per group, and never filled in. No quantity is
computed here.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from vaft.diagram._op_space import OperationalProjection, get_projection
from vaft.diagram._similarity_space import UNAVAILABLE_REFERENCES, ReferencePoint, reference_points
from vaft.plot.operational_space import operational_space_population

__all__ = ["dimensionless_similarity", "SimilarityExclusions", "AXIS_SCALES", "AXIS_LABELS"]

#: Axis scales per projection: rho*, nu* and Omega_ci tau_E span decades between devices; beta_N does not.
AXIS_SCALES = {
    "rho_star_nu_star": ("log", "log"),
    "rho_star_beta_n": ("log", "linear"),
    "rho_star_omega_ci_tau_e": ("log", "log"),
}
#: Axis labels that name the convention, so a figure cannot be read as another nu* or rho*.
AXIS_LABELS = {
    "rho_star_verdoolaege_2021": r"$\rho_*$ (ITPA DB, Verdoolaege 2021 Eq. 1a)",
    "nu_star_verdoolaege_2021": r"$\nu_*$ (ITPA DB, Verdoolaege 2021 Eq. 1c)",
    "omega_ci_tau_e_th": r"$\Omega_{ci}\tau_{E,\mathrm{th}}$",
    "normalized_beta": r"$\beta_N$ [% m T / MA]",
}
REFERENCE_MARKER = dict(marker="*", s=180.0, facecolors="#1a1a19", edgecolors="white", linewidths=0.6, zorder=5)


@dataclass(frozen=True)
class SimilarityExclusions:
    """The states that could not be placed on a projection, and why.

    ``unassessed`` maps a group (the ``group`` column's value, or ``"all"``) to
    the number of its rows left off. ``reasons`` maps each axis column to the
    number of rows where it is missing, not finite or not positive.
    """

    projection: str
    total: int
    assessed: int
    unassessed: Mapping[str, int]
    reasons: Mapping[str, int]

    @property
    def excluded(self) -> int:
        return self.total - self.assessed

    def summary(self) -> str:
        if not self.excluded:
            return f"{self.projection}: all {self.total} states placed"
        groups = ", ".join(f"{k}: {v}" for k, v in self.unassessed.items())
        why = ", ".join(f"{k} missing or non-positive in {v}" for k, v in self.reasons.items() if v)
        return (f"{self.projection}: {self.excluded} of {self.total} states UNASSESSED ({groups}); {why}. "
                "They are left off, not approximated.")


def _similarity_projection(projection) -> OperationalProjection:
    proj = get_projection(projection) if isinstance(projection, str) else projection
    meta = proj.interpretation
    if meta is None or meta.category == "stability_limit":
        raise ValueError(f"projection {proj.key!r} is not a dimensionless-similarity or regime space; "
                         "use vaft.plot.operational_space.operational_space_population")
    return proj


def _check_columns(table: pd.DataFrame, proj: OperationalProjection, units: Optional[Mapping[str, str]]) -> None:
    declared = dict(getattr(table, "attrs", {}).get("units", {}) or {})
    declared.update(units or {})
    for quantity in (proj.x, proj.y):
        if quantity.name not in table.columns:
            convention = proj.interpretation.convention(quantity.name)
            raise KeyError(
                f"table has no column {quantity.name!r}. The axis is {convention.expression} "
                f"({convention.source.citation}); a value computed with another convention is another quantity "
                "and is not drawn on it"
            )
        unit = declared.get(quantity.name, "-" if quantity.unit == "-" else None)
        if unit != quantity.unit:
            raise ValueError(f"column {quantity.name!r} must be declared in {quantity.unit!r}, not {unit!r}")


def _exclusions(table: pd.DataFrame, proj: OperationalProjection, group: Optional[str]) -> Tuple[np.ndarray, SimilarityExclusions]:
    ok = np.ones(len(table), bool)
    reasons = {}
    for quantity in (proj.x, proj.y):
        values = pd.to_numeric(table[quantity.name], errors="coerce").to_numpy(float)
        bad = ~(np.isfinite(values) & (values > 0))
        reasons[quantity.name] = int(bad.sum())
        ok &= ~bad
    labels = (table[group].astype(object).where(table[group].notna(), "unknown").astype(str)
              if group else pd.Series("all", index=table.index))
    unassessed = {str(k): int(v) for k, v in labels[~ok].value_counts(sort=False).items()}
    report = SimilarityExclusions(projection=proj.key, total=len(table), assessed=int(ok.sum()),
                                  unassessed=unassessed, reasons=reasons)
    return ok, report


def _reference_rows(proj: OperationalProjection, references, reference_table: Optional[pd.DataFrame]):
    """``(label, x, y, kind, citation)`` for every reference point to draw."""
    rows = []
    if references is True or references == "default":
        points: Sequence[ReferencePoint] = reference_points(proj)
    elif references in (False, None):
        points = ()
    else:
        raise ValueError(f"references must be 'default', True, False or None, not {references!r}")
    for p in points:
        rows.append((p.label, p.values[proj.x.name], p.values[proj.y.name], p.kind, p.source.citation))
    if reference_table is not None:
        needed = ["label", "kind", "source", proj.x.name, proj.y.name]
        missing = [c for c in needed if c not in reference_table.columns]
        if missing:
            raise KeyError(f"reference_table needs columns {needed}; missing {missing}")
        for _, r in reference_table.iterrows():
            if not str(r["source"]).strip() or pd.isna(r["source"]):
                raise ValueError(f"reference {r['label']!r} has no source; every reference point needs one")
            rows.append((str(r["label"]), float(r[proj.x.name]), float(r[proj.y.name]), str(r["kind"]),
                         str(r["source"])))
    return rows


def _span(values, scale: str) -> Optional[Tuple[float, float]]:
    values = np.asarray([v for v in values if np.isfinite(v) and (v > 0 or scale == "linear")], float)
    if values.size == 0:
        return None
    lo, hi = float(values.min()), float(values.max())
    if scale == "log":
        return (lo / 1.6, hi * 1.6)
    pad = 0.1 * (hi - lo) if hi > lo else 0.1 * max(abs(hi), 1.0)
    return (min(0.0, lo - pad) if lo >= 0 else lo - pad, hi + pad)


def dimensionless_similarity(table: pd.DataFrame, projection: Union[str, OperationalProjection] = "rho_star_nu_star",
                             *, group: Optional[str] = None,
                             references: Union[str, bool, None] = "default",
                             reference_table: Optional[pd.DataFrame] = None,
                             units: Optional[Mapping[str, str]] = None,
                             scales: Optional[Tuple[str, str]] = None,
                             x_range: Optional[Tuple[float, float]] = None,
                             y_range: Optional[Tuple[float, float]] = None,
                             boundaries: Union[str, bool, Sequence[str]] = "default",
                             title: Optional[str] = None, ax=None, show: bool = False, **population_kwargs):
    """Scatter a population on a global dimensionless-similarity projection.

    Parameters
    ----------
    table : pandas.DataFrame
        One row per state. The axis columns must be the projection's quantity
        names (``rho_star_verdoolaege_2021``, ``nu_star_verdoolaege_2021``,
        ``normalized_beta``, ``omega_ci_tau_e_th``); ``normalized_beta`` needs
        its unit declared in ``table.attrs["units"]`` or ``units=``.
    projection : str or OperationalProjection
        A registered projection carrying an interpretation, see
        ``vaft.diagram._similarity_space.SIMILARITY_PROJECTIONS``.
    group : str, optional
        Categorical column (machine, scenario, ...) for colour and for the
        per-group count of unassessed states.
    references : "default", True, False or None
        Draw the registered reference points that state both axis values
        (``vaft.diagram._similarity_space.REFERENCE_POINTS``).
    reference_table : pandas.DataFrame, optional
        Further reference points with columns ``label``, ``kind``, ``source``
        and the two axis quantities; a row without a source is refused.
    units : mapping, optional
        Column units.
    scales : (str, str), optional
        Axis scales; default :data:`AXIS_SCALES` for the projection.
    x_range, y_range : (float, float), optional
        Axis limits; default the span of the placed states and references.
    boundaries : "default", False or sequence of str
        Registered boundaries; the similarity projections have none by
        default.
    title : str, optional
        Axes title; default the projection's title.
    ax : matplotlib.axes.Axes, optional
        Target axes.
    show : bool
        Call ``plt.show()``.
    **population_kwargs
        Passed to :func:`operational_space_population` (``marker``,
        ``format``, ``theme``, ``trajectories``, ...).

    Returns
    -------
    (Figure, Axes)
        ``ax.vaft_interpretation`` holds the projection's
        :class:`~vaft.diagram._projection_interpretation.ProjectionInterpretation`,
        ``ax.vaft_exclusions`` the :class:`SimilarityExclusions`,
        ``ax.vaft_references`` the drawn reference rows and
        ``ax.vaft_unavailable_references`` the source's references that have
        no stated values.

    Raises
    ------
    KeyError
        An axis column is absent (for example a nu* under another convention's name).
    ValueError
        The projection has no interpretation, a unit does not match, or a
        reference has no source.
    """
    import matplotlib.pyplot as plt

    proj = _similarity_projection(projection)
    _check_columns(table, proj, units)
    scales = tuple(scales or AXIS_SCALES.get(proj.key, ("linear", "linear")))
    ok, report = _exclusions(table, proj, group)
    if report.excluded:
        warnings.warn(report.summary(), stacklevel=2)
    placed = table.copy()
    for quantity in (proj.x, proj.y):   # non-positive values are unassessed, not plotted at an axis edge
        placed.loc[~ok, quantity.name] = np.nan
    refs = _reference_rows(proj, references, reference_table)
    if x_range is None:
        x_range = _span(list(placed[proj.x.name][ok]) + [r[1] for r in refs], scales[0])
    if y_range is None:
        y_range = _span(list(placed[proj.y.name][ok]) + [r[2] for r in refs], scales[1])
    fig, ax = operational_space_population(placed, proj, color=group, boundaries=boundaries, units=units,
                                           x_range=x_range, y_range=y_range, title=title, ax=ax, **population_kwargs)
    ax.set_xscale(scales[0])
    ax.set_yscale(scales[1])
    if x_range is not None:
        ax.set_xlim(x_range)
    if y_range is not None:
        ax.set_ylim(y_range)
    drawn = []
    for label, xv, yv, kind, citation in refs:
        drawn.append(ax.scatter([xv], [yv], label=f"{label} ({kind}; {citation.split(',')[0]})", **REFERENCE_MARKER))
        ax.annotate(label, (xv, yv), xytext=(5, 5), textcoords="offset points", fontsize="small")
    ax.set_xlabel(AXIS_LABELS.get(proj.x.name, ax.get_xlabel()))
    ax.set_ylabel(AXIS_LABELS.get(proj.y.name, ax.get_ylabel()))
    if report.excluded:   # the figure itself says who is missing, not only the warning
        from matplotlib.lines import Line2D
        for name, count in report.unassessed.items():
            drawn.append(Line2D([], [], linestyle="none", marker="x", color="#6b6b66",
                                label=f"{name}: {count} unassessed (input missing)"))
    if drawn:
        old = ax.get_legend()   # keep the renderer's entries, including the patches it added by hand
        handles = list(getattr(old, "legend_handles", None) or getattr(old, "legendHandles", [])) if old else []
        labels = [t.get_text() for t in old.get_texts()] if old else []
        handles += drawn
        labels += [h.get_label() for h in drawn]
        if population_kwargs.get("boundary_style") == "inline":
            ax.legend(handles, labels, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                      borderaxespad=0.0, fontsize="small")
        else:
            ax.legend(handles, labels, fontsize="x-small", frameon=False, loc="best")
    ax.vaft_interpretation = proj.interpretation
    ax.vaft_exclusions = report
    ax.vaft_references = tuple(refs)
    ax.vaft_unavailable_references = UNAVAILABLE_REFERENCES
    if show:
        plt.show()
    return fig, ax
