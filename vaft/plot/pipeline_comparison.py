"""Legacy archive against the current pipeline: distributions, matched pairs and a scorecard.

These draw the tables ``workflow/pipeline_improvement/build_comparison_tables.py``
writes, one row per side and slice with a ``side`` column (``"legacy"`` or
``"current"``) and, for slices matched by time, a shared ``pair_id``. They
follow the renderer contract of :mod:`vaft.plot` (``ax=None``, ``show=False``,
return ``(Figure, Axes)``) and take the table, as :mod:`vaft.plot.population`
and :mod:`vaft.plot.stability_atlas` do. No ODS, database or solver layer is
imported.

The two sides keep one colour each in every figure: legacy takes the second
categorical slot and is drawn dashed, current takes the first. Each curve is
named on the data, so identity never rests on colour alone.

A metric's direction comes from
:data:`vaft.validation.pipeline_comparison.METRICS`. ``"zero"`` metrics (log
ratios) are drawn as ``|value|``, so that in every panel further left (or
lower) is better.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .population import AXIS_LABEL_SCALE, CATEGORICAL, TICK_LABEL_SCALE, _font

__all__ = [
    "pipeline_comparison_distribution",
    "pipeline_comparison_measured_vs_reconstructed",
    "pipeline_comparison_paired",
    "pipeline_comparison_scorecard",
]

SIDES = ("legacy", "current")
SIDE_STYLE = {
    "legacy": {"color": CATEGORICAL[1], "linestyle": "--", "marker": "s", "label": "legacy archive"},
    "current": {"color": CATEGORICAL[0], "linestyle": "-", "marker": "o", "label": "current pipeline"},
}
INK = "#1a1a19"
MUTED = "#7c7b73"
GRID = "#e4e3dc"


def _axes(ax, figsize):
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax


def _finish(fig, show: bool):
    if show:
        import matplotlib.pyplot as plt

        plt.show()


def _recessive(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=_font("xtick.labelsize", TICK_LABEL_SCALE))
    ax.xaxis.label.set_size(_font("axes.labelsize", AXIS_LABEL_SCALE))
    ax.yaxis.label.set_size(_font("axes.labelsize", AXIS_LABEL_SCALE))


def _oriented(values: np.ndarray, better: str | None) -> np.ndarray:
    return np.abs(values) if better == "zero" else values


def _values(table: pd.DataFrame, value: str, side: str, by: str, better: str | None, log: bool) -> np.ndarray:
    column = pd.to_numeric(table.loc[table[by] == side, value], errors="coerce").to_numpy(dtype=float)
    column = _oriented(column, better)
    column = column[np.isfinite(column)]
    return column[column > 0] if log else column


def pipeline_comparison_distribution(
    table: pd.DataFrame,
    value: str,
    *,
    by: str = "side",
    sides: Sequence[str] = SIDES,
    better: str | None = "lower",
    log: bool = False,
    linthresh: float | None = None,
    xlabel: str | None = None,
    ax=None,
    show: bool = False,
    figsize: tuple[float, float] = (6.0, 4.0),
):
    """Empirical CDF of one metric for each side, with each median marked and labelled.

    ``log`` draws a log axis over the positive values; ``linthresh`` instead
    draws a symmetric-log axis with that linear range, for a signed quantity
    such as a stability energy whose sign is the result.

    Every slice of a side counts, paired or not; :func:`pipeline_comparison_paired`
    is the slice-by-slice view. The label at each median names the side, its
    count and the median, so the figure reads without a legend.
    """
    fig, ax = _axes(ax, figsize)
    curves = []
    for k, side in enumerate(sides):
        data = np.sort(_values(table, value, side, by, better, log))
        style = SIDE_STYLE.get(side, {"color": CATEGORICAL[k % len(CATEGORICAL)], "linestyle": "-",
                                      "label": side})
        if not data.size:
            continue
        y = np.arange(1, data.size + 1) / data.size
        ax.step(data, y, where="post", color=style["color"], linestyle=style["linestyle"], linewidth=2.0)
        median = float(np.median(data))
        ax.plot([median], [0.5], marker="o", markersize=8, color=style["color"],
                markeredgecolor="white", markeredgewidth=2.0, zorder=3)
        curves.append((median, data.size, style))
    # Labels sit in the two corners the curves leave empty: the curve further
    # left (smaller median) is named top-left, the other bottom-right. Each
    # label carries a short line sample, so identity is not colour alone.
    from matplotlib.lines import Line2D

    for rank, (median, count, style) in enumerate(sorted(curves, key=lambda c: c[0])):
        top = rank == 0
        # (text x, text y, sample x0, sample x1, sample y) in axes fractions.
        tx, ty, x0, x1, sy = (0.13, 0.97, 0.03, 0.11, 0.935) if top else (0.87, 0.03, 0.89, 0.97, 0.115)
        ax.text(tx, ty, f"{style['label']}\nn = {count}, median {median:.3g}", transform=ax.transAxes,
                ha="left" if top else "right", va="top" if top else "bottom", color=INK,
                fontsize=_font("legend.fontsize", 1.1))
        ax.add_line(Line2D((x0, x1), (sy, sy), transform=ax.transAxes, color=style["color"],
                           linestyle=style["linestyle"], linewidth=2.0))
    if linthresh is not None:
        ax.set_xscale("symlog", linthresh=float(linthresh))
        ax.axvline(0.0, color=MUTED, linewidth=1.0, linestyle=":")
    elif log:
        ax.set_xscale("log")
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("fraction of slices")
    name = xlabel or value
    ax.set_xlabel(f"|{name}|" if better == "zero" else name)
    _recessive(ax)
    _finish(fig, show)
    return fig, ax


def _pairs(table: pd.DataFrame, value: str, by: str, pair: str) -> pd.DataFrame:
    rows = table.loc[table[pair].notna(), [pair, by, value]].copy()
    rows[value] = pd.to_numeric(rows[value], errors="coerce")
    wide = rows.pivot_table(index=pair, columns=by, values=value, aggfunc="first")
    return wide.dropna()


def pipeline_comparison_paired(
    table: pd.DataFrame,
    value: str,
    *,
    by: str = "side",
    pair: str = "pair_id",
    better: str | None = "lower",
    log: bool = False,
    label: str | None = None,
    ax=None,
    show: bool = False,
    figsize: tuple[float, float] = (4.6, 4.6),
):
    """Legacy (x) against current (y) for every slice pair, with the line of no change.

    Points under the diagonal improved when ``better`` is set. The fraction of
    pairs on each side of it is written in the corner, not left to the eye.
    """
    fig, ax = _axes(ax, figsize)
    wide = _pairs(table, value, by, pair)
    if {"legacy", "current"} <= set(wide.columns):
        x = _oriented(wide["legacy"].to_numpy(dtype=float), better)
        y = _oriented(wide["current"].to_numpy(dtype=float), better)
        keep = np.isfinite(x) & np.isfinite(y)
        if log:
            keep &= (x > 0) & (y > 0)
        x, y = x[keep], y[keep]
    else:
        x = y = np.array([], dtype=float)
    ax.scatter(x, y, s=22, color=SIDE_STYLE["current"]["color"], alpha=0.55, edgecolors="white",
               linewidths=0.6, zorder=3)
    if x.size:
        low = float(min(x.min(), y.min()))
        high = float(max(x.max(), y.max()))
        if log:
            low, high = low / 1.3, high * 1.3
        else:
            pad = 0.05 * (high - low or 1.0)
            low, high = low - pad, high + pad
        ax.plot([low, high], [low, high], color=MUTED, linewidth=1.2, linestyle=":", zorder=2)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        if better is not None:
            improved = float(np.mean(y < x))
            ax.text(0.97, 0.04, f"current lower: {improved:.0%} of {x.size} pairs", transform=ax.transAxes,
                    ha="right", va="bottom", color=INK, fontsize=_font("legend.fontsize", 1.05))
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    name = label or value
    shown = f"|{name}|" if better == "zero" else name
    ax.set_xlabel(f"legacy  {shown}")
    ax.set_ylabel(f"current  {shown}")
    ax.set_aspect("equal", adjustable="box")
    _recessive(ax)
    _finish(fig, show)
    return fig, ax


def pipeline_comparison_measured_vs_reconstructed(
    points: pd.DataFrame,
    family: str,
    *,
    by: str = "side",
    sides: Sequence[str] = SIDES,
    scale: float = 1.0,
    unit: str = "",
    ax=None,
    show: bool = False,
    figsize: tuple[float, float] = (4.8, 4.8),
):
    """Measured (x) against reconstructed (y) for one constraint family, one colour per side.

    Each side is labelled with its point count (channels times slices) and the RMS of
    ``reconstructed - measured`` over its points, in the family's unit times
    ``scale``. Only the channels the fit used are in ``points``.
    """
    fig, ax = _axes(ax, figsize)
    data = points[points["family"] == family]
    extent = []
    lines = []
    for side in sides:
        rows = data[data[by] == side]
        x = pd.to_numeric(rows["measured"], errors="coerce").to_numpy(float) * scale
        y = pd.to_numeric(rows["reconstructed"], errors="coerce").to_numpy(float) * scale
        keep = np.isfinite(x) & np.isfinite(y)
        x, y = x[keep], y[keep]
        if not x.size:
            continue
        style = SIDE_STYLE[side]
        ax.scatter(x, y, s=14, marker=style["marker"], color=style["color"], alpha=0.45,
                   edgecolors="white", linewidths=0.4, zorder=3 if side == "current" else 2)
        extent += [x.min(), x.max(), y.min(), y.max()]
        rms = float(np.sqrt(np.mean((y - x) ** 2)))
        lines.append(f"{style['label']}: {x.size} points, RMS(r - m) {rms:.3g} {unit}".rstrip())
    if extent:
        low, high = min(extent), max(extent)
        pad = 0.05 * (high - low or 1.0)
        ax.plot([low - pad, high + pad], [low - pad, high + pad], color=MUTED, linestyle=":", linewidth=1.2)
        ax.set_xlim(low - pad, high + pad)
        ax.set_ylim(low - pad, high + pad)
    for k, line in enumerate(lines):
        ax.text(0.03, 0.97 - 0.07 * k, line, transform=ax.transAxes, ha="left", va="top", color=INK,
                fontsize=_font("legend.fontsize", 1.0))
    suffix = f" [{unit}]" if unit else ""
    ax.set_xlabel(f"measured{suffix}")
    ax.set_ylabel(f"reconstructed{suffix}")
    ax.set_aspect("equal", adjustable="box")
    _recessive(ax)
    _finish(fig, show)
    return fig, ax


def pipeline_comparison_scorecard(
    rows: Sequence[Mapping[str, Any]],
    *,
    ax=None,
    show: bool = False,
    figsize: tuple[float, float] | None = None,
):
    """Median current/legacy ratio per metric on one log axis; left of 1 is better.

    ``rows`` holds one mapping per metric with ``label``, ``legacy_median``,
    ``current_median`` (both on the oriented scale) and, from a paired summary,
    ``improved_fraction``, ``n_pairs`` and ``wilcoxon_p``. A metric with no
    direction is skipped: its ratio has no "better" side. The annotation beside
    each dot is the paired evidence, so the dot is never read alone.
    """
    usable = [r for r in rows if r.get("better") is not None
              and np.isfinite(r.get("legacy_median", np.nan)) and np.isfinite(r.get("current_median", np.nan))
              and r["legacy_median"] > 0 and r["current_median"] > 0]
    if figsize is None:
        figsize = (7.2, 0.55 * max(len(usable), 1) + 1.2)
    fig, ax = _axes(ax, figsize)
    ratios = [r["current_median"] / r["legacy_median"] for r in usable]
    positions = np.arange(len(usable))[::-1]
    for y, ratio, row in zip(positions, ratios, usable):
        ax.plot([1.0, ratio], [y, y], color=GRID, linewidth=3.0, solid_capstyle="round", zorder=1)
        ax.plot([ratio], [y], marker="o", markersize=9, color=SIDE_STYLE["current"]["color"],
                markeredgecolor="white", markeredgewidth=2.0, zorder=3)
        parts = [f"{ratio:.2g}x"]
        if row.get("improved_fraction") is not None and row.get("n_pairs"):
            parts.append(f"{row['improved_fraction']:.0%} of {row['n_pairs']} pairs improved")
        p = row.get("wilcoxon_p")
        if p is not None and np.isfinite(p):
            parts.append("p < 1e-3" if p < 1e-3 else f"p = {p:.2g}")
        ax.annotate("  ".join(parts), (max(ratio, 1.0), y), xytext=(10, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=_font("legend.fontsize", 1.05))
    ax.axvline(1.0, color=MUTED, linewidth=1.2, linestyle=":")
    ax.set_xscale("log")
    ax.set_yticks(positions)
    ax.set_yticklabels([r["label"] for r in usable])
    if ratios:
        low = min(min(ratios), 1.0) / 1.6
        high = max(max(ratios), 1.0) * 60.0
        ax.set_xlim(low, high)
    ax.set_xlabel("current median / legacy median  (left of 1 is better)")
    _recessive(ax)
    ax.grid(False, axis="y")
    _finish(fig, show)
    return fig, ax
