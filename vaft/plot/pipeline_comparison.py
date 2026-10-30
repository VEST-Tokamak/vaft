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
from .presentation import presented, resolve_presentation

__all__ = [
    "pipeline_comparison_distribution",
    "pipeline_comparison_measured_vs_reconstructed",
    "pipeline_comparison_onset_population",
    "pipeline_comparison_onset_scatter",
    "pipeline_comparison_onset_traces",
    "pipeline_comparison_overview",
    "pipeline_comparison_paired",
    "pipeline_comparison_quality_checks",
    "pipeline_comparison_scorecard",
    "pipeline_comparison_stability",
    "pipeline_comparison_stability_overview",
    "pipeline_comparison_thomson",
    "pipeline_comparison_virial",
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


def _thin_symlog_ticks(ax, linthresh: float, max_ticks: int = 9) -> None:
    """Decade ticks on a symlog axis, every k-th decade so at most ``max_ticks`` show."""
    from matplotlib.ticker import NullLocator

    low, high = ax.get_xlim()
    base = np.log10(linthresh)
    top = [int(np.ceil(np.log10(v))) if v > linthresh else None for v in (abs(low), abs(high))]
    neg = range(int(np.floor(base)), (top[0] or int(base)) + 1) if low < -linthresh else range(0)
    pos = range(int(np.floor(base)), (top[1] or int(base)) + 1) if high > linthresh else range(0)
    count = len(neg) + len(pos) + 1
    step = max(1, int(np.ceil(count / max_ticks)))
    ticks = [-(10.0 ** d) for d in list(neg)[::-1][::step]][::-1] + [0.0] + [10.0 ** d for d in list(pos)[::step]]
    ax.set_xticks([t for t in ticks if low <= t <= high])
    ax.xaxis.set_minor_locator(NullLocator())


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
    # Each side is named in a row above the axes, with a line sample, so no
    # label can sit on a curve and identity is not colour alone.
    from matplotlib.lines import Line2D

    for k, (median, count, style) in enumerate(curves):
        x0 = 0.0 + 0.5 * k
        ax.add_line(Line2D((x0, x0 + 0.07), (1.075, 1.075), transform=ax.transAxes, color=style["color"],
                           linestyle=style["linestyle"], linewidth=2.0, clip_on=False))
        ax.text(x0 + 0.09, 1.04, f"{style['label']}\nn = {count}, median {median:.3g}", transform=ax.transAxes,
                ha="left", va="bottom", color=INK, fontsize=_font("legend.fontsize", 1.0))
    if linthresh is not None:
        ax.set_xscale("symlog", linthresh=float(linthresh))
        ax.axvline(0.0, color=MUTED, linewidth=1.0, linestyle=":")
        _thin_symlog_ticks(ax, float(linthresh))
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


#: Display of each constraint family in the comparison tables (SI values):
#: title, unit and the factor from SI, as in :mod:`vaft.plot.equilibrium_quality`.
FAMILY_DISPLAY = {
    "probe": ("Poloidal probes", "mT", 1e3),
    "loop": ("Flux loops", "mWb", 1e3),
    "ip": ("Plasma current", "kA", 1e-3),
    "dia": ("Diamagnetic flux", "mWb", 1e3),
}
#: Volume-integral and pair-13 virial columns of the virial table, per quantity.
VIRIAL_QUANTITIES = {
    "beta_p": ("beta_p_integral", "beta_p_pair_13", r"\beta_p", "Poloidal beta", None),
    "li": ("li_volume", "li_pair_13", r"l_i", "Internal inductance", None),
}
COMPARISON_LABELS = {"legacy": "Legacy archive", "current": "Current pipeline"}


def _side_points(points: pd.DataFrame, family: str, by: str) -> pd.DataFrame:
    """Rows of one family in display units: per-channel points, or the single-channel columns of the fit table."""
    _title, _unit, scale = FAMILY_DISPLAY[family]
    if "family" in points:
        data = points[points["family"] == family]
        x, y = data["measured"], data["reconstructed"]
    else:
        data = points
        x, y = data[f"{family}_measured"], data[f"{family}_reconstructed"]
    return pd.DataFrame({by: data[by].to_numpy(),
                         "x": pd.to_numeric(x, errors="coerce").to_numpy(float) * scale,
                         "y": pd.to_numeric(y, errors="coerce").to_numpy(float) * scale})


def _limits(limits):
    values = np.asarray(limits, dtype=float)
    if values.shape != (2,) or not np.isfinite(values).all() or values[0] >= values[1]:
        raise ValueError("limits must be two finite increasing values")
    return tuple(values)


@presented(default_figsize=(5.2, 5.0))
def pipeline_comparison_measured_vs_reconstructed(points: pd.DataFrame, *, family: str, by: str = "side",
                                                  ax=None, show: bool = False, figsize=None,
                                                  format: str | None = None, theme: str | None = None,
                                                  limits=None):
    """Measured (x) against reconstructed (y) for one constraint family, legacy and current, with ``y = x``.

    The view of :func:`vaft.plot.equilibrium_quality.equilibrium_quality_measured_vs_reconstructed`
    with the pipeline as the cohort. ``points`` is either the per-channel
    ``constraint_points`` table (``family``, ``measured``, ``reconstructed``,
    SI) or, for ``family="ip"`` / ``"dia"``, the per-slice fit table with
    ``<family>_measured`` / ``<family>_reconstructed``. The legend gives each
    side's point count and mean absolute error in the display unit; points
    outside ``limits=(min, max)`` still count.
    """
    if family not in FAMILY_DISPLAY:
        raise ValueError(f"unknown family {family!r}; known: {sorted(FAMILY_DISPLAY)}")
    title, unit, _scale = FAMILY_DISPLAY[family]
    limits = None if limits is None else _limits(limits)
    fig, ax = _axes(ax, figsize or (5.2, 5.0))
    data = _side_points(points, family, by)
    finite = np.isfinite(data["x"]) & np.isfinite(data["y"])
    import matplotlib.pyplot as plt

    font_scale = plt.rcParams["font.size"] / 10.0
    hidden = []
    for layer, side in enumerate(SIDES, start=1):
        use = (data[by] == side).to_numpy() & finite.to_numpy()
        if not use.any():
            continue
        x, y = data["x"].to_numpy()[use], data["y"].to_numpy()[use]
        if limits is not None:
            outside = (x < limits[0]) | (x > limits[1]) | (y < limits[0]) | (y > limits[1])
            hidden.append(f"{COMPARISON_LABELS[side].split()[0].lower()} {int(outside.sum()):,}")
        style = SIDE_STYLE[side]
        dense = use.sum() > 2000
        ax.scatter(x, y, s=(3 if dense else 14) * font_scale ** 2, marker=style["marker"], color=style["color"],
                   alpha=(0.15 if dense else 0.6), zorder=layer + 1, linewidths=0 if dense else 0.4,
                   edgecolors=None if dense else "white", rasterized=dense,
                   label=f"{COMPARISON_LABELS[side]} (n={int(use.sum()):,}, MAE {np.mean(np.abs(y - x)):.2f} {unit})")
    if finite.any():
        lo = float(np.nanmin(np.r_[data["x"][finite], data["y"][finite]]))
        hi = float(np.nanmax(np.r_[data["x"][finite], data["y"][finite]]))
        pad = 0.05 * (hi - lo or 1.0)
        reference = limits if limits is not None else (lo - pad, hi + pad)
        ax.plot(reference, reference, color=INK, lw=1.0, ls="--", label="y = x", zorder=5)
        ax.set_xlim(reference)
        ax.set_ylim(reference)
    if hidden:
        ax.text(0.97, 0.03, "not shown: " + ", ".join(hidden), transform=ax.transAxes, ha="right", va="bottom",
                color=MUTED, fontsize=plt.rcParams["font.size"] * 0.75, zorder=6,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.0})
    ax.set_xlabel(f"measured [{unit}]")
    ax.set_ylabel(f"reconstructed [{unit}]")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", frameon=False, markerscale=2.0)
    _finish(fig, show)
    return fig, ax


@presented(default_figsize=(5.2, 5.0))
def pipeline_comparison_virial(table: pd.DataFrame, *, quantity: str = "beta_p", by: str = "side", ax=None,
                               show: bool = False, figsize=None, format: str | None = None,
                               theme: str | None = None, limits=None, log: bool | None = None,
                               band: float | None | str = "auto"):
    """Volume-integral against pair-13 virial ``beta_p`` or ``l_i``, legacy and current, with ``y = x``.

    The view of :func:`vaft.plot.kinetic_state.kinetic_state_virial_pair13_comparison`
    with the pipeline as the cohort, on the virial table of the comparison
    (one row per side and slice). ``band`` shades agreement within that
    factor; ``"auto"`` draws the QA criterion where one exists: for ``beta_p`` the
    virial check of :data:`vaft.validation._efit_study_criteria.CRITERIA`,
    ``|ln(beta_p^int / beta_p^vir)| <= virial_log_ratio_max`` (ln 2), and none
    for ``l_i``, which no criterion grades against its closure (only the
    plausibility bound ``0 < l_i <= 3``). ``log`` defaults to ``True`` for ``beta_p``, whose two pipelines
    differ by an order of magnitude. Each legend entry counts the slices
    outside the view (a non-positive value on a log axis among them), so no
    slice is dropped silently.
    """
    if quantity not in VIRIAL_QUANTITIES:
        raise ValueError(f"unknown quantity {quantity!r}; known: {sorted(VIRIAL_QUANTITIES)}")
    volume, closure, symbol, title, _ = VIRIAL_QUANTITIES[quantity]
    criterion = band == "auto"
    if criterion:
        band = _virial_criterion_band(quantity)
    log = (quantity == "beta_p") if log is None else bool(log)
    fig, ax = _axes(ax, figsize or (5.2, 5.0))
    x_all = pd.to_numeric(table[closure], errors="coerce").to_numpy(float)
    y_all = pd.to_numeric(table[volume], errors="coerce").to_numpy(float)
    finite = np.isfinite(x_all) & np.isfinite(y_all)
    usable = finite & ((x_all > 0) & (y_all > 0) if log else True)
    if limits is None:
        if not usable.any():
            raise ValueError(f"no finite {quantity} pairs to draw")
        lo = float(np.percentile(np.r_[x_all[usable], y_all[usable]], 0.5))
        hi = float(np.percentile(np.r_[x_all[usable], y_all[usable]], 99.5))
        limits = (lo / 1.5, hi * 1.5) if log else (lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
    limits = _limits(limits)
    if log and limits[0] <= 0:
        raise ValueError("logarithmic limits must be positive")
    import matplotlib.pyplot as plt

    font_scale = plt.rcParams["font.size"] / 10.0
    inside = finite & (x_all >= limits[0]) & (x_all <= limits[1]) & (y_all >= limits[0]) & (y_all <= limits[1])
    hidden = []
    for layer, side in enumerate(SIDES, start=1):
        rows = (table[by] == side).to_numpy()
        use = rows & inside
        if not rows.any():
            continue
        style = SIDE_STYLE[side]
        outside = int((rows & ~inside).sum())
        hidden.append(f"{COMPARISON_LABELS[side].split()[0].lower()} {outside:,}")
        note = f", {outside:,} not shown" if outside else ""
        ax.scatter(x_all[use], y_all[use], s=6 * font_scale ** 2, marker=style["marker"], color=style["color"],
                   alpha=0.35, linewidths=0, zorder=layer + 1, rasterized=True,
                   label=f"{COMPARISON_LABELS[side]} (n={int(rows.sum()):,}{note})")
    grid = np.geomspace(*limits, 100) if log else np.linspace(*limits, 100)
    if band:
        label = (rf"QA: $|\ln({symbol}^{{\mathrm{{int}}}}/{symbol}^{{\mathrm{{vir}}}})| \leq \ln {band:g}$"
                 if criterion else f"within a factor of {band:g}")
        ax.fill_between(grid, grid / band, grid * band, color="gray", alpha=0.14, zorder=1, label=label)
    ax.plot(grid, grid, color=INK, ls="--", lw=1.0, label="y = x", zorder=5)
    # Written on the axes as well as in the legend, so a grid that drops the
    # legends still says how many slices fall outside the view.
    # "Not shown" is outside the view or undefined (no virial value, or <= 0 on a log axis).
    ax.text(0.97, 0.03, "not shown: " + ", ".join(hidden), transform=ax.transAxes, ha="right", va="bottom",
            color=MUTED, fontsize=plt.rcParams["font.size"] * 0.75, zorder=6,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.0})
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    else:
        from matplotlib.ticker import MaxNLocator

        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel(rf"virial closure ${symbol}^{{\mathrm{{vir}}}}$")
    ax.set_ylabel(rf"volume integral ${symbol}^{{\mathrm{{int}}}}$")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", frameon=False, markerscale=1.5)
    _finish(fig, show)
    return fig, ax


def _virial_criterion_band(quantity: str) -> float | None:
    """The agreement factor the virial QA criterion allows, or ``None`` when no criterion grades ``quantity``."""
    if quantity != "beta_p":
        return None
    from vaft.validation._efit_study_criteria import CRITERIA

    return float(np.exp(CRITERIA["virial_log_ratio_max"]))


def _panel_letter(ax, letter: str, y: float = 1.04, x: float = -0.18, scale: float = 1.2) -> None:
    import matplotlib.pyplot as plt

    ax.text(x, y, letter, transform=ax.transAxes, ha="left", va="bottom", fontweight="bold",
            fontsize=plt.rcParams["font.size"] * scale)


def pipeline_comparison_overview(points: pd.DataFrame, virial: pd.DataFrame, *, show: bool = False,
                                 figsize: tuple[float, float] = (9.8, 10.0), limits_by_panel=None,
                                 panel_labels: bool = True):
    """A presentation-size 2×2: probes and flux loops, then ``beta_p`` and ``l_i`` against the virial closure.

    The layout of :func:`vaft.plot.equilibrium_quality.equilibrium_quality_diagnostic_grid`,
    one shared legend under the panels; the single panels keep their counts
    and errors. ``limits_by_panel`` maps ``"probe"``, ``"loop"``, ``"beta_p"``
    or ``"li"`` to ``(min, max)``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    limits = dict(limits_by_panel or {})
    unknown = set(limits) - {"probe", "loop", "beta_p", "li"}
    if unknown:
        raise ValueError(f"unknown panel in limits_by_panel: {sorted(unknown)}")
    presentation = resolve_presentation("slide", None)
    with presentation.context():
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        for ax, family in zip(axes[0], ("probe", "loop")):
            pipeline_comparison_measured_vs_reconstructed(points, family=family, ax=ax, limits=limits.get(family))
        for ax, quantity in zip(axes[1], ("beta_p", "li")):
            pipeline_comparison_virial(virial, quantity=quantity, ax=ax, limits=limits.get(quantity))
        for letter, ax in zip("abcd", axes.ravel()):
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
            if panel_labels:
                _panel_letter(ax, letter)
        handles = [Line2D([], [], color=SIDE_STYLE[side]["color"], marker=SIDE_STYLE[side]["marker"],
                          linestyle="none", markersize=10, label=COMPARISON_LABELS[side]) for side in SIDES]
        handles.append(Line2D([], [], color=INK, linestyle="--", label="y = x"))
        band = _virial_criterion_band("beta_p")
        if band:
            handles.append(Patch(facecolor="gray", alpha=0.14, edgecolor="none",
                                 label=rf"QA in (c): $|\ln(\beta_p^{{\mathrm{{int}}}}/\beta_p^{{\mathrm{{vir}}}})| \leq \ln {band:g}$"))
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=2, frameon=False)
        fig.tight_layout(rect=(0.0, 0.11, 1.0, 1.0), pad=0.8, w_pad=0.8, h_pad=1.2)
    if show:
        plt.show()
    return fig, axes


#: Stability quantities: binning and ticks of :mod:`vaft.plot.stability_atlas`, the
#: side of zero that is unstable, and the axis label.
STABILITY_QUANTITIES = {
    "w_t": ("ideal", False, r"$\delta W$", (-1e3, -1.0, 0.0, 1.0, 1e2), "Ideal MHD stability (DCON)"),
    "delta_prime": ("resistive", True, r"$\Delta'$", None, "Resistive MHD stability (RDCON)"),
}


#: Line style per toroidal mode; the pipeline is the colour.
MODE_LINESTYLES = ("-", "--", ":", "-.", (0, (5, 1, 1, 1, 1, 1)), (0, (1, 3)))


def _mode_styles(modes) -> dict[int, Any]:
    """One line style per toroidal mode, in mode order."""
    return {n: MODE_LINESTYLES[i % len(MODE_LINESTYLES)] for i, n in enumerate(sorted(int(n) for n in modes))}


def pipeline_comparison_stability(table: pd.DataFrame, *, quantity: str = "w_t", value: str = "value",
                                  by: str = "side", mode: str = "n", modes=None, styles=None,
                                  normalize: bool = True, ax=None, show: bool = False,
                                  figsize: tuple[float, float] = (5.2, 3.0)):
    """Legacy and current populations of ``delta W`` or ``Delta'`` per toroidal mode, on a symlog axis.

    The panel of :func:`vaft.plot.stability_atlas.stability_atlas_population`,
    with its bins, ticks and stable/unstable zones. Colour is the pipeline
    (legacy archive and current pipeline, as in every figure of this module);
    the toroidal mode (column ``mode``) is the line style, solid, dashed,
    dotted, dash-dotted in mode order. The
    count axis is linear. ``normalize`` divides every (pipeline, mode) curve by
    its own number of slices, because the two populations differ in size by
    more than an order of magnitude. A value past the outermost bin is drawn in
    the edge bin, as the atlas figure does. The key is left to the caller
    (:func:`pipeline_comparison_stability_overview` draws one for both panels);
    each step carries a ``"<side>:<mode>"`` label.
    """
    from .stability_atlas import _BINNING, _TICKS, _prepare, _symlog_bins, _zones

    if quantity not in STABILITY_QUANTITIES:
        raise ValueError(f"unknown quantity {quantity!r}; known: {sorted(STABILITY_QUANTITIES)}")
    panel, unstable_right, xlabel, ticks, title = STABILITY_QUANTITIES[quantity]
    fig, ax = _axes(ax, figsize)
    lowest, highest, linthresh, per_decade = _BINNING[panel]
    bins = _symlog_bins(lowest, highest, linthresh, per_decade)
    # The atlas's ticks, thinned where a slide-size font would make them touch.
    _prepare(ax, bins, linthresh, ticks or _TICKS[panel])
    present = sorted(int(n) for n in pd.unique(table[mode].dropna()))
    modes = present if modes is None else [int(n) for n in modes]
    styles = styles or _mode_styles(modes)
    tallest = 0.0
    for side in SIDES:
        for n in modes:
            rows = (table[by] == side) & (table[mode] == n)
            values = pd.to_numeric(table.loc[rows, value], errors="coerce").to_numpy(float)
            values = values[np.isfinite(values)]
            if not values.size:
                continue
            values = np.clip(values, bins[0], bins[-1])
            weights = np.full(values.size, 1.0 / values.size) if normalize else None
            counts, _, _ = ax.hist(values, bins=bins, weights=weights, histtype="step", lw=1.6,
                                   color=SIDE_STYLE[side]["color"], linestyle=styles[n], label=f"{side}:{n}")
            tallest = max(tallest, float(np.max(counts)))
    # Headroom above the tallest bar for the zone names.
    ax.set_ylim(0, max(tallest, 1e-9) * 1.3)
    ax.set_ylabel("fraction of slices" if normalize else "slices")
    _zones(ax, unstable_right=unstable_right)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    _finish(fig, show)
    return fig, ax


def pipeline_comparison_stability_overview(w_t: pd.DataFrame, delta_prime: pd.DataFrame, *,
                                           normalize: bool = True, show: bool = False,
                                           figsize: tuple[float, float] = (8.0, 7.6),
                                           panel_labels: bool = True):
    """``delta W`` over ``Delta'``, legacy and current, per toroidal mode, stacked as the atlas figure is.

    Both tables are long form (``side``, ``n``, ``value``). One line style per
    mode across both panels; the colour is the pipeline. Both keys sit under
    the panels: the pipeline row, then a "toroidal mode" row.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    modes = sorted({int(n) for n in pd.unique(w_t["n"].dropna())} | {int(n) for n in pd.unique(delta_prime["n"].dropna())})
    styles = _mode_styles(modes)
    presentation = resolve_presentation("slide", None)
    with presentation.context():
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        pipeline_comparison_stability(w_t, quantity="w_t", styles=styles, normalize=normalize, ax=axes[0])
        pipeline_comparison_stability(delta_prime, quantity="delta_prime", styles=styles, normalize=normalize,
                                      ax=axes[1])
        if panel_labels:
            for letter, ax in zip("ab", axes):
                _panel_letter(ax, letter)
        pipelines = [Line2D([], [], color=SIDE_STYLE[side]["color"], lw=3.0, label=COMPARISON_LABELS[side])
                     for side in SIDES]
        # Both keys sit under the panels, one row each: the pipelines, then the
        # modes, whose row is headed by an inline "toroidal mode" label.
        heading = Line2D([], [], linestyle="none", label="toroidal mode")
        swatches = [heading] + [Line2D([], [], color=INK, linestyle=styles[n], lw=1.6, label=str(n)) for n in modes]
        first = fig.legend(handles=pipelines, loc="lower center", bbox_to_anchor=(0.5, 0.045), ncol=2, frameon=False)
        fig.legend(handles=swatches, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=len(swatches),
                   frameon=False, handlelength=2.4, columnspacing=1.2, handletextpad=0.4)
        fig.add_artist(first)
        fig.tight_layout(rect=(0.0, 0.11, 1.0, 1.0), pad=0.8, h_pad=1.4)
    if show:
        plt.show()
    return fig, axes


@presented(default_figsize=(6.3, 5.5))
def pipeline_comparison_thomson(channels: pd.DataFrame, *, by: str = "side", paired_only: bool = True,
                                ax=None, show: bool = False, figsize=None, format: str | None = None,
                                theme: str | None = None, limits=None, errorbars: bool = True):
    """EFIT pressure at each matched Thomson channel against ``2 p_e``, legacy and current.

    The view of :func:`vaft.plot.kinetic_state.kinetic_state_pressure_comparison`
    with the pipeline as the colour: the band ``p_e <= p_EFIT <= 2 p_e``
    (``x/2 <= y <= x`` for ``x = 2 p_e``), ``p_EFIT = 2 p_e`` dashed and
    ``p_EFIT = p_e`` dotted, on log axes. ``channels`` is the comparison's
    ``thomson_channels`` table (``side``, ``pair_id``, ``p_e``, ``p_recon`` in
    Pa). ``paired_only`` keeps the slices matched by time on both sides. A
    channel with a non-positive pressure has no place on a log axis; it is
    counted on the panel, not dropped silently. The band is the Thomson QA
    criterion, ``thomson_log_ratio_band`` of
    :data:`vaft.validation._efit_study_criteria.CRITERIA`; the figure grades no point.

    ``errorbars`` draws a translucent horizontal bar on every point that has
    ``n_e_error`` and ``t_e_error``: ``2 sigma(p_e)`` with
    ``sigma(p_e)/p_e = sqrt((sigma_n/n_e)^2 + (sigma_T/T_e)^2)`` (the stored
    ``data_error_upper`` taken as a symmetric 1-sigma, the two independent),
    the lower arm clipped at the axis. EFIT gives ``p_EFIT`` no uncertainty,
    so there is no vertical bar.
    """
    from vaft.validation._efit_study_criteria import CRITERIA
    import matplotlib.pyplot as plt

    data = channels[channels["pair_id"].notna()] if paired_only else channels
    x_all = 2.0 * pd.to_numeric(data["p_e"], errors="coerce").to_numpy(float)
    y_all = pd.to_numeric(data["p_recon"], errors="coerce").to_numpy(float)
    positive = np.isfinite(x_all) & np.isfinite(y_all) & (x_all > 0) & (y_all > 0)
    if not positive.any():
        raise ValueError("Thomson comparison needs positive finite matched-channel pressures")
    if limits is None:
        values = np.r_[x_all[positive], y_all[positive]]
        limits = (float(values.min()) / 1.3, float(values.max()) * 1.3)
    limits = _limits(limits)
    if limits[0] <= 0:
        raise ValueError("logarithmic limits must be positive")
    fig, ax = _axes(ax, figsize or (6.3, 5.5))
    font_scale = plt.rcParams["font.size"] / 10.0
    if errorbars and {"n_e", "t_e", "n_e_error", "t_e_error"} <= set(data.columns):
        relative = np.hypot(pd.to_numeric(data["n_e_error"], errors="coerce").to_numpy(float)
                            / pd.to_numeric(data["n_e"], errors="coerce").to_numpy(float),
                            pd.to_numeric(data["t_e_error"], errors="coerce").to_numpy(float)
                            / pd.to_numeric(data["t_e"], errors="coerce").to_numpy(float))
        x_error = x_all * relative
    else:
        x_error = np.full_like(x_all, np.nan)
    hidden = []
    for layer, side in enumerate(SIDES, start=1):
        rows = (data[by] == side).to_numpy()
        if not rows.any():
            continue
        inside = positive & (x_all >= limits[0]) & (x_all <= limits[1]) & (y_all >= limits[0]) & (y_all <= limits[1])
        use = rows & inside
        hidden.append(f"{COMPARISON_LABELS[side].split()[0].lower()} {int((rows & ~inside).sum()):,}")
        style = SIDE_STYLE[side]
        bar = use & np.isfinite(x_error) & (x_error > 0)
        if bar.any():
            upper = x_error[bar]
            lower = np.minimum(upper, x_all[bar] - limits[0])
            ax.errorbar(x_all[bar], y_all[bar], xerr=[lower, upper], fmt="none", ecolor=style["color"],
                        elinewidth=0.9 * font_scale, alpha=0.25, capsize=0, zorder=layer)
        ax.scatter(x_all[use], y_all[use], s=19 * font_scale ** 2, marker=style["marker"], color=style["color"],
                   alpha=0.62, linewidths=0, zorder=layer + 1, label=COMPARISON_LABELS[side])
    grid = np.geomspace(*limits, 100)
    low, high = CRITERIA["thomson_log_ratio_band"]
    ax.fill_between(grid, grid * np.exp(low), grid * np.exp(high), color="gray", alpha=0.14, zorder=1,
                    label=r"QA: $p_e \leq p_{EFIT} \leq 2p_e$")
    ax.plot(grid, grid / 2, color="0.4", ls=":", lw=1.2 * font_scale, label=r"$p_{EFIT}=p_e$", zorder=4)
    ax.plot(grid, grid, color=INK, ls="--", lw=1.2 * font_scale, label=r"$p_{EFIT}=2p_e$", zorder=4)
    if any(not h.endswith(" 0") for h in hidden):
        ax.text(0.97, 0.03, "not shown: " + ", ".join(hidden), transform=ax.transAxes, ha="right", va="bottom",
                color=MUTED, fontsize=plt.rcParams["font.size"] * 0.75, zorder=6,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.0})
    ax.set(xscale="log", yscale="log", xlim=limits, ylim=limits,
           xlabel=r"$2p_e$ from Thomson [Pa]", ylabel=r"$p_{EFIT}$ at TS channel [Pa]",
           title="Matched Thomson channels")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
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


#: The onset tables name the HSDS source; the figures name the pipeline.
ONSET_SOURCES = {"public": "legacy", "main": "current"}
#: Panels of the onset figures: family, title, y label and display scale of a trace.
ONSET_PANELS = (
    ("ip", "Plasma current", r"$I_p$ [kA]", 1.0),
    ("probe", "Poloidal probes", "[mT]", 1e3),
    ("loop", "Flux loops", "[mWb]", 1e3),
)


def _onset_axes(n: int, figsize):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, n, figsize=figsize, sharex=True)
    for ax in np.atleast_1d(axes):
        ax.axvline(0.0, color=MUTED, ls=":", lw=1.2, zorder=1)
        ax.set_xlabel(r"$t - t_\mathrm{onset}$ [ms]")
    return fig, np.atleast_1d(axes)


def _onset_legend(fig, extra) -> None:
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=SIDE_STYLE[side]["color"], lw=3.0, label=COMPARISON_LABELS[side])
               for side in SIDES]
    handles += list(extra)
    handles.append(Line2D([], [], color=MUTED, ls=":", lw=1.2, label="plasma onset"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=len(handles), frameon=False)
    fig.tight_layout(rect=(0.0, 0.1, 1.0, 1.0), pad=0.8, w_pad=1.0)


def pipeline_comparison_onset_population(onset: pd.DataFrame, *, sources: Mapping[str, str] = ONSET_SOURCES,
                                         show: bool = False, figsize: tuple[float, float] = (15.0, 4.8),
                                         panel_labels: bool = True):
    """Measured against vacuum magnetics over many shots, on one clock aligned at the plasma onset.

    ``onset`` is the onset table of the comparison: one row per source, shot,
    family and ``t_ms`` (``t - t_onset``), where ``value`` is, for ``probe``
    and ``loop``, the shot's median over channels of
    ``|measured - vacuum (coil+eddy)|`` over the channel's ``max|measured|``
    before the onset, and
    for ``ip`` the measured plasma current [kA], whose vacuum value is zero.
    Each side draws the median over shots and shades the interquartile range;
    the residual panels are logarithmic. Before the onset the residual is the
    vacuum model's error; after it, the plasma's own signal.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    data = onset.assign(side=onset["source"].map(sources))
    presentation = resolve_presentation("slide", None)
    with presentation.context():
        fig, axes = _onset_axes(len(ONSET_PANELS), figsize)
        for ax, (family, title, unit, _scale) in zip(axes, ONSET_PANELS):
            for side in SIDES:
                rows = data[(data["side"] == side) & (data["family"] == family)]
                if rows.empty:
                    continue
                table = rows.pivot_table(index="t_ms", columns="shot", values="value")
                quartiles = table.quantile([0.25, 0.5, 0.75], axis=1).T
                color = SIDE_STYLE[side]["color"]
                ax.fill_between(quartiles.index, quartiles[0.25], quartiles[0.75], color=color, alpha=0.18, lw=0)
                ax.plot(quartiles.index, quartiles[0.5], color=color, lw=2.0,
                        label=f"{COMPARISON_LABELS[side]} ({table.shape[1]:,} shots)")
            if family == "ip":
                ax.axhline(0.0, color=INK, ls="--", lw=1.0)
                ax.set_ylabel(unit)
            else:
                ax.set_yscale("log")
                ax.set_ylabel(r"$|m - m_\mathrm{vac}|\,/\,\max|m|$")
            ax.set_title(title)
        if panel_labels:
            for letter, ax in zip("abc", axes):
                _panel_letter(ax, letter)
        _onset_legend(fig, [Patch(facecolor=MUTED, alpha=0.3, label="interquartile range over shots")])
    if show:
        plt.show()
    return fig, axes


def pipeline_comparison_onset_traces(traces: pd.DataFrame, *, shot: int, channels: Mapping[str, str] | None = None,
                                     sources: Mapping[str, str] = ONSET_SOURCES, show: bool = False,
                                     figsize: tuple[float, float] = (15.0, 4.8), panel_labels: bool = True):
    """One shot's plasma current, one probe and one flux loop, measured (solid) and vacuum (dashed).

    ``traces`` is the onset-traces table (``source``, ``shot``, ``family``,
    ``channel``, ``t_ms``, ``measured``, ``vacuum``; SI, ``ip`` in kA). The
    vacuum of ``ip`` is zero. ``channels`` maps ``probe`` and ``loop`` to a
    channel name; unset, the typical channel: of those both sides carry, the
    one whose RMS ``|measured - vacuum| / max|measured|`` before the onset,
    averaged over the sides, is the family's median (the largest field would
    pick an outlier as often as not). Probes are drawn in mT and loops in mWb.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    data = traces[traces["shot"] == shot].assign(side=lambda t: t["source"].map(sources))
    if data.empty:
        raise ValueError(f"no onset traces for shot {shot}")
    chosen = dict(channels or {})
    for family in ("probe", "loop"):
        if family in chosen:
            continue
        rows = data[(data["family"].str.contains("flux_loop") == (family == "loop")) & (data["family"] != "ip")
                    & (data["t_ms"] < 0)]
        both = rows.groupby("channel")["side"].nunique()
        rows = rows[rows["channel"].isin(both[both == len(SIDES)].index)]
        if rows.empty:
            raise ValueError(f"shot {shot} has no {family} channel on both sides")
        grouped = rows.groupby(["channel", "side"])
        error = np.sqrt(grouped.apply(lambda g: np.nanmean((g["measured"] - g["vacuum"]) ** 2), include_groups=False))
        error = (error / grouped["measured"].apply(lambda m: np.nanmax(np.abs(m)))).groupby("channel").mean().dropna()
        if error.empty:
            raise ValueError(f"shot {shot} has no finite {family} channel before the onset")
        ranked = error.sort_values()
        chosen[family] = ranked.index[(len(ranked) - 1) // 2]
    presentation = resolve_presentation("slide", None)
    with presentation.context():
        fig, axes = _onset_axes(len(ONSET_PANELS), figsize)
        for ax, (family, title, unit, scale) in zip(axes, ONSET_PANELS):
            name = "ip" if family == "ip" else chosen[family]
            for side in SIDES:
                rows = data[(data["side"] == side) & (data["channel"] == name)].sort_values("t_ms")
                color = SIDE_STYLE[side]["color"]
                ax.plot(rows["t_ms"], rows["measured"] * scale, color=color, lw=1.8)
                if family != "ip":
                    ax.plot(rows["t_ms"], rows["vacuum"] * scale, color=color, lw=1.8, ls="--")
            if family == "ip":
                ax.axhline(0.0, color=INK, ls="--", lw=1.0)
            ax.set_ylabel(unit)
            ax.set_title(title)
            if family != "ip":
                from vaft.plot.backend.recipes import channel_short_name

                ax.text(0.03, 0.04, channel_short_name(name), transform=ax.transAxes, ha="left", va="bottom",
                        color=MUTED, fontsize=plt.rcParams["font.size"] * 0.8)
        if panel_labels:
            for letter, ax in zip("abc", axes):
                _panel_letter(ax, letter)
        fig.suptitle(f"Shot {shot}", y=1.0)
        _onset_legend(fig, [Line2D([], [], color=INK, lw=1.8, label="measured"),
                            Line2D([], [], color=INK, lw=1.8, ls="--", label="vacuum (coil + eddy)")])
    if show:
        plt.show()
    return fig, axes


def _onset_points(points: pd.DataFrame, sources: Mapping[str, str], t_ms: float | None):
    """The onset points as the measured-vs-reconstructed renderer reads them, and which the input QA rejects."""
    if t_ms is not None:
        points = points[np.isclose(pd.to_numeric(points["t_ms"], errors="coerce"), t_ms)]
        if points.empty:
            raise ValueError(f"no onset points at t - t_onset = {t_ms} ms")
    data = points.assign(side=points["source"].map(sources),
                         reconstructed=pd.to_numeric(points["vacuum"], errors="coerce"))
    # The table holds I_p in kA; the shared renderer takes SI.
    is_ip = (data["family"] == "ip").to_numpy()
    for column in ("measured", "reconstructed"):
        values = pd.to_numeric(data[column], errors="coerce").to_numpy(float)
        data[column] = np.where(is_ip, values * 1e3, values)
    code = pd.to_numeric(data["qa_code"], errors="coerce") if "qa_code" in data else pd.Series(np.nan, data.index)
    return data, (code < 0).to_numpy()


def _onset_panel(ax, data: pd.DataFrame, rejected: np.ndarray, family: str, limits,
                 show_rejected: bool = True) -> None:
    """One family of the onset points: accepted points as dots, input-QA rejects as crosses, counted.

    ``show_rejected=False`` leaves the rejects out altogether: no crosses, no count.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    pipeline_comparison_measured_vs_reconstructed(data[~rejected], family=family, ax=ax, limits=limits)
    font_scale = plt.rcParams["font.size"] / 10.0
    if not show_rejected:
        rejected = np.zeros_like(rejected)
    failed = data[rejected & (data["family"] == family).to_numpy()]
    scale = FAMILY_DISPLAY[family][2]
    counts = []
    for layer, side in enumerate(SIDES, start=1):
        rows = failed[failed["side"] == side]
        counts.append(f"{COMPARISON_LABELS[side].split()[0].lower()} {len(rows):,}")
        if len(rows):
            # Small, faint and under the accepted points: the rejects show where
            # the input QA acts without hiding what it lets through.
            dense = len(rows) > 500
            ax.scatter(rows["measured"].to_numpy(float) * scale, rows["reconstructed"].to_numpy(float) * scale,
                       marker="x", s=(5 if dense else 14) * font_scale ** 2, color=SIDE_STYLE[side]["color"],
                       alpha=0.25 if dense else 0.7, linewidths=0.5 if dense else 0.9, zorder=layer * 0.5,
                       rasterized=dense)
    if rejected.any() and family != "ip":
        ax.text(0.03, 0.97, "input QA rejects: " + ", ".join(counts), transform=ax.transAxes, ha="left",
                va="top", color=MUTED, fontsize=plt.rcParams["font.size"] * 0.75, zorder=6,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.0})
    if family == "ip":
        ax.axhline(0.0, color=MUTED, lw=0.8, zorder=1)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.set_ylabel(f"vacuum (coil + eddy) [{FAMILY_DISPLAY[family][1]}]")
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()


def pipeline_comparison_onset_scatter(points: pd.DataFrame, *, sources: Mapping[str, str] = ONSET_SOURCES,
                                      t_ms: float | None = None, show: bool = False,
                                      figsize: tuple[float, float] = (15.0, 5.6), limits_by_panel=None,
                                      panel_labels: bool = True):
    """Measured (x) against vacuum (y) before the plasma onset: plasma current, probes, flux loops.

    The layout of :func:`pipeline_comparison_overview` for the vacuum model:
    ``points`` is the onset-points table (``source``, ``family`` of ``ip``,
    ``probe`` or ``loop``, ``measured``, ``vacuum``; SI with ``ip`` in kA),
    every channel at a few fixed instants before the onset, pooled. The
    vacuum (coil + eddy) model gives the plasma current no value but zero, so
    in (a) the points lie on ``y = 0`` and their spread along ``x`` is the
    measured offset. ``limits_by_panel`` maps ``"ip"``, ``"probe"`` or
    ``"loop"`` to ``(min, max)`` in display units; ``t_ms`` keeps one instant.

    With a ``qa_code`` column (the per-sample validity the diagnostics
    magnetics QA, :func:`vaft.validation.magnetics.validate_magnetics_signals`,
    gives the channel at that instant) a point the input QA rejects
    (``qa_code < 0``: invalid or suspect) is drawn as a cross in its
    pipeline's colour and counted on the panel; the plasma current has no
    such QA. The QA judges the waveform alone, never its agreement with the
    model, so a cross far from ``y = x`` is a sensor the input QA already
    caught, and a dot far from it is one it did not.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    limits = dict(limits_by_panel or {})
    unknown = set(limits) - {"ip", "probe", "loop"}
    if unknown:
        raise ValueError(f"unknown panel in limits_by_panel: {sorted(unknown)}")
    data, rejected = _onset_points(points, sources, t_ms)
    presentation = resolve_presentation("slide", None)
    with presentation.context():
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        for ax, family in zip(axes, ("ip", "probe", "loop")):
            _onset_panel(ax, data, rejected, family, limits.get(family))
        if panel_labels:
            for letter, ax in zip("abc", axes):
                _panel_letter(ax, letter)
        instants = sorted(pd.unique(pd.to_numeric(points["t_ms"], errors="coerce").dropna()))
        fig.suptitle(r"Before the plasma onset, $t - t_\mathrm{onset}$ = "
                     + ", ".join(f"{t:g}" for t in instants) + " ms", y=0.98)
        handles = [Line2D([], [], color=SIDE_STYLE[side]["color"], marker=SIDE_STYLE[side]["marker"],
                          linestyle="none", markersize=10, label=COMPARISON_LABELS[side]) for side in SIDES]
        if rejected.any():
            handles.append(Line2D([], [], color=INK, marker="x", linestyle="none", markersize=9,
                                  label="rejected by the input QA"))
        handles.append(Line2D([], [], color=INK, linestyle="--", label="y = x"))
        fig.tight_layout(rect=(0.0, 0.12, 1.0, 0.95), pad=0.8, w_pad=1.2)
        # tight_layout keeps the axis labels inside the rect only to within the
        # equal-aspect boxes; the key needs its own band below them.
        fig.subplots_adjust(bottom=0.24)
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=len(handles),
                   frameon=False)
    if show:
        plt.show()
    return fig, axes


#: The four quality checks of :func:`pipeline_comparison_quality_checks`, one per panel.
QUALITY_CHECK_TITLES = (
    "Input check\nProbes vs vacuum model",
    "Fit check\nProbes vs EFIT",
    "Force-balance check\n" r"Virial $\beta_p$",
    "Consistency check\nThomson vs EFIT pressure",
)


def pipeline_comparison_quality_checks(onset_points: pd.DataFrame, points: pd.DataFrame, virial: pd.DataFrame,
                                       channels: pd.DataFrame, *, t_ms: float = -1.0, show_rejected: bool = False,
                                       sources: Mapping[str, str] = ONSET_SOURCES, limits_by_panel=None,
                                       titles: Sequence[str] = QUALITY_CHECK_TITLES, show: bool = False,
                                       figsize: tuple[float, float] = (10.4, 11.0), panel_labels: bool = True):
    """Four quality checks, legacy against current, as one 2×2: input, fit, force balance, consistency.

    (a) every probe the input QA accepts, its measurement against the vacuum
    (coil + eddy) model at ``t_ms`` before the plasma onset; the rejects are
    left out unless ``show_rejected`` (:func:`pipeline_comparison_onset_scatter`); (b) every fitted probe
    against the EFIT reconstruction (:func:`pipeline_comparison_measured_vs_reconstructed`);
    (c) volume-integral ``beta_p`` against the pair-13 virial closure with its
    QA band (:func:`pipeline_comparison_virial`); (d) EFIT pressure at the
    matched Thomson channels against ``2 p_e``, with its QA band and error
    bars (:func:`pipeline_comparison_thomson`); ``y = x`` is the only reference
    line, so (d) drops the ``p_EFIT = p_e`` line its band's lower edge already
    marks. The panels' "not shown" counts are dropped (they belong in the
    caption); one key under the panels. ``limits_by_panel`` maps ``"onset"``,
    ``"probe"``, ``"beta_p"`` or ``"thomson"`` to ``(min, max)``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    limits = dict(limits_by_panel or {})
    unknown = set(limits) - {"onset", "probe", "beta_p", "thomson"}
    if unknown:
        raise ValueError(f"unknown panel in limits_by_panel: {sorted(unknown)}")
    if len(titles) != 4:
        raise ValueError("titles must name the four panels")
    data, rejected = _onset_points(onset_points, sources, t_ms)
    presentation = resolve_presentation("slide", None)
    with presentation.context():
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        (a, b), (c, d) = axes
        _onset_panel(a, data, rejected, "probe", limits.get("onset"), show_rejected=show_rejected)
        pipeline_comparison_measured_vs_reconstructed(points, family="probe", ax=b, limits=limits.get("probe"))
        b.set_ylabel(f"EFIT reconstruction [{FAMILY_DISPLAY['probe'][1]}]")
        pipeline_comparison_virial(virial, quantity="beta_p", ax=c, limits=limits.get("beta_p"))
        pipeline_comparison_thomson(channels, ax=d, limits=limits.get("thomson"))
        base = plt.rcParams["font.size"]
        for ax, title in zip(axes.ravel(), titles):
            ax.set_title(title, fontsize=base * 0.95)
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
            # One visual hierarchy for a print figure: the in-panel counts recede,
            # the QA bands read as regions, and y = x is one line everywhere.
            # The "not shown" counts belong in the caption, not on a print figure.
            for text in list(ax.texts):
                if text.get_text().startswith("not shown"):
                    text.remove()
                elif text.get_text().startswith("input QA rejects"):
                    text.set_fontsize(base * 0.6)
                    text.set_color("#a19f97")
            for band in ax.collections:
                if type(band).__name__ in ("PolyCollection", "FillBetweenPolyCollection"):
                    band.set_alpha(0.22)
        for line in d.get_lines():
            if line.get_linestyle() == ":":
                line.remove()
            elif line.get_linestyle() == "--":
                line.set_linewidth(1.0)
                line.set_color(INK)
        if panel_labels:
            # Small, outside the axes' top-left corner, clear of the centred titles.
            for letter, ax in zip("abcd", axes.ravel()):
                _panel_letter(ax, letter, x=-0.24, y=1.08, scale=1.0)
        handles = [Line2D([], [], color=SIDE_STYLE[side]["color"], marker=SIDE_STYLE[side]["marker"],
                          linestyle="none", markersize=10, label=COMPARISON_LABELS[side]) for side in SIDES]
        if show_rejected and rejected.any():
            handles.append(Line2D([], [], color=INK, marker="x", linestyle="none", markersize=9,
                                  label="rejected by the input QA (a)"))
        handles += [Line2D([], [], color=INK, linestyle="--", label=r"$y = x$"),
                    Patch(facecolor="gray", alpha=0.22, edgecolor="none", label="QA band (c, d)")]
        fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0), pad=0.8, w_pad=1.6, h_pad=1.8)
        # The key is its own block under the figure, two rows, clear of (c)'s axis label.
        fig.subplots_adjust(bottom=0.17)
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=(len(handles) + 1) // 2,
                   frameon=False, fontsize=base * 0.85, columnspacing=2.0)
    if show:
        plt.show()
    return fig, axes
