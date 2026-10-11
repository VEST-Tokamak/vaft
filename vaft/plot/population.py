"""Population renderers for canonical multi-machine tables (#1205).

These draw the canonical confinement and transition tables of
:mod:`vaft.data.public` -- DB5.2.3, TCV L-H, VEST summaries, any source
normalised into them -- and know nothing about which database a row came from.  They follow the renderer contract of :mod:`vaft.plot`
(``ax=None``, ``show=False``, return ``(Figure, Axes)``) but take the canonical
table rather than a view model: a population is a table, not a trace.

Colour carries machine identity in a fixed order: the ``max_groups`` machines
with most rows take the categorical slots, the rest fold into a grey
``"Other"``.  The highlighted machine (VEST by default) is drawn on top with a
distinct marker, so identity never rests on colour alone.

Groups are named on the data, not in a legend (#1946): each label sits above
its own cluster in the cluster's colour, placed by
:func:`vaft.plot._direct_labels.add_direct_labels` so labels never overlap and
a displaced one gets a leader line.  ``labels="legend"`` restores the legend.
Axis and tick text are sized from the active ``rcParams`` (so a
:mod:`vaft.plot.presentation` format scales them) and set larger than
Matplotlib's defaults.

:func:`machine_lcfs_comparison` draws the other multi-machine view: each
machine's last closed flux surface in one R-Z axes, labelled the same way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "confinement_coverage_strip",
    "machine_lcfs_comparison",
    "confinement_h_factor_distribution",
    "confinement_population",
    "confinement_predicted_vs_measured",
    "lh_threshold_population",
    "transition_margin",
    "transition_predicted_vs_measured",
]

#: Categorical slots in fixed order (validated default palette, light mode).
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7")
OTHER_COLOR = "#b8b7ae"
OTHER_TEXT_COLOR = "#7c7b73"
HIGHLIGHT_COLOR = "#1a1a19"
MARKERS = ("o", "s", "^", "D", "v", "P", "X")

LABELS = {
    "i_p_A": ("$I_p$", "MA", 1e-6),
    "b_t_T": ("$B_t$", "T", 1.0),
    "n_e_line_avg_m3": (r"$\bar n_e$", r"$10^{19}$ m$^{-3}$", 1e-19),
    "p_loss_W": ("$P_{loss}$", "MW", 1e-6),
    "w_th_J": ("$W_{th}$", "MJ", 1e-6),
    "tau_e_th_s": (r"$\tau_{E,th}$", "s", 1.0),
    "r_geo_m": ("$R$", "m", 1.0),
    "a_m": ("$a$", "m", 1.0),
    "epsilon": (r"$\epsilon$", "", 1.0),
    "kappa": (r"$\kappa$", "", 1.0),
    "kappa_area": (r"$\kappa_a$", "", 1.0),
    "delta": (r"$\delta$", "", 1.0),
    "m_eff_amu": ("$M_{eff}$", "amu", 1.0),
    "p_lh_scaling_W": ("$P_{LH}$ scaling", "MW", 1e-6),
    "surface_area_m2": ("$S$", "m$^2$", 1.0),
    "q95": ("$q_{95}$", "", 1.0),
}


#: Text sizes relative to the active rcParams entry (#1946): larger than the
#: Matplotlib defaults, and scaled by a presentation format because the base is
#: read when the figure is drawn into, inside the format's rc context.
AXIS_LABEL_SCALE = 1.2
TICK_LABEL_SCALE = 1.15
ROW_LABEL_SCALE = 1.2
LABEL_MODES = ("direct", "legend", "none")


def _font(key: str, scale: float) -> float:
    """``rcParams[key]`` in points (a relative size such as "medium" resolved), times ``scale``."""
    import matplotlib
    from matplotlib.font_manager import FontProperties

    return scale * FontProperties(size=matplotlib.rcParams[key]).get_size_in_points()


def _legible_axes(ax, *, x: bool = True, y: bool = True) -> None:
    """Enlarge the axis labels and tick labels already set on ``ax``."""
    if x:
        ax.xaxis.label.set_fontsize(_font("axes.labelsize", AXIS_LABEL_SCALE))
        ax.tick_params(axis="x", which="both", labelsize=_font("xtick.labelsize", TICK_LABEL_SCALE))
    if y:
        ax.yaxis.label.set_fontsize(_font("axes.labelsize", AXIS_LABEL_SCALE))
        ax.tick_params(axis="y", which="both", labelsize=_font("ytick.labelsize", TICK_LABEL_SCALE))


def _text_color(color):
    """The folded "Other" grey is too light for text: its label uses a darker grey."""
    return OTHER_TEXT_COLOR if color == OTHER_COLOR else color


def _check_labels(labels: str) -> None:
    if labels not in LABEL_MODES:
        raise ValueError(f"labels must be one of {LABEL_MODES}; got {labels!r}")


def _name_groups(ax, named, labels: str, legend_loc: str, *, avoid=()) -> None:
    """Name the drawn groups: on their clusters, in a legend, or not at all."""
    if labels == "direct":
        from ._direct_labels import DirectLabel, add_direct_labels

        add_direct_labels(ax, [DirectLabel(text, _text_color(color), x, y, "above")
                               for text, color, x, y in named if np.size(x)], avoid=avoid)
    elif labels == "legend":
        ax.legend(fontsize="x-small", markerscale=1.5, frameon=False, loc=legend_loc)


def _label(column: str) -> str:
    name, unit, _ = LABELS.get(column, (column, "", 1.0))
    return f"{name} [{unit}]" if unit else name


def _scaled(table: pd.DataFrame, column: str) -> np.ndarray:
    scale = LABELS.get(column, (None, None, 1.0))[2]
    return pd.to_numeric(table[column], errors="coerce").to_numpy(float) * scale


def _group_labels(table: pd.DataFrame, by: str) -> pd.Series:
    column = table[by].astype(object)
    return column.where(column.notna(), "unknown").astype(str)


def _groups(table: pd.DataFrame, by: str, highlight: str | None, max_groups: int):
    """Ordered (label, mask, color, marker) for the background population."""
    labels = _group_labels(table, by)
    background = labels != highlight if highlight is not None else pd.Series(True, index=table.index)
    order = labels[background].value_counts().index.tolist()
    head = order[:max_groups]
    groups = [
        (name, (labels == name).to_numpy(), CATEGORICAL[i % len(CATEGORICAL)], MARKERS[i % len(MARKERS)])
        for i, name in enumerate(head)
    ]
    rest = background.to_numpy() & ~labels.isin(head).to_numpy()
    if rest.any():
        groups.insert(0, (f"Other: {len(order) - len(head)} groups", rest, OTHER_COLOR, "."))
    return groups


def _aligned(series: pd.Series, table: pd.DataFrame, name: str) -> np.ndarray:
    """Values of a per-row series, refusing one computed on another table.

    Aligning by index would silently pair rows of different tables (a
    concatenated table re-numbers its rows), so the index must be identical.
    """
    if not series.index.equals(table.index):
        raise ValueError(
            f"{name} is not indexed like the table; compute it on this table "
            "(rows are paired by the table's own rows, not by position)"
        )
    return pd.to_numeric(series, errors="coerce").to_numpy(float)


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


def _draw_highlight(ax, x, y, label):
    """Stars on top; returns ``(text, colour, x, y)`` for the group's name."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.any():
        text = f"{label} ({int(ok.sum())})"
        ax.scatter(x[ok], y[ok], s=140, marker="*", color=HIGHLIGHT_COLOR,
                   edgecolor="white", linewidth=1.0, zorder=5, label=text)
        return [(text, HIGHLIGHT_COLOR, x[ok], y[ok])]
    return []


def confinement_population(
    table: pd.DataFrame,
    x: str = "i_p_A",
    y: str = "tau_e_th_s",
    *,
    by: str = "machine",
    highlight: str | None = "VEST",
    max_groups: int = 7,
    log: bool = True,
    labels: str = "direct",
    ax=None,
    show: bool = False,
    figsize=(6.4, 5.0),
):
    """Scatter one canonical column against another across the whole population.

    Parameters
    ----------
    table : pandas.DataFrame
        Canonical confinement table [table].
    x, y : str, optional
        Canonical column names; default ``"i_p_A"`` and ``"tau_e_th_s"`` [str].
    by : str, optional
        Column whose values get colours, default ``"machine"`` [str].
    highlight : str or None, optional
        Value of ``by`` drawn on top as stars, default ``"VEST"`` [str].
    max_groups : int, optional
        Groups with their own colour; the rest fold into "Other"; default 7 [-].
    log : bool, optional
        Log-log axes, default ``True`` [bool].
    labels : str, optional
        ``"direct"`` (default) names each group above its own cluster in its
        colour; ``"legend"`` draws a legend instead; ``"none"`` neither [str].
    ax : matplotlib.axes.Axes or None, optional
        Target axes, default ``None`` creates a figure [Axes].
    show : bool, optional
        Call ``plt.show()``, default ``False`` [bool].
    figsize : tuple, optional
        Size of a created figure, default ``(6.4, 5.0)`` [in].

    Returns
    -------
    tuple
        ``(Figure, Axes)`` [matplotlib].
    """
    _check_labels(labels)
    fig, ax = _axes(ax, figsize)
    xs, ys = _scaled(table, x), _scaled(table, y)
    named = []
    for name, mask, color, marker in _groups(table, by, highlight, max_groups):
        ok = mask & np.isfinite(xs) & np.isfinite(ys)
        text = f"{name} ({int(ok.sum())})"
        ax.scatter(xs[ok], ys[ok], s=10, color=color, marker=marker, alpha=0.55,
                   linewidth=0, label=text)
        named.append((text, color, xs[ok], ys[ok]))
    if highlight is not None:
        mask = (_group_labels(table, by) == highlight).to_numpy()
        named += _draw_highlight(ax, xs[mask], ys[mask], highlight)
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel(_label(x))
    ax.set_ylabel(_label(y))
    _legible_axes(ax)
    ax.grid(alpha=0.2, which="both")
    _name_groups(ax, named, labels, "best")
    _finish(fig, show)
    return fig, ax


def confinement_predicted_vs_measured(
    table: pd.DataFrame,
    predicted: pd.Series,
    *,
    scaling_label: str = "IPB98(y,2)",
    by: str = "machine",
    highlight: str | None = "VEST",
    max_groups: int = 7,
    band: float = 2.0,
    labels: str = "direct",
    ax=None,
    show: bool = False,
    figsize=(5.6, 5.4),
):
    """Measured against scaling-predicted thermal confinement time, log-log.

    Parameters
    ----------
    table : pandas.DataFrame
        Canonical confinement table [table].
    predicted : pandas.Series
        Prediction computed on ``table`` itself (identical index), e.g. by
        :func:`vaft.data.public.predict_confinement_time` [s].
    scaling_label : str, optional
        Name shown on the axis, default ``"IPB98(y,2)"`` [str].
    by, highlight, max_groups : optional
        Grouping as in :func:`confinement_population`.
    band : float, optional
        Factor of the dashed band around unity, default 2 [-].
    labels : str, optional
        Group names, as in :func:`confinement_population`; default
        ``"direct"`` [str].
    ax, show, figsize : optional
        Renderer contract, as in :func:`confinement_population`.

    Returns
    -------
    tuple
        ``(Figure, Axes)`` [matplotlib].
    """
    _check_labels(labels)
    fig, ax = _axes(ax, figsize)
    measured = pd.to_numeric(table["tau_e_th_s"], errors="coerce").to_numpy(float)
    pred = _aligned(predicted, table, "predicted")
    named = []
    for name, mask, color, marker in _groups(table, by, highlight, max_groups):
        ok = mask & np.isfinite(measured) & np.isfinite(pred)
        text = f"{name} ({int(ok.sum())})"
        ax.scatter(pred[ok], measured[ok], s=10, color=color, marker=marker, alpha=0.55,
                   linewidth=0, label=text)
        named.append((text, color, pred[ok], measured[ok]))
    if highlight is not None:
        mask = (_group_labels(table, by) == highlight).to_numpy()
        named += _draw_highlight(ax, pred[mask], measured[mask], highlight)
    finite = np.concatenate([v[np.isfinite(v) & (v > 0)] for v in (measured, pred)])
    references = []
    if finite.size:
        lo, hi = finite.min() / 1.5, finite.max() * 1.5
        line = np.array([lo, hi])
        ax.plot(line, line, color="#52514e", lw=1.0)
        ax.plot(line, line * band, color="#52514e", lw=0.8, ls="--")
        ax.plot(line, line / band, color="#52514e", lw=0.8, ls="--")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        references = [(line, line * factor) for factor in (1.0, band, 1.0 / band)]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(rf"$\tau_E$ {scaling_label} [s]")
    ax.set_ylabel(r"$\tau_{E,th}$ measured [s]")
    _legible_axes(ax)
    ax.grid(alpha=0.2, which="both")
    _name_groups(ax, named, labels, "upper left", avoid=references)
    _finish(fig, show)
    return fig, ax


def confinement_h_factor_distribution(
    table: pd.DataFrame,
    h: pd.Series,
    *,
    scaling_label: str = "IPB98(y,2)",
    by: str = "machine",
    highlight: str | None = "VEST",
    ax=None,
    show: bool = False,
    figsize=(6.4, 5.0),
):
    """H-factor distribution per group, as horizontal box plots.

    Parameters
    ----------
    table : pandas.DataFrame
        Canonical confinement table [table].
    h : pandas.Series
        H-factor computed on ``table`` itself (identical index), e.g. by
        :func:`vaft.data.public.h_factor` [-].
    scaling_label : str, optional
        Scaling named on the axis, default ``"IPB98(y,2)"`` [str].
    by : str, optional
        Grouping column, default ``"machine"`` [str].
    highlight : str or None, optional
        Group drawn in the highlight colour, default ``"VEST"`` [str].
    ax, show, figsize : optional
        Renderer contract, as in :func:`confinement_population`.

    Returns
    -------
    tuple
        ``(Figure, Axes)`` [matplotlib].
    """
    fig, ax = _axes(ax, figsize)
    values = _aligned(h, table, "h")
    frame = pd.DataFrame({"group": _group_labels(table, by).to_numpy(), "h": values}).dropna()
    order = frame.groupby("group")["h"].median().sort_values().index.tolist()
    data = [frame.loc[frame.group == g, "h"].to_numpy() for g in order]
    if data:
        boxes = ax.boxplot(data, vert=False, widths=0.6, patch_artist=True,
                           flierprops={"markersize": 2, "alpha": 0.4},
                           medianprops={"color": HIGHLIGHT_COLOR})
        for patch, name in zip(boxes["boxes"], order):
            patch.set_facecolor(HIGHLIGHT_COLOR if name == highlight else "#cde2fb")
            patch.set_edgecolor("#52514e")
        ax.set_yticks(range(1, len(order) + 1))
        ax.set_yticklabels([f"{g} ({len(d)})" for g, d in zip(order, data)], fontsize="small")
    ax.axvline(1.0, color="#52514e", lw=1.0, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel(f"$H$ = measured / {scaling_label}")
    ax.grid(alpha=0.2, axis="x", which="both")
    _finish(fig, show)
    return fig, ax


def confinement_coverage_strip(
    table: pd.DataFrame,
    columns: tuple[str, ...] = ("i_p_A", "b_t_T", "n_e_line_avg_m3", "r_geo_m", "epsilon", "kappa_area"),
    *,
    by: str = "machine",
    highlight: str | None = "VEST",
    max_groups: int = 7,
    axes=None,
    show: bool = False,
    figsize=None,
):
    """Where each group sits along each engineering parameter, one panel per column.

    Rows missing a quantity are simply absent from that panel, so a partial
    record (e.g. a VEST slice without loss power) still shows where it sits in
    the parameters it has.

    Parameters
    ----------
    table : pandas.DataFrame
        Canonical confinement table [table].
    columns : tuple of str, optional
        Canonical columns, one panel each [str].
    by, highlight, max_groups : optional
        Grouping as in :func:`confinement_population`.
    axes : sequence of Axes or None, optional
        One axes per column, default ``None`` creates a figure [Axes].
    show : bool, optional
        Call ``plt.show()``, default ``False`` [bool].
    figsize : tuple or None, optional
        Size of a created figure, default scales with the column count [in].

    Returns
    -------
    tuple
        ``(Figure, ndarray[Axes])`` [matplotlib].
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, MaxNLocator

    if axes is None:
        fig, axes = plt.subplots(1, len(columns), figsize=figsize or (2.5 * len(columns), 5.0), sharey=True)
    else:
        fig = np.ravel(axes)[0].figure
    axes = np.atleast_1d(axes)
    groups = _groups(table, by, highlight, max_groups)
    rows = [g[0] for g in groups] + ([highlight] if highlight is not None else [])
    rng = np.random.default_rng(0)
    for ax, column in zip(axes, columns):
        values = _scaled(table, column)
        for position, (name, mask, color, marker) in enumerate(groups):
            ok = mask & np.isfinite(values)
            jitter = position + rng.uniform(-0.25, 0.25, int(ok.sum()))
            ax.scatter(values[ok], jitter, s=6, color=color, marker=marker, alpha=0.5, linewidth=0)
        if highlight is not None:
            mask = (_group_labels(table, by) == highlight).to_numpy() & np.isfinite(values)
            ax.scatter(values[mask], np.full(int(mask.sum()), len(groups)), s=120, marker="*",
                       color=HIGHLIGHT_COLOR, edgecolor="white", linewidth=1.0, zorder=5)
        positive = values[np.isfinite(values) & (values > 0)]
        # Few ticks: the strips are narrow and their tick text is enlarged (#1946).
        if positive.size and positive.max() / positive.min() > 20:
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
        else:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.set_xlabel(_label(column), fontsize=_font("axes.labelsize", AXIS_LABEL_SCALE))
        ax.tick_params(axis="x", which="both", labelsize=_font("xtick.labelsize", TICK_LABEL_SCALE))
        ax.grid(alpha=0.2, axis="x", which="both")
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels(rows, fontsize=_font("ytick.labelsize", ROW_LABEL_SCALE))
    fig.tight_layout()
    _finish(fig, show)
    return fig, axes


def _observed_legend(ax, location="best"):
    """Groups carry colour; filled vs hollow carries whether the event occurred."""
    from matplotlib.lines import Line2D

    handles, labels = ax.get_legend_handles_labels()
    handles += [
        Line2D([], [], marker="o", ls="", color="#52514e", markerfacecolor="#52514e", label="transition observed"),
        Line2D([], [], marker="o", ls="", color="#52514e", markerfacecolor="none", label="not observed"),
    ]
    labels += ["transition observed", "not observed"]
    ax.legend(handles, labels, fontsize="x-small", frameon=False, loc=location)


def _transition_scatter(ax, table, x, y, by, max_groups):
    observed = table["transition_observed"].astype(bool).to_numpy()
    for name, mask, color, marker in _groups(table, by, None, max_groups):
        ok = mask & np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[ok & observed], y[ok & observed], s=36, marker=marker, color=color,
                   edgecolor=color, linewidth=1.0, label=f"{name} ({int(ok.sum())})")
        ax.scatter(x[ok & ~observed], y[ok & ~observed], s=36, marker=marker,
                   facecolor="none", edgecolor=color, linewidth=1.2)


def lh_threshold_population(
    table: pd.DataFrame,
    x: str = "n_e_line_avg_m3",
    y: str = "p_loss_W",
    *,
    by: str = "main_ion",
    max_groups: int = 7,
    log: bool = False,
    ax=None,
    show: bool = False,
    figsize=(6.4, 5.0),
):
    """L-H transition records in any two canonical columns, e.g. P_loss against density.

    Parameters
    ----------
    table : pandas.DataFrame
        Canonical transition table [table].
    x, y : str, optional
        Canonical columns; default ``"n_e_line_avg_m3"`` and ``"p_loss_W"`` [str].
    by : str, optional
        Column whose values get colours, default ``"main_ion"`` [str].
    max_groups : int, optional
        Groups with their own colour, the rest fold into "Other"; default 7 [-].
    log : bool, optional
        Log-log axes, default ``False`` [bool].
    ax, show, figsize : optional
        Renderer contract, as in :func:`confinement_population`.

    Returns
    -------
    tuple
        ``(Figure, Axes)`` [matplotlib].

    Notes
    -----
    Filled markers are records where the transition occurred; hollow ones are
    records where the source sought it and did not see it.
    """
    fig, ax = _axes(ax, figsize)
    _transition_scatter(ax, table, _scaled(table, x), _scaled(table, y), by, max_groups)
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel(_label(x))
    ax.set_ylabel(_label(y))
    ax.grid(alpha=0.2, which="both")
    _observed_legend(ax)
    _finish(fig, show)
    return fig, ax


def transition_predicted_vs_measured(
    table: pd.DataFrame,
    *,
    by: str = "main_ion",
    max_groups: int = 7,
    band: float = 2.0,
    ax=None,
    show: bool = False,
    figsize=(5.6, 5.4),
):
    """Measured loss power at the transition against the record's scaling threshold.

    Parameters
    ----------
    table : pandas.DataFrame
        Canonical transition table; plots ``p_loss_W`` against
        ``p_lh_scaling_W`` [table].
    by, max_groups : optional
        Grouping as in :func:`lh_threshold_population`.
    band : float, optional
        Factor of the dashed band around unity, default 2 [-].
    ax, show, figsize : optional
        Renderer contract, as in :func:`confinement_population`.

    Returns
    -------
    tuple
        ``(Figure, Axes)`` [matplotlib].
    """
    fig, ax = _axes(ax, figsize)
    predicted = _scaled(table, "p_lh_scaling_W")
    measured = _scaled(table, "p_loss_W")
    _transition_scatter(ax, table, predicted, measured, by, max_groups)
    finite = np.concatenate([v[np.isfinite(v) & (v > 0)] for v in (measured, predicted)])
    if finite.size:
        lo, hi = finite.min() / 1.5, finite.max() * 1.5
        line = np.array([lo, hi])
        ax.plot(line, line, color="#52514e", lw=1.0)
        ax.plot(line, line * band, color="#52514e", lw=0.8, ls="--")
        ax.plot(line, line / band, color="#52514e", lw=0.8, ls="--")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$P_{LH}$ scaling [MW]")
    ax.set_ylabel(r"$P_{loss}$ at the record [MW]")
    ax.grid(alpha=0.2, which="both")
    _observed_legend(ax, "upper left")
    _finish(fig, show)
    return fig, ax


def transition_margin(
    table: pd.DataFrame,
    margin: pd.Series,
    x: str = "n_e_line_avg_m3",
    *,
    by: str = "main_ion",
    max_groups: int = 7,
    ax=None,
    show: bool = False,
    figsize=(6.4, 4.6),
):
    """Transition margin ``P_loss / P_LH`` against a canonical column.

    Parameters
    ----------
    table : pandas.DataFrame
        Canonical transition table [table].
    margin : pandas.Series
        Margin computed on ``table`` itself (identical index), e.g. by
        :func:`vaft.data.public.transition_margin` [-].
    x : str, optional
        Canonical column on the horizontal axis, default ``"n_e_line_avg_m3"``
        [str].
    by, max_groups : optional
        Grouping as in :func:`lh_threshold_population`.
    ax, show, figsize : optional
        Renderer contract, as in :func:`confinement_population`.

    Returns
    -------
    tuple
        ``(Figure, Axes)`` [matplotlib].
    """
    fig, ax = _axes(ax, figsize)
    _transition_scatter(ax, table, _scaled(table, x), _aligned(margin, table, "margin"), by, max_groups)
    from matplotlib.ticker import FuncFormatter

    ax.axhline(1.0, color="#52514e", lw=1.0, ls="--")
    ax.set_yscale("log")
    plain = FuncFormatter(lambda value, _: f"{value:g}")
    ax.yaxis.set_major_formatter(plain)
    ax.yaxis.set_minor_formatter(plain)
    ax.set_xlabel(_label(x))
    ax.set_ylabel(r"$P_{loss}\,/\,P_{LH}$ scaling")
    ax.grid(alpha=0.2, which="both")
    _observed_legend(ax)
    _finish(fig, show)
    return fig, ax


def machine_lcfs_comparison(
    equilibria,
    *,
    normalized_by=None,
    limiter: bool = True,
    magnetic_axis: bool = True,
    labels: str = "direct",
    ax=None,
    show: bool = False,
    figsize=(6.4, 6.0),
):
    """Last closed flux surfaces of several machines in one R-Z axes, each named on its curve.

    Parameters
    ----------
    equilibria : mapping
        Machine name -> equilibrium with ``lcfs`` (``.r``, ``.z`` in m) and,
        optionally, ``limiter`` (``.r``, ``.z``) and ``magnetic_axis``
        (``(R, Z)``), e.g. :class:`vaft.process.equilibrium.EquilibriumData`;
        an entry without an LCFS is skipped [m].
    normalized_by : mapping or None, optional
        Machine name -> ``(R0, a)``; when given, draws ``(R - R0) / a``
        against ``Z / a`` and skips machines it does not name, default
        ``None`` draws physical R-Z [m].
    limiter : bool, optional
        Thin limiter outline in the machine's colour (physical view only),
        default ``True`` [bool].
    magnetic_axis : bool, optional
        ``+`` at the magnetic axis (physical view only), default ``True`` [bool].
    labels : str, optional
        ``"direct"`` (default) writes each machine's name beside its LCFS in
        its colour, with leader lines where machines nest; ``"legend"`` draws
        a legend; ``"none"`` neither [str].
    ax, show, figsize : optional
        Renderer contract, as in :func:`confinement_population`.

    Returns
    -------
    tuple
        ``(Figure, Axes)`` [matplotlib].

    Notes
    -----
    Colours follow the categorical slots in the mapping's order.  The axes are
    equal-aspect, so a spherical tokamak and ITER keep their true shapes.
    """
    _check_labels(labels)
    fig, ax = _axes(ax, figsize)
    named, outlines = [], []
    for index, (name, eq) in enumerate(equilibria.items()):
        lcfs = getattr(eq, "lcfs", None)
        if lcfs is None:
            continue
        color = CATEGORICAL[index % len(CATEGORICAL)]
        r, z = np.asarray(lcfs.r, float), np.asarray(lcfs.z, float)
        if normalized_by is not None:
            if name not in normalized_by:
                continue
            r0, a = (float(v) for v in normalized_by[name])
            r, z = (r - r0) / a, z / a
        else:
            wall = getattr(eq, "limiter", None)
            if limiter and wall is not None:
                wr, wz = np.asarray(wall.r, float), np.asarray(wall.z, float)
                ax.plot(wr, wz, color=color, lw=0.6, alpha=0.4)
                outlines.append((wr, wz))
            axis = getattr(eq, "magnetic_axis", None)
            if magnetic_axis and axis is not None:
                ax.plot(*axis, marker="+", color=color, ms=8)
        ax.plot(r, z, color=color, label=name)
        named.append((name, color, r, z))
    if normalized_by is not None:
        ax.set_xlabel(r"$(R - R_0)\,/\,a$")
        ax.set_ylabel(r"$Z\,/\,a$")
    else:
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
    _legible_axes(ax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    if labels == "direct":
        from ._direct_labels import DirectLabel, add_direct_labels

        add_direct_labels(ax, [DirectLabel(name, color, r, z, "outside") for name, color, r, z in named],
                          avoid=outlines)
    elif labels == "legend":
        ax.legend(fontsize="small")
    _finish(fig, show)
    return fig, ax
