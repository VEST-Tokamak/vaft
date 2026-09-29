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
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "confinement_coverage_strip",
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


def _label(column: str) -> str:
    name, unit, _ = LABELS.get(column, (column, "", 1.0))
    return f"{name} [{unit}]" if unit else name


def _scaled(table: pd.DataFrame, column: str) -> np.ndarray:
    scale = LABELS.get(column, (None, None, 1.0))[2]
    return pd.to_numeric(table[column], errors="coerce").to_numpy(float) * scale


def _groups(table: pd.DataFrame, by: str, highlight: str | None, max_groups: int):
    """Ordered (label, mask, color, marker) for the background population."""
    column = table[by].astype(object)
    labels = column.where(column.notna(), "unknown").astype(str)
    background = labels != highlight if highlight is not None else pd.Series(True, index=table.index)
    order = labels[background].value_counts().index.tolist()
    head = order[:max_groups]
    groups = [
        (name, (labels == name).to_numpy(), CATEGORICAL[i % len(CATEGORICAL)], MARKERS[i % len(MARKERS)])
        for i, name in enumerate(head)
    ]
    rest = background.to_numpy() & ~labels.isin(head).to_numpy()
    if rest.any():
        groups.insert(0, (f"Other ({len(order) - len(head)})", rest, OTHER_COLOR, "."))
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
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.any():
        ax.scatter(x[ok], y[ok], s=140, marker="*", color=HIGHLIGHT_COLOR,
                   edgecolor="white", linewidth=1.0, zorder=5, label=f"{label} ({int(ok.sum())})")


def confinement_population(
    table: pd.DataFrame,
    x: str = "i_p_A",
    y: str = "tau_e_th_s",
    *,
    by: str = "machine",
    highlight: str | None = "VEST",
    max_groups: int = 7,
    log: bool = True,
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
    fig, ax = _axes(ax, figsize)
    xs, ys = _scaled(table, x), _scaled(table, y)
    for name, mask, color, marker in _groups(table, by, highlight, max_groups):
        ok = mask & np.isfinite(xs) & np.isfinite(ys)
        ax.scatter(xs[ok], ys[ok], s=10, color=color, marker=marker, alpha=0.55,
                   linewidth=0, label=f"{name} ({int(ok.sum())})")
    if highlight is not None:
        mask = (table[by].astype(str) == highlight).to_numpy()
        _draw_highlight(ax, xs[mask], ys[mask], highlight)
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel(_label(x))
    ax.set_ylabel(_label(y))
    ax.grid(alpha=0.2, which="both")
    ax.legend(fontsize="x-small", markerscale=1.5, frameon=False, loc="best")
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
    ax, show, figsize : optional
        Renderer contract, as in :func:`confinement_population`.

    Returns
    -------
    tuple
        ``(Figure, Axes)`` [matplotlib].
    """
    fig, ax = _axes(ax, figsize)
    measured = pd.to_numeric(table["tau_e_th_s"], errors="coerce").to_numpy(float)
    pred = _aligned(predicted, table, "predicted")
    for name, mask, color, marker in _groups(table, by, highlight, max_groups):
        ok = mask & np.isfinite(measured) & np.isfinite(pred)
        ax.scatter(pred[ok], measured[ok], s=10, color=color, marker=marker, alpha=0.55,
                   linewidth=0, label=f"{name} ({int(ok.sum())})")
    if highlight is not None:
        mask = (table[by].astype(str) == highlight).to_numpy()
        _draw_highlight(ax, pred[mask], measured[mask], highlight)
    finite = np.concatenate([v[np.isfinite(v) & (v > 0)] for v in (measured, pred)])
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
    ax.set_xlabel(rf"$\tau_E$ {scaling_label} [s]")
    ax.set_ylabel(r"$\tau_{E,th}$ measured [s]")
    ax.grid(alpha=0.2, which="both")
    ax.legend(fontsize="x-small", markerscale=1.5, frameon=False, loc="upper left")
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
    frame = pd.DataFrame({"group": table[by].astype(str).to_numpy(), "h": values}).dropna()
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

    if axes is None:
        fig, axes = plt.subplots(1, len(columns), figsize=figsize or (2.1 * len(columns), 4.6), sharey=True)
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
            mask = (table[by].astype(str) == highlight).to_numpy() & np.isfinite(values)
            ax.scatter(values[mask], np.full(int(mask.sum()), len(groups)), s=120, marker="*",
                       color=HIGHLIGHT_COLOR, edgecolor="white", linewidth=1.0, zorder=5)
        positive = values[np.isfinite(values) & (values > 0)]
        if positive.size and positive.max() / positive.min() > 20:
            ax.set_xscale("log")
        ax.set_xlabel(_label(column), fontsize="small")
        ax.grid(alpha=0.2, axis="x", which="both")
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels(rows, fontsize="small")
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
