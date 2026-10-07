"""Plots of matched Thomson and EFIT kinetic-state Atlas tables."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .presentation import presented

__all__ = ["kinetic_state_pressure_comparison", "kinetic_state_virial_pair13_comparison",
           "kinetic_state_virial_li_comparison"]

_LINEAGE_COLORS = {"magnetics": "#2a78d6", "electron_kinetic": "#eb6834"}


@presented(default_figsize=(6.3, 5.5))
def kinetic_state_pressure_comparison(
    points: pd.DataFrame,
    *,
    ax=None,
    show: bool = False,
    figsize: tuple[float, float] | None = None,
    format: str | None = None,
    theme: str | None = None,
):
    """Compare EFIT pressure with twice Thomson electron pressure at matched channels.

    The pointwise ``p_e <= p_EFIT <= 2 p_e`` reference becomes
    ``x/2 <= y <= x`` because ``x = 2 p_e``.  The study verdict instead uses
    the ratio of channel sums; this figure does not grade individual points.
    ``points`` must contain ``p_e_pa``, ``p_efit_pa``, and ``efit_lineage``.
    ``format='slide'`` uses the shared VAFT presentation size and typography.
    """
    required = {"p_e_pa", "p_efit_pa", "efit_lineage"}
    missing = required - set(points.columns)
    if missing:
        raise ValueError(f"pressure comparison missing columns: {', '.join(sorted(missing))}")
    data = points.copy()
    data["p_e_pa"] = pd.to_numeric(data["p_e_pa"], errors="coerce")
    data["p_efit_pa"] = pd.to_numeric(data["p_efit_pa"], errors="coerce")
    data = data.loc[np.isfinite(data.p_e_pa) & np.isfinite(data.p_efit_pa)
                    & (data.p_e_pa > 0) & (data.p_efit_pa > 0)]
    if data.empty:
        raise ValueError("pressure comparison needs positive finite matched-channel pressures")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (6.3, 5.5))
    else:
        fig = ax.figure
    font_scale = plt.rcParams["font.size"] / 10.0
    for lineage, group in data.groupby("efit_lineage", sort=False):
        ax.scatter(2 * group.p_e_pa, group.p_efit_pa,
                   s=19 * font_scale ** 2, alpha=0.62,
                   color=_LINEAGE_COLORS.get(lineage, "#8b8a82"),
                   label=f"{lineage.replace('_', ' ')} (n={len(group)})")

    limit = (min(2 * data.p_e_pa.min(), data.p_efit_pa.min()),
             max(2 * data.p_e_pa.max(), data.p_efit_pa.max()))
    x = np.geomspace(*limit, 100)
    ax.fill_between(x, x / 2, x, color="gray", alpha=0.14, label=r"$p_e \leq p_{EFIT} \leq 2p_e$")
    ax.plot(x, x / 2, color="0.4", ls=":", lw=1.2 * font_scale, label=r"$p_{EFIT}=p_e$")
    ax.plot(x, x, color="black", ls="--", lw=1.2 * font_scale, label=r"$p_{EFIT}=2p_e$")
    ax.set(xscale="log", yscale="log", xlim=limit, ylim=limit,
           xlabel=r"$2p_e$ from Thomson [Pa]", ylabel=r"$p_{EFIT}$ at TS channel [Pa]",
           title="Matched Thomson channels")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    if show:
        plt.show()
    return fig, ax


@presented(default_figsize=(6.3, 5.5))
def kinetic_state_virial_pair13_comparison(
    states: pd.DataFrame,
    *,
    ax=None,
    show: bool = False,
    figsize: tuple[float, float] | None = None,
    format: str | None = None,
    theme: str | None = None,
):
    """Compare volume-integral and pair-13 virial estimates of poloidal beta.

    Both coordinates must come from the same selected EFIT product and time
    slice. This visualizes the pair-13 closure separately from the direct
    E1/E2/E3 identity residuals and does not apply an acceptance threshold.
    """
    return _virial_pair13_comparison(states, quantity="beta_p", symbol=r"\beta_p",
                                     title="Virial pair-13 closure", ax=ax,
                                     show=show, figsize=figsize)


@presented(default_figsize=(6.3, 5.5))
def kinetic_state_virial_li_comparison(
    states: pd.DataFrame,
    *,
    ax=None,
    show: bool = False,
    figsize: tuple[float, float] | None = None,
    format: str | None = None,
    theme: str | None = None,
):
    """Compare volume-integral and pair-13 virial internal inductance."""
    return _virial_pair13_comparison(states, quantity="li", symbol=r"l_i",
                                     title="Virial pair-13 internal inductance", ax=ax,
                                     show=show, figsize=figsize)


def _virial_pair13_comparison(states, *, quantity, symbol, title, ax, show, figsize):
    volume_column = f"{quantity}_volume"
    pair_column = f"{quantity}_pair_13"
    required = {volume_column, pair_column, "efit_lineage"}
    missing = required - set(states.columns)
    if missing:
        raise ValueError(f"virial pair-13 comparison missing columns: {', '.join(sorted(missing))}")
    data = states.copy()
    for column in (volume_column, pair_column):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.loc[np.isfinite(data[volume_column]) & np.isfinite(data[pair_column])]
    if data.empty:
        raise ValueError(f"virial pair-13 comparison needs finite {quantity} pairs")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (6.3, 5.5))
    else:
        fig = ax.figure
    font_scale = plt.rcParams["font.size"] / 10.0
    values = data[[volume_column, pair_column]].to_numpy()
    lower, upper = float(values.min()), float(values.max())
    padding = max((upper - lower) * 0.06, 0.01)
    limit = (lower - padding, upper + padding)
    reference_label = (r"$l_{i,13}=l_{i,vol}$" if quantity == "li"
                       else r"$\beta_{p,13}=\beta_{p,vol}$")
    ax.plot(limit, limit, color="black", ls="--", lw=1.2 * font_scale,
            label=reference_label, zorder=1)
    for lineage, group in data.groupby("efit_lineage", sort=False):
        ax.scatter(group[volume_column], group[pair_column],
                   s=23 * font_scale ** 2, alpha=0.72,
                   color=_LINEAGE_COLORS.get(lineage, "#8b8a82"),
                   label=f"{lineage.replace('_', ' ')} (n={len(group)})", zorder=2)
    ax.set(xlim=limit, ylim=limit, aspect="equal",
           xlabel=rf"Volume-integral ${symbol}$", ylabel=rf"Pair-13 virial ${symbol}$",
           title=title)
    relative = np.abs(data[pair_column] - data[volume_column]) / np.abs(data[volume_column])
    relative = relative.loc[np.isfinite(relative)]
    if not relative.empty:
        ax.text(0.98, 0.04, f"Median relative\ndifference: {relative.median():.2%}",
                transform=ax.transAxes, ha="right", va="bottom")
    ax.legend(frameon=False, loc="upper left")
    if show:
        plt.show()
    return fig, ax
