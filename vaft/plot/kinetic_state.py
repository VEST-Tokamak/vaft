"""Plots of matched Thomson and EFIT kinetic-state Atlas tables."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .presentation import presented

__all__ = ["kinetic_state_pressure_comparison", "kinetic_state_virial_pair13_comparison",
           "kinetic_state_virial_li_comparison", "kinetic_state_pressure_ratio_distribution",
           "kinetic_state_virial_identity_distribution"]

_LINEAGE_COLORS = {"magnetics": "#2a78d6", "electron_kinetic": "#eb6834"}


def _distribution_input(evidence, quantities, limits, positive=False):
    required = {"efit_lineage", "quantity", "status", "value"}
    missing = required - set(evidence.columns)
    if missing:
        raise ValueError(f"distribution missing columns: {', '.join(sorted(missing))}")
    data = evidence.loc[evidence.quantity.isin(quantities)].copy()
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data["usable"] = (data.status == "available") & np.isfinite(data.value)
    if positive:
        data["usable"] &= data.value > 0
    resolved = {}
    for quantity, bounds in (limits or {}).items():
        if quantity not in quantities:
            raise ValueError(f"unknown distribution panel: {quantity}")
        bounds = np.asarray(bounds, dtype=float)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
            raise ValueError("limits must contain two finite increasing values per panel")
        if positive and bounds[0] <= 0:
            raise ValueError("logarithmic pressure limits must be positive")
        resolved[quantity] = tuple(bounds)
    return data, resolved


def _distribution_axes(ax, count, figsize):
    if ax is None:
        return plt.subplots(1, count, figsize=figsize)
    axes = np.asarray(ax, dtype=object).ravel()
    if len(axes) != count or any(axis.figure is not axes[0].figure for axis in axes):
        raise ValueError(f"distribution needs {count} axes in one figure")
    return axes[0].figure, axes


def _distribution_panel(ax, data, bins):
    for lineage, group in data.groupby("efit_lineage", sort=False):
        values = group.loc[group.usable, "value"]
        ax.hist(values, bins=bins, alpha=0.55, color=_LINEAGE_COLORS.get(lineage, "#8b8a82"),
                label=f"{lineage.replace('_', ' ')}\n(n={len(values)}/{len(group)})")
    if not data.usable.any():
        ax.text(0.5, 0.5, "No available evidence", ha="center", va="center", transform=ax.transAxes)
    if not data.empty:
        ax.legend(frameon=False)


@presented(default_figsize=(12.0, 4.0))
def kinetic_state_pressure_ratio_distribution(evidence: pd.DataFrame, *, ax=None,
                                              show=False, figsize=None, format=None,
                                              theme=None, limits=None):
    """Three pressure-support distributions from ``pressure_evidence``.

    Only available finite positive values are drawn. Legend counts are
    usable/total evidence rows, including unavailable and failed states.
    ``limits`` maps r_sum/r_w/r_w_full to view bounds; counts use all data.
    The [1, 2] interval is a reference illustration, not a new verdict.
    """
    quantities = ("r_sum", "r_w", "r_w_full")
    data, limits = _distribution_input(evidence, quantities, limits, positive=True)
    fig, axes = _distribution_axes(ax, 3, figsize or (12.0, 4.0))
    for axis, quantity, title in zip(axes, quantities,
                                     ("TS channel sums", "TS-covered volume", "Full plasma, extrapolated")):
        group = data.loc[data.quantity == quantity]
        values = group.loc[group.usable, "value"]
        lo = min(0.5, values.min() * 0.9) if len(values) else 0.5
        hi = max(5.0, values.max() * 1.1) if len(values) else 5.0
        _distribution_panel(axis, group, np.geomspace(lo, hi, 25))
        axis.axvspan(1, 2, color="gray", alpha=0.12)
        axis.set(xscale="log", xlabel=rf"{quantity}: $p_{{EFIT}}/p_e$", title=title)
        if quantity in limits:
            axis.set_xlim(limits[quantity])
    axes[0].set_ylabel("State count")
    if show:
        plt.show()
    return fig, axes


@presented(default_figsize=(11.0, 4.2))
def kinetic_state_virial_identity_distribution(evidence: pd.DataFrame, *, ax=None,
                                               show=False, figsize=None, format=None,
                                               theme=None, limits=None):
    """Direct E1/E2/E3 and RMS distributions from ``virial_evidence``.

    The first panel pools the three identities (identity counts); the second
    counts states with RMS. Legend n is usable/total observations per panel.
    ``limits`` accepts ``identity`` and ``rms`` bounds. No threshold is drawn.
    """
    identities = ("e1_normalized", "e2_normalized", "e3_normalized")
    data, _ = _distribution_input(evidence, (*identities, "rms"), None)
    # View keys name the two panels, rather than the three pooled identities.
    _, limits = _distribution_input(evidence, ("identity", "rms"), limits)
    fig, axes = _distribution_axes(ax, 2, figsize or (11.0, 4.2))
    for axis, panel in zip(axes, ("identity", "rms")):
        group = data.loc[data.quantity.isin(identities)] if panel == "identity" else data.loc[data.quantity == "rms"]
        values = group.loc[group.usable, "value"]
        if panel == "identity":
            lo, hi = (float(values.min()), float(values.max())) if len(values) else (-0.01, 0.01)
            if lo == hi:
                padding = max(abs(lo) * 0.05, 1e-6)
                lo, hi = lo - padding, hi + padding
            bins = np.linspace(lo, hi, 36)
        else:
            bins = np.linspace(0, max(float(values.max()) * 1.05, 1e-6) if len(values) else 1, 21)
        _distribution_panel(axis, group, bins)
        axis.set(xlabel="Normalized E1, E2, E3 residual" if panel == "identity" else "RMS of normalized identity residuals",
                 ylabel="Identity count" if panel == "identity" else "State count")
        if panel in limits:
            axis.set_xlim(limits[panel])
    if show:
        plt.show()
    return fig, axes


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
