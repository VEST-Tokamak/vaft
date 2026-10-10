"""Population histograms of the Tier A stability atlas (#1852).

These draw the populations :func:`vaft.process.mhd_stability.stability_atlas_populations`
reads from the atlas that ``workflow/stability_atlas/build_atlas.py`` writes. They
follow the renderer contract of :mod:`vaft.plot` (``ax=None``, ``show=False``,
return ``(Figure, Axes)``) but take the reader's result rather than a view model, as
:mod:`vaft.plot.population` and :mod:`vaft.plot.transport_atlas` do: an atlas is
a table, not a trace. No ODS, database or solver layer is imported.

One figure, three stacked panels, each marking its stability boundary and the
two sides of it:

1. ideal MHD -- DCON's least-stable ``W_t`` for each ``n``; ``W_t < 0`` is
   unstable (the sign only: there is no ``|W_t|`` band);
2. resistive -- each slice's ``Delta'_max`` over its rational surfaces, RDCON and
   STRIDE at ``n = 1, 2``; ``Delta'_max > 0``, a classical tearing drive at any
   surface, is unstable;
3. local criteria -- Mercier ``max D_I`` (unstable ``> 0``) and high-n
   ballooning ``min C_A`` (unstable ``< 0``), whose unstable sides are opposite.

Counts (slices, QA, unstable) are not printed on the figure; they are the
reader's ``summary`` table.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

__all__ = ["stability_atlas_population"]

#: Unstable and stable zone colours; the series take :data:`vaft.plot.population.CATEGORICAL`.
UNSTABLE_COLOR = "#c0392b"
STABLE_COLOR = "#2a78d6"
ZERO_LINE_COLOR = "#1a1a19"

#: (lowest negative decade, highest positive decade, linear threshold, bins per decade) per panel.
_BINNING = {
    "ideal": (4.0, 1.5, 1e-3, 3),
    "resistive": (2.0, 15.0, 1.0, 2),
    "local": (1.0, 1.0, 1e-2, 6),
}
_TICKS = {
    "ideal": (-1e3, -1.0, -1e-3, 0.0, 1e-3, 1.0, 1e2),
    "resistive": (-1e2, 0.0, 1e4, 1e8, 1e12),
    "local": (-1.0, -1e-2, 0.0, 1e-2, 1.0),
}
_RESISTIVE_SERIES = (("rdcon", 1, 1), ("rdcon", 2, 3), ("stride", 1, 2), ("stride", 2, 0))


def _symlog_bins(lowest: float, highest: float, linthresh: float, per_decade: int) -> np.ndarray:
    start = np.log10(linthresh)
    positive = np.logspace(start, highest, int(round((highest - start) * per_decade)) + 1)
    negative = -np.logspace(start, lowest, int(round((lowest - start) * per_decade)) + 1)[::-1]
    return np.concatenate([negative, [0.0], positive])


def _prepare(ax, bins: np.ndarray, linthresh: float, ticks: Sequence[float]) -> None:
    ax.set_xscale("symlog", linthresh=linthresh)
    # Pinned to the bin range so the zone shading fills the axis edge to edge.
    ax.set_xlim(bins[0], bins[-1])
    ax.set_xticks(list(ticks))


def _headroom(ax, highest_count: float) -> None:
    # One decade above the tallest bar keeps the zone and key text clear of the data.
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(0, max(highest_count, 1.0) * 10)
    ax.set_ylabel("slices")


def _zones(ax, *, unstable_right: bool) -> None:
    lo, hi = ax.get_xlim()
    right, left = (UNSTABLE_COLOR, STABLE_COLOR) if unstable_right else (STABLE_COLOR, UNSTABLE_COLOR)
    ax.axvspan(0.0, hi, color=right, alpha=0.06, zorder=0)
    ax.axvspan(lo, 0.0, color=left, alpha=0.06, zorder=0)
    ax.axvline(0.0, color=ZERO_LINE_COLOR, lw=1.2, zorder=3)
    for x, ha, unstable in ((0.98, "right", unstable_right), (0.02, "left", not unstable_right)):
        ax.text(x, 0.96, "unstable" if unstable else "stable", transform=ax.transAxes, ha=ha, va="top",
                fontsize="medium", fontweight="bold", color=UNSTABLE_COLOR if unstable else STABLE_COLOR)


def _step(ax, values: np.ndarray, bins: np.ndarray, color: Any) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return 0.0
    # Delta' is unbounded at low-shear first surfaces: a value past the last bin is
    # drawn in the edge bin rather than dropped, so every counted slice is drawn.
    values = np.clip(values, bins[0], bins[-1])
    counts, _, _ = ax.hist(values, bins=bins, histtype="step", lw=1.5, color=color)
    return float(np.max(counts))


def _ideal(ax, populations) -> None:
    import matplotlib.pyplot as plt

    lowest, highest, linthresh, per_decade = _BINNING["ideal"]
    bins = _symlog_bins(lowest, highest, linthresh, per_decade)
    _prepare(ax, bins, linthresh, _TICKS["ideal"])
    modes = sorted(populations.ideal_w_t)
    cmap = plt.get_cmap("viridis")
    colors = {n: cmap(i / max(len(modes) - 1, 1) * 0.92) for i, n in enumerate(modes)}
    tallest = max([_step(ax, populations.ideal_w_t[n], bins, colors[n]) for n in modes] or [0.0])
    _headroom(ax, tallest)
    _zones(ax, unstable_right=False)
    ax.text(0.02, 0.80, "n =", transform=ax.transAxes, ha="left", va="top", fontsize="small")
    for i, n in enumerate(modes):
        ax.text(0.10 + 0.045 * i, 0.80, str(n), transform=ax.transAxes, ha="left", va="top",
                fontsize="small", fontweight="bold", color=colors[n])
    ax.set_xlabel(r"$W_t$")


def _resistive(ax, populations) -> None:
    from vaft.plot.population import CATEGORICAL

    lowest, highest, linthresh, per_decade = _BINNING["resistive"]
    bins = _symlog_bins(lowest, highest, linthresh, per_decade)
    _prepare(ax, bins, linthresh, _TICKS["resistive"])
    tallest = 0.0
    keys = []
    for solver, n, slot in _RESISTIVE_SERIES:
        values = populations.delta_prime_max.get((solver, n))
        if values is None:
            continue
        tallest = max(tallest, _step(ax, values, bins, CATEGORICAL[slot]))
        keys.append((f"{solver.upper()} n={n}", CATEGORICAL[slot]))
    _headroom(ax, tallest)
    _zones(ax, unstable_right=True)
    for i, (label, color) in enumerate(keys):
        ax.text(0.98, 0.84 - 0.085 * i, label, transform=ax.transAxes, ha="right", va="top",
                fontsize="small", color=color)
    ax.set_xlabel(r"$\Delta'_{\max}$")


def _local(ax, populations) -> None:
    from vaft.plot.population import CATEGORICAL

    lowest, highest, linthresh, per_decade = _BINNING["local"]
    bins = _symlog_bins(lowest, highest, linthresh, per_decade)
    _prepare(ax, bins, linthresh, _TICKS["local"])
    mercier, ballooning = CATEGORICAL[6], CATEGORICAL[2]
    tallest = max(_step(ax, populations.mercier_max_d_i, bins, mercier),
                  _step(ax, populations.ballooning_min_c_a, bins, ballooning))
    _headroom(ax, tallest)
    ax.axvline(0.0, color=ZERO_LINE_COLOR, lw=1.2, zorder=3)
    # The two criteria are unstable on opposite sides of zero, so each names its own.
    ax.text(0.02, 0.96, "$\\leftarrow$ ballooning unstable\n" + r"(min $C_A<0$)", transform=ax.transAxes,
            ha="left", va="top", fontsize=9.5, fontweight="bold", color=ballooning)
    ax.text(0.98, 0.96, "Mercier unstable $\\rightarrow$\n" + r"(max $D_I>0$)", transform=ax.transAxes,
            ha="right", va="top", fontsize=9.5, fontweight="bold", color=mercier)
    ax.set_xlabel("local stability index")


def stability_atlas_population(populations, *, ax=None, show: bool = False, figsize=(5.2, 7.0)):
    """The stability atlas population as three stacked histograms.

    Parameters
    ----------
    populations : vaft.process.mhd_stability.StabilityAtlasPopulations
        What :func:`vaft.process.mhd_stability.stability_atlas_populations` returns [-].
    ax : sequence of matplotlib Axes, optional
        Exactly one per panel (three), top to bottom; a new figure when omitted [-].
    show : bool, optional
        Call ``plt.show()`` after drawing [-].
    figsize : tuple of float, optional
        Size of a new figure [in].

    Returns
    -------
    tuple
        ``(Figure, ndarray)``, the three Axes top to bottom [matplotlib].
    """
    import matplotlib.pyplot as plt

    style = {"font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5}
    with plt.rc_context(style):
        if ax is None:
            fig, axes = plt.subplots(3, 1, figsize=figsize, layout="constrained")
        else:
            axes = np.asarray(ax, dtype=object).ravel()
            if axes.size != 3:
                raise ValueError(f"stability_atlas_population draws three panels, got {axes.size} axes")
            fig = axes[0].figure
        _ideal(axes[0], populations)
        _resistive(axes[1], populations)
        _local(axes[2], populations)
    if show:
        plt.show()
    return fig, axes
