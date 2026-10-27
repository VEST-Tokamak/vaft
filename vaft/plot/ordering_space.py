"""Ordering spaces and ordering margins of a population of plasma states.

Two plots read the asymptotic orderings of :mod:`vaft.validation.orderings`
for a table of states -- one row per state, slice or flux-surface point, with
columns named as in :data:`~vaft.validation.orderings.ORDERING_QUANTITIES`
(the rows :mod:`vaft.process.ordering_state` builds):

* :func:`ordering_space` -- two ordering quantities on log axes, with the
  order-unity line of each, the region where both orderings hold, and the one-
  and two-decade margin lines. Each quadrant is named by the regime it means
  (``ideal / single-fluid``, ``resistive / Hall``, ...). It is an *ordering*
  space: the lines are where an expansion parameter reaches one, not an
  empirical stability or operating boundary.
* :func:`ordering_margins_plot` -- the distribution of each ordering's margin
  $m = \\mp\\log_{10}(x/x_0)$ in decades, on the bands of
  :data:`~vaft.validation.orderings.MARGIN_LABELS`. The continuous margin is the
  result; the band names read it and are not an applicability status. Like
  :func:`vaft.process.ordering_state.ordering_margins`, it counts a global or
  time-history ordering once per state and a profile ordering once per point.

Each ordering is drawn against its own threshold ``x_0``
(:func:`~vaft.validation.orderings.ordering_threshold`), order unity for every
registered contract today.

Nothing is computed or filled in here. A row whose value is missing, not
finite or not positive has no ordering statement and is counted as
unassessed, never drawn at a guessed position.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = ["ordering_space", "ordering_margins_plot", "SPACE_STYLE"]

#: Colours of the permitted region, the order-unity lines and the margin lines.
SPACE_STYLE = {
    "permitted": "#dbe9df",
    "unity": "#9a3324",
    "decade": "#7a7a7a",
    "points": "#2b2b2b",
    "bands": {"outside": "#f2d7d3", "0-1 decade": "#f6ead0", "1-2 decades": "#e3ecdc", "over 2 decades": "#cfe3d4"},
}


def _space(space):
    from vaft.validation.orderings import ORDERING_SPACES, OrderingSpace

    if isinstance(space, OrderingSpace):
        return space
    try:
        return ORDERING_SPACES[space]
    except KeyError:
        raise KeyError(f"no ordering space {space!r}; known: {', '.join(ORDERING_SPACES)}") from None


def _values(table: pd.DataFrame, column: str) -> np.ndarray:
    if column not in table.columns:
        raise KeyError(f"table has no column {column!r}: the axis is the ordering quantity of that name")
    return pd.to_numeric(table[column], errors="coerce").to_numpy(float)


def _assessable(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values) & (values > 0)


def _axes(ax):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 4.4))
    return ax.figure, ax


def _limits(values: np.ndarray, ordering: str, depth: float = 2.0, threshold: float = 1.0) -> Tuple[float, float]:
    """Log-axis limits: ``depth`` decades on the permitted side of the threshold, one on the other."""
    lo, hi = (10.0 ** -depth, 10.0) if ordering == "small" else (0.1, 10.0 ** depth)
    lo, hi = lo * threshold, hi * threshold
    finite = values[_assessable(values)]
    if finite.size:
        lo, hi = min(lo, finite.min()), max(hi, finite.max())
    return 10 ** (math.floor(math.log10(lo)) - 0.3), 10 ** (math.ceil(math.log10(hi)) + 0.3)


def _permitted(lim: Tuple[float, float], ordering: str, threshold: float = 1.0) -> Tuple[float, float]:
    return (lim[0], threshold) if ordering == "small" else (threshold, lim[1])


def ordering_space(table: pd.DataFrame, space="resistive_two_fluid", *, group: Optional[str] = None,
                   decades: Sequence[float] = (1.0, 2.0), quadrants: bool = True,
                   title: Optional[str] = None, ax=None, show: bool = False, **style):
    """Scatter a population on one ordering space.

    Parameters
    ----------
    table : pandas.DataFrame
        One row per state or point; the two axis columns are the space's
        ordering quantities, by name.
    space : str or OrderingSpace
        A key of :data:`vaft.validation.orderings.ORDERING_SPACES`, or an
        :class:`~vaft.validation.orderings.OrderingSpace` pairing any two
        ordering quantities.
    group : str, optional
        Categorical column for colour.
    decades : sequence of float
        Margins, in decades, at which to draw a dashed line on the permitted
        side of each axis.
    quadrants : bool
        Name the four regions by the regime each side means.
    title : str, optional
        Axes title; default the space's title.
    ax : matplotlib Axes, optional
        Authoritative axes; none is created when given.
    show : bool
        Call ``plt.show()``.
    **style
        Passed to ``Axes.scatter``.

    Returns
    -------
    (Figure, Axes)
        The axes carry the exclusion count in ``ax.ordering_exclusions``:
        ``{"total": n, "placed": k, "unassessed": n - k}``.
    """
    from vaft.validation.orderings import SYMBOLS, ordering_threshold, required_ordering

    s = _space(space)
    x, y = _values(table, s.x), _values(table, s.y)
    ox, oy = required_ordering(s.x), required_ordering(s.y)
    tx, ty = ordering_threshold(s.x), ordering_threshold(s.y)
    ok = _assessable(x) & _assessable(y)
    fig, ax = _axes(ax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    depth = max([2.0, *decades])
    xlim, ylim = _limits(x, ox, depth, tx), _limits(y, oy, depth, ty)
    px, py = _permitted(xlim, ox, tx), _permitted(ylim, oy, ty)
    ax.fill_between(px, py[0], py[1], color=SPACE_STYLE["permitted"], zorder=0, lw=0)
    ax.axvline(tx, color=SPACE_STYLE["unity"], lw=1.2, zorder=1)
    ax.axhline(ty, color=SPACE_STYLE["unity"], lw=1.2, zorder=1)
    for d in decades:
        fx = tx * (10 ** (-d) if ox == "small" else 10 ** d)
        fy = ty * (10 ** (-d) if oy == "small" else 10 ** d)
        ax.axvline(fx, color=SPACE_STYLE["decade"], lw=0.7, ls="--", zorder=1)
        ax.axhline(fy, color=SPACE_STYLE["decade"], lw=0.7, ls="--", zorder=1)
        ax.annotate(f"$m = {d:g}$", (fx, ylim[1]), xytext=(2, -2), textcoords="offset points", fontsize=7,
                    color=SPACE_STYLE["decade"], ha="left", va="top")
        ax.annotate(f"$m = {d:g}$", (xlim[1], fy), xytext=(-2, 2), textcoords="offset points", fontsize=7,
                    color=SPACE_STYLE["decade"], ha="right", va="bottom")
    kwargs = dict(s=10, color=SPACE_STYLE["points"], alpha=0.75, lw=0, zorder=3)
    kwargs.update(style)
    if group is not None:
        if group not in table.columns:
            raise KeyError(f"table has no group column {group!r}")
        kwargs.pop("color", None)
        labels = table[group].astype(object).where(table[group].notna(), "unknown").astype(str).to_numpy()
        for name in pd.unique(labels[ok]):
            sel = ok & (labels == name)
            ax.scatter(x[sel], y[sel], label=str(name), **kwargs)
        if ok.any():
            ax.legend(fontsize=7, frameon=False, loc="best")
    else:
        ax.scatter(x[ok], y[ok], **kwargs)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    side = {"small": "\\ll", "large": "\\gg"}
    ax.set_xlabel(f"{SYMBOLS[s.x]}   (needs ${side[ox]} {tx:g}$)")
    ax.set_ylabel(f"{SYMBOLS[s.y]}   (needs ${side[oy]} {ty:g}$)")
    if quadrants:
        for hx in (True, False):
            for hy in (True, False):
                # the centre of the quadrant in log space
                qx = px if hx else ((tx, xlim[1]) if ox == "small" else (xlim[0], tx))
                qy = py if hy else ((ty, ylim[1]) if oy == "small" else (ylim[0], ty))
                cx, cy = math.sqrt(qx[0] * qx[1]), math.sqrt(qy[0] * qy[1])
                text = f"{s.y_reading[0 if hy else 1]}\n{s.x_reading[0 if hx else 1]}"
                ax.text(cx, cy, text, ha="center", va="center", fontsize=7.5, zorder=2,
                        color="#3d5a45" if (hx and hy) else "#6b6b6b",
                        fontweight="bold" if (hx and hy) else "normal")
    n, k = len(table), int(ok.sum())
    ax.set_title(title if title is not None else s.title, fontsize=10, loc="left")
    ax.text(1.0, 1.01, f"{k} of {n} placed" + (f", {n - k} unassessed" if n - k else ""),
            transform=ax.transAxes, fontsize=7, color="#6b6b6b", ha="right", va="bottom")
    ax.ordering_exclusions = {"total": n, "placed": k, "unassessed": n - k}
    if show:
        import matplotlib.pyplot as plt

        plt.show()
    return fig, ax


def ordering_margins_plot(states: pd.DataFrame, points: Optional[pd.DataFrame] = None,
                          quantities: Optional[Iterable[str]] = None, *,
                          title: Optional[str] = None, ax=None, show: bool = False):
    """The margin of each ordering across a population, in decades, on the margin bands.

    Parameters
    ----------
    states : pandas.DataFrame
        One row per state: the global and time-history ordering quantities.
    points : pandas.DataFrame, optional
        One row per flux-surface point: the profile and other local ordering
        quantities. Without it every quantity is read from ``states``.
    quantities : iterable of str, optional
        The orderings to show, top to bottom; default every ordering quantity
        the tables have a column for, in registry order.
    title : str, optional
        Axes title.
    ax : matplotlib Axes, optional
        Authoritative axes; none is created when given.
    show : bool
        Call ``plt.show()``.

    Returns
    -------
    (Figure, Axes)
        One row per ordering: the margins, their median, and on the right the
        fraction with ``m > 0`` and the number assessed. A global or
        time-history ordering is counted once per state, never once per
        profile point, so ``n`` and the fraction match
        :func:`vaft.process.ordering_state.ordering_margins` on the same data.
    """
    from vaft.validation.applicability import ordering_margin
    from vaft.validation.orderings import (MARGIN_LABELS, ORDERING_QUANTITIES, SYMBOLS, ordering_threshold,
                                           required_ordering)

    def source(q):
        state_level = ORDERING_QUANTITIES[q].group in ("global", "time_history")
        return states if (points is None or state_level) else points

    if quantities is None:
        names = [q for q in ORDERING_QUANTITIES if q in source(q).columns]
    else:
        names = list(quantities)
    if not names:
        raise ValueError("the tables have no ordering-quantity column")
    margins = {}
    for q in names:
        side, threshold = required_ordering(q), ordering_threshold(q)
        margins[q] = np.array([ordering_margin(v, side, threshold) for v in _values(source(q), q)], float)
    finite = np.concatenate([m[np.isfinite(m)] for m in margins.values()] + [np.array([-1.0, 3.0])])
    lo, hi = math.floor(finite.min()) - 0.5, math.ceil(finite.max()) + 0.5
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6.8, 0.42 * len(names) + 1.2))
        # room on the right for the per-ordering "% hold, n" column
        ax.figure.subplots_adjust(left=0.2, right=0.8)
    fig = ax.figure
    bounds = [b for b, _ in MARGIN_LABELS]
    for k, ((bound, label), upper) in enumerate(zip(MARGIN_LABELS, [hi] + bounds[:-1])):
        ax.axvspan(max(bound, lo), upper, color=SPACE_STYLE["bands"][label], zorder=0, lw=0)
        # one-decade bands are narrower than their names, so neighbours alternate between two heights
        ax.text(0.5 * (max(bound, lo) + upper), 1.005 + 0.03 * (k % 2), label, ha="center", va="bottom",
                fontsize=6.5, color="#555555", transform=ax.get_xaxis_transform())
    ax.axvline(0.0, color=SPACE_STYLE["unity"], lw=1.2, zorder=1)
    rng = np.random.default_rng(0)
    for i, q in enumerate(names):
        m = margins[q]
        ok = np.isfinite(m)
        jitter = rng.uniform(-0.18, 0.18, ok.sum())
        ax.scatter(m[ok], i + jitter, s=6, color=SPACE_STYLE["points"], alpha=0.45, lw=0, zorder=2)
        if ok.any():
            ax.plot([np.median(m[ok])] * 2, [i - 0.3, i + 0.3], color="#1a1a19", lw=2.0, zorder=3)
            frac = float((m[ok] > 0).mean())
            ax.text(hi + 0.1, i, f"{100 * frac:.0f}% hold, n={int(ok.sum())}", va="center", fontsize=7,
                    clip_on=False)
        else:
            ax.text(hi + 0.1, i, "unassessed", va="center", fontsize=7, color="#6b6b6b", clip_on=False)
    ax.set_yticks(range(len(names)))
    need = {"small": "\\ll", "large": "\\gg"}
    ax.set_yticklabels([f"{SYMBOLS[q]}  ${need[required_ordering(q)]} {ordering_threshold(q):g}$" for q in names])
    ax.set_ylim(len(names) - 0.5, -0.5)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("ordering margin $m = \\mp\\log_{10}(x/x_0)$  [decades]")
    if title:
        ax.set_title(title, fontsize=10, pad=22)
    if show:
        plt.show()
    return fig, ax
