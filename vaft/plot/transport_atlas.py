"""Renderers for the (shot, t, rho) transport atlas table (#1427).

These draw the table that ``workflow/transport_atlas/build_atlas.py`` writes: one
row per (shot, time_efit_s, efit_lineage, r_over_a) surface, with TGLF, NEO and
classical model predictions. They follow the renderer contract of :mod:`vaft.plot`
(``ax=None``, ``show=False``, return ``(Figure, Axes)``) but take the table rather
than a view model, as :mod:`vaft.plot.population` does: an atlas is a table, not a
trace. No ODS, database or solver layer is imported.

Every axis, legend and colour-bar label comes from :data:`LABELS`. The atlas schema
publishes the same symbols, so a consumer drawing the table elsewhere labels a
quantity the same way.

Encoding: marker shape is the EFIT lineage (``o`` magnetics, ``s`` electron_kinetic);
filled markers are ``good`` slices and open markers are ``admissible``. Everything
drawn is a model prediction under the atlas's declared assumptions, not an
experimental operating boundary.

``transport_atlas_mode_branch`` colours by the sign of the real frequency of the
fastest-growing ion-scale mode (``k_y rho_s <= 1``), on surfaces where that mode
grows (``gamma > 0``); a stable surface has no direction and is not drawn. In TGLF a negative frequency
is the ion diamagnetic direction (``tglf/src/tglf_max.f90``). The colours are
directions, not ITG/TEM labels.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

__all__ = [
    "LABELS",
    "axis_label",
    "transport_atlas_mode_branch",
    "transport_atlas_scatter",
]

#: column -> (mathtext symbol, display unit). An empty unit is dimensionless or
#: already normalised inside the symbol.
LABELS: dict[str, tuple[str, str]] = {
    "r_over_a": (r"$r/a$", ""),
    "rho_tor_norm": (r"$\rho_{\mathrm{tor}}$", ""),
    "a_m": (r"$a$", "m"),
    "b_unit_T": (r"$B_{\mathrm{unit}}$", "T"),
    "q": (r"$q$", ""),
    "shear": (r"$\hat{s}$", ""),
    "kappa": (r"$\kappa$", ""),
    "delta": (r"$\delta$", ""),
    "betae": (r"$\beta_e$", ""),
    "xnue": (r"$\nu_{ei}\,a/c_s$", ""),
    "zeff": (r"$Z_{\mathrm{eff}}$", ""),
    "a_over_lne": (r"$a/L_{n_e}$", ""),
    "a_over_lte": (r"$a/L_{T_e}$", ""),
    "a_over_lti": (r"$a/L_{T_i}$", ""),
    "a_over_lni": (r"$a/L_{n_i}$", ""),
    "ti_over_te": (r"$T_i/T_e$", ""),
    "q_gb_W_m2": (r"$Q_{\mathrm{GB}}$", r"W m$^{-2}$"),
    "qe_gb": (r"$Q_e/Q_{\mathrm{GB}}$", ""),
    "qi_gb": (r"$Q_i/Q_{\mathrm{GB}}$", ""),
    "gamma_e_gb": (r"$\Gamma_e/\Gamma_{\mathrm{GB}}$", ""),
    "q_tot_gb": (r"$(Q_e+Q_i)/Q_{\mathrm{GB}}$", ""),
    "qe_tglf_W_m2": (r"$Q_e^{\mathrm{TGLF}}$", r"W m$^{-2}$"),
    "qi_tglf_W_m2": (r"$Q_i^{\mathrm{TGLF}}$", r"W m$^{-2}$"),
    "gamma_e_tglf_m2_s": (r"$\Gamma_e^{\mathrm{TGLF}}$", r"m$^{-2}$ s$^{-1}$"),
    "f_e": (r"$f_e = |Q_e|/(|Q_e|+|Q_i|)$", ""),
    "gamma_max": (r"$\gamma_{\max}$", r"$c_s/a$"),
    "ky_at_gamma_max": (r"$k_y\rho_s$ at $\gamma_{\max}$", ""),
    "omega_at_gamma_max": (r"$\omega_r$ at $\gamma_{\max}$", r"$c_s/a$"),
    "gamma_max_ion_scale": (r"$\gamma_{\max}\,(k_y\rho_s\leq 1)$", r"$c_s/a$"),
    "ky_at_gamma_max_ion_scale": (r"$k_y\rho_s$ at $\gamma_{\max}\,(k_y\rho_s\leq 1)$", ""),
    "omega_at_gamma_max_ion_scale": (r"$\omega_r$ at $\gamma_{\max}\,(k_y\rho_s\leq 1)$", r"$c_s/a$"),
    "n_unstable_ky": (r"$N_{k_y}(\gamma>0)$", ""),
    "ky_q_mean": (r"$\langle k_y\rho_s\rangle_Q$", ""),
    "f_em": (r"$f_{\mathrm{EM}}$", ""),
    "qe_neo_W_m2": (r"$Q_e^{\mathrm{NEO}}$", r"W m$^{-2}$"),
    "qi_neo_W_m2": (r"$Q_i^{\mathrm{NEO}}$", r"W m$^{-2}$"),
    "gamma_e_neo_m2_s": (r"$\Gamma_e^{\mathrm{NEO}}$", r"m$^{-2}$ s$^{-1}$"),
    "qe_classical_W_m2": (r"$Q_e^{\mathrm{cl}}$", r"W m$^{-2}$"),
    "qi_classical_W_m2": (r"$Q_i^{\mathrm{cl}}$", r"W m$^{-2}$"),
    "chi_e_classical_m2_s": (r"$\chi_e^{\mathrm{cl}}$", r"m$^2$ s$^{-1}$"),
    "chi_i_classical_m2_s": (r"$\chi_i^{\mathrm{cl}}$", r"m$^2$ s$^{-1}$"),
    "f_neo_qe": (r"$f_{\mathrm{neo}}(Q_e)$", ""),
    "f_neo_qi": (r"$f_{\mathrm{neo}}(Q_i)$", ""),
    "f_neo_gamma_e": (r"$f_{\mathrm{neo}}(\Gamma_e)$", ""),
    "f_classical_qe": (r"$f_{\mathrm{cl}}(Q_e)$", ""),
    "f_classical_qi": (r"$f_{\mathrm{cl}}(Q_i)$", ""),
    "qe_model_W_m2": (r"$Q_e^{\mathrm{model}}$", r"W m$^{-2}$"),
    "qi_model_W_m2": (r"$Q_i^{\mathrm{model}}$", r"W m$^{-2}$"),
}

#: Marker per EFIT lineage (lane K's State key contract v1 spellings, #1454).
MARKERS = {"magnetics": "o", "electron_kinetic": "s"}
QUALITIES = ("good", "admissible")
DIRECTION_COLORS = {"electron": "#2a78d6", "ion": "#d6402a"}


def axis_label(column: str) -> str:
    """``<symbol> [<unit>]`` for an atlas column; the bare name when it has no symbol."""
    symbol, unit = LABELS.get(column, (column, ""))
    return f"{symbol} [{unit}]" if unit else symbol


def _numeric(table: pd.DataFrame, column: str) -> np.ndarray:
    if column not in table:
        raise KeyError(f"the atlas table has no column {column!r}")
    return pd.to_numeric(table[column], errors="coerce").to_numpy(float)


def _figure(ax):
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.2, 4.2), layout="constrained")
        return fig, ax
    return ax.figure, ax


def _finish(fig, show: bool):
    if show:
        import matplotlib.pyplot as plt

        plt.show()


def transport_atlas_scatter(
    table: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str] = None,
    *,
    where: Optional[Callable[[pd.DataFrame], Any]] = None,
    symlog: bool = False,
    cmap: str = "viridis",
    ax=None,
    show: bool = False,
):
    """Scatter two atlas columns, coloured by a third.

    Parameters
    ----------
    table : pandas.DataFrame
        The atlas (``atlas.csv``), one row per surface.
    x, y, color : str
        Column names. Rows missing any of them are not drawn.
    where : callable, optional
        ``table -> boolean mask`` selecting the rows to draw. The colour scale spans
        only the rows drawn.
    symlog : bool
        Symmetric-log axes, for signed fluxes spanning decades.

    Returns
    -------
    (Figure, Axes)
    """
    import matplotlib as mpl

    fig, ax = _figure(ax)
    mask = np.ones(len(table), bool) if where is None else np.asarray(where(table), bool)
    xs, ys = _numeric(table, x), _numeric(table, y)
    mask &= np.isfinite(xs) & np.isfinite(ys)
    cs = _numeric(table, color) if color else None
    if cs is not None:
        mask &= np.isfinite(cs)
    lineage = table.get("efit_lineage", pd.Series("", index=table.index)).astype(str).to_numpy()
    quality = table.get("efit_quality", pd.Series("", index=table.index)).astype(str).to_numpy()
    # Only rows with a known lineage and quality are drawn, so only they set the scale.
    mask &= np.isin(lineage, list(MARKERS)) & np.isin(quality, QUALITIES)
    norm = None
    if cs is not None and mask.any():
        norm = mpl.colors.Normalize(vmin=float(cs[mask].min()), vmax=float(cs[mask].max()))
    drawn = 0
    for position, (name, marker) in enumerate(MARKERS.items()):
        for label in QUALITIES:
            sel = mask & (lineage == name) & (quality == label)
            if not sel.any():
                continue
            kwargs = dict(marker=marker, s=26, linewidths=0.8, label=f"{name}, {label} ({sel.sum()})")
            if cs is not None:
                colors = mpl.colormaps[cmap](norm(cs[sel]))
                if label == "good":
                    ax.scatter(xs[sel], ys[sel], c=colors, **kwargs)
                else:
                    ax.scatter(xs[sel], ys[sel], facecolors="none", edgecolors=colors, **kwargs)
            else:
                # An open marker needs an explicit edge colour or matplotlib draws nothing.
                edge = f"C{position}"
                ax.scatter(xs[sel], ys[sel], facecolors=edge if label == "good" else "none",
                           edgecolors=edge, **kwargs)
            drawn += int(sel.sum())
    ax.set_xlabel(axis_label(x))
    ax.set_ylabel(axis_label(y))
    if symlog:
        ax.set_xscale("symlog", linthresh=1e-2)
        ax.set_yscale("symlog", linthresh=1e-2)
    if norm is not None:
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=axis_label(color))
    if drawn:
        ax.legend(fontsize=7, loc="best")
    ax.vaft_drawn = drawn
    _finish(fig, show)
    return fig, ax


def transport_atlas_mode_branch(
    table: pd.DataFrame,
    *,
    x: str = "a_over_lne",
    y: str = "a_over_lte",
    ax=None,
    show: bool = False,
):
    """Drive plane coloured by the direction of the fastest-growing ion-scale mode.

    Marker area grows with ``log10(1 + |Q_e + Q_i| / Q_GB)``; filled markers are good
    slices and open ones admissible, as in :func:`transport_atlas_scatter`. The counts drawn in each
    direction are left on ``ax.vaft_counts`` as ``{"electron": n, "ion": n}``.

    Returns
    -------
    (Figure, Axes)
    """
    fig, ax = _figure(ax)
    omega = _numeric(table, "omega_at_gamma_max_ion_scale")
    gamma = _numeric(table, "gamma_max_ion_scale")
    xs, ys = _numeric(table, x), _numeric(table, y)
    q = np.abs(_numeric(table, "q_tot_gb")) if "q_tot_gb" in table else np.zeros(len(table))
    size = 6 + 10 * np.log10(1 + np.nan_to_num(q))
    lineage = table.get("efit_lineage", pd.Series("", index=table.index)).astype(str).to_numpy()
    quality = table.get("efit_quality", pd.Series("", index=table.index)).astype(str).to_numpy()
    # A direction is a property of a growing mode: a stable surface has none.
    base = (np.isfinite(omega) & (omega != 0) & (gamma > 0)
            & np.isfinite(xs) & np.isfinite(ys))
    counts = {}
    for direction, sign, text in (("electron", 1, r"$\omega_r > 0$"), ("ion", -1, r"$\omega_r < 0$")):
        sel_dir = base & (np.sign(omega) == sign)
        counts[direction] = int(sel_dir.sum())
        colour = DIRECTION_COLORS[direction]
        for name, marker in MARKERS.items():
            for label in QUALITIES:
                sel = sel_dir & (lineage == name) & (quality == label)
                if sel.any():
                    ax.scatter(xs[sel], ys[sel], s=size[sel], marker=marker,
                               facecolors=colour if label == "good" else "none",
                               edgecolors=colour, linewidths=0.8,
                               label=f"{direction} ({text}), {name}, {label} ({sel.sum()})")
    ax.set_xlabel(axis_label(x))
    ax.set_ylabel(axis_label(y))
    ax.text(0.99, 0.01, r"$\omega_r$ of the fastest-growing mode with $k_y\rho_s\leq 1$;"
            "\n" r"marker area $\propto \log_{10}(1+|Q_e+Q_i|/Q_{\mathrm{GB}})$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color="0.35")
    if any(counts.values()):
        ax.legend(fontsize=6.5, loc="upper left")
    ax.vaft_counts = counts
    _finish(fig, show)
    return fig, ax
