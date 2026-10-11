"""Local MHD stability criteria of one equilibrium, drawn (#1902).

These draw what :mod:`vaft.process.local_mhd_criteria` computes: the Mercier
and GGJ interchange indices against normalized flux, each surface on the
s-alpha plane, and the field-line curvature and local shear over the
poloidal cross-section.  They follow the table-input renderer contract of
:mod:`vaft.plot` (``ax=None``, ``show=False``, return ``(Figure, Axes)``),
like :mod:`vaft.plot.stability_atlas`; the R-Z maps go through
:class:`vaft.plot.models.Field2D` and
:func:`vaft.plot.renderers.fields.render_field_2d`.

Every criterion is drawn as the universal one only -- ``D_I > 0``,
``D_R > 0`` unstable; the s-alpha boundary is the circular model's and is
labelled so.  No region is shaded beyond what the model defines.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "local_mhd_criteria_field_curvature",
    "local_mhd_criteria_profile_interchange",
    "local_mhd_criteria_s_alpha",
]

_CURVATURE_LABELS = {
    "kappa_n": r"normal curvature $\kappa_n$ [m$^{-1}$]",
    "kappa_g": r"geodesic curvature $\kappa_g$ [m$^{-1}$]",
    "local_shear": r"local shear $\mathbf{h}\cdot\nabla\times\mathbf{h}$ [m$^{-3}$]",
}
_ZERO = "#1a1a19"


def local_mhd_criteria_profile_interchange(
    profile,
    *,
    comparison: Mapping[str, Any] | None = None,
    rational_surfaces: Sequence[tuple[float, str]] = (),
    ax=None,
    show: bool = False,
    figsize=(5.6, 3.6),
):
    """Mercier ``D_I`` and GGJ ``D_R`` against normalized flux.

    Parameters
    ----------
    profile : vaft.process.local_mhd_criteria.InterchangeProfile
        The indices to draw [-].
    comparison : mapping, optional
        From :func:`vaft.process.local_mhd_criteria.interchange_comparison`;
        its ``di``/``dr`` are drawn as markers labelled with its ``label`` [-].
    rational_surfaces : sequence of (float, str), optional
        ``(psi_n, label)`` of rational surfaces to mark, e.g. from
        :func:`vaft.process.core_q_context` [-].
    ax : matplotlib Axes, optional
        Axes to draw into; a new figure when omitted [-].
    show : bool, optional
        Call ``plt.show()`` [-].
    figsize : tuple of float, optional
        Size of a new figure [in].

    Returns
    -------
    tuple
        ``(Figure, Axes)`` [matplotlib].
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    else:
        fig = ax.figure
    x = np.asarray(profile.psi_n)
    ax.plot(x, profile.D_I, color="C0", lw=1.6, label=r"$D_I$ (Mercier)")
    ax.plot(x, profile.D_R, color="C3", lw=1.6, label=r"$D_R$ (GGJ resistive)")
    if comparison is not None:
        label = comparison.get("label", "code")
        ax.plot(comparison["psi_n"], comparison["di"], "o", ms=3, mfc="none", color="C0", label=f"di, {label}")
        ax.plot(comparison["psi_n"], comparison["dr"], "s", ms=3, mfc="none", color="C3", label=f"dr, {label}")
    ax.axhline(0.0, color=_ZERO, lw=1.0)
    ax.text(0.99, 0.97, r"$D > 0$: unstable", transform=ax.transAxes, ha="right", va="top", fontsize="small")
    for psi_s, name in rational_surfaces:
        ax.axvline(psi_s, color="0.6", ls=":", lw=0.9)
        ax.text(psi_s, 0.02, name, transform=ax.get_xaxis_transform(), rotation=90, ha="right", va="bottom",
                fontsize="x-small", color="0.4")
    # Near the axis the indices run to large negative values while the criterion
    # lives at zero: a symmetric-log axis keeps both readable.
    ax.set_yscale("symlog", linthresh=0.5)
    ax.set_xlabel(r"$\psi_N$")
    ax.set_ylabel("interchange index (symlog)")
    # "best", not a fixed corner: the rational-surface labels sit at the bottom.
    ax.legend(fontsize="small", loc="best")
    if show:
        plt.show()
    return fig, ax


def local_mhd_criteria_s_alpha(profile, *, ax=None, show: bool = False, figsize=(4.6, 3.8)):
    """Each surface on the s-alpha plane, with the circular model's first-stability boundary.

    Parameters
    ----------
    profile : vaft.process.local_mhd_criteria.SAlphaProfile
        Per-surface ``s`` and ``alpha`` [-].
    ax : matplotlib Axes, optional
        Axes to draw into; a new figure when omitted [-].
    show : bool, optional
        Call ``plt.show()`` [-].
    figsize : tuple of float, optional
        Size of a new figure [in].

    Returns
    -------
    tuple
        ``(Figure, Axes)`` [matplotlib].
    """
    import matplotlib.pyplot as plt

    from vaft.formula.stability import s_alpha_marginal_alpha

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    else:
        fig = ax.figure
    s, alpha, psi_n = (np.asarray(v, dtype=float) for v in (profile.s, profile.alpha, profile.psi_n))
    ok = np.isfinite(s) & np.isfinite(alpha)
    s_top = max(float(np.nanmax(s[ok])) * 1.05 if ok.any() else 1.0, 0.5)
    s_low = min(float(np.nanmin(s[ok])) * 1.05 if ok.any() else 0.0, 0.0)  # reversed shear stays visible
    grid = np.linspace(0.0, s_top, 41)  # the model has no first-stability boundary at s <= 0
    first, _ = s_alpha_marginal_alpha(grid)
    ax.plot(grid, first, color=_ZERO, lw=1.2, label="s–α circular, first stability")
    points = ax.scatter(s[ok], alpha[ok], c=psi_n[ok], cmap="viridis", s=18, zorder=3)
    ax.plot(s[ok], alpha[ok], color="0.5", lw=0.8, zorder=2)
    fig.colorbar(points, ax=ax, label=r"$\psi_N$")
    ax.set_xlim(s_low, s_top)
    ax.set_ylim(0.0, max(float(np.nanmax(alpha[ok])) * 1.3 if ok.any() else 1.0, 0.5))
    ax.set_xlabel(r"shear $\hat s_V$")
    ax.set_ylabel(r"$\alpha$")
    ax.legend(fontsize="small", loc="upper left")
    if show:
        plt.show()
    return fig, ax


def local_mhd_criteria_field_curvature(
    maps: Mapping[str, np.ndarray],
    *,
    quantity: str = "all",
    boundary: tuple[np.ndarray, np.ndarray] | None = None,
    ax=None,
    clip_percentile: float = 98.0,
    show: bool = False,
    figsize=None,
):
    """Normal and geodesic curvature and local shear over the poloidal cross-section.

    Parameters
    ----------
    maps : mapping
        From :func:`vaft.process.local_mhd_criteria.curvature_map`: ``R``,
        ``Z`` indexed ``(R, Z)`` on a regular grid, and the quantities, NaN
        outside the confined band [-].
    quantity : str, optional
        ``"kappa_n"``, ``"kappa_g"``, ``"local_shear"``, or ``"all"`` for a
        1 x 3 panel [-].
    boundary : tuple of np.ndarray, optional
        ``(R, Z)`` of an outline to overlay, e.g. the LCFS [m].
    ax : matplotlib Axes or sequence of three, optional
        One Axes, or three for ``"all"``; a new figure when omitted [-].
    clip_percentile : float, optional
        The colour scale spans +/- this percentile of ``|value|``; values
        beyond saturate (``extend="both"``).  The local shear peaks sharply at
        the inboard edge, so a lower percentile shows more of the interior [-].
    show : bool, optional
        Call ``plt.show()`` [-].
    figsize : tuple of float, optional
        Size of a new figure [in].

    Returns
    -------
    tuple
        ``(Figure, Axes)``, an ndarray of three Axes for ``"all"`` [matplotlib].

    Raises
    ------
    ValueError
        Unknown ``quantity`` or the wrong number of axes.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    from vaft.plot.models import Field2D, GeometryLayer
    from vaft.plot.renderers.fields import render_field_2d

    names = tuple(_CURVATURE_LABELS) if quantity == "all" else (quantity,)
    if any(name not in _CURVATURE_LABELS for name in names):
        raise ValueError(f"quantity must be one of {tuple(_CURVATURE_LABELS)} or 'all', got {quantity!r}")
    if ax is None:
        fig, axes = plt.subplots(1, len(names), figsize=figsize or (3.6 * len(names), 4.2), layout="constrained",
                                 squeeze=False)
        axes = axes.ravel()
    else:
        axes = np.atleast_1d(np.asarray(ax, dtype=object)).ravel()
        if axes.size != len(names):
            raise ValueError(f"{len(names)} panel(s) need {len(names)} axes, got {axes.size}")
        fig = axes[0].figure
    r = np.asarray(maps["R"])[:, 0]
    z = np.asarray(maps["Z"])[0, :]
    overlays = ()
    if boundary is not None:
        overlays = (GeometryLayer(r=np.asarray(boundary[0]), z=np.asarray(boundary[1]), kind="polygon",
                                  label="boundary", style={"color": "k", "linewidth": 0.8}, role="equilibrium"),)
    for axis, name in zip(axes, names):
        values = np.asarray(maps[name], dtype=float).T  # Field2D is indexed (Z, R)
        finite = values[np.isfinite(values)]
        top = float(np.percentile(np.abs(finite), clip_percentile)) if finite.size else 1.0
        top = top if top > 0 else 1.0
        model = Field2D(r=r, z=z, values=values, value_label=_CURVATURE_LABELS[name],
                        contour_levels=list(np.linspace(-top, top, 21)), extend="both",
                        region=np.isfinite(values), overlays=overlays)
        render_field_2d(model, ax=axis, cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0.0, vmin=-top, vmax=top))
        kept = np.isfinite(values)
        if kept.any():  # frame the confined region, not the whole computational grid
            rr, zz = np.meshgrid(r, z)
            pad = 0.08 * max(np.ptp(rr[kept]), np.ptp(zz[kept]))
            axis.set_xlim(rr[kept].min() - pad, rr[kept].max() + pad)
            axis.set_ylim(zz[kept].min() - pad, zz[kept].max() + pad)
    if show:
        plt.show()
    return fig, (axes if quantity == "all" else axes[0])
