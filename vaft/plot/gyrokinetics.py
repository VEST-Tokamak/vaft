"""Gyrokinetic figures: linear spectra, eigenfunctions, convergence, nonlinear fluxes.

Plain functions over arrays and native CGYRO results, for the Lane V conference notebook
and the #1354 validation report. They are deliberately **not** registered in
:mod:`vaft.plot.registry` yet (the plotting registry is frozen until 2026-10-06);
registration is a follow-up on the Lane Y issue.

Conventions every function here keeps:

* wavenumbers are ``k_y rho_s`` and rates ``c_s/a`` -- the CGYRO/TGLF normalisation,
  not the IMAS ``gyrokinetics_local`` one, so the overlays need no rescaling;
* frequencies are drawn with the **ion diamagnetic direction negative** (TGLF's
  convention, :attr:`CgyroOutputs.frequency_ion_negative` for CGYRO), and the panel says
  so;
* every function returns ``(figure, axes)`` -- a flat array of axes for the multi-panel
  ones -- and honours ``ax=`` and ``show=`` as the rest of :mod:`vaft.plot` does.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .style import axis_label, finalize, resolve_axes

__all__ = [
    "cgyro_linear_spectrum",
    "plot_convergence",
    "plot_eigenfunction",
    "plot_flux_ky_spectrum",
    "plot_flux_trace",
    "plot_linear_spectrum",
]

_KY = axis_label(r"$k_y \rho_s$")
_GAMMA = axis_label(r"$\gamma$", "c_s/a")
_OMEGA = axis_label(r"$\omega$ (ion dia. < 0)", "c_s/a")


def cgyro_linear_spectrum(runs: Sequence[Any]) -> dict[str, np.ndarray]:
    """Collect single-mode linear CGYRO runs into one spectrum, sorted by ``ky``.

    ``runs`` are :class:`~vaft.code.gacode.cgyro.outputs.CgyroOutputs` (one per ky).
    Unsolved runs are dropped; ``converged`` marks which of the rest met ``FREQ_TOL``.
    """
    rows = []
    for run in runs:
        if run is None or not run.solved or run.ky is None:
            continue
        omega = run.frequency_ion_negative
        rows.append((
            float(run.ky[0]), float(run.final_growth_rate[0]),
            float("nan") if omega is None else float(omega[0]), bool(run.converged),
        ))
    rows.sort()
    columns = list(zip(*rows)) if rows else [(), (), (), ()]
    return {
        "ky": np.asarray(columns[0], dtype=float),
        "gamma": np.asarray(columns[1], dtype=float),
        "omega": np.asarray(columns[2], dtype=float),
        "converged": np.asarray(columns[3], dtype=bool),
    }


def plot_linear_spectrum(
    cgyro: Mapping[str, Any],
    *,
    references: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ax: Any = None,
    show: bool = False,
    figsize: Optional[tuple[float, float]] = None,
    title: Optional[str] = None,
    log_ky: bool = True,
) -> tuple[Any, Any]:
    """Growth rate and frequency against ``k_y rho_s``: CGYRO with reference overlays.

    Parameters
    ----------
    cgyro
        ``{"ky", "gamma", "omega"}`` and optionally ``"converged"`` (from
        :func:`cgyro_linear_spectrum`). The line joins converged points only;
        unconverged ones are drawn hollow and never set the axis range.
    references
        ``{label: {"ky", "gamma", "omega"}}`` -- TGLF linear at the same ky, TGLF's
        SAT0-3 spectra, a second resolution. ``gamma``/``omega`` may be 2-D
        ``(ky, mode)``; the leading mode is drawn solid and the others dotted.
    """
    figure, axes = resolve_axes(ax, nrows=2, ncols=1, figsize=figsize or (6.0, 6.0), sharex=True)
    top, bottom = np.asarray(axes).ravel()

    ky = np.asarray(cgyro["ky"], dtype=float)
    gamma = np.asarray(cgyro["gamma"], dtype=float)
    omega = np.asarray(cgyro["omega"], dtype=float)
    converged = np.asarray(cgyro.get("converged", np.ones_like(ky, dtype=bool)), dtype=bool)
    shown: dict[int, list[np.ndarray]] = {0: [], 1: []}
    for index, (panel, values) in enumerate(((top, gamma), (bottom, omega))):
        # The line joins converged eigenvalues only: an unconverged initial-value run's
        # last sample is not an eigenvalue and must not shape the spectrum.
        good = converged & np.isfinite(values)
        line, = panel.plot(ky[good], values[good], "o-", color="black", linewidth=1.6,
                           label="CGYRO")
        shown[index].append(values[good])
        if np.any(~converged):
            panel.plot(ky[~converged], values[~converged], "o", mfc="none",
                       color=line.get_color(), label="CGYRO (not converged)")

    for label, reference in (references or {}).items():
        ref_ky = np.asarray(reference["ky"], dtype=float)
        for index, (panel, key) in enumerate(((top, "gamma"), (bottom, "omega"))):
            values = np.asarray(reference[key], dtype=float)
            if values.ndim == 1:
                values = values[:, None]
            line, = panel.plot(ref_ky, values[:, 0], "-", linewidth=1.0, label=label)
            shown[index].append(values[:, 0])
            for mode in range(1, values.shape[1]):
                panel.plot(ref_ky, values[:, mode], ":", linewidth=0.8, color=line.get_color())

    # Scale each panel to the converged CGYRO points and the references, so a stray
    # unconverged sample (|gamma| ~ 10-40 at high collisionality) cannot flatten the
    # spectrum; the points left outside are counted on the panel instead.
    for index, (panel, values) in enumerate(((top, gamma), (bottom, omega))):
        finite = np.concatenate([v[np.isfinite(v)] for v in shown[index]] or [np.zeros(1)])
        if finite.size == 0:
            continue
        low, high = float(min(finite.min(), 0.0)), float(max(finite.max(), 0.0))
        pad = 0.08 * (high - low or 1.0)
        panel.set_ylim(low - pad, high + pad)
        outside = int(np.count_nonzero(
            ~converged & np.isfinite(values) & ((values < low - pad) | (values > high + pad))))
        if outside:
            panel.text(0.99, 0.02, f"{outside} unconverged point(s) off scale",
                       transform=panel.transAxes, ha="right", va="bottom",
                       fontsize="small", alpha=0.7)

    bottom.axhline(0.0, color="0.6", linewidth=0.8)
    top.set_ylabel(_GAMMA)
    bottom.set_ylabel(_OMEGA)
    bottom.set_xlabel(_KY)
    if log_ky:
        bottom.set_xscale("log")
    top.legend(fontsize="small")
    if title:
        top.set_title(title)
    return finalize(figure, np.array([top, bottom], dtype=object), show=show,
                    tight_layout=ax is None)


def plot_eigenfunction(
    theta: Any,
    fields: Mapping[str, Any],
    *,
    ax: Any = None,
    show: bool = False,
    figsize: Optional[tuple[float, float]] = None,
    title: Optional[str] = None,
    normalise_to: Optional[str] = "phi",
) -> tuple[Any, Any]:
    """Ballooning-space eigenfunctions, one panel per field: Re, Im and ``|.|``.

    ``theta`` is the extended ballooning angle (``CgyroOutputs.grid["thetab"]``) and
    ``fields`` maps a name (``"phi"``, ``"a_parallel"``) to the complex field on it.
    With ``normalise_to`` every field is divided by that field's value at its peak, so
    the relative amplitude and phase of A_par against phi survive.
    """
    names = list(fields)
    figure, axes = resolve_axes(ax, nrows=len(names), ncols=1,
                                figsize=figsize or (6.0, 2.6 * len(names)), sharex=True)
    panels = np.atleast_1d(np.asarray(axes, dtype=object)).ravel()
    theta = np.asarray(theta, dtype=float)
    order = np.argsort(theta)
    scale = 1.0
    if normalise_to and normalise_to in fields:
        reference = np.asarray(fields[normalise_to])
        scale = reference[int(np.argmax(np.abs(reference)))]
        if scale == 0:
            scale = 1.0
    for panel, name in zip(panels, names):
        values = np.asarray(fields[name])[order] / scale
        x = theta[order] / np.pi
        panel.plot(x, values.real, label="Re")
        panel.plot(x, values.imag, label="Im")
        panel.plot(x, np.abs(values), color="black", linewidth=1.2, label="|.|")
        panel.set_ylabel(name)
        panel.axhline(0.0, color="0.7", linewidth=0.6)
    panels[0].legend(fontsize="small", ncol=3)
    panels[-1].set_xlabel(r"$\theta_b / \pi$")
    if title:
        panels[0].set_title(title)
    return finalize(figure, panels, show=show, tight_layout=ax is None)


def plot_convergence(
    scans: Mapping[str, Mapping[str, Any]],
    *,
    baseline: Mapping[str, float],
    tolerance: float = 0.05,
    ax: Any = None,
    show: bool = False,
    figsize: Optional[tuple[float, float]] = None,
    title: Optional[str] = None,
) -> tuple[Any, Any]:
    """Relative change of ``gamma`` and ``omega`` against each resolution parameter.

    ``scans`` maps a parameter (``"n_theta"``) to ``{"value", "gamma", "omega"}`` arrays;
    ``baseline`` is ``{"gamma", "omega"}`` at the production resolution. The shaded band
    is the acceptance tolerance.
    """
    names = list(scans)
    figure, axes = resolve_axes(ax, nrows=1, ncols=len(names),
                                figsize=figsize or (3.0 * len(names), 3.0), sharey=True)
    panels = np.atleast_1d(np.asarray(axes, dtype=object)).ravel()
    for panel, name in zip(panels, names):
        scan = scans[name]
        values = np.asarray(scan["value"], dtype=float)
        for key, marker in (("gamma", "o"), ("omega", "s")):
            reference = float(baseline[key])
            change = (np.asarray(scan[key], dtype=float) - reference) / abs(reference)
            panel.plot(values, change, marker=marker, label=key)
        panel.axhspan(-tolerance, tolerance, color="0.9", zorder=0)
        panel.axhline(0.0, color="0.6", linewidth=0.6)
        panel.set_xlabel(name)
    panels[0].set_ylabel("relative change")
    panels[0].legend(fontsize="small")
    if title:
        figure.suptitle(title)
    return finalize(figure, panels, show=show, tight_layout=ax is None)


def plot_flux_trace(
    time: Any,
    traces: Mapping[str, Any],
    *,
    window: Optional[tuple[float, float]] = None,
    references: Optional[Mapping[str, float]] = None,
    ax: Any = None,
    show: bool = False,
    figsize: Optional[tuple[float, float]] = None,
    ylabel: str = axis_label(r"$Q/Q_{GB}$"),
    title: Optional[str] = None,
) -> tuple[Any, Any]:
    """Nonlinear flux against time, with the averaging window and reference levels.

    ``traces`` maps a label (``"Q_i"``, ``"Q_e"``) to a flux time series in gyro-Bohm
    units; ``references`` maps a label (``"TGLF SAT2"``) to a level drawn dashed, so the
    saturated CGYRO flux reads directly against the SAT rules.
    """
    figure, axes = resolve_axes(ax, figsize=figsize or (6.5, 3.5))
    t = np.asarray(time, dtype=float)
    for label, values in traces.items():
        values = np.asarray(values, dtype=float)
        axes.plot(t[: values.size], values, label=label)
    if window is not None:
        axes.axvspan(window[0], window[1], color="0.9", zorder=0, label="average window")
    for label, level in (references or {}).items():
        axes.axhline(float(level), linestyle="--", linewidth=0.9, label=label)
    axes.set_xlabel(axis_label(r"$t$", "a/c_s"))
    axes.set_ylabel(ylabel)
    axes.legend(fontsize="small")
    if title:
        axes.set_title(title)
    return finalize(figure, axes, show=show, tight_layout=ax is None)


def plot_flux_ky_spectrum(
    ky: Any,
    spectra: Mapping[str, Any],
    *,
    ax: Any = None,
    show: bool = False,
    figsize: Optional[tuple[float, float]] = None,
    ylabel: str = axis_label(r"$Q/Q_{GB}$ per mode"),
    title: Optional[str] = None,
) -> tuple[Any, Any]:
    """Time-averaged flux per toroidal mode against ``k_y rho_s``."""
    figure, axes = resolve_axes(ax, figsize=figsize or (6.0, 3.5))
    ky = np.asarray(ky, dtype=float)
    for label, values in spectra.items():
        axes.plot(ky, np.asarray(values, dtype=float), marker="o", label=label)
    axes.set_xlabel(_KY)
    axes.set_ylabel(ylabel)
    axes.legend(fontsize="small")
    if title:
        axes.set_title(title)
    return finalize(figure, axes, show=show, tight_layout=ax is None)
