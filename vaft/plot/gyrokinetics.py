"""Native gyrokinetic figures: TGLF/CGYRO validation in the solvers' own units.

This is the **native** layer of the gyrokinetic plot family (#1591): plain functions
over arrays and native results (:class:`~vaft.code.gacode.tglf.outputs.TglfOutputs`,
:class:`~vaft.code.gacode.cgyro.outputs.CgyroOutputs`), meant for matched TGLF-CGYRO
comparisons and solver diagnostics. The **standardized** layer -- registered plots that
read IMAS ``gyrokinetics_local`` / ``core_transport`` -- is separate and uses the IMAS/GKDB
normalisation; the two are never mixed (#1591 section 8). These functions are not in
:mod:`vaft.plot.registry`.

The inventory follows the scientific questions MITIM's ``TGLF.plot`` answers (Summary,
Contributors, Spectra, Model Details, SAT Parameters, Input Plasma), as a benchmark only:
MITIM is not imported, and a quantity is drawn here only from a parsed native output
whose definition is cited in :mod:`vaft.code.gacode.tglf.outputs`.

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
    "plot_fluctuation_spectra",
    "plot_flux_contributors",
    "plot_flux_ky_spectrum",
    "plot_flux_trace",
    "plot_linear_spectrum",
    "plot_local_state",
    "plot_mixing_length_proxy",
    "plot_predator_prey_lag",
    "plot_predator_prey_phase_space",
    "plot_predator_prey_trace",
    "plot_model_details",
    "plot_saturation_parameters",
    "tglf_flux_contributors",
    "tglf_linear_spectrum",
    "tglf_local_state",
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


# ---------------------------------------------------------------------------------
# TGLF native extraction: slicing of parsed arrays into named series, no physics.
# ---------------------------------------------------------------------------------

_FIELD_LABELS = {"phi": r"$\phi$", "a_par": r"$A_\parallel$", "b_par": r"$B_\parallel$"}
_SPECIES_SYMBOL = {"particle": r"\Gamma", "energy": "Q"}


def tglf_linear_spectrum(outputs: Any) -> dict[str, Any]:
    """``{"ky", "gamma", "omega", "preset"}`` from one TGLF run, ``(ky, mode)`` arrays.

    ``preset`` names the linear model family: TGLF's presets couple the linear model to
    the saturation rule (SAT2/3 use ``XNU_MODEL=3`` and ``WDIA_TRAPPED=1``), so a TGLF
    linear spectrum is only meaningful with the family it was computed in.
    """
    if outputs is None or outputs.growth_rate is None or outputs.ky_spectrum is None:
        raise ValueError("this TGLF run carries no eigenvalue spectrum")
    params = outputs.saturation_parameters or {}
    preset = None
    if params:
        preset = (f"SAT{params.get('SAT_RULE')} / XNU_MODEL={params.get('XNU_MODEL')} / "
                  f"UNITS={params.get('UNITS')}")
    return {"ky": np.asarray(outputs.ky_spectrum, dtype=float),
            "gamma": np.asarray(outputs.growth_rate, dtype=float),
            "omega": np.asarray(outputs.frequency, dtype=float), "preset": preset}


def tglf_flux_contributors(
    outputs: Any, *, quantity: str = "energy", species: Sequence[str] = ("e", "i"),
) -> dict[str, Any]:
    """Field-resolved saturated flux spectra from ``sum_flux_spectrum``.

    Returns ``{"ky", "series": {label: {field: (ky,) array, "total": ...}}}`` where each
    value is TGLF's ky-weighted flux per ky bin (the bins sum to the total flux). Only the
    fields TGLF wrote are present; the electromagnetic part is *never* formed as a
    difference of totals. ``species``: ``"e"`` (electrons, TGLF species 1), ``"i"``
    (sum over all ions) or an integer index (1-based, TGLF order).
    """
    from vaft.code.gacode.tglf.outputs import FLUX_SPECTRUM_FIELDS, FLUX_SPECTRUM_QUANTITIES

    spectrum = None if outputs is None else outputs.sum_flux_spectrum
    if spectrum is None:
        raise ValueError("this TGLF run carries no sum_flux_spectrum")
    q = FLUX_SPECTRUM_QUANTITIES.index(quantity)
    nfield = spectrum.shape[1]
    count = spectrum.shape[0]
    series: dict[str, dict[str, np.ndarray]] = {}
    for spec in species:
        if spec == "e":
            block, name = spectrum[0:1], "e"
        elif spec == "i":
            if count < 2:
                raise ValueError("this run has no ion species")
            block, name = spectrum[1:], "i"
        else:
            index = int(spec)
            if not 1 <= index <= count:
                raise ValueError(f"species index {spec!r} is outside 1..{count} (TGLF order)")
            block, name = spectrum[index - 1:index], f"s{index}"
        fields = {}
        for f in range(nfield):
            values = block[:, f, :, q]
            if np.all(np.isnan(values)):
                continue          # a field the run did not include: absent, not zero
            fields[FLUX_SPECTRUM_FIELDS[f]] = values.sum(axis=0)
        if fields:
            fields["total"] = sum(fields.values())
        else:
            # No field carries this quantity for the species (a partial or malformed
            # spectrum): the total is unknown, not zero, and keeps the ky shape.
            fields["total"] = np.full(block.shape[2], np.nan)
        series[name] = fields
    return {"ky": np.asarray(outputs.ky_spectrum, dtype=float), "series": series,
            "quantity": quantity}


def tglf_local_state(local: Any) -> dict[str, Any]:
    """The local input TGLF was given, with each value's provenance kind.

    ``local`` is a :class:`~vaft.code.gacode.tglf.inputs.TGLFInput`. Returns
    ``{"species": [{name, a_over_Ln, a_over_LT, n_over_ne, T_over_Te}], "scalars":
    {...}, "provenance": {scalar label: kind}}``; kinds come from the input's own
    provenance record (``derived``, ``unavailable``, ``caller_supplied``, ...), falling
    back to the record of the whole projection (``source``), never guessed.
    """
    species = []
    for index, name in enumerate(local.names or [f"s{i + 1}" for i in range(local.n_species)]):
        species.append({"name": name, "a_over_Ln": float(local.rlns[index]),
                        "a_over_LT": float(local.rlts[index]),
                        "n_over_ne": float(local.as_[index]),
                        "T_over_Te": float(local.taus[index])})
    rmin, q = float(local.rmin_loc), float(local.q_loc)
    scalars = {
        "r/a": rmin, "R0/a": float(local.rmaj_loc), "q": q,
        "s": float(local.q_prime_loc) * (rmin / q) ** 2 if q else float("nan"),
        "kappa": float(local.kappa_loc), "delta": float(local.delta_loc),
        "beta_e": float(local.betae), "nu_ee a/c_s": float(local.xnue),
        "Z_eff": float(local.zeff),
        "ExB shear": None if local.vexb_shear is None else float(local.vexb_shear),
    }
    records = {key: dict(value).get("kind") for key, value in (local.provenance or {}).items()}
    # Every scalar gets a kind: its own record when the input has one, otherwise the
    # input's overall source kind (the projection that produced all local values).
    keys = {"r/a": "rho", "R0/a": "rmaj_loc", "q": "q_loc", "s": "q_prime_loc",
            "kappa": "kappa_loc", "delta": "delta_loc", "beta_e": "betae",
            "nu_ee a/c_s": "xnue", "Z_eff": "zeff", "ExB shear": "vexb_shear"}
    default = records.get("source")
    provenance = {label: records.get(keys[label], default) for label in scalars}
    return {"species": species, "scalars": scalars, "provenance": provenance}


# ---------------------------------------------------------------------------------
# Native renderers
# ---------------------------------------------------------------------------------


def plot_flux_contributors(
    contributors: Mapping[str, Any],
    *,
    ax: Any = None,
    show: bool = False,
    figsize: Optional[tuple[float, float]] = None,
    title: Optional[str] = None,
    log_ky: bool = True,
) -> tuple[Any, Any]:
    """Saturated flux per ky bin split by field, one panel per species group.

    ``contributors`` is :func:`tglf_flux_contributors`'s result (or the same shape from
    another solver). Each panel draws the total and every field the solver wrote
    (phi, A_parallel, B_parallel); a field it did not write is absent, not zero.
    """
    series = contributors["series"]
    names = list(series)
    figure, axes = resolve_axes(ax, nrows=1, ncols=len(names),
                                figsize=figsize or (4.2 * len(names), 3.4), sharey=False)
    panels = np.atleast_1d(np.asarray(axes, dtype=object)).ravel()
    ky = np.asarray(contributors["ky"], dtype=float)
    symbol = _SPECIES_SYMBOL.get(contributors.get("quantity", "energy"), contributors.get("quantity"))
    for panel, name in zip(panels, names):
        fields = series[name]
        for field, values in fields.items():
            if field == "total":
                continue
            panel.plot(ky, values, linewidth=1.2, label=_FIELD_LABELS.get(field, field))
        # Dashed and on top: when one field carries the flux the total coincides with
        # it, and a solid line underneath would vanish.
        panel.plot(ky, fields["total"], color="black", linewidth=1.1, linestyle="--",
                   zorder=5, label="total")
        if np.all(np.isnan(fields["total"])):
            panel.text(0.5, 0.5, f"no {contributors.get('quantity', 'energy')} flux "
                       f"written for {name}", transform=panel.transAxes, ha="center",
                       va="center", fontsize="small", color="0.4")
        panel.axhline(0.0, color="0.7", linewidth=0.6)
        panel.set_xlabel(_KY)
        panel.set_ylabel(axis_label(fr"${symbol}_{{{name}}}$ per $k_y$ bin", "GB"))
        if log_ky:
            panel.set_xscale("log")
        panel.legend(fontsize="small")
    if title:
        figure.suptitle(title)
    return finalize(figure, panels, show=show, tight_layout=ax is None)


def plot_mixing_length_proxy(
    spectrum: Mapping[str, Any],
    *,
    definition: str = "gamma_over_ky2",
    ax: Any = None,
    show: bool = False,
    figsize: Optional[tuple[float, float]] = None,
    title: Optional[str] = None,
) -> tuple[Any, Any]:
    """A reduced mixing-length diagnostic from a linear spectrum, with its definition.

    ``definition``:

    ``"gamma_over_ky2"``
        ``gamma / k_y^2`` -- the mixing-length estimate ``gamma / k_perp^2`` evaluated at
        ``k_x = 0`` (``k_perp = k_y``); a diffusivity scale in gyro-Bohm units.
    ``"gamma_over_ky"``
        ``gamma / k_y`` -- a different quantity (a velocity scale), drawn when asked for.

    Neither is a saturated-flux or zonal-flow predictor; the panel says which one it is.
    Only positive growth rates enter (a stable mode has no mixing length).
    """
    ky = np.asarray(spectrum["ky"], dtype=float)
    gamma = np.asarray(spectrum["gamma"], dtype=float)
    if gamma.ndim == 1:
        gamma = gamma[:, None]
    power = {"gamma_over_ky2": 2, "gamma_over_ky": 1}.get(definition)
    if power is None:
        raise ValueError(f"definition is 'gamma_over_ky2' or 'gamma_over_ky'; got {definition!r}")
    figure, axes = resolve_axes(ax, figsize=figsize or (5.5, 3.4))
    for mode in range(gamma.shape[1]):
        values = np.where(gamma[:, mode] > 0, gamma[:, mode] / ky**power, np.nan)
        axes.plot(ky, values, "-" if mode == 0 else ":", label=f"mode {mode + 1}")
    label = r"$\gamma / k_y^2$ ($k_x=0$)" if power == 2 else r"$\gamma / k_y$"
    axes.set_xscale("log")
    axes.set_xlabel(_KY)
    axes.set_ylabel(axis_label(label, "GB"))
    axes.text(0.99, 0.97, "reduced diagnostic, not a flux model", transform=axes.transAxes,
              ha="right", va="top", fontsize="small", alpha=0.7)
    axes.legend(fontsize="small")
    if title:
        axes.set_title(title)
    return finalize(figure, axes, show=show, tight_layout=ax is None)


def plot_fluctuation_spectra(
    ky: Any,
    amplitudes: Mapping[str, Any],
    *,
    cross_phase: Optional[Any] = None,
    ax: Any = None,
    show: bool = False,
    figsize: Optional[tuple[float, float]] = None,
    title: Optional[str] = None,
) -> tuple[Any, Any]:
    """Fluctuation amplitude spectra and, optionally, the n_e-T_e cross phase.

    ``amplitudes`` maps labels (``"dn_e/n_e"``, ``"dT_e/T_e"``) to ``(ky,)`` gyro-Bohm
    amplitudes -- for TGLF, the columns of ``density_spectrum``/``temperature_spectrum``
    (``sqrt`` of the mode-summed intensity). ``cross_phase`` is ``(ky, mode)`` [rad],
    drawn in degrees.
    """
    rows = 2 if cross_phase is not None else 1
    figure, axes = resolve_axes(ax, nrows=rows, ncols=1, figsize=figsize or (5.5, 3.0 * rows),
                                sharex=True)
    panels = np.atleast_1d(np.asarray(axes, dtype=object)).ravel()
    ky = np.asarray(ky, dtype=float)
    for label, values in amplitudes.items():
        panels[0].plot(ky, np.asarray(values, dtype=float), label=label)
    panels[0].set_ylabel(axis_label("amplitude", "GB"))
    panels[0].legend(fontsize="small")
    if cross_phase is not None:
        phase = np.atleast_2d(np.asarray(cross_phase, dtype=float).T).T
        for mode in range(phase.shape[1]):
            panels[1].plot(ky, np.degrees(phase[:, mode]), "-" if mode == 0 else ":",
                           label=f"mode {mode + 1}")
        panels[1].axhline(0.0, color="0.7", linewidth=0.6)
        panels[1].set_ylabel(axis_label(r"$n_e$-$T_e$ cross phase", "deg"))
        panels[1].legend(fontsize="small")
    panels[-1].set_xscale("log")
    panels[-1].set_xlabel(_KY)
    if title:
        panels[0].set_title(title)
    return finalize(figure, panels, show=show, tight_layout=ax is None)


def plot_model_details(
    ky: Any,
    details: Mapping[str, Any],
    *,
    ax: Any = None,
    show: bool = False,
    figsize: Optional[tuple[float, float]] = None,
    title: Optional[str] = None,
) -> tuple[Any, Any]:
    """TGLF model internals vs ky: Gaussian width, spectral shift, SAT0 normalisation.

    ``details`` maps a label to a ``(ky,)`` array -- for TGLF, ``width_spectrum``,
    ``spectral_shift_spectrum`` and ``ave_p0_spectrum``. Absent entries are skipped.
    """
    items = [(k, v) for k, v in details.items() if v is not None]
    figure, axes = resolve_axes(ax, nrows=1, ncols=max(len(items), 1),
                                figsize=figsize or (3.4 * max(len(items), 1), 3.0))
    panels = np.atleast_1d(np.asarray(axes, dtype=object)).ravel()
    ky = np.asarray(ky, dtype=float)
    for panel, (label, values) in zip(panels, items):
        panel.plot(ky, np.asarray(values, dtype=float), marker=".")
        panel.set_xscale("log")
        panel.set_xlabel(_KY)
        panel.set_ylabel(label)
    if title:
        figure.suptitle(title)
    return finalize(figure, panels, show=show, tight_layout=ax is None)


def _text_table(panel: Any, rows: Sequence[tuple[str, str]], *, title: str) -> None:
    panel.axis("off")
    panel.set_title(title, fontsize="medium", loc="left")
    text = "\n".join(f"{key:<16s} {value}" for key, value in rows)
    panel.text(0.0, 1.0, text, family="monospace", fontsize="small", va="top",
               transform=panel.transAxes)


def plot_saturation_parameters(
    parameters: Mapping[str, Any],
    *,
    ax: Any = None,
    show: bool = False,
    figsize: Optional[tuple[float, float]] = None,
    title: str = "TGLF saturation parameters",
) -> tuple[Any, Any]:
    """The scalar saturation parameters of a TGLF run as a text panel.

    Values are shown as TGLF wrote them (``out.tglf.scalar_saturation_parameters``); the
    preset line notes that SAT2/3 also change the linear model (``XNU_MODEL``).
    """
    figure, axes = resolve_axes(ax, figsize=figsize or (4.5, 4.2))
    rows = []
    for key, value in parameters.items():
        rows.append((key, f"{value:.4g}" if isinstance(value, float) else str(value)))
    _text_table(axes, rows, title=title)
    return finalize(figure, axes, show=show, tight_layout=ax is None)


def plot_local_state(
    state: Mapping[str, Any],
    *,
    ax: Any = None,
    show: bool = False,
    figsize: Optional[tuple[float, float]] = None,
    title: Optional[str] = None,
) -> tuple[Any, Any]:
    """The local input a solver was given: gradients per species and geometry scalars.

    ``state`` is :func:`tglf_local_state`'s result. Left: ``a/L_n`` and ``a/L_T`` per
    species. Right: geometry and plasma scalars, each tagged with its provenance kind
    when the input records one (``[derived]``, ``[unavailable]``, ...), so an assumed or
    absent quantity is visibly different from a reconstructed one.
    """
    figure, axes = resolve_axes(ax, nrows=1, ncols=2, figsize=figsize or (9.0, 3.6))
    left, right = np.atleast_1d(np.asarray(axes, dtype=object)).ravel()
    species = state["species"]
    x = np.arange(len(species))
    left.bar(x - 0.2, [s["a_over_Ln"] for s in species], width=0.4, label=r"$a/L_n$")
    left.bar(x + 0.2, [s["a_over_LT"] for s in species], width=0.4, label=r"$a/L_T$")
    left.set_xticks(x, [s["name"] for s in species])
    left.axhline(0.0, color="0.7", linewidth=0.6)
    left.legend(fontsize="small")
    left.set_ylabel("normalised gradient")
    kinds = state.get("provenance") or {}
    rows = []
    for key, value in state["scalars"].items():
        if value is None:
            # The solver still ran with its default (zero for the ExB shear); the tag
            # says nobody measured it.
            text = "0 (solver default)"
        else:
            text = f"{value:.4g}" if isinstance(value, float) else str(value)
        kind = kinds.get(key)
        rows.append((key, f"{text}  [{kind}]" if kind else text))
    _text_table(right, rows, title="local state")
    if title:
        figure.suptitle(title)
    return finalize(figure, np.array([left, right], dtype=object), show=show,
                    tight_layout=ax is None)


# ---------------------------------------------------------------------------------
# Turbulence--zonal-flow predator--prey views (#1820). They draw what
# vaft.process.gyrokinetics derived; no peak detection, correlation or fitting here.
# Fitted overlays are dashed, data solid, so the two never read as one curve.
# ---------------------------------------------------------------------------------

_FIT_STYLE = {"linestyle": "--", "linewidth": 1.4, "color": "k", "alpha": 0.8}


def _fit_times(time: np.ndarray, fit: Any) -> np.ndarray:
    """The data times a ``PredatorPreyFit`` was evaluated on, recovered from its window."""
    t0, t1 = fit.time_window
    inside = time[(time >= t0) & (time <= t1)]
    if inside.size != np.asarray(fit.turbulence_fit).size:
        raise ValueError(
            "fit does not belong to this time axis: its window holds "
            f"{inside.size} samples here but the fit has {np.asarray(fit.turbulence_fit).size}")
    return inside


def plot_predator_prey_trace(
    time: Any,
    turbulence: Any,
    zonal: Any,
    *,
    transport: Optional[Any] = None,
    fit: Optional[Any] = None,
    metrics: Optional[Any] = None,
    log: bool = True,
    ax: Any = None,
    show: bool = False,
    figsize: Optional[tuple[float, float]] = None,
    time_label: str = axis_label(r"$t$", "a/c_s"),
    transport_label: str = axis_label(r"$Q/Q_{GB}$"),
    title: Optional[str] = None,
) -> tuple[Any, Any]:
    """Turbulence and zonal-flow intensity against time, with optional transport and fit.

    Panels: ``N(t)``, ``E(t)``, and ``Q(t)`` when ``transport`` is given. ``fit`` (a
    :class:`~vaft.process.gyrokinetics.PredatorPreyFit` on the same ``time``) overlays the
    fitted trajectories dashed; ``metrics`` (a
    :class:`~vaft.process.gyrokinetics.PredatorPreyCycleMetrics`) marks the peaks it
    found. Nothing is detected or fitted here.
    """
    t = np.asarray(time, dtype=float)
    n = np.asarray(turbulence, dtype=float)
    e = np.asarray(zonal, dtype=float)
    rows = 3 if transport is not None else 2
    figure, axes = resolve_axes(ax, nrows=rows, ncols=1, sharex=True,
                                figsize=figsize or (7.0, 2.2 * rows + 0.6))
    panels = np.atleast_1d(np.asarray(axes, dtype=object)).ravel()
    panels[0].plot(t, n, color="C3", label="N (turbulence)")
    panels[1].plot(t, e, color="C2", label="E (zonal)")
    if fit is not None:
        tf = _fit_times(t, fit)
        panels[0].plot(tf, fit.turbulence_fit, label=f"LV fit (R$^2$ {fit.r2_turbulence:.2f})", **_FIT_STYLE)
        panels[1].plot(tf, fit.zonal_fit, label=f"LV fit (R$^2$ {fit.r2_zonal:.2f})", **_FIT_STYLE)
    if metrics is not None:
        for peak in metrics.turbulence_peak_times:
            panels[0].axvline(peak, color="C3", linewidth=0.6, alpha=0.5)
        for peak in metrics.zonal_peak_times:
            panels[1].axvline(peak, color="C2", linewidth=0.6, alpha=0.5)
    panels[0].set_ylabel(axis_label("N", "potential intensity"))
    panels[1].set_ylabel(axis_label("E", "potential intensity"))
    if transport is not None:
        q = np.asarray(transport, dtype=float)
        panels[2].plot(t[: q.size], q, color="k", linewidth=1.0, label="Q")
        panels[2].set_ylabel(transport_label)
    for panel in panels[:2]:
        if log:
            panel.set_yscale("log")
        panel.legend(fontsize="small", loc="upper right")
    panels[-1].set_xlabel(time_label)
    if title:
        panels[0].set_title(title)
    return finalize(figure, panels, show=show, tight_layout=ax is None)


def plot_predator_prey_phase_space(
    turbulence: Any,
    zonal: Any,
    *,
    time: Optional[Any] = None,
    fit: Optional[Any] = None,
    arrows: int = 8,
    vector_field: bool = True,
    log: bool = True,
    ax: Any = None,
    show: bool = False,
    figsize: Optional[tuple[float, float]] = None,
    title: Optional[str] = None,
) -> tuple[Any, Any]:
    """The measured ``N``-``E`` trajectory, with the fitted model's structure when given.

    The data trajectory is drawn with ``arrows`` direction markers (coloured by ``time``
    when it is given). With ``fit``: the fitted trajectory (dashed), both nullclines
    (``E = gamma_eff/c1`` where ``dN/dt = 0``, ``N = gamma_Z/c2`` where ``dE/dt = 0``),
    their intersection, the positive fixed point, and the model's vector field. The
    model quantities are evaluated from :mod:`vaft.formula.turbulence`; nothing is fitted.
    """
    from matplotlib.collections import LineCollection

    figure, axes = resolve_axes(ax, figsize=figsize or (5.2, 4.6))
    n = np.asarray(turbulence, dtype=float)
    e = np.asarray(zonal, dtype=float)
    if n.shape != e.shape or n.ndim != 1 or n.size < 2:
        raise ValueError("turbulence and zonal must be 1-D traces of equal length (>= 2)")
    if log and (np.any(n <= 0) or np.any(e <= 0)):
        raise ValueError("log axes need strictly positive intensities")
    segments = np.stack([np.column_stack([n[:-1], e[:-1]]), np.column_stack([n[1:], e[1:]])], axis=1)
    colour = np.asarray(time, dtype=float)[:-1] if time is not None else np.arange(n.size - 1)
    line = LineCollection(segments, cmap="viridis", linewidths=1.2)
    line.set_array(colour)
    axes.add_collection(line)
    if time is not None:
        figure.colorbar(line, ax=axes, label=axis_label(r"$t$", "a/c_s"))
    step = max(1, n.size // max(int(arrows), 1))
    for k in range(step // 2, n.size - 1, step):
        axes.annotate("", xy=(n[k + 1], e[k + 1]), xytext=(n[k], e[k]),
                      arrowprops={"arrowstyle": "->", "color": "0.2", "lw": 1.0})
    if fit is not None:
        from vaft.formula.turbulence import predator_prey_fixed_point, predator_prey_rhs

        params = (fit.gamma_eff, fit.coupling_suppression, fit.coupling_drive, fit.gamma_zonal)
        n_star, e_star = predator_prey_fixed_point(*params)
        axes.plot(fit.turbulence_fit, fit.zonal_fit, label="LV fit", **_FIT_STYLE)
        axes.axhline(e_star, color="C3", linestyle=":", linewidth=1.0, label=r"$\dot N = 0$")
        axes.axvline(n_star, color="C2", linestyle=":", linewidth=1.0, label=r"$\dot E = 0$")
        axes.plot([n_star], [e_star], marker="*", markersize=11, color="k", linestyle="none",
                  label="fixed point")
        if vector_field:
            lo_n, hi_n = min(n.min(), n_star), max(n.max(), n_star)
            lo_e, hi_e = min(e.min(), e_star), max(e.max(), e_star)
            grid = (np.geomspace if log else np.linspace)
            gn, ge = np.meshgrid(grid(lo_n, hi_n, 12), grid(lo_e, hi_e, 12))
            dn, de = predator_prey_rhs(gn, ge, *params)
            if log:   # direction in log coordinates: d ln N/dt, d ln E/dt
                dn, de = dn / gn, de / ge
            norm = np.hypot(dn, de)
            norm[norm == 0] = 1.0
            axes.quiver(gn, ge, dn / norm, de / norm, color="0.6", alpha=0.6, width=0.003,
                        angles="uv", pivot="mid")
        axes.legend(fontsize="small", loc="best")
    axes.autoscale_view()
    if log:
        axes.set_xscale("log")
        axes.set_yscale("log")
    axes.set_xlabel(axis_label("N (turbulence)", "potential intensity"))
    axes.set_ylabel(axis_label("E (zonal)", "potential intensity"))
    if title:
        axes.set_title(title)
    return finalize(figure, axes, show=show, tight_layout=ax is None)


def plot_predator_prey_lag(
    lags: Any,
    cross_correlation: Any,
    *,
    peak_lag: Optional[float] = None,
    model_lag: Optional[float] = None,
    correlation_lag: Optional[float] = None,
    ax: Any = None,
    show: bool = False,
    figsize: Optional[tuple[float, float]] = None,
    time_label: str = axis_label(r"lag $\tau$ (zonal after turbulence)", "a/c_s"),
    title: Optional[str] = None,
) -> tuple[Any, Any]:
    """Cross-correlation of ``N`` and ``E`` against lag, with the lag estimates marked.

    ``lags`` and ``cross_correlation`` are the curve
    :func:`~vaft.process.gyrokinetics.predator_prey_cycle_metrics` returned (positive lag =
    zonal flow after turbulence); ``correlation_lag`` (its maximum), ``peak_lag`` (the mean
    peak-to-peak response lag) and ``model_lag`` (e.g. the linearized LV ``T/4`` of a fit)
    are drawn as vertical markers. The correlation is not recomputed.
    """
    figure, axes = resolve_axes(ax, figsize=figsize or (5.5, 3.4))
    tau = np.asarray(lags, dtype=float)
    cc = np.asarray(cross_correlation, dtype=float)
    if tau.shape != cc.shape:
        raise ValueError("lags and cross_correlation must have the same shape")
    axes.plot(tau, cc, color="k", linewidth=1.2, label="corr(N(t), E(t+$\\tau$))")
    axes.axhline(0.0, color="0.7", linewidth=0.6)
    axes.axvline(0.0, color="0.7", linewidth=0.6)
    for value, label, style in (
        (correlation_lag, "correlation maximum", {"color": "C0", "linestyle": "-"}),
        (peak_lag, "peak response lag", {"color": "C1", "linestyle": "--"}),
        (model_lag, "LV model lag", {"color": "C2", "linestyle": ":"}),
    ):
        if value is not None and np.isfinite(value):
            axes.axvline(float(value), linewidth=1.3, label=f"{label} {float(value):.3g}", **style)
    axes.set_xlabel(time_label)
    axes.set_ylabel("cross-correlation")
    axes.legend(fontsize="small")
    if title:
        axes.set_title(title)
    return finalize(figure, axes, show=show, tight_layout=ax is None)

