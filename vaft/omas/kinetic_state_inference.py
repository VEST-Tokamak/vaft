"""The kinetic state inferred in named, inspectable stages: Romero first, ``T_i = T_e``, ``n_C = n_O``.

The inference behind the ``kinetic_overview_state`` figure uses only the
equilibrium (``p_eq`` and the geometry) and Thomson scattering (``n_e``, ``T_e``),
and two assumptions it states and records:

* ``T_i = T_e``, and
* ``n_C = n_O`` -- the impurity mix is carbon and oxygen at equal densities.

It runs in seven stages, each of which adds quantities and says where they land
in ``core_profiles``:

====  ====================  =======================================================
 #    stage                 adds
====  ====================  =======================================================
 1    ``data``              ``p_eq(rho)``, the geometry, the Thomson channel points
 2    ``thomson_fits``      ``n_e(rho)``, ``T_e(rho)``, ``p_e``
 3    ``romero_zeff``       the Romero resistive ``Z_eff`` of the discharge window
 4    ``impurity_amplitude``  the one impurity amplitude ``c`` the Romero value fixes:
                            ``n_I,eff(rho)``, ``Z_I,eff(rho)``
 5    ``ion_density``       ``n_main(rho)`` by quasi-neutrality, the charge states,
                            ``n_i,tot(rho)``
 6    ``zeff_profile``      ``Z_eff(rho)``, and the resistive scalar it projects to
 7    ``pressure_closure``  ``T_i = T_e``, ``p_e + p_i + p_imp`` against ``p_eq``
====  ====================  =======================================================

The physics is not here.  Stage 3 is Lane Z's estimator
(:func:`vaft.omas.resistive_zeff.compute_resistive_zeff_ods`); stages 4-6 are Lane
L's :func:`vaft.process.impurity.resolve_radial_composition` with
``normalization="resistive_closure"`` -- the amplitude whose resistive projection
(:func:`vaft.process.zeff_projection.project_zeff_profile_to_resistive_scalar`) equals
the Romero scalar, slice by slice; stage 7 is the dilution-aware closure of
:mod:`vaft.process.kinetic_closure`.  This module orders them, keeps each stage's
result, and writes the product the last stage leaves behind.

**Stage 7 is a consistency check, not a fit.**  By then every degree of freedom is
spent: ``T_i`` is assumed equal to ``T_e`` and the impurity amplitude was fixed in
stage 4.  The residual ``p_eq - p_thermal`` says whether the equilibrium pressure is
compatible with the assumptions; nothing is adjusted to reduce it.

**A stage that cannot be made says why, and stops the chain.**  Every later stage is
``unavailable`` with the reason, never silently filled: a Romero estimate on a bound
or not identifiable gives no impurity amplitude, hence no ``Z_eff`` profile and no
closure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
import math
from typing import Any, Mapping, Sequence

import numpy as np
from omas import ODS

__all__ = [
    "DEFAULT_ELEMENTAL_WEIGHTS",
    "ION_TEMPERATURE_MODES",
    "STAGE_NAMES",
    "InferenceStage",
    "KineticStateStages",
    "combine_ti_ratios",
    "kinetic_state_stages",
]

#: The stages in the order they are computed.
STAGE_NAMES = (
    "data",
    "thomson_fits",
    "romero_zeff",
    "impurity_amplitude",
    "ion_density",
    "zeff_profile",
    "pressure_closure",
)

#: ``n_C = n_O``: the elemental densities are in this ratio, only their common amplitude is inferred.
DEFAULT_ELEMENTAL_WEIGHTS: Mapping[str, float] = {"C": 1.0, "O": 1.0}

_STAGE_LABELS = {
    "data": "data only: p_eq, geometry, Thomson points",
    "thomson_fits": "Thomson fits: n_e, T_e, p_e",
    "romero_zeff": "Romero Z_eff (plasma transformer)",
    "impurity_amplitude": "effective impurity density n_I,eff (n_C = n_O)",
    "ion_density": "ion density n_i by quasi-neutrality and dilution",
    "zeff_profile": "Z_eff(rho) profile",
    "pressure_closure": "pressure closure p_e + p_i + p_imp vs p_eq (T_i = T_e; a check, not a fit)",
}

_CP = "core_profiles.profiles_1d[j]"


@dataclass(frozen=True)
class InferenceStage:
    """One stage of the chain: what it added, how it is known, and where it lands.

    ``fields`` holds the quantities this stage *adds* (arrays on the shared ``rho``
    grid, or scalars, or small mappings); ``adds`` names them.  ``provenance`` says
    how they were obtained; ``assumptions`` carries the flags that are also recorded
    in the stored ``*_fit.parameters`` records; ``lands`` maps a quantity to the IMAS
    path it occupies in the stored product (``j`` is the ``core_profiles`` slice),
    and omits what is derived rather than stored.  ``status`` is ``ok`` or
    ``unavailable``, with ``reason`` saying why.
    """

    name: str
    label: str
    adds: tuple[str, ...]
    fields: Mapping[str, Any]
    provenance: Mapping[str, Any]
    assumptions: Mapping[str, Any]
    lands: Mapping[str, str]
    status: str = "ok"
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def record(self) -> dict[str, Any]:
        """The stage without its arrays: what a log or a legend needs."""
        return {
            "name": self.name, "label": self.label, "status": self.status, "reason": self.reason,
            "adds": list(self.adds), "assumptions": dict(self.assumptions),
            "provenance": dict(self.provenance), "lands": dict(self.lands),
        }


@dataclass(frozen=True)
class KineticStateStages:
    """The ordered stages of one kinetic state, and the product the last one leaves behind."""

    time: float
    rho: np.ndarray | None
    stages: tuple[InferenceStage, ...]
    #: The ODS (a copy) as the chain leaves it -- ``ion[]``, ``zeff``, the assumed ``T_i``,
    #: the pressures and ``z_eff_resistive`` -- or ``None`` when the chain did not complete.
    product: ODS | None
    assumptions: Mapping[str, Any]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(stage.name for stage in self.stages)

    @property
    def complete(self) -> bool:
        return all(stage.ok for stage in self.stages)

    def stage(self, name: str) -> InferenceStage:
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise KeyError(f"no stage {name!r}; the stages are {', '.join(self.names)}")

    def upto(self, name: str) -> dict[str, Any]:
        """Everything known after stage ``name``: its fields and those of the ok stages before it.

        One animation frame.  The fields of an unavailable stage are not included.
        """
        known: dict[str, Any] = {}
        for stage in self.stages:
            if stage.ok:
                known.update(stage.fields)
            if stage.name == name:
                return known
        raise KeyError(f"no stage {name!r}; the stages are {', '.join(self.names)}")

    def records(self) -> list[dict[str, Any]]:
        return [stage.record() for stage in self.stages]


def _stage(name: str, *, fields: Mapping[str, Any], provenance: Mapping[str, Any],
           assumptions: Mapping[str, Any] | None = None, lands: Mapping[str, str] | None = None) -> InferenceStage:
    return InferenceStage(
        name=name, label=_STAGE_LABELS[name], adds=tuple(fields), fields=dict(fields),
        provenance=dict(provenance), assumptions=dict(assumptions or {}), lands=dict(lands or {}),
    )


def _unavailable(name: str, reason: str) -> InferenceStage:
    return InferenceStage(
        name=name, label=_STAGE_LABELS[name], adds=(), fields={}, provenance={}, assumptions={},
        lands={}, status="unavailable", reason=reason,
    )


class _Stop(Exception):
    """A stage that cannot be made; the message is the reason recorded."""


def _full(inside: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.full(inside.shape, np.nan)
    out[inside] = values
    return out


#: ``assumed`` T_i = T_e; ``inferred`` the pointwise pressure partition (NaN where it
#: carries no information); ``inferred_ratio`` one T_i/T_e from that partition.
ION_TEMPERATURE_MODES = ("assumed", "inferred", "inferred_ratio", "inferred_profile")

#: Highest polynomial order in rho of the ``inferred_profile`` T_i/T_e.
TI_PROFILE_MAX_ORDER = 2

#: A sloped (order >= 1) ratio profile whose first significant point lies beyond this
#: rho is flagged ``axis_extrapolated``: its axis value is set by the polynomial alone.
TI_PROFILE_AXIS_GAP = 0.2


def combine_ti_ratios(ratios: Sequence[float], sigmas: Sequence[float]) -> dict[str, float]:
    """One ``T_i/T_e`` for a time window from per-slice estimates: inverse-variance mean.

    The slices of one window are treated as independent repeats; when they scatter
    more than their sigmas allow (``chi2/dof > 1``) the error of the mean is scaled
    up by ``sqrt(chi2/dof)``, so a drifting ratio is not reported with a false
    precision.  Non-finite entries are ignored.

    Raises:
        ValueError: Fewer than two usable slices.
    """
    r = np.asarray(ratios, dtype=float)
    s = np.asarray(sigmas, dtype=float)
    ok = np.isfinite(r) & np.isfinite(s) & (s > 0)
    if ok.sum() < 2:
        raise ValueError("fewer than two slices with a T_i/T_e and its sigma")
    w = 1.0 / s[ok] ** 2
    mean = float(np.sum(w * r[ok]) / np.sum(w))
    formal = float(1.0 / np.sqrt(np.sum(w)))
    chi2 = float(np.sum(w * (r[ok] - mean) ** 2) / (ok.sum() - 1))
    return {"ratio": mean, "sigma": formal * float(np.sqrt(max(1.0, chi2))), "sigma_formal": formal,
            "chi2_per_dof": chi2, "slices": int(ok.sum()), "std": float(np.std(r[ok], ddof=1))}


def _ti_ratio_profile(
    rho: np.ndarray, ti_te: np.ndarray, sigma_t_i: np.ndarray, t_e: np.ndarray, rho_out: np.ndarray,
    *, n_independent: int = 5, max_order: int = TI_PROFILE_MAX_ORDER,
) -> dict[str, Any]:
    """A smooth ``T_i/T_e(rho)`` from the significant points: a polynomial in ``rho`` of order 0..max.

    Weighted least squares with weights ``1/sigma^2``.  The points of one profile
    are smooth fits of ``n_independent`` Thomson channels, so chi-square and the
    covariance are rescaled to that many independent points (the covariance by
    ``N/n_independent``, then by ``chi2/dof`` when it exceeds one).  The order is
    the lowest BIC, ``chi2_eff + k ln(n_independent)``; an order whose ratio is not
    positive everywhere on ``rho_out``, or that has no degree of freedom left, is
    rejected.  Returns the ratio and its one-sigma on ``rho_out``, the chosen
    order, coefficients and every order's BIC; outside the span of the points the
    values are an extrapolation (``extrapolated`` mask).
    """
    rho, ratio = np.asarray(rho, dtype=float), np.asarray(ti_te, dtype=float)
    sigma = np.asarray(sigma_t_i, dtype=float) / np.asarray(t_e, dtype=float)
    ok = np.isfinite(rho) & np.isfinite(ratio) & np.isfinite(sigma) & (sigma > 0)
    if ok.sum() < 3:
        raise ValueError("fewer than three significant T_i/T_e points: no ratio profile can be fitted")
    x, y, w = rho[ok], ratio[ok], 1.0 / sigma[ok] ** 2
    n, n_eff = int(ok.sum()), max(1, int(n_independent))
    out = np.asarray(rho_out, dtype=float)
    fits: dict[int, dict[str, Any]] = {}
    for order in range(max_order + 1):
        k = order + 1
        if k >= n_eff:
            break
        design = np.vander(x, k, increasing=True)
        normal = design.T @ (w[:, None] * design)
        try:
            cov = np.linalg.inv(normal)
        except np.linalg.LinAlgError:
            continue
        coef = cov @ (design.T @ (w * y))
        chi2 = float(np.sum(w * (y - design @ coef) ** 2))
        chi2_eff = chi2 * n_eff / n
        scale = (n / n_eff) * max(1.0, chi2_eff / max(1, n_eff - k))
        basis = np.vander(out, k, increasing=True)
        values = basis @ coef
        if not np.all(values[np.isfinite(values)] > 0):
            continue
        band = np.sqrt(np.einsum("ij,jk,ik->i", basis, cov * scale, basis))
        fits[order] = {"coef": coef, "ratio": values, "sigma": band,
                       "bic": chi2_eff + k * float(np.log(n_eff)), "chi2_eff": chi2_eff}
    if not fits:
        raise ValueError("no polynomial T_i/T_e stays positive over the profile")
    best = min(fits, key=lambda o: fits[o]["bic"])
    span = (float(x.min()), float(x.max()))
    warnings = []
    if best >= 1 and span[0] > TI_PROFILE_AXIS_GAP:
        # a sloped fit carried over a wide gap is the polynomial speaking, not the data
        warnings.append(f"axis_extrapolated: order-{best} fit carried over rho < {span[0]:.2f} with no point; "
                        "do not quote the axis value")
    return {"order": int(best), "warnings": warnings, "coefficients": [float(c) for c in fits[best]["coef"]],
            "ratio": fits[best]["ratio"], "sigma": fits[best]["sigma"],
            "bic": {int(o): float(f["bic"]) for o, f in fits.items()},
            "rho_span": span, "extrapolated": (out < span[0]) | (out > span[1]),
            "points": n, "n_independent": n_eff}


#: Elementary charge [J/eV]: p [Pa] = e n [m^-3] T [eV].
_QE = 1.602176634e-19


def _ti_ratio_estimate(
    ti_te: np.ndarray, sigma_t_i: np.ndarray, t_e: np.ndarray, *, n_independent: int = 5,
) -> dict[str, float]:
    """Inverse-variance mean of the significant ``T_i/T_e`` points, with a conservative sigma.

    The points of one profile are strongly correlated (smooth fits of a few Thomson
    channels), so treating them as independent understates the error.  The formal
    error is rescaled from ``N`` points to ``n_independent``; the reported
    ``sigma`` is the larger of that formal error (scaled by ``sqrt(chi2/dof)`` when
    the points scatter more than their sigmas allow) and the radial spread of the
    points, half the 16-84 percentile range, divided by ``sqrt(n_independent)``
    (the Thomson channel count).  A radially varying ``T_i/T_e`` therefore widens it.
    """
    ratio = np.asarray(ti_te, dtype=float)
    sigma = np.asarray(sigma_t_i, dtype=float) / np.asarray(t_e, dtype=float)
    ok = np.isfinite(ratio) & np.isfinite(sigma) & (sigma > 0)
    if ok.sum() < 3:
        raise ValueError("fewer than three significant T_i/T_e points: no ratio can be estimated")
    w = 1.0 / sigma[ok] ** 2
    mean = float(np.sum(w * ratio[ok]) / np.sum(w))
    formal = float(1.0 / np.sqrt(np.sum(w)))
    chi2 = float(np.sum(w * (ratio[ok] - mean) ** 2) / (ok.sum() - 1))
    q16, q50, q84 = (float(x) for x in np.percentile(ratio[ok], [16, 50, 84]))
    spread = 0.5 * (q84 - q16) / float(np.sqrt(max(1, n_independent)))
    # N grid points carry the information of n_independent channels, as in _ti_ratio_profile
    formal *= float(np.sqrt(ok.sum() / max(1, n_independent)))
    return {"ratio": mean, "sigma_formal": formal,
            "sigma": max(formal * float(np.sqrt(max(1.0, chi2))), spread), "n_independent": int(n_independent),
            "chi2_per_dof": chi2, "points": int(ok.sum()), "median": q50, "p16": q16, "p84": q84}


def kinetic_state_stages(
    ods: ODS,
    *,
    time: float,
    equilibrium_lineage: str | None = None,
    elemental_weights: Mapping[str, float] | None = None,
    conductivity_model: str = "redl",
    ln_lambda: Any = "sauter",
    bounds: tuple[float, float] = (1.0, 8.0),
    i_ni_a: float = 0.0,
    ionization: str = "transient",
    plasma_age_s: float | None = None,
    equilibrium_tolerance_s: float = 5e-4,
    max_excluded_current: float = 0.05,
    inductive_max: float = 1.0,
    tables: Mapping[str, Any] | None = None,
    cache_dir: str | None = None,
    ion_temperature: str = "assumed",
    ti_floor: float = 0.17,
    ti_window_s: float = 1e-3,
) -> KineticStateStages:
    """Infer the kinetic state of the slice at ``time`` in the seven named stages.

    ``ods`` carries a multi-slice ``equilibrium`` (the Romero balance needs the
    current-carrying window, not one slice), the ``core_profiles`` slice at ``time``
    (the Thomson fits) and, optionally, the ``thomson_scattering`` channels.  It is
    not modified.  ``equilibrium_lineage`` is the caller's statement of where the
    equilibrium came from, recorded in the stored provenance (``"magnetics"`` when
    omitted with ``T_i`` assumed).  The inferred ``ion_temperature`` modes partition
    ``T_i`` out of that equilibrium's pressure, so they require it to be stated and
    accept only ``"magnetics"``: an electron-kinetic EFIT was fitted with an assumed
    ``T_i/T_e`` and would return it.  Nothing in an ODS records the lineage,
    so it cannot be read here -- the caller must know where it loaded it from.

    ``elemental_weights`` is the fixed elemental ratio (default ``C : O = 1 : 1``);
    ``ionization`` is ``"transient"`` (needs ``plasma_age_s``) or ``"coronal"``;
    ``conductivity_model``, ``ln_lambda``, ``bounds``, ``i_ni_a`` are Lane Z's
    estimator settings (``i_ni_a = 0`` asserts a purely Ohmic window).

    ``ion_temperature`` is ``"assumed"`` (``T_i = T_e``; the pressure closure is then a
    consistency check), ``"inferred_ratio"`` (the three closures with one
    ``T_i/T_e``: the inverse-variance mean of the significant points of the partition,
    ``T_i = r T_e`` over the whole profile, the pointwise result kept as a diagnostic)
    or ``"inferred"`` (the three closures together: the Romero
    scalar closes the impurity amplitude, and ``T_i`` is partitioned out of the
    magnetic-EFIT pressure with that composition's ``n_i``, as
    :func:`vaft.process.kinetic_closure.infer_ti_pressure_partition`; ``p_thermal``
    then equals ``p_eq`` where ``T_i`` is defined by construction).  ``ti_floor`` and
    ``ti_window_s`` are the pressure sigma floor and neighbour window of that inference.
    """
    if ion_temperature not in ION_TEMPERATURE_MODES:
        raise ValueError(f"ion_temperature must be one of {ION_TEMPERATURE_MODES}, got {ion_temperature!r}")
    pointwise = ion_temperature in ("inferred", "inferred_ratio", "inferred_profile")
    if equilibrium_lineage is None:
        if pointwise:
            raise ValueError(
                f"ion_temperature={ion_temperature!r} partitions T_i out of the equilibrium pressure, so "
                "equilibrium_lineage must be stated (only 'magnetics' is accepted)"
            )
        equilibrium_lineage = "magnetics"
    from vaft.omas.vest_upstream import (
        PRESSURE_CLOSURE_RECORD,
        _append_record_fields,
        _equilibrium_pressure_on,
        _romero_window,
        _thomson_relative_errors,
        _time_array_ms,
        _volume_weights_on_grid,
    )
    from vaft.ods_access import path_value
    from vaft.process.impurity import populate_radial_impurity_profiles, resolve_radial_composition
    from vaft.process.kinetic_closure import infer_kinetic_closure
    from vaft.process.resistive_zeff import Smoothing
    from vaft.process.zeff_projection import project_zeff_profile_to_resistive_scalar
    from vaft.validation import kinetic_state as ks

    time = float(time)
    weights = dict(DEFAULT_ELEMENTAL_WEIGHTS if elemental_weights is None else elemental_weights)
    nc_no = (
        math.isclose(weights.get("C", math.nan), weights.get("O", math.nan))
        and set(weights) == {"C", "O"}
    )
    assumptions: dict[str, Any] = {"assume_ti_eq_te": int(ion_temperature == "assumed"), "assume_ohmic": int(float(i_ni_a) == 0.0)}
    if nc_no:
        assumptions.update(assume_nc_eq_no=1, n_c_over_n_o=1.0)
    else:
        assumptions.update(elemental_weights=",".join(f"{k}:{v:g}" for k, v in sorted(weights.items())))

    ctx: dict[str, Any] = {}
    stages: list[InferenceStage] = []
    had_wall = "wall" in ods.keys()

    # -- 1. data -------------------------------------------------------------------------------
    def data() -> InferenceStage:
        try:
            eq_index, offset = ks.slice_at_time(ods, time, tolerance_s=equilibrium_tolerance_s)
        except LookupError as error:
            raise _Stop(str(error))
        coordinate = ks.rho_tor_norm_of(ods, eq_index)
        electron = ks.core_profiles_electron_pressure(ods, time_s=time, tolerance_s=equilibrium_tolerance_s)
        if not electron["available"]:
            raise _Stop(electron["reason"])
        placed = ks.core_profiles_grid_on_slice(coordinate, electron)
        if not placed["available"]:
            raise _Stop(placed["reason"])
        psi = np.asarray(placed["psi_norm"], dtype=float)
        inside = psi <= 1.0
        if inside.sum() < 3:
            raise _Stop("fewer than three profile points inside the last closed flux surface")
        p_eq = _equilibrium_pressure_on(ods, eq_index, psi[inside])
        if p_eq is None:
            raise _Stop("the equilibrium pressure cannot be sampled on the profile grid")
        ctx.update(eq_index=eq_index, cp_index=int(electron["index"]), coordinate=coordinate, inside=inside,
                   rho=np.asarray(electron["rho_tor_norm"], dtype=float), psi_norm=psi, electron=electron)
        ctx["p_eq"] = _full(inside, p_eq)
        fields: dict[str, Any] = {
            "rho": ctx["rho"], "psi_norm": psi, "inside": inside, "p_eq": ctx["p_eq"],
            "equilibrium_time_s": float(ods[f"equilibrium.time_slice.{eq_index}.time"])
            if path_value(ods, f"equilibrium.time_slice.{eq_index}.time", None) is not None
            else time - offset,
        }
        thomson: dict[str, np.ndarray] = {}
        note = ""
        if path_value(ods, "thomson_scattering.channel", None) is not None:
            from vaft.validation.equilibrium import thomson_pressure_samples

            samples = thomson_pressure_samples(ods, eq_index, ods, tolerance=equilibrium_tolerance_s)
            if samples.get("available") and samples.get("channels"):
                order = np.argsort(coordinate["psi_norm"])
                rows = samples["channels"]
                psi_ts = np.array([row["psi_norm"] for row in rows], dtype=float)
                thomson = {
                    "r": np.array([row["r"] for row in rows], dtype=float),
                    "z": np.array([row["z"] for row in rows], dtype=float),
                    "psi_norm": psi_ts,
                    "rho": np.interp(psi_ts, coordinate["psi_norm"][order], coordinate["rho_tor_norm"][order]),
                    "n_e": np.array([row["n_e"] for row in rows], dtype=float),
                    "t_e": np.array([row["t_e"] for row in rows], dtype=float),
                    "n_e_error": np.array([row["n_e_error"] for row in rows], dtype=float),
                    "t_e_error": np.array([row["t_e_error"] for row in rows], dtype=float),
                    "p_e": np.array([row["p_e"] for row in rows], dtype=float),
                }
            else:
                note = samples.get("reason", "no Thomson channel could be placed on the equilibrium")
        else:
            note = "no thomson_scattering channels in the ODS"
        fields["thomson"] = thomson
        ctx["thomson"] = thomson
        return _stage(
            "data", fields=fields,
            provenance={"origin": "measured", "equilibrium_lineage": equilibrium_lineage,
                        "equilibrium_occurrence": 0, "equilibrium_offset_s": float(offset),
                        "rho_tor_norm_source": coordinate.get("source", ""), "thomson_note": note},
        )

    # -- 2. thomson_fits -----------------------------------------------------------------------
    def thomson_fits() -> InferenceStage:
        electron, inside = ctx["electron"], ctx["inside"]
        ne = np.asarray(electron["n_e"], dtype=float)
        te = np.asarray(electron["t_e"], dtype=float)
        rel_n, rel_t = _thomson_relative_errors(ods, time)
        ctx.update(ne=ne, te=te)
        record = path_value(ods, f"core_profiles.profiles_1d.{ctx['cp_index']}.electrons.temperature_fit.parameters", None)
        return _stage(
            "thomson_fits",
            fields={"n_e": ne, "T_e": te, "p_e": np.asarray(electron["p_e"], dtype=float),
                    "sigma_n_e_rel": rel_n, "sigma_t_e_rel": rel_t},
            provenance={"origin": "measured", "method": "thomson_fit", "record": None if record is None else str(record),
                        "density_leaf": electron.get("density_leaf", "")},
            lands={"n_e": f"{_CP}.electrons.density", "T_e": f"{_CP}.electrons.temperature"},
        )

    # -- 3. romero_zeff ------------------------------------------------------------------------
    def romero() -> InferenceStage:
        smoothing = Smoothing("none")
        window = _romero_window(
            ods, model=conductivity_model, ln_lambda=ln_lambda, bounds=bounds, i_ni_a=i_ni_a, smoothing=smoothing,
            time_range=None, max_excluded_current=max_excluded_current, inductive_max=inductive_max,
            time_tolerance_s=equilibrium_tolerance_s,
        )
        if window["result"] is None or window["inference"] is None:
            raise _Stop(window["error"] or "no Romero estimate")
        inference = window["inference"]
        mine = [s for s in window["usable"] if int(s.source["cp_index"]) == ctx["cp_index"]]
        if not mine:
            reasons = "; ".join(r for t, r in window["skipped"] if abs(t - time) < 1e-3) or "filtered out"
            raise _Stop(f"the slice at t = {time:g} s is not a usable flux-surface state ({reasons})")
        ctx["state"] = mine[0]
        estimate = inference.estimate
        diagnostic = window["diagnostic"]
        fields = {
            "z_eff_resistive": inference.zeff if inference.status == "ok" else None,
            "z_eff_resistive_sigma": estimate.get("uncertainty"),
            "estimate_status": inference.status,
            "estimate_value": inference.zeff,
            "window_s": (float(window["window"][0]), float(window["window"][1])),
            "n_states": len(window["usable"]),
            "per_slice_diagnostic": {
                "note": "each solved alone: the full noise of a differentiated V_R; a diagnostic, not the estimate",
                "time_s": [float(t) for t in diagnostic["time"]],
                "zeff": [None if not np.isfinite(z) else float(z) for z in diagnostic["zeff"]],
                "status": list(diagnostic["status"]),
            },
        }
        provenance = {
            "origin": "inferred", "model": "romero_transformer", "scope": "window",
            "conductivity": conductivity_model, "ln_lambda": str(ln_lambda), "i_ni_a": float(i_ni_a),
            "smoothing": smoothing.describe(), "bounds": list(bounds),
            "equilibrium_lineage": equilibrium_lineage, "role": "constraint",
        }
        if inference.status != "ok":
            stage = _unavailable("romero_zeff", f"the window estimate is {inference.status}"
                                 + (f" ({inference.reason})" if inference.reason else "")
                                 + ": no constraint for the impurity amplitude")
            return InferenceStage(**{**stage.__dict__, "fields": fields, "adds": tuple(fields),
                                     "provenance": provenance})
        ctx["z_res"] = float(inference.zeff)
        ctx["romero_provenance"] = provenance
        ctx["romero_fields"] = fields
        return _stage("romero_zeff", fields=fields, provenance=provenance,
                      assumptions={"assume_ohmic": assumptions["assume_ohmic"]},
                      lands={"z_eff_resistive": "core_profiles.global_quantities.z_eff_resistive[j]"})

    # -- 4. impurity_amplitude -----------------------------------------------------------------
    def projection_for(state):
        rho_state = None

        def projection(z_on_cp_grid: np.ndarray) -> float:
            nonlocal rho_state
            coordinate = ctx["coordinate"]
            if rho_state is None:
                order = np.argsort(coordinate["psi_norm"])
                rho_state = np.interp(state.psi_norm, coordinate["psi_norm"][order], coordinate["rho_tor_norm"][order])
            z = np.asarray(z_on_cp_grid, dtype=float)
            good = np.isfinite(z)
            if good.sum() < 2:
                return math.nan
            order = np.argsort(ctx["rho"][good])
            z_state = np.interp(rho_state, ctx["rho"][good][order], z[good][order])
            return float(project_zeff_profile_to_resistive_scalar(
                state, z_state, model=conductivity_model, ln_lambda=ln_lambda, bounds=(1.0, max(bounds[1], 12.0)),
            ).zeff_equivalent)

        return projection

    def amplitude() -> InferenceStage:
        projection = projection_for(ctx["state"])
        try:
            radial = resolve_radial_composition(
                ctx["te"], ctx["ne"], ctx["rho"], weights, normalization="resistive_closure",
                ionization=ionization, plasma_age_s=plasma_age_s, projection=projection,
                resistive_target=ctx["z_res"], tables=tables, cache_dir=cache_dir, time=time,
            )
        except ValueError as error:
            raise _Stop(f"no impurity amplitude closes the Romero constraint: {error}")
        ctx.update(radial=radial, projection=projection)
        ne = ctx["ne"]
        elements = list(radial.elements)
        n_total = {e: ne * radial.elemental_fractions[:, k] for k, e in enumerate(elements)}
        charged = {e: n_total[e] * (1.0 - radial.charge_state_fractions[k][:, 0]) for k, e in enumerate(elements)}
        ctx.update(n_total=n_total, n_charged=charged)
        ratio = None
        if {"C", "O"} <= set(elements):
            first = int(np.flatnonzero(np.isfinite(n_total["O"]) & (n_total["O"] > 0))[0])
            ratio = float(n_total["C"][first] / n_total["O"][first])
        n_i_eff = sum(charged.values())
        return _stage(
            "impurity_amplitude",
            fields={"scale": float(radial.scale), "n_I_eff": n_i_eff,
                    "n_I_eff_total": sum(n_total.values()), "n_by_element": n_total,
                    "n_c_over_n_o": ratio, "Z_I_eff": np.asarray(radial.effective_charge, dtype=float)},
            provenance={"origin": "inferred", "method": f"openadas_{ionization}",
                        "normalization": "resistive_closure", "resistive_target": ctx["z_res"],
                        "constraint": "resistive_projection", "scope": "slice",
                        "plasma_age_s": plasma_age_s, "role": "inferred_from_constraint"},
            assumptions={k: v for k, v in assumptions.items() if k.startswith(("assume_nc", "n_c", "elemental"))},
        )

    # -- 5. ion_density ------------------------------------------------------------------------
    def ion_density() -> InferenceStage:
        radial, ne = ctx["radial"], ctx["ne"]
        n_main = ne * np.asarray(radial.main_ion_fraction, dtype=float)
        ctx["n_main"] = n_main
        states = {
            e: [ctx["n_total"][e] * radial.charge_state_fractions[k][:, q] for q in range(1, radial.charge_state_fractions[k].shape[1])]
            for k, e in enumerate(radial.elements)
        }
        n_i_total = n_main + sum(ctx["n_charged"].values())
        return _stage(
            "ion_density",
            fields={"n_main": n_main, "n_state": states, "n_i_total": n_i_total,
                    "dilution": np.asarray(radial.dilution_fraction, dtype=float)},
            provenance={"origin": "inferred", "method": "quasi_neutrality", "main_ion": radial.main_ion,
                        "ionization": ionization},
            assumptions={k: v for k, v in assumptions.items() if k.startswith(("assume_nc", "n_c"))},
            lands={"n_main": f"{_CP}.ion[0].density", "n_impurity": f"{_CP}.ion[k].density",
                   "n_state": f"{_CP}.ion[k].state[q].density"},
        )

    # -- 6. zeff_profile -----------------------------------------------------------------------
    def zeff_profile() -> InferenceStage:
        radial = ctx["radial"]
        z = np.asarray(radial.zeff, dtype=float)
        try:
            equivalent = float(ctx["projection"](z))
        except ValueError as error:
            raise _Stop(f"the Z_eff profile has no resistive projection: {error}")
        return _stage(
            "zeff_profile",
            fields={"z_eff": z, "z_eff_resistive_equivalent": equivalent,
                    "closure_error": equivalent - ctx["z_res"]},
            provenance={"origin": "inferred", "method": f"openadas_{ionization}",
                        "normalization": "resistive_closure", "resistive_target": ctx["z_res"]},
            assumptions={k: v for k, v in assumptions.items() if k.startswith(("assume_nc", "n_c"))},
            lands={"z_eff": f"{_CP}.zeff"},
        )

    # -- 7. pressure_closure -------------------------------------------------------------------
    def partition_ti() -> tuple[np.ndarray, str]:
        """``T_i`` from the magnetic-EFIT pressure and the Romero-closed ``n_i``; and its record."""
        from vaft.machine_mapping.core_profiles import inferred_ti_text
        from vaft.omas.vest_upstream import usable_equilibrium_times_ms
        from vaft.process.kinetic_closure import infer_ti_pressure_partition

        if equilibrium_lineage != "magnetics":
            raise _Stop(f"the T_i inference accepts only the magnetics lineage, got {equilibrium_lineage!r}")
        inside, ne, te, eq = ctx["inside"], ctx["ne"], ctx["te"], ctx["eq_index"]
        fraction = (ctx["n_main"] + sum(ctx["n_charged"].values())) / ne
        psi = ctx["psi_norm"][inside]
        usable = usable_equilibrium_times_ms(ods)
        others = []
        eq_ms = _time_array_ms(ods, "equilibrium.time")
        for k, t_ms in enumerate([] if eq_ms is None else eq_ms):
            if k == eq or not abs(float(t_ms) * 1e-3 - time) <= ti_window_s + 1e-9:
                continue
            if usable is not None and not np.any(np.abs(usable - t_ms) < 1e-6):
                continue
            pressure_k = _equilibrium_pressure_on(ods, k, psi)
            if pressure_k is not None:
                others.append(pressure_k)
        sigma_p, basis = ks.pressure_sigma(ctx["p_eq"][inside], others, floor=ti_floor)
        rel_n, rel_t = _thomson_relative_errors(ods, time)
        result = infer_ti_pressure_partition(
            ctx["p_eq"][inside], ne[inside], te[inside],
            composition={"ion_density_per_electron": fraction[inside],
                         "composition_source": "resistive-closed impurity amplitude (stage 4)"},
            sigma_p_eq=sigma_p, sigma_n_e=rel_n * ne[inside], sigma_t_e=rel_t * te[inside],
            equilibrium_lineage=equilibrium_lineage,
        )
        if not result["eligible"]:
            raise _Stop(result["reason"])
        valid = np.isfinite(result["t_i"])
        if not valid.any():
            raise _Stop("no point has a positive, significant p_i")
        t_i = _full(inside, result["t_i"])
        record = "; ".join([
            inferred_ti_text("equilibrium_pressure_partition"),
            f"sigma_p_eq_basis={basis}", f"valid_points={int(valid.sum())}/{valid.size}",
            "not measured",
        ])
        ctx["ti_ratio"] = _full(inside, result["ti_te"])
        ctx["ti_result"] = (result, inside)
        ctx["ti_sigma"] = _full(inside, result["sigma_t_i"])
        return t_i, record

    def pressure_closure() -> InferenceStage:
        radial, ne, te, cp = ctx["radial"], ctx["ne"], ctx["te"], ctx["cp_index"]
        product = populate_radial_impurity_profiles(ods, radial, time=time)
        base = f"core_profiles.profiles_1d.{cp}"
        lineage = {"equilibrium_lineage": equilibrium_lineage, "equilibrium_occurrence": 0}
        if pointwise:
            t_i, ti_record = partition_ti()
            if ion_temperature == "inferred_profile":
                result, inside = ctx["ti_result"]
                n_ts = int(np.size(ctx["thomson"]["rho"])) if ctx.get("thomson") else 5
                try:
                    fit = _ti_ratio_profile(ctx["rho"][inside], result["ti_te"], result["sigma_t_i"],
                                            te[inside], ctx["rho"], n_independent=n_ts)
                except ValueError as error:
                    raise _Stop(str(error))
                ctx["ti_pointwise"], ctx["ti_profile"] = t_i, fit
                t_i = fit["ratio"] * te
                ti_record = "; ".join([
                    "origin=inferred; method=equilibrium_pressure_partition_profile",
                    f"ti_te_order={fit['order']}",
                    "ti_te_coefficients=" + ",".join(f"{c:.4g}" for c in fit["coefficients"]),
                    f"rho_valid={fit['rho_span'][0]:.4g}:{fit['rho_span'][1]:.4g}",
                    "extrapolated_outside_rho_valid=1",
                    *(["axis_extrapolated=1"] if fit["warnings"] else []), "not measured",
                ])
            if ion_temperature == "inferred_ratio":
                result, inside = ctx["ti_result"]
                try:
                    n_ts = int(np.size(ctx["thomson"]["rho"])) if ctx.get("thomson") else 5
                    estimate = _ti_ratio_estimate(result["ti_te"], result["sigma_t_i"], te[inside],
                                                  n_independent=n_ts)
                except ValueError as error:
                    raise _Stop(str(error))
                ctx["ti_pointwise"], ctx["ti_estimate"] = t_i, estimate
                t_i = estimate["ratio"] * te
                ti_record = "; ".join([
                    "origin=inferred; method=equilibrium_pressure_partition_scalar",
                    f"ti_te_ratio={estimate['ratio']:.4g}; sigma_ti_te_ratio={estimate['sigma']:.3g}",
                    f"ratio_points={estimate['points']}", "not measured",
                ])
        else:
            t_i, ti_record = te.copy(), "ti_te_ratio=1; status=assumed; source=assumption; assume_ti_eq_te=1"
        for k in range(len(product[f"{base}.ion"])):
            product[f"{base}.ion.{k}.temperature"] = t_i
            product[f"{base}.ion.{k}.temperature_fit.parameters"] = ti_record
        product[f"{base}.t_i_average"] = t_i
        _append_record_fields(product, f"{base}.ion.0.temperature_fit.parameters", lineage)
        # t_i_average says what it is too (same record, lineage included) -- a reader of
        # the average must not have to know it was copied from the main ion
        product[f"{base}.t_i_average_fit.parameters"] = str(product[f"{base}.ion.0.temperature_fit.parameters"])
        romero = ctx["romero_fields"]
        closure_fields = {
            "resistive_target": ctx["z_res"], "resistive_method": "romero",
            **({"resistive_sigma": float(romero["z_eff_resistive_sigma"])}
               if romero.get("z_eff_resistive_sigma") is not None else {}),
            "resistive_window": f"{romero['window_s'][0]:.6g}:{romero['window_s'][1]:.6g}",
            "resistive_status": romero["estimate_status"], "resistive_n_states": int(romero["n_states"]),
        }
        zeff_extra = {**closure_fields, "ionization": ionization,
                      **{k: v for k, v in assumptions.items() if k.startswith(("assume_nc", "n_c"))}}
        _append_record_fields(product, f"{base}.zeff_fit.parameters", zeff_extra)
        # the record that states the normalization carries the closure it was normalised to
        for k in range(len(product[f"{base}.ion"])):
            _append_record_fields(product, f"{base}.ion.{k}.density_fit.parameters", closure_fields)
        closure = infer_kinetic_closure(product, time=time, dilution="species", ti_te_ratio=1.0)
        pressure = closure.pressure
        if pointwise:
            from vaft.process.kinetic_closure import assemble_pressure

            pressure = assemble_pressure(
                closure.n_e, closure.T_e, closure.ion_densities,
                {label: t_i for label in closure.ion_densities},
            )
        product[f"{base}.electrons.pressure"] = pressure.p_e
        product[f"{base}.electrons.pressure_thermal"] = pressure.p_e
        product[f"{base}.pressure_ion_total"] = pressure.p_i_thermal
        product[f"{base}.pressure_thermal"] = pressure.p_thermal
        # per species, so p_imp is read rather than rebuilt: p_k = e n_k T_i for every ion
        for k in range(len(product[f"{base}.ion"])):
            n_k = np.asarray(product[f"{base}.ion.{k}.density"], dtype=float)
            p_k = _QE * n_k * np.asarray(t_i, dtype=float)
            product[f"{base}.ion.{k}.pressure"] = p_k
            product[f"{base}.ion.{k}.pressure_thermal"] = p_k
        _append_record_fields(product, f"{base}.ion.0.density_fit.parameters", PRESSURE_CLOSURE_RECORD)
        if not had_wall and "wall" in product.keys():
            # a read somewhere in the chain materialised an empty, NaN wall (OMAS creates
            # what it is asked for); the input had none, so neither does the product
            del product["wall"]
        z_stored = np.full(np.atleast_1d(np.asarray(product["core_profiles.time"], dtype=float)).shape, np.nan)
        z_stored[cp] = ctx["z_res"]
        product["core_profiles.global_quantities.z_eff_resistive"] = z_stored
        ctx["product"] = product

        species = pressure.p_i_species
        p_main = np.asarray(species.get(radial.main_ion + "+", 0.0), dtype=float)
        p_imp = np.asarray(pressure.p_i_thermal, dtype=float) - p_main
        p_eq = ctx["p_eq"]
        residual = p_eq - pressure.p_thermal
        with np.errstate(invalid="ignore", divide="ignore"):
            relative = residual / p_eq
        volume = _volume_weights_on_grid(ods, ctx["eq_index"], ctx["rho"])
        weights_name = "equilibrium dV" if volume is not None else "rho drho (no equilibrium volume)"
        if volume is None:
            edges = np.concatenate(([0.0], 0.5 * (ctx["rho"][1:] + ctx["rho"][:-1]), [ctx["rho"][-1]]))
            volume = np.diff(edges**2)
        ok = np.isfinite(p_eq) & np.isfinite(pressure.p_thermal) & (volume > 0)
        summary = {
            "weights": weights_name,
            "volume_ratio": float(np.sum((pressure.p_thermal * volume)[ok]) / np.sum((p_eq * volume)[ok])) if ok.any() else None,
            "median_relative_residual": float(np.nanmedian(relative[ok])) if ok.any() else None,
        }
        return _stage(
            "pressure_closure",
            fields={"T_i": t_i, "T_i_over_T_e": t_i / te, "T_i_pointwise": ctx.get("ti_pointwise"),
                    "ti_ratio_estimate": ctx.get("ti_estimate"), "ti_ratio_profile": ctx.get("ti_profile"),
                    "T_i_pointwise_sigma": ctx.get("ti_sigma"),
                    "p_e": np.asarray(pressure.p_e, dtype=float), "p_i": p_main,
                    "p_imp": p_imp, "p_thermal": np.asarray(pressure.p_thermal, dtype=float), "p_eq": p_eq,
                    "residual": residual, "relative_residual": relative, "closure_summary": summary},
            provenance=(
                ({"origin": "inferred", "method": "equilibrium_pressure_partition_profile", "dilution": "species",
                  "role": "closure_profile",
                  "note": "a low-order T_i/T_e(rho) fitted to the significant points of the pressure partition "
                          "(order by BIC); continuous over the profile, extrapolated outside rho_valid; the "
                          "residual to p_eq is reported"}
                 if ion_temperature == "inferred_profile" else
                 {"origin": "inferred", "method": "equilibrium_pressure_partition_scalar", "dilution": "species",
                  "role": "closure_scalar",
                  "note": "one T_i/T_e from the significant points of the pressure partition (Romero-closed "
                          "n_i); T_i = r T_e over the whole profile, so the profile shape is assumed and the "
                          "residual to p_eq is reported"}
                 if ion_temperature == "inferred_ratio" else
                 {"origin": "inferred", "method": "equilibrium_pressure_partition", "dilution": "species",
                  "role": "closure",
                  "note": "T_i is partitioned out of p_eq with the Romero-closed n_i, so p_thermal equals p_eq "
                          "where T_i is defined; the residual is an identity, not an agreement test"})
                if pointwise else
                {"origin": "assumed", "method": "infer_kinetic_closure", "dilution": "species",
                 "role": "consistency_check",
                 "note": "T_i = T_e and the impurity amplitude are fixed before this stage; the residual "
                         "is reported, never minimised"}),
            assumptions={"assume_ti_eq_te": int(ion_temperature == "assumed")}
            | ({"assume_ti_profile_shape_eq_te": 1} if ion_temperature == "inferred_ratio" else {})
            | ({"ti_extrapolated_outside_rho_valid": 1} if ion_temperature == "inferred_profile" else {}),
            lands={"p_e": f"{_CP}.electrons.pressure", "p_i_total": f"{_CP}.pressure_ion_total",
                   "p_thermal": f"{_CP}.pressure_thermal", "T_i": f"{_CP}.t_i_average"},
        )

    builders = (data, thomson_fits, romero, amplitude, ion_density, zeff_profile, pressure_closure)
    stopped: str | None = None
    for name, build in zip(STAGE_NAMES, builders):
        if stopped is not None:
            stages.append(_unavailable(name, stopped))
            continue
        try:
            stage = build()
        except _Stop as stop:
            stage = _unavailable(name, str(stop))
        except (ValueError, KeyError, IndexError) as error:
            stage = _unavailable(name, f"{type(error).__name__}: {error}"[:300])
        stages.append(stage)
        if not stage.ok:
            stopped = f"stage {name!r} is unavailable: {stage.reason}"

    complete = all(stage.ok for stage in stages)
    return KineticStateStages(
        time=time, rho=ctx.get("rho"), stages=tuple(stages),
        product=ctx.get("product") if complete else None, assumptions=assumptions,
    )
