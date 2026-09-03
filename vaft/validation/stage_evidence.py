"""Per-stage validation evidence for the FileDB pipeline (issues #139, #338).

What a production stage's data product *shows* -- the preconditions that say
whether it is empty for a legitimate reason, and the quantitative metrics that
back its validation figures.  Every function here composes a domain provider
(:mod:`vaft.omas.efit_quality`, :mod:`vaft.omas.vacuum_magnetics`,
:mod:`vaft.validation.magnetics`) into a manifest block; none renders, hashes
or persists anything.

That is the whole reason this module is separate from
:mod:`vaft.database.production_qa` (#253 §11).  Which figures a stage *owes*,
what they are called on disk and how they are written are questions about the
artifact, and belong to the database layer.  Whether the stage's product is
credible enough to be worth a figure is a question about the science, and
belongs here.  The dependency runs one way: the artifact executor imports this
module for :data:`STAGE_PRECONDITIONS` and :data:`STAGE_METRICS`; nothing here
knows a FileDB path exists.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# Iterating an OMAS AOS yields its integer keys rather than its entries, so
# everything here counts and indexes positionally (issue #118).
from vaft.ods_access import path_count as _ods_count

__all__ = [
    "STAGE_METRICS",
    "STAGE_PRECONDITIONS",
]


def _no_equilibrium_slices(source: Any) -> str | None:
    """EFIT can legitimately produce nothing for a shot; that is not a plot bug."""
    if _ods_count(source, "equilibrium.time_slice") == 0:
        return (
            "EFIT produced no accepted equilibrium time slice for this shot; "
            "see the stage's efit_status"
        )
    return None


def _no_plasma_onset(source: Any) -> str | None:
    """A discharge that never formed a plasma has no residual onset to validate.

    The eddy figures are about where the plasma signal emerges from the vacuum
    response. Without a plasma current there is no such time, and that is a
    property of the shot rather than a fault in the reconstruction.
    """
    from vaft.machine_mapping.magnetics import vfit_plasma_mgods_startend

    try:
        start, end = vfit_plasma_mgods_startend(source)
    except Exception:
        return "magnetics.ip is unreadable, so no plasma-current onset can be located"
    if start < 0 or end <= start:
        return (
            "no plasma-current onset can be located in magnetics.ip; this shot "
            "carries no plasma phase to separate from the vacuum response"
        )
    return None


#: Fields a `toroidal_mode` entry only carries when a solver actually produced
#: a result for that cell. `n_tor` is deliberately not among them: the IDS is
#: laid out as a dense (time, n_tor) grid, so *every* requested mode has an
#: entry stating its own `n_tor` whether or not it was solved for.
_MHD_LINEAR_MODE_PAYLOAD = ("energy_perturbed", "ballooning_type.name")


def _no_toroidal_modes(source: Any) -> str | None:
    count = _ods_count(source, "mhd_linear.time_slice")
    if count == 0:
        return "the GPEC suite mapped no mhd_linear time slice for this shot"
    if not any(
        _ods_count(source, f"mhd_linear.time_slice.{index}.toroidal_mode")
        for index in range(count)
    ):
        return "the GPEC suite mapped no toroidal mode for this shot"
    # A dense grid is structurally present even for a shot no solver produced
    # anything for, so emptiness is a question about payloads, not entries.
    for index in range(count):
        root = f"mhd_linear.time_slice.{index}.toroidal_mode"
        for position in range(_ods_count(source, root)):
            if any(f"{root}.{position}.{field}" in source for field in _MHD_LINEAR_MODE_PAYLOAD):
                return None
    return (
        "the GPEC suite produced no usable result for any toroidal mode of this "
        "shot; the mhd_linear grid is padding only"
    )


def _no_chease_slices(source: Any) -> str | None:
    """CHEASE can legitimately produce nothing: disabled, no executable, or
    every input g-file failed to refine. `run_chease_refinement.py` writes an
    empty `equilibrium.time_slice` product on every such path.
    """
    if _ods_count(source, "equilibrium.time_slice") == 0:
        return (
            "CHEASE produced no refined equilibrium time slice for this shot; "
            "see the stage's chease_status"
        )
    return None


#: Stages whose data product can be legitimately empty.  The callable returns a
#: reason when it is, and every declared plot -- required ones included -- is
#: then recorded as skipped with that reason instead of failing the stage.  This
#: is not a silent swallow: the empty product is a known, reported state that
#: lands in the manifest, unlike a required plot whose data is unexpectedly
#: absent, which still raises.
STAGE_PRECONDITIONS: dict[str, Any] = {
    "chease": _no_chease_slices,
    "eddy": _no_plasma_onset,
    "efit": _no_equilibrium_slices,
    "mhd_linear": _no_toroidal_modes,
}


def _efit_metrics(source: Any, **_context: Any) -> dict[str, Any]:
    """Coverage, residuals, goodness of fit and convergence, slice by slice.

    Every quantity is either submitted to EFIT or produced by the EFIT run.
    Each block carries a ``tier`` so a consumer can tell a primary validation
    metric from a diagnostic or from solver metadata without reading the design.
    """
    import numpy as np

    from vaft.formula.statistics import rms
    from vaft.omas.efit_quality import (
        CONSTRAINT_STATES,
        FAMILIES,
        constraint_table,
        efit_quality_metrics,
        family_chi_squared_sum,
        fitted_mask,
        slice_times,
    )

    quality = efit_quality_metrics(source)
    times = slice_times(source)
    for index in range(times.size):
        families: dict[str, Any] = {}
        for family, _title, _unit, _scale, is_array in FAMILIES:
            table = constraint_table(
                source, time_slice=index, family=family, is_array=is_array
            )
            # One convention, defined once in vaft.omas.efit_quality: the
            # residual RMS runs over the fitted channels and the chi-square
            # aggregate over the enabled ones, exactly as fit_quality_metrics
            # reports them.
            fitted = fitted_mask(table)
            families[family] = {
                **{state: table.count(state) for state in CONSTRAINT_STATES},
                "residual_rms": rms(table.residual[fitted]),
                # The stored value in SI units: EFIT's normalization for these is
                # not recorded anywhere, so this is not a reduced chi-square.
                "chi_squared_sum": family_chi_squared_sum(table),
            }
        quality["slices"][index]["families"] = families
        quality["slices"][index]["grad_shafranov_deviation"] = (
            quality["slices"][index]["convergence"]["error"]["final_error"]
        )
        quality["slices"][index]["iterations"] = (
            quality["slices"][index]["convergence"]["iterations"]["iterations"]
        )
    return quality


def _mhd_linear_metrics(source: Any, **context: Any) -> dict[str, Any]:
    """Solver run status and the Delta-prime values that have no IDS slot.

    Both live only in the stage manifest, so this reads it when the workflow
    passes one; without it only what the IDS carries is reported.
    """
    slice_count = _ods_count(source, "mhd_linear.time_slice")
    modes: dict[str, Any] = {}
    for index in range(slice_count):
        root = f"mhd_linear.time_slice.{index}.toroidal_mode"
        for position in range(_ods_count(source, root)):
            entry = source[f"{root}.{position}"]
            n_tor = entry.get("n_tor")
            if n_tor is None:
                continue
            energy = entry.get("energy_perturbed")
            modes.setdefault(str(int(n_tor)), []).append(
                {
                    "time_slice": index,
                    "energy_perturbed": None if energy is None else float(energy),
                }
            )

    metrics: dict[str, Any] = {
        "schema_version": 1,
        "time_slice_count": slice_count,
        "modes": modes,
    }
    manifest_path = context.get("stage_manifest")
    if manifest_path:
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        cells = payload.get("modules_modes", {})
        metrics["solver_runs"] = {
            key: {
                "status": cell.get("status"),
                "reason": cell.get("reason"),
                # RDCON/STRIDE Delta-prime: mapped into `extras`, never into the
                # IDS, so the manifest is its only home.
                "modes": {
                    str(mode): {
                        "module": detail.get("module"),
                        "variable": detail.get("variable"),
                        "value": detail.get("value"),
                    }
                    for mode, detail in (cell.get("modes") or {}).items()
                },
            }
            for key, cell in sorted(cells.items())
        }
        metrics["solver_status_counts"] = {
            status: sum(
                1 for cell in cells.values() if cell.get("status") == status
            )
            for status in sorted({cell.get("status") for cell in cells.values()})
        }
    return metrics


def _eddy_metrics(source: Any, **_context: Any) -> dict[str, Any]:
    """Quantitative QA behind the eddy stage's two validation figures."""
    from vaft.omas.vacuum_magnetics import (
        DEFAULT_MIN_WALL_AUTHORITY,
        plasma_onset_time,
        synthetic_vacuum_magnetics,
        vacuum_magnetics_metrics,
    )

    # Metrics run even for an empty product, so the no-plasma case is reported
    # rather than raised -- the same condition the precondition reports.
    reason = _no_plasma_onset(source)
    if reason is not None:
        return {"schema_version": 1, "status": "unavailable", "reason": reason}

    onset = plasma_onset_time(source)
    # The validation window is the pre-plasma stretch, so that is the interval
    # the channel selection asks about too (#189): a probe that fails after
    # breakdown is still a good witness before it.
    channels = synthetic_vacuum_magnetics(
        source, validity_window=(float("-inf"), onset)
    )
    ip_time = source.get("magnetics.ip.0.time", None)
    ip_data = source.get("magnetics.ip.0.data", None)
    return vacuum_magnetics_metrics(
        channels,
        plasma_onset=onset,
        plasma_current=(
            None if ip_time is None or ip_data is None else (ip_time, ip_data)
        ),
        # The scored block leaves out channels the wall barely reaches (VEST's
        # inboard flux loops), whose improvement sign is noise.
        min_wall_authority=DEFAULT_MIN_WALL_AUTHORITY,
    )


def _chease_metrics(source: Any, **_context: Any) -> dict[str, Any]:
    """Refinement-comparison and physics-flag QA behind the chease figures.

    Primary tier: per-slice profile/boundary RMS change, psi and Ip diffs --
    `comparison_metrics`, computed once in `vaft.code.chease` and embedded onto
    the refined ODS's `equilibrium.code.parameters` by `generate_chease_ods.py`
    -- plus q0/q95, q-monotonicity and pressure positivity, read directly off
    each refined time slice. Metadata tier: `records_summary`, which input
    gfile refined into a time slice against which failed, coming from the same
    embedded block. There is no CHEASE `chease.log`/`NOUT` parser anywhere in
    this codebase; the diagnostic tier the issue calls for is therefore scoped
    to the coarse per-record status already carried in `records_summary`
    rather than full log parsing.
    """
    embedded: dict[str, Any] = {}
    raw_parameters = source.get("equilibrium.code.parameters", None)
    if raw_parameters:
        try:
            embedded = json.loads(raw_parameters)
        except (TypeError, ValueError):
            embedded = {}
    comparison_by_slice = embedded.get("comparison_metrics", {}) or {}

    slice_count = _ods_count(source, "equilibrium.time_slice")
    slices: dict[str, Any] = {}
    for index in range(slice_count):
        root = f"equilibrium.time_slice.{index}"
        q = np.asarray(source.get(f"{root}.profiles_1d.q", []), dtype=float)
        pressure = np.asarray(source.get(f"{root}.profiles_1d.pressure", []), dtype=float)
        diffs = np.diff(q)
        slices[str(index)] = {
            "q0": source.get(f"{root}.global_quantities.q_axis", None),
            "q95": source.get(f"{root}.global_quantities.q_95", None),
            "q_monotonic": bool(diffs.size == 0 or np.all(diffs >= 0) or np.all(diffs <= 0)),
            "pressure_positive": bool(pressure.size == 0 or np.all(pressure >= 0)),
            "comparison": comparison_by_slice.get(str(index), {}),
        }

    return {
        "schema_version": 1,
        "time_slice_count": slice_count,
        "slices": slices,
        "records_summary": embedded.get("records_summary", []),
    }


def _diagnostics_metrics(source: Any, **_context: Any) -> dict[str, Any]:
    """Magnetics signal-quality QA for the diagnostics stage (issue #189).

    Quantitative, so a channel that degraded between shots is visible without
    opening a figure.  Deliberately report-only: none of these numbers gates
    the stage, because their thresholds have not been justified across a
    representative VEST population.

    The detectors are re-run here rather than read back off the ODS because the
    metrics -- noise, drift, dynamic range, event extents -- are richer than
    what the Data Dictionary's validity fields can carry.  The *verdict* the
    ODS carries and the verdict computed here are the same function of the same
    waveforms.
    """
    from vaft.validation.magnetics import (
        magnetics_quality_metrics,
        validate_magnetics_signals,
    )

    if "magnetics" not in source:
        return {
            "schema_version": 1,
            "status": "unavailable",
            "reason": "the shot has no magnetics IDS to assess",
        }
    return magnetics_quality_metrics(source, validate_magnetics_signals(source))


#: Stages that record scalar validation results alongside their figures.  The
#: callable takes the stage's data product and returns the block the plot
#: manifest carries under ``"metrics"``.
STAGE_METRICS: dict[str, Any] = {
    "chease": _chease_metrics,
    "diagnostics": _diagnostics_metrics,
    "eddy": _eddy_metrics,
    "efit": _efit_metrics,
    "mhd_linear": _mhd_linear_metrics,
}
