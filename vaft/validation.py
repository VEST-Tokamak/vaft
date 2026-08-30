"""Post-generation validation plots for the FileDB pipeline (issue #139).

Each production stage owes a small, fixed set of *validation plots*: production
QA artifacts, not publication figures, whose job is to make bad calibration,
timing, geometry or reconstruction obvious at a glance.  This module is the one
declarative source of truth for which figures a stage owes, and the one executor
that renders and persists them.

It sits above both plotting layers rather than inside either:

* :mod:`vaft.plot` owns rendering and nothing else;
* :mod:`vaft.omas` owns ODS interpretation and the ``plot_*`` adapters;
* this module owns the *policy* -- which plots each stage must produce, which
  are optional, what they are called on disk, and what the stage records about
  them in its metadata.

The workflow resolves the destination through the canonical FileDB ``plot``
artifact class (``.../plot/``) and never joins a plot path by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "STAGE_METRICS",
    "STAGE_PRECONDITIONS",
    "STAGE_VALIDATION_PLOTS",
    "ValidationPlot",
    "mhd_linear_run_coverage_model",
    "raw_acquisition_qa_model",
    "render_stage_plots",
    "stage_plot_filenames",
    "stages",
    "validation_plots",
]

#: Suffix used for every persisted validation plot.
PLOT_SUFFIX = ".png"

#: How a stage's plots are produced.  ``"ods"`` entries name a canonical
#: ``vaft.plot`` renderer and are rendered from an ODS through the
#: ``vaft.omas`` adapter layer.  ``"raw"`` entries are built here, because the
#: raw DAQ archive is not an ODS and therefore has no adapter home.
KINDS = ("ods", "raw")


@dataclass(frozen=True)
class ValidationPlot:
    """One figure a stage owes after its data product is generated."""

    plot: str
    filename: str = ""
    required: bool = True
    kind: str = "ods"
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown validation-plot kind {self.kind!r}; expected one of {KINDS}")
        if not self.plot:
            raise ValueError("ValidationPlot.plot must name a renderer")
        object.__setattr__(self, "filename", self.filename or f"{self.plot}{PLOT_SUFFIX}")
        object.__setattr__(self, "options", dict(self.options))


#: The per-stage contract.  Stage keys match the FileDB domain/``OMASStage``
#: names so a stage's plots resolve next to its own output and metadata.
STAGE_VALIDATION_PLOTS: dict[str, tuple[ValidationPlot, ...]] = {
    # Raw acquisition already has strong programmatic validation; this is one
    # compact overview for human inspection of those checks, deliberately not
    # one figure per raw field.
    "raw": (
        ValidationPlot(
            plot="raw_acquisition_qa",
            filename="raw_overview_acquisition.png",
            kind="raw",
        ),
    ),
    # Enough machine geometry to catch a wrong machine era or malformed
    # geometry before anything downstream trusts it.  ``machine_geometry_poloidal``
    # carries the PF coils; the dedicated ``pf_active_geometry_poloidal`` renderer
    # needs element outlines, and VEST's PF elements are rectangles, so it is
    # deliberately not declared here rather than being permanently skipped.
    "static": (
        ValidationPlot("machine_geometry_poloidal"),
        ValidationPlot("wall_geometry_poloidal", required=False),
        ValidationPlot("pf_passive_geometry_poloidal", required=False),
        ValidationPlot("magnetics_geometry_poloidal", required=False),
    ),
    # A fixed per-shot visual QA set, not everything the registry can draw for
    # the shot; the manifest's ``available`` list carries the rest.
    "diagnostics": (
        ValidationPlot("magnetics_time_ip"),
        ValidationPlot("magnetics_overview"),
        ValidationPlot("pf_active_time_current"),
        ValidationPlot("magnetics_time_b_field_pol_probe_field", required=False),
        ValidationPlot("magnetics_time_flux_loop_flux", required=False),
        ValidationPlot("tf_time_coil_current", required=False),
        ValidationPlot("interferometer_overview", required=False),
    ),
    # Validated physically, by forward-modeling the magnetic response of the
    # reconstructed vacuum current system, rather than by plotting the fitted
    # eddy currents. See vaft.omas.vacuum_magnetics.
    "eddy": (
        ValidationPlot(
            plot="magnetics_overview_vacuum",
            filename="vacuum_magnetics_overview.png",
        ),
        ValidationPlot(
            plot="magnetics_overview_plasma_residual",
            filename="residual_plasma_signal.png",
        ),
    ),
    # What was submitted to EFIT and what came back, in that order: a converged
    # solution built on a dead channel set is still a bad one, so the submitted
    # constraints are validated before the reconstruction is interpreted.
    "efit": (
        ValidationPlot(
            plot="equilibrium_overview_constraints",
            filename="efit_constraints_submitted.png",
        ),
        ValidationPlot(
            plot="equilibrium_overview_constraint_coverage",
            filename="efit_constraint_coverage.png",
        ),
        ValidationPlot(
            plot="equilibrium_overview_residuals",
            filename="efit_reconstruction_residuals.png",
        ),
        ValidationPlot(
            plot="equilibrium_overview_fit_quality",
            filename="efit_fit_quality.png",
        ),
        ValidationPlot(
            plot="equilibrium_overview_convergence",
            filename="efit_convergence.png",
        ),
        ValidationPlot("equilibrium_overview_verification"),
        ValidationPlot("equilibrium_overview", required=False),
        ValidationPlot("equilibrium_field_psi", required=False),
    ),
    # Deliberately minimal: only `n_tor` and DCON's `energy_perturbed` reach the
    # IDS today. RDCON/STRIDE's Delta-prime has no IDS slot and survives only in
    # the stage manifest, so it is recorded as a metric rather than invented into
    # a figure.
    "mhd_linear": (
        ValidationPlot(
            plot="mhd_linear_time_energy_perturbed",
            filename="stability_energy_perturbed.png",
        ),
        # Issue #173 phase 1: which (module, mode, time) cells actually ran and
        # succeeded, independent of the #170 IDS-contract work -- its data is
        # the stage manifest's `modules_modes` table, not the ODS, so it is a
        # "raw" plot (see `mhd_linear_run_coverage_model`) even though this
        # stage's other plot is ODS-driven.
        ValidationPlot(
            plot="mhd_linear_run_coverage",
            filename="stability_run_coverage.png",
            kind="raw",
            required=False,
        ),
    ),
    # CHEASE refines an EFIT equilibrium at fixed boundary: the per-slice
    # comparison figure (`chease_comparison.png`, one per time label) stays a
    # side effect of `run_chease`, tracked by its own manifest, because its
    # count and filenames vary per shot and a required plot needs a
    # deterministic name. Only the two per-shot summaries -- which the
    # per-slice figures cannot give -- are declared here. Both are rendered
    # purely from the refined ODS: `comparison_metrics` per time slice rides
    # along on `equilibrium.code.parameters`, embedded by
    # `generate_chease_ods.py`, so the ODS alone is what is validated.
    "chease": (
        ValidationPlot(
            plot="chease_overview_refinement_summary",
            filename="chease_refinement_summary.png",
        ),
        ValidationPlot(
            plot="chease_overview_profile_validity",
            filename="chease_profile_validity.png",
        ),
    ),
}


def _ods_count(source: Any, path: str) -> int:
    """Length of an ODS array of structures, 0 when the node is absent.

    Iterating an OMAS AOS yields its integer keys rather than its entries, so
    everything here indexes positionally instead.
    """
    try:
        return len(source[path])
    except (KeyError, ValueError, IndexError, TypeError):
        return 0


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
        plasma_onset_time,
        synthetic_vacuum_magnetics,
        vacuum_magnetics_metrics,
    )

    # Metrics run even for an empty product, so the no-plasma case is reported
    # rather than raised -- the same condition the precondition reports.
    reason = _no_plasma_onset(source)
    if reason is not None:
        return {"schema_version": 1, "status": "unavailable", "reason": reason}

    channels = synthetic_vacuum_magnetics(source)
    ip_time = source.get("magnetics.ip.0.time", None)
    ip_data = source.get("magnetics.ip.0.data", None)
    return vacuum_magnetics_metrics(
        channels,
        plasma_onset=plasma_onset_time(source),
        plasma_current=(
            None if ip_time is None or ip_data is None else (ip_time, ip_data)
        ),
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


#: Stages that record scalar validation results alongside their figures.  The
#: callable takes the stage's data product and returns the block the plot
#: manifest carries under ``"metrics"``.
STAGE_METRICS: dict[str, Any] = {
    "chease": _chease_metrics,
    "eddy": _eddy_metrics,
    "efit": _efit_metrics,
    "mhd_linear": _mhd_linear_metrics,
}


def stages() -> tuple[str, ...]:
    """Every stage with a declared validation-plot set, sorted."""
    return tuple(sorted(STAGE_VALIDATION_PLOTS))


def validation_plots(stage: str) -> tuple[ValidationPlot, ...]:
    """The declared plots for ``stage``."""
    try:
        return STAGE_VALIDATION_PLOTS[stage]
    except KeyError:
        raise KeyError(
            f"no validation-plot set for stage {stage!r}; "
            f"declared stages are: {', '.join(stages())}"
        ) from None


def stage_plot_filenames(stage: str, *, required_only: bool = False) -> tuple[str, ...]:
    """File names ``stage`` writes into its ``plot/`` directory.

    Snakemake declares the required names as real rule outputs, so a validation
    plot cannot silently become an untracked side effect.
    """
    return tuple(
        entry.filename
        for entry in validation_plots(stage)
        if entry.required or not required_only
    )


# ---------------------------------------------------------------------------
# Raw acquisition QA
# ---------------------------------------------------------------------------

def _field_arrays(payload: Mapping[str, Any]) -> dict[int, np.ndarray]:
    fields = payload.get("fields")
    if not isinstance(fields, Mapping):
        raise ValueError("raw payload has no 'fields' mapping")
    arrays: dict[int, np.ndarray] = {}
    for code, entry in fields.items():
        data = entry.get("data") if isinstance(entry, Mapping) else None
        arrays[int(code)] = np.asarray(data if data is not None else [], dtype=float)
    return arrays


def _field_types(payload: Mapping[str, Any]) -> dict[int, str]:
    fields = payload.get("fields", {})
    return {
        int(code): str(entry.get("type", "unknown")) if isinstance(entry, Mapping) else "unknown"
        for code, entry in fields.items()
    }


def _quality_fractions(data: np.ndarray) -> tuple[float, float]:
    """Return ``(non-finite fraction, zero fraction)`` for one field."""
    if data.size == 0:
        return 1.0, 1.0
    finite = np.isfinite(data)
    non_finite = float(1.0 - finite.mean())
    zero = float((data[finite] == 0.0).mean()) if finite.any() else 1.0
    return non_finite, zero


def raw_acquisition_qa_model(
    payload: Mapping[str, Any],
    *,
    required_fields: Sequence[int] = (),
    shot: int | None = None,
    max_signal_panels: int = 4,
):
    """Build the compact raw-acquisition QA view model for one shot.

    Shows what identifies an unusable acquisition -- per-field sample coverage,
    zero/non-finite fractions, and a few representative mandatory signals --
    rather than duplicating the archive one figure per field.  The raw dump
    stores samples without a per-field time base, so signals are drawn against
    sample index and window consistency is read off the sample-count panel:
    fields of the same DAQ class must share a sample count.
    """
    from vaft.plot import LineSeries, Panels, Series

    arrays = _field_arrays(payload)
    types = _field_types(payload)
    if shot is None and payload.get("shot") is not None:
        shot = int(payload["shot"])
    required = [int(code) for code in required_fields]
    codes = sorted(arrays)

    panels: list[Any] = []

    # 1) Sample coverage, split by DAQ class so a short/truncated acquisition
    #    stands out against the fields it should match.
    coverage: list[Series] = []
    for daq_type in sorted({types.get(code, "unknown") for code in codes}):
        selected = [code for code in codes if types.get(code, "unknown") == daq_type]
        if not selected:
            continue
        coverage.append(
            Series(
                x=np.asarray(selected, dtype=float),
                y=np.asarray([arrays[code].size for code in selected], dtype=float),
                label=f"{daq_type} DAQ",
                style={"marker": ".", "linestyle": "none"},
            )
        )
    missing_required = [code for code in required if arrays.get(code, np.empty(0)).size == 0]
    present_required = [code for code in required if code in arrays and arrays[code].size]
    if present_required:
        coverage.append(
            Series(
                x=np.asarray(present_required, dtype=float),
                y=np.asarray([arrays[code].size for code in present_required], dtype=float),
                label="required",
                style={"marker": "o", "linestyle": "none", "markerfacecolor": "none"},
            )
        )
    if coverage:
        panels.append(
            LineSeries(
                series=tuple(coverage),
                x_label="field code",
                y_label="samples",
                title=(
                    f"Sample coverage — {len(codes)} fields"
                    + (f", {len(missing_required)} required field(s) missing" if missing_required else "")
                ),
            )
        )

    # 2) Zero / non-finite fractions: a flatlined or dead channel reads 1.0.
    fractions = [_quality_fractions(arrays[code]) for code in codes]
    if codes:
        panels.append(
            LineSeries(
                series=(
                    Series(
                        x=np.asarray(codes, dtype=float),
                        y=np.asarray([value[1] for value in fractions], dtype=float),
                        label="zero fraction",
                        style={"marker": ".", "linestyle": "none"},
                    ),
                    Series(
                        x=np.asarray(codes, dtype=float),
                        y=np.asarray([value[0] for value in fractions], dtype=float),
                        label="non-finite fraction",
                        style={"marker": "x", "linestyle": "none"},
                    ),
                ),
                x_label="field code",
                y_label="fraction",
                y_limits=(-0.05, 1.05),
                title="Zero / non-finite sample fraction",
            )
        )

    # 3) A few representative mandatory signals, against sample index.
    for code in present_required[:max_signal_panels]:
        data = arrays[code]
        panels.append(
            LineSeries(
                series=(
                    Series(
                        x=np.arange(data.size, dtype=float),
                        y=data,
                        label=f"field {code}",
                    ),
                ),
                x_label="sample index",
                y_label="raw value",
                title=f"Field {code} ({types.get(code, 'unknown')} DAQ)",
            )
        )

    if not panels:
        raise ValueError("raw payload contains no fields to validate")

    flagged = payload.get("field_quality") or {}
    suptitle = (
        f"Raw acquisition QA — shot {shot if shot is not None else 'unknown'}: "
        f"{len(codes)} fields, {len(flagged)} flagged"
    )
    return Panels(models=tuple(panels), ncols=2, share_x=False, suptitle=suptitle)


#: `build_mhd_linear_ods`'s manifest cell key format (`vaft/omas/vest_upstream.py`).
_COVERAGE_KEY_RE = re.compile(r"^t=(?P<time>[^/]+)/(?P<module>[^/]+)/n=(?P<mode>\d+)$")

#: One marker style per solver-run status, shared by every module's panel so
#: the same status always reads the same way across panels.
_COVERAGE_STATUS_STYLE: dict[str, dict[str, Any]] = {
    "success": {"marker": "o", "color": "tab:green"},
    "missing": {"marker": "x", "color": "tab:gray"},
    "failed": {"marker": "X", "color": "tab:red"},
    "no_output": {"marker": "s", "color": "tab:orange", "markerfacecolor": "none"},
}


def mhd_linear_run_coverage_model(manifest: Mapping[str, Any]):
    """Build the DCON/RDCON/STRIDE run-coverage view model (issue #173 phase 1).

    One panel per solver module; within a panel, one point per ``(time,
    n_tor)`` cell the workflow attempted, colored by the manifest's own
    ``status`` (``success``/``missing``/``failed``/``no_output``) -- this is
    "did the run happen and produce usable output", not a physics
    comparison, so it is independent of how the #170 data-model work settles
    DCON's delta-W against RDCON/STRIDE's Delta-prime.
    """
    from vaft.plot import LineSeries, Panels, Series

    cells = manifest.get("modules_modes") or {}
    rows: list[tuple[str, int, float, str]] = []
    for key, cell in cells.items():
        match = _COVERAGE_KEY_RE.match(key)
        if not match:
            continue
        try:
            time_value = float(match.group("time"))
        except ValueError:
            continue
        rows.append((match.group("module"), int(match.group("mode")), time_value, str(cell.get("status", "unknown"))))
    if not rows:
        raise ValueError("stage manifest carries no modules_modes coverage cells")

    modules = sorted({module for module, _, _, _ in rows})
    panels = []
    for module in modules:
        module_rows = [row for row in rows if row[0] == module]
        series = []
        for status in sorted({row[3] for row in module_rows}):
            selected = [row for row in module_rows if row[3] == status]
            style = dict(_COVERAGE_STATUS_STYLE.get(status, {"marker": "."}))
            style["linestyle"] = "none"
            series.append(
                Series(
                    x=np.asarray([row[2] for row in selected], dtype=float),
                    y=np.asarray([row[1] for row in selected], dtype=float),
                    label=status,
                    style=style,
                )
            )
        n_success = sum(1 for row in module_rows if row[3] == "success")
        panels.append(
            LineSeries(
                series=tuple(series),
                x_label="time",
                y_label="toroidal mode n",
                title=f"{module} — {n_success}/{len(module_rows)} cells succeeded",
            )
        )

    shot = manifest.get("shot")
    return Panels(
        models=tuple(panels),
        ncols=min(2, len(panels)) or 1,
        share_x=False,
        suptitle=f"Stability run coverage — shot {shot if shot is not None else 'unknown'}",
    )


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def _use_non_interactive_backend() -> None:
    """Automated workflows must never need a display or a writable ``$HOME``."""
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="vaft-mpl-"))
    import matplotlib

    matplotlib.use("Agg", force=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _render_ods_plot(entry: ValidationPlot, source: Any):
    from vaft.omas.plotting import render

    result = render(entry.plot, source, show=False, **entry.options)
    return result[0]


def _raw_plot_unavailable_reason(entry: ValidationPlot, context: Mapping[str, Any]) -> str | None:
    """Whether a "raw" kind plot's data is missing, mirroring the "ods" availability check.

    ``raw_acquisition_qa`` has always had no such pre-check (its payload *is*
    the stage's `source`, never absent) and keeps raising from inside its own
    model builder on genuinely bad data; only plots whose data lives
    elsewhere (e.g. `mhd_linear_run_coverage`'s stage manifest) need one.
    """
    if entry.plot == "mhd_linear_run_coverage" and not context.get("stage_manifest"):
        return "no stage_manifest supplied; run coverage cannot be computed"
    return None


def _render_raw_plot(entry: ValidationPlot, source: Any, context: Mapping[str, Any]):
    from vaft.plot import render_panels

    if entry.plot == "raw_acquisition_qa":
        model = raw_acquisition_qa_model(
            source,
            required_fields=context.get("required_fields", ()),
            shot=context.get("shot"),
        )
    elif entry.plot == "mhd_linear_run_coverage":
        manifest_path = context.get("stage_manifest")
        if not manifest_path:
            raise ValueError("mhd_linear_run_coverage requires context['stage_manifest']")
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        model = mhd_linear_run_coverage_model(manifest)
    else:
        raise KeyError(f"no raw validation renderer named {entry.plot!r}")
    figure, _axes = render_panels(model, show=False, figsize=(13.0, 8.0))
    return figure


def render_stage_plots(
    stage: str,
    source: Any,
    output_dir: str | os.PathLike[str],
    **context: Any,
) -> dict[str, Any]:
    """Render and persist ``stage``'s validation plots into ``output_dir``.

    ``source`` is the stage's own data product: an ODS (or anything
    :mod:`vaft.omas` normalizes) for ``ods`` entries, the decoded raw payload
    for the raw stage.

    A **required** plot whose data is missing, or which fails to render, raises:
    a declared validation artifact that cannot be produced is an actionable
    stage failure, never a silent gap.  An **optional** plot whose data is
    absent is recorded as ``skipped`` with a reason; an optional plot whose data
    *is* present but which fails to render still raises, because that is a bug
    rather than an absent IDS.

    Returns the manifest fragment the stage records in its metadata.
    """
    _use_non_interactive_backend()
    from vaft.plot import save_figure

    entries = validation_plots(stage)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    available: tuple[str, ...] = ()
    if any(entry.kind == "ods" for entry in entries):
        from vaft.omas import available_plots

        available = tuple(sorted(row["name"] for row in available_plots(source)))

    empty_reason = None
    precondition = STAGE_PRECONDITIONS.get(stage)
    if precondition is not None:
        empty_reason = precondition(source)

    records: list[dict[str, Any]] = []
    for entry in entries:
        target = directory / entry.filename
        if empty_reason is not None:
            records.append(
                {
                    "name": entry.plot,
                    "file": entry.filename,
                    "status": "skipped",
                    "reason": empty_reason,
                }
            )
            continue
        if entry.kind == "ods" and entry.plot not in available:
            reason = f"ODS does not carry the data {entry.plot!r} requires"
            if entry.required:
                raise ValueError(
                    f"required validation plot {entry.plot!r} for stage {stage!r}: {reason}"
                )
            records.append(
                {
                    "name": entry.plot,
                    "file": entry.filename,
                    "status": "skipped",
                    "reason": reason,
                }
            )
            continue
        if entry.kind == "raw":
            reason = _raw_plot_unavailable_reason(entry, context)
            if reason is not None:
                if entry.required:
                    raise ValueError(
                        f"required validation plot {entry.plot!r} for stage {stage!r}: {reason}"
                    )
                records.append(
                    {
                        "name": entry.plot,
                        "file": entry.filename,
                        "status": "skipped",
                        "reason": reason,
                    }
                )
                continue

        if entry.kind == "ods":
            figure = _render_ods_plot(entry, source)
        else:
            figure = _render_raw_plot(entry, source, context)
        save_figure(figure, target)
        records.append(
            {
                "name": entry.plot,
                "file": entry.filename,
                "status": "generated",
                "bytes": target.stat().st_size,
                "sha256": _sha256(target),
            }
        )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "stage": stage,
        "plots": records,
    }
    if available:
        manifest["available"] = list(available)
    if empty_reason is not None:
        manifest["status"] = "empty"
        manifest["reason"] = empty_reason
    # Metrics are computed even for an empty product: a stage is empty precisely
    # when something upstream did not produce what it should have, which is when
    # its diagnostics matter most. `mhd_linear`'s solver-run block, for one,
    # explains *why* no toroidal mode was mapped.
    compute_metrics = STAGE_METRICS.get(stage)
    if compute_metrics is not None:
        manifest["metrics"] = compute_metrics(source, **context)
    return manifest
