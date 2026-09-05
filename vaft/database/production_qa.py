"""Post-generation validation plots for the FileDB pipeline (issue #139).

Each production stage owes a small, fixed set of *validation plots*: production
QA artifacts, not publication figures, whose job is to make bad calibration,
timing, geometry or reconstruction obvious at a glance.  This module is the one
declarative source of truth for which figures a stage owes, and the one executor
that renders and persists them.

It is a **database-layer** concern (#253 §11, #338): the question it answers is
whether an artifact is complete and fit to publish -- every required figure
present, hashed, and recorded in a manifest next to the product it validates.
The scientific question of whether the product is *credible* is answered
upstream, by :mod:`vaft.validation.stage_evidence`, whose
:data:`~vaft.validation.stage_evidence.STAGE_PRECONDITIONS` and
:data:`~vaft.validation.stage_evidence.STAGE_METRICS` this module consumes.  The
dependency runs only that way; this module must not reach into
``vaft.machine_mapping``, which imports ``vaft.database`` and would close a
cycle.

It sits above both plotting layers rather than inside either:

* :mod:`vaft.plot` owns rendering and nothing else;
* :mod:`vaft.omas` owns ODS interpretation and the ``plot_*`` adapters;
* this module owns the *artifact policy* -- which plots each stage must produce,
  which are optional, what they are called on disk, and what the stage records
  about them in its metadata.

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

from vaft.validation.stage_evidence import STAGE_METRICS, STAGE_PRECONDITIONS

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
    # carries the PF coils; the dedicated ``pf_coil_geometry_poloidal`` renderer
    # needs element outlines, and VEST's PF elements are rectangles, so it is
    # deliberately not declared here rather than being permanently skipped.
    "static": (
        ValidationPlot("machine_geometry_poloidal"),
        ValidationPlot("wall_geometry_poloidal", required=False),
        ValidationPlot("passive_structure_geometry_poloidal", required=False),
        ValidationPlot("magnetics_geometry_poloidal", required=False),
    ),
    # A fixed per-shot visual QA set, not everything the registry can draw for
    # the shot; the manifest's ``available`` list carries the rest.
    "diagnostics": (
        ValidationPlot("plasma_current_time"),
        ValidationPlot("magnetics_overview"),
        ValidationPlot("pf_coil_time_current"),
        ValidationPlot("b_field_probe_time_field", required=False),
        ValidationPlot("flux_loop_time_flux", required=False),
        ValidationPlot("tf_coil_time_current", required=False),
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
        # The stage's artifact keeps its content and name: the four time
        # histories, which plot_equilibrium_overview no longer draws (#261).
        ValidationPlot(
            "equilibrium_overview_histories", filename="equilibrium_overview.png", required=False
        ),
        ValidationPlot("equilibrium_field_psi", required=False),
    ),
    # DCON's eigenfunction now reaches the IDS as well as `n_tor` and
    # `energy_perturbed`, so it gets a figure. RDCON/STRIDE's Delta-prime still
    # has no `mhd_linear` slot and survives only in the stage manifest, so it
    # stays a recorded metric rather than being invented into a figure.
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
        # The eigenfunction reaches the IDS only when DCON's companion `match`
        # ran (it is what writes solutions.bin), which is a legitimate thing for
        # an installation not to have -- so this is optional, and its absence is
        # recorded as a skip rather than failing the stage.
        ValidationPlot(
            plot="mhd_linear_overview_eigenfunction",
            filename="stability_eigenfunction.png",
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
    from vaft.plot.environment import use_non_interactive_backend

    use_non_interactive_backend()


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
