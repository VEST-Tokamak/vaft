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
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "STAGE_METRICS",
    "STAGE_VALIDATION_PLOTS",
    "ValidationPlot",
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
    # Seeded with the figure the pipeline already produced; the constraint and
    # residual validation issue #139 asks for lands as a follow-up.
    "efit": (
        ValidationPlot("equilibrium_overview_verification"),
        ValidationPlot("equilibrium_overview", required=False),
        ValidationPlot("equilibrium_field_psi", required=False),
    ),
}


def _eddy_metrics(source: Any, **_context: Any) -> dict[str, Any]:
    """Quantitative QA behind the eddy stage's two validation figures."""
    from vaft.omas.vacuum_magnetics import (
        plasma_onset_time,
        synthetic_vacuum_magnetics,
        vacuum_magnetics_metrics,
    )

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


#: Stages that record scalar validation results alongside their figures.  The
#: callable takes the stage's data product and returns the block the plot
#: manifest carries under ``"metrics"``.
STAGE_METRICS: dict[str, Any] = {"eddy": _eddy_metrics}


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


def _render_raw_plot(entry: ValidationPlot, source: Any, context: Mapping[str, Any]):
    from vaft.plot import render_panels

    if entry.plot != "raw_acquisition_qa":
        raise KeyError(f"no raw validation renderer named {entry.plot!r}")
    model = raw_acquisition_qa_model(
        source,
        required_fields=context.get("required_fields", ()),
        shot=context.get("shot"),
    )
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

    records: list[dict[str, Any]] = []
    for entry in entries:
        target = directory / entry.filename
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
    compute_metrics = STAGE_METRICS.get(stage)
    if compute_metrics is not None:
        manifest["metrics"] = compute_metrics(source, **context)
    return manifest
