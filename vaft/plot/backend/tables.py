"""Builders of the non-graphical canonical views: tables and text summaries (issue #1180).

These are recipes like any other: they select a slice, read or compute the
quantities, and return a typed view model -- a
:class:`~vaft.plot.models.Table` or :class:`~vaft.plot.models.TextSummary` --
holding numbers, stored units and classifications.  Nothing here formats a
number; :mod:`vaft.plot.renderers.tables` does that.  The selection and the
reductions are the existing ones: the slice summary is
:func:`~vaft.plot.backend.recipes.slice_global_cells` at the slice
:func:`~vaft.plot.backend.recipes.resolve_time_slice` chooses (the slice
``equilibrium_overview`` draws), and the fit-quality table is
:func:`vaft.omas.efit_quality.fit_quality_metrics` at one slice (what
``equilibrium_overview_fit_quality`` plots).

Imported at the end of :mod:`vaft.plot.backend.recipes`, so the entries land in
the one ``RECIPES`` table.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from vaft.plot.models import Table, TableCell, TableColumn, TextItem, TextSection, TextSummary

from . import recipes as _recipes
from .recipes import (
    NEUTRAL,
    OMAS_BOUND,
    OWN_TIME,
    RECIPES,
    CallableRecipe,
    _array,
    _count,
    _derived_profiles_for,
    _flux_display,
    _efit_suptitle,
    _get,
    _require_slices,
    _slice_times,
    resolve_time_slice,
    slice_global_cells,
)

__all__ = ["slice_boundary_cells", "validation_verdicts"]

#: The boundary shape a slice summary states: label, ``boundary`` leaf, stored unit.
_SLICE_SHAPE: tuple[tuple[str, str, str], ...] = (
    ("R_geo", "geometric_axis.r", "m"),
    ("Z_geo", "geometric_axis.z", "m"),
    ("minor radius", "minor_radius", "m"),
    ("elongation", "elongation", ""),
    ("triangularity (upper)", "triangularity_upper", ""),
    ("triangularity (lower)", "triangularity_lower", ""),
)

#: The note a value read from the derived copy carries.
_DERIVED_NOTE = (
    "not stored in the slice; derived from it by "
    "vaft.omas.update_equilibrium_derived_profiles on a private copy"
)


def _boundary_scalar(ods: Any, index: int, leaf: str) -> float:
    raw = _get(ods, f"equilibrium.time_slice.{index}.boundary.{leaf}")
    try:
        return float(np.asarray(raw, dtype=float).ravel()[0]) if raw is not None else np.nan
    except (IndexError, TypeError, ValueError):
        return np.nan


def slice_boundary_cells(ods: Any, index: int, derived: Any = None) -> list[tuple[str, TableCell]]:
    """``(label, TableCell)`` per stored boundary-shape quantity of one slice.

    Stored first, else read from the derived copy (and noted as such), else
    empty -- the rule of :func:`~vaft.plot.backend.recipes.slice_global_cells`.
    """
    cells = []
    for label, leaf, unit in _SLICE_SHAPE:
        value = _boundary_scalar(ods, index, leaf)
        note = ""
        if not np.isfinite(value) and derived is not None:
            value = _boundary_scalar(derived, index, leaf)
            note = _DERIVED_NOTE if np.isfinite(value) else ""
        if not np.isfinite(value):
            cells.append((label, TableCell(None)))
            continue
        cells.append((label, TableCell(value, unit=unit, subject="equilibrium" if unit else "", note=note)))
    return cells


def _psi_display(ods: Any, index: int, units: Any) -> Any:
    """The flux display ``units=`` asks for, resolved as ``equilibrium_overview`` resolves it.

    ``None`` keeps the stored convention's default; an explicit unit or
    ``"auto"`` (judged on the slice's own psi map, as the overview's map is)
    goes through the same :func:`~vaft.plot.backend.recipes._flux_display`,
    so the table states psi in the unit the overview's panel does.
    """
    if units is None:
        return None
    from .convention import psi_convention

    data = _array(ods, f"equilibrium.time_slice.{index}.profiles_2d.0.psi")
    return _flux_display(ods, index, units, convention=psi_convention(ods, index), data=data)


def _global_cells_with_provenance(
    ods: Any, index: int, derived: Any, units: Any = None
) -> list[tuple[str, TableCell]]:
    """The slice's global quantities, each derived one carrying :data:`_DERIVED_NOTE`."""
    flux_display = _psi_display(ods, index, units)
    stored = dict(slice_global_cells(ods, index, flux_display=flux_display))
    cells = []
    for label, cell in slice_global_cells(ods, index, derived, flux_display=flux_display):
        if stored[label].missing and not cell.missing:
            cell = TableCell(
                cell.value, unit=cell.unit, subject=cell.subject, quantity=cell.quantity,
                display_unit=cell.display_unit, format=cell.format, note=_DERIVED_NOTE,
            )
        cells.append((label, cell))
    return cells


def _selected_slice(ods: Any, options: dict[str, Any]) -> tuple[int, float, str, int, Any]:
    """``(index, stored time, reason, slice count, pulse)`` of the requested slice.

    The selection of ``equilibrium_overview``: the representative slice unless
    ``time=`` (snapped to a stored slice, never interpolated) or
    ``time_slice=`` names one.
    """
    index, time_value, reason = resolve_time_slice(
        ods, time=options.get("time"), time_slice=options.get("time_slice")
    )
    reason = options.get("_slice_reason") or reason
    return index, time_value, reason, _count(ods, "equilibrium.time_slice"), _get(
        ods, "dataset_description.data_entry.pulse", ""
    )


def _slice_title(pulse: Any, time_value: float, index: int, total: int, reason: str) -> str:
    """The heading ``equilibrium_overview`` gives the same slice."""
    shot = f" #{pulse}" if pulse not in (None, "") else ""
    time_text = f"t = {time_value * 1e3:.2f} ms" if np.isfinite(time_value) else "time not stored"
    return f"Equilibrium slice{shot} — {time_text} (slice {index + 1} of {total}, {reason})"


def _build_equilibrium_table_summary(ods: Any, **options: Any) -> Table:
    """The selected slice's global quantities as a ``Quantity | Value | Unit`` table."""
    index, time_value, reason, total, pulse = _selected_slice(ods, options)
    derived, _ = _derived_profiles_for(ods, index)
    rows = [
        (TableCell(label), cell) for label, cell in _global_cells_with_provenance(ods, index, derived, options.get("units"))
    ]
    return Table(
        columns=(TableColumn("Quantity"), TableColumn("Value", kind="value", units="column")),
        rows=tuple(rows),
        title=options.get("title") or _slice_title(pulse, time_value, index, total, reason),
        missing="not stored",
    )


def _build_equilibrium_text_summary(ods: Any, **options: Any) -> TextSummary:
    """The selected slice in three sections: which slice, its global quantities, its shape."""
    index, time_value, reason, total, pulse = _selected_slice(ods, options)
    derived, _ = _derived_profiles_for(ods, index)
    code = _get(ods, "equilibrium.code.name", "")
    identity = [
        TextItem("shot", None if pulse in (None, "") else pulse),
        TextItem(
            "time", time_value if np.isfinite(time_value) else None,
            unit="s", display_unit="ms", format=".2f",
        ),
        TextItem("slice", f"{index + 1} of {total}"),
        TextItem("selection", reason),
    ]
    if code not in (None, ""):
        identity.append(TextItem("code", str(code)))

    def items(cells):
        return tuple(
            TextItem(
                label, cell.value, unit=cell.unit, subject=cell.subject, quantity=cell.quantity,
                display_unit=cell.display_unit, format=cell.format, note=cell.note,
            )
            for label, cell in cells
        )

    return TextSummary(
        sections=(
            TextSection("Slice", tuple(identity)),
            TextSection("Global quantities", items(_global_cells_with_provenance(ods, index, derived, options.get("units")))),
            TextSection("Shape", items(slice_boundary_cells(ods, index, derived))),
        ),
        title=options.get("title") or _slice_title(pulse, time_value, index, total, reason),
    )


def _build_equilibrium_table_fit_quality(ods: Any, **options: Any) -> Table:
    """EFIT's goodness of fit at one slice, one row per constraint family.

    The numbers are :func:`vaft.omas.efit_quality.fit_quality_metrics` for the
    slice -- the metrics ``equilibrium_overview_fit_quality`` plots -- and the
    only classification is the one that function makes: a residual bias beyond
    two standard errors is flagged ``warn``.
    """
    from vaft.omas.efit_quality import CONSTRAINT_FAMILIES, fit_quality_metrics

    count = _require_slices(ods)
    time_slice = int(options.get("time_slice", 0) or 0)
    if not 0 <= time_slice < count:
        raise ValueError(f"time_slice={time_slice} is outside the {count} stored slices")
    times = _slice_times(ods)
    fit = fit_quality_metrics(ods, time_slice=time_slice)
    share = fit["chi_squared_share"]
    rows = []
    for family, title, _unit, _scale, _is_array in CONSTRAINT_FAMILIES:
        entry = fit["families"].get(family)
        if entry is None:
            continue
        channels = entry["channels"]
        significant = bool(entry.get("z_bias_significant"))
        rows.append((
            TableCell(title),
            TableCell(entry["fit_role"]),
            TableCell(channels.get("enabled")),
            TableCell(channels.get("disabled")),
            TableCell(channels.get("missing")),
            TableCell(entry["chi_squared_sum"] if entry["fit_role"] == "fitted" else None),
            TableCell(share.get(family), format=".3f"),
            TableCell(entry.get("z_rms"), format=".3g"),
            TableCell(entry.get("z_bias"), format="+.3g"),
            TableCell(entry.get("z_abs_max"), format=".3g"),
            TableCell(entry.get("z_abs_max_channel")),
            TableCell(
                status="warn" if significant else "",
                note="residual bias beyond two standard errors" if significant else "",
            ),
        ))
    for family, title in (("ip", "Plasma current"), ("diamagnetic_flux", "Diamagnetic flux")):
        entry = fit["scalars"].get(family)
        if entry is None:
            continue
        z = entry.get("z", float("nan"))
        rows.append((
            TableCell(title), TableCell("fitted"),
            TableCell(None), TableCell(None), TableCell(None),
            TableCell(entry["chi_squared"]),
            TableCell(share.get(family), format=".3f"),
            TableCell(None), TableCell(z, format="+.3g"),
            TableCell(abs(z) if np.isfinite(z) else None, format=".3g"),
            TableCell(None), TableCell(),
        ))
    if not rows:
        raise ValueError("equilibrium ODS carries no fitted constraints to assess")
    rows.append((
        TableCell("Total"), TableCell(None), TableCell(None), TableCell(None), TableCell(None),
        TableCell(fit["chi_squared_total"]), TableCell(None), TableCell(None), TableCell(None),
        TableCell(None), TableCell(None), TableCell(),
    ))
    reduced, dof = fit["chi_squared_reduced"], fit["degrees_of_freedom"]
    caption = (
        f"reduced chi-square {reduced:.3g} over {dof:.0f} degrees of freedom (EFIT's own count)"
        if np.isfinite(reduced) and np.isfinite(dof)
        else "reduced chi-square not available: EFIT reported no degrees of freedom"
    )
    return Table(
        columns=(
            TableColumn("Family"),
            TableColumn("Role"),
            TableColumn("Enabled", kind="value"),
            TableColumn("Disabled", kind="value"),
            TableColumn("Missing", kind="value"),
            TableColumn("χ²", kind="value"),
            TableColumn("χ² share", kind="value"),
            TableColumn("z RMS", kind="value"),
            TableColumn("z bias", kind="value"),
            TableColumn("max |z|", kind="value"),
            TableColumn("at"),
            TableColumn("Flag", kind="status"),
        ),
        rows=tuple(rows),
        title=options.get("title") or _efit_suptitle(ods, "EFIT fit quality", times, time_slice),
        caption=caption + f"; uncertainty model: {fit['uncertainty_model']}.",
        notes=("z = (measured − reconstructed) · weight / k, in units of the uncertainty EFIT was given.",),
    )


#: Per check, the result field holding the number its status was decided on,
#: and what that number is.  For a graded check this is the normalized
#: residual the registry tolerance is read against; for a rule, the one
#: quantity the rule thresholds, when there is a single one.  ``None``: the
#: rule weighs several facts at once (codes, bounds on two parameters), so no
#: one number stands for it and the cell is left empty rather than invented.
#: The ``"*"`` field of ``virial_pair_consistency`` is the largest
#: ``|pair_*_residual|`` -- the value that check grades.
_VERDICT_MEASURE: dict[str, tuple[str, str] | None] = {
    "verification.structure": None,
    "verification.continuity": ("ip_max_relative_step", "largest Ip step / median |Ip|"),
    "verification.convention": None,
    "verification.convergence": ("final_error", "final iteration error"),
    "diagnostic_fit.bpol_probe": ("z_rms", "z RMS"),
    "diagnostic_fit.flux_loop": ("z_rms", "z RMS"),
    "diagnostic_fit.pf_current": ("z_rms", "z RMS"),
    "diagnostic_fit.ip": ("z", "z"),
    "diagnostic_fit.diamagnetic_flux": ("z", "z"),
    "diagnostic_fit.global": ("chi_squared_reduced", "reduced χ²"),
    "physical_validity.virial_identity": ("rms", "RMS normalized residual"),
    "physical_validity.virial_pair_consistency": ("*", "largest |leave-one-out residual|"),
    "physical_validity.virial_conditioning": ("rt_denominator_ratio", "RT denominator ratio"),
    "physical_validity.virial_parameter_plausibility": None,
    "physical_validity.pressure_consistency": ("log_ratio", "ln(β_p integral / β_p virial)"),
    "physical_validity.q_profile": ("non_decreasing_fraction", "non-decreasing |q| fraction"),
    "physical_validity.pressure_profile": None,
    "physical_validity.diamagnetic_flux": ("relative_error", "relative error"),
    "independent_validation.kinetic_pressure": ("log_ratio", "ln(kinetic / reconstructed)"),
    "independent_validation.thomson_pressure": ("log_ratio", "ln(Thomson / reconstructed)"),
    "independent_validation.diamagnetic_energy": ("log_ratio", "ln(W_dia / W_kin virial)"),
    "independent_validation.virial_measured_mu_i": ("log_ratio", "ln(β_p measured μ_i / β_p)"),
}

#: The order statuses are counted in the caption.
_VERDICT_ORDER = ("pass", "warn", "fail", "indeterminate", "not_available")


def _verdict_value(key: str, result: dict[str, Any]) -> tuple[Any, str]:
    """``(value, measure label)`` of one check's result, ``(None, "")`` without one."""
    entry = _VERDICT_MEASURE.get(key)
    if entry is None:
        return None, ""
    field, label = entry
    if field == "*":
        residuals = [
            abs(float(result[name])) for name in ("pair_12_residual", "pair_13_residual", "pair_23_residual")
            if isinstance(result.get(name), (int, float)) and np.isfinite(result[name])
        ]
        return (max(residuals) if residuals else None), label
    value = result.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        return None, label
    return float(value), label


def _verdict_criterion(spec: Any) -> TableCell:
    """The registry's criterion: its tolerance pair, or ``rule``; the method as a note.

    The method is always attached, graded checks included: it states what the
    tolerance pair alone cannot (the one-sided electron-only coverage of the
    pressure checks, for one).
    """
    from vaft.validation.registry import MEASURES

    if spec.tolerance is None:
        return TableCell("rule", note=f"{spec.key}: {spec.method}")
    warn, fail = spec.tolerance
    return TableCell(
        f"|·| ≤ {warn:g} pass, ≤ {fail:g} warn, else fail",
        note=f"{spec.key}: {spec.method} (measure {spec.measure}: {MEASURES[spec.measure]})",
    )


def _verdict_note(key: str, result: dict[str, Any]) -> str:
    """The result's own reason, else the finding codes it raised."""
    reason = result.get("reason")
    if reason:
        return str(reason)
    issues = result.get("issues")
    if issues:
        return "findings: " + ", ".join(str(code) for code in issues)
    return ""


#: A reason longer than this is listed once beneath the table, behind a
#: marker in its row, instead of widening the column (several checks share
#: one long reason, e.g. the #891 uncertainty-model one).
_INLINE_NOTE = 80


def _verdict_note_cell(note: str) -> TableCell:
    if len(note) <= _INLINE_NOTE:
        return TableCell(note or None)
    return TableCell(None, note=note)


def validation_verdicts(ods: Any, index: int) -> dict[str, dict[str, Any]]:
    """Every registered check's result at one slice, keyed by its registry key.

    The per-slice results are :func:`vaft.validation.equilibrium.validate_equilibrium`
    at ``time_slice=index`` -- every category, on the ODS alone (its own
    ``magnetics``, ``core_profiles`` and ``thomson_scattering`` serve as the
    diagnostics and kinetic profiles).  ``verification.continuity`` is a
    whole-IDS check, so it is :func:`~vaft.validation.equilibrium.verify_continuity`
    over every stored slice instead: at one slice it is ``not_available`` by
    definition.  Nothing is recomputed here and no tolerance is applied.
    """
    from vaft.validation.equilibrium import EQUILIBRIUM_CATEGORIES, validate_equilibrium, verify_continuity
    from vaft.validation.registry import CHECKS

    report = validate_equilibrium(ods, time_slice=index)
    verdicts: dict[str, dict[str, Any]] = {}
    for key in CHECKS:
        category, check = key.split(".", 1)
        if key == "verification.continuity":
            verdicts[key] = dict(verify_continuity(ods))
            continue
        entry = report.get(category, {}).get(check) if category in EQUILIBRIUM_CATEGORIES else None
        if entry is None:
            verdicts[key] = {"status": "not_available", "reason": "not evaluated by validate_equilibrium"}
            continue
        slices = entry.get("slices")
        verdicts[key] = dict(slices[0]) if slices else dict(entry)
    return verdicts


def _build_equilibrium_table_validation(ods: Any, **options: Any) -> Table:
    """Every registered validation check's verdict at the selected slice.

    One row per :data:`vaft.validation.registry.CHECKS` entry, in registry
    order: category, check, the number its status was decided on, the
    registry's criterion, the status and the result's own reason.  The
    statuses are those of :func:`validation_verdicts` -- the existing
    validation functions, unchanged; the view adds no physics and no
    tolerance.  A check that cannot be evaluated on this input is a
    ``not_available`` row carrying the reason the check gave, never dropped.
    Every provider is cheap on an EFIT slice (the virial set is the costliest,
    well under a second), so none is skipped by this view.  The caption is
    :func:`~vaft.validation.equilibrium.aggregate_status` over the rows and
    the count of each status.
    """
    from vaft.validation.equilibrium import aggregate_status
    from vaft.validation.registry import CHECKS

    index, time_value, reason, total, pulse = _selected_slice(ods, options)
    # validate_equilibrium deep-copies whatever it is given; handing it only
    # the IDS its checks read keeps the copy (and the reads) to those.
    verdicts = validation_verdicts(_recipes._isolated_copy(ods, _VALIDATION_ROOTS), index)
    rows = []
    for key, spec in CHECKS.items():
        result = verdicts[key]
        status = str(result.get("status", "not_available"))
        value, measure = _verdict_value(key, result)
        note = _verdict_note(key, result)
        if status in ("not_available", "indeterminate") and value is not None:
            # A number the verdict was not decided on reads as a grade; keep it
            # out of the Value column and say what it was in the note instead.
            ungraded = f"{measure or 'measure'} = {value:.3g}, not graded"
            note = f"{ungraded}; {note}" if note else ungraded
            value = None
        if key == "verification.continuity":
            whole = f"whole IDS, {total} stored slices"
            note = f"{whole}; {note}" if note else whole
        if status == "not_available" and not note:
            note = "the check produced no evidence and gave no reason"
        rows.append((
            TableCell(spec.category),
            TableCell(key.split(".", 1)[1]),
            TableCell(measure or None),
            TableCell(value, format=".3g"),
            _verdict_criterion(spec),
            TableCell(status=status),
            _verdict_note_cell(note),
        ))
    statuses = [row[5].status for row in rows]
    overall = str(aggregate_status(statuses))
    counts = ", ".join(f"{statuses.count(status)} {status.upper()}" for status in _VERDICT_ORDER)
    shot = f" #{pulse}" if pulse not in (None, "") else ""
    time_text = f"t = {time_value * 1e3:.2f} ms" if np.isfinite(time_value) else "time not stored"
    return Table(
        columns=(
            TableColumn("Category"),
            TableColumn("Check"),
            TableColumn("Measure"),
            TableColumn("Value", kind="value"),
            TableColumn("Criterion"),
            TableColumn("Status", kind="status"),
            TableColumn("Note"),
        ),
        rows=tuple(rows),
        title=options.get("title") or (
            f"Equilibrium validation{shot} — {time_text} (slice {index + 1} of {total}, {reason})"
        ),
        caption=(
            f"overall {overall.upper()} (aggregate of {len(rows)} checks, continuity over the "
            f"whole IDS): {counts}"
        ),
        notes=(
            "NOT_AVAILABLE: the evidence was never produced; INDETERMINATE: produced but "
            "not deciding. Neither is a pass.",
            "Statuses from vaft.validation.equilibrium.validate_equilibrium at this slice; "
            "continuity from verify_continuity over all stored slices.",
        ),
        missing="",
    )


_SUMMARY_READS = RECIPES["equilibrium_overview"].reads

#: The IDS the equilibrium checks read: the slice itself, the measured
#: diamagnetic flux, and the kinetic profiles and Thomson channels of the
#: independent checks.  The builder hands validate_equilibrium a private copy
#: of just these (it deep-copies its argument whole).
_VALIDATION_ROOTS = ("equilibrium", "magnetics", "core_profiles", "thomson_scattering", "dataset_description")
_VALIDATION_READS = (*_SUMMARY_READS, *_VALIDATION_ROOTS)

RECIPES["equilibrium_table_summary"] = CallableRecipe(
    builder=_build_equilibrium_table_summary,
    description="The selected equilibrium slice's global quantities as a table.",
    reads=_SUMMARY_READS,
    backend=OMAS_BOUND,
    reason=RECIPES["equilibrium_overview"].reason,
    time_axis=OWN_TIME,
)
RECIPES["equilibrium_text_summary"] = CallableRecipe(
    builder=_build_equilibrium_text_summary,
    description="The selected equilibrium slice as a text summary: slice, globals, shape.",
    reads=(*_SUMMARY_READS, "equilibrium.code.name"),
    backend=OMAS_BOUND,
    reason=RECIPES["equilibrium_overview"].reason,
    time_axis=OWN_TIME,
)
RECIPES["equilibrium_table_fit_quality"] = CallableRecipe(
    builder=_build_equilibrium_table_fit_quality,
    description="EFIT goodness of fit at one slice, per constraint family, as a table.",
    reads=_recipes._EFIT_QUALITY_READS,
    backend=NEUTRAL,
    time_axis="equilibrium.time_slice",
)
RECIPES["equilibrium_table_validation"] = CallableRecipe(
    builder=_build_equilibrium_table_validation,
    description="Every registered validation check's verdict at one equilibrium slice, as a table.",
    reads=_VALIDATION_READS,
    backend=OMAS_BOUND,
    reason=(
        "vaft.validation.equilibrium.validate_equilibrium deep-copies the ODS and runs "
        "vaft.omas.process_wrapper.compute_virial_equilibrium_quantities_ods on the copy"
    ),
    time_axis=OWN_TIME,
)


# ---------------------------------------------------------------------------
# The plasma-free vacuum benchmark as tables (issue #190, roadmap #1242 C3)
# ---------------------------------------------------------------------------

def _score_cell(value: Any) -> TableCell:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return TableCell(None)
    return TableCell(number, format=".3g")


def _flagged_note(entry: Mapping[str, Any]) -> str:
    fraction = entry.get("fraction")
    share = f" for {float(fraction):.2f} of its scorable samples" if fraction is not None else ""
    return (
        f"{str(entry.get('reason', 'flagged')).replace('_', ' ')}{share}; "
        "kept out of the scored medians"
    )


def _yes_no(value: Any) -> str:
    return "yes" if value else "no"


def _build_magnetics_table_vacuum_benchmark(ods: Any, **options: Any) -> Table:
    """The benchmark's per-channel scores of one shot, one row per scored channel.

    One row per ``metrics.channels`` entry of
    :func:`~vaft.validation.vacuum_benchmark.run_benchmark_case`, in its
    order: channel, family, kind, status (evaluated, flagged or excluded --
    :func:`~vaft.plot.backend.recipes.benchmark_channel_status`), the eddy
    improvement, normalized residual, correlation and wall authority, and the
    reason for a flagged or excluded row.  The numbers are the benchmark's;
    no threshold is applied and no row is graded.
    """
    case = _recipes.vacuum_benchmark_case(ods, options)
    rows = case["metrics"]["channels"]
    flagged = {entry["channel"]: entry for entry in case["channels"]["flagged"]}
    cells = []
    for position, row in enumerate(rows, start=1):
        status = _recipes.benchmark_channel_status(row, flagged)
        if status == "flagged":
            note = _flagged_note(flagged[row["name"]])
        elif status == "excluded":
            note = str(row.get("reason") or "excluded by the benchmark, no reason given")
        else:
            note = ""
        cells.append((
            TableCell(position),
            TableCell(_recipes.channel_short_name(row["name"])),
            TableCell(str(row["family"])),
            TableCell(str(row["kind"])),
            TableCell(status),
            _score_cell(row.get("improvement")),
            _score_cell(row.get("normalized_residual")),
            _score_cell(row.get("correlation")),
            _score_cell(row.get("wall_authority")),
            _verdict_note_cell(note),
        ))

    statuses = [row[4].value for row in cells]
    counts = ", ".join(
        f"{statuses.count(status)} {status}" for status in ("evaluated", "flagged", "excluded")
    )
    pulse = _get(ods, "dataset_description.data_entry.pulse", "")
    shot = f" — shot {pulse}" if pulse not in (None, "") else ""
    solver = case["solver"]
    drive = case["coil_drive"]
    fraction = drive.get("coil_drive_fraction")
    scored = case["metrics"]["summary"]["scored"]
    caption = (
        f"{case['case_type']} case, {_recipes.vacuum_benchmark_window_text(case)}; "
        f"solver history {_recipes._ms(solver.get('available_history'))} ms of "
        f"{_recipes._ms(solver.get('required_history'))} ms required "
        f"({solver.get('n_tau', float('nan')):g} × slowest wall time constant "
        f"{_recipes._ms(solver.get('slowest_wall_time_constant'))} ms; sufficient: "
        f"{_yes_no(solver.get('sufficient'))}); coil drive "
        f"{'—' if fraction is None else f'{fraction:.2f}'} of the shot peak inside the window "
        f"(sufficiently driven: {_yes_no(drive.get('sufficiently_driven'))}); "
        f"resistance scale {case['static_model']['resistance_scale']:g}. "
        f"Scored medians over {scored['count']} channels (flagged channels and channels "
        f"with an undefined wall authority left out): improvement "
        f"{scored['improvement']['median']:.3g}, normalized residual "
        f"{scored['normalized_residual']['median']:.3g}, correlation "
        f"{scored['correlation']['median']:.3g}."
    )
    notes = [
        "No thresholds and no verdict (#190): the scores are reported, not graded. "
        "Improvement = 1 − RMS(measured − (coil+eddy)) / RMS(measured − coil): 1 is "
        "perfect, 0 means the wall term added nothing, negative that it made agreement worse.",
        "flagged: evaluated, but the probe contradicts its own array "
        "(vaft.validation.vacuum_benchmark.array_contradictions) -- a sensor finding, kept "
        "out of the scored medians; excluded: too few usable samples to score.",
        "Channel names drop the instrument prefix ("
        + ", ".join(f"'{prefix}'" for prefix in _recipes.CHANNEL_NAME_PREFIXES) + ").",
    ]
    if drive.get("reason"):
        notes.append(f"coil drive: {drive['reason']}")
    return Table(
        columns=(
            TableColumn("#", kind="value"),
            TableColumn("Channel"),
            TableColumn("Family"),
            TableColumn("Kind"),
            TableColumn("Status"),
            TableColumn("Improvement", kind="value"),
            TableColumn("Normalized residual", kind="value"),
            TableColumn("Correlation", kind="value"),
            TableColumn("Wall authority", kind="value"),
            TableColumn("Note"),
        ),
        rows=tuple(cells),
        title=options.get("title") or f"Vacuum benchmark{shot} ({len(cells)} channels: {counts})",
        caption=caption,
        notes=tuple(notes),
        missing="",
    )


#: The aggregate's cross-tabs, in the order the table lists them.
_AGGREGATE_AXES = (
    ("case", "by_case"),
    ("family", "by_family"),
    ("excitation", "by_excitation"),
    ("machine era", "by_machine_era"),
)


def _unique_labels(labels: list[str]) -> list[str]:
    """``labels`` made distinct: a repeat gets the next free `` (n)`` suffix.

    A suffix is taken only when no label -- one given or one already made --
    uses it, so ``["1 (2)", "1", "1"]`` becomes ``["1 (2)", "1", "1 (3)"]``.
    """
    given = set(labels)
    used: set[str] = set()
    unique = []
    for label in labels:
        candidate, count = label, 1
        while candidate in used or (candidate != label and candidate in given):
            count += 1
            candidate = f"{label} ({count})"
        used.add(candidate)
        unique.append(candidate)
    return unique


def _build_magnetics_table_vacuum_benchmark_aggregate(
    entries: Sequence[tuple[str, Any]], **options: Any
) -> Table:
    """The benchmark across entries: every cross-tab of :func:`aggregate_benchmark`.

    Each entry is run through
    :func:`~vaft.validation.vacuum_benchmark.run_benchmark_case` as
    ``magnetics_table_vacuum_benchmark`` runs it, and the cases go to
    :func:`~vaft.validation.vacuum_benchmark.aggregate_benchmark` with the
    entry's label as their case name.  An entry that cannot supply a case --
    one missing a path the plot requires, or one the benchmark refuses (a
    :class:`~vaft.validation.vacuum_benchmark.BenchmarkError`, no usable
    magnetic channel, no ``em_coupling`` to solve the wall from: every such
    refusal is a ``ValueError``) -- is a case row carrying the reason, never
    dropped; so is a case whose every channel was excluded, which
    :func:`aggregate_benchmark` holds no row for.  Rows: by case, by family,
    by PF excitation, by machine era.
    """
    from vaft.validation.vacuum_benchmark import aggregate_benchmark

    _recipes.vacuum_benchmark_options(options)  # refuse a bad option before any solve
    labels = _unique_labels([str(label) for label, _ods in entries])
    cases: list[dict[str, Any]] = []
    failed: list[tuple[str, str]] = []
    for label, (_entry_label, ods) in zip(labels, entries):
        # An entry without the circuits a case is solved from (a kinetic-only
        # or a magnetics-only sample) is listed, as a shot whose plasma-free
        # stretch cannot certify a case is.
        absent = _recipes.missing_required_path(ods, "magnetics_table_vacuum_benchmark_aggregate")
        if absent is not None:
            failed.append((label, f"the entry carries no {absent}"))
            continue
        try:
            case = _recipes.vacuum_benchmark_case(ods, options)
        except ValueError as error:
            # BenchmarkError, VacuumMagneticsError (no usable magnetic channel)
            # and CouplingGeometryMismatch (no em_coupling) all subclass it;
            # the options were refused above, before any entry was run.
            failed.append((label, str(error) or type(error).__name__))
            continue
        # The aggregate names a case by its "shot"; the entry label is that
        # name here, so a row reads as the entry the caller passed.
        cases.append({**case, "shot": label})
    aggregate = aggregate_benchmark(cases)
    # aggregate_benchmark rows only evaluated channels, so a case whose every
    # channel was excluded has no by_case entry; it is listed with its count.
    scored_labels = set(aggregate.get("by_case", {}))
    unscored = [
        (case["shot"], len(case["channels"]["excluded"]))
        for case in cases if case["shot"] not in scored_labels
    ]

    columns = (
        TableColumn("Group"),
        TableColumn("Value"),
        TableColumn("Cases", kind="value"),
        TableColumn("Channels", kind="value"),
        TableColumn("Median improvement", kind="value"),
        TableColumn("Median normalized residual", kind="value"),
        TableColumn("Median correlation", kind="value"),
        TableColumn("Worst channel"),
        TableColumn("Note"),
    )
    failed_rows = [
        (TableCell("case"), TableCell(label), TableCell(None), TableCell(None), TableCell(None),
         TableCell(None), TableCell(None), TableCell(None), _verdict_note_cell(note))
        for label, note in [
            (label, f"no evaluated channel: {excluded} excluded") for label, excluded in unscored
        ] + [(label, f"no benchmark case: {reason}") for label, reason in failed]
    ]
    n_scored = len(cases) - len(unscored)
    title = options.get("title") or (
        f"Vacuum benchmark across {len(entries)} entr{'y' if len(entries) == 1 else 'ies'} "
        f"({n_scored} case{'' if n_scored == 1 else 's'} scored, {len(unscored)} with no "
        f"evaluated channel, {len(failed)} without a case)"
    )
    notes = (
        "No thresholds and no verdict (#190): the medians are reported, not graded; "
        "which axis a poor result concentrates on is the diagnosis "
        "(vaft.validation.vacuum_benchmark.aggregate_benchmark).",
        "Cross-tab rows count every evaluated channel, flagged and undriven ones included; "
        "the summary medians leave flagged probes and undriven cases out.",
        "Channel names drop the instrument prefix ("
        + ", ".join(f"'{prefix}'" for prefix in _recipes.CHANNEL_NAME_PREFIXES) + ").",
    )
    if aggregate.get("status") == "empty":
        return Table(
            columns=columns,
            rows=tuple(failed_rows),
            title=title,
            caption=f"aggregate empty: {aggregate['reason']} ({aggregate['case_count']} cases)",
            notes=notes,
            missing="",
        )

    undriven = set(aggregate["undriven_cases"])
    flagged = aggregate["flagged_channels"]
    rows = []
    for group, key in _AGGREGATE_AXES:
        for value, stats in aggregate[key].items():
            note = []
            if group == "case":
                if value in undriven:
                    note.append("not sufficiently driven: left out of the summary medians")
                if value in flagged:
                    names = ", ".join(_recipes.channel_short_name(name) for name in flagged[value])
                    note.append(f"flagged: {names}")
            rows.append((
                TableCell(group),
                TableCell(str(value)),
                TableCell(int(stats["cases"])),
                TableCell(int(stats["channels"])),
                _score_cell(stats["median_improvement"]),
                _score_cell(stats["median_normalized_residual"]),
                _score_cell(stats["median_correlation"]),
                TableCell(_recipes.channel_short_name(stats["worst_channel"])),
                _verdict_note_cell("; ".join(note)),
            ))
        if group == "case":
            rows.extend(failed_rows)

    summary = aggregate["summary"]
    flagged_text = "; ".join(
        f"{case}: {', '.join(_recipes.channel_short_name(name) for name in names)}"
        for case, names in flagged.items()
    ) or "none"
    caption = (
        f"{aggregate['case_count']} cases, {aggregate['channel_rows']} channel rows; "
        f"undriven cases: {', '.join(aggregate['undriven_cases']) or 'none'}; "
        f"flagged channels: {flagged_text}. "
        f"Summary over {summary['driven_channel_rows']} driven, unflagged channel rows: "
        f"median improvement {summary['median_improvement']:.3g}, improved fraction "
        f"{summary['improved_fraction']:.3g}, median normalized residual "
        f"{summary['median_normalized_residual']:.3g}, median wall authority "
        f"{summary['median_wall_authority']:.3g}."
    )
    return Table(
        columns=columns,
        rows=tuple(rows),
        title=title,
        caption=caption,
        notes=notes,
        missing="",
    )


RECIPES["magnetics_table_vacuum_benchmark"] = CallableRecipe(
    builder=_build_magnetics_table_vacuum_benchmark,
    description="The plasma-free vacuum benchmark's per-channel scores of one shot, as a table.",
    reads=_recipes._VACUUM_BENCHMARK_READS,
    backend=OMAS_BOUND,
    reason=RECIPES["magnetics_overview_vacuum_benchmark"].reason,
)
RECIPES["magnetics_table_vacuum_benchmark_aggregate"] = CallableRecipe(
    builder=_build_magnetics_table_vacuum_benchmark_aggregate,
    description="The plasma-free vacuum benchmark across entries: medians by case, family, "
                "PF excitation and machine era, as a table.",
    reads=_recipes._VACUUM_BENCHMARK_READS,
    backend=OMAS_BOUND,
    reason=RECIPES["magnetics_overview_vacuum_benchmark"].reason,
    multi_entry=True,
)
