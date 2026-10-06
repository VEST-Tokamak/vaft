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

from typing import Any

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
