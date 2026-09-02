"""ODS-aware plot discovery (issue #262).

:mod:`vaft.plot.discovery` knows what this build of VAFT can plot; this module
knows what a particular ODS can plot, and how.  It fills the capability fields
of each :class:`~vaft.plot.discovery.PlotCapability` from two places:

* the recipe table -- the display unit a plot uses, the layouts its builder
  accepts, the analysis method a transformed view implements, the subjects an
  overview composes; and
* the input itself -- whether the required data is present, how many channels
  carry data and how many of those are flagged, which physical regions and
  representatives the family has, whether uncertainty and validity metadata
  sit beside the signal.

Every instance-level answer is computed by the same helper the adapters run
(:func:`missing_required_path`, :func:`_channel_has_data`, :func:`_resolve_preset`,
:func:`_validity_of` ...), so discovery cannot disagree with rendering: a plot
reported available is one :func:`vaft.omas.plotting.render` will not refuse
for missing data, and the channel counts are the ones ``selection=`` resolves.
Discovery reads policies; it decides none.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from vaft.plot import discovery as _core
from vaft.plot.discovery import PlotCapability, PlotCatalog, with_capabilities
from vaft.plot.display import (
    DIMENSIONLESS_DISPLAY,
    QUANTITIES,
    channel_label,
    quantity_for_unit,
    resolve_display,
)
from vaft.plot.registry import get_spec
from vaft.plot.selection import (
    INBOARD,
    OUTBOARD,
    REPRESENTATIVE_PRESETS,
    UNCLASSIFIED,
    classify_regions,
    radial_divider,
)
from vaft.plot.style import UNCERTAINTY_MODES, VALIDITY_MODES

from ._plot_recipes import (
    RECIPES,
    SYNTHETIC_CONSTRAINTS,
    LineRecipe,
    PanelRecipe,
    PowerSpectrumRecipe,
    ProfileRecipe,
    SpectrogramRecipe,
    _channel_has_data,
    _channel_identifiers,
    _channel_positions,
    _container_of,
    _count,
    _resolve_preset,
    _uncertainty_of,
    _validity_of,
    diagnoses_itself,
    has_synthetic_values,
    missing_required_path,
    normalize_entries,
)

__all__ = ["describe"]

#: What the transformed views compute.  Spectrograms are short-time Fourier
#: transforms (:func:`vaft.process.mirnov_spectrogram`); spectra are Welch
#: power spectral densities (:func:`vaft.process.fluctuation.compute_psd`).
#: Only methods that exist are listed -- discovery never advertises a
#: capability that would raise.
ANALYSIS_METHODS: dict[type, tuple[str, ...]] = {
    SpectrogramRecipe: ("STFT",),
    PowerSpectrumRecipe: ("Welch PSD",),
}


def describe(
    source: Any = None,
    *,
    query: str | None = None,
    detail: bool = False,
    available_only: bool | None = None,
    **filters: Any,
) -> PlotCatalog:
    """The catalog of plots, evaluated against ``source`` when one is given.

    Without ``source`` the records carry the registry's identity plus what the
    recipes declare (units, layouts, methods, overview members).  With an
    ``ODS``, ``ODC`` or list, each record also says whether the input can draw
    it and, for multi-channel plots, what the selection policy finds there.

    ``available_only`` defaults to ``True`` with a source -- the catalog then
    holds exactly the plots the input can draw, which is what
    ``vaft.omas.available_plots(ods)`` has always meant -- and pass ``False``
    to keep the unavailable ones with their reasons.  It is meaningless
    without a source.
    """
    base = _core.catalog(query=query, detail=detail, **filters)
    records = [_declare(record) for record in base]
    if source is None:
        return base.with_records(records)
    entries = normalize_entries(source, label="key")
    shots = [label for label, _ in normalize_entries(source, label="shot")]
    if available_only is None:
        available_only = True
    if not entries:
        # An empty collection can draw nothing; say so rather than evaluate
        # every plot against no data.
        return PlotCatalog((), source="no entries", query=query, detail=detail)
    evaluated = [_evaluate(record, entries) for record in records]
    if available_only:
        evaluated = [record for record in evaluated if record.available]
    return PlotCatalog(
        evaluated,
        source=_source_label(shots),
        query=query,
        detail=detail,
        available_only=False,
    )


def _source_label(shots: Sequence[str]) -> str:
    labels = [f"#{shot}" if str(shot).strip() else "ODS" for shot in shots]
    return ", ".join(labels) if labels else "ODS"


# ---------------------------------------------------------------------------
# Recipe-declared capabilities (no ODS needed)
# ---------------------------------------------------------------------------


def _declare(record: PlotCapability) -> PlotCapability:
    recipe = RECIPES.get(record.name)
    updates: dict[str, Any] = {}
    unit = getattr(recipe, "y_unit", None)
    if isinstance(recipe, (LineRecipe, ProfileRecipe)):
        updates["display"] = _display_block(record, unit or "")
    if isinstance(recipe, LineRecipe):
        # Only the line-series builder takes layout= (issue #260); grouped
        # needs a radial split, which an ODS decides -- see _evaluate.
        updates["layouts"] = (
            ("overlay", "subplots", "grouped") if recipe.index == "channel" else ("overlay",)
        )
    for kind, methods in ANALYSIS_METHODS.items():
        if isinstance(recipe, kind):
            updates["analysis_methods"] = methods
    if isinstance(recipe, PanelRecipe):
        updates["overview_members"] = _member_subjects(recipe)
    if record.name in SYNTHETIC_CONSTRAINTS:
        updates["synthetic"] = {"overlay": "equilibrium"}
    return with_capabilities(record, **updates) if updates else record


def _display_block(record: PlotCapability, unit: str) -> dict[str, Any]:
    """Default unit, alternatives and notation from the display policy (#256)."""
    if not unit and (record.subject, record.quantity) not in DIMENSIONLESS_DISPLAY:
        return {}
    try:
        display = resolve_display(unit, subject=record.subject, quantity=record.quantity)
    except ValueError:
        return {}
    quantity = quantity_for_unit(unit) if unit else None
    units = tuple(QUANTITIES[quantity].units) if quantity else (display.unit,)
    return {"unit": display.unit, "units": units, "notation": display.notation}


def _member_subjects(recipe: PanelRecipe) -> tuple[str, ...]:
    subjects: list[str] = []
    for member in recipe.members:
        try:
            subject = get_spec(member).subject or member
        except KeyError:
            subject = member
        if subject not in subjects:
            subjects.append(subject)
    return tuple(subjects)


# ---------------------------------------------------------------------------
# Instance-level evaluation
# ---------------------------------------------------------------------------


def _evaluate(record: PlotCapability, entries: Sequence[tuple[str, Any]]) -> PlotCapability:
    missing = {str(label): missing_required_path(ods, record.name) for label, ods in entries}
    per_entry = {label: path is None for label, path in missing.items()}
    available = any(per_entry.values())
    reason = ""
    if not available:
        wanted = next(path for path in missing.values() if path is not None)
        recipe = RECIPES.get(record.name)
        reason = (
            f"requires {wanted}"
            if not isinstance(recipe, PanelRecipe)
            else f"none of its members are available ({wanted})"
        )
    elif diagnoses_itself(record.name):
        reason = "checked at render time"
    updates: dict[str, Any] = {
        "available": available,
        "reason": reason,
        "entries": per_entry,
    }
    if available:
        label, ods = next((label, ods) for label, ods in entries if per_entry[str(label)])
        recipe = RECIPES.get(record.name)
        if isinstance(recipe, LineRecipe):
            updates.update(_line_facts(record, recipe, ods))
        if record.synthetic:
            updates["synthetic"] = {
                **record.synthetic,
                "available": has_synthetic_values(ods, record.name),
            }
    return with_capabilities(record, **updates)


def _line_facts(record: PlotCapability, recipe: LineRecipe, ods: Any) -> dict[str, Any]:
    """Channel, layout and metadata facts for one line-series plot."""
    facts: dict[str, Any] = {}
    if recipe.index != "channel":
        code, _ = _validity_of(ods, recipe.y_path)
        facts["validity"] = _validity_block(present=code is not None, flagged=int(code is not None and code < 0))
        facts["uncertainty"] = _uncertainty_block(_uncertainty_of(ods, recipe.y_path) is not None)
        return facts

    container = _container_of(recipe.y_path, "{i}")
    total = _count(ods, container)
    candidates = (recipe.y_path,) + tuple(recipe.fallback_y_paths)
    with_data = [i for i in range(total) if _channel_has_data(ods, candidates, i)]
    codes = {i: _validity_of(ods, recipe.y_path, i)[0] for i in with_data}
    flagged = [i for i, code in codes.items() if code is not None and code < 0]
    channels: dict[str, Any] = {
        "total": total,
        "with_data": len(with_data),
        "usable": len(with_data) - len(flagged),
        "flagged": len(flagged),
    }
    r_values, z_values = _channel_positions(ods, container, total)
    # The divider is the whole family's (that is what grouped infers it from);
    # the counts are of the channels that carry data, because those are the
    # traces grouped actually places -- an empty channel builds no trace.
    split = radial_divider(r_values)
    layouts: tuple[str, ...] = ("overlay", "subplots")
    if split:
        regions = classify_regions(r_values[with_data], split=split)
        counts = {name: regions.count(name) for name in (INBOARD, OUTBOARD, UNCLASSIFIED)}
        channels["regions"] = {name: count for name, count in counts.items() if count}
        representatives: dict[str, int | None] = {}
        for term in REPRESENTATIVE_PRESETS:
            try:
                chosen = _resolve_preset(ods, container, total, term, candidates)
            except ValueError:
                chosen = None
            representatives[term] = chosen[0] if chosen else None
        channels["representatives"] = representatives
        layouts = layouts + ("grouped",)
    channels["identifiers"] = tuple(_channel_identifiers(ods, container, total))
    channels["positions"] = tuple(
        channel_label(i, r_values[i], z_values[i]) for i in range(total)
    )
    facts["channels"] = channels
    facts["layouts"] = layouts
    facts["validity"] = _validity_block(
        present=any(code is not None for code in codes.values()), flagged=len(flagged)
    )
    facts["uncertainty"] = _uncertainty_block(
        any(_uncertainty_of(ods, recipe.y_path, i) is not None for i in with_data)
    )
    return facts


def _validity_block(*, present: bool, flagged: int) -> dict[str, Any]:
    if not present:
        return {}
    return {"available": True, "flagged": flagged, "modes": VALIDITY_MODES}


def _uncertainty_block(present: bool) -> dict[str, Any]:
    if not present:
        return {}
    return {"available": True, "modes": UNCERTAINTY_MODES}
