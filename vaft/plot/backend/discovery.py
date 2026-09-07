"""Instance-aware plot discovery, shared by every namespace (issue #262).

:mod:`vaft.plot.discovery` knows what this build of VAFT can plot; this module
knows what a particular input can plot, and how.  It fills the capability
fields of each :class:`~vaft.plot.discovery.PlotCapability` from two places:

* the recipe table -- the display unit a plot uses, the layouts its builder
  accepts, the analysis method a transformed view implements, the subjects an
  overview composes; and
* the input itself -- whether the required data is present, how many channels
  carry data and how many of those are flagged, which physical regions and
  representatives the family has, whether uncertainty and validity metadata
  sit beside the signal.

Every instance-level answer is computed by the same helper the adapters run
(:func:`missing_required_path`, :func:`_channel_has_data`, :func:`_resolve_preset`,
:func:`_validity_of` ...) through the shared path accessor, so discovery
cannot disagree with rendering for any data model: a plot reported available
is one ``render_entries`` will not refuse for missing data, and the channel
counts are the ones ``selection=`` resolves.  Discovery reads policies; it
decides none.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from vaft.plot import discovery as _core
from vaft.validation.validity import is_condemned, record_from_mask
from vaft.plot.discovery import PlotCapability, PlotCatalog, with_capabilities
from vaft.plot.display import (
    allowed_units,
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

from .recipes import (
    CAMERA_OVERLAYS,
    ChannelProfileRecipe,
    SPECTROGRAM_METHODS,
    SPECTROGRAM_PARAMETERS,
    _coordinate_options,
    abscissa_options,
    _has,
    CAMERA_PROJECTIONS,
    RECIPES,
    SYNTHETIC_CONSTRAINTS,
    FieldRecipe,
    LineRecipe,
    PanelRecipe,
    PowerSpectrumRecipe,
    ProfileRecipe,
    SpectrogramRecipe,
    _channel_carries_signal,
    _channel_has_data,
    _channel_identifiers,
    _channel_positions,
    _container_of,
    _count,
    _get,
    _resolve_preset,
    _uncertainty_of,
    _validity_of,
    converts_for_builder,
    diagnoses_itself,
    has_synthetic_values,
    missing_required_path,
)

__all__ = ["describe_by_ids", "describe_entries", "INTERACTION", "INTERACTION_ENTRY_POINTS", "OVERVIEW_CONTENTS", "ANALYSIS_METHODS"]

#: Interaction modes a plot offers (issue #261 sections 14-17).  A static
#: summary is the baseline; a time-navigable entry point appears beside it.
INTERACTION: dict[str, tuple[str, ...]] = {
    "equilibrium_overview": ("static", "time-navigable"),
}

#: The public entry point behind an interaction mode that is not a view.
INTERACTION_ENTRY_POINTS: dict[str, str] = {
    "time-navigable": "plot_equilibrium_interactive()",
    "controls": "plot_<name>(..., interactive=True)",
}

#: What a composite built by code (not a PanelRecipe) draws, for discovery's
#: ``overview:`` note (issue #262 section 13).
OVERVIEW_CONTENTS: dict[str, tuple[str, ...]] = {
    "equilibrium_overview": ("poloidal flux", "pressure", "q", "global quantities"),
}

#: What the transformed views compute.  Spectrograms are short-time Fourier
#: transforms (:func:`vaft.process.mirnov_spectrogram`); spectra are Welch
#: power spectral densities (:func:`vaft.process.fluctuation.compute_psd`).
#: Only methods that exist are listed -- discovery never advertises a
#: capability that would raise.
def _analysis_methods() -> dict[str, tuple[str, ...]]:
    """The analysis each plot offers, by plot name (issue #484).

    Keyed by name rather than by recipe kind because the spectrograms no
    longer share one transform: the two path-driven ones and the
    interferometer's own builder all take ``method=``.
    """
    methods: dict[str, tuple[str, ...]] = {}
    for name, recipe in RECIPES.items():
        if isinstance(recipe, SpectrogramRecipe) or name.endswith("_spectrogram"):
            methods[name] = SPECTROGRAM_METHODS
        elif isinstance(recipe, PowerSpectrumRecipe):
            methods[name] = ("Welch PSD",)
    return methods


#: Plot name -> the analyses it offers.
ANALYSIS_METHODS: dict[str, tuple[str, ...]] = _analysis_methods()



def describe_entries(
    entries: Sequence[tuple[str, Any]] | None,
    shots: Sequence[str] = (),
    *,
    query: str | None = None,
    detail: bool = False,
    available_only: bool | None = None,
    **filters: Any,
) -> PlotCatalog:
    """The catalog of plots, evaluated against ``entries`` when given.

    ``entries`` is ``None`` for the registry-level catalog (identity plus what
    the recipes declare: units, layouts, methods, overview members) and a
    namespace's ``(label, object)`` sequence otherwise; ``shots`` are the
    entries' shot labels for the heading.  With entries, each record also says
    whether the input can draw it and, for multi-channel plots, what the
    selection policy finds there.  ``available_only`` defaults to ``True``
    with entries -- the catalog then holds exactly the plots the input can
    draw -- and ``False`` keeps the unavailable ones with their reasons.
    """
    base = _core.catalog(query=query, detail=detail, **filters)
    records = [_declare(record) for record in base]
    if entries is None:
        return base.with_records(records)
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


def describe_by_ids(
    present: Sequence[str],
    source: str,
    *,
    query: str | None = None,
    detail: bool = False,
    available_only: bool | None = None,
    **filters: Any,
) -> PlotCatalog:
    """The catalog judged at IDS level: which IDS a shot holds, nothing read.

    A plot is available when every IDS it needs is among ``present`` -- a
    composite when any member's are -- and the reason names the missing IDS.
    This is what a database shot answers before anything is downloaded; it
    is deliberately optimistic about leaves (an IDS may exist with the leaf a
    plot wants empty), and it reports no channel facts, which need a loaded
    object.
    """
    base = _core.catalog(query=query, detail=detail, **filters)
    present_set = set(present)
    records = []
    for record in base:
        record = _declare(record)
        missing = missing_required_ids(present_set, record.name)
        available = missing is None
        reason = "" if available else f"requires IDS {missing}"
        records.append(with_capabilities(record, available=available, reason=reason))
    if available_only is None or available_only:
        records = [record for record in records if record.available]
    return PlotCatalog(records, source=source, query=query, detail=detail, available_only=False)


def missing_required_ids(present: set[str], name: str) -> str | None:
    """The first IDS plot ``name`` needs that ``present`` lacks, or ``None``."""
    recipe = RECIPES.get(name)
    if isinstance(recipe, PanelRecipe):
        members = [missing_required_ids(present, member) for member in recipe.members]
        if any(missing is None for missing in members):
            return None
        return " or ".join(m for m in members if m) or None
    spec = get_spec(name)
    if not spec.required_paths:
        # The same rule missing_required_path applies to a plot that declares
        # IDS but no paths: its builder treats them as what it may draw, and
        # any one present is enough.
        if not spec.ids or any(root in present for root in spec.ids):
            return None
        return " or ".join(spec.ids)
    for root in dict.fromkeys(path.split(".")[0] for path in spec.required_paths):
        if root not in present:
            return root
    return None


def _source_label(shots: Sequence[str]) -> str:
    labels = [f"#{shot}" if str(shot).strip() else "ODS" for shot in shots]
    return ", ".join(labels) if labels else "ODS"


# ---------------------------------------------------------------------------
# Recipe-declared capabilities (no ODS needed)
# ---------------------------------------------------------------------------


#: Plots whose value unit is a poloidal flux: ``None`` means "the stored
#: convention of the input"; the vacuum map is always full weber (Green's
#: functions), whatever the equilibrium declares.
PSI_FIELD_CONVENTIONS: dict[str, str | None] = {
    "equilibrium_field_psi": None,
    "equilibrium_field_psi_vacuum": "Wb",
    "equilibrium_overview": None,
}


def _declare(record: PlotCapability) -> PlotCapability:
    recipe = RECIPES.get(record.name)
    updates: dict[str, Any] = {}
    unit = getattr(recipe, "y_unit", None)
    if isinstance(recipe, (LineRecipe, ProfileRecipe)):
        updates["display"] = _display_block(record, unit or "")
    if isinstance(recipe, ProfileRecipe):
        options = _coordinate_options(recipe)
        updates["coordinates"] = {
            "default": recipe.default_coordinate, "options": options, "declared": options,
        }
    if isinstance(recipe, ChannelProfileRecipe):
        updates["display"] = _display_block(record, unit or "")
        updates["coordinates"] = {
            "default": recipe.default_coordinate,
            "options": tuple(recipe.coordinates),
            "declared": tuple(recipe.coordinates),
        }
    elif record.name in PSI_FIELD_CONVENTIONS:
        # The psi maps take units= (issue #478); the default unit follows the
        # stored convention, which only an input can tell -- see _evaluate.
        updates["display"] = {
            "unit": None,
            "units": allowed_units("magnetic_flux", "equilibrium"),
            "notation": "auto",
            "convention": PSI_FIELD_CONVENTIONS[record.name],
        }
    if isinstance(recipe, LineRecipe):
        options = abscissa_options(recipe)
        updates["abscissa"] = {"default": "time", "options": options, "declared": options}
        # Only the line-series builder takes layout= (issue #260); grouped
        # needs a radial split, which an ODS decides -- see _evaluate.
        updates["layouts"] = (
            ("overlay", "subplots", "grouped") if recipe.index == "channel" else ("overlay",)
        )
    methods = ANALYSIS_METHODS.get(record.name)
    if methods:
        updates["analysis_methods"] = methods
        if methods == SPECTROGRAM_METHODS:
            updates["analysis"] = {
                "default": SPECTROGRAM_METHODS[0],
                "methods": {name: SPECTROGRAM_PARAMETERS[name] for name in methods},
            }
    if isinstance(recipe, PanelRecipe):
        updates["overview_members"] = _member_subjects(recipe)
        # A composite is drawable by a backend only when every member is.
        from vaft.plot.backends import render_backends_for

        members = [get_spec(member) for member in recipe.members]
        updates["backends"] = tuple(
            name for name in record.backends
            if all(name in render_backends_for(spec) for spec in members)
        )
    elif record.name in OVERVIEW_CONTENTS:
        updates["overview_members"] = OVERVIEW_CONTENTS[record.name]
    if record.name in SYNTHETIC_CONSTRAINTS:
        updates["synthetic"] = {"overlay": "equilibrium"}
    if record.name == "camera_visible_image":
        updates["overlays"] = CAMERA_OVERLAYS
        updates["projection"] = {"methods": CAMERA_PROJECTIONS}
    if record.name in INTERACTION:
        updates["interaction"] = INTERACTION[record.name]
        updates["interaction_entry_points"] = {
            mode: INTERACTION_ENTRY_POINTS[mode]
            for mode in INTERACTION[record.name] if mode in INTERACTION_ENTRY_POINTS
        }
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
    units = allowed_units(quantity, record.subject) if quantity else (display.unit,)
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


def describe_one(name: str, entries: Sequence[tuple[str, Any]]) -> PlotCapability:
    """The capability record of one plot, evaluated against ``entries``.

    What ``describe_entries`` does for the whole catalog, for the single
    record a control layer needs (issue #480); the unavailable record is
    returned with its reason rather than dropped.
    """
    base = _core.catalog(status=None)
    record = next((record for record in base if record.name == name), None)
    if record is None:
        raise KeyError(f"no plot named {name!r}")
    return _evaluate(_declare(record), entries)


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
        if any(converts_for_builder(obj, record.name) for _, obj in entries):
            reason = "checked at render time; converted per IDS for this input"
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
            updates["abscissa"] = _abscissa_block(record, recipe, ods)
        elif isinstance(recipe, ChannelProfileRecipe):
            updates.update(_channel_facts(record, recipe, ods))
            updates["times"] = _times_block(ods, recipe)
        elif isinstance(recipe, ProfileRecipe):
            # Profiles take the same sign policy (issue #307); the catalog has
            # to say which default a profile applies, or a caller cannot know
            # whether the curve it gets back was flipped.
            updates["orientation"] = _orientation_block(recipe)
            if record.coordinates and recipe.index == "time_slice":
                updates["coordinates"] = _coordinates_block(record, recipe, ods)
        if _takes_time_slice(record.name):
            updates["slices"] = _slices_block(ods)
        if record.name in PSI_FIELD_CONVENTIONS:
            from .convention import psi_convention

            convention = PSI_FIELD_CONVENTIONS[record.name] or psi_convention(ods)
            display = resolve_display(convention, subject="equilibrium")
            updates["display"] = {
                **record.display,
                "unit": display.unit,
                "convention": convention,
            }
        if record.synthetic:
            updates["synthetic"] = {
                **record.synthetic,
                "available": has_synthetic_values(ods, record.name),
            }
        if record.projection:
            updates["projection"] = {**record.projection, **_projection_state(ods)}
        # A plot with something to change offers ``interactive=True`` (issue #480).
        from vaft.plot.controls import controls_for

        evaluated = with_capabilities(record, **updates)
        offered = controls_for(evaluated)
        if offered:
            updates["interaction"] = tuple(dict.fromkeys((*record.interaction, "controls")))
            updates["interaction_entry_points"] = {
                **record.interaction_entry_points,
                "controls": f"{record.function.removesuffix('()')}(..., interactive=True)",
            }
            updates["controls"] = tuple(c.name for c in offered)
    return with_capabilities(record, **updates)


def _coordinates_block(record: PlotCapability, recipe: ProfileRecipe, ods: Any) -> dict[str, Any]:
    """The coordinates this input can resolve for a slice-indexed profile.

    Evaluated on the representative slice: a stored leaf, or the leaves a
    derivation needs (``q``/``psi`` for the toroidal ones, the 2-D map with
    the axis and the boundary outline for the radial ones, issue #479).
    """
    from .recipes import _count, resolve_time_slice

    declared = tuple(record.coordinates.get("declared") or record.coordinates.get("options") or ())
    if not _count(ods, "equilibrium.time_slice"):
        return {**record.coordinates, "options": ()}
    try:
        index = int(resolve_time_slice(ods)[0])
    except Exception:
        index = 0
    base = f"equilibrium.time_slice.{index}.profiles_1d"
    has = lambda leaf: _has(ods, f"{base}.{leaf}")  # noqa: E731
    radial = (has("r_inboard") and has("r_outboard")) or (
        _has(ods, f"equilibrium.time_slice.{index}.profiles_2d.0.psi")
        and _has(ods, f"equilibrium.time_slice.{index}.global_quantities.magnetic_axis.r")
        and _has(ods, f"equilibrium.time_slice.{index}.boundary.outline.r")
        and has("psi")
    )
    supported = {
        "psi_norm": has("psi"),
        "rho_tor_norm": has("psi") and (has("q") or has("rho_tor_norm")),
        "sqrt_phi_norm": has("psi") and (has("phi") or has("q")),
        "r_major": radial,
        "r_minor": radial,
    }
    options = tuple(name for name in declared if supported.get(name, True))
    default = record.coordinates.get("default")
    if default not in options and options:
        default = options[0]
    return {"default": default, "options": options, "declared": declared}


def _takes_time_slice(name: str) -> bool:
    """Whether ``name`` draws one stored equilibrium slice (``time_slice=``).

    A profile indexed by slice and a 2-D map do; a time history gathers every
    slice into one trace and takes no ``time_slice=`` (``LineRecipe`` with
    ``index="time_slice"`` is that history, not a selection), so the check is
    on the recipe's kind, never on the text of its paths.
    """
    recipe = RECIPES.get(name)
    if name in PSI_FIELD_CONVENTIONS:
        return True
    if isinstance(recipe, ProfileRecipe):
        return recipe.index == "time_slice"
    if isinstance(recipe, FieldRecipe):
        return "{i}" in recipe.value_path
    return False


def _slices_block(ods: Any) -> dict[str, Any]:
    """The stored equilibrium slices: how many, which are usable, their times."""
    from .recipes import _count, _get, _usable_slices, resolve_time_slice

    total = _count(ods, "equilibrium.time_slice")
    times = []
    for index in range(total):
        raw = _get(ods, f"equilibrium.time_slice.{index}.time")
        try:
            times.append(float(np.asarray(raw, dtype=float).ravel()[0]))
        except (IndexError, TypeError, ValueError):
            times.append(float("nan"))
    usable = tuple(int(i) for i in _usable_slices(ods)) if total else ()
    try:
        selected = int(resolve_time_slice(ods)[0]) if total else None
    except Exception:
        selected = None
    return {"total": total, "usable": usable, "times": tuple(times), "selected": selected}


def _projection_state(ods: Any) -> dict[str, Any]:
    """Whether the calibrated projection is available for this shot, and why not."""
    from vaft.omas.process_wrapper import camera_projection_for

    shot = _get(ods, "dataset_description.data_entry.pulse")
    if shot in (None, ""):
        return {"available": False, "reason": "no pulse number to look a pose up by"}
    try:
        camera_projection_for(int(shot))
    except (FileNotFoundError, TypeError, ValueError) as exc:
        return {"available": False, "reason": str(exc).splitlines()[0]}
    return {"available": True}


def _line_facts(record: PlotCapability, recipe: LineRecipe, ods: Any) -> dict[str, Any]:
    """Channel, layout and metadata facts for one line-series plot."""
    facts: dict[str, Any] = {"orientation": _orientation_block(recipe)}
    if recipe.index != "channel":
        code, mask = _validity_of(ods, recipe.y_path)
        facts["validity"] = _validity_block(
            present=code is not None, flagged=int(is_condemned(record_from_mask(code, mask)))
        )
        facts["uncertainty"] = _uncertainty_block(_uncertainty_of(ods, recipe.y_path) is not None)
        return facts

    facts.update(_channel_facts(record, recipe, ods))
    return facts


def _channel_facts(record: PlotCapability, recipe: Any, ods: Any) -> dict[str, Any]:
    """Channel, layout and metadata facts of a channel-indexed recipe.

    Shared by the time histories (``LineRecipe``) and the spatial plots
    (``ChannelProfileRecipe``, issue #486): the same family, the same split.
    """
    facts: dict[str, Any] = {}
    container = _container_of(recipe.y_path, "{i}")
    total = _count(ods, container)
    candidates = (recipe.y_path,) + tuple(recipe.fallback_y_paths)
    with_data = [i for i in range(total) if _channel_has_data(ods, candidates, i)]
    verdicts = {i: _validity_of(ods, recipe.y_path, i) for i in with_data}
    codes = {i: code for i, (code, _mask) in verdicts.items()}
    # "Nothing usable to draw" is the same reading ``Series.is_invalid_channel``
    # applies when the trace is drawn: the per-sample mask decides when stored.
    flagged = [i for i, (code, mask) in verdicts.items() if is_condemned(record_from_mask(code, mask))]
    channels: dict[str, Any] = {
        "total": total,
        "with_data": len(with_data),
        "usable": len(with_data) - len(flagged),
        "flagged": len(flagged),
    }
    r_values, z_values = _channel_positions(ods, container, total)
    # The divider is the whole family's (that is what grouped infers it from);
    # the counts are of the channels the default selection draws -- flagged
    # valid and carrying a signal -- because those are the traces grouped
    # actually places (``active`` in vaft.plot.selection).
    active = [i for i in with_data if i not in flagged and _channel_carries_signal(ods, candidates, i)]
    channels["active"] = len(active)
    split = radial_divider(r_values)
    layouts: tuple[str, ...] = ("overlay", "subplots")
    if split:
        regions = classify_regions(r_values[active], split=split)
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
    if isinstance(recipe, LineRecipe):
        facts["layouts"] = layouts
    facts["validity"] = _validity_block(
        present=any(code is not None for code in codes.values()), flagged=len(flagged)
    )
    facts["uncertainty"] = _uncertainty_block(
        any(_uncertainty_of(ods, recipe.y_path, i) is not None for i in with_data)
    )
    return facts


def _abscissa_block(record: PlotCapability, recipe: LineRecipe, ods: Any) -> dict[str, Any]:
    """The abscissae this input can actually supply (issue #481).

    ``time`` and ``index`` always can be; a declared sibling needs its own
    leaf, so a shot without one is not offered a control that would silently
    fall back to the index.
    """
    declared = tuple(record.abscissa.get("declared") or abscissa_options(recipe))
    available = []
    for name in declared:
        entry = next((a for a in recipe.abscissae if a.name == name), None)
        if entry is None or _abscissa_is_stored(ods, entry):
            available.append(name)
    return {"default": "time", "options": tuple(available), "declared": declared}


def _abscissa_is_stored(ods: Any, entry: Any) -> bool:
    """Whether this input holds a declared sibling abscissa anywhere.

    The builder gathers a slice-indexed sibling across every slice and fills
    what is missing, so one slice without the leaf does not make the abscissa
    unavailable -- discovery must not refuse what rendering would draw.
    """
    from .recipes import _container_of

    for path in entry.paths:
        if "{i}" not in path:
            if _has(ods, path):
                return True
            continue
        total = _count(ods, _container_of(path, "{i}"))
        if any(_has(ods, path.format(i=index)) for index in range(total)):
            return True
    return False


def _times_block(ods: Any, recipe: Any) -> dict[str, Any]:
    """The stored time axis a spatial plot samples: start, stop, count."""
    from .recipes import _first_time

    container = _container_of(recipe.y_path, "{i}")
    for index in range(_count(ods, container)):
        axis = _first_time(ods, recipe.time_paths, i=index)
        if axis is not None and axis.size:
            return {"start": float(axis.min()), "stop": float(axis.max()), "count": int(axis.size)}
    return {}


def _validity_block(*, present: bool, flagged: int) -> dict[str, Any]:
    if not present:
        return {}
    return {"available": True, "flagged": flagged, "modes": VALIDITY_MODES}


def _orientation_block(recipe: LineRecipe | ProfileRecipe) -> dict[str, Any]:
    """The display sign policy a line or profile plot applies by default (#307)."""
    from vaft.plot.backend.recipes import ORIENTATIONS

    return {"default": recipe.orientation, "options": list(ORIENTATIONS)}


def _uncertainty_block(present: bool) -> dict[str, Any]:
    if not present:
        return {}
    return {"available": True, "modes": UNCERTAINTY_MODES}
