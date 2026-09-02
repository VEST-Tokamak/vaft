"""Semantic discovery of what can be plotted (issue #262).

:func:`vaft.plot.available_plots` used to return a flat tuple of registry rows
-- ``name``, ``domain``, ``required_paths`` -- which answers "which renderers
are registered?" but not "what can I look at, and how?".  This module keeps
the structure and layers a semantic view on top of it.

Two objects do the work:

:class:`PlotCapability`
    one canonical plot as a record: its ``subject / view / [quantity]``
    identity from :mod:`vaft.plot.taxonomy`, the public function that draws it,
    and the capabilities it offers -- display units, layouts, analysis methods,
    channel information, availability.  The developer-facing registry fields
    (``ids``, ``required_paths``, the view model) are kept but move out of the
    default rendering.  A record still answers ``record["name"]`` so every
    caller that iterated the old rows keeps working unchanged.

:class:`PlotCatalog`
    a sequence of records that prints as a tree grouped by subject, view and
    quantity.  Printing never discards the structure: iterate the catalog for
    the records, or call :meth:`PlotCatalog.rows` for the plain dictionaries.

This module reads the taxonomy and the registry and nothing else: it never
imports OMAS.  Instance-level facts -- whether an ODS actually holds the data,
how many channels are usable, which regions exist -- are filled in by
:mod:`vaft.omas.discovery`, which reuses the same extraction helpers the
adapters run so discovery cannot disagree with rendering.  Discovery only
*reports* the policies decided in issues #251 (taxonomy), #256 (display),
#259 (selection) and #260 (layout); it decides nothing of its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields, replace
from typing import Any, Iterator, Mapping, Sequence

from . import taxonomy
from .registry import VIEWS, PlotSpec, specs

__all__ = [
    "PlotCapability",
    "PlotCatalog",
    "QueryMatch",
    "catalog",
    "match_query",
    "normalise_query",
]

#: Registry fields kept for developers; hidden from the default tree (#262 §17).
DEVELOPER_KEYS = (
    "name",
    "domain",
    "ids",
    "required_paths",
    "optional_paths",
    "model",
    "status",
    "description",
)

_WORD_BREAK = re.compile(r"[\s\-/]+")


@dataclass(frozen=True)
class PlotCapability:
    """One canonical plot and everything discovery knows about it."""

    # -- identity (issue #251) ------------------------------------------------
    name: str
    subject: str
    view: str
    quantity: str
    aliases: tuple[str, ...]
    function: str
    # -- developer block (registry metadata, hidden by default) ---------------
    domain: str
    ids: tuple[str, ...]
    required_paths: tuple[str, ...]
    optional_paths: tuple[str, ...]
    model: str
    status: str
    description: str
    # -- capabilities declared by the recipe (filled by vaft.omas.discovery) --
    #: Display unit the plot uses by default, the alternatives it accepts and
    #: the notation, from the display policy (#256).  Empty when the quantity
    #: has no conversion table.
    display: Mapping[str, Any] = field(default_factory=dict)
    #: Layouts this plot's builder accepts (#260).  ``grouped`` appears only
    #: when the family has a radial split, which needs an ODS to know.
    layouts: tuple[str, ...] = ()
    #: Analysis methods a transformed view implements (spectra, spectrograms).
    analysis_methods: tuple[str, ...] = ()
    #: Subjects an overview composes, in panel order.
    overview_members: tuple[str, ...] = ()
    #: Sibling plots that overlay something onto this one (camera views).
    overlays: tuple[str, ...] = ()
    #: A reconstruction's prediction that can be overlaid on this measurement
    #: (issue #261 section 9): ``{"overlay": "equilibrium"}`` when supported,
    #: plus ``"available"`` once an ODS says whether it holds finite values.
    synthetic: Mapping[str, Any] = field(default_factory=dict)
    # -- instance-level (only with an ODS) ------------------------------------
    #: ``None`` without an ODS; otherwise whether the input can draw it, by the
    #: same test :func:`vaft.omas.plotting.render` applies.
    available: bool | None = None
    #: Machine-readable reason when unavailable: the first required path the
    #: input lacks, or the composite members it lacks.
    reason: str = ""
    #: Per-entry availability for multi-shot input, keyed by entry label.
    entries: Mapping[str, bool] = field(default_factory=dict)
    #: Channel facts for multi-channel plots, from the selection policy (#259).
    channels: Mapping[str, Any] = field(default_factory=dict)
    #: Uncertainty / validity metadata present beside the signal (#256).
    uncertainty: Mapping[str, Any] = field(default_factory=dict)
    validity: Mapping[str, Any] = field(default_factory=dict)
    # -- reserved for issue #261; empty and unprinted until it lands ----------
    sources: Mapping[str, Any] = field(default_factory=dict)
    interaction: tuple[str, ...] = ()
    #: Public function behind an interaction mode that is not itself a plot.
    interaction_entry_points: Mapping[str, str] = field(default_factory=dict)
    projection: Mapping[str, Any] = field(default_factory=dict)

    # The old ``available_plots`` rows were plain dictionaries; keep that
    # access so ``row["name"]`` and friends still work on a record.
    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __eq__(self, other: object) -> bool:
        # A record equals the flat row it replaces, so code that compared rows
        # to literal dictionaries keeps its answer.
        if isinstance(other, Mapping):
            return self.row() == dict(other) or self.as_dict() == dict(other)
        if isinstance(other, PlotCapability):
            return self.as_dict() == other.as_dict()
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.name, self.subject, self.view, self.quantity))

    def keys(self) -> tuple[str, ...]:
        return tuple(f.name for f in fields(self))

    def as_dict(self) -> dict[str, Any]:
        """Every field, as a plain dictionary."""
        return {key: getattr(self, key) for key in self.keys()}

    def row(self) -> dict[str, Any]:
        """The developer row in the shape the flat listing always had."""
        return {key: getattr(self, key) for key in _LEGACY_ROW_KEYS}

    @property
    def identity(self) -> str:
        """``subject / view`` or ``subject / view / quantity``."""
        parts = [self.subject, self.view] + ([self.quantity] if self.quantity else [])
        return " / ".join(parts)

    @property
    def heading(self) -> str:
        """The canonical subject with its strict aliases: ``plasma_current [ip, I_p]``."""
        return _subject_heading(self.subject, self.aliases)


#: Keys of the flat rows returned before #262, in their historical order.
_LEGACY_ROW_KEYS = (
    "name",
    "domain",
    "subject",
    "view",
    "quantity",
    "model",
    "ids",
    "required_paths",
    "description",
    "status",
)


def _subject_heading(subject: str, aliases: Sequence[str]) -> str:
    return f"{subject} [{', '.join(aliases)}]" if aliases else subject


def capability_for(spec: PlotSpec) -> PlotCapability:
    """The registry-level record for ``spec``: identity plus developer block."""
    aliases: tuple[str, ...] = ()
    if spec.subject in taxonomy.SUBJECTS:
        aliases = taxonomy.SUBJECTS[spec.subject].aliases
    return PlotCapability(
        name=spec.name,
        subject=spec.subject,
        view=spec.view,
        quantity=spec.quantity,
        aliases=aliases,
        function=f"plot_{spec.stem}()",
        domain=spec.domain,
        ids=tuple(spec.ids),
        required_paths=tuple(spec.required_paths),
        optional_paths=tuple(spec.optional_paths),
        model=spec.model.__name__,
        status=spec.status,
        description=spec.description,
        overlays=_overlays_for(spec),
    )


def _overlays_for(spec: PlotSpec) -> tuple[str, ...]:
    """Sibling image plots that overlay something onto this frame.

    ``camera_visible / image / frame`` is the bare image; its siblings under the
    same subject and view (``efit_overlay``, ``field_line``) draw something over
    it.  This is read from the registry, not declared, so a new overlay
    renderer is discovered the moment it registers.
    """
    if spec.view != "image" or spec.quantity != "frame":
        return ()
    return tuple(
        sibling.quantity
        for sibling in specs(subject=spec.subject, view="image")
        if sibling.quantity and sibling.quantity != "frame"
    )


# ---------------------------------------------------------------------------
# Query resolution -- strict aliases only (#262 §3, §20, §21)
# ---------------------------------------------------------------------------


def normalise_query(term: str) -> str:
    """``"Electron Density"`` -> ``"electron_density"``; identity for canonical names."""
    return _WORD_BREAK.sub("_", term.strip()).lower()


@dataclass(frozen=True)
class QueryMatch:
    """What a query resolved to: subjects, and the quantities that narrow them."""

    subjects: tuple[str, ...] = ()
    #: Empty when the query named a subject; otherwise the canonical quantities
    #: it named, and only plots of those quantities match.
    quantities: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.subjects)

    def accepts(self, spec: PlotSpec) -> bool:
        if spec.subject not in self.subjects:
            return False
        return not self.quantities or spec.quantity in self.quantities


def match_query(term: str) -> QueryMatch:
    """Resolve a query to canonical subjects, strictly.

    A canonical subject name or one of the taxonomy's registered aliases
    (``ip`` -> ``plasma_current``) names that subject.  A quantity family or
    quantity alias (``beta`` -> ``beta_n``, ``beta_p``, ``beta_t``) names the
    subjects that plot those quantities, narrowed to those plots.  Otherwise a
    whole-word prefix of a canonical subject name (``flux`` -> ``flux_loop``,
    not ``diamagnetic_flux``).  Descriptions are never searched, so
    related-but-distinct concepts do not match: ``Rogowski coil`` is not a
    ``plasma_current`` alias and ``line_radiation`` is not a
    ``spectrometer_uv`` alias (issue #262).  An unknown query matches nothing.
    """
    query = normalise_query(term)
    if not query:
        return QueryMatch()
    for candidate in (query, term.strip()):
        try:
            return QueryMatch(subjects=(taxonomy.resolve_subject(candidate).name,))
        except KeyError:
            pass
    quantities: set[str] = set()
    for candidate in (query, term.strip()):
        try:
            family = taxonomy.resolve_family(candidate)
        except KeyError:
            pass
        else:
            # A family names its members and its own composite plot
            # (equilibrium / time / beta draws beta_n, beta_p and beta_t).
            quantities.update(family.members)
            quantities.add(family.name)
        try:
            quantities.add(taxonomy.resolve_quantity(candidate))
        except KeyError:
            pass
    if quantities:
        return _by_quantities(quantities)
    prefixed = tuple(
        name
        for name in taxonomy.subject_names()
        if name == query or name.startswith(query + "_")
    )
    if prefixed:
        return QueryMatch(subjects=prefixed)
    # A canonical quantity spelled exactly ("pressure", "voltage") is an
    # identity, not a synonym, so it names the plots that carry it.
    if any(spec.quantity == query for spec in specs()):
        return _by_quantities({query})
    return QueryMatch()


def _by_quantities(quantities: set[str]) -> QueryMatch:
    subjects = sorted({spec.subject for spec in specs() if spec.quantity in quantities})
    return QueryMatch(subjects=tuple(subjects), quantities=tuple(sorted(quantities)))


# ---------------------------------------------------------------------------
# The catalog
# ---------------------------------------------------------------------------


class PlotCatalog(Sequence[PlotCapability]):
    """A sequence of :class:`PlotCapability` that prints as a semantic tree.

    Indexing, iteration and ``len`` behave like the tuple of rows that
    :func:`vaft.plot.available_plots` returned before issue #262, and each
    element still supports ``row["name"]``; the tree is what ``print`` and the
    REPL show.  ``source`` names the input the catalog was evaluated against
    (``"#39915"``), or is empty for the registry alone.
    """

    def __init__(
        self,
        records: Sequence[PlotCapability],
        *,
        source: str = "",
        query: str | None = None,
        detail: bool = False,
        available_only: bool = False,
    ) -> None:
        self._records = tuple(records)
        self.source = source
        self.query = query
        self.detail = detail
        self.available_only = available_only

    # -- sequence protocol ---------------------------------------------------
    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index):  # type: ignore[override]
        if isinstance(index, slice):
            return PlotCatalog(
                self._records[index],
                source=self.source,
                query=self.query,
                detail=self.detail,
                available_only=self.available_only,
            )
        return self._records[index]

    def __iter__(self) -> Iterator[PlotCapability]:
        return iter(self._records)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PlotCatalog):
            return self._records == other._records
        if isinstance(other, (tuple, list)):
            return self._records == tuple(other)
        return NotImplemented

    def __add__(self, other):
        # The flat listing was a tuple; concatenation still yields one.
        return self.rows() + tuple(other)

    def __radd__(self, other):
        return tuple(other) + self.rows()

    def __hash__(self) -> int:  # pragma: no cover - sequences are rarely hashed
        return hash(self._records)

    # -- structured access ----------------------------------------------------
    def rows(self) -> tuple[dict[str, Any], ...]:
        """The plain developer rows, exactly as the flat listing returned them."""
        return tuple(record.row() for record in self._records)

    def names(self) -> tuple[str, ...]:
        return tuple(record.name for record in self._records)

    def find(self, name: str) -> PlotCapability:
        for record in self._records:
            if record.name == name:
                return record
        raise KeyError(f"no plot named {name!r} in this catalog")

    def with_records(self, records: Sequence[PlotCapability]) -> "PlotCatalog":
        return PlotCatalog(
            records,
            source=self.source,
            query=self.query,
            detail=self.detail,
            available_only=self.available_only,
        )

    # -- rendering ------------------------------------------------------------
    def tree(self, *, detail: bool | None = None, available_only: bool | None = None) -> str:
        """Render the catalog as a ``subject / view / quantity`` tree."""
        detail = self.detail if detail is None else detail
        available_only = self.available_only if available_only is None else available_only
        return render_tree(
            self._records,
            source=self.source,
            query=self.query,
            detail=detail,
            available_only=available_only,
        )

    def __str__(self) -> str:
        return self.tree()

    def __repr__(self) -> str:
        return self.tree()

    def _repr_pretty_(self, printer: Any, cycle: bool) -> None:  # IPython
        printer.text(self.tree())


def catalog(
    *,
    query: str | None = None,
    domain: str | None = None,
    subject: str | None = None,
    view: str | None = None,
    model: type | None = None,
    status: str | None = "canonical",
    detail: bool = False,
) -> PlotCatalog:
    """The registry-level catalog: what this build of VAFT can plot.

    ``query`` resolves through the strict alias registry (see
    :func:`match_query`); the other filters are the developer filters the flat
    listing always took.  Nothing here consults an ODS -- see
    :func:`vaft.omas.available_plots` for that.
    """
    selected = specs(domain=domain, subject=subject, view=view, model=model, status=status)
    if query is not None:
        match = match_query(query)
        selected = tuple(spec for spec in selected if match.accepts(spec))
    records = tuple(capability_for(spec) for spec in selected)
    return PlotCatalog(records, query=query, detail=detail)


# ---------------------------------------------------------------------------
# Tree rendering
# ---------------------------------------------------------------------------


def _view_order(view: str) -> int:
    return VIEWS.index(view) if view in VIEWS else len(VIEWS)


def _group(records: Sequence[PlotCapability]):
    """``{subject: {view: [records]}}`` with views in canonical order."""
    tree: dict[str, dict[str, list[PlotCapability]]] = {}
    for record in records:
        tree.setdefault(record.subject, {}).setdefault(record.view, []).append(record)
    ordered = {}
    for subject in sorted(tree):
        views = tree[subject]
        ordered[subject] = {
            view: sorted(views[view], key=lambda r: r.name)
            for view in sorted(views, key=_view_order)
        }
    return ordered


def _compact_notes(record: PlotCapability) -> list[str]:
    """The one-line capability summary shown beneath a plot in compact mode."""
    notes: list[str] = []
    if record.available is False:
        notes.append(f"unavailable — {record.reason}" if record.reason else "unavailable")
        return notes
    if record.display.get("unit"):
        note = f"unit: {record.display['unit']}"
        if record.display.get("notation") not in (None, "", "auto"):
            note += f" ({record.display['notation']})"
        notes.append(note)
    channels = record.channels
    if channels:
        total = channels.get("total")
        usable = channels.get("usable")
        if total is not None:
            text = f"channels: {usable} / {total} usable" if usable is not None else f"channels: {total}"
            regions = [name for name, count in channels.get("regions", {}).items() if count]
            if regions:
                text += " · regions: " + ", ".join(regions)
            reps = [name for name, index in channels.get("representatives", {}).items() if index is not None]
            if reps:
                text += " · representatives: " + ", ".join(reps)
            notes.append(text)
    if record.layouts:
        layouts = list(record.layouts)
        # Without an ODS nobody knows whether the family has a radial split,
        # which is what grouped needs; say so rather than promise it.
        if record.available is None and "grouped" in layouts:
            layouts[layouts.index("grouped")] = "grouped (with a radial split)"
        notes.append("layout: " + " | ".join(layouts))
    if record.analysis_methods:
        notes.append("methods: " + " | ".join(record.analysis_methods))
    if record.overview_members:
        notes.append("overview: " + " · ".join(record.overview_members))
    if record.overlays:
        notes.append("overlays: " + ", ".join(record.overlays))
    if record.synthetic:
        notes.append(_synthetic_note(record.synthetic))
    if record.interaction:
        notes.append("interaction: " + " | ".join(record.interaction))
    flags = []
    if record.uncertainty.get("available"):
        flags.append("uncertainty")
    if record.validity.get("available"):
        flagged = record.validity.get("flagged")
        flags.append(f"validity ({flagged} flagged)" if flagged else "validity")
    if flags:
        notes.append("metadata: " + ", ".join(flags))
    return notes


def _synthetic_note(synthetic: Mapping[str, Any]) -> str:
    """``synthetic overlay: equilibrium``, qualified by the ODS when evaluated."""
    note = f"synthetic overlay: {synthetic.get('overlay', '')}"
    if synthetic.get("available") is False:
        note += " (supported, unavailable in this ODS)"
    return note


def _detail_lines(record: PlotCapability) -> list[str]:
    """Everything, key by key, for ``detail=True``."""
    lines = [f"identity: {record.identity}"]
    if record.aliases:
        lines.append("aliases: " + ", ".join(record.aliases))
    lines.append(f"function: vaft.omas.{record.function}")
    if record.available is not None:
        state = "available" if record.available else "unavailable"
        lines.append(f"availability: {state}" + (f" — {record.reason}" if record.reason else ""))
        if len(record.entries) > 1:
            per_entry = ", ".join(f"{k}: {'yes' if v else 'no'}" for k, v in record.entries.items())
            lines.append(f"entries: {per_entry}")
    if record.display:
        lines.append(f"unit: {record.display.get('unit', '')}")
        if record.display.get("units"):
            lines.append("units: " + " | ".join(record.display["units"]))
        if record.display.get("notation"):
            lines.append(f"notation: {record.display['notation']}")
    if record.channels:
        c = record.channels
        if "total" in c:
            lines.append(f"channels: {c.get('usable', '?')} / {c['total']} usable")
        for key in ("regions", "representatives"):
            if c.get(key):
                lines.append(f"{key}: " + ", ".join(f"{k} -> {v}" for k, v in c[key].items()))
        if c.get("identifiers"):
            lines.append("identifiers: " + ", ".join(map(str, c["identifiers"])))
        if c.get("positions"):
            lines.append("positions: " + ", ".join(c["positions"]))
    if record.layouts:
        lines.append("layouts: " + " | ".join(record.layouts))
    if record.analysis_methods:
        lines.append("methods: " + " | ".join(record.analysis_methods))
    if record.overview_members:
        lines.append("includes: " + ", ".join(record.overview_members))
    if record.overlays:
        lines.append("overlays: " + ", ".join(record.overlays))
    if record.synthetic:
        lines.append(_synthetic_note(record.synthetic))
    for key in ("uncertainty", "validity"):
        block = getattr(record, key)
        if block:
            state = "available" if block.get("available") else "absent"
            extra = f" ({block['flagged']} flagged)" if block.get("flagged") else ""
            lines.append(f"{key}: {state}{extra}")
            if block.get("modes"):
                lines.append(f"{key} handling: " + " | ".join(block["modes"]))
    if record.sources:
        lines.append("sources: " + ", ".join(record.sources))
    if record.interaction:
        lines.append("interaction: " + " | ".join(record.interaction))
        for mode, function in record.interaction_entry_points.items():
            lines.append(f"{mode}: vaft.omas.{function}")
    if record.projection:
        lines.append("projection: " + ", ".join(f"{k}: {v}" for k, v in record.projection.items()))
    lines.append(f"domain: {record.domain}")
    lines.append("ids: " + (", ".join(record.ids) or "—"))
    lines.append("required: " + (", ".join(record.required_paths) or "—"))
    if record.optional_paths:
        lines.append("optional: " + ", ".join(record.optional_paths))
    lines.append(f"model: {record.model}")
    lines.append(f"status: {record.status}")
    if record.description:
        lines.append(f"description: {record.description}")
    return lines


def render_tree(
    records: Sequence[PlotCapability],
    *,
    source: str = "",
    query: str | None = None,
    detail: bool = False,
    available_only: bool = False,
) -> str:
    """Render records as the ``subject / view / [quantity]`` tree."""
    if available_only:
        records = [r for r in records if r.available is not False]
    header = "Available plots"
    header += f" — {source}" if source else " — registry"
    if query is not None:
        header += f" — query {query!r}"
    if not records:
        return header + ("\n\n(no plots match)" if query is not None else "\n\n(nothing to plot)")
    out = [header, ""]
    for subject, views in _group(records).items():
        any_record = next(iter(next(iter(views.values()))))
        out.append(_subject_heading(subject, any_record.aliases))
        view_items = list(views.items())
        for v_index, (view, members) in enumerate(view_items):
            last_view = v_index == len(view_items) - 1
            branch = "└─ " if last_view else "├─ "
            child_indent = "   " if last_view else "│  "
            single = len(members) == 1 and not members[0].quantity
            if single:
                record = members[0]
                out.append(f"{branch}{view}  {record.function}")
                out.extend(_leaf_body(record, child_indent, detail))
                continue
            out.append(f"{branch}{view}")
            for q_index, record in enumerate(members):
                last = q_index == len(members) - 1
                q_branch = "└─ " if last else "├─ "
                q_indent = child_indent + ("   " if last else "│  ")
                label = record.quantity or "—"
                out.append(f"{child_indent}{q_branch}{label}  {record.function}")
                out.extend(_leaf_body(record, q_indent, detail))
        out.append("")
    return "\n".join(out).rstrip()


def _leaf_body(record: PlotCapability, indent: str, detail: bool) -> list[str]:
    lines = _detail_lines(record) if detail else _compact_notes(record)
    return [f"{indent}{line}" for line in lines]


def with_capabilities(record: PlotCapability, **updates: Any) -> PlotCapability:
    """A copy of ``record`` with the given capability fields replaced."""
    return replace(record, **updates)
