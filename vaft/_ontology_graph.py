"""The scientific ontology of VAFT, generated from its registries (#1702).

``python -m vaft._ontology_graph --output docs/_data/ontology_graph.yml``
writes what ``/reference/ontology/`` renders: the third graph beside the
source dependency graph (what imports what) and the pipeline lineage (what
produced what).  This one answers *what a thing means*: which concept a plot
visualizes, which diagnostic measures a quantity, which Data Dictionary path
represents it, which check assesses it, which convention a code uses.

It is a generated view, not a second scientific database.  Every node and
edge comes from a registry that already owns the fact, and records which:

* :mod:`vaft.plot.taxonomy` -- the controlled vocabulary: canonical subjects
  and their kinds, strict aliases, canonical quantities, and families (a
  family is membership, never synonymy: ``beta_n`` is not ``beta_p``);
* :mod:`vaft.plot.registry` -- each plot's subject, quantity, IDS and the
  Data Dictionary paths it reads (through :mod:`vaft.plot.backend.dd`, which
  supplies canonical spellings, units, coordinates and lifecycle);
* :mod:`vaft.machine_mapping.registry` -- each VEST diagnostic, its IDS, what
  it measures and derives, and the mapping function;
* :mod:`vaft.validation.registry` -- named checks and their providers;
* :mod:`vaft.data.cocos` -- the COCOS convention each code or format uses;
* :mod:`vaft._ecosystem` (#1648) -- external codes, their adapters and the
  IDS they are mapped into;
* :mod:`vaft.formula._taxonomy` (#1626) -- which quantity a formula reduces to
  which, with each formula's ``Reduction`` section as facets;
* the ``Semantics`` sections of :mod:`vaft.formula` and :mod:`vaft.process`
  docstrings -- the quantities a function consumes and produces, in the vocabulary;
* :mod:`vaft.validation.orderings` (#1627) -- the physical models' approximation
  contracts, the ordering quantities they assume and the kernels computing them.

**Identity is strict.**  A term becomes a concept only through
:func:`resolve_term`, which accepts a canonical name or a registered alias
and nothing else.  A term no registry can resolve is not canonicalized: it is
listed in the snapshot's ``unresolved`` audit with where it came from, so a
gap is visible instead of silently becoming a new concept.  Nothing is read
from docstring prose and nothing from Python imports.

Edges use a small, typed vocabulary (:data:`RELATIONS`); node ids are
namespaced (``concept:``, ``diagnostic:``, ``dd:``, ...) so names the
registries legitimately reuse never collide.  Generation is offline: the Data
Dictionary comes with omas, and no database, service or solver is touched.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

SCHEMA_VERSION = 1
_GENERATOR = "python -m vaft._ontology_graph --output docs/_data/ontology_graph.yml"
_PACKAGE = Path(__file__).resolve().parent
_ROOT = _PACKAGE.parent

#: Node kinds, and the views each appears in.
KINDS = {
    "concept": "a scientific quantity, reconstruction, model or composite subject",
    "concept_family": "a family of distinct concepts (membership, not synonymy)",
    "diagnostic": "a measurement system",
    "machine": "a machine component or system",
    "code": "an external scientific code",
    "data_format": "a data model or file format with a fixed convention",
    "plot": "a canonical VAFT plot",
    "api": "a VAFT function: a mapping, a validation provider, a code adapter, a formula or a process",
    "ids": "an IMAS IDS",
    "dd_path": "a Data Dictionary path",
    "validation": "a named validation check",
    "convention": "a COCOS convention",
    "model": "a physical model's approximation contract: the orderings it assumes",
    "ordering_quantity": "a dimensionless ordering quantity a model's validity is judged by",
}

#: The relation vocabulary. Every edge is one of these and nothing else.
RELATIONS = {
    "member_of": "a concept belongs to a family",
    "measures": "a diagnostic measures a quantity directly",
    "derives": "a diagnostic's processing derives a quantity",
    "represented_by": "a concept or diagnostic is stored in an IDS or a Data Dictionary path",
    "part_of": "a Data Dictionary path belongs to its IDS; a diagnostic system belongs to a larger one",
    "mapped_by": "a diagnostic is mapped into IMAS by a VAFT function",
    "visualized_by": "a concept is drawn by a canonical plot",
    "reads": "a plot reads a Data Dictionary path",
    "assessed_by": "a concept is assessed by a named validation check",
    "provided_by": "a check or an ordering quantity is computed by a VAFT function",
    "uses_convention": "a code or data format uses a COCOS convention",
    "implemented_by": "an external code is integrated through a VAFT adapter",
    "produces": "an external code's result is mapped into an IDS; a formula or process computes a quantity",
    "consumes": "a formula or process takes a quantity as input",
    "derived_from": "a quantity is reduced from another by a step VAFT performs outside a formula",
    "assumes": "a physical model is valid only where an ordering quantity is small or large",
}

#: The id namespace of each kind, where it differs from the kind's name.
NAMESPACES = {"dd_path": "dd"}

#: Which node kinds each view shows. Concepts is the compact default.
VIEWS = {
    "concepts": ("concept", "concept_family", "diagnostic", "machine", "code"),
    "representations": ("concept", "diagnostic", "machine", "ids", "dd_path", "api"),
    "implementations": ("concept", "diagnostic", "machine", "code", "plot", "api", "ids"),
    "assessment": ("concept", "diagnostic", "validation", "api", "convention", "code", "data_format",
                   "model", "ordering_quantity"),
}

#: Taxonomy subject kind -> ontology node kind.
_SUBJECT_KIND = {
    "quantity": "concept",
    "reconstruction": "concept",
    "model": "concept",
    "composite": "concept",
    "diagnostic": "diagnostic",
    "machine": "machine",
    "code": "code",
}

#: COCOS registry names that are data formats or models rather than codes.
_FORMATS = {"imas", "omas", "geqdsk"}


class OntologyError(ValueError):
    """Two registries disagree in a way the ontology cannot represent."""


def _relative(path: str | Path) -> str:
    return Path(path).resolve().relative_to(_ROOT).as_posix()


# --------------------------------------------------------------------------
# strict identity
# --------------------------------------------------------------------------


def node_id_for_subject(name: str) -> str:
    from vaft.plot import taxonomy

    subject = taxonomy.SUBJECTS[name]
    return f"{_SUBJECT_KIND[subject.kind]}:{name}"


def resolve_term(term: str, *, quantities_only: bool = False) -> Optional[str]:
    """The node id of ``term`` if a registry names it, else ``None`` -- never a guess.

    A canonical subject or one of its strict aliases resolves to that subject;
    a canonical quantity or one of its aliases to that quantity.  Case,
    spacing and near-spellings are not folded: ``"Te"`` resolves only if the
    taxonomy registers it.  ``quantities_only`` refuses subjects that are
    not quantities (a reconstruction, a model, a composite overview).
    """
    from vaft.plot import taxonomy

    if not term:
        return None
    try:
        subject = taxonomy.resolve_subject(term)
    except KeyError:
        subject = None
    if subject is not None:
        # "what a diagnostic measures" must be a quantity, not a reconstruction or an overview
        if quantities_only and subject.kind != "quantity":
            return None
        return node_id_for_subject(subject.name)
    try:
        return f"concept:{taxonomy.resolve_quantity(term)}"
    except KeyError:
        return None


def canonical_dd(path: str) -> str:
    """A Data Dictionary path in canonical spelling, every index as ``(:)``."""
    return re.sub(r"\(\d+\)", "(:)", path)


# --------------------------------------------------------------------------
# the graph
# --------------------------------------------------------------------------


class _Graph:
    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.edges: dict[tuple[str, str, str], dict] = {}
        self.unresolved: dict[tuple[str, str], dict] = {}

    def node(self, node_id: str, kind: str, label: str, origin: str, **facets) -> dict:
        if kind not in KINDS:
            raise OntologyError(f"unknown node kind {kind!r}")
        node = self.nodes.get(node_id)
        if node is None:
            node = self.nodes[node_id] = {"id": node_id, "kind": kind, "label": label, "origins": [], "facets": {}}
        elif node["kind"] != kind:
            raise OntologyError(f"{node_id} is both a {node['kind']} and a {kind}")
        if origin not in node["origins"]:
            node["origins"].append(origin)
        for key, value in facets.items():
            if value not in (None, "", [], ()):
                node["facets"].setdefault(key, value)
        return node

    def edge(self, source: str, target: str, kind: str, origin: str) -> None:
        if kind not in RELATIONS:
            raise OntologyError(f"unknown relation {kind!r}")
        for end in (source, target):
            if end not in self.nodes:
                raise OntologyError(f"{kind} edge names {end}, which no registry defined")
        edge = self.edges.setdefault((source, target, kind), {"source": source, "target": target, "kind": kind,
                                                              "origins": []})
        if origin not in edge["origins"]:
            edge["origins"].append(origin)

    def miss(self, term: str, origin: str, context: str) -> None:
        row = self.unresolved.setdefault((term, origin), {"term": term, "origin": origin, "contexts": []})
        if context not in row["contexts"]:
            row["contexts"].append(context)


def _api(graph: _Graph, dotted: str, origin: str, role: str) -> str:
    """An API node for a dotted function, linked to its API page when it has one."""
    from vaft import _api_catalog

    module, _, name = dotted.replace(":", ".").rpartition(".")
    function = getattr(importlib.import_module(module), name, None)
    if not callable(function):
        raise OntologyError(f"{origin} names {dotted}, which is not a callable")
    page = _api_catalog.page_for(module, _api_catalog.load_inventory()) if _api_catalog.is_public(module) else None
    source = {}
    try:
        from vaft._docstring import source_span

        span = source_span(function, _ROOT, inline=False)
        source = {"path": span["path"], "line": int(span["line"]), "end_line": int(span["end_line"])}
    except Exception:  # noqa: BLE001 - a builtin or a wrapped callable has no span
        source = {}
    node_id = f"api:{module}.{name}"
    graph.node(node_id, "api", f"{module}.{name}", origin, role=role,
               api_url=f"/reference/api/{page}/#{module}.{name}" if page else "", source=source)
    return node_id


def _taxonomy(graph: _Graph) -> None:
    from vaft.plot import taxonomy

    origin = "vaft.plot.taxonomy"
    for subject in taxonomy.SUBJECTS.values():
        graph.node(node_id_for_subject(subject.name), _SUBJECT_KIND[subject.kind], subject.name, origin,
                   concept_kind=subject.kind, aliases=sorted(subject.aliases))
    quantities: dict[str, list[str]] = {name: [] for name in taxonomy.CANONICAL_QUANTITIES}
    for alias, canonical in taxonomy.QUANTITY_ALIASES.items():
        quantities.setdefault(canonical, [])
        if alias != canonical:
            quantities[canonical].append(alias)
    for canonical, aliases in sorted(quantities.items()):
        graph.node(f"concept:{canonical}", "concept", canonical, origin, concept_kind="quantity",
                   aliases=sorted(aliases))
    for family in taxonomy.FAMILIES.values():
        family_id = f"concept_family:{family.name}"
        graph.node(family_id, "concept_family", family.name, origin, aliases=sorted(family.aliases))
        for member in family.members:
            member_id = resolve_term(member)
            if member_id is None:
                raise OntologyError(f"family {family.name} names {member}, which the taxonomy does not define")
            graph.edge(member_id, family_id, "member_of", origin)


def _ids(graph: _Graph, name: str, origin: str) -> str:
    graph.node(f"ids:{name}", "ids", name, origin)
    return f"ids:{name}"


def _dd(graph: _Graph, path: str, origin: str) -> Optional[str]:
    from vaft.plot.backend import dd

    canonical = canonical_dd(path)
    try:
        info = dd.resolve(canonical, cross_check=False)
    except KeyError:
        graph.miss(canonical, origin, "not in the Data Dictionary")
        return None
    ids = canonical.split("/", 1)[0]
    node_id = f"dd:{canonical}"
    graph.node(node_id, "dd_path", canonical, origin, ids=ids, units=info.units,
               coordinates=list(info.coordinates), data_type=info.data_type,
               lifecycle=info.lifecycle_status, dd_version=info.dd_version,
               documentation=(info.documentation or "").split("\n", 1)[0][:240])
    graph.edge(node_id, _ids(graph, ids, origin), "part_of", origin)
    return node_id


def _plots(graph: _Graph) -> None:
    import vaft.plot  # noqa: F401 - registers every renderer
    from vaft.plot import registry
    from vaft.plot.backend import dd

    origin = "vaft.plot.registry"
    for spec in sorted(registry.specs(status=None), key=lambda s: s.name):
        plot_id = f"plot:{spec.name}"
        graph.node(plot_id, "plot", spec.name, origin, view=spec.view, domain=spec.domain,
                   quantity=spec.quantity, description=spec.description,
                   plot_url=f"/reference/plot/#{spec.name}")
        subject_id = node_id_for_subject(spec.subject)
        graph.edge(subject_id, plot_id, "visualized_by", origin)
        quantity_id = resolve_term(spec.quantity, quantities_only=True) if spec.quantity else None
        if spec.quantity and quantity_id is None:
            graph.miss(spec.quantity, origin, f"quantity of {spec.name}")
        elif quantity_id and quantity_id != subject_id:
            graph.edge(quantity_id, plot_id, "visualized_by", origin)
        data_paths = []
        for path in dd.dd_paths(spec.name):
            node = _dd(graph, path.canonical, origin)
            if node is None:
                continue
            graph.edge(plot_id, node, "reads", origin)
            if path.role == "data":
                data_paths.append(node)
        # The one unambiguous representation rule: a plot of a single quantity
        # that reads exactly one quantity path shows where that quantity lives.
        if spec.quantity:
            drawn = quantity_id  # an unresolved quantity never falls back to the subject
        else:
            drawn = subject_id if subject_id.startswith("concept:") else None
        concept_kind = graph.nodes[drawn]["facets"].get("concept_kind") if drawn else None
        if drawn and concept_kind == "quantity" and len(set(data_paths)) == 1:
            graph.edge(drawn, data_paths[0], "represented_by", origin)


def _diagnostics(graph: _Graph) -> None:
    from vaft.machine_mapping.registry import load_diagnostic_registry, validate_diagnostic_registry
    from vaft.plot import taxonomy

    origin = "vaft.machine_mapping.registry"
    records = load_diagnostic_registry()
    # A record's `subject` must be a vocabulary subject; the registry loader does not
    # know the vocabulary (machine_mapping never imports vaft.plot), so check it here.
    validate_diagnostic_registry(records, subjects=taxonomy.SUBJECTS)
    # Identity is the record's declared taxonomy `subject`, never a spelling match: a
    # record that alone names its subject *is* that diagnostic; several records naming
    # one subject (magnetics.ip, magnetics.internal_probe, ...) are parts of it.
    named: dict[str, list[str]] = {}
    for record_id, record in records.items():
        if record.get("subject"):
            named.setdefault(record["subject"], []).append(record_id)
    for record_id, record in sorted(records.items()):
        subject = record.get("subject")
        parent = None
        if subject and len(named[subject]) == 1:
            node_id = node_id_for_subject(subject)
        else:
            node_id = f"diagnostic:{record_id}"
            if subject:
                parent = node_id_for_subject(subject)
        kind = node_id.split(":", 1)[0]
        graph.node(node_id, kind, record.get("name") or record_id, origin, registry_id=record_id,
                   family=record.get("family"), category=record.get("category"),
                   availability=record.get("availability"), lifecycle=record.get("lifecycle"),
                   mapping_status=record.get("mapping_status"), ids_path=record.get("ids_path"),
                   diagnostics_url="/reference/vest-diagnostics/")
        if parent:
            graph.edge(node_id, parent, "part_of", origin)
        ids = record.get("ids")
        if ids and ids != "not_developed":
            graph.edge(node_id, _ids(graph, ids, origin), "represented_by", origin)
        quantities = record.get("quantities") or {}
        for relation, group in (("measures", "measured"), ("derives", "derived")):
            for term in quantities.get(group) or []:
                target = resolve_term(term, quantities_only=True)
                if target is None:
                    graph.miss(term, origin, f"{group} by {record_id}")
                elif target != node_id:
                    graph.edge(node_id, target, relation, origin)
        mapping = record.get("mapping") or {}
        if mapping.get("module") and mapping.get("entrypoint"):
            api = _api(graph, f"{mapping['module']}.{mapping['entrypoint']}", origin, "mapping")
            graph.edge(node_id, api, "mapped_by", origin)


def _validation(graph: _Graph) -> None:
    from vaft.validation.registry import CHECKS

    origin = "vaft.validation.registry"
    for key, spec in sorted(CHECKS.items()):
        check_id = f"validation:{key}"
        graph.node(check_id, "validation", key, origin, category=spec.category, unit=spec.unit,
                   method=spec.method, measure=spec.measure,
                   tolerance=list(spec.tolerance) if spec.tolerance else None)
        graph.edge(check_id, _api(graph, spec.provider, origin, "validation provider"), "provided_by", origin)
        # a check key names what it checks: <category>.<checked thing>
        checked = key.split(".", 1)[1]
        target = resolve_term(checked)
        if target and target.split(":", 1)[0] not in {"concept", "diagnostic"}:
            target = None
        if target and target.startswith("concept:") and graph.nodes[target]["facets"].get("concept_kind") != "quantity":
            target = None
        if target is None:
            graph.miss(checked, origin, f"checked by {key}")
        else:
            graph.edge(target, check_id, "assessed_by", origin)


def _code_node(graph: _Graph, name: str, origin: str, label: str = "", **facets) -> str:
    if name in _FORMATS:
        graph.node(f"data_format:{name}", "data_format", label or name, origin, **facets)
        return f"data_format:{name}"
    resolved = resolve_term(name)
    node_id = resolved if resolved and resolved.startswith("code:") else f"code:{name}"
    graph.node(node_id, "code", label or name, origin, **facets)
    return node_id


def _conventions(graph: _Graph) -> None:
    from vaft.data import cocos

    origin = "vaft.data.cocos"
    for name in sorted(cocos.known_codes()):
        convention = cocos.convention_for(name)
        node_id = _code_node(graph, name, origin, psi_unit=convention.psi_unit,
                             convention_confirmed=bool(convention.confirmed),
                             convention_note=convention.notes)
        if convention.cocos is None:
            graph.nodes[node_id]["facets"].setdefault("cocos", "identified per file")
            continue
        convention_id = f"convention:cocos_{convention.cocos}"
        graph.node(convention_id, "convention", f"COCOS {convention.cocos}", origin)
        graph.edge(node_id, convention_id, "uses_convention", origin)


def _external_codes(graph: _Graph) -> None:
    from vaft import _ecosystem
    from vaft.database.sources import STAGE_REPLICATION

    origin = "vaft._ecosystem"
    for code in _ecosystem.EXTERNAL_CODES:
        node_id = _code_node(graph, code.id, origin, label=code.name, roles=list(code.roles), mode=code.mode,
                             maturity=code.maturity, code_url=f"/reference/external-codes/#code-{code.id}")
        graph.nodes[node_id]["label"] = code.name  # the catalog names a code, whichever registry met it first
        adapter_id = f"api:{code.adapter}"
        graph.node(adapter_id, "api", code.adapter, origin, role="code adapter",
                   api_url=f"/reference/api/code/#module-{code.adapter}")
        graph.edge(node_id, adapter_id, "implemented_by", origin)
        for entry in code.standardized:
            ids = entry.ids
            if entry.via.startswith("stage:"):
                ids = STAGE_REPLICATION[entry.via.split(":", 1)[1]].ids
            for name in ids:
                graph.edge(node_id, _ids(graph, name, origin), "produces", origin)


def _formula_api(graph: _Graph, key: str, origin: str) -> str:
    """The API node of a ``"category.name"`` formula, carrying its ``Reduction`` section."""
    from vaft.formula.catalog import describe

    spec = describe(key)
    node_id = _api(graph, f"{spec.module}.{spec.qualname.rsplit('.', 1)[-1]}", origin, "formula")
    if spec.reduction is not None:
        reduction = spec.reduction
        # the Reduction section is the formula catalog's fact, whichever registry reached the formula
        graph.node(node_id, "api", graph.nodes[node_id]["label"], "vaft.formula.catalog",
                   reduction_input=list(reduction.input),
                   reduction_output=reduction.output, reduction_kind=reduction.kind, locality=reduction.locality,
                   physical_role=reduction.role)
    return node_id


def _reductions(graph: _Graph) -> None:
    """#1626: which quantity is reduced to which, and by which formula.

    The reduction graphs name quantities by their own keys (``s_hat``,
    ``j_phi_field``).  A key becomes an edge only through the ``concept`` its
    ``Quantity`` declares, resolved by :func:`resolve_term`; a key without one
    is audited, never matched by spelling.
    """
    from vaft.formula import _taxonomy

    origin = "vaft.formula._taxonomy"

    def concept(key: str) -> Optional[str]:
        declared = _taxonomy.QUANTITIES[key].concept
        if declared is None:
            return None
        resolved = resolve_term(declared, quantities_only=True)
        if resolved is None:
            raise OntologyError(f"reduction quantity {key} declares concept {declared}, which the vocabulary lacks")
        return resolved

    for family, relations in _taxonomy.REDUCTION_FAMILIES.items():
        for relation in relations:
            target = concept(relation.target)
            sources = [(key, concept(key)) for key in relation.sources]
            step = relation.formula or relation.kind
            if target is None:
                graph.miss(relation.target, origin,
                           f"produced by {step} from {' + '.join(relation.sources)} ({family})")
            for key, source in sources:
                if source is None:
                    graph.miss(key, origin, f"reduced by {step} to {relation.target} ({family})")
            if relation.formula:
                if target is None and not any(source for _, source in sources):
                    continue  # nothing the vocabulary knows: the formula would be an island
                api = _formula_api(graph, relation.formula, origin)
                for _, source in sources:
                    if source:
                        graph.edge(api, source, "consumes", origin)
                if target:
                    graph.edge(api, target, "produces", origin)
            elif target:
                for _, source in sources:
                    if source and source != target:
                        graph.edge(target, source, "derived_from", origin)


def _semantics(graph: _Graph) -> None:
    """The ``Semantics`` sections of formulas and processes (#1702 phase 3).

    Every term must be a quantity of the vocabulary: the section is a contract,
    so an unknown term is an error here, not an audit row.
    """
    from vaft.formula.catalog import list_formulas
    from vaft.process.catalog import list_processes

    for origin, specs in (("vaft.formula.catalog", list_formulas()), ("vaft.process.catalog", list_processes())):
        for spec in specs:
            if spec.semantics is None:
                continue
            if origin == "vaft.formula.catalog":
                api = _formula_api(graph, spec.qualname, origin)  # with its Reduction facets
            else:
                api = _api(graph, f"{spec.module}.{spec.name}", origin, "process")
            for relation, terms in (("consumes", spec.semantics.consumes), ("produces", spec.semantics.produces)):
                for term in terms:
                    target = resolve_term(term, quantities_only=True)
                    if target is None:
                        raise OntologyError(f"{spec.module}.{spec.name} Semantics {relation} {term!r}, "
                                            "which is no quantity of vaft.plot.taxonomy")
                    graph.edge(api, target, relation, origin)


def _orderings(graph: _Graph) -> None:
    """#1627: approximation contracts, the ordering quantities they assume and their kernels."""
    from vaft.validation import orderings

    origin = "vaft.validation.orderings"
    for name, quantity in sorted(orderings.ORDERING_QUANTITIES.items()):
        node_id = f"ordering_quantity:{name}"
        graph.node(node_id, "ordering_quantity", name, origin, definition=quantity.definition,
                   scale=quantity.scale, scope=quantity.scope, group=quantity.group)
        for kernel in quantity.kernels:
            graph.edge(node_id, _formula_api(graph, kernel, origin), "provided_by", origin)
    for name, contract in sorted(orderings.CONTRACTS.items()):
        node_id = f"model:{name}"
        graph.node(node_id, "model", name, origin, physical_model=contract.physical_model,
                   assumptions=[f"{a.quantity} {a.ordering} ({a.scope}): {a.meaning}" for a in contract.assumptions],
                   limitations=list(contract.limitations), references=list(contract.references))
        for assumption in contract.assumptions:
            target = f"ordering_quantity:{assumption.quantity}"
            if target not in graph.nodes:
                raise OntologyError(f"contract {name} assumes {assumption.quantity}, which is no ordering quantity")
            graph.edge(node_id, target, "assumes", origin)


def _ambiguous_aliases(graph: _Graph) -> None:
    """An alias that identifies two different subjects identifies nothing; drop and audit it.

    Ids are namespaced, so an alias may share its bare spelling with a node of
    another kind (``tf`` is an alias of ``machine:tf_coil`` and the name of
    ``ids:tf``): the alias still resolves to exactly one subject.  What makes an
    alias ambiguous is resolving elsewhere -- two nodes claiming it, or the
    vocabulary resolving it to a different concept than the node that lists it.
    """
    claims: dict[str, set] = {}
    for node in graph.nodes.values():
        for alias in node["facets"].get("aliases") or []:
            claims.setdefault(alias, set()).add(node["id"])
    for node in graph.nodes.values():
        aliases = node["facets"].get("aliases") or []
        if not aliases:
            continue
        kept = []
        for alias in aliases:
            resolved = resolve_term(alias)
            others = (claims[alias] - {node["id"]}) | ({resolved} if resolved not in (None, node["id"]) else set())
            if others:
                graph.miss(alias, node["origins"][0],
                           f"alias of {node['id']} also identifies {', '.join(sorted(others))}")
            else:
                kept.append(alias)
        node["facets"]["aliases"] = kept


def _views(graph: _Graph) -> None:
    for node in graph.nodes.values():
        node["views"] = sorted(view for view, kinds in VIEWS.items() if node["kind"] in kinds)


def ontology_snapshot(provenance: Mapping[str, str] | None = None) -> dict:
    """The ontology as one deterministic, documentation-ready mapping."""
    graph = _Graph()
    _taxonomy(graph)
    _plots(graph)
    _diagnostics(graph)
    _validation(graph)
    _conventions(graph)
    _external_codes(graph)
    _reductions(graph)
    _semantics(graph)
    _orderings(graph)
    _ambiguous_aliases(graph)
    _views(graph)

    import vaft.data.cocos as cocos_module
    import vaft.formula._taxonomy as reduction_module
    import vaft.plot.taxonomy as taxonomy_module
    import vaft.validation.orderings as orderings_module
    import vaft.validation.registry as validation_module

    files = [
        Path(__file__).resolve(), Path(taxonomy_module.__file__).resolve(),
        Path(validation_module.__file__).resolve(), Path(cocos_module.__file__).resolve(),
        _ROOT / "vaft" / "machine_mapping" / "vest.yaml", _ROOT / "vaft" / "_ecosystem.py",
        _ROOT / "vaft" / "plot" / "registry.py", _ROOT / "vaft" / "plot" / "backend" / "dd.py",
        Path(reduction_module.__file__).resolve(), Path(orderings_module.__file__).resolve(),
    ]
    nodes = sorted(graph.nodes.values(), key=lambda n: n["id"])
    for node in nodes:
        node["origins"] = sorted(node["origins"])
        node["facets"] = {key: node["facets"][key] for key in sorted(node["facets"])}
    edges = sorted(graph.edges.values(), key=lambda e: (e["source"], e["target"], e["kind"]))
    for edge in edges:
        edge["origins"] = sorted(edge["origins"])
    unresolved = sorted(graph.unresolved.values(), key=lambda r: (r["origin"], r["term"]))
    snapshot: dict = {
        "schema_version": SCHEMA_VERSION,
        "generator": _GENERATOR,
        "source": [
            {"path": _relative(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in sorted(set(files), key=_relative)
        ],
        "kinds": dict(KINDS),
        "relations": dict(RELATIONS),
        "views": {view: list(kinds) for view, kinds in VIEWS.items()},
        "nodes": nodes,
        "edges": edges,
        "unresolved": unresolved,
    }
    if provenance:
        snapshot["provenance"] = {key: provenance[key] for key in sorted(provenance)}
    return snapshot


def _dump(snapshot: Mapping[str, Any]) -> str:
    import yaml

    return yaml.safe_dump(dict(snapshot), allow_unicode=True, sort_keys=False, default_flow_style=False, width=100)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export VAFT's generated scientific ontology for the documentation site.")
    parser.add_argument("--output", required=True, help="YAML destination for the snapshot")
    parser.add_argument("--provenance-commit", help="Commit the source tree was taken from, recorded in the snapshot")
    parser.add_argument("--provenance-ref", help="Ref that commit was resolved from, recorded in the snapshot")
    parser.add_argument("--check", action="store_true",
                        help="Do not write; exit 1 unless --output already holds what this tree derives")
    arguments = parser.parse_args(argv)
    try:
        if arguments.check:
            import yaml

            recorded = yaml.safe_load(Path(arguments.output).read_text(encoding="utf-8"))
            if ontology_snapshot(recorded.get("provenance")) != recorded:
                raise SystemExit(f"{arguments.output} is stale: regenerate it with {_GENERATOR}")
            return
        provenance = {
            key: value
            for key, value in (("commit", arguments.provenance_commit), ("ref", arguments.provenance_ref))
            if value
        }
        destination = Path(arguments.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(_dump(ontology_snapshot(provenance or None)), encoding="utf-8")
    except OntologyError as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":  # pragma: no cover - exercised through the module CLI
    main()
