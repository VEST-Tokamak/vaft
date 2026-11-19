"""Plasma-model guide figures, read from the formalism records.

Every node and edge here is derived from :mod:`vaft.code.formalism` (``AXES``
and the :class:`~vaft.code.formalism.PlasmaFormalism` fields) and from
:mod:`vaft.validation.model_contracts` (``ADAPTER_MODES``, ``contracts_for``,
``ROLES``): a node lists the ``adapter:mode`` ids whose record matches it,
computed rather than typed, so the figures cannot drift from the registry.
Codes that are not in VAFT are drawn hollow and labelled so
(:data:`EXTERNAL_CODES`).

``plasma_model_reduction_map``
    one starting point (Vlasov / Fokker--Planck + Maxwell) and the reductions
    to the fluid, drift-kinetic, gyrokinetic, particle, hybrid, reduced and
    equilibrium descriptions, with the VAFT modes under each;
``formalism_axes_map``
    the independent axes of the record, one column each, with three example
    runs drawn across them;
``solver_placement_map``
    every adapter mode against the main axes, then the codes not in VAFT;
``kinetic_mhd_coupling``
    kinetic-MHD as a coupling loop between a fluid bulk and a drift-kinetic
    population, with the GPEC-suite modes placed on it by their records;
``reduced_model_ancestry``
    parent theory, model reduction and computational realization of every
    reduced mode.

There is no fidelity axis: no position means "simpler" or "more accurate".
Four edge kinds are used, each with one line style (:data:`EDGE_KINDS`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

from ._concept import Box, box, connector, escape_latex
from ._geometry import _check_labels
from ._render import Diagram
from ._scene import Arrow, Label, Polyline, Scene

#: The four edge kinds of the guide, and the line style of each.
EDGE_KINDS = ("asymptotic_reduction", "model_coupling", "software_dependency", "surrogate_ancestry")
_EDGE_STYLES = {
    "asymptotic_reduction": "->,black!75,line width=1.0pt",
    "model_coupling": "<->,islandblue!85!black,line width=1.1pt",
    "software_dependency": "->,black!60,dotted,line width=1.2pt",
    "surrogate_ancestry": "->,driftred!80!black,dashed,line width=1.0pt",
}
_EDGE_NAMES = {
    "asymptotic_reduction": "asymptotic reduction",
    "model_coupling": "model coupling",
    "software_dependency": "software dependency",
    "surrogate_ancestry": "surrogate ancestry",
}
#: a code or theory with no VAFT solver
_HOLLOW = "draw=black!50,dashed,line width=0.8pt,rounded corners=3pt,fill=white"
_HOLLOW_TEXT = "concept text,text=black!60"
_GREY = "draw=black!30,line width=0.6pt,fill=black!5"


@dataclass(frozen=True)
class ExternalCode:
    """A code classified for comparison only: no VAFT run adapter. Cells mirror the guide's table."""

    code: str
    bulk: str
    model: str
    domain: str
    realization: str


#: The guide's "Not in VAFT" table (``/reference/plasma-model-guide/`` section 9), row for row.
EXTERNAL_CODES: Tuple[ExternalCode, ...] = (
    ExternalCode("GENE / GENE-X", "kinetic", "gyrokinetic; GENE-X full-f, across the separatrix",
                 "local or global; GENE-X whole volume", "continuum"),
    ExternalCode("GTC", "kinetic", "gyrokinetic ions and energetic particles (GTC-X: no primary reference identified)",
                 "global", "PIC"),
    ExternalCode("ORB5", "kinetic", "gyrokinetic, δf with control variates", "global", "PIC"),
    ExternalCode("MARS-F / MARS-K", "fluid / hybrid",
                 "resistive MHD with flow and wall; MARS-K adds drift-kinetic resonances", "whole volume", "—"),
    ExternalCode("ASCOT5", "particle", "test-particle Fokker–Planck", "whole volume",
                 "Monte Carlo, full or guiding-centre orbit"),
    ExternalCode("SIMPLE", "particle", "none (orbit integration)", "whole volume",
                 "symplectic guiding-centre integrator"),
)


#: A learned surrogate and the reduced model it reproduces. A surrogate descends from that model
#: (surrogate_ancestry), not by an asymptotic reduction from the theory.
_SURROGATES = {"gacode:tglf_nn": "gacode:tglf"}


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _modes():
    from vaft.validation.model_contracts import ADAPTER_MODES

    return list(ADAPTER_MODES)


def _record(key: str):
    from vaft.validation.model_contracts import adapter_mode

    return adapter_mode(key).formalism


def _ids(predicate: Callable) -> List[str]:
    """The ``adapter:mode`` ids whose record satisfies ``predicate``."""
    return [m.key for m in _modes() if m.formalism is not None and predicate(m.formalism)]


def _node(x: float, y: float, width: float, height: float, title: str, ids: Sequence[str] = (), *,
          detail: str = "", style: str = "concept box", role: str = "", hollow: bool = False) -> Box:
    """A box with a title, an optional detail line and the implemented ids (or "not in VAFT")."""
    lines = [f"\\textbf{{{escape_latex(title)}}}"]
    if detail:
        lines.append(f"{{\\footnotesize {detail}}}")
    if hollow:
        lines.append("{\\footnotesize\\itshape not in VAFT}")
    elif ids:
        lines.append("{\\scriptsize\\ttfamily " + ", ".join(escape_latex(i) for i in ids) + "}")
    return box(x, y, width, height, "\\\\ ".join(lines), style=_HOLLOW if hollow else style,
               text_style=_HOLLOW_TEXT if hollow else "concept text", role=role, latex=True)


_UNICODE_TEX = {"δ": "$\\delta$", "—": "---", "–": "--"}


def _tex(text: str) -> str:
    """Plain text for a LaTeX label: escaped, with the few Unicode characters the registry uses spelled in TeX."""
    out = escape_latex(text)
    for char, tex in _UNICODE_TEX.items():
        out = out.replace(char, tex)
    return out


def _edge(start: Box, end: Box, kind: str, *, role: str = "") -> Arrow:
    return connector(start, end, style=_EDGE_STYLES[kind], role=role or f"edge:{kind}")


def _legend(kinds: Sequence[str], x: float, y: float, *, step: float = 0.55) -> List:
    """One sample line per edge kind used, top to bottom from ``(x, y)``."""
    items: List = []
    for i, kind in enumerate(kinds):
        yy = y - i * step
        items.append(Arrow((x, yy), (x + 1.2, yy), _EDGE_STYLES[kind], role="legend"))
        items.append(Label((x + 1.35, yy), _EDGE_NAMES[kind], "small label", anchor="west", role="legend"))
    return items


def _edge_label(a: Arrow, text: str, *, side: str = "east", shift: float = 0.0, frac: float = 0.5) -> Label:
    mx = a.start[0] + frac * (a.end[0] - a.start[0])
    my = a.start[1] + frac * (a.end[1] - a.start[1]) + shift
    anchor = {"east": "west", "west": "east", "north": "south", "south": "north"}[side]
    dx = {"east": 0.1, "west": -0.1}.get(side, 0.0)
    return Label((mx + dx, my), text, "small label,align=left", anchor=anchor, role="edge_label")


# ---------------------------------------------------------------------------
# E2.1 one starting point, many reductions
# ---------------------------------------------------------------------------

def plasma_model_reduction_map(*, labels: bool = True) -> Diagram:
    r"""One starting point and the reductions to every description VAFT uses.

    The root is Vlasov / Fokker--Planck + Maxwell, which no VAFT solver
    solves. Below it, left to right, the fluid (with its ``FLUID_MODELS``),
    drift-kinetic, gyrokinetic and particle descriptions; the hybrid, which
    couples a fluid model and a drift-kinetic response; and last the reduced
    models (by ``derived_from``) and the equilibrium. Each node lists the
    ``ADAPTER_MODES`` ids whose record matches it. Solid edges are asymptotic
    reductions, labelled with the ordering that makes them; the double-headed
    edges a model coupling; dashed edges a surrogate's ancestry.
    """
    labels = _check_labels(labels)
    stab = lambda f: f.scientific_operation != "equilibrium"
    items: List = []
    root = box(0.0, 0.0, 7.0, 1.2, "\\textbf{Vlasov / Fokker--Planck + Maxwell}\\\\ "
               "{\\footnotesize\\itshape no VAFT solver}", style=_HOLLOW, text_style=_HOLLOW_TEXT,
               role="node:vlasov_maxwell", latex=True)
    fluid_ids = {m: _ids(lambda f, m=m: f.bulk_description == "fluid" and f.fluid_model == m and stab(f))
                 for m in ("ideal_mhd", "resistive_mhd", "extended_mhd")}
    y1, y2, y3 = -3.8, -7.8, -11.6
    fluid = _node(-12.5, y1, 4.6, 1.3, "fluid", (), detail="bulk = fluid", role="node:fluid")
    children = {}
    for i, (model, ids) in enumerate(fluid_ids.items()):
        children[model] = _node(-15.0 + i * 2.5, y2, 2.35, 2.4, model.replace("_", " "), ids,
                                role=f"node:{model}", style="concept leaf")
    dk = _node(-4.0, y1, 3.9, 1.6, "drift kinetic", _dk_ids(), role="node:drift_kinetic")
    gk = _node(2.5, y1, 3.9, 1.6, "gyrokinetic", _gk_ids(), role="node:gyrokinetic")
    particle_ids = _ids(lambda f: f.bulk_description == "particle")
    used_orbits = {_record(k).orbit_representation for k in particle_ids}
    orbits = ", ".join(o.replace("_", " ") + ("" if o in used_orbits else " (not in VAFT)")
                       for o in ("full_orbit", "guiding_center"))
    particle = _node(9.5, y1, 3.9, 2.1, "particle", particle_ids, detail=f"orbits: {orbits}", role="node:particle")
    hybrid_ids = _ids(lambda f: f.bulk_description == "hybrid")
    couplings = sorted({m.formalism.kinetic_coupling for m in _modes()
                        if m.formalism is not None and m.formalism.bulk_description == "hybrid"})
    hybrid = _node(-6.5, y2, 4.0, 2.6, "hybrid", hybrid_ids,
                   detail="coupling: " + ", ".join(escape_latex(c) for c in couplings), role="node:hybrid")
    red_dk = _node(-4.0, y3, 3.9, 1.9, "reduced", _ids(lambda f: f.bulk_description == "reduced"
                                                        and f.derived_from == "drift_kinetic"),
                   detail="derived from drift kinetics", role="node:reduced_drift_kinetic")
    red_gk_ids = [k for k in _ids(lambda f: f.bulk_description == "reduced" and f.derived_from == "gyrokinetic")
                  if k not in _SURROGATES]
    red_nn_ids = [k for k in _SURROGATES if _record(k) is not None]
    red_gk = _node(2.5, y2, 3.9, 1.9, "reduced", red_gk_ids, detail="derived from gyrokinetics",
                   role="node:reduced_gyrokinetic")
    red_nn = _node(2.5, y3, 3.9, 1.6, "learned surrogate", red_nn_ids, role="node:reduced_surrogate")
    equilibrium = _node(-12.5, y3 - 0.8, 4.6, 3.4, "equilibrium",
                        _ids(lambda f: f.scientific_operation == "equilibrium"),
                        detail="operation = equilibrium", role="node:equilibrium")
    for b in (root, fluid, *children.values(), dk, gk, particle, hybrid, red_dk, red_gk, red_nn, equilibrium):
        items += list(b.items)
    edges = [
        (root, fluid, "asymptotic_reduction", "velocity moments\\\\ + closure", "west", 0.55),
        (root, dk, "asymptotic_reduction", "gyro-phase average;\\\\ $\\rho/L, \\omega/\\Omega, k_\\perp\\rho \\ll 1$",
         "west", 0.62),
        (root, gk, "asymptotic_reduction", "$\\omega/\\Omega \\sim k_\\parallel/k_\\perp$\\\\ $\\sim \\delta f/F \\sim \\rho_*$,\\\\ "
         "$k_\\perp\\rho \\sim 1$", "west", 0.62),
        (root, particle, "asymptotic_reduction", "test-particle orbits;\\\\ $f$ sampled by\\\\ markers, not gridded",
         "east", 0.5),
        (gk, dk, "asymptotic_reduction", "$k_\\perp\\rho \\to 0$", "north", 0.5),
        (gk, red_gk, "asymptotic_reduction", "quasilinear\\\\ gyro-Landau-fluid\\\\ closure", "east", 0.5),
        (red_gk, red_nn, "surrogate_ancestry", "learned", "east", 0.5),
        (dk, red_dk, "asymptotic_reduction", "analytic theory,\\\\ fits (+ NN)", "east", 0.5),
        (fluid, hybrid, "model_coupling", "", "east", 0.5),
        (dk, hybrid, "model_coupling", "", "east", 0.5),
    ]
    for child in children.values():
        items.append(connector(fluid, child, style="connector line,black!45", role="edge:fluid_model"))
    for a, b, kind, text, side, frac in edges:
        arrow = _edge(a, b, kind)
        items.append(arrow)
        if labels and text:
            items.append(_edge_label(arrow, text, side=side, frac=frac))
    # fluid -> equilibrium runs down the left margin, past the fluid-model leaves
    x_eq = -16.6
    path = [(fluid.x - 0.5 * fluid.width - 0.08, fluid.y), (x_eq, fluid.y), (x_eq, equilibrium.y),
            (equilibrium.x - 0.5 * equilibrium.width - 0.08, equilibrium.y)]
    items.append(Polyline.of(path[:-1], "black!75,line width=1.0pt", role="edge:asymptotic_reduction"))
    items.append(Arrow(path[-2], path[-1], _EDGE_STYLES["asymptotic_reduction"], role="edge:asymptotic_reduction"))
    if labels:
        items.append(Label((x_eq - 0.1, 0.5 * (y2 + y3)), "static force\\\\ balance,\\\\ "
                           "$\\tau_{evol}/\\tau_A \\gg 1$", "small label,align=right", anchor="east", role="edge_label"))
        items.append(Label((hybrid.x, hybrid.y - 0.5 * hybrid.height - 0.12), "couples a fluid model\\\\ "
                           "and a kinetic response", "small label,align=center", anchor="north", role="edge_label"))
        items += _legend(("asymptotic_reduction", "model_coupling", "surrogate_ancestry"), 8.0, y2 + 0.4)
        items.append(Label((-3.0, y3 - 2.7), "No direction is a fidelity axis: a branch is a different set of "
                           "orderings, not a simpler or a more accurate model.\\\\ Ids are the VAFT adapter modes "
                           "whose record matches the node (vaft.validation.model\\_contracts.ADAPTER\\_MODES).",
                           "note", anchor="north", role="note"))
    nodes = {
        "fluid": sum(fluid_ids.values(), []), **fluid_ids, "drift_kinetic": _dk_ids(), "gyrokinetic": _gk_ids(),
        "particle": particle_ids, "hybrid": hybrid_ids,
        "equilibrium": _ids(lambda f: f.scientific_operation == "equilibrium"),
        "reduced_drift_kinetic": _ids(lambda f: f.bulk_description == "reduced" and f.derived_from == "drift_kinetic"),
        "reduced_gyrokinetic": red_gk_ids, "reduced_surrogate": red_nn_ids,
    }
    placed = {k for ids in nodes.values() for k in ids}
    missing = [m.key for m in _modes() if m.formalism is not None and m.key not in placed]
    if missing:
        raise RuntimeError(f"plasma_model_reduction_map places no node for {missing}: add one")
    model = {"nodes": nodes, "couplings": tuple(couplings)}
    return Diagram("plasma_model_reduction_map", Scene(tuple(items)), model=model)


def _dk_ids():
    return _ids(lambda f: f.bulk_description == "kinetic" and f.kinetic_equation == "drift_kinetic")


def _gk_ids():
    return _ids(lambda f: f.bulk_description == "kinetic" and f.kinetic_equation == "gyrokinetic")


# ---------------------------------------------------------------------------
# E2.4 the axes are independent
# ---------------------------------------------------------------------------

#: the three runs drawn across the axes, and their colours
_AXES_EXAMPLES = (("gacode:cgyro_nonlinear", "islandblue"), ("gpec:dcon_kinetic", "driftred"),
                  ("nubeam:fast_ions", "absequil!80!black"))


def formalism_axes_map(*, labels: bool = True) -> Diagram:
    r"""The independent axes of a plasma-formalism record, with three runs drawn across them.

    One column per field of ``vaft.code.formalism.AXES``, in record order,
    each listing its controlled values and an "N/A" cell (the field has no
    meaning for the run). Three ``ADAPTER_MODES`` runs -- CGYRO nonlinear,
    DCON kinetic, NUBEAM -- are polylines through their values; no column is
    a level of another. The numerical representation is not an axis: it
    stays in ``extensions`` and has its own box, which no run line enters.
    """
    from vaft.code.formalism import AXES
    from vaft.validation.model_contracts import adapter_mode

    labels = _check_labels(labels)
    col, row, top = 2.75, 0.72, 0.0
    width = col - 0.3
    cells: Dict[Tuple[str, str], Tuple[float, float]] = {}
    for j, (field, values) in enumerate(AXES.items()):
        for i, value in enumerate((*values, "N/A")):
            cells[(field, value)] = (j * col, top - (i + 0.5) * row)
    lines: List = []
    marks: List = []
    runs = {}
    for k, (key, colour) in enumerate(_AXES_EXAMPLES):
        record = adapter_mode(key).formalism
        points, dy = [], (k - 1) * 0.14
        for field in AXES:
            value = getattr(record, field)
            x, y = cells[(field, "N/A" if value is None else value)]
            points += [(x - 0.5 * width, y + dy), (x + 0.5 * width, y + dy)]
            for side in (-1, 1):
                marks.append(Polyline.of([(x + side * 0.5 * width - 0.07, y + dy - 0.07),
                                          (x + side * 0.5 * width + 0.07, y + dy - 0.07),
                                          (x + side * 0.5 * width + 0.07, y + dy + 0.07),
                                          (x + side * 0.5 * width - 0.07, y + dy + 0.07)],
                                         f"draw={colour},fill={colour}", role=f"run:{key}", closed=True))
        runs[key] = tuple(getattr(record, f) for f in AXES)
        # the line runs behind the cells: visible between columns, marked at each chosen cell
        lines.append(Polyline.of(points, f"{colour},line width=1.6pt,opacity=0.85", role=f"run:{key}"))
        if labels:
            lines.append(Label((-0.5 * col - 0.15, points[0][1]), escape_latex(key), f"small label,text={colour}",
                               anchor="east", role=f"run:{key}"))
    items: List = list(lines)
    for j, (field, values) in enumerate(AXES.items()):
        x = j * col
        items.append(Label((x - 0.4, top + 0.35), field.replace("_", "\\_"), "small label,rotate=35",
                           anchor="south west", role=f"column:{field}"))
        for value in (*values, "N/A"):
            cx, cy = cells[(field, value)]
            style = _GREY if value == "N/A" else "concept leaf"
            items += list(box(cx, cy, width, row - 0.16, value.replace("_", " "), style=style,
                              text_style="concept text,font=\\scriptsize", role=f"value:{field}:{value}").items)
    items += marks
    depth = max(len(v) for v in AXES.values()) + 1
    bottom = top - depth * row
    side_x = (len(AXES) - 1) * col + 0.5 * col + 2.0
    items += list(box(side_x + 1.4, top - 2.0, 3.2, 2.4, "\\textbf{numerical representation}\\\\ "
                      "{\\footnotesize (extensions, not an axis)}\\\\ {\\footnotesize continuum, PIC, Monte Carlo, "
                      "finite element}", style=_GREY, role="extensions", latex=True).items)
    if labels:
        items.append(Label((0.5 * (len(AXES) - 1) * col, bottom - 0.4),
                           "Each column is one field of vaft.code.formalism.AXES, in record order. A run takes one value "
                           "per column, or N/A when the field has no meaning for it.\\\\ No column is a level of another, "
                           "and no direction is a fidelity axis.", "note", anchor="north", role="note"))
    model = {"columns": tuple(AXES), "runs": runs}
    return Diagram("formalism_axes_map", Scene(tuple(items)), model=model)


# ---------------------------------------------------------------------------
# E2.5 every VAFT mode, placed
# ---------------------------------------------------------------------------

#: family order of the placement matrix (vaft.validation.model_contracts.AdapterMode.family)
_FAMILY_ORDER = ("equilibrium", "mhd_stability", "neoclassical", "gyrokinetic", "fast_ions", "waves",
                 "integrated", "field_lines", "drivers")
_FAMILY_TITLES = {"equilibrium": "equilibrium", "mhd_stability": "MHD stability", "neoclassical": "neoclassical",
                  "gyrokinetic": "gyrokinetic", "fast_ions": "fast ions", "waves": "waves",
                  "integrated": "integrated modelling", "field_lines": "field lines", "drivers": "drivers"}
_PLACEMENT_COLUMNS = ("bulk", "model", "spatial domain", "regime")


def _model_cell(record) -> str:
    parts = [v for v in (record.fluid_model, record.kinetic_equation) if v]
    if record.derived_from:
        parts.append(f"from {record.derived_from}")
    return " + ".join(parts) if parts else "—"


def solver_placement_map(*, labels: bool = True) -> Diagram:
    r"""Every VAFT adapter mode against the main formalism axes, then the codes not in VAFT.

    Rows are ``ADAPTER_MODES`` grouped by family; columns the bulk
    description, the fluid model / kinetic equation (or a reduced model's
    parent), the spatial domain and the regime, filled from each record. A
    mode with no record is greyed "unclassified". Below, the codes of
    :data:`EXTERNAL_CODES` are drawn hollow and labelled "not in VAFT", so no
    code looks implemented that is not.
    """
    labels = _check_labels(labels)
    widths = (2.2, 4.4, 2.6, 2.4)
    left, row = 4.3, 0.55
    xs, x = [], left
    for w in widths:
        xs.append(x + 0.5 * w)
        x += w
    right = x
    items: List = []
    for cx, name in zip(xs, _PLACEMENT_COLUMNS):
        items.append(Label((cx, 0.35), name, "small label,font=\\bfseries", anchor="south", role=f"column:{name}"))
    modes = _modes()
    families = [f for f in _FAMILY_ORDER if any(m.family == f for m in modes)]
    families += sorted({m.family for m in modes} - set(families))
    y = 0.0
    placed = {}
    for family in families:
        y -= row
        items.append(Polyline.of([(0.0, y - 0.5 * row), (right, y - 0.5 * row), (right, y + 0.5 * row),
                                  (0.0, y + 0.5 * row)], "concept band", role=f"family:{family}", closed=True))
        items.append(Label((0.1, y), escape_latex(_FAMILY_TITLES.get(family, family)), "concept band label", anchor="west",
                           role=f"family:{family}"))
        for mode in (m for m in modes if m.family == family):
            y -= row
            items.append(Label((0.25, y), escape_latex(mode.key), "small label,font=\\ttfamily\\scriptsize",
                               anchor="west", role=f"mode:{mode.key}"))
            if mode.formalism is None:
                items.append(Polyline.of([(left, y - 0.22), (right, y - 0.22), (right, y + 0.22), (left, y + 0.22)],
                                         _GREY, role=f"mode:{mode.key}", closed=True))
                items.append(Label((left + 0.1, y), "unclassified: " + _tex(mode.note),
                                   f"small label,text=black!55,font=\\itshape\\tiny,text width={right - left - 0.3:.2f}cm,"
                                   "align=left", anchor="west", role=f"mode:{mode.key}"))
                placed[mode.key] = None
                continue
            r = mode.formalism
            values = (r.bulk_description, _model_cell(r), r.spatial_domain or "—", r.regime or "—")
            placed[mode.key] = values
            for cx, w, v in zip(xs, widths, values):
                items.append(Label((cx, y), _tex(v).replace("\\_", " "), "small label,font=\\scriptsize",
                                   role=f"cell:{mode.key}"))
    y -= row
    items.append(Polyline.of([(0.0, y - 0.5 * row), (right, y - 0.5 * row), (right, y + 0.5 * row),
                              (0.0, y + 0.5 * row)], "concept band", role="family:external", closed=True))
    items.append(Label((0.1, y), "not in VAFT (comparison only)", "concept band label", anchor="west",
                       role="family:external"))
    for code in EXTERNAL_CODES:
        y -= row
        items.append(Polyline.of([(0.05, y - 0.24), (right - 0.05, y - 0.24), (right - 0.05, y + 0.24),
                                  (0.05, y + 0.24)], _HOLLOW, role=f"external:{code.code}", closed=True))
        items.append(Label((0.25, y), _tex(code.code), "small label,text=black!60,font=\\scriptsize",
                           anchor="west", role=f"external:{code.code}"))
        for cx, v in zip(xs, (code.bulk, code.model.split(";")[0].split(" (")[0], code.domain.split(";")[0], "—")):
            items.append(Label((cx, y), _tex(v), "small label,text=black!60,font=\\scriptsize",
                               role=f"external:{code.code}"))
    if labels:
        items.append(Label((0.5 * right, y - 0.6), "Implemented rows are filled from each mode's PlasmaFormalism record; "
                           "unclassified modes have no record.\\\\ Hollow rows are not in VAFT "
                           "(EXTERNAL\\_CODES; full cells in the guide's table).", "note", anchor="north", role="note"))
    model = {"rows": placed, "external": tuple(c.code for c in EXTERNAL_CODES), "families": tuple(families)}
    return Diagram("solver_placement_map", Scene(tuple(items)), model=model)


# ---------------------------------------------------------------------------
# E2.6 kinetic-MHD is a coupling, not an equation
# ---------------------------------------------------------------------------

def _roles(key: str) -> str:
    from vaft.validation.model_contracts import adapter_mode

    uses, _ = adapter_mode(key).contracts()
    return ", ".join(f"{u.contract.replace('_', ' ')} ({u.role.replace('_', ' ')})" for u in uses)


def kinetic_mhd_coupling(*, labels: bool = True) -> Diagram:
    r"""Kinetic-MHD as a loop between a fluid bulk and a drift-kinetic population.

    The fluid bulk (``fluid_model="ideal_mhd"``) passes its displacement and
    fields to a drift-kinetic population, whose response returns through the
    record's ``kinetic_coupling``: ``energy`` closes the loop
    self-consistently (DCON and GPEC kinetic, a delta W_k term), ``closure``
    is one-way (PENTRC evaluates a torque on a given displacement and feeds
    nothing back). The ideal-only modes sit inside the bulk. RDCON + RMATCH
    stands outside the loop: its T_e, n_e input is a profile-informed
    resistivity, a coefficient, not a model coupling. Each placed mode is
    annotated with the roles of the contracts it consumes.
    """
    from vaft.validation.model_contracts import ADAPTER_MODES

    labels = _check_labels(labels)
    by_key = {m.key: m for m in ADAPTER_MODES}
    hybrid = [k for k, m in by_key.items() if m.formalism is not None and m.formalism.bulk_description == "hybrid"]
    loop = [k for k in hybrid if by_key[k].formalism.kinetic_coupling == "energy"]
    one_way = [k for k in hybrid if by_key[k].formalism.kinetic_coupling == "closure"]
    if set(loop) | set(one_way) != set(hybrid):
        raise RuntimeError("kinetic_mhd_coupling draws energy and closure couplings only; a hybrid mode with "
                           f"another kinetic_coupling needs a place: {sorted(set(hybrid) - set(loop) - set(one_way))}")
    stability = lambda m: (m.formalism is not None and m.formalism.bulk_description == "fluid"
                           and m.formalism.scientific_operation != "equilibrium")
    ideal = [k for k, m in by_key.items() if stability(m) and m.formalism.fluid_model == "ideal_mhd"]
    outside = [k for k, m in by_key.items() if stability(m) and m.formalism.fluid_model == "resistive_mhd"]
    populations = sorted({p for k in hybrid for p in (by_key[k].formalism.kinetic_population or ())})
    equations = sorted({by_key[k].formalism.kinetic_equation for k in hybrid})
    items: List = []
    bulk = _node(-5.0, 0.0, 5.0, 2.6, "fluid bulk", ideal, detail="fluid\\_model = ideal\\_mhd", role="node:bulk")
    kinetic = _node(5.0, 0.0, 5.0, 2.0, "kinetic population", (),
                    detail="kinetic\\_equation = " + ", ".join(_tex(e) for e in equations) + "\\\\ population: " +
                    ", ".join(_tex(p).replace("\\_", " ") for p in populations), role="node:kinetic")
    items += list(bulk.items) + list(kinetic.items)
    top, bottom = 0.75, -0.75
    # one model coupling, drawn as its two directions so the loop reads clockwise
    one_way_style = _EDGE_STYLES["model_coupling"].replace("<->", "->")
    out = Arrow((bulk.x + 0.5 * bulk.width + 0.08, top), (kinetic.x - 0.5 * kinetic.width - 0.08, top),
                one_way_style, role="edge:model_coupling")
    back = Arrow((kinetic.x - 0.5 * kinetic.width - 0.08, bottom), (bulk.x + 0.5 * bulk.width + 0.08, bottom),
                 one_way_style, role="edge:model_coupling")
    items += [out, back]
    # PENTRC: a one-way closure, drawn as an open arrow to its output rather than back into the bulk
    torque = _node(5.0, -4.0, 5.0, 1.6, "closure output", one_way, detail="NTV torque on a given $\\xi$",
                   style="concept leaf", role="node:closure")
    items += list(torque.items)
    items.append(Arrow((kinetic.x, kinetic.y - 0.5 * kinetic.height - 0.08),
                       (torque.x, torque.y + 0.5 * torque.height + 0.08), "-{Stealth[open,length=3.2mm,width=2.6mm]},black!70,line width=1.0pt",
                       role="edge:closure"))
    # RDCON + RMATCH: outside the loop, with a coefficient input
    rdcon = _node(-5.0, -4.6, 5.0, 1.8, "outside the loop", outside,
                  detail="matched ideal outer + resistive layer", style="concept leaf", role="node:rdcon")
    items += list(rdcon.items)
    items.append(Arrow((-13.0, rdcon.y), (rdcon.x - 0.5 * rdcon.width - 0.08, rdcon.y),
                       "->,black!55,line width=0.9pt", role="edge:coefficient"))
    if labels:
        items.append(Label(((out.start[0] + out.end[0]) / 2, top + 0.12), "displacement $\\xi$ / fields",
                           "small label", anchor="south", role="edge_label"))
        items.append(Label(((back.start[0] + back.end[0]) / 2, bottom - 0.12),
                           "kinetic\\_coupling = energy\\\\ ($\\delta W_k$, self-consistent)", "small label,align=center",
                           anchor="north", role="edge_label"))
        items.append(Label((kinetic.x + 0.15, 0.5 * (kinetic.y - 0.5 * kinetic.height + torque.y + 0.5 * torque.height)),
                           "kinetic\\_coupling = closure\\\\ (one-way, not fed back)", "small label,align=left",
                           anchor="west", role="edge_label"))
        items.append(Label((-13.0, rdcon.y - 0.15), "$T_e, n_e \\to$ Spitzer $\\eta$\\\\ (profile-informed coefficient,\\\\ "
                           "not a model coupling)", "small label,align=left", anchor="north west", role="edge_label"))
        # the loop's modes and every placed mode's contract roles
        items.append(Label((0.0, 0.0), "on the loop:\\\\ " + "\\\\ ".join(escape_latex(k) for k in loop),
                           "small label,align=center,font=\\ttfamily\\scriptsize", role="loop_modes"))
        notes = []
        for key in ideal + loop + one_way + outside:
            notes.append(f"{escape_latex(key)}: {escape_latex(_roles(key))}")
        items.append(Label((0.0, -6.1), "\\\\ ".join(notes), "small label,align=left,font=\\scriptsize",
                           anchor="north", role="roles"))
        # legend: the coupling pair, the one-way closure, the coefficient input
        lx, ly = 9.0, -6.3
        for i, (style, text) in enumerate(((one_way_style, "model coupling (both arrows: one loop)"),
                                           ("-{Stealth[open,length=3.2mm,width=2.6mm]},black!70,line width=1.0pt", "one-way closure"),
                                           ("->,black!55,line width=0.9pt", "input coefficient, not a coupling"))):
            items.append(Arrow((lx, ly - 0.5 * i), (lx + 1.2, ly - 0.5 * i), style, role="legend"))
            items.append(Label((lx + 1.35, ly - 0.5 * i), text, "small label", anchor="west", role="legend"))
    model = {"loop": tuple(loop), "one_way": tuple(one_way), "bulk": tuple(ideal), "populations": tuple(populations),
             "outside": tuple(outside)}
    return Diagram("kinetic_mhd_coupling", Scene(tuple(items)), model=model)


# ---------------------------------------------------------------------------
# E2.7 reduced models and surrogates
# ---------------------------------------------------------------------------

#: the model reduction each reduced mode makes, in words (its record says only what it is derived from)
_REDUCTIONS = {
    "gacode:tglf_linear": "gyro-Landau-fluid eigenmodes",
    "gacode:tglf": "quasilinear gyro-Landau-fluid fluxes",
    "gacode:tglf_nn": "neural-network fit to TGLF",
    "gacode:neo_analytic": "analytic theory, fits and NN (HH, Chang--Hinton, Sauter, Redl, NCLASS)",
}


def reduced_model_ancestry(*, labels: bool = True) -> Diagram:
    r"""Parent theory, model reduction and computational realization of every reduced mode.

    Rows are the ``ADAPTER_MODES`` with ``bulk_description="reduced"``;
    the left column is their ``derived_from`` theory, the middle the
    reduction, the right the VAFT mode that realizes it. Parent to reduction
    is an asymptotic reduction; TGLF to TGLF-NN a surrogate's ancestry. Every
    row consumes its parent's contract in the ``inherited`` role, and adds a
    closure or fit no ordering contract covers.
    """
    from vaft.validation.model_contracts import ADAPTER_MODES

    labels = _check_labels(labels)
    reduced = [m for m in ADAPTER_MODES if m.formalism is not None and m.formalism.bulk_description == "reduced"]
    if {m.key for m in reduced} != set(_REDUCTIONS):
        raise RuntimeError("reduced_model_ancestry is out of step with ADAPTER_MODES")
    items: List = []
    xs, row = (-7.0, 0.0, 7.0), 2.1
    for x, title in zip(xs, ("parent theory (derived\\_from)", "model reduction", "computational realization")):
        items.append(Label((x, 1.1), title, "concept group title", anchor="south", role="column"))
    parents: Dict[str, Box] = {}
    reductions: Dict[str, Box] = {}
    # parents grouped; within a parent, models before their surrogates, in registry order
    order = sorted(reduced, key=lambda m: (m.formalism.derived_from != "gyrokinetic", m.key in _SURROGATES))
    for i, mode in enumerate(order):
        y = -i * row
        parent = mode.formalism.derived_from
        reduction = _node(xs[1], y, 5.4, 1.4, _REDUCTIONS[mode.key], (), role=f"reduction:{mode.key}")
        realization = _node(xs[2], y, 4.6, 1.5, mode.key.split(":")[1].replace("_", " "), [mode.key],
                            detail="inherited + own closure/fit", style="concept leaf", role=f"mode:{mode.key}")
        reductions[mode.key] = reduction
        items += list(reduction.items) + list(realization.items)
        items.append(connector(reduction, realization, style="connector", role="edge:realization"))
        if parent not in parents:
            members = [m for m in order if m.formalism.derived_from == parent]
            first, last = order.index(members[0]), order.index(members[-1])
            parents[parent] = _node(xs[0], -0.5 * (first + last) * row, 4.0, 1.4, parent.replace("_", " "), (),
                                    role=f"parent:{parent}")
            items += list(parents[parent].items)
        if mode.key not in _SURROGATES:
            # a surrogate descends from TGLF (dashed, below), not by reduction from the theory
            items.append(_edge(parents[parent], reduction, "asymptotic_reduction"))
    for surrogate, model in _SURROGATES.items():
        items.append(_edge(reductions[model], reductions[surrogate], "surrogate_ancestry"))
    bottom = -(len(order) - 0.4) * row
    if labels:
        items += _legend(("asymptotic_reduction", "surrogate_ancestry"), xs[0] - 2.0, bottom - 0.3)
        items.append(Label((xs[1], bottom - 1.6), "Every row consumes its parent's ordering contract in the inherited "
                           "role: necessary, not sufficient.\\\\ The reduction's own closure or fit is not covered by "
                           "any contract (contracts\\_for, unmapped).", "note", anchor="north", role="note"))
    model = {"rows": tuple(m.key for m in order), "parents": {m.key: m.formalism.derived_from for m in order}}
    return Diagram("reduced_model_ancestry", Scene(tuple(items)), model=model)
