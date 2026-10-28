"""The cross-domain figures: equilibrium, MHD and transport joined through shared representations.

The technical figure (``cross_domain_scientific_workflows``) draws every element from the workflow
spine (#1585), names real APIs and Data Dictionary leaves, and makes only the claims the
implementation supports: a shared IDS is not a shared state, MHD and transport are complementary, and
only the models that read the resolved state share its identity. The conceptual figure
(``cross_domain_scientific_architecture``) summarizes it, and every card names the technical elements
it stands for, so the two cannot drift apart.
"""

import inspect
import itertools
import re

import pytest

import vaft.diagram
from vaft.diagram import _cross_domain as X
from vaft.diagram import _cross_domain_concept as C
from vaft.diagram._scene import Arrow, Label, Polyline
from vaft.diagram._workflow import resolve
from vaft.diagram._workflow_specs import WORKFLOWS

from test_diagram_workflows import DIAGRAMS_MD, _ids_names, _ids_roots

REGIONS = {key for key, *_ in X.REGIONS}
#: the manuscript column: both figures are checked at this width
MANUSCRIPT_WIDTH_CM = 18.0
_POINTS = {"\\tiny": 5.0, "\\scriptsize": 7.0, "\\footnotesize": 8.0, "\\small": 9.0}


def _edges(connection=None):
    return [(s, d) for s, d, c, _ in X.EDGES if connection in (None, c)]


def _technical():
    return vaft.diagram.cross_domain_scientific_workflows()


def _concept():
    return vaft.diagram.cross_domain_scientific_architecture()


def _methods():
    return {e.key for e in X.ELEMENTS if e.kind in ("solver", "derived") and e.method}


# --------------------------------------------------------------------------- technical: sources and claims


def test_every_element_is_drawn_from_existing_spine_nodes_and_its_apis_resolve():
    for e in X.ELEMENTS:
        for spec, node in e.sources:
            assert node in {n.key for n in WORKFLOWS[spec].nodes}, (e.key, spec, node)
        for api in e.apis:
            assert resolve(api) is not None, (e.key, api)


def test_an_explicit_api_only_fills_what_the_source_nodes_leave_unnamed():
    for e in X.ELEMENTS:
        if e.api:
            assert e.api not in {n.api for n in e.source_nodes}, e.key


def test_standardized_leaves_exist_in_the_installed_data_dictionary_and_match_their_sources():
    from omas import omas_info_node
    from omas.omas_setup import omas_rcparams

    version = omas_rcparams["default_imas_version"]
    for e in X.ELEMENTS:
        if e.kind != "standardized":
            continue
        assert omas_info_node(e.dd_path, imas_version=version), (e.key, e.dd_path, version)
        roots = {root for ids in e.ids for root in _ids_roots(ids)}
        assert e.ids_name in roots, (e.key, e.ids_name, roots)


def test_edges_join_known_ends_and_every_element_is_connected():
    known = set(X.ELEMENT) | REGIONS
    assert X.CONNECTIONS == ("flow", "alternative", "mapping", "native")
    for src, dst, connection, _ in X.EDGES:
        assert {src, dst} <= known and connection in ("flow", "mapping", "native"), (src, dst, connection)
    assert len(set(_edges())) == len(X.EDGES)
    assert set(X.ELEMENT) <= {k for edge in _edges() for k in edge}


def test_an_input_read_by_two_methods_is_drawn_once():
    inputs = [e for e in X.ELEMENTS if e.kind in ("measured", "machine", "prior", "convention", "reconstructed")]
    seen = {}
    for e in inputs:
        assert (e.kind, e.domain, e.detail) not in seen, (e.key, seen.get((e.kind, e.domain, e.detail)))
        seen[(e.kind, e.domain, e.detail)] = e.key
    for key, readers in (("eq_coils", {"tokamaker", "efit"}), ("eq_machine", {"tokamaker", "efit"}),
                         ("tr_profiles", {"sauter_redl", "state"})):
        assert {d for s, d in _edges("flow") if s == key} == readers, key


def test_the_four_relations_differ_without_colour():
    from vaft.diagram._render import template

    styles = X.CONNECTION_STYLE
    flow = re.search(r"^\s*" + re.escape(styles["flow"]) + r"/\.style=\{(.*)\},?$", template(), re.M).group(1)
    assert "->" in flow and "dash" not in flow and "dotted" not in flow
    assert "dash dot" in styles["alternative"] and "fill" not in styles["alternative"]
    assert "dash" not in styles["mapping"] and "fill" in styles["mapping"]
    assert "dashed" in styles["native"] and "fill" in styles["native"]


def test_mappings_end_in_ids_and_native_results_stay_native():
    for src, dst in _edges("mapping"):
        assert X.ELEMENT[dst].kind == "standardized" and X.ELEMENT[src].kind in ("solver", "derived"), (src, dst)
    for src, dst in _edges("native"):
        assert X.ELEMENT[dst].kind == "native_result", (src, dst)
    for e in X.ELEMENTS:
        if e.kind == "standardized":
            assert [s for s, d in _edges("mapping") if d == e.key], e.key


def test_one_equilibrium_ids_holds_four_kinds_of_state_and_only_the_reconstructed_one_feeds_transport():
    entries = [e for e in X.ELEMENTS if e.domain == "equilibrium" and e.kind == "standardized"]
    assert {e.ids_name for e in entries} == {"equilibrium"} and len(entries) == 4
    producers = {d: s for s, d in _edges("mapping") if d in {e.key for e in entries}}
    assert producers == {"eq_analytic": "analytic", "eq_model": "tokamaker", "eq_reconstructed": "efit",
                         "eq_refined": "chease"}
    into_state = {s for s, d in _edges("flow") if d == "state"}
    assert into_state == {"eq_reconstructed", "tr_profiles", "tr_closures"}
    # CHEASE refines the reconstruction: it reads that entry and is no alternative to EFIT
    assert ("eq_reconstructed", "chease") in _edges("flow")
    assert not [s for s in X.ALTERNATIVES if "chease" in s]


def test_element_kinds_are_those_of_the_nodes_they_are_drawn_from():
    for e in X.ELEMENTS:
        kinds = {n.kind for n in e.source_nodes}
        # a result the specs leave inside its step (the Sauter / Redl return value) is drawn from that step
        assert e.kind in kinds or (e.kind == "native_result" and kinds <= {"derived", "solver"}), (e.key, e.kind)


def test_reconstruction_reads_either_lineage_and_gpec_reads_dcon():
    efit_inputs = {s for s, d in _edges("flow") if d == "efit"}
    assert {"ef_magnetics", "ef_kinetic"} <= efit_inputs
    assert ("kinetic_efit", "efit") in X.ELEMENT["efit"].sources
    # an ideal-GPEC run consumes the same-mode DCON products (euler.bin): vaft.code.gpec checks it
    assert ("dcon", "gpec") in _edges("flow")
    assert "euler.bin" in inspect.getsource(resolve("vaft.code.gpec"))


def test_alternatives_are_methods_of_one_domain_and_mhd_methods_are_complementary():
    assert X.ALTERNATIVES == (("analytic", "tokamaker", "efit"), ("sauter_redl", "neo"), ("tglf", "cgyro"))
    for group in X.ALTERNATIVES:
        assert set(group) <= _methods() and len({X.ELEMENT[k].domain for k in group}) == 1, group
    assert not {"dcon", "rdcon", "gpec"} & {k for group in X.ALTERNATIVES for k in group}


def test_mhd_and_transport_are_never_chained():
    for src, dst in _edges():
        domains = {X.ELEMENT[k].domain if k in X.ELEMENT else k for k in (src, dst)}
        assert domains != {"mhd", "transport"}, (src, dst)


def test_only_the_models_reading_the_resolved_state_share_its_identity():
    readers = {d for s, d in _edges("flow") if s == "state"}
    assert readers == {"neo", "classical", "tglf", "cgyro"} == set(X.STATE_IDENTITY_JOIN)
    # the identity keys runs, not IDS entries: the NEO writer records none, so no join is drawn on the results
    assert not [it for it in _technical().scene.items if "join" in (getattr(it, "role", "") or "")]
    assert X.ELEMENT["state"].api == "vaft.process.transport_state.ResolvedTransportState.identity"
    # Sauter / Redl read the IMAS state directly and are retained, not mapped
    assert "sauter_redl" not in readers
    assert {d for s, d in _edges() if s == "sauter_redl"} == {"sr_native"}
    writers = [it for it in _technical().scene.items if (getattr(it, "role", "") or "").startswith("no writer:")]
    assert [it.role for it in writers] == ["no writer:sauter_redl"]


def test_cgyro_is_drawn_as_linear_eigenmodes_only():
    assert [d for s, d in _edges("mapping") if s == "cgyro"] == ["gk_cgyro"]
    assert X.ELEMENT["gk_cgyro"].dd_path.startswith("gyrokinetics_local.linear.")
    assert X.ELEMENT["cgyro"].basis == "gyrokinetic" and X.ELEMENT["gk_cgyro"].label == "Linear eigenmodes"
    assert "gyrofluid" in X.ELEMENT["tglf"].basis  # TGLF is a quasilinear gyrofluid model, not gyrokinetic
    assert not [e for e in X.ELEMENTS if e.dd_path and "non_linear" in e.dd_path]


def test_mapping_caveats_are_stated_where_the_quantity_does_not_match_exactly():
    assert "normalized" in X.ELEMENT["ml_dcon"].detail and "J" in X.ELEMENT["ml_dcon"].note
    assert "diagonal" in X.ELEMENT["ntms_rdcon"].detail
    assert "identifier.index" in X.ELEMENT["ct_classical"].note
    assert X.ELEMENT["gpec_coils"].kind == "machine"  # coils_non_axisymmetric is an input, not a GPEC result


# --------------------------------------------------------------------------- technical: drawing


def test_the_builders_are_public_registered_and_deterministic():
    from vaft.diagram.build import CANONICAL

    for name in ("cross_domain_scientific_workflows", "cross_domain_scientific_architecture"):
        fn = getattr(vaft.diagram, name)
        assert name in vaft.diagram.__all__ and CANONICAL[f"{name}.svg"] == (name, {})
        assert fn().tikz == fn().tikz
        assert not [it for it in fn(labels=False).scene.items if isinstance(it, Label)]
        with pytest.raises(ValueError):
            fn(labels="yes")


def test_flows_are_arrows_and_each_result_sits_under_its_producer_in_its_band():
    items = _technical().scene.items
    boxes = _technical().model["boxes"]
    for src, dst, connection, text in X.EDGES:
        drawn = [it for it in items if getattr(it, "role", "") == f"edge:{connection}:{src}->{dst}"]
        if connection == "flow":
            assert len(drawn) == 1 and drawn[0].style == X.CONNECTION_STYLE["flow"], (src, dst)
            assert bool(text) == ((src, dst) in X._LABEL_AT), (src, dst)
            continue
        assert not drawn, (src, dst)  # placement says it: no line
        (sx, sy, *_), (dx, dy, *_) = boxes[src], boxes[dst]
        assert abs(sx - dx) < 1e-9 and dy < sy, (src, dst)
        band = "native" if connection == "native" else ("hub" if X.ELEMENT[dst].domain == "equilibrium" else "imas")
        x0, x1, y0, y1 = X._REGION_BOUNDS[band]
        assert x0 < dx < x1 and y1 < dy < y0, (src, dst, band)
    for group in X.ALTERNATIVES:
        assert [it for it in items if getattr(it, "role", "") == "alternative:" + ",".join(group)]
    for connection in X.CONNECTIONS:
        assert [it for it in items if isinstance(it, (Arrow, Polyline)) and it.role == f"legend:{connection}"]


def _segments(item):
    points = [item.start, item.end] if isinstance(item, Arrow) else [tuple(p) for p in item.points]
    return list(zip(points[:-1], points[1:]))


def _crosses(p, q, r, s):
    """Whether the open segments pq and rs properly intersect."""
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    d1, d2, d3, d4 = orient(r, s, p), orient(r, s, q), orient(p, q, r), orient(p, q, s)
    return d1 * d2 < -1e-12 and d3 * d4 < -1e-12


def _through(segment, box, margin=0.05):
    """Whether a segment passes through the inside of ``box`` (x, y, w, h), sampled finely."""
    (ax, ay), (bx, by) = segment
    x, y, w, h = box
    for t in [i / 200 for i in range(1, 200)]:
        px, py = ax + t * (bx - ax), ay + t * (by - ay)
        if abs(px - x) < 0.5 * w - margin and abs(py - y) < 0.5 * h - margin:
            return True
    return False


def test_no_flow_arrow_crosses_another_or_passes_through_a_box():
    diagram = _technical()
    boxes = {k: v for k, v in diagram.model["boxes"].items() if k in X.ELEMENT}
    flows = {}
    for it in diagram.scene.items:
        role = getattr(it, "role", "") or ""
        if role.startswith("edge:flow:"):
            src, dst = role[len("edge:flow:"):].split("->")
            flows[(src, dst)] = _segments(it)
    for (a, sa), (b, sb) in itertools.combinations(flows.items(), 2):
        for p, q in sa:
            for r, s in sb:
                assert not _crosses(p, q, r, s), (a, b)
    for (src, dst), segments in flows.items():
        for key, box in boxes.items():
            if key in (src, dst):
                continue
            assert not any(_through(seg, box) for seg in segments), (src, dst, key)


def test_no_two_element_boxes_overlap_and_each_sits_in_its_region():
    boxes = _technical().model["boxes"]
    for a, b in itertools.combinations(sorted(X.ELEMENT), 2):
        (ax, ay, aw, ah), (bx, by, bw, bh) = boxes[a], boxes[b]
        assert abs(ax - bx) >= 0.5 * (aw + bw) or abs(ay - by) >= 0.5 * (ah + bh), (a, b)
    for e in X.ELEMENTS:
        if e.kind == "standardized":
            region = "hub" if e.domain == "equilibrium" else "imas"
        else:
            region = "native" if e.kind == "native_result" else e.domain
        x0, x1, y0, y1 = X._REGION_BOUNDS[region]
        x, y, w, h = boxes[e.key]
        assert x0 <= x - 0.5 * w and x + 0.5 * w <= x1 and y1 <= y - 0.5 * h and y + 0.5 * h <= y0, e.key


def test_an_enclosure_holds_its_alternatives_and_nothing_else():
    boxes = _technical().model["boxes"]
    for group in X.ALTERNATIVES:
        outline = next(it for it in _technical().scene.items if getattr(it, "role", "") == "alternative:" + ",".join(group))
        xs, ys = [p[0] for p in outline.points], [p[1] for p in outline.points]
        for key, (x, y, w, h) in boxes.items():
            if key not in X.ELEMENT:
                continue
            inside = min(xs) < x < max(xs) and min(ys) < y < max(ys)
            assert inside == (key in group), (group, key)


def test_no_python_path_is_drawn():
    for diagram in (_technical(), _concept()):
        for it in diagram.scene.items:
            if isinstance(it, Label):
                assert "vaft." not in it.text and "_workflow" not in it.text, it.text


def test_the_documentation_table_is_generated_from_the_module():
    assert X.cross_domain_markdown() in DIAGRAMS_MD.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- the conceptual figure


def test_every_technical_input_is_summarized_by_one_context_card_of_its_kind():
    inputs = {e.key for e in X.ELEMENTS if e.kind in ("measured", "machine", "prior", "convention")}
    covered = [k for card in C.CONTEXT for k in card.covers]
    assert sorted(covered) == sorted(inputs)
    for card in C.CONTEXT:
        assert {X.ELEMENT[k].kind for k in card.covers} == {card.kind}, card.key
    assert set(_ids_names()) >= {i for card in C.CONTEXT for i in card.idss}
    for card in C.CONTEXT:  # each IDS printed on a card is one a covered element is held in or written by
        held = {root for k in card.covers for ids in X.ELEMENT[k].ids for root in _ids_roots(ids)}
        written = {api.rsplit(".", 1)[-1] for k in card.covers for api in X.ELEMENT[k].apis}
        assert set(card.idss) <= held | written, (card.key, set(card.idss) - held - written)
    assert "coils_non_axisymmetric" in next(c for c in C.CONTEXT if c.key == "machine").idss


def test_data_and_metadata_stay_apart():
    """Data is what was measured; metadata is the context needed to interpret it and run an operation."""
    groups = {title: keys for title, _, keys in C.CONTEXT_GROUPS}
    assert list(groups) == ["Data", "Metadata"]
    kinds = {title: {c.kind for c in C.CONTEXT if c.key in keys} for title, keys in groups.items()}
    assert kinds == {"Data": {"measured"}, "Metadata": {"machine", "prior", "convention"}}
    assert sorted(k for keys in groups.values() for k in keys) == sorted(c.key for c in C.CONTEXT)
    titles = {it.role: it.text for it in _concept().scene.items if isinstance(it, Label)}
    assert "Shared data \\& metadata" in titles["region title:context"]


def test_every_technical_method_is_one_operation_with_the_same_name_and_domain():
    covered = [k for op in C.OPERATION.values() for k in op.covers]
    assert sorted(covered) == sorted(_methods())
    for domain, ops in C.OPERATIONS.items():
        for op in ops:
            assert {X.ELEMENT[k].domain for k in op.covers} == {domain}, op.key
            assert op.title.replace("\n", " ") in {X.ELEMENT[k].label for k in op.covers}, op.key  # one vocabulary
            assert op.kind in {X.ELEMENT[k].kind for k in op.covers}, op.key
            if len(op.covers) > 1:  # alternatives in one card: each code on its own line, in order, model under it
                assert tuple(op.covers) in X.ALTERNATIVES, op.key
                codes = [X.ELEMENT[k].method for k in op.covers]
                assert [line for line in op.lines if not line.startswith("\\itshape")] == codes, op.key
    # GPEC reads DCON's eigenfunctions; with no room for an arrow, the card says so
    assert any("DCON" in line for line in C.OPERATION["response"].lines) and ("dcon", "gpec") in _edges("flow")
    turbulent = C.OPERATION["turbulent"].lines
    assert turbulent[turbulent.index("TGLF") + 1].endswith("gyrofluid")
    assert turbulent[turbulent.index("CGYRO") + 1].endswith("gyrokinetic")


def test_alternatives_and_flows_are_those_of_the_technical_figure():
    for group in C.ALTERNATIVES:
        assert tuple(k for op in group for k in C.OPERATION[op].covers) in X.ALTERNATIVES
    tech_flows = set(_edges("flow"))
    for src, dst, _ in C.EDGES:
        if src in C.OPERATION and dst in C.OPERATION:
            # an operation consumes another directly or through the IDS entry the first one writes
            outputs = set(C.OPERATION[src].covers) | {d for s, d in _edges("mapping") if s in C.OPERATION[src].covers}
            assert any((o, t) in tech_flows for o in outputs for t in C.OPERATION[dst].covers), (src, dst)
    edges = {(s, d): t for s, d, t in C.EDGES}
    assert ("hub", "mhd") in tech_flows and ("eq_reconstructed", "state") in tech_flows
    # from the equilibrium section, not from one entry: eligibility is per model
    assert "eligible" in edges[("equilibrium", "mhd")] and "eligible" in edges[("equilibrium", "transport")]
    # the reconstruction requirement binds only the resolved state, which this figure does not draw (Sauter /
    # Redl reads the ODS), so the concept says nothing about it; the technical figure draws it on the state
    assert "reconstructed" not in edges[("equilibrium", "transport")]
    assert not [e for e in C.EDGES if {e[0], e[1]} == {"mhd", "transport"}]
    assert {("context", "equilibrium"), ("context", "mhd")} <= set(edges)


def test_each_equilibrium_entry_sits_inside_the_equilibrium_section_under_its_operation():
    boxes = _concept().model["boxes"]
    writes = {d: s for s, d in _edges("mapping")}
    for entry in C.EQUILIBRIUM_ENTRIES:
        (producer,) = [op for op in C.OPERATIONS["equilibrium"] if writes[entry.covers[0]] in op.covers]
        assert abs(boxes[entry.key][0] - boxes[producer.key][0]) < 1e-9 and entry.title == X.ELEMENT[entry.covers[0]].label
        assert boxes[entry.key][1] < boxes[producer.key][1]
    titles = {it.role: it.text for it in _concept().scene.items if isinstance(it, Label)}
    subtitle = titles["region subtitle:equilibrium"]
    assert "one IDS format" in subtitle and "not one plasma state" in subtitle


def test_results_cover_every_technical_result_once_and_print_the_ids_they_write():
    results = {e.key for e in X.ELEMENTS if e.kind in ("standardized", "native_result") and e.domain != "equilibrium"}
    assert sorted(k for item in C.RESULT.values() for k in item.covers) == sorted(results)
    for item, lines in C.RESULTS:
        assert {X.ELEMENT[k].kind for k in item.covers} == {item.kind}, item.key
        assert len({X.ELEMENT[k].domain for k in item.covers}) == 1, item.key
        printed = {ids for ids, _ in lines if ids}
        if item.kind == "standardized":
            written = {X.ELEMENT[k].ids_name for k in item.covers}
            sourced = {root for k in item.covers for ids in X.ELEMENT[k].ids for root in _ids_roots(ids)}
            assert written <= printed <= sourced, (item.key, printed, written, sourced)
        else:
            assert not printed, item.key
    # CGYRO linear only and not yet stored; NEO's j_bs in core_profiles
    transport = dict(next(lines for item, lines in C.RESULTS if item.key == "transport_imas"))
    assert "linear" in transport["gyrokinetics_local"] and "not yet stored" in transport["gyrokinetics_local"]
    assert "NEO" in transport["core_profiles"]


def test_one_output_section_separates_implemented_mappings_from_native_outputs():
    diagram = _concept()
    titles = {it.role: it.text for it in diagram.scene.items if isinstance(it, Label)}
    assert C.OUTPUT_TITLE == "IMAS-compatible representations with complementary native results"
    assert C.OUTPUT_TITLE in titles["region title:outputs"]
    assert [r for r, *_ in C.OUTPUT_ROWS] == ["imas", "native"]
    assert "Implemented" in titles["output row:imas"] and "no IMAS mapping implemented" in titles["output row:native"]
    regions = [it.role for it in diagram.scene.items if (getattr(it, "role", "") or "").startswith("region:")]
    assert regions.count("region:outputs") == 1 and not {"region:imas", "region:native", "region:equilibrium_ids"} & set(regions)
    boxes = diagram.model["boxes"]
    for item in C.RESULT.values():  # a mapped result above every native one
        row = "imas" if item.kind == "standardized" else "native"
        assert abs(boxes[item.key][1] - C._OUTPUT_ROW[row][1]) < 1e-9, item.key


def test_the_conceptual_figure_shares_the_technical_vocabulary_and_reads_clearly():
    items = _concept().scene.items
    titles = {it.role: it.text for it in items if isinstance(it, Label)}
    for key in ("equilibrium", "mhd", "transport"):
        assert dict((k, t) for k, t, _ in X.REGIONS)[key] in titles[f"region title:{key}"]
    # the limit of an arrow is in the legend, not a separate footnote
    assert "not a validated workflow" in titles["legend:flow"] and "footnote" not in titles
    concept_styles = {it.style for it in items if (getattr(it, "role", "") or "").startswith("alternative:")}
    assert concept_styles == {X.CONNECTION_STYLE["alternative"]}
    # no pale annotation: every gray label is at least 75 % black, and no wedge connector
    for it in items:
        if isinstance(it, Label):
            assert "black!60" not in it.text + it.style and (it.style.split(",")[0] != "concept annotation"
                                                            or "text=black!80" in it.style), it.role
    assert not [it for it in items if (getattr(it, "role", "") or "") == "edge:context"]


def test_no_two_concept_cards_overlap_and_each_sits_in_its_section():
    boxes = _concept().model["boxes"]
    section = {**{c.key: "context" for c in C.CONTEXT},
               **{op.key: d for d, ops in C.OPERATIONS.items() for op in ops},
               **{e.key: "equilibrium" for e in C.EQUILIBRIUM_ENTRIES},
               **{item.key: "outputs" for item in C.RESULT.values()}}
    for a, b in itertools.combinations(sorted(section), 2):
        (ax, ay, aw, ah), (bx, by, bw, bh) = boxes[a], boxes[b]
        assert abs(ax - bx) >= 0.5 * (aw + bw) or abs(ay - by) >= 0.5 * (ah + bh), (a, b)
    for key, region in section.items():
        x0, x1, y0, y1 = C._BOUNDS[region]
        x, y, w, h = boxes[key]
        assert x0 <= x - 0.5 * w and x + 0.5 * w <= x1 and y1 <= y - 0.5 * h and y + 0.5 * h <= y0, key
    # cards of one kind of row share one size
    for domain in ("mhd", "transport"):
        assert len({boxes[op.key][3] for op in C.OPERATIONS[domain]}) == 1
    assert len({boxes[e.key][2:] for e in C.EQUILIBRIUM_ENTRIES}) == 1


def _label_sizes(diagram):
    """The nominal size [pt] of every drawn label: its own size commands, else its style's font."""
    from vaft.diagram._render import template

    fonts = dict(re.findall(r"^\s*([a-z ]+)/\.style=\{font=(\\[a-z]+)", template(), re.M))
    named = {"\\normalsize": 10.0, "\\large": 12.0, **_POINTS}
    sizes = []
    for it in diagram.scene.items:
        if not isinstance(it, Label):
            continue
        own = [pt for name, pt in _POINTS.items() if name in it.text]
        style = fonts.get(it.style.split(",")[0].strip(), "\\normalsize")
        sizes += own or [named.get(style, 10.0)]
    return sizes


@pytest.mark.parametrize("module, width_cm", [(C, MANUSCRIPT_WIDTH_CM), (X, 24.0)])
def test_text_stays_legible_at_its_print_width(module, width_cm):
    """The concept figure at the 180 mm manuscript column, the technical one at a landscape page: the
    smallest nominal label size, scaled with the figure, stays at or above 5.5 pt (a box that shrinks its
    text to fit is measured from the rendered PDF in the handoff, not here)."""
    diagram = _concept() if module is C else _technical()
    scale = width_cm / (module._WIDTH + 0.3)  # the standalone border adds 4 pt a side
    sizes = _label_sizes(diagram)
    assert min(sizes) * scale >= 5.5, (min(sizes), scale)
