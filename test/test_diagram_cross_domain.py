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
from vaft.diagram import _workflow as W
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


def test_data_and_metadata_stay_apart():
    """Data is what was measured; metadata is the context needed to interpret it and run an operation."""
    groups = dict(C.CONTEXT_GROUPS)
    assert list(groups) == ["Experimental data", "Metadata"]
    kinds = {title: {c.kind for c in C.CONTEXT if c.key in keys} for title, keys in groups.items()}
    assert kinds == {"Experimental data": {"measured"}, "Metadata": {"machine", "prior", "convention"}}
    assert sorted(k for keys in groups.values() for k in keys) == sorted(c.key for c in C.CONTEXT)
    titles = {it.role: it.text for it in _concept().scene.items if isinstance(it, Label)}
    assert "Shared data \\& metadata" in titles["region title:context"]


def test_every_technical_method_is_one_operation_with_the_same_name_and_domain():
    covered = [k for op in C.OPERATION.values() for k in op.covers]
    assert sorted(covered) == sorted(_methods())
    for domain, ops in C.OPERATIONS.items():
        for op in ops:
            (k,) = op.covers  # one code per card
            assert X.ELEMENT[k].domain == domain, op.key
            # the method kind is the primary computational approach: analytical model / formula or numerical solver
            assert op.kind == ("derived" if op.key in C.ANALYTICAL else "solver"), op.key
            # one vocabulary: a card is titled by the technical label, or (inside a titled group) by its code
            in_group = any(op.key in g for g in C.GROUP_TITLE)
            assert op.title.replace("\n", " ") == (X.ELEMENT[k].method if in_group else X.ELEMENT[k].label), op.key
    assert C.OPERATION["response"].lines[-1].endswith("perturbed equilibria")
    assert C.OPERATION["tglf"].lines[-1].endswith("gyrofluid") and C.OPERATION["cgyro"].lines[-1].endswith("gyrokinetic")
    # the equilibrium formulations read: reconstruction, analytic, fixed-boundary, free-boundary
    assert [op.key for op in C.OPERATIONS["equilibrium"]] == ["reconstruction", "analytic", "refinement", "free_boundary"]
    assert set(C.ANALYTICAL) == {"analytic", "classical", "sauter_redl"}
    assert C.KIND_WORDS["derived"] == "analytical model / formula"
    assert C.KIND_WORDS["solver"] == "numerical method / solver"


def test_alternatives_produce_one_scientific_output_and_the_equilibria_are_peers():
    """A dash-dot group holds methods for the same output: a technical alternative set, or methods all writing one
    IDS. The four equilibrium formulations are peers in one group -- no edge runs between them."""
    tech_flows = set(_edges("flow"))
    writes = {}
    for s_, d in _edges("mapping"):
        writes.setdefault(s_, set()).add(X.ELEMENT[d].ids_name)
    for group in C.ALTERNATIVES:
        methods = tuple(k for op in group for k in C.OPERATION[op].covers)
        assert len({X.ELEMENT[k].domain for k in methods}) == 1, group
        same_ids = len(set.intersection(*(writes.get(k, set()) for k in methods))) == 1
        assert methods in X.ALTERNATIVES or same_ids, group
    eq = tuple(op.key for op in C.OPERATIONS["equilibrium"])
    assert eq in C.ALTERNATIVES  # all four, CHEASE included
    assert not [e for e in C.EDGES if e[0] in eq and e[1] in eq]
    # a titled group carries the technical figure's physics label of its members
    for group, title in C.GROUP_TITLE.items():
        assert group in C.ALTERNATIVES and title in {X.ELEMENT[k].label for op in group for k in C.OPERATION[op].covers}
    titles = {it.role: it.text for it in _concept().scene.items if isinstance(it, Label)}
    for group, title in C.GROUP_TITLE.items():
        assert title in titles["group title:" + ",".join(group)]
    for group in C.ALTERNATIVES:  # every member card inside its enclosure
        x, y, w, h = _concept().model["boxes"]["alternative:" + ",".join(group)]
        for op in group:
            ox, oy, ow, oh = _concept().model["boxes"][op]
            assert x - 0.5 * w < ox - 0.5 * ow and ox + 0.5 * ow < x + 0.5 * w and y - 0.5 * h < oy - 0.5 * oh, op
    assert ("hub", "mhd") in tech_flows and ("eq_reconstructed", "state") in tech_flows
    # one shared equilibrium state, written by the equilibrium section, branches to both consumers: no chain
    assert {("equilibrium", "state"), ("state", "mhd"), ("state", "transport")} <= set(C.EDGES)
    assert not [e for e in C.EDGES if set(e) == {"mhd", "transport"}]
    assert C.STATE.covers == ("eq_analytic", "eq_model", "eq_reconstructed", "eq_refined")
    # the shared data & metadata feed the equilibrium and the downstream input row
    assert {("context", "equilibrium"), ("context", "inputs")} <= set(C.EDGES)


def test_each_section_names_only_ids_its_operations_write():
    """Every IDS named has a writer for a covered element of the same domain; every technical IDS result is
    named; no native output and no Sauter / Redl result is given an IDS."""
    standardized = {e.key for e in X.ELEMENTS if e.kind == "standardized"}
    named = {k for entries in C.IMAS.values() for _, _, covers in entries for k in covers}
    assert named == standardized
    writes = {d: s for s, d in _edges("mapping")}
    for domain, entries in C.IMAS.items():
        for ids, quantities, covers in entries:
            for k in covers:
                assert X.ELEMENT[k].domain == domain and X.ELEMENT[k].kind == "standardized", (ids, k)
                roots = {root for ref in X.ELEMENT[k].ids for root in _ids_roots(ref)}
                assert ids in roots, (ids, k, roots)
                assert writes[k] != "sauter_redl"
    # a code is named where one card holds several and only one writes
    transport = {ids: q for ids, q, _ in C.IMAS["transport"]}
    assert "NEO" not in transport["core_profiles"] and writes["ct_neo"] == "neo"  # NEO writes it; not repeated
    # TGLF and CGYRO both have gyrokinetics_local writers (gyrokinetics.py): neither may be named alone
    assert "CGYRO" not in transport["gyrokinetics_local"] and "TGLF" not in transport["gyrokinetics_local"]
    # every native result is listed once, as native, in its own domain, and never beside an IDS
    native = [k for entries in C.NATIVE.values() for _, covers in entries for k in covers]
    assert sorted(native + list(C.OMITTED)) == sorted(e.key for e in X.ELEMENTS if e.kind == "native_result")
    assert C.OMITTED == ("gpec_native", "sr_native")  # representative lists; the technical figure has both
    for domain, entries in C.NATIVE.items():
        for _, covers in entries:
            assert {X.ELEMENT[k].domain for k in covers} == {domain}
    lists = {it.role: it.text for it in _concept().scene.items if isinstance(it, Label)}
    for domain in C.NATIVE:  # the native line has the italic key, no typewriter IDS
        native_line = lists[f"quantities:{domain}"].split("\\\\")[-1]
        assert "\\textit{native}" in native_line and "\\texttt" not in native_line
    assert "IMAS" not in " ".join(lists.values())
    # no implementation-status annotation in the figure
    text = " ".join(it.text for it in _concept().scene.items if isinstance(it, Label))
    for phrase in ("not yet", "stored", "not mapped", "implemented"):
        assert phrase not in text.replace("implemented interface", ""), phrase


def test_the_figure_does_not_narrate_its_own_layout():
    items = _concept().scene.items
    roles = {it.role for it in items if isinstance(it, Label)}
    assert not [r for r in roles if r.startswith(("region subtitle:", "edge label:", "output row:", "footnote"))]
    assert not [it for it in items if (getattr(it, "role", "") or "") in ("region:outputs", "region:equilibrium_ids")]
    assert not [it for it in items if (getattr(it, "role", "") or "").startswith("representation:")]
    titles = {it.role: it.text for it in items if isinstance(it, Label)}
    for key in ("equilibrium", "mhd", "transport"):
        assert C._TITLE[key] in titles[f"region title:{key}"]
        assert C._TITLE[key] == dict((k, t) for k, t, _ in X.REGIONS)[key] or key == "mhd"
    assert "MHD" not in titles["region title:mhd"]  # every domain is MHD-based; the title names stability only
    # sections are solid: a dash marks the alternatives' enclosure or a card's kind (assumption, convention), nothing else
    from vaft.diagram._render import template

    patterned = {name for name, body in re.findall(r"^\s*([a-z ]+)/\.style=\{(.*)\},?$", template(), re.M)
                 if "dash" in body or "dotted" in body}
    dashed = {it.role for it in items if isinstance(it, Polyline) and not (it.role or "").startswith("legend:")
              and ("dash" in it.style or "dotted" in it.style or it.style.split(",")[0] in patterned)}
    cards = _cards()
    assert dashed - {"alternative:" + ",".join(g) for g in C.ALTERNATIVES} == \
        {role for role, card in cards.items() if W.kind_style(card.kind) in patterned}
    # one legend: each kind on a card, with its pictogram, then the two line styles
    legend = {it.role for it in items if (getattr(it, "role", "") or "").startswith("legend:")}
    assert legend == ({f"legend:{c.kind}" for c in cards.values()}
                      | {"legend:flow", "legend:write", "legend:alternative"})
    # the equilibrium operations *write* the shared state: an open head, not the input arrow
    writes = [it for it in items if (getattr(it, "role", "") or "").startswith("edge:write:")]
    assert [it.role for it in writes] == ["edge:write:equilibrium->state"] and writes[0].style == C.WRITE_STYLE
    assert "open" in C.WRITE_STYLE and C.WRITE_STYLE != X.CONNECTION_STYLE["flow"]
    assert not [it for it in items if isinstance(it, Label) and "black!60" in it.text + it.style]


def _cards():
    """Every drawn card by its role: the left panel, the operations, the downstream inputs and the shared state."""
    return ({f"context:{c.key}": c for c in C.CONTEXT} | {f"operation:{k}": op for k, op in C.OPERATION.items()}
            | {f"input:{c.key}": c for row in C.INPUTS.values() for c in row} | {"state": C.STATE})


def test_downstream_inputs_are_the_shared_data_and_metadata_they_come_from():
    """Kinetic profiles and closures feed transport; modes, wall and excitation feed MHD -- each chip in the kind
    of the left-panel card that holds it, and the left panel's arrow meets that row."""
    held = {}
    for card in C.CONTEXT:
        for k in card.covers:
            held[k] = card.kind
    for domain, chips in C.INPUTS.items():
        for chip in chips:
            assert all(held[k] == chip.kind for k in chip.covers), chip.key
            reads = {d for s, d in _edges("flow") if s in chip.covers}
            assert {X.ELEMENT[d].domain for d in reads} == {domain}, (chip.key, reads)
    transport = {c.kind for c in C.INPUTS["transport"]}
    assert {"measured", "prior"} <= transport  # kinetic profiles and modelling assumptions, explicitly
    # no rotation: TGLF and CGYRO run with gamma_E = 0 (tglf/inputs.py, #553), so v_phi is no transport input
    measured = " ".join(line for c in C.INPUTS["transport"] if c.kind == "measured" for line in c.lines)
    assert "v_" not in measured and {"T_e", "n_e"} <= set(measured.replace("$", "").replace(",", " ").split())
    # the closures chip names the general inputs only: Z_eff and gamma_E, no T_i (a kinetic profile) and no value
    closures = " ".join(C.INPUTS["transport"][1].lines)
    assert "Z_{\\mathrm{eff}}" in closures and "gamma_E" in closures and "T_i" not in closures and "=" not in closures
    models = " ".join(next(c for c in C.CONTEXT if c.key == "models").lines)
    assert "T_i" not in models and "gamma_E" in models and "=" not in models
    (start, end) = C._ROUTES[("context", "inputs")]
    assert abs(start[1] - C._INPUT_Y) < 0.5 * C._INPUT_H and end[0] < C._BOUNDS["mhd"][0]


def test_every_card_carries_its_kinds_pictogram_in_its_kinds_style():
    items = _concept().scene.items
    cards = _cards()
    for role, card in cards.items():
        outline = [it for it in items if isinstance(it, Polyline) and it.role == role]
        assert len(outline) == 1 and outline[0].style == W.kind_style(card.kind), role
        symbol = [it for it in items if getattr(it, "role", "") == f"symbol:{role}"]
        legend = [it for it in items if getattr(it, "role", "") == f"symbol:legend:{card.kind}"]
        assert symbol and len(symbol) == len(legend), role  # the same pictogram as the legend's, drawn once
        (x, y, w, h) = _concept().model["boxes"][card.key if role != "state" else "state"]
        xs = [p[0] for it in symbol for p in it.points]
        ys = [p[1] for it in symbol for p in it.points]
        assert x - 0.5 * w < min(xs) and max(xs) < x + 0.5 * w and y - 0.5 * h < min(ys) and max(ys) < y + 0.5 * h, role


def test_no_two_concept_cards_overlap_and_each_sits_in_its_section():
    boxes = _concept().model["boxes"]
    section = {**{c.key: "context" for c in C.CONTEXT}, **{op.key: d for d, ops in C.OPERATIONS.items() for op in ops},
               **{c.key: d for d, row in C.INPUTS.items() for c in row}}
    for a, b in itertools.combinations(sorted(section), 2):
        (ax, ay, aw, ah), (bx, by, bw, bh) = boxes[a], boxes[b]
        assert abs(ax - bx) >= 0.5 * (aw + bw) or abs(ay - by) >= 0.5 * (ah + bh), (a, b)
    for key, region in section.items():
        x0, x1, y0, y1 = C._BOUNDS[region]
        x, y, w, h = boxes[key]
        assert x0 <= x - 0.5 * w and x + 0.5 * w <= x1 and y1 <= y - 0.5 * h and y + 0.5 * h <= y0, key
    for domain in ("equilibrium", "mhd"):
        assert len({boxes[op.key][2:] for op in C.OPERATIONS[domain]}) == 1
    for group in C.GROUP_TITLE:  # the enclosed codes share one size
        assert len({boxes[op][2:] for op in group}) == 1
    # a balanced transport row: the groups share one width and height, their rows align (Sauter / Redl with TGLF,
    # NEO with CGYRO), and Classical and the groups share the MHD cards' top and bottom edges
    neo, turb = (boxes["alternative:" + ",".join(g)] for g in C.GROUP_TITLE)
    assert neo[2:] == turb[2:]
    assert boxes["sauter_redl"][1] == boxes["tglf"][1] and boxes["neo"][1] == boxes["cgyro"][1]
    top = lambda b: b[1] + 0.5 * b[3]
    bottom = lambda b: b[1] - 0.5 * b[3]
    for b in (neo, turb, boxes["classical"]):
        assert abs(top(b) - top(boxes["ideal"])) < 1e-9 and abs(bottom(b) - bottom(boxes["ideal"])) < 1e-9
    # the shared state sits between the equilibrium section and its consumers, overlapping none
    sx, sy, sw, sh = boxes["state"]
    assert C._BOUNDS["equilibrium"][3] > sy + 0.5 * sh and sy - 0.5 * sh > C._BOUNDS["mhd"][2]
    # each downstream section's IDS list sits below its cards, inside the section
    for domain in ("mhd", "transport"):
        lowest = min(boxes[op.key][1] - 0.5 * boxes[op.key][3] for op in C.OPERATIONS[domain])
        assert C._BOUNDS[domain][3] < C._LIST_TOP[domain] < lowest, domain


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
