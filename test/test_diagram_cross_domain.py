"""The cross-domain figure: equilibrium, MHD and transport joined through shared representations.

Every element is drawn from the workflow spine (#1585), names real APIs and Data Dictionary leaves,
and makes only the claims the implementation supports: a shared IDS is not a shared state, MHD and
transport are complementary, and only the models that read the resolved state share its identity.
"""

import inspect
import itertools
import re

import pytest

import vaft.diagram
from vaft.diagram import _cross_domain as X
from vaft.diagram._scene import Arrow, Label, Polyline
from vaft.diagram._workflow import resolve
from vaft.diagram._workflow_specs import WORKFLOWS

from test_diagram_workflows import DIAGRAMS_MD, _ids_names, _ids_roots

REGIONS = {key for key, *_ in X.REGIONS}


def _edges(connection=None):
    return [(s, d) for s, d, c, _ in X.EDGES if connection in (None, c)]


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


def test_context_tiles_name_real_ids():
    known = _ids_names()
    for e in X.ELEMENTS:
        assert set(e.idss) <= known, (e.key, e.idss)
    machine = X.ELEMENT["machine"]
    assert "coils_non_axisymmetric" in machine.idss and machine.api.endswith(".coils_non_axisymmetric")


def test_edges_join_known_ends_with_one_of_four_connections():
    known = set(X.ELEMENT) | REGIONS
    assert X.CONNECTIONS == ("flow", "alternative", "mapping", "native")
    for src, dst, connection, _ in X.EDGES:
        assert {src, dst} <= known and connection in X.CONNECTIONS, (src, dst, connection)
    assert len(set(_edges())) == len(X.EDGES)
    # every element is connected: nothing floats unexplained
    ends = {k for edge in _edges() for k in edge}
    assert set(X.ELEMENT) - {"measurements", "assumptions", "machine"} <= ends


def test_the_four_connections_differ_without_colour():
    from vaft.diagram._render import template

    styles = X.CONNECTION_STYLE
    flow = re.search(r"^\s*" + re.escape(styles["flow"]) + r"/\.style=\{(.*)\},?$", template(), re.M).group(1)
    assert "->" in flow and "dash" not in flow and "dotted" not in flow
    assert "dash dot" in styles["alternative"] and "->" not in styles["alternative"]
    assert "open" in styles["mapping"] and "dash" not in styles["mapping"]
    assert "dashed" in styles["native"] and "->" in styles["native"]


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
    assert not [e for e in _edges("alternative") if "chease" in e]


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


def test_alternatives_stay_inside_one_domain_and_mhd_methods_are_complementary():
    for src, dst in _edges("alternative"):
        assert X.ELEMENT[src].domain == X.ELEMENT[dst].domain, (src, dst)
    for a, b in itertools.combinations(("dcon", "rdcon", "gpec"), 2):
        assert (a, b) not in _edges("alternative") and (b, a) not in _edges("alternative")
    assert set(_edges("alternative")) == {("analytic", "tokamaker"), ("tokamaker", "efit"),
                                          ("sauter_redl", "neo"), ("tglf", "cgyro")}


def test_mhd_and_transport_are_never_chained():
    for src, dst in _edges():
        domains = {X.ELEMENT[k].domain if k in X.ELEMENT else k for k in (src, dst)}
        assert domains != {"mhd", "transport"}, (src, dst)


def test_only_the_models_reading_the_resolved_state_share_its_identity():
    readers = {d for s, d in _edges("flow") if s == "state"}
    assert readers == {"neo", "classical", "tglf", "cgyro"} == set(X.STATE_IDENTITY_JOIN)
    # the identity keys runs, not IDS entries: the NEO writer records none, so no join is drawn on the results
    assert not [it for it in vaft.diagram.cross_domain_scientific_workflows().scene.items
                if "join" in (getattr(it, "role", "") or "")]
    assert X.ELEMENT["state"].api == "vaft.process.transport_state.ResolvedTransportState.identity"
    # Sauter / Redl read the IMAS state directly and are retained, not mapped
    assert "sauter_redl" not in readers
    assert {d for s, d in _edges() if s == "sauter_redl"} == {"neo", "sr_native"}
    assert not [d for s, d in _edges("mapping") if s == "sauter_redl"]


def test_cgyro_is_drawn_as_linear_eigenmodes_only():
    assert [d for s, d in _edges("mapping") if s == "cgyro"] == ["gk_cgyro"]
    assert X.ELEMENT["gk_cgyro"].dd_path.startswith("gyrokinetics_local.linear.")
    assert "linear" in X.ELEMENT["cgyro"].basis
    assert not [e for e in X.ELEMENTS if e.dd_path and "non_linear" in e.dd_path]


def test_mapping_caveats_are_stated_where_the_quantity_does_not_match_exactly():
    assert "normalized" in X.ELEMENT["ml_dcon"].detail and "J" in X.ELEMENT["ml_dcon"].note
    assert "diagonal" in X.ELEMENT["ntms_rdcon"].detail
    assert "identifier.index" in X.ELEMENT["ct_classical"].note
    assert X.ELEMENT["gpec_coils"].kind == "machine"  # coils_non_axisymmetric is an input, not a GPEC result


def test_the_builder_is_public_registered_and_deterministic():
    fn = vaft.diagram.cross_domain_scientific_workflows
    assert "cross_domain_scientific_workflows" in vaft.diagram.__all__
    from vaft.diagram.build import CANONICAL

    assert CANONICAL["cross_domain_scientific_workflows.svg"] == ("cross_domain_scientific_workflows", {})
    assert fn().tikz == fn().tikz
    assert not [it for it in fn(labels=False).scene.items if isinstance(it, Label)]
    with pytest.raises(ValueError):
        fn(labels="yes")


def test_every_edge_is_drawn_once_in_its_style_and_the_legend_shows_each_connection():
    items = vaft.diagram.cross_domain_scientific_workflows().scene.items
    for src, dst, connection, text in X.EDGES:
        drawn = [it for it in items if getattr(it, "role", "") == f"edge:{connection}:{src}->{dst}"]
        assert len(drawn) == 1 and drawn[0].style == X.CONNECTION_STYLE[connection], (src, dst)
        assert bool(text) == ((src, dst) in X._LABEL_AT), (src, dst)
    for connection in X.CONNECTIONS:
        assert [it for it in items if isinstance(it, (Arrow, Polyline)) and it.role == f"legend:{connection}"]


def test_no_two_element_boxes_overlap_and_each_sits_in_its_region():
    boxes = vaft.diagram.cross_domain_scientific_workflows().model["boxes"]
    keys = sorted(X.ELEMENT)
    for a, b in itertools.combinations(keys, 2):
        (ax, ay, aw, ah), (bx, by, bw, bh) = boxes[a], boxes[b]
        assert abs(ax - bx) >= 0.5 * (aw + bw) or abs(ay - by) >= 0.5 * (ah + bh), (a, b)
    for e in X.ELEMENTS:
        region = "hub" if e.domain == "equilibrium" and e.kind == "standardized" else e.domain
        x0, x1, y0, y1 = X._REGION_BOUNDS[region]
        x, y, w, h = boxes[e.key]
        assert x0 <= x - 0.5 * w and x + 0.5 * w <= x1 and y1 <= y - 0.5 * h and y + 0.5 * h <= y0, e.key


def test_no_python_path_is_drawn():
    for it in vaft.diagram.cross_domain_scientific_workflows().scene.items:
        if isinstance(it, Label):
            assert "vaft." not in it.text and "_workflow" not in it.text, it.text


def test_the_documentation_table_is_generated_from_the_module():
    assert X.cross_domain_markdown() in DIAGRAMS_MD.read_text(encoding="utf-8")
