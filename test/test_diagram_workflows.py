"""The physics-workflow spine (#1585): specs that name real APIs, real formulas and real IDS."""

import inspect
import re
from pathlib import Path

import pytest

import vaft.diagram
from vaft.diagram import _workflow as W
from vaft.diagram._equations import formula_equation
from vaft.diagram._scene import Label
from vaft.diagram._workflow_specs import WORKFLOWS

SPECS = list(WORKFLOWS.values())
DIAGRAMS_MD = Path(__file__).resolve().parents[1] / "docs" / "_guide" / "Diagrams.md"


def _ids_names():
    from omas.omas_setup import omas_rcparams
    from omas.omas_utils import list_structures

    return set(list_structures(omas_rcparams["default_imas_version"]))


def _ids_roots(path):
    """The IDS names a node's path starts from: ``a.b``, ``{a | b}.c`` and ``a.x; b`` forms."""
    roots = []
    for alternative in path.split(";"):
        alternative = alternative.strip()
        if alternative.startswith("{"):
            roots += [re.split(r"[.\[]", name.strip(), maxsplit=1)[0]
                      for name in re.split(r"[|,]", alternative[1:alternative.index("}")])]
        else:
            roots.append(re.split(r"[.\[{]", alternative, maxsplit=1)[0])
    return roots


def test_the_spine_reads_from_inference_through_reconstruction_to_transport():
    assert list(WORKFLOWS) == [
        "plasma_parameter_inference", "romero_transformer_balance", "resistive_zeff_inference", "magnetic_efit",
        "kinetic_efit", "analytic_mhd_equilibrium", "chease_coupling", "tokamaker_coupling", "dcon_rdcon_stability",
        "gpec_plasma_response", "flare_field_line_topology", "neo_neoclassical", "tglf_cgyro_local_transport",
        "parameter_inference_overview", "parameter_inference_dependency_graph", "eddy_current_model",
        "kinetic_profile_fitting", "transport_hierarchy", "neoclassical_model_comparison",
        "machine_model_history", "signal_processing", "quality_validation", "fault_anomaly_detection",
        "shot_classification", "event_detection",
        "vacuum_field_proxies", "derived_mhd_parameters", "synthetic_diagnostics",
        "coordinate_cocos_conversions", "power_balance_confinement"]


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.key)
def test_every_api_and_equation_resolves_on_this_tree(spec):
    for node in spec.nodes:
        if node.api:
            assert callable(W.resolve(node.api)), node.api
        if node.equation:
            fn = W.resolve(node.equation)
            assert fn.__module__.startswith("vaft.formula."), node.equation
            assert formula_equation(fn)  # the catalog documents an equation for it


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.key)
def test_ids_paths_start_from_imas_structures(spec):
    known = _ids_names()
    for node in spec.nodes:
        if node.ids:
            assert set(_ids_roots(node.ids)) <= known, f"{spec.key}: {node.ids}"


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.key)
def test_status_and_kind_are_explicit(spec):
    assert spec.status in W.STATUSES
    for node in spec.nodes:
        assert node.kind in W.KINDS
        assert node.status is None or node.status in W.STATUSES


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.key)
def test_priors_conventions_and_machine_data_enter_from_the_side(spec):
    d = vaft.diagram.__getattribute__(spec.key)()
    centers = d.model["centers"]
    for src, dst in spec.side:
        assert spec.node(src).kind in W.SIDE_KINDS, src
        assert abs(centers[src][0] - centers[dst][0]) >= W._W + 0.5  # beside the step it enters, not in the chain
    placed = [k for row in spec.rows for k in row if k is not None]
    row_of = {k: r for r, row in enumerate(spec.rows) for k in row if k is not None}
    measured = [k for k in placed if spec.node(k).kind == "measured"]
    for key in placed:  # a prior in the main chain is never above a measurement
        if spec.node(key).kind == "prior":
            assert all(row_of[key] >= row_of[m] for m in measured), key


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.key)
def test_no_two_boxes_overlap(spec):
    d = vaft.diagram.__getattribute__(spec.key)()
    boxes = {}
    for it in d.scene.items:
        if isinstance(it, Label) and it.role.startswith("node:"):
            continue
        role = getattr(it, "role", "") or ""
        if role.startswith("node:") and hasattr(it, "points"):
            xs, ys = zip(*it.points)
            boxes[role] = (min(xs), max(xs), min(ys), max(ys))
    keys = sorted(boxes)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            ax0, ax1, ay0, ay1 = boxes[a]
            bx0, bx1, by0, by1 = boxes[b]
            assert ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0, (a, b)


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.key)
def test_equations_shown_are_the_catalog_ones(spec):
    d = vaft.diagram.__getattribute__(spec.key)()
    text = {it.role.split(":", 1)[1]: it.text for it in d.scene.items
            if isinstance(it, Label) and it.role.startswith("node:")}
    for node in spec.nodes:
        if node.equation:  # drawn inside the node, under its label, one line per joined relation
            parts = W._equation_parts(node)
            assert parts and all(part in text[node.key] for part in parts)
            def flat(tex):
                return tex.replace("\\qquad", "").replace(" ", "").replace(",", "")

            assert flat("".join(parts)) in flat(formula_equation(W.resolve(node.equation)))
        for part in W._symbol_parts(node):
            assert f"${part}$" in text[node.key]
        if node.ids:
            assert W.escape_latex(node.ids) in text[node.key]
        if node.mapping_todo:
            assert "IMAS mapping TODO: " + W.escape_latex(node.mapping_todo) in text[node.key]


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.key)
def test_no_status_word_is_drawn(spec):
    d = vaft.diagram.__getattribute__(spec.key)()
    status_word = re.compile(r"status:|(?<![\\a-z])(implemented|partial|planned|design shell)\b")  # not \partial
    assert not [it.text for it in d.scene.items if isinstance(it, Label) and status_word.search(it.text)]
    assert spec.status == "implemented" and all(node.status is None for node in spec.nodes)


def test_inputs_and_outputs_carry_their_variables():
    for spec in SPECS:
        for row in (spec.rows[0], spec.rows[-1]):
            for key in row:
                if key is not None and spec.node(key).kind in ("measured", "standardized"):
                    assert spec.node(key).symbols, f"{spec.key}: {key}"


def test_resistive_zeff_is_inferred_and_built_on_the_romero_balance():
    romero, zeff = WORKFLOWS["romero_transformer_balance"], WORKFLOWS["resistive_zeff_inference"]
    assert zeff.node("zeff").kind == "inferred" and zeff.node("observed").kind == "inferred"
    assert romero.node("rp").kind == "inferred"
    assert W.resolve(romero.node("voltages").api).__name__ == "romero_flux_balance"
    assert zeff.node("model").kind == zeff.node("lnlambda").kind == "convention"
    assert "zeff" not in zeff.node("eq").ids and "core_profiles" in zeff.node("cp").ids


def test_no_todo_claims_shipped_work_is_off_develop():
    # cold review 0.8.0 delta-absorb-17 species-docs F5: a TODO names a gap, not a branch state
    for spec in SPECS:
        for todo in spec.todos:
            assert "not on develop" not in todo.lower(), (spec.key, todo)


def test_unmapped_results_say_so_and_every_spec_lists_its_gaps_in_the_docs():
    tagged = {(spec.key, n.key) for spec in SPECS for n in spec.nodes if n.mapping_todo}
    assert {("plasma_parameter_inference", "state"), ("romero_transformer_balance", "rp"),
            ("resistive_zeff_inference", "zeff"), ("gpec_plasma_response", "mhd_linear"),
            ("tglf_cgyro_local_transport", "ct_tglf")} <= tagged
    docs = DIAGRAMS_MD.read_text(encoding="utf-8")
    for spec in SPECS:
        for todo in spec.todos:
            assert f"- {todo}" in docs, todo


def test_gpec_and_flare_are_separate_and_tglf_and_cgyro_outputs_are_distinct():
    gpec, flare = WORKFLOWS["gpec_plasma_response"], WORKFLOWS["flare_field_line_topology"]
    assert gpec.node("coil_geometry").kind == "machine" and flare.node("targets").kind == "machine"
    assert "delta\\mathbf B_{\\mathrm{vac}}" in gpec.node("fields").symbols
    assert flare.node("proxy").kind == "derived" and "rel" in flare.node("proxy").relation
    gk = WORKFLOWS["tglf_cgyro_local_transport"]
    assert {gk.node(k).kind for k in ("tglf_out", "cgyro_lin", "cgyro_nl")} == {"native_result"}
    assert gk.node("rotation").kind == "prior"


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.key)
def test_every_spec_is_a_builder_that_renders_deterministically(spec):
    fn = getattr(vaft.diagram, spec.key)
    assert spec.key in vaft.diagram.__all__
    assert fn().tikz == fn().tikz
    assert not [it for it in fn(labels=False).scene.items if isinstance(it, Label)]
    with pytest.raises(ValueError):
        fn(labels="yes")
    # the builder is a thin wrapper: its spec, not its own drawing
    assert "_render_workflow(WORKFLOWS[" in inspect.getsource(fn)


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.key)
def test_the_documentation_table_is_generated_from_the_spec(spec):
    rows = W.workflow_table(spec)
    assert [r["node"] for r in rows] == [n.label for n in spec.nodes]
    assert W.workflow_markdown(spec) in DIAGRAMS_MD.read_text(encoding="utf-8"), spec.key


def test_specs_refuse_inconsistent_structure():
    node = W.Node("a", "A", "measured")
    with pytest.raises(ValueError):
        W.Node("x", "X", "observed")  # not a kind
    with pytest.raises(ValueError):
        W.Node("r", "R", "derived", relation="a = b")  # a relation without the api that implements it
    with pytest.raises(ValueError):
        W.WorkflowSpec("k", "T", "f", "done", (node,), (("a",),), ())  # not a status
    with pytest.raises(ValueError):
        W.WorkflowSpec("k", "T", "f", "implemented", (node,), (), ())  # a node not placed
    with pytest.raises(ValueError):
        W.WorkflowSpec("k", "T", "f", "implemented", (node,), (("a",),), (("a", "b", ""),))  # unknown edge end
    with pytest.raises(ValueError):
        W.Node("y", "Y", "solver", api="code.efit.run")  # not a vaft path


def test_inference_figures_draw_only_implemented_closures_and_name_no_code():
    assert "synthetic" in W.KINDS and "synthetic" not in W.SIDE_KINDS  # the grammar keeps synthetic apart
    overview = WORKFLOWS["parameter_inference_overview"]
    assert {overview.node(k).kind for k in ("thermo", "composition", "resistive")} == {"inferred"}
    graph = WORKFLOWS["parameter_inference_dependency_graph"]
    for spec in (overview, graph):
        assert all(n.api for n in spec.nodes if n.kind == "inferred")  # nothing future is drawn
        drawn = " ".join(f"{n.label} {n.symbols or ''}" for n in spec.nodes).lower()
        assert not re.search(r"\b(tglf|cgyro|neo|chease|efit)\b", drawn)  # "neoclassical" is physics, not a code
    assert graph.node("gamma").kind == "prior"  # gamma_E = 0 is an assumption, not an inference (#553)


def test_references_are_optional_numbered_by_first_use_and_listed():
    spec = WORKFLOWS["parameter_inference_overview"]
    plain = vaft.diagram.parameter_inference_overview()
    cited = vaft.diagram.parameter_inference_overview(references=True)
    assert not [it for it in plain.scene.items if getattr(it, "role", "").startswith("reference:")]
    listed = [it.text for it in cited.scene.items if getattr(it, "role", "").startswith("reference:")]
    numbers = W.reference_numbers(spec)
    assert len(listed) == len(numbers) == len({r for n in spec.nodes for r in n.references}) > 0
    assert listed[0].startswith("[1] ") and "Lao" in listed[0]  # first cited node: the equilibrium
    resistive = next(it.text for it in cited.scene.items if isinstance(it, Label) and it.role == "node:resistive")
    assert "[3,4,5,6]" in resistive and "Spitzer" in resistive
    with pytest.raises(ValueError):
        vaft.diagram.parameter_inference_overview(references="yes")


def test_every_transport_route_starts_from_the_state_and_external_codes_are_optional():
    """#1901: no route feeds another; NEO, TGLF and CGYRO are optional external codes, each with its own writer."""
    spec = WORKFLOWS["transport_hierarchy"]
    resolved = ("classical", "neo_code", "turb_ql", "turb_gk")
    routes = resolved + ("neo_fit",)
    # Sauter/Redl reads the ODS (compute_bootstrap_current(ods, ...)), not the resolved state built from it
    assert {(src, dst) for src, dst, _ in spec.edges if dst in routes + ("state",)} == \
        {("state", r) for r in resolved} | {("source", "neo_fit"), ("source", "state")}
    for key in ("neo_code", "turb_ql", "turb_gk"):
        assert spec.node(key).kind == "solver" and "optional" in spec.node(key).symbols, key
    for key in ("classical", "neo_fit"):
        assert "software" in spec.node(key).symbols and "always" in spec.node(key).symbols, key
    # closures, not fluxes: Sauter/Redl has no writer and no flux
    assert not [e for e in spec.edges if e[0] == "neo_fit"] and "no" in spec.node("neo_fit").symbols
    # TGLF and CGYRO land in separate entries; CGYRO needs a saturation window
    assert ("turb_ql", "ct_tglf", "") in spec.edges and ("turb_gk", "ct_cgyro", "") in spec.edges
    assert ("window", "ct_cgyro") in spec.side
    for key in ("ct_classical", "ct_neo", "ct_tglf", "ct_cgyro"):
        assert "model[k]" in spec.node(key).ids and "identifier.index" in spec.node(key).ids, key


@pytest.mark.parametrize("key", ["transport_hierarchy", "neo_neoclassical"])
def test_only_the_steps_drawn_from_the_source_read_the_ods(key):
    """Sauter / Redl and the resolver read the ODS; everything downstream of the resolved state reads its outputs."""
    spec = WORKFLOWS[key]

    def first_parameter(node_key):
        return next(iter(inspect.signature(W.resolve(spec.node(node_key).api)).parameters))

    from_source = {dst for src, dst, _ in spec.edges if src == "source"}
    assert from_source == {"state", "neo_fit" if key == "transport_hierarchy" else "fits"}
    assert all(first_parameter(k) == "ods" for k in from_source)
    from_state = {dst for src, dst, _ in spec.edges if src == "state"}
    assert from_state and all(first_parameter(k) != "ods" for k in from_state if spec.node(k).api)
    # the fits are returned, not written: nothing leaves them
    assert not [e for e in spec.edges if e[0] in from_source - {"state"}]


def test_the_comparison_separates_the_shared_state_from_two_input_paths():
    spec = WORKFLOWS["neoclassical_model_comparison"]
    assert ("state", "analytic_in", "VAFT path") in spec.edges and ("state", "neo_in", "GACODE path") in spec.edges
    assert ("neo_in", "neo_cond", "") in spec.edges and "E_\\parallel = 1" in spec.node("neo_cond").symbols
    assert "not\\ ground\\ truth" in spec.node("compare").symbols
    assert "B_0" in spec.node("basis").symbols and "COCOS" in spec.node("mapping").symbols
    assert ("mapping", "compare", "reference") in spec.edges


def test_the_eddy_model_is_a_forward_solve_checked_against_vacuum_magnetics():
    spec = WORKFLOWS["eddy_current_model"]
    assert spec.node("solve").kind == "solver" and spec.node("currents").ids == "pf_passive.loop[:].current"
    assert spec.node("machine").kind == "machine" and spec.node("filaments").kind == "prior"
    assert ("vacuum", "check", "") in spec.edges and any("not fitted" in t for t in spec.todos)


def test_vacuum_products_are_returned_and_the_vessel_write_is_drawn():
    """#1901: the evaluators return dicts; the only IDS write is the vessel currents they solve when absent."""
    spec = WORKFLOWS["vacuum_field_proxies"]
    assert spec.node("vessel").kind == "standardized" and spec.node("vessel").ids.startswith("pf_passive.")
    assert ("pf", "vessel", "if absent") in spec.edges and ("vessel", "field", "") in spec.edges
    written = [n.key for n in spec.nodes if n.kind == "standardized"]
    assert written == ["vessel"]
    assert spec.node("summary").mapping_todo and any("No vacuum field" in t for t in spec.todos)
    assert "not\\ Wb/rad" in spec.node("field").symbols  # full weber from the Green's functions


def test_mhd_parameters_write_the_dd_leaves_and_return_the_rest():
    spec = WORKFLOWS["derived_mhd_parameters"]
    for key in ("beta_pol", "beta_tor", "li", "beta_n", "energy"):
        node = spec.node(key)
        assert node.kind == "standardized" and ".global_quantities." in node.ids and node.equation, key
    assert spec.node("beta_pol").equation.endswith("beta_poloidal_from_pressure_integral")  # the DD form
    assert "LCFS" in spec.node("descriptors").relation  # the boundary-average form, kept apart
    for key in ("descriptors", "edge_q"):
        assert "not written" in spec.node(key).mapping_todo, key
    assert ("reference", "surfaces") in spec.side and "tf.r0" in spec.node("reference").ids


def test_synthetic_diagnostics_are_the_three_forward_models_and_the_rest_is_a_gap():
    spec = WORKFLOWS["synthetic_diagnostics"]
    models = {n.api.rsplit(".", 1)[1] for n in spec.nodes if n.key in ("vacuum", "diamagnetic", "sxr")}
    assert models == {"synthetic_vacuum_magnetics", "compute_reconstructed_diamagnetic_flux",
                      "synthetic_island_soft_x_rays"}
    assert spec.node("island").kind == "prior" and ("chords", "sxr") in spec.side
    gap = spec.todos[0]
    assert all(word in gap for word in ("Thomson", "interferometer", "camera"))
    assert spec.node("vac_out").mapping_todo and spec.node("sxr_out").mapping_todo


def test_cocos_is_transformed_at_two_independent_layers():
    spec = WORKFLOWS["coordinate_cocos_conversions"]
    assert spec.node("normalize").api == "vaft.omas.general.equilibrium_psi_to_weber"
    assert spec.node("convert").api == "vaft.process.equilibrium.convert_cocos"
    ods_lane, record_lane = {"normalize", "ods11", "psi_norm", "grid", "rho_tor"}, {"record", "convert", "radial", "sfl"}
    assert not [e for e in spec.edges if {e[0], e[1]} & ods_lane and {e[0], e[1]} & record_lane]
    assert ("declared", "normalize") in spec.side and ("target", "convert") in spec.side
    assert "grid_type 11" in spec.node("grid").mapping_todo and not spec.node("grid").equation
    assert spec.node("sfl").equation.endswith("straight_field_line_angle")


def test_the_two_power_balances_stay_apart_and_write_no_summary():
    spec = WORKFLOWS["power_balance_confinement"]
    experimental, modelled = {"split", "balance", "tau", "evidence", "scaling"}, {"spitzer", "radiation", "budget"}
    assert not [e for e in spec.edges if {e[0], e[1]} & experimental and {e[0], e[1]} & modelled]
    assert spec.node("tau").api == "vaft.omas.confinement.ohmic_confinement_ods"
    assert spec.node("budget").api == "vaft.omas.formula_wrapper.compute_power_balance"
    for key in ("tau", "budget"):
        assert "not written to the summary IDS" in spec.node(key).mapping_todo, key
    assert "NaN" in spec.node("balance").symbols
    assert not [n for n in spec.nodes if n.ids and n.ids.startswith("summary")]


def test_every_platform_science_box_has_a_diagram_or_a_tracked_gap():
    from vaft.diagram import _architecture_coverage as C

    assert [(row.area, row.box) for row in C.COVERAGE] == C.platform_boxes()
    for row in C.COVERAGE:
        assert row.diagrams or re.search(r"#\d+", row.gaps), row.box
        for key in row.diagrams:
            assert key in vaft.diagram.__all__, key
    assert C.coverage_markdown() in DIAGRAMS_MD.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# kind pictograms in the shared legend (#2114)
# --------------------------------------------------------------------------


def test_every_kind_has_one_distinct_registered_symbol():
    from vaft.diagram import pictograms

    assert list(W.KIND_SYMBOL) == list(W.KINDS)
    assert len(set(W.KIND_SYMBOL.values())) == len(W.KINDS)  # no two kinds merged under one metaphor
    for symbol in W.KIND_SYMBOL.values():
        pictograms.symbol(symbol)


@pytest.mark.parametrize("spec", SPECS, ids=[s.key for s in SPECS])
def test_the_legend_pairs_each_used_kind_with_its_symbol(spec):
    from vaft.diagram import pictograms
    from vaft.diagram._scene import Polyline

    scene = getattr(vaft.diagram, spec.key)().scene
    used = [k for k in W.KINDS if any(n.kind == k for n in spec.nodes)]
    legend_kinds = list(dict.fromkeys(i.role[len("legend:"):] for i in scene.items if i.role.startswith("legend:")))
    assert legend_kinds == used  # the kinds drawn, in KINDS order
    for kind in used:
        swatch, label = scene.role(f"legend:{kind}")  # the existing swatch and label, unchanged in role and order
        assert isinstance(swatch, Polyline) and swatch.style.startswith(W.kind_style(kind))
        assert label.text == W.escape_latex(W.KIND_LABEL[kind])
        icon = scene.role(f"symbol:legend:{kind}")
        assert len(icon) == len(pictograms.symbol(W.KIND_SYMBOL[kind]).strokes)
        # the pictogram sits inside its swatch, and the label follows on the same line
        sx = [p[0] for p in swatch.points]
        sy = [p[1] for p in swatch.points]
        pts = [p for i in icon for p in i.points]
        assert all(min(sx) < x < max(sx) and min(sy) < y < max(sy) for x, y in pts)
        assert max(sx) < label.at[0] and min(sy) < label.at[1] < max(sy)
    assert {i.role for i in scene.items if i.role.startswith("symbol:")} == {f"symbol:legend:{k}" for k in used}


@pytest.mark.parametrize("spec", SPECS, ids=[s.key for s in SPECS])
def test_legend_pictograms_go_with_the_legend(spec, monkeypatch):
    bare = getattr(vaft.diagram, spec.key)(labels=False)
    assert not [i for i in bare.scene.items if i.role.startswith(("symbol:", "legend:"))]
    # with the pictograms stubbed out the figure is what it was: nodes, edges, text, model
    full = getattr(vaft.diagram, spec.key)()
    monkeypatch.setattr(W, "pictogram", lambda *a, **k: vaft.diagram._scene.Scene(()))
    plain = getattr(vaft.diagram, spec.key)()
    assert [i for i in full.scene.items if not i.role.startswith("symbol:")] == [
        i for i in plain.scene.items]  # (legend widths include the icon slot either way)
    assert full.model == plain.model


@pytest.mark.parametrize("spec", SPECS, ids=[s.key for s in SPECS])
def test_the_legend_never_widens_the_figure(spec):
    from vaft.diagram._scene import Polyline

    scene = getattr(vaft.diagram, spec.key)().scene
    boxes = [p[0] for i in scene.items if isinstance(i, Polyline) and i.role.startswith("node:") for p in i.points]
    left, right = min(boxes), max(boxes)
    legend = [i for i in scene.items if i.role.startswith(("legend:", "symbol:legend:"))]
    xs = [p[0] for i in legend if isinstance(i, Polyline) for p in i.points]
    half = 0.5 * max(right - left, 12.0)
    assert 0.5 * (left + right) - half - 1e-9 <= min(xs) and max(xs) <= 0.5 * (left + right) + half + 1e-9
