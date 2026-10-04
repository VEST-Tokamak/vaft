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
        "gpec_plasma_response", "flare_field_line_topology", "neo_neoclassical", "tglf_cgyro_local_transport"]


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
    assert "render_workflow(WORKFLOWS[" in inspect.getsource(fn)


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
