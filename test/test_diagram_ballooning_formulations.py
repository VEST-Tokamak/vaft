"""The ballooning-formulation hierarchy (#1637): reduction chain, implementations, shared normalisation."""

import pytest

import vaft.diagram
from vaft.diagram._ballooning_formulations import IMPLEMENTATIONS, REDUCTION_STEPS
from vaft.diagram._equations import formula_equation
from vaft.formula.equilibrium import ballooning_alpha_from_volume, shear_from_volume
from vaft.formula.stability import s_alpha_ballooning_stable


@pytest.fixture(scope="module")
def diagram():
    return vaft.diagram.ballooning_formulation_hierarchy()


def test_the_reduction_is_a_chain_from_the_general_equation_to_the_cht_model(diagram):
    edges = set(diagram.model["edges"])
    chain = ["general", *[k for k, _ in REDUCTION_STEPS], "vaft"]
    for a, b in zip(chain, chain[1:]):
        assert (a, b) in edges, (a, b)
    # the full-geometry codes come straight from the general equation, not through any reduction step
    for code in ("dcon", "gpec_jl"):
        assert ("general", code) in edges
        assert not any(b == code and a != "general" for a, b in edges)
    # every formulation reaches the comparison
    for code in IMPLEMENTATIONS:
        assert (code, "compare") in edges


def test_the_two_full_geometry_indices_have_opposite_stable_signs():
    assert ">" in IMPLEMENTATIONS["dcon"][1] and "<" in IMPLEMENTATIONS["gpec_jl"][1]


def test_equations_come_from_the_catalog(diagram):
    for role, fn in (("equation:cht", s_alpha_ballooning_stable), ("equation:shear", shear_from_volume),
                     ("equation:alpha", ballooning_alpha_from_volume)):
        texts = [it.text for it in diagram.scene.role(role)]
        assert any(formula_equation(fn) in t for t in texts), role


def test_deterministic_exported_and_labels_switch_off_the_note():
    fn = vaft.diagram.ballooning_formulation_hierarchy
    assert fn().tikz == fn().tikz
    assert "ballooning_formulation_hierarchy" in vaft.diagram.__all__
    assert fn().scene.role("note") and not fn(labels=False).scene.role("note")
    with pytest.raises(ValueError):
        fn(labels="yes")
