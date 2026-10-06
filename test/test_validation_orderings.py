"""Ordering contracts of the reduced models (#1627 phase C): registry, quantities and evaluation."""

import math

import pytest

from vaft.formula import catalog
from vaft.validation.applicability import SCOPES, evaluate_contract, evaluate_population
from vaft.validation.orderings import CONTRACTS, ORDERING_QUANTITIES, contract


def test_every_assumption_names_a_defined_quantity_with_its_scale_and_scope():
    for c in CONTRACTS.values():
        assert c.references, c.name
        for a in c.assumptions:
            q = ORDERING_QUANTITIES[a.quantity]
            assert (a.scale, a.scope) == (q.scale, q.scope), (c.name, a.quantity)
            assert a.scope in SCOPES and a.threshold == 1.0  # order unity, nothing invented


def test_every_kernel_a_quantity_cites_is_catalogued():
    for name, q in ORDERING_QUANTITIES.items():
        for kernel in q.kernels:
            assert catalog.describe(kernel).qualname == kernel, (name, kernel)


def test_ideal_mhd_is_four_separate_orderings_not_one():
    ideal = contract("ideal_single_fluid_mhd")
    assert {a.quantity: a.ordering for a in ideal.assumptions} == {
        "lundquist_number": "large", "ion_skin_depth_over_L": "small",
        "rho_i_over_LTi": "small", "tau_evolution_over_tau_alfven": "large"}
    # resistive MHD keeps resistivity: it makes no claim about S
    assert "lundquist_number" not in {a.quantity for a in contract("resistive_mhd").assumptions}


def test_evaluation_names_the_limiting_ordering_and_reports_missing_ones():
    ideal = contract("ideal_single_fluid_mhd")
    state = {"lundquist_number": 5.7e4, "ion_skin_depth_over_L": 0.29, "rho_i_over_LTi": 0.02,
             "tau_evolution_over_tau_alfven": 1.4e4}
    result = evaluate_contract(ideal, state)
    assert result.status == "SUPPORTED" and result.limiting == "ion_skin_depth_over_L"
    margins = {r.quantity: r.margin for r in result.assumptions}
    assert margins["lundquist_number"] == pytest.approx(math.log10(5.7e4))
    assert margins["ion_skin_depth_over_L"] == pytest.approx(-math.log10(0.29))
    # a Hall-scale state fails on d_i/a alone
    assert evaluate_contract(ideal, {**state, "ion_skin_depth_over_L": 1.5}).status == "OUTSIDE"
    # a missing quantity is unassessed, never a violation
    partial = {k: v for k, v in state.items() if k != "rho_i_over_LTi"}
    assert evaluate_contract(ideal, partial).status == "UNASSESSED"


def test_contracts_evaluate_over_a_population():
    pd = pytest.importorskip("pandas")
    table = pd.DataFrame([{"electron_knudsen_number": 0.01, "ion_knudsen_number": 0.05},
                          {"electron_knudsen_number": 3.0, "ion_knudsen_number": 0.05}])
    out = evaluate_population(contract("local_collisional_fluid"), table)
    assert list(out["status"]) == ["SUPPORTED", "OUTSIDE"]
    assert list(out["limiting"]) == ["ion_knudsen_number", "electron_knudsen_number"]


def test_unknown_contract_lists_the_known_ones():
    with pytest.raises(KeyError, match="ideal_single_fluid_mhd"):
        contract("hall_mhd")
