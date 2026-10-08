"""Ordering contracts of the reduced models (#1627 phase C): registry, quantities and evaluation."""

import math

import pytest

from vaft.formula import catalog
from vaft.validation.applicability import SCOPES, evaluate_contract, evaluate_population
from vaft.validation.orderings import CONTRACTS, GROUPS, ORDERING_QUANTITIES, OrderingQuantity, contract


def _orderings(name):
    return {a.quantity: a.ordering for a in contract(name).assumptions}


def test_every_assumption_names_a_defined_quantity_with_its_scale_and_scope():
    for c in CONTRACTS.values():
        assert c.references, c.name
        for a in c.assumptions:
            q = ORDERING_QUANTITIES[a.quantity]
            assert (a.scale, a.scope) == (q.scale, q.scope), (c.name, a.quantity)
            assert a.scope in SCOPES and a.threshold == 1.0  # order unity, nothing invented


def test_every_quantity_is_used_and_every_cited_kernel_is_catalogued():
    used = {a.quantity for c in CONTRACTS.values() for a in c.assumptions}
    assert used == set(ORDERING_QUANTITIES)
    for name, q in ORDERING_QUANTITIES.items():
        for kernel in q.kernels:
            assert catalog.describe(kernel).qualname == kernel, (name, kernel)


def test_groups_follow_the_scope_and_are_contiguous():
    groups = [q.group for q in ORDERING_QUANTITIES.values()]
    # drawn in GROUPS order, each group one block
    assert groups == sorted(groups, key=GROUPS.index)
    for q in ORDERING_QUANTITIES.values():
        if q.group == "perturbation":
            assert q.scope == "mode"
        if q.group == "layer":
            assert q.scope == "local"
    with pytest.raises(ValueError, match="group"):
        OrderingQuantity("x", "L", "global", "everywhere", ())


def test_the_ion_skin_depth_is_three_quantities_on_three_scales():
    scales = {k: (ORDERING_QUANTITIES[k].scale, ORDERING_QUANTITIES[k].scope)
              for k in ("ion_skin_depth_over_a", "k_ion_skin_depth", "ion_skin_depth_over_layer")}
    assert scales == {"ion_skin_depth_over_a": ("a", "global"), "k_ion_skin_depth": ("1 / k", "mode"),
                      "ion_skin_depth_over_layer": ("delta_layer", "local")}
    # resistive MHD also needs the layer to be wider than d_i and rho_s: a small global d_i/a is not enough
    resistive = _orderings("resistive_mhd")
    assert resistive["ion_skin_depth_over_a"] == resistive["ion_skin_depth_over_layer"] == "small"
    assert resistive["rho_s_over_layer"] == "small"


def test_slow_evolution_is_the_equilibrium_sequence_not_mhd():
    # ideal MHD describes omega tau_A ~ 1 (kinks, Alfven waves): tau_evol/tau_A >> 1 is not its condition
    for name in ("ideal_single_fluid_mhd", "resistive_mhd", "hall_mhd"):
        assert "tau_evolution_over_tau_alfven" not in _orderings(name), name
    assert _orderings("quasi_static_equilibrium")["tau_evolution_over_tau_alfven"] == "large"


def test_k_perp_rho_i_separates_gyrokinetics_from_mhd_and_drift_kinetics():
    for name in ("ideal_single_fluid_mhd", "resistive_mhd", "drift_kinetic", "flr_small_fluid"):
        assert _orderings(name)["k_perp_rho_i"] == "small", name
    gk = _orderings("gyrokinetic_delta_f")
    assert "k_perp_rho_i" not in gk
    assert gk["rho_i_over_LTi"] == gk["k_par_over_k_perp"] == gk["omega_over_ion_gyrofrequency"] == "small"


def test_hall_mhd_keeps_d_i_and_orders_out_only_electron_inertia():
    hall = _orderings("hall_mhd")
    assert "k_ion_skin_depth" not in hall and "ion_skin_depth_over_layer" not in hall
    assert hall["k_electron_skin_depth"] == hall["electron_skin_depth_over_layer"] == "small"


def test_reduced_mhd_beta_orderings_differ():
    assert _orderings("low_beta_reduced_mhd")["beta_over_inverse_aspect_ratio"] == "small"
    assert _orderings("high_beta_reduced_mhd")["beta"] == "small"
    assert "beta_over_inverse_aspect_ratio" not in _orderings("high_beta_reduced_mhd")


def test_pfirsch_schlueter_and_braginskii_share_the_parallel_knudsen_number():
    # lambda/(qR) = 1/nu_hat: the collisional closure is the Pfirsch-Schlueter ordering
    ps, brag = _orderings("pfirsch_schlueter_neoclassical"), _orderings("braginskii_two_fluid")
    for species in ("electron", "ion"):
        key = f"{species}_parallel_knudsen_number"
        assert ps[key] == brag[key] == "small"
        assert ORDERING_QUANTITIES[key].scale == "q R"
    assert _orderings("banana_regime_neoclassical") == {"electron_collisionality": "small",
                                                        "ion_collisionality": "small"}


def test_evaluation_names_the_limiting_ordering_and_reports_missing_ones():
    ideal = contract("ideal_single_fluid_mhd")
    state = {"debye_length_over_L": 1e-5, "omega_over_ion_gyrofrequency": 1e-3, "lundquist_number": 5.7e4,
             "ion_skin_depth_over_a": 0.29, "k_ion_skin_depth": 0.1, "k_perp_rho_i": 0.05,
             "rho_i_over_LTi": 0.02, "pressure_anisotropy": 0.01}
    result = evaluate_contract(ideal, state)
    assert result.status == "SUPPORTED" and result.limiting == "ion_skin_depth_over_a"
    margins = {r.quantity: r.margin for r in result.assumptions}
    assert margins["lundquist_number"] == pytest.approx(math.log10(5.7e4))
    assert margins["ion_skin_depth_over_a"] == pytest.approx(-math.log10(0.29))
    # a Hall-scale perturbation fails on k d_i alone, though d_i/a is unchanged
    assert evaluate_contract(ideal, {**state, "k_ion_skin_depth": 1.5}).status == "OUTSIDE"
    # a missing quantity is unassessed, never a violation
    partial = {k: v for k, v in state.items() if k != "rho_i_over_LTi"}
    assert evaluate_contract(ideal, partial).status == "UNASSESSED"


def test_contracts_evaluate_over_a_population():
    pd = pytest.importorskip("pandas")
    table = pd.DataFrame([{"electron_parallel_knudsen_number": 0.01, "ion_parallel_knudsen_number": 0.05},
                          {"electron_parallel_knudsen_number": 3.0, "ion_parallel_knudsen_number": 0.05}])
    out = evaluate_population(contract("pfirsch_schlueter_neoclassical"), table)
    assert list(out["status"]) == ["SUPPORTED", "OUTSIDE"]
    assert list(out["limiting"]) == ["ion_parallel_knudsen_number", "electron_parallel_knudsen_number"]


def test_unknown_contract_lists_the_known_ones():
    with pytest.raises(KeyError, match="ideal_single_fluid_mhd"):
        contract("full_kinetic")


def test_braginskii_is_slow_against_collisions():
    assert _orderings("braginskii_two_fluid")["omega_tau_i"] == "small"
    assert ORDERING_QUANTITIES["omega_tau_i"].scope == "mode"


def test_signed_ratios_enter_as_magnitudes_and_zero_is_unassessed():
    from vaft.formula.ordering import pressure_anisotropy

    ideal = contract("ideal_single_fluid_mhd")
    delta = pressure_anisotropy(1.0, 2.0)  # parallel heating: Delta = -0.75
    assert delta < 0
    margin = {r.quantity: r.margin for r in evaluate_contract(ideal, {"pressure_anisotropy": abs(delta)}).assumptions}
    assert margin["pressure_anisotropy"] == pytest.approx(-math.log10(0.75))
    # an exactly isotropic state has no logarithmic margin: applicability reports it, it does not pass it
    zero = {r.quantity: r for r in evaluate_contract(ideal, {"pressure_anisotropy": 0.0}).assumptions}
    assert math.isnan(zero["pressure_anisotropy"].margin)
