"""The ordering contract map: every cell is a registered assumption."""

import pytest

import vaft.diagram
from vaft.validation.orderings import CONTRACTS, GROUPS, ORDERING_QUANTITIES


def test_cells_are_exactly_the_registered_assumptions():
    d = vaft.diagram.ordering_contract_map()
    expected = {(c.name, a.quantity): a.ordering for c in CONTRACTS.values() for a in c.assumptions}
    assert d.model["cells"] == expected
    assert d.model["rows"] == tuple(ORDERING_QUANTITIES) and sorted(d.model["columns"]) == sorted(CONTRACTS)
    for (name, quantity), ordering in expected.items():
        texts = [it.text for it in d.scene.role(f"cell:{name}:{quantity}") if hasattr(it, "text")]
        assert texts == ["$\\gg 1$" if ordering == "large" else "$\\ll 1$"]


def test_deterministic_exported_and_labels():
    fn = vaft.diagram.ordering_contract_map
    assert fn().tikz == fn().tikz
    assert "ordering_contract_map" in vaft.diagram.__all__
    assert fn().scene.role("note") and not fn(labels=False).scene.role("note")
    with pytest.raises(ValueError):
        fn(labels="yes")


def test_rows_are_grouped_by_scale_and_empty_cells_are_unordered():
    d = vaft.diagram.ordering_contract_map()
    assert [r.split(":", 1)[1] for r in (it.role for it in d.scene.items) if r.startswith("group:")][::2] == list(GROUPS)
    # gyrokinetics leaves k_perp rho_i unordered: no cell, while MHD and drift kinetics have one
    assert not d.scene.role("cell:gyrokinetic_delta_f:k_perp_rho_i")
    assert d.scene.role("cell:ideal_single_fluid_mhd:k_perp_rho_i") and d.scene.role("cell:drift_kinetic:k_perp_rho_i")
    # slow evolution belongs to the equilibrium sequence, not to MHD
    assert not d.scene.role("cell:ideal_single_fluid_mhd:tau_evolution_over_tau_alfven")
    assert d.scene.role("cell:quasi_static_equilibrium:tau_evolution_over_tau_alfven")


def test_the_adapter_map_is_the_model_contracts_registry():
    from vaft.validation.model_contracts import ADAPTER_MODES

    d = vaft.diagram.adapter_contract_map()
    expected = {(m.key, u.contract): u.role for m in ADAPTER_MODES for u in m.contracts()[0]}
    assert d.model["uses"] == expected
    # only the consumed contracts are columns, and every unclassified mode is named in the note
    assert set(d.model["columns"]) == {c for (_, c) in expected}
    assert set(d.model["uncovered"]) == {m.key for m in ADAPTER_MODES if not m.contracts()[0]}
    assert d.scene.role("use:gpec:rdcon_rmatch:resistive_mhd")[-1].text == "layer solution"
    # inherited is drawn apart from governing, and the note says an empty cell is not independence
    assert d.scene.role("use:gacode:tglf:gyrokinetic_delta_f")[0].style != d.scene.role(
        "use:gacode:cgyro_linear:gyrokinetic_delta_f")[0].style
    assert "not physical independence" in d.scene.role("note")[0].text
    assert "adapter_contract_map" in vaft.diagram.__all__
    assert d.tikz == vaft.diagram.adapter_contract_map().tikz
    assert d.scene.role("note") and not vaft.diagram.adapter_contract_map(labels=False).scene.role("note")
