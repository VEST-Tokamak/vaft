"""The ordering contract map (#1627): every cell is a registered assumption."""

import pytest

import vaft.diagram
from vaft.validation.orderings import CONTRACTS, ORDERING_QUANTITIES


def test_cells_are_exactly_the_registered_assumptions():
    d = vaft.diagram.ordering_contract_map()
    expected = {(c.name, a.quantity): a.ordering for c in CONTRACTS.values() for a in c.assumptions}
    assert d.model["cells"] == expected
    assert d.model["rows"] == tuple(CONTRACTS) and d.model["columns"] == tuple(ORDERING_QUANTITIES)
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
