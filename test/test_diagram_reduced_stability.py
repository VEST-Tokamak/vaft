"""Reduced stability diagnostic diagrams (#1635): the taxonomy and the Suydam / Mercier chart."""

import numpy as np
import pytest

import vaft.diagram
from vaft.diagram._equations import formula_equation
from vaft.diagram._reduced_stability import CELLS, PROBLEMS, STATUSES
from vaft.formula.stability import mercier_criterion_circular, suydam_criterion


def test_every_cell_sits_in_its_row_and_column():
    d = vaft.diagram.stability_diagnostic_taxonomy()
    centers = d.model["centers"]
    assert set(centers) == set(CELLS)
    rows = {p: i for i, (p, _) in enumerate(PROBLEMS)}
    cols = {s: j for j, (s, _) in enumerate(STATUSES)}
    xs = {cols[s]: x for (_, s), (x, _) in centers.items()}
    ys = {rows[p]: y for (p, _), (_, y) in centers.items()}
    for (problem, status), (x, y) in centers.items():
        assert x == pytest.approx(xs[cols[status]]) and y == pytest.approx(ys[rows[problem]])
    # columns run left to right and rows top to bottom in the declared order
    assert all(np.diff([xs[j] for j in sorted(xs)]) > 0)
    assert all(np.diff([ys[i] for i in sorted(ys)]) < 0)


def test_taxonomy_keeps_heuristics_and_solver_outputs_apart_from_criteria():
    cells = vaft.diagram.stability_diagnostic_taxonomy().model["cells"]
    # the deprecated heuristics are flagged as such and never shown as reduced models
    for key, text in cells.items():
        if "deprecated" in text:
            assert key[1] == "heuristic", key
    # the 0.6 s line is not the canonical s-alpha model
    assert "0.6" in cells[("pressure", "heuristic")]
    assert "0.6" not in cells[("pressure", "reduced")]
    # Mercier D_I from DCON is read, not computed; the equilibrium-native one is listed as absent
    assert "D_I" in cells[("pressure", "solver")]
    assert any("Mercier" in item for item in vaft.diagram.stability_diagnostic_taxonomy().model["absent"])


def test_interchange_chart_draws_the_formulas_and_their_toroidal_difference():
    d = vaft.diagram.interchange_criteria()
    chart = d.model
    x, s = chart.curves["suydam"].T
    _, m = chart.curves["mercier"].T
    # Mercier = Suydam - p' q^2 > Suydam where the pressure falls: never less stable
    assert np.all(m >= s - 1e-12)
    # Suydam's violated region extends past q = 1; Mercier's stays inside it
    p = chart.parameters
    assert p["q0"] < 1.0
    assert p["suydam_violated_to"] > p["r1"] > p["mercier_violated_to"] > 0.0
    # the equations shown are the catalog's
    texts = [it.text for it in d.scene.role("equations")]
    assert any(formula_equation(suydam_criterion) in t for t in texts)
    assert any(formula_equation(mercier_criterion_circular) in t for t in texts)


@pytest.mark.parametrize("name", ["stability_diagnostic_taxonomy", "interchange_criteria"])
def test_deterministic_exported_and_labels_switch_off_the_note(name):
    fn = getattr(vaft.diagram, name)
    assert fn().tikz == fn().tikz
    assert name in vaft.diagram.__all__
    assert fn().scene.role("note") and not fn(labels=False).scene.role("note")
    with pytest.raises(ValueError):
        fn(labels="yes")
