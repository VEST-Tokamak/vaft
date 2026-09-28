"""Disruption diagrams (#1041): the timeline is the reference model, the model is the formulas."""

import numpy as np
import pytest

import vaft.diagram
from vaft.diagram import _disruption as dz
from vaft.diagram._scene import Label
from vaft.formula.disruption import avalanche_efolds_from_current_drop


def test_the_reference_model_goes_through_the_disruption_sequence():
    m = dz.reference_model()
    p = m["params"]
    t = m["t"]
    # the thermal quench precedes the current quench, which precedes the runaway plateau
    edges = dz._phase_edges(m)
    assert 0.0 == edges["thermal_quench"] < edges["current_quench"] < edges["re_plateau"]
    # cooling raises the resistivity by (T_0/T_f)^(3/2)
    assert m["eta"][-1] / m["eta"][0] == pytest.approx((p["T_0"] / p["T_final"]) ** 1.5, rel=1e-9)
    # the induced field goes far above E_c during the current quench, and is below it before
    assert m["E"][0] < m["E_c"] < np.max(m["E"]) and np.max(m["E"]) > 100 * m["E_c"]
    # runaways appear only after the field exceeds E_c, and carry part of the current at the end
    first_re = t[np.argmax(m["I_RE"] > 1.0)]
    first_supercritical = t[np.argmax(m["E"] > m["E_c"])]
    assert first_re >= first_supercritical
    assert 0.05 * p["I_0"] < m["I_RE"][-1] < p["I_0"]
    # the total current never grows
    assert np.all(np.diff(m["I_p"]) <= 1e-6 * p["I_0"])


def test_the_model_reports_its_seed_and_avalanche_honestly():
    m = dz.reference_model()
    p = m["params"]
    # at 1 MA the avalanche gives a few e-folds, bounded by the Rosenbluth-Putvinski estimate for this drop
    efolds = avalanche_efolds_from_current_drop(p["I_0"] - m["I_p"][-1], m["L_p"], p["R0"], p["Z_eff"],
                                                p["ln_Lambda_rel"])
    assert 1.0 < m["avalanche_gain"] < np.exp(efolds)
    assert m["I_RE"][-1] == pytest.approx(m["seed_current"] * m["avalanche_gain"])
    assert m["seed_current"] < 0.2 * m["I_RE"][-1]  # the avalanche does most of the work, even at 1 MA
    # the cached model cannot be corrupted by a caller
    with pytest.raises(ValueError):
        m["I_p"][0] = 0.0


def test_the_generation_chart_keeps_dreicer_inside_its_validity():
    m = vaft.diagram.runaway_generation().model
    assert np.log10(m["E_D"] / m["E_c"]) > 3
    assert np.all(m["avalanche"][m["x"] < 0] == 0.0)
    assert m["dreicer_drawn"].max() <= np.log10(0.1 * m["E_D"] / m["E_c"]) + 1e-9
    # the avalanche rate grows linearly in E - E_c
    above = m["x"] > 0.5
    ratio = m["avalanche"][above] / (10 ** m["x"][above] - 1)
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-9)


@pytest.mark.parametrize("name", ["disruption_timeline", "disruption_causal_chain", "runaway_generation",
                                  "disruption_energy_pathways"])
def test_every_disruption_diagram_is_deterministic_and_exported(name):
    fn = getattr(vaft.diagram, name)
    assert fn().tikz == fn().tikz
    assert name in vaft.diagram.__all__
    assert fn().scene.role("note") and not fn(labels=False).scene.role("note")
    assert sum(isinstance(i, Label) for i in fn(labels=False).scene.items) <= sum(
        isinstance(i, Label) for i in fn().scene.items)
