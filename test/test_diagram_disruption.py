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


def test_the_runaway_gain_stays_below_the_avalanche_upper_bound():
    m = dz.reference_model()
    p = m["params"]
    efolds = avalanche_efolds_from_current_drop(p["I_0"] - m["I_p"][-1], m["L_p"], p["R0"], p["Z_eff"],
                                                p["ln_Lambda_rel"])
    seed = np.trapezoid(m["dreicer"], m["t"]) * m["area"] * 1.602176634e-19 * 299792458.0
    assert m["I_RE"][-1] <= seed * np.exp(efolds) * 1.01


def test_the_generation_chart_orders_the_regimes():
    m = vaft.diagram.runaway_generation().model
    x = np.log10(m["E_D"] / m["E_c"])
    assert x > 3
    below = m["x"] < 0
    assert np.all(m["avalanche"][below] == 0.0)
    # near E_D the Dreicer rate per electron overtakes the avalanche rate per runaway, far below it does not
    assert m["dreicer"][-1] > m["avalanche"][-1] or m["dreicer"][-1] > 1e-3
    mid = np.argmin(np.abs(m["x"] - 1.0))
    assert m["dreicer"][mid] < 1e-12 * m["avalanche"][mid]


@pytest.mark.parametrize("name", ["disruption_timeline", "disruption_causal_chain", "runaway_generation",
                                  "disruption_energy_pathways"])
def test_every_disruption_diagram_is_deterministic_and_exported(name):
    fn = getattr(vaft.diagram, name)
    assert fn().tikz == fn().tikz
    assert name in vaft.diagram.__all__
    assert fn().scene.role("note") and not fn(labels=False).scene.role("note")
    assert sum(isinstance(i, Label) for i in fn(labels=False).scene.items) <= sum(
        isinstance(i, Label) for i in fn().scene.items)
