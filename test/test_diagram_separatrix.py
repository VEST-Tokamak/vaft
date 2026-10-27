"""Separatrix q asymptotics diagrams (#2152): traced Solov'ev q against the X-point coefficient, and the figures."""

import re
from pathlib import Path

import numpy as np
import pytest

import vaft
import vaft.diagram
from vaft.diagram import _separatrix as sep
from vaft.diagram._scene import Label
from vaft.formula.geometry import xpoint_resolution_length

ROOT = Path(vaft.__file__).resolve().parents[1]


def _slope(topology, d1=1e-5, d2=1e-7):
    q1, q2 = sep.traced_q(topology, 1.0 - d1), sep.traced_q(topology, 1.0 - d2)
    return (q2 - q1) / np.log(d1 / d2)


def test_registered_and_canonical():
    from vaft.diagram import build
    for name in ("separatrix_q_asymptotics", "xpoint_refinement_hierarchy"):
        assert name in vaft.diagram.__all__
        assert build.CANONICAL[f"{name}.svg"] == (name, {})


@pytest.mark.parametrize("topology, n_x", [("single_null", 1), ("double_null", 2)])
def test_the_traced_q_diverges_with_the_x_point_coefficient(topology, n_x):
    model = sep._equilibrium(topology)
    assert len(model["x_points"]) == n_x
    # every X-point on the separatrix contributes; the traced slope is their sum, not fitted into it
    assert _slope(topology) == pytest.approx(model["A"], rel=2e-3)


def test_the_limited_q_stays_finite():
    near, nearer = sep.traced_q("limited", 1.0 - 1e-4), sep.traced_q("limited", 1.0 - 1e-6)
    assert abs(nearer - near) < 1e-3
    assert abs(_slope("limited")) < 1e-3


def test_the_figure_model_agrees_with_the_formulas():
    d = vaft.diagram.separatrix_q_asymptotics()
    m = d.model
    assert m["fitted_slope"] == pytest.approx(m["A"], rel=5e-3)
    # rational surfaces deep in the logarithmic regime: found and predicted agree in ln(1/delta)
    for entry in m["rational"].values():
        for xn, xp in zip(entry["numeric"], entry["predicted"]):
            if xn > 5.0:
                assert xn == pytest.approx(xp, abs=0.05)
    # the grid limit is where the X-point resolution length equals the grid spacing
    eq = sep._equilibrium("single_null")
    saddle = eq["x_points"][0]["saddle"]
    span = abs(eq["psi_axis"] * eq["psi0"])
    ell = xpoint_resolution_length(m["delta_grid"] * span, min(saddle.kappa_u, saddle.kappa_v))
    assert ell == pytest.approx(m["grid_spacing"], rel=1e-12)
    text = " ".join(i.text for i in d.scene.items if isinstance(i, Label))
    assert f"{m['A']:.3f}" in text and "Hessian" in text


def test_the_marked_psihigh_is_the_packaged_one():
    text = (ROOT / "vaft/data/gpec/equil.in").read_text()
    value = float(re.search(r"^\s*psihigh\s*=\s*([0-9.eE+-]+)", text, re.M).group(1))
    assert sep.DCON_PSIHIGH == value


def test_the_hierarchy_names_real_solver_parameters():
    d = vaft.diagram.xpoint_refinement_hierarchy()
    assert d.model["causes"] == ("field", "q", "rational", "mapping")
    assert d.model["strategies"] == ("chease", "dcon", "coordinates")
    drawn = sorted(i.role[len("edge:"):] for i in d.scene.items if i.role.startswith("edge:"))
    assert drawn == sorted(f"{a}->{b}" for a, b in d.model["edges"])
    assert {b for a, b in d.model["edges"] if a in d.model["strategies"]} == {"validation"}
    chease = (ROOT / "vaft/code/chease.py").read_text()
    for name in ("target_psin", "ns", "nt"):
        assert re.search(rf"^\s+{name}: ", chease, re.M), name
    gpec = (ROOT / "vaft/data/gpec/equil.in").read_text() + (ROOT / "vaft/data/gpec/dcon.in").read_text()
    for name in ("psihigh", "qhigh", "sas_flag", "dmlim", "psiedge"):
        assert name in gpec, name


@pytest.mark.parametrize("name", ["separatrix_q_asymptotics", "xpoint_refinement_hierarchy"])
def test_labels_false_drops_the_note(name):
    build = getattr(vaft.diagram, name)
    assert build().scene.role("note") and not build(labels=False).scene.role("note")
    with pytest.raises(ValueError, match="labels"):
        build(labels="yes")
