"""SOL blob diagrams (#1211): signs from the drawn charges, curves and boundaries against the papers."""

import math

import numpy as np
import pytest

import vaft.diagram
from vaft.diagram._scene import Arrow, Label, Marker
from vaft.formula.sol import blob_crossover_size, interpolated_blob_velocity


def _arrows(diagram):
    return {a.role: np.subtract(a.end, a.start) for a in diagram.scene.items if isinstance(a, Arrow)}


def _at(diagram, role):
    return np.asarray([i.at for i in diagram.scene.items if isinstance(i, Label) and i.role == role][0])


@pytest.mark.parametrize("perturbation, outward", [("blob", True), ("hole", False)])
def test_the_drawn_dipole_drives_the_drawn_motion(perturbation, outward):
    diagram = vaft.diagram.blob_polarization(perturbation=perturbation)
    arrows = _arrows(diagram)
    B = np.array([0.0, 0.0, -1.0])  # into the page
    # ions drift along B x grad B (q > 0)
    ion = np.cross(B, np.append(arrows["grad_b"], 0.0))
    assert np.sign(ion[1]) == np.sign(arrows["ion_drift"][1])
    # E points from the drawn + to the drawn -; E x B is the drawn motion
    E = np.append(_at(diagram, "charge_negative") - _at(diagram, "charge_positive"), 0.0)
    assert np.dot(E[:2], arrows["electric_field"]) > 0
    v = np.cross(E, B)
    assert np.sign(v[0]) == np.sign(arrows["exb_velocity"][0])
    # a blob moves down grad B (outward), a hole up it (inward)
    assert (np.sign(arrows["exb_velocity"][0]) == -np.sign(arrows["grad_b"][0])) == outward
    # on a blob the positive charge sits where the ions drift to
    toward_plus = _at(diagram, "charge_positive")[1] - _at(diagram, "charge_negative")[1]
    assert (np.sign(toward_plus) == np.sign(arrows["ion_drift"][1])) == outward


@pytest.mark.parametrize("regime", ["sheath", "inertial"])
def test_the_current_closes_where_the_regime_says(regime):
    diagram = vaft.diagram.blob_current_closure(regime=regime)
    roles = {getattr(i, "role", "") for i in diagram.scene.items}
    if regime == "sheath":
        assert {"parallel_current", "sheath_current"} <= roles and "polarization_current" not in roles
    else:
        assert {"polarization_current", "disconnection"} <= roles and "sheath_current" not in roles
    # current flows from the + side toward the - side
    plus_y = _at(diagram, "charge_positive")[1]
    minus_y = _at(diagram, "charge_negative")[1]
    down = [a for r, a in _arrows(diagram).items() if r in ("sheath_current", "polarization_current")]
    assert all(np.sign(a[1]) == np.sign(minus_y - plus_y) for a in down)


def test_velocity_curve_and_crossing():
    diagram = vaft.diagram.blob_velocity_scaling()
    chart = diagram.model
    x, y = chart.curves["interpolated"].T
    np.testing.assert_allclose(10**y, interpolated_blob_velocity(10**x))
    # the crossing marker sits where the two drawn limits meet: delta_hat = 1 at delta n/n = 1
    assert chart.points["crossover"][0] == pytest.approx(math.log10(blob_crossover_size()))
    assert [m for m in diagram.scene.items if isinstance(m, Marker) and m.role == "crossover"]
    # and the interpolation peaks below it, at 0.574 of the crossing
    assert 10 ** x[np.argmax(y)] == pytest.approx(0.574, abs=0.02)


@pytest.mark.parametrize("eps", [0.1, 0.3, 0.01])
def test_regime_boundaries_are_the_printed_lines(eps):
    # D'Ippolito, Myra and Zweben (2011) Fig. 23: Lambda = Theta, Lambda = eps Theta, Lambda = 1, Theta = 1/eps
    chart = vaft.diagram.blob_regimes(epsilon_x=eps).model
    lt, ll = chart.curves["rb_rx"].T
    np.testing.assert_allclose(ll, lt, atol=1e-12)
    lt, ll = chart.curves["rx_ci"].T
    np.testing.assert_allclose(ll, lt + math.log10(eps), atol=1e-12)
    assert lt.max() <= math.log10(1 / eps) + 1e-12
    assert np.allclose(chart.curves["rx_cs"][:, 1], 0.0) and chart.curves["rx_cs"][0, 0] == pytest.approx(
        math.log10(1 / eps))
    assert chart.curves["ci_cs"][0, 0] == pytest.approx(math.log10(1 / eps))
    # the region labels lie in their regions
    le = math.log10(1 / eps)
    x, y = chart.labels["RB"]
    assert y > x
    x, y = chart.labels["Cs"]
    assert x > le and y < 0
    x, y = chart.labels["Ci"]
    assert x < le and y < x + math.log10(eps)


def test_labels_off_and_inputs():
    for build in (vaft.diagram.blob_polarization, vaft.diagram.blob_velocity_scaling, vaft.diagram.blob_regimes,
                  vaft.diagram.blob_current_closure):
        assert not [i for i in build(labels=False).scene.items if isinstance(i, Label)
                    and i.role not in ("axes", "ticks", "charge_positive", "charge_negative")]  # charges are content
    with pytest.raises(ValueError):
        vaft.diagram.blob_polarization(perturbation="spot")
    with pytest.raises(ValueError):
        vaft.diagram.blob_current_closure(regime="x-point")
