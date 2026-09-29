"""SOL blob diagrams (#1211): signs self-consistent, curves and boundaries from vaft.formula.blob."""

import numpy as np
import pytest

import vaft.diagram
from vaft.diagram._scene import Arrow, Label
from vaft.formula.blob import blob_regime_velocities, interpolated_blob_velocity


def test_polarization_signs_are_self_consistent():
    diagram = vaft.diagram.blob_polarization()
    arrows = {a.role: np.subtract(a.end, a.start) for a in diagram.scene.items if isinstance(a, Arrow)}
    # B into the page (-z), x outward: grad B points -x
    B = np.array([0.0, 0.0, -1.0])
    grad_b = np.append(arrows["grad_b"], 0.0)
    ion = np.cross(B, grad_b)  # q > 0: v_gradB ~ B x grad B
    assert np.sign(ion[1]) == np.sign(arrows["ion_drift"][1]) > 0
    assert np.sign(arrows["electron_drift"][1]) < 0
    # + above, - below: E points down; E x B points outward, down grad B
    E = np.append(arrows["electric_field"], 0.0)
    assert E[1] < 0
    v_e = np.cross(E, B)
    assert np.sign(v_e[0]) == np.sign(arrows["exb_velocity"][0]) > 0
    assert np.sign(arrows["exb_velocity"][0]) == -np.sign(arrows["grad_b"][0])


def test_velocity_curve_is_the_formula():
    chart = vaft.diagram.blob_velocity_scaling().model
    x, y = chart.curves["interpolated"].T
    np.testing.assert_allclose(10**y, interpolated_blob_velocity(10**x))


@pytest.mark.parametrize("eps", [0.1, 0.3])
def test_regime_boundaries_are_where_the_scalings_meet(eps):
    chart = vaft.diagram.blob_regimes(epsilon_x=eps).model
    lt, ll = chart.curves["rb_rx"].T
    v = blob_regime_velocities((10**lt) ** 0.4, 10**ll, eps)
    np.testing.assert_allclose(v.resistive_ballooning, v.resistive_x_point, rtol=1e-10)
    lt, ll = chart.curves["rx_ci"].T
    v = blob_regime_velocities((10**lt) ** 0.4, 10**ll, eps)
    np.testing.assert_allclose(v.resistive_x_point, v.connected_ideal_interchange, rtol=1e-10)
    assert chart.curves["ci_cs"][0, 0] == pytest.approx(np.log10(1 / eps))


def test_labels_off():
    for build in (vaft.diagram.blob_polarization, vaft.diagram.blob_velocity_scaling, vaft.diagram.blob_regimes):
        assert not [i for i in build(labels=False).scene.items if isinstance(i, Label)
                    and i.role not in ("axes", "ticks", "charge_positive", "charge_negative")]  # charges are content
