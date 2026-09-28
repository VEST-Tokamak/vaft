"""Cold-plasma wave diagrams (#1113): every boundary is a zero or pole of the Stix parameters."""

import numpy as np
import pytest

import vaft.diagram
from vaft.diagram import _cold_plasma_waves as cw
from vaft.diagram._scene import Label
from vaft.formula.constants import ME, QE
from vaft.formula.waves import perpendicular_refractive_index_squared, stix_parameters


def test_o_mode_cutoff_is_at_the_plasma_frequency():
    chart = vaft.diagram.o_mode_cutoff().model
    assert chart.parameters["cutoff_omega_over_omega_pe"] == pytest.approx(1.0, abs=1e-9)
    x, y = chart.curves["n2_O"].T
    assert np.all(y[x < 1.0] < 0) and np.all(y[x > 1.0] > 0)


@pytest.mark.parametrize("ratio", [0.6, 1.2, 2.0, 3.5])
def test_x_mode_cutoffs_and_resonance_are_the_textbook_frequencies(ratio):
    p = vaft.diagram.x_mode_dispersion(omega_pe_over_omega_ce=ratio).model.parameters
    # in units of |Omega_e|: omega_{R,L} = sqrt(1/4 + ratio^2) +- 1/2, omega_UH = sqrt(1 + ratio^2)
    assert p["omega_R"] == pytest.approx(np.sqrt(0.25 + ratio**2) + 0.5, rel=1e-9)
    assert p["omega_L"] == pytest.approx(np.sqrt(0.25 + ratio**2) - 0.5, rel=1e-9)
    assert p["omega_UH"] == pytest.approx(np.sqrt(1.0 + ratio**2), rel=1e-9)
    assert p["omega_L"] < p["omega_UH"] < p["omega_R"]
    chart = vaft.diagram.x_mode_dispersion(omega_pe_over_omega_ce=ratio).model
    assert chart.x_range[1] > p["omega_R"]  # every layer on the chart, whatever the density
    for name, xy in chart.labels.items():
        assert chart.x_range[0] <= xy[0] <= chart.x_range[1], name


def test_x_mode_curve_is_masked_at_the_pole():
    chart = vaft.diagram.x_mode_dispersion().model
    runs = [c for name, c in chart.curves.items() if name.startswith("n2_X")]
    assert len(runs) >= 2
    omega_uh = chart.parameters["omega_UH"]
    for run in runs:
        assert not (run[0, 0] < omega_uh < run[-1, 0])  # no run crosses the resonance


def test_cma_boundaries_are_the_stix_zeros():
    Y = np.array([0.1, 0.5, 0.9, 1.5])
    b = cw.cma_boundaries(Y)
    np.testing.assert_allclose(b["P"], 1.0, rtol=1e-9)
    np.testing.assert_allclose(b["L"], 1.0 + Y, rtol=1e-9)
    np.testing.assert_allclose(b["R"][:3], 1.0 - Y[:3], rtol=1e-9)
    np.testing.assert_allclose(b["S"][:3], 1.0 - Y[:3] ** 2, rtol=1e-9)
    assert np.isnan(b["R"][3]) and np.isnan(b["S"][3])  # no R or S zero above the cyclotron resonance
    chart = vaft.diagram.cma_diagram().model
    assert {"P=0", "R=0", "L=0", "S=0", "Y=1"} <= set(chart.curves)


def test_profile_layers_are_sign_changes_of_the_formula():
    p = cw.EXAMPLE_PROFILE
    layers = cw.profile_layers(p)
    assert {k: len(v) for k, v in layers.items()} == {"P": 2, "R": 1, "L": 2, "S": 1, "ECR": 1}
    omega = 2 * np.pi * p["frequency"]
    for name in ("P", "R", "L", "S"):
        for r in layers[name]:
            _, _, s, _, _ = cw.profile_quantities(np.array([r - 1e-6, r + 1e-6]), p)
            values = getattr(s, name)
            assert np.sign(values[0]) != np.sign(values[1]), name
            assert abs(getattr(cw.profile_quantities(np.array([r]), p)[2], name)[0]) < 1e-6
    (ecr,) = layers["ECR"]
    assert QE * p["B0"] * p["R0"] / ecr / ME == pytest.approx(omega, rel=1e-9)
    # from the low-field edge the X mode meets its R cutoff before the upper hybrid layer
    assert max(layers["R"]) > max(layers["S"])
    # the O-mode cutoffs bracket the dense core
    assert min(layers["P"]) < p["R0"] < max(layers["P"])


def test_profile_curves_are_the_perpendicular_indices():
    chart = vaft.diagram.profile_propagation().model
    R, n2 = chart.curves["n2_O"].T
    p = cw.EXAMPLE_PROFILE
    B = p["B0"] * p["R0"] / R
    n_e = p["n0"] * np.clip(1 - ((R - p["R0"]) / p["a"]) ** 2, 0, None)
    s = stix_parameters(2 * np.pi * p["frequency"], n_e[None, :], np.array([-QE]), np.array([ME]), B)
    np.testing.assert_allclose(n2, perpendicular_refractive_index_squared(s.R, s.L, s.P)[0])


def test_labels_off_and_determinism():
    for build in (vaft.diagram.o_mode_cutoff, vaft.diagram.x_mode_dispersion, vaft.diagram.cma_diagram,
                  vaft.diagram.profile_propagation):
        assert build().scene == build().scene
        assert not [i for i in build(labels=False).scene.items
                    if isinstance(i, Label) and i.role not in ("axes", "ticks")]
    with pytest.raises(ValueError):
        vaft.diagram.x_mode_dispersion(omega_pe_over_omega_ce=0.0)
