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


# --- #1113 section A and E: omega-k, n^2-X, n^2-Y, oblique -----------------------------------------


def test_omega_k_branches_start_on_their_cutoffs():
    m = vaft.diagram.wave_dispersion_omega_k(omega_pe_over_omega_ce=1.2).model
    r = 1.2
    assert m.parameters["omega_P"] == pytest.approx(r, rel=1e-6)
    assert m.parameters["omega_L"] == pytest.approx((-1 + np.sqrt(1 + 4 * r**2)) / 2, rel=1e-6)
    assert m.parameters["omega_R"] == pytest.approx((1 + np.sqrt(1 + 4 * r**2)) / 2, rel=1e-6)
    assert m.parameters["omega_S"] == pytest.approx(np.sqrt(1 + r**2), rel=1e-6)
    x_branches = [c for n, c in m.curves.items() if n.startswith("X ")]
    lower = min(x_branches, key=lambda c: c[:, 1].min())
    k, w = lower.T
    # the lower X branch starts at the L cutoff and runs to the upper-hybrid resonance from below, turning slow
    assert w.min() == pytest.approx(m.parameters["omega_L"], abs=0.01)
    assert w.max() < m.parameters["omega_S"] and w.max() == pytest.approx(m.parameters["omega_S"], abs=0.05)
    assert np.any(w < k)  # it crosses the light line
    upper = max(x_branches, key=lambda c: c[:, 1].min())
    assert upper[:, 1].min() == pytest.approx(m.parameters["omega_R"], abs=0.01)
    assert np.all(upper[:, 1] >= upper[:, 0] - 1e-9)  # the R-cutoff branch stays fast


def test_refractive_index_vs_X_layers_are_the_closed_forms():
    Y = 0.5
    p = vaft.diagram.refractive_index_vs_X(Y=Y).model.parameters
    assert p["X_P"] == pytest.approx(1.0, abs=1e-9)
    assert p["X_R"] == pytest.approx(1 - Y, abs=1e-9)
    assert p["X_L"] == pytest.approx(1 + Y, abs=1e-9)
    assert p["X_S"] == pytest.approx(1 - Y**2, abs=1e-9)


def test_refractive_index_vs_Y_o_branch_is_flat_and_layers_located():
    X = 0.5
    m = vaft.diagram.refractive_index_vs_Y(X=X).model
    o = np.vstack([c for n, c in m.curves.items() if n.startswith("O ")])
    np.testing.assert_allclose(o[:, 1], 1 - X, atol=1e-12)
    assert m.parameters["Y_R"] == pytest.approx(1 - X, abs=1e-9)
    assert m.parameters["Y_S"] == pytest.approx(np.sqrt(1 - X), abs=1e-9)


@pytest.mark.parametrize("deg", [30.0, 60.0])
def test_oblique_resonance_is_the_cone_A_equals_zero(deg):
    th = np.radians(deg)
    Y = 0.5
    p = vaft.diagram.refractive_index_vs_X(Y=Y, theta=th).model.parameters
    # A = S sin^2 + P cos^2 with S = 1 - X/(1 - Y^2), P = 1 - X
    X_A = 1.0 / (np.sin(th) ** 2 / (1 - Y**2) + np.cos(th) ** 2)
    assert p["X_A"] == pytest.approx(X_A, rel=1e-6)
    assert "X_S" not in p


def test_oblique_branches_are_unnamed_and_one_colour():
    d = vaft.diagram.wave_dispersion_omega_k(theta=np.radians(60))
    names = {n.split(" ")[0] for n in d.model.curves if n.split(" ")[0] in ("O", "X", "+", "-")}
    assert names == {"+", "-"}
    assert {it.style for it in d.scene.items if getattr(it, "role", "").split(" ")[0] in ("+", "-")} == {"boundary"}
    assert d.model.parameters["omega_A"] < 1.0 < d.model.parameters["omega_A2"]  # below and above |Omega_e|


def test_profile_theta_default_and_perpendicular_are_the_committed_figure():
    base = vaft.diagram.profile_propagation()
    assert vaft.diagram.profile_propagation(theta=np.pi / 2).tikz == base.tikz
    oblique = vaft.diagram.profile_propagation(theta=np.radians(60)).model
    assert "any_propagating_fraction" in oblique.parameters and "O_propagating_fraction" not in oblique.parameters
    assert any(n.startswith("A ") for n in oblique.curves) and not any(n.startswith("S ") for n in oblique.curves)


@pytest.mark.parametrize("build", ["wave_dispersion_omega_k", "refractive_index_vs_X", "refractive_index_vs_Y",
                                   "profile_propagation"])
@pytest.mark.parametrize("bad", [-0.1, 0.0, 0.05, 2.0, "1", True])
def test_theta_is_validated(build, bad):
    with pytest.raises(ValueError):
        getattr(vaft.diagram, build)(theta=bad)


def test_new_views_are_deterministic_with_labels_off():
    for name in ("wave_dispersion_omega_k", "refractive_index_vs_X", "refractive_index_vs_Y"):
        build = getattr(vaft.diagram, name)
        assert build().scene == build().scene
        assert not [i for i in build(labels=False).scene.items
                    if isinstance(i, Label) and i.role not in ("axes", "ticks")]
    with pytest.raises(ValueError):
        vaft.diagram.refractive_index_vs_X(Y=1.5)
    with pytest.raises(ValueError):
        vaft.diagram.refractive_index_vs_Y(X=0.0)


@pytest.mark.parametrize("deg", [5.0, 30.0, 60.0])
def test_oblique_profile_curves_never_chord_across_a_gap(deg):
    m = vaft.diagram.profile_propagation(theta=np.radians(deg)).model
    R = np.linspace(m.parameters["R0"] - m.parameters["a"], m.parameters["R0"] + m.parameters["a"], 1201)
    step = R[1] - R[0]
    for name, curve in m.curves.items():
        if name.startswith(("n2_O", "n2_X")):
            assert np.all(np.diff(curve[:, 0]) < 1.5 * step), name


@pytest.mark.parametrize("deg", [5.0, 20.0, 60.0])
def test_both_oblique_resonances_are_found_down_to_the_smallest_angle(deg):
    p = vaft.diagram.wave_dispersion_omega_k(theta=np.radians(deg)).model.parameters
    assert p["omega_A"] < 1.0 < p["omega_A2"]
    q = vaft.diagram.refractive_index_vs_Y(X=0.5, theta=np.radians(deg)).model.parameters
    assert "Y_A" in q


def test_omega_k_labels_only_layers_on_the_chart_and_numbers_are_validated():
    d = vaft.diagram.wave_dispersion_omega_k(omega_pe_over_omega_ce=3.0)
    for item in d.scene.items:
        if getattr(item, "role", "").startswith("layer") and hasattr(item, "text"):
            name = item.role.split(" ")[1]
            assert d.model.parameters[f"omega_{name}"] < 4.0
    for bad in (True, "1.2", 0.0, 3.5):
        with pytest.raises(ValueError):
            vaft.diagram.wave_dispersion_omega_k(omega_pe_over_omega_ce=bad)
    with pytest.raises(ValueError):
        vaft.diagram.refractive_index_vs_X(Y="0.5")
