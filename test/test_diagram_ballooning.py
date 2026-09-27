"""Clebsch labels and ballooning representation (#1075): the drawn physics is the formulas'."""

import math

import numpy as np
import pytest

import vaft.diagram
from vaft.formula.stability import (
    field_line_label,
    helical_phase,
    s_alpha_ballooning_solution,
    s_alpha_ballooning_stable,
    s_alpha_curvature_drive,
)


# --- formulas ------------------------------------------------------------------------------------


def test_the_label_is_constant_along_a_field_line_and_is_the_helical_phase():
    q = 1.5
    theta = np.linspace(0, 4 * np.pi, 50)
    phi = 0.7 + q * theta
    np.testing.assert_allclose(field_line_label(phi, theta, q), 0.7)
    m, n = 3, 2
    th, ph = 0.4, 1.3
    assert field_line_label(ph, th, m / n) == pytest.approx(-float(helical_phase(th, ph, m, n)) / n)


def test_the_curvature_drive_is_the_equations_coefficient():
    s, a = 0.8, 0.5
    theta = np.linspace(-6, 6, 13)
    lam = s * theta - a * np.sin(theta)
    np.testing.assert_allclose(s_alpha_curvature_drive(theta, s, a), np.cos(theta) + lam * np.sin(theta))
    assert s_alpha_curvature_drive(0.0, s, a) == pytest.approx(1.0)  # outboard: bad
    assert s_alpha_curvature_drive(np.pi, s, a) == pytest.approx(-1.0)  # inboard: good
    # theta0 moves the zero of the local shear to theta0
    th0 = 0.6
    lam0 = s * (theta - th0) - a * (np.sin(theta) - np.sin(th0))
    np.testing.assert_allclose(s_alpha_curvature_drive(theta, s, a, theta0=th0),
                               np.cos(theta) + lam0 * np.sin(theta))


@pytest.mark.parametrize("s, alpha", [(1.0, 0.3), (1.0, 0.9), (0.5, 0.2), (0.5, 0.6), (1.5, 4.5)])
def test_the_solution_crosses_zero_exactly_when_the_verdict_is_unstable(s, alpha):
    theta, F = s_alpha_ballooning_solution(s, alpha, theta_max=40 * np.pi)
    assert F[0] == 1.0 and np.all(np.diff(theta) > 0)
    assert bool(np.all(F > 0)) == s_alpha_ballooning_stable(s, alpha)


# --- diagrams ------------------------------------------------------------------------------------


def test_the_clebsch_field_is_along_the_line_and_grad_alpha_across_it():
    m = vaft.diagram.clebsch_field_line_label().model
    t = m["tangent"] / np.linalg.norm(m["tangent"])
    b = m["B_direction"] / np.linalg.norm(m["B_direction"])
    assert b @ t == pytest.approx(1.0, abs=1e-6)  # B ~ grad(psi) x grad(alpha) runs along +phi, +theta
    assert abs(m["grad_alpha"] @ t) < 1e-3 * np.linalg.norm(m["grad_alpha"])
    assert abs(m["normal"] @ t) < 1e-3
    for k, line in m["lines"].items():
        np.testing.assert_allclose(line["label"], line["alpha"])  # each drawn line has one label


def test_bad_curvature_bands_sit_on_the_outboard_crossings():
    chart = vaft.diagram.ballooning_curvature_drive().model
    bands = chart.parameters["bad_bands"]
    for k in (-2, 0, 2):
        assert any(b0 <= k * np.pi <= b1 for b0, b1 in bands), k
    assert not any(b0 <= np.pi <= b1 for b0, b1 in bands)


def test_the_eigenfunction_cases_are_on_the_right_sides_of_the_boundaries():
    chart = vaft.diagram.ballooning_eigenfunction().model
    p = chart.parameters
    a1, a2 = p["alpha1"], p["alpha2"]
    cases = p["alpha"]
    assert cases["stable"] < a1 < cases["unstable"] < a2 < cases["second_stable"]
    assert np.all(chart.curves["stable"][:, 1] > 0)
    assert np.any(chart.curves["unstable"][:, 1] < 0)
    assert np.all(chart.curves["second_stable"][:, 1] > 0)


def test_the_harmonics_centre_on_nq_and_the_mode_balloons_outboard():
    chart = vaft.diagram.ballooning_harmonic_envelope(20).model
    p = chart.parameters
    m, amps = p["m"], p["amplitudes"]
    assert abs(m[np.argmax(amps)] - 20 * p["q"]) <= 1.0
    assert np.count_nonzero(amps > 0.1 * amps.max()) >= 3  # many harmonics coupled
    grid, field = p["field"].T
    assert abs(grid[np.argmax(np.abs(field))]) < 0.3  # localised at the outboard midplane
    with pytest.raises(ValueError):
        vaft.diagram.ballooning_harmonic_envelope(2)


def test_the_workflow_names_the_hierarchy_and_the_global_alternative():
    d = vaft.diagram.ballooning_workflow()
    for key in ("sfl", "label", "aligned", "tube", "global", "transform", "ode", "test"):
        assert key in d.model["nodes"]
    assert d.scene.role("node:global_mhd")


@pytest.mark.parametrize("name", ["clebsch_field_line_label", "ballooning_curvature_drive", "ballooning_eigenfunction",
                                  "ballooning_harmonic_envelope", "ballooning_workflow"])
def test_every_ballooning_diagram_is_deterministic_and_exported(name):
    fn = getattr(vaft.diagram, name)
    assert fn().tikz == fn().tikz
    assert name in vaft.diagram.__all__
    assert fn().scene.role("note") and not fn(labels=False).scene.role("note")
