"""3-D harmonic diagrams: one concept each, and every drawn phasor is the complex number it claims (#1088)."""

import math

import numpy as np
import pytest

import vaft.diagram
from vaft.formula.stability import helical_harmonic, helical_phase

NAMES = ("normal_field_component", "complex_harmonic", "toroidal_harmonic_phase",
         "harmonic_real_space_projection", "complex_field_superposition")


def _arrow(diagram, role):
    arrows = [it for it in diagram.scene.role(role) if hasattr(it, "end")]
    assert len(arrows) == 1, role
    return np.asarray(arrows[0].start), np.asarray(arrows[0].end)


# --- the formula ---------------------------------------------------------------------------


def test_helical_harmonic_is_the_real_part_in_the_helical_phase():
    theta, phi = np.linspace(0, 2 * np.pi, 7), np.linspace(0, np.pi, 7)
    b = 0.8 - 0.3j
    xi = helical_phase(theta, phi, 3, 2)
    np.testing.assert_allclose(helical_harmonic(b, theta, phi, 3, 2), b.real * np.cos(xi) - b.imag * np.sin(xi))
    # amplitude and a crest where xi = -arg(b)
    alpha = np.angle(b)
    assert helical_harmonic(b, -alpha / 3, 0.0, 3, 2) == pytest.approx(abs(b))
    # a real coefficient needs no imaginary part; i*b is a quarter period ahead
    assert helical_harmonic(1.0, 0.0, 0.0, 1, 1) == pytest.approx(1.0)
    assert helical_harmonic(1j, 0.0, np.pi / 2, 1, 1) == pytest.approx(1.0)  # crest at xi = -pi/2
    with pytest.raises(ValueError):
        helical_harmonic(1.0, 0.0, 0.0, 0, 1)


def test_a_sampled_toroidal_decomposition_is_the_conjugate_coefficient():
    from vaft.process import toroidal_mode_decomposition

    n, A, delta = 2, 1.3, 0.7
    phi = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    C = toroidal_mode_decomposition(phi, A * np.cos(n * phi + delta), [n])[n]
    b_hat = 2 * np.conj(C)  # as helical_harmonic's Convention states
    # at theta = 0 the m-part drops out, whatever m is
    np.testing.assert_allclose(helical_harmonic(b_hat, 0.0, phi, 1, n), A * np.cos(n * phi + delta), atol=1e-12)
    assert not np.allclose(helical_harmonic(2 * C, 0.0, phi, 1, n), A * np.cos(n * phi + delta))


# --- normal component ----------------------------------------------------------------------


def test_the_normal_and_tangential_parts_decompose_the_perturbation():
    d = vaft.diagram.normal_field_component()
    v = d.model.vectors
    n, t = v["normal"], v["tangent"]
    assert np.linalg.norm(n) == pytest.approx(1.0) and n @ t == pytest.approx(0.0, abs=1e-12)
    nc = v["normal_component"]
    assert nc[0] * n[1] - nc[1] * n[0] == pytest.approx(0.0, abs=1e-12)  # parallel to n
    assert v["tangential_component"] @ n == pytest.approx(0.0, abs=1e-12)
    assert v["normal_component"] + v["tangential_component"] == pytest.approx(v["perturbation"])
    # n is the surface normal and t its tangent at the point
    radial = v["point"] - v["surface_centre"]
    assert radial / np.linalg.norm(radial) == pytest.approx(n)
    assert np.min(np.linalg.norm(v["surface"] - v["point"], axis=1)) < 0.1  # the point is on the drawn surface
    for role, key in (("perturbation", "perturbation"), ("normal_component", "normal_component"),
                      ("tangential_component", "tangential_component")):
        start, end = _arrow(d, role)
        assert start == pytest.approx(v["point"]) and end - start == pytest.approx(v[key])


# --- complex harmonic -------------------------------------------------------------------------


@pytest.mark.parametrize("amplitude, phase", [(1.0, math.pi / 3), (2.5, -2.0), (0.3, math.pi), (1.0, 0.0)])
def test_the_quadratures_are_a_cos_and_a_sin(amplitude, phase):
    d = vaft.diagram.complex_harmonic(amplitude, phase)
    p = d.model.parameters
    assert p["real"] == pytest.approx(amplitude * math.cos(phase), abs=1e-12)
    assert p["imag"] == pytest.approx(amplitude * math.sin(phase), abs=1e-12)
    start, end = _arrow(d, "phasor")
    assert start == pytest.approx(0.0) and end == pytest.approx(np.array([p["real"], p["imag"]]) * p["cm_per_unit"])
    (real,) = [it for it in d.scene.role("real_component") if hasattr(it, "points")]
    (imag,) = [it for it in d.scene.role("imag_component") if hasattr(it, "points")]
    assert real.points[1][0] == pytest.approx(end[0]) and real.points[1][1] == 0.0
    assert imag.points[1][1] == pytest.approx(end[1]) and imag.points[1][0] == 0.0


def test_the_amplitude_is_printed_on_its_circle():
    (label,) = [it for it in vaft.diagram.complex_harmonic(2.5, 0.4).scene.role("amplitude") if hasattr(it, "text")]
    assert "2.5" in label.text
    assert vaft.diagram.complex_harmonic(2.5, 0.4).tikz != vaft.diagram.complex_harmonic(1.0, 0.4).tikz


@pytest.mark.parametrize("kw", [{"amplitude": 0.0}, {"amplitude": -1.0}, {"amplitude": float("nan")},
                                {"phase": float("inf")}, {"amplitude": True}])
def test_complex_harmonic_rejects_bad_values(kw):
    with pytest.raises(ValueError):
        vaft.diagram.complex_harmonic(**kw)


# --- toroidal phase -----------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 3, 7])
def test_a_toroidal_shift_turns_the_coefficient_by_minus_n_delta_phi(n):
    m = vaft.diagram.toroidal_harmonic_phase(n).model
    b0, b1 = m.values["phase_zero"], m.values["phase_shifted"]
    assert abs(b1) == pytest.approx(abs(b0))  # the amplitude is invariant
    turn = np.angle(b1 / b0)
    expected = -n * m.parameters["delta_phi"]
    assert turn == pytest.approx(math.atan2(math.sin(expected), math.cos(expected)))
    assert m.parameters["rotation"] == pytest.approx(expected)
    # the quadratures are not invariant
    assert abs(b1.real - b0.real) + abs(b1.imag - b0.imag) > 0.1
    # and the same shift reproduces the real field at shifted phi
    for theta in (0.0, 0.7):
        assert helical_harmonic(b1, theta, 0.3, 2, n) == pytest.approx(
            helical_harmonic(b0, theta, 0.3 + m.parameters["delta_phi"], 2, n))


@pytest.mark.parametrize("n", [0, -1, 8, 1.0, True])
def test_toroidal_harmonic_phase_needs_a_small_positive_n(n):
    with pytest.raises(ValueError):
        vaft.diagram.toroidal_harmonic_phase(n)


# --- real-space projection -----------------------------------------------------------------------


@pytest.mark.parametrize("m, n, phase", [(2, 1, math.pi / 3), (3, 2, -1.0), (1, 1, 0.0)])
def test_crests_and_troughs_are_the_formulas_extrema(m, n, phase):
    chart = vaft.diagram.harmonic_real_space_projection(m, n, phase).model
    b = complex(math.cos(phase), math.sin(phase))
    crests = [c for k, c in chart.curves.items() if k.startswith("crest_")]
    troughs = [c for k, c in chart.curves.items() if k.startswith("trough_")]
    assert crests and troughs
    for curves, value in ((crests, 1.0), (troughs, -1.0)):
        for line in curves:
            phi, theta = line[::20].T
            np.testing.assert_allclose(helical_harmonic(b, theta, phi, m, n), value, atol=1e-12)
            slope = np.diff(line[:, 1]) / np.diff(line[:, 0])
            np.testing.assert_allclose(slope, n / m)  # a field line of q = m/n keeps the phase
    assert chart.parameters["sample_value"] == pytest.approx(1.0)


def test_the_drawn_stripes_and_the_sample_are_on_the_chart():
    d = vaft.diagram.harmonic_real_space_projection()
    assert len(d.scene.role("spatial_pattern")) >= 4
    (marker,) = d.scene.role("real_projection")
    assert np.allclose(marker.at, d.model.to_cm(np.array(d.model.points["sample"])))
    (box,) = d.scene.role("equations")
    from vaft.diagram._equations import formula_equation

    assert formula_equation(helical_harmonic) in box.text


# --- superposition ---------------------------------------------------------------------------------


@pytest.mark.parametrize("case", ["screening", "amplification", "phase_shift"])
def test_the_total_closes_the_complex_triangle(case):
    d = vaft.diagram.complex_field_superposition(case)
    v, unit = d.model.values, d.model.parameters["cm_per_unit"]
    assert v["total"] == pytest.approx(v["external"] + v["plasma"])
    ext0, ext1 = _arrow(d, "external")
    pl0, pl1 = _arrow(d, "plasma")
    tot0, tot1 = _arrow(d, "total")
    assert ext1 == pytest.approx(pl0) and pl1 == pytest.approx(tot1) and ext0 == pytest.approx(tot0)
    assert tot1 == pytest.approx(np.array([v["total"].real, v["total"].imag]) * unit)
    ratio = abs(v["total"]) / abs(v["external"])
    assert {"screening": ratio < 0.5, "amplification": ratio > 1.5, "phase_shift": abs(np.angle(v["total"])) > 0.5}[case]


def test_superposition_takes_a_named_case():
    with pytest.raises(ValueError, match="case"):
        vaft.diagram.complex_field_superposition("shielding")


# --- common --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_every_harmonic_diagram_is_deterministic_lazy_and_has_no_labels_on_request(name):
    fn = getattr(vaft.diagram, name)
    assert fn().tikz == fn().tikz
    assert name in vaft.diagram.__all__
    assert fn().scene.role("equations") and not fn(labels=False).scene.role("equations")
    with pytest.raises(ValueError, match="labels"):
        fn(labels=1)
