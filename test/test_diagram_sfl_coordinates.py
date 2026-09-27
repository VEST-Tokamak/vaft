"""Straight-field-line coordinates (#1074): one equilibrium, one family of angles."""

import math

import numpy as np
import pytest

import vaft.diagram
from vaft.diagram import _sfl_coordinates as sfl
from vaft.formula.equilibrium import generalized_straight_field_line_angle, straight_field_line_angle


def test_every_member_is_a_monotonic_angle_and_they_differ():
    s = sfl._surface(0.8)
    angles = s["angles"]
    for name, a in angles.items():
        assert np.all(np.diff(a) > 0), name
        assert a[-1] - a[0] == pytest.approx(2 * math.pi), name
    names = list(angles)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            diff = np.max(np.abs(angles[names[i]] - angles[names[j]]))
            if {names[i], names[j]} == {"PEST", "Boozer"}:
                continue  # nearly equal when B_p << B_phi: B^2 ~ R^-2
            assert diff > 0.05, (names[i], names[j])


def test_equal_arc_spaces_the_angle_evenly_along_the_surface():
    s = sfl._surface(0.8)
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(s["R"]), np.diff(s["Z"])))])
    np.testing.assert_allclose(s["angles"]["equal-arc"], 2 * math.pi * arc / arc[-1], atol=2e-3)


def test_the_generalised_formula_reduces_to_pest_and_is_invariant_to_scale():
    theta = np.linspace(0, 2 * np.pi, 401)
    R = 3 + 0.5 * np.cos(theta)
    jac = R * 0.5 * (1 + 0.1 * np.cos(theta))
    Bp = 0.2 * (1 + 0.3 * np.cos(theta))
    B = 3 / R
    np.testing.assert_allclose(generalized_straight_field_line_angle(theta, jac, R, Bp, B),
                               straight_field_line_angle(theta, jac, R))
    a = generalized_straight_field_line_angle(theta, jac, R, Bp, B, 1, 2, 0)
    b = generalized_straight_field_line_angle(theta, 7 * jac, R, 3 * Bp, 2 * B, 1, 2, 0)
    np.testing.assert_allclose(a, b)  # only the variation along the surface matters
    with pytest.raises(ValueError):
        generalized_straight_field_line_angle(theta, jac, R, -Bp, B)


def test_the_grids_share_the_surfaces_and_differ_only_in_angle():
    d = vaft.diagram.sfl_coordinate_grids()
    names = list(d.model["coordinates"])
    surfaces = {n: [np.asarray(it.points) for it in d.scene.role(f"surface:{n}") if hasattr(it, "points")]
                for n in names}
    ref = surfaces[names[0]]
    ref_x0 = np.mean(ref[-1][:, 0])
    for n in names[1:]:
        shift = np.mean(surfaces[n][-1][:, 0]) - ref_x0
        for a, b in zip(ref, surfaces[n]):
            np.testing.assert_allclose(b - np.array([shift, 0.0]), a, atol=1e-9)
    grids = {n: np.vstack([np.asarray(it.points) for it in d.scene.role(f"grid:{n}")]) for n in names}
    assert not np.allclose(grids["PEST"][:, 0] - grids["PEST"][:, 0].mean(),
                           grids["Hamada"][:, 0] - grids["Hamada"][:, 0].mean())


def test_the_spectra_legend_states_the_computed_widths():
    d = vaft.diagram.sfl_fourier_convergence()
    spectra = sfl.perturbation_spectra()
    for name, v in spectra.items():
        (label,) = [it for it in d.scene.role(f"legend:{name}") if hasattr(it, "text")]
        assert f"m \\le {v['m99']}" in label.text
    widths = [v["m99"] for v in spectra.values()]
    assert max(widths) > 2 * min(widths)  # the same structure costs very different numbers of harmonics


def test_cocos_stands_apart_from_the_coordinate_tree():
    d = vaft.diagram.sfl_coordinate_taxonomy()
    assert "cocos" not in [n for n in d.model["nodes"]]
    edges = {it.role for it in d.scene.items if it.role.startswith("edge:")}
    assert not any("cocos" in e for e in edges)
    assert d.scene.role("node:cocos")
    for node in ("node:pest", "node:boozer", "node:hamada", "node:equal_arc", "node:clebsch", "node:derived",
                 "node:canonical", "node:generalized_boozer"):
        assert d.scene.role(node), node


@pytest.mark.parametrize("name", ["sfl_coordinate_grids", "sfl_coordinate_taxonomy", "sfl_fourier_convergence"])
def test_every_sfl_diagram_is_deterministic_and_exported(name):
    fn = getattr(vaft.diagram, name)
    assert fn().tikz == fn().tikz
    assert name in vaft.diagram.__all__
    assert fn().scene.role("note") and not fn(labels=False).scene.role("note")
