"""Cylindrical geometry (#1072): the q profile is the formula's, the pictures follow it."""

import math

import numpy as np
import pytest

import vaft.diagram
from vaft.formula.constants import MU0
from vaft.formula.geometry import (
    cylindrical_poloidal_field,
    cylindrical_safety_factor_from_r_B,
    peaked_current_safety_factor,
)
from vaft.formula.stability import delta_prime_from_outer_derivatives


def test_the_peaked_current_q_is_ampere_plus_the_cylindrical_q():
    a, R0, Bz, Ia, nu = 0.5, 1.5, 2.0, 2e5, 1.5
    x = np.array([0.2, 0.5, 0.9, 1.0])
    I = Ia * (1 - (1 - x**2) ** (nu + 1))
    q = cylindrical_safety_factor_from_r_B(x * a, cylindrical_poloidal_field(x * a, I), Bz, R0)
    q_a = cylindrical_safety_factor_from_r_B(a, cylindrical_poloidal_field(a, Ia), Bz, R0)
    np.testing.assert_allclose(peaked_current_safety_factor(x, q_a, nu), q)
    assert peaked_current_safety_factor(0.0, 3.0, 2.0) == pytest.approx(1.0)  # q_a / (nu + 1)
    assert cylindrical_poloidal_field(0.1, 1e5) == pytest.approx(MU0 * 1e5 / (2 * math.pi * 0.1))
    for bad in ({"x": 1.2}, {"q_a": 0.0}, {"nu": -1.0}):
        with pytest.raises(ValueError):
            peaked_current_safety_factor(**{"x": 0.5, "q_a": 3.0, "nu": 1.0, **bad})


def test_the_profile_panels_are_the_chain():
    m = vaft.diagram.current_to_q_profile(nu=2.0, q_a=3.0).model
    assert m["q"][0] == pytest.approx(1.0)
    assert np.all(np.diff(m["q"]) >= -1e-12)
    assert m["j"][-1] == pytest.approx(0.0) and m["j"][0] == 1.0


@pytest.mark.parametrize("n", [1, 2])
def test_every_rational_surface_sits_where_q_is_m_over_n(n):
    chart = vaft.diagram.cylindrical_rational_surfaces(n).model
    surfaces = chart.parameters["surfaces"]
    assert surfaces
    for m, rs in surfaces.items():
        assert float(peaked_current_safety_factor(rs, 3.5, 1.0)) == pytest.approx(m / n, abs=1e-9)
        assert math.gcd(m, n) == 1
    assert list(surfaces.values()) == sorted(surfaces.values())


def test_the_mode_shapes_have_their_m():
    m = vaft.diagram.cylindrical_mode_morphology().model
    for mm, pts in m["shapes"].items():
        if mm == 1:
            continue
        cx = 0.5 * (pts[:, 0].max() + pts[:, 0].min())
        r = np.hypot(pts[:, 0] - cx, pts[:, 1])
        spec = np.abs(np.fft.rfft(r[:-1] - r[:-1].mean()))
        assert int(np.argmax(spec)) == mm


def test_the_internal_kink_stops_at_q_equal_one():
    chart = vaft.diagram.internal_external_kink().model
    r1 = chart.parameters["r1"]
    assert float(peaked_current_safety_factor(r1, 2.5, 2.0)) == pytest.approx(1.0, abs=1e-9)
    x, xi = chart.curves["internal"].T
    assert xi[x < r1 - 0.1].min() > 0.99 and xi[x > r1 + 0.1].max() < 0.01
    x, xi = chart.curves["external"].T
    assert xi[-1] == pytest.approx(1.0)


@pytest.mark.parametrize("m", [1, 2, 3])
def test_the_vacuum_solution_meets_the_plasma_and_vanishes_on_the_wall(m):
    chart = vaft.diagram.plasma_vacuum_wall(m).model
    p = chart.parameters
    plasma, wall = chart.curves["plasma"], chart.curves["vacuum_wall"]
    assert plasma[-1, 1] == pytest.approx(wall[0, 1])
    assert wall[-1, 1] == pytest.approx(0.0, abs=1e-12)
    r = wall[:, 0]
    np.testing.assert_allclose(wall[:, 1], p["A"] * r**m + p["B"] * r ** (-m))


def test_the_tearing_outer_solution_meets_at_r_s_with_its_delta_prime():
    chart = vaft.diagram.cylindrical_tearing_outer(2, 1).model
    p = chart.parameters
    left, right = chart.curves["outer_left"], chart.curves["outer_right"]
    assert left[-1, 1] == pytest.approx(1.0) and right[0, 1] == pytest.approx(1.0)
    assert right[-1, 1] == pytest.approx(0.0, abs=1e-12)  # ideal wall
    assert float(peaked_current_safety_factor(p["r_s"], 3.5, 1.0)) == pytest.approx(2.0, abs=1e-9)
    assert p["delta_prime"] == pytest.approx(
        delta_prime_from_outer_derivatives(1.0, p["dpsi_dr_minus"], p["dpsi_dr_plus"]))
    with pytest.raises(ValueError):
        vaft.diagram.cylindrical_tearing_outer(1, 1)  # no q = 1 surface in this profile


@pytest.mark.parametrize("name", ["current_to_q_profile", "cylindrical_rational_surfaces",
                                  "cylindrical_mode_morphology", "internal_external_kink", "plasma_vacuum_wall",
                                  "cylindrical_tearing_outer"])
def test_every_cylindrical_diagram_is_deterministic_and_exported(name):
    fn = getattr(vaft.diagram, name)
    assert fn().tikz == fn().tikz
    assert name in vaft.diagram.__all__
    assert fn().scene.role("note") and not fn(labels=False).scene.role("note")
