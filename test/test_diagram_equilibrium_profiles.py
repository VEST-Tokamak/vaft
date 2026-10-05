"""Current-profile and q topology (#1604): every drawn number is the cylinder's, every crossing the processor's."""

import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq

import vaft.diagram
from vaft.diagram._equilibrium_profiles import _cylinder
from vaft.formula.equilibrium import li_3_from_Bp2_volume_integral
from vaft.formula.geometry import (
    cylindrical_enclosed_current,
    cylindrical_internal_inductance,
    cylindrical_poloidal_field,
    cylindrical_poloidal_flux,
    peaked_current_safety_factor,
)
from vaft.process.equilibrium import find_rational_surfaces


# ---------------------------------------------------------------------------
# the formulas
# ---------------------------------------------------------------------------


def test_the_enclosed_current_of_a_uniform_column_is_its_area_times_j():
    r = np.linspace(0.0, 0.4, 401)
    I = cylindrical_enclosed_current(r, np.full_like(r, 2e6))
    np.testing.assert_allclose(I, 2e6 * math.pi * r**2, rtol=1e-12, atol=1e-6)
    # a parabolic current (nu = 1): I = I_p [1 - (1 - x^2)^2], to second order in the spacing
    x = np.linspace(0.0, 1.0, 2001)
    I = cylindrical_enclosed_current(x, 1.0 - x**2)
    np.testing.assert_allclose(I / I[-1], 1.0 - (1.0 - x**2) ** 2, atol=1e-6)


def test_the_flux_of_a_uniform_current_is_quadratic_and_scales_with_R0():
    r = np.linspace(0.0, 0.5, 501)
    B = 0.2 * r / 0.5  # uniform current: B_theta grows linearly to 0.2 T at the edge
    psi = cylindrical_poloidal_flux(r, B, 1.5)
    np.testing.assert_allclose(psi, 1.5 * 0.2 * r**2 / (2 * 0.5), rtol=1e-12, atol=1e-15)
    assert cylindrical_poloidal_flux(r, B, 3.0)[-1] == pytest.approx(2.0 * psi[-1])


def test_a_uniform_current_has_l_i_one_half_and_a_peaked_one_more():
    r = np.linspace(0.0, 0.3, 3001)
    I = cylindrical_enclosed_current(r, np.full_like(r, 1e6))
    B = np.zeros_like(r)
    B[1:] = cylindrical_poloidal_field(r[1:], I[1:])
    assert cylindrical_internal_inductance(r, B) == pytest.approx(0.5, rel=1e-6)
    # j ~ 1 - x^2: B_theta / B_theta(a) = 2x - x^3, so l_i = 2 int_0^1 (2x - x^3)^2 x dx = 2 (1 - 2/3 + 1/8)
    x = np.linspace(0.0, 1.0, 4001)
    b = 2.0 * x - x**3
    expected = 2.0 * (1.0 - 2.0 / 3.0 + 1.0 / 8.0)
    assert cylindrical_internal_inductance(x, b) == pytest.approx(expected, rel=1e-6)
    assert expected > 0.5
    assert cylindrical_internal_inductance(r, 3.0 * B) == pytest.approx(0.5, rel=1e-6)  # a ratio of B^2


def test_the_cylinder_l_i_is_the_imas_l_i_3_of_a_straight_torus():
    a, R0, Ip = 0.4, 1.8, 3e5
    r = np.linspace(0.0, a, 4001)
    I = Ip * (1.0 - (1.0 - (r / a) ** 2) ** 3)  # j ~ (1 - x^2)^2
    B = np.zeros_like(r)
    B[1:] = cylindrical_poloidal_field(r[1:], I[1:])
    bp2_dv = 2.0 * math.pi * R0 * np.trapezoid(B**2 * 2.0 * math.pi * r, r)
    assert cylindrical_internal_inductance(r, B) == pytest.approx(li_3_from_Bp2_volume_integral(bp2_dv, Ip, R0),
                                                                  rel=1e-9)


@pytest.mark.parametrize("bad", [
    {"r": np.array([0.1, 0.2, 0.3])},          # not from the axis
    {"r": np.array([0.0, 0.2, 0.1])},          # not increasing
    {"j_z": np.ones(2)},                        # wrong shape
])
def test_the_cylinder_integrals_refuse_a_bad_grid(bad):
    kwargs = {"r": np.array([0.0, 0.1, 0.2]), "j_z": np.ones(3), **bad}
    with pytest.raises(ValueError):
        cylindrical_enclosed_current(**kwargs)
    with pytest.raises(ValueError):
        cylindrical_poloidal_flux(kwargs["r"], kwargs["j_z"], 1.0)
    with pytest.raises(ValueError):
        cylindrical_internal_inductance(kwargs["r"], kwargs["j_z"])


def test_l_i_needs_a_boundary_field_and_flux_a_positive_R0():
    r = np.linspace(0.0, 1.0, 11)
    with pytest.raises(ValueError):
        cylindrical_internal_inductance(r, np.zeros_like(r))
    for R0 in (0.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            cylindrical_poloidal_flux(r, r, R0)


# ---------------------------------------------------------------------------
# the reduced cylinder
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", ["peaked", "broad", "hollow"])
def test_every_shape_carries_the_same_current_and_its_field_is_ampere(shape):
    m = _cylinder(shape)
    x = m["x"]
    assert m["I"][-1] == pytest.approx(1.0)
    # j is j / j_bar with j_bar = I_p / pi a^2: its area average is one
    assert 2.0 * np.trapezoid(m["j"] * x, x) == pytest.approx(1.0, rel=1e-5)
    # B_theta / B_theta(a) = I(r) a / (I_p r), Ampère
    np.testing.assert_allclose(m["B_theta"][1:], m["I"][1:] / x[1:], rtol=1e-12)
    # q = q_a (r/a)^2 I_p / I(r) against quadrature, and on axis q0 = q_a j_bar / j(0)
    f = vaft.diagram._equilibrium_profiles._CURRENTS[shape]
    total = quad(lambda s: f(s) * s, 0.0, 1.0, epsabs=1e-13)[0]
    for xx in (0.2, 0.5, 0.8):
        expected = m["q_a"] * xx**2 * total / quad(lambda s: f(s) * s, 0.0, xx, epsabs=1e-13)[0]
        assert np.interp(xx, x, m["q"]) == pytest.approx(expected, rel=1e-4)
    assert m["q0"] == pytest.approx(m["q_a"] * 2.0 * total / f(0.0), rel=1e-5)
    assert m["q"][-1] == pytest.approx(m["q_a"])


def test_the_cylinder_reproduces_the_analytic_peaked_q():
    # the "peaked" shape is (1 - x^2)^2, nu = 2 of peaked_current_safety_factor
    m = _cylinder("peaked")
    np.testing.assert_allclose(m["q"], peaked_current_safety_factor(m["x"], m["q_a"], 2.0), rtol=1e-4)


def test_the_shapes_are_peaked_broad_and_hollow_and_l_i_follows_the_formula():
    models = {s: _cylinder(s) for s in ("peaked", "broad", "hollow")}
    assert np.argmax(models["peaked"]["j"]) == 0
    assert models["peaked"]["j"][0] > models["broad"]["j"][0] > models["hollow"]["j"][0]
    hollow = models["hollow"]["j"]
    assert hollow[0] < 0.5 * hollow.max() and 0.4 < models["hollow"]["x"][np.argmax(hollow)] < 0.8
    for m in models.values():
        assert m["l_i"] == pytest.approx(cylindrical_internal_inductance(m["x"], m["B_theta"]))
    assert models["peaked"]["l_i"] > models["broad"]["l_i"] > models["hollow"]["l_i"]


def test_the_diagram_draws_the_models_it_reports():
    d = vaft.diagram.current_profile_shapes()
    models = d.model["profiles"]
    assert set(models) == {"peaked", "broad", "hollow"}
    for key in ("j", "I", "B_theta"):
        for name, m in models.items():
            np.testing.assert_array_equal(d.model["charts"][key].curves[name][:, 1], m[key])
    text = " ".join(item.text for item in d.scene.items if hasattr(item, "text"))
    for m in models.values():
        assert f"l_i = {m['l_i']:.2f}" in text
    assert "peaked" in text and "hollow" in text


# ---------------------------------------------------------------------------
# q landmarks and topologies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("profile", ["monotonic", "reversed_shear"])
def test_the_landmarks_sit_where_they_are_defined(profile):
    chart = vaft.diagram.q_profile_landmarks(profile).model
    p = chart.parameters
    q = chart.curves["q"]
    assert chart.points["q0"] == (0.0, p["q0"]) and p["q0"] == q[0, 1]
    assert p["q_min"] == pytest.approx(q[:, 1].min())
    # q95 is q at psi_N = 0.95, not at r/a = 0.95
    m = _cylinder({"monotonic": "peaked", "reversed_shear": "hollow"}[profile])
    assert p["q95"] == pytest.approx(np.interp(p["x95"], m["x"], m["q"]))
    assert p["x95"] > 0.953 and p["q95"] > np.interp(0.95, m["x"], m["q"]) + 0.02
    # the edge q is its own landmark, never substituted for q95
    assert chart.points["q_a"] == (1.0, p["q_a"]) and p["q95"] < p["q_a"]


def test_psi_n_is_the_closed_form_of_the_peaked_current():
    # j ~ (1 - x^2)^2: I ~ 1 - (1 - u)^3 with u = x^2, so psi ~ (3u - 3u^2/2 + u^3/3)/2
    def psi_n(x):
        u = x * x
        return (3 * u - 1.5 * u * u + u**3 / 3) / (11.0 / 6.0)

    m = _cylinder("peaked")
    np.testing.assert_allclose(m["psi_n"], psi_n(m["x"]), atol=2e-6)
    x95 = brentq(lambda x: psi_n(x) - 0.95, 0.5, 1.0)
    assert vaft.diagram.q_profile_landmarks("monotonic").model.parameters["x95"] == pytest.approx(x95, abs=1e-5)
    assert x95 == pytest.approx(0.9551, abs=1e-4)


def test_q_min_is_on_axis_only_for_the_monotonic_profile():
    mono = vaft.diagram.q_profile_landmarks("monotonic").model.parameters
    rev = vaft.diagram.q_profile_landmarks("reversed_shear").model.parameters
    assert mono["x_min"] == 0.0 and mono["q_min"] == mono["q0"]
    assert 0.5 < rev["x_min"] < 0.9 and rev["q_min"] < rev["q0"]


def test_the_topologies_have_their_shear_signs():
    d = vaft.diagram.q_profile_topologies()
    runs = d.model["shear_runs"]
    signs = {p: {sign for sign, a, b in r} for p, r in runs.items()}
    assert "-" not in signs["monotonic"] and "-" not in signs["weak_shear"]
    assert "-" in signs["reversed_shear"]
    rev = d.model["profiles"]["reversed_shear"]
    # negative shear lies inside q_min, positive outside
    for sign, a, b in runs["reversed_shear"]:
        if sign == "-":
            assert b <= rev["x_min"] + 0.02
    assert all(s > 0 for s in rev["s"][rev["x"] > rev["x_min"] + 0.05])
    # the weak-shear core: a broad current keeps |s| small over half the radius
    weak = d.model["profiles"]["weak_shear"]
    assert np.all(np.abs(weak["s"][weak["x"] < 0.5]) < 0.1)


# ---------------------------------------------------------------------------
# rational-surface topology
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m, n", [(2, 1), (3, 2), (3, 1)])
def test_monotonic_q_crosses_m_over_n_once_and_reversed_shear_twice(m, n):
    one = vaft.diagram.rational_surface_topology("monotonic", m, n).model.parameters
    two = vaft.diagram.rational_surface_topology("reversed_shear", m, n).model.parameters
    assert len(one["crossings"]) == 1
    r1, r2 = two["crossings"]
    assert r1 < two["x_min"] < r2
    for p in (one, two):
        assert p["level"] == m / n
        f = vaft.diagram._equilibrium_profiles._CURRENTS[{"monotonic": "peaked",
                                                          "reversed_shear": "hollow"}[p["profile"]]]
        total = quad(lambda s: f(s) * s, 0.0, 1.0, epsabs=1e-13)[0]

        def q(xx):
            return p["q_a"] * xx**2 * total / quad(lambda s: f(s) * s, 0.0, xx, epsabs=1e-13)[0]

        edges = [0.01, p["x_min"], 1.0] if p["profile"] == "reversed_shear" else [0.01, 1.0]
        roots = [brentq(lambda xx: q(xx) - m / n, lo, hi) for lo, hi in zip(edges, edges[1:])]
        np.testing.assert_allclose(p["crossings"], roots, atol=1e-5)


def test_the_crossings_are_the_processors():
    chart = vaft.diagram.rational_surface_topology("reversed_shear", 2, 1).model
    q = chart.curves["q"]
    found = find_rational_surfaces(q[:, 0], q[:, 1], 1, m_range=(2, 2))
    assert list(found["m"]) == [2, 2]  # one helicity, two distinct surfaces
    np.testing.assert_array_equal(found["psi_n_rational"], chart.parameters["crossings"])


def test_double_rational_surfaces_are_not_called_a_double_tearing_mode():
    d = vaft.diagram.rational_surface_topology("reversed_shear", 2, 1)
    texts = [item.text for item in d.scene.items if hasattr(item, "text")]
    joined = " ".join(texts)
    assert "double-resonant configuration" in joined
    # "double tearing" appears only as the last, conditional step of the chain, never as the configuration
    tearing = [t for t in texts if "double tearing" in t or "double \\mbox{tearing" in t or "mbox{double tearing}" in t]
    assert len(tearing) == 1 and "mode" in tearing[0]
    roles = {item.role for item in d.scene.items if hasattr(item, "text") and item.text in tearing}
    assert roles == {"chain:4"}
    assert "stability calculation" in joined


def test_the_single_crossing_rational_surface_is_unchanged():
    chart = vaft.diagram.rational_surface(2, 1).model
    q = chart.curves["q_profile"]
    found = find_rational_surfaces(q[:, 0], q[:, 1], 1, m_range=(2, 2))
    assert len(found["m"]) == 1


@pytest.mark.parametrize("call", [
    lambda: vaft.diagram.current_profile_shapes(profiles=("peaked", "spiky")),
    lambda: vaft.diagram.current_profile_shapes(profiles=("peaked", "peaked")),
    lambda: vaft.diagram.current_profile_shapes(labels=1),
    lambda: vaft.diagram.q_profile_landmarks("weak_shear"),
    lambda: vaft.diagram.q_profile_landmarks(("monotonic", "reversed_shear")),
    lambda: vaft.diagram.rational_surface_topology(1, 2, 1),
    lambda: vaft.diagram.q_profile_topologies(profiles=()),
    lambda: vaft.diagram.rational_surface_topology("reversed_shear", 4, 2),
    lambda: vaft.diagram.rational_surface_topology("reversed_shear", 0, 1),
    lambda: vaft.diagram.rational_surface_topology("reversed_shear", True, 1),
    lambda: vaft.diagram.rational_surface_topology("hollow", 2, 1),
])
def test_bad_arguments_are_refused(call):
    with pytest.raises(ValueError):
        call()


@pytest.mark.parametrize("builder", ["current_profile_shapes", "q_profile_landmarks", "q_profile_topologies",
                                     "rational_surface_topology"])
def test_labels_false_draws_no_text_beyond_the_axes(builder):
    d = getattr(vaft.diagram, builder)(labels=False)
    roles = {item.role for item in d.scene.items if hasattr(item, "text")}
    assert roles <= {"axes", "ticks", "psi_n_axis"}
