"""Field-aligned coordinates part 2 (#1075): eigenmode, k_x(theta), basis, shear, flux tube."""

import math

import numpy as np
import pytest

import vaft.diagram
from vaft.diagram import _field_aligned as fa
from vaft.diagram._scene import Label, Polyline
from vaft.formula.stability import (
    ballooning_radial_wavenumber,
    field_line_label,
    s_alpha_ballooning_eigenmode,
    s_alpha_ballooning_stable,
    s_alpha_curvature_drive,
)


@pytest.mark.parametrize("s, alpha", [(1.0, 0.3), (1.0, 0.7), (1.0, 1.2), (0.4, 0.5), (1.5, 2.0), (0.5, 2.5)])
def test_the_eigenmode_agrees_with_newcomb_and_localises_when_unstable(s, alpha):
    g, theta, F = s_alpha_ballooning_eigenmode(s, alpha)
    stable = s_alpha_ballooning_stable(s, alpha)
    assert (g < 0.0) == stable  # an independent integrator decides the same way
    np.testing.assert_allclose(F, F[::-1], atol=1e-6)  # even
    tail = np.max(np.abs(F[np.abs(theta) > 4 * math.pi]))
    assert np.all(F >= -1e-9)  # the top mode is nodeless
    if not stable:
        assert F[len(F) // 2] == pytest.approx(1.0)  # peaked at the outboard midplane
        assert tail < 0.05  # it balloons: localised near theta = 0


def test_the_growth_rate_is_converged_in_the_box_and_the_grid():
    # CHT s-alpha at (1, 1.2): an independent tridiagonal solve gives 0.3867
    g, _, _ = s_alpha_ballooning_eigenmode(1.0, 1.2)
    assert g == pytest.approx(0.3867, abs=2e-4)
    assert s_alpha_ballooning_eigenmode(1.0, 1.2, theta_max=12 * math.pi, n_points=4801)[0] == pytest.approx(g, abs=2e-4)
    # a stable surface: the top of the continuum approaches zero from below as the box grows
    small = s_alpha_ballooning_eigenmode(1.0, 0.3)[0]
    large = s_alpha_ballooning_eigenmode(1.0, 0.3, theta_max=12 * math.pi, n_points=2401)[0]
    assert small < large < 0.0


def test_the_radial_wavenumber_is_the_lambda_of_the_s_alpha_model():
    theta = np.linspace(-6.0, 6.0, 25)
    for s, alpha, t0 in ((1.0, 0.0, 0.0), (0.7, 0.5, 0.3)):
        lam = ballooning_radial_wavenumber(1.0, s, theta, alpha, t0)
        # s_alpha_curvature_drive's K = cos(theta) + Lambda sin(theta), with the same Lambda
        K = s_alpha_curvature_drive(theta, s, alpha, t0)
        np.testing.assert_allclose(K, np.cos(theta) + lam * np.sin(theta), atol=1e-12)
    # alpha = 0: the sheared slab, k_x = k_x0 + k_y s z with k_x0 = -k_y s theta0
    assert ballooning_radial_wavenumber(2.0, 0.8, 3.0, 0.0, 0.5) == pytest.approx(2.0 * 0.8 * 3.0 - 2.0 * 0.8 * 0.5)
    with pytest.raises(ValueError):
        ballooning_radial_wavenumber(np.inf, 1.0, 0.0)


def test_the_basis_has_b_along_the_lines_and_grad_alpha_across():
    q = 3.0
    model = vaft.diagram.field_aligned_basis(q=q).model
    b, g = np.asarray(model["B_direction"]), np.asarray(model["grad_alpha_direction"])
    # derive both from the formula: d_i alpha by finite differences, B^i from integrating d phi / d theta = q
    phi, theta, h = 0.7, 0.4, 1e-6
    d_alpha = np.array([(field_line_label(phi + h, theta, q) - field_line_label(phi - h, theta, q)) / (2 * h),
                        (field_line_label(phi, theta + h, q) - field_line_label(phi, theta - h, q)) / (2 * h)])
    step = 1e-3
    line = np.array([phi + q * step, theta + step]) - np.array([phi, theta])  # one step along B
    assert field_line_label(*(np.array([phi, theta]) + line), q) == pytest.approx(field_line_label(phi, theta, q))
    np.testing.assert_allclose(b, line / np.linalg.norm(line), atol=1e-9)
    np.testing.assert_allclose(g, d_alpha / np.linalg.norm(d_alpha), atol=1e-6)
    assert np.dot(line, d_alpha) == pytest.approx(0.0, abs=1e-9)  # B^i d_i alpha = 0


def test_shear_rotates_the_drawn_fronts_at_fixed_binormal_period():
    diagram = vaft.diagram.magnetic_shear_field_aligned(shear=1.5)
    model = diagram.model
    for i, (theta, k_x) in enumerate(zip(model["theta"], model["k_x"])):
        assert k_x == pytest.approx(ballooning_radial_wavenumber(model["k_y"], 1.5, theta))
        fronts = [np.asarray(p.points) for p in diagram.scene.items
                  if isinstance(p, Polyline) and p.role == f"front_{i}"]
        assert len(fronts) >= 3
        k = np.array([k_x, model["k_y"]])
        # each drawn front is normal to (k_x, k_y), i.e. a line of constant phase k . r
        for f in fronts:
            assert np.dot(f[1] - f[0], k) == pytest.approx(0.0, abs=1e-9)
        # consecutive fronts differ in phase by one binormal wavelength: fixed k_y period in every plane
        phases = np.sort([np.dot(f[0], k) for f in fronts])
        np.testing.assert_allclose(np.diff(phases), model["k_y"] * model["wavelength_y"], rtol=1e-9)


def test_the_figure_contrasts_an_unstable_and_a_stable_surface():
    chart = vaft.diagram.ballooning_eigenfunction().model
    assert chart.parameters["growth_rate_squared_unstable"] > 0 > chart.parameters["growth_rate_squared_stable"]
    assert not s_alpha_ballooning_stable(*fa.UNSTABLE) and s_alpha_ballooning_stable(*fa.STABLE)


def test_flux_tube_steps_and_labels_off():
    assert vaft.diagram.flux_tube_patch().model["steps"][-1] == "flux_tube"
    for build in (vaft.diagram.field_aligned_basis, vaft.diagram.flux_tube_patch,
                  vaft.diagram.magnetic_shear_field_aligned, vaft.diagram.ballooning_eigenfunction):
        assert not [i for i in build(labels=False).scene.items
                    if isinstance(i, Label) and i.role not in ("axes", "ticks")]


# --- #1075 remainder: transits, boundary conditions, X-point limit ---------------------------------


def test_each_transit_revisits_the_outboard_point_with_less_of_the_mode():
    d = vaft.diagram.ballooning_transit_map(transits=2)
    a = d.model["amplitude"]
    assert a[0] == pytest.approx(1.0)
    for k in (1, 2):
        assert a[k] == pytest.approx(a[-k], rel=1e-6)  # symmetric mode
        assert abs(a[k]) < abs(a[k - 1])
    _, th, F = s_alpha_ballooning_eigenmode(*fa.UNSTABLE)
    assert a[1] == pytest.approx(np.interp(2 * math.pi, th, F / np.max(np.abs(F))), rel=1e-9)
    assert len([it for it in d.scene.role("cross_section")]) == 5
    with pytest.raises(ValueError):
        vaft.diagram.ballooning_transit_map(transits=0)


def test_the_flux_tube_end_rejoins_with_the_sheared_kx():
    d = vaft.diagram.ballooning_boundary_conditions(shear=1.0)
    assert d.model["kx_after_one_turn"] == pytest.approx(float(ballooning_radial_wavenumber(1.0, 1.0, 2 * math.pi)))
    assert d.scene.role("twist_and_shift") and d.scene.role("decay")


def test_theta_star_crowds_into_the_x_point_and_q_diverges():
    m = vaft.diagram.field_aligned_xpoint_limitation().model
    q = np.array(m["q_relative"])
    assert np.all(np.diff(q) > 0) and q[-1] > 4.0
    assert 0.3 < m["fraction_near_x_point"] < 0.9
    # a property of the surface, not of how many lines are drawn
    assert vaft.diagram.field_aligned_xpoint_limitation(n_theta=8).model["fraction_near_x_point"] == \
        pytest.approx(m["fraction_near_x_point"])
    # the core surface spends none of its angle near the X-point
    from vaft.diagram._gs_equilibrium import flux_model

    model = flux_model("diverted")
    *_, core = fa._straight_field_line_points(model, 0.3, 24, near=(model["x_point"], 0.15))
    assert core == 0.0
    with pytest.raises(ValueError):
        vaft.diagram.field_aligned_xpoint_limitation(n_theta=4)


def test_theta_star_steps_are_equal_in_the_line_integral():
    from vaft.diagram._gs_equilibrium import flux_model

    model = flux_model("diverted")
    a, loop_a = fa._straight_field_line_points(model, 0.5, 12)
    b, loop_b = fa._straight_field_line_points(model, 0.5, 24)
    assert loop_a == pytest.approx(loop_b)
    assert np.allclose(a, b[::2], atol=1e-9)  # every other point of the finer set


@pytest.mark.parametrize("name", ["ballooning_transit_map", "ballooning_boundary_conditions",
                                  "field_aligned_xpoint_limitation"])
def test_the_remainder_figures_are_deterministic_and_exposed(name):
    fn = getattr(vaft.diagram, name)
    assert name in vaft.diagram.__all__
    assert fn().tikz == fn().tikz
    assert fn(labels=False).tikz != fn().tikz


def test_the_transit_map_box_reaches_past_the_drawn_range():
    d = vaft.diagram.ballooning_transit_map(transits=3)
    a = d.model["amplitude"]
    assert a[3] != 0.0 and abs(a[3]) < abs(a[2])  # not the Dirichlet end of the box
    assert d.model["amplitude"][1] == pytest.approx(vaft.diagram.ballooning_transit_map().model["amplitude"][1],
                                                    rel=1e-3)


@pytest.mark.parametrize("bad", [0.0, -1.0, 10.0, "1", True])
def test_the_boundary_condition_shear_is_validated(bad):
    with pytest.raises(ValueError):
        vaft.diagram.ballooning_boundary_conditions(shear=bad)


def test_numpy_integers_are_accepted_as_counts():
    assert vaft.diagram.ballooning_transit_map(transits=np.int64(1)).model["transits"] == (-1, 0, 1)
    assert vaft.diagram.field_aligned_xpoint_limitation(n_theta=np.int64(16)).model["n_theta"] == 16
