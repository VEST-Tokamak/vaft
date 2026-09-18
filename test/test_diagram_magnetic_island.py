"""Physics and topology of the magnetic-island schematic (issue #890).

These run on the model and the renderer-neutral scene, so they need no TeX:
a picture can only be right if the geometry it is drawn from is.
"""

import math

import numpy as np
import pytest

import vaft.diagram
from vaft.diagram import _magnetic_island as mi
from vaft.diagram._scene import Marker, Polyline
from vaft.formula.equilibrium import straight_field_line_angle
from vaft.formula.stability import (
    helical_phase,
    island_pendulum_hamiltonian,
    island_separatrix_half_width,
)

PROJECTIONS = ("poloidal", "top", "3d")


def _wrap(angle):
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


D_SHAPE = {"elongation": 1.7, "triangularity": 0.4}


def _model(**kw):
    args = dict(m=2, n=1, width=0.24, phase=0.0, projection="poloidal", r_s=0.64, aspect_ratio=3.2, elongation=1.0)
    args.update(kw)
    return mi._validate(**args)


# --- formula layer -----------------------------------------------------------


def test_helical_phase_follows_the_documented_convention():
    assert helical_phase(0.3, 0.7, 3, 2, 0.1) == pytest.approx(3 * 0.3 - 2 * 0.7 - 0.1)
    # constant along a field line of q = m/n: dtheta/dphi = n/m
    phi = np.linspace(0, 4 * np.pi, 9)
    assert np.allclose(helical_phase(0.2 + phi / 2, phi, 2, 1), helical_phase(0.2, 0.0, 2, 1))


@pytest.mark.parametrize("m, n", [(0, 1), (2, 0), (-2, 1), (2.5, 1)])
def test_helical_phase_rejects_bad_mode_numbers(m, n):
    with pytest.raises(ValueError):
        helical_phase(0.0, 0.0, m, n)


def test_pendulum_hamiltonian_has_its_minimum_at_the_o_point_and_saddle_at_the_x_point():
    w = 0.3
    A = (w / 4) ** 2
    assert island_pendulum_hamiltonian(0.0, 0.0, w) == pytest.approx(-A)
    assert island_pendulum_hamiltonian(0.0, np.pi, w) == pytest.approx(A)
    # O: a minimum in both directions; X: rises in x, falls in xi
    eps = 1e-3
    assert island_pendulum_hamiltonian(eps, 0.0, w) > -A
    assert island_pendulum_hamiltonian(0.0, eps, w) > -A
    assert island_pendulum_hamiltonian(eps, np.pi, w) > A
    assert island_pendulum_hamiltonian(0.0, np.pi + eps, w) < A


def test_separatrix_is_the_x_point_level_and_gives_the_full_width():
    w = 0.24
    xi = np.linspace(-np.pi, np.pi, 101)
    x = island_separatrix_half_width(xi, w)
    assert np.allclose(island_pendulum_hamiltonian(x, xi, w), island_pendulum_hamiltonian(0.0, np.pi, w))
    assert 2 * island_separatrix_half_width(0.0, w) == pytest.approx(w)
    assert island_separatrix_half_width(np.pi, w) == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("fn", [island_pendulum_hamiltonian, island_separatrix_half_width])
def test_island_formulas_reject_a_non_positive_width(fn):
    args = (0.0, 0.0, 0.0) if fn is island_pendulum_hamiltonian else (0.0, -1.0)
    with pytest.raises(ValueError):
        fn(*args)


@pytest.mark.parametrize("eps", [0.05, 0.3, 0.7])
def test_straight_field_line_angle_matches_concentric_circles(eps):
    # concentric circles: J = r R, so J / R^2 = r / R
    theta = np.linspace(0.0, 2 * np.pi, 4001)
    R = 1.0 + eps * np.cos(theta)
    star = straight_field_line_angle(theta, eps * R, R)
    exact = np.mod(2 * np.arctan(np.sqrt((1 - eps) / (1 + eps)) * np.tan(theta / 2)), 2 * np.pi)
    exact[-1] = 2 * np.pi
    inner = slice(1, -1)
    assert np.allclose(_wrap(star[inner] - exact[inner]), 0.0, atol=1e-6)
    assert star[0] == 0.0 and star[-1] == pytest.approx(2 * np.pi)


def test_straight_field_line_angle_is_geometric_at_large_aspect_ratio():
    theta = np.linspace(-np.pi, np.pi, 721)
    R = 1e6 + np.cos(theta)
    assert np.allclose(straight_field_line_angle(theta, R, R), theta, atol=1e-5)


@pytest.mark.parametrize("theta", [np.linspace(0, np.pi, 50), np.linspace(0, 2 * np.pi, 50)[::-1]])
def test_straight_field_line_angle_needs_one_increasing_period(theta):
    with pytest.raises(ValueError):
        straight_field_line_angle(theta, np.ones_like(theta), np.ones_like(theta))


# --- model topology -------------------------------------------------------------


@pytest.mark.parametrize("m, n", [(2, 1), (3, 2), (1, 1), (5, 3)])
@pytest.mark.parametrize("phase", [0.0, 0.7])
def test_a_fixed_phi_section_has_m_o_and_m_x_points_half_a_period_apart(m, n, phase):
    model = _model(m=m, n=n, phase=phase)
    for phi in (0.0, 1.1):
        o, x = model.o_theta(phi), model.x_theta(phi)
        assert len(o) == len(x) == m
        assert len(np.unique(np.round(_wrap(o), 9))) == m
        assert np.allclose(_wrap(model.xi(o, phi)), 0.0, atol=1e-5)
        assert np.allclose(np.abs(_wrap(model.xi(x, phi))), np.pi, atol=1e-5)
        # O and X alternate poloidally, a half period 2 pi / (2 m) apart in theta*
        star = model.theta_star(model.r_s, x) - model.theta_star(model.r_s, o)
        assert np.allclose(np.exp(1j * star), np.exp(1j * np.pi / m), atol=1e-5)


def test_the_helices_close_after_m_toroidal_turns_and_stay_on_the_rational_surface():
    model = _model(m=3, n=2)
    for kind, xi0 in (("o", 0.0), ("x", np.pi)):
        theta, phi = model.locus(kind, np.linspace(0, 2 * np.pi * model.m, 50))
        assert np.allclose(_wrap(model.xi(theta, phi) - xi0), 0.0, atol=1e-5)
        assert math.isclose(math.cos(theta[0]), math.cos(theta[-1]), abs_tol=1e-9)


@pytest.mark.parametrize("kw, match", [
    ({"m": 0}, "positive integer"),
    ({"n": 1.5}, "positive integer"),
    ({"m": True}, "positive integer"),
    ({"m": 4, "n": 2}, "lowest terms"),
    ({"projection": "side"}, "projection"),
    ({"r_s": 1.2}, "r_s"),
    ({"width": 0.0}, "width"),
    ({"width": 0.8}, "width"),
    ({"phase": float("nan")}, "phase"),
    ({"aspect_ratio": 0.9}, "aspect_ratio"),
    ({"elongation": -1.0}, "elongation"),
    ({"triangularity": 1.0}, "triangularity"),
    ({"triangularity": -1.2}, "triangularity"),
])
def test_invalid_parameters_fail_explicitly(kw, match):
    with pytest.raises(ValueError, match=match):
        vaft.diagram.magnetic_island(**kw)


# --- projections -----------------------------------------------------------------


def _markers(diagram, role):
    return [item.at for item in diagram.scene.role(role) if isinstance(item, Marker)]


def test_every_projection_is_drawn_from_one_model():
    diagrams = [vaft.diagram.magnetic_island(projection=p, phase=0.4) for p in PROJECTIONS]
    assert len({d.model for d in diagrams}) == 1
    assert diagrams[0].model == _model(phase=0.4)


@pytest.mark.parametrize("phase", [0.0, 0.9, -2.0])
def test_the_poloidal_markers_sit_on_the_rational_surface_at_the_helical_nodes(phase):
    d = vaft.diagram.magnetic_island(phase=phase, projection="poloidal")
    model, S = d.model, mi.POLOIDAL_SCALE
    for role, expected in (("o_point", model.o_theta(0.0)), ("x_point", model.x_theta(0.0))):
        pts = np.asarray(_markers(d, role)) / S
        assert np.allclose(np.hypot(*pts.T), model.r_s)
        got = np.sort(_wrap(np.arctan2(pts[:, 1], pts[:, 0])))
        assert np.allclose(got, np.sort(_wrap(expected)))


def test_the_poloidal_separatrix_reproduces_the_requested_width_at_the_o_point():
    width = 0.2
    d = vaft.diagram.magnetic_island(width=width, projection="poloidal", phase=0.3)
    model, S = d.model, mi.POLOIDAL_SCALE
    lobe = np.asarray(d.scene.role("separatrix")[0].points) / S
    r = np.hypot(*lobe.T)
    theta = np.arctan2(lobe[:, 1], lobe[:, 0])
    at_o = np.abs(_wrap(model.xi(theta, 0.0, r=r))) < 1e-4
    assert at_o.sum() >= 2
    assert r[at_o].max() - r[at_o].min() == pytest.approx(width, rel=1e-6)
    # the arrow spans the island in the flux label (circles here, so r = |point|); with a
    # phase it follows the xi = 0 line, which bends slightly in theta between surfaces
    (arrow,) = [item for item in d.scene.role("width") if not hasattr(item, "text")]
    assert (np.hypot(*arrow.end) - np.hypot(*arrow.start)) / S == pytest.approx(width)


def test_island_contours_are_closed_level_sets_inside_the_separatrix():
    d = vaft.diagram.magnetic_island(projection="poloidal")
    model, S = d.model, mi.POLOIDAL_SCALE
    for curve in d.scene.role("island"):
        pts = np.asarray(curve.points) / S
        r, theta = np.hypot(*pts.T), np.arctan2(pts[:, 1], pts[:, 0])
        H = island_pendulum_hamiltonian(r - model.r_s, model.xi(theta, 0.0, r=r), model.width)
        assert np.ptp(H) < 1e-4 * model.width ** 2
        assert model.o_level < H.mean() < model.separatrix_level


@pytest.mark.parametrize("shape", [{}, D_SHAPE], ids=["circle", "d_shape"])
@pytest.mark.parametrize("phase", [0.0, 1.3])
def test_top_and_3d_views_agree_with_the_poloidal_section(phase, shape):
    model = _model(phase=phase, **shape)
    top = vaft.diagram.magnetic_island(projection="top", phase=phase, **shape)
    R0, S = model.major_radius, mi.TOP_SCALE

    def major_radius_at(theta):
        return R0 + model.section(model.r_s, theta)[..., 0]

    # phi = 0 markers: on the x axis at the rational surface's R(theta_O)
    got = np.sort(np.asarray(_markers(top, "o_point"))[:, 0] / S)
    assert np.allclose(np.asarray(_markers(top, "o_point"))[:, 1], 0.0, atol=1e-12)
    assert np.allclose(got, np.sort(major_radius_at(model.o_theta(0.0))))
    # every locus point lies within the rational-surface bounds, at R0 + r_s cos(theta_O(phi))
    (locus,) = [item for item in top.scene.role("o_locus") if isinstance(item, Polyline)]
    xy = np.asarray(locus.points) / S
    R, phi = np.hypot(*xy.T), np.arctan2(xy[:, 1], xy[:, 0])
    assert R.max() <= R0 + model.r_s + 1e-9 and R.min() >= R0 - model.r_s - 1e-9
    branches = major_radius_at(np.array([model.theta_at(0.0, phi, k) for k in range(model.m)]))
    assert np.allclose(np.min(np.abs(branches - R), axis=0), 0.0, atol=1e-9)

    d3 = vaft.diagram.magnetic_island(projection="3d", phase=phase, **shape)
    expected = mi._project(model.cartesian(model.r_s, model.o_theta(0.0), 0.0))
    assert np.allclose(np.asarray(_markers(d3, "o_point")), expected)


def test_a_phase_shift_rotates_every_projection_consistently():
    a, b = _model(phase=0.0), _model(phase=0.8)
    # the section rotates by delta/m in theta*, the helix is the same helix shifted in phi by delta/n
    star = b.theta_star(b.r_s, b.o_theta(0.0)) - a.theta_star(a.r_s, a.o_theta(0.0))
    assert np.allclose(_wrap(star), 0.8 / 2, atol=1e-5)
    assert np.allclose(_wrap(b.o_theta(0.0) - a.o_theta(0.8 / 1)), 0.0, atol=1e-9)


def test_the_toggles_remove_their_elements():
    d = vaft.diagram.magnetic_island(show_separatrix=False, show_o_points=False, show_x_points=False,
                                     show_rational_surface=False, labels=False)
    for role in ("separatrix", "o_point", "x_point", "rational", "title"):
        assert not d.scene.role(role)


# --- plasma shaping (Miller, delta(r) = delta * r) --------------------------------


def test_default_shaping_is_a_circle():
    model = _model()
    theta = np.linspace(0, 2 * np.pi, 13)
    assert np.allclose(model.section(0.7, theta), np.stack([0.7 * np.cos(theta), 0.7 * np.sin(theta)], axis=-1))


def test_shaped_surfaces_follow_miller_with_triangularity_growing_linearly():
    model = _model(**D_SHAPE)
    theta = np.linspace(0, 2 * np.pi, 13)
    for r in (0.3, 1.0):
        expected_R = r * np.cos(theta + np.arcsin(0.4 * r) * np.sin(theta))
        assert np.allclose(model.section(r, theta), np.stack([expected_R, 1.7 * r * np.sin(theta)], axis=-1))
    # the top of each surface sits at R - R0 = -delta(r) r: the D leans inward
    assert model.section(1.0, np.pi / 2)[0] == pytest.approx(-0.4)
    assert model.section(0.5, np.pi / 2)[0] == pytest.approx(-0.5 * 0.2)


def test_shaped_surfaces_are_nested_and_contains_agrees_with_the_embedding():
    model = _model(**D_SHAPE)
    theta = np.linspace(0, 2 * np.pi, 181)
    for inner, outer in ((0.2, 0.4), (0.5, 0.64), (0.8, 1.0)):
        pts = model.section(inner, theta)
        assert model.contains(outer, model.major_radius + pts[:, 0], pts[:, 1]).all()
        pts = model.section(outer, theta)
        assert not model.contains(inner, model.major_radius + pts[:, 0], pts[:, 1]).any()


def test_shaped_poloidal_markers_and_width_sit_where_the_model_puts_them():
    d = vaft.diagram.magnetic_island(m=3, n=2, width=0.16, r_s=0.55, projection="poloidal", **D_SHAPE)
    model, S = d.model, mi.POLOIDAL_SCALE
    got = np.asarray(_markers(d, "o_point")) / S
    assert np.allclose(got, model.section(model.r_s, model.o_theta(0.0)))
    got = np.asarray(_markers(d, "x_point")) / S
    assert np.allclose(got, model.section(model.r_s, model.x_theta(0.0)))
    # phase 0 puts an O-point on the outboard midplane, where the label width is physical
    (arrow,) = [item for item in d.scene.role("width") if not hasattr(item, "text")]
    assert np.hypot(*np.subtract(arrow.end, arrow.start)) / S == pytest.approx(0.16)
    assert arrow.start[1] == pytest.approx(0.0, abs=1e-12)


def test_shaped_island_contours_are_level_sets_in_the_label_coordinates():
    d = vaft.diagram.magnetic_island(projection="poloidal", **D_SHAPE)
    model, S = d.model, mi.POLOIDAL_SCALE
    curve = d.scene.role("island")[0]
    pts = np.asarray(curve.points) / S
    # every contour point lies on some surface r and some angle theta the model maps back to it
    r = np.linspace(model.r_s - model.width, model.r_s + model.width, 801)
    theta = np.linspace(-np.pi, np.pi, 1441)
    grid = model.section(r[:, None], theta[None, :])
    for p in pts[::15]:
        dist = np.hypot(*(grid - p).transpose(2, 0, 1))
        i, j = np.unravel_index(np.argmin(dist), dist.shape)
        H = island_pendulum_hamiltonian(r[i] - model.r_s, model.xi(theta[j], 0.0, r=r[i]), model.width)
        assert model.o_level < H < model.separatrix_level


def test_the_3d_view_hides_the_far_side_of_a_shaped_torus():
    model = _model(**D_SHAPE)
    phi_view = mi.CAMERA_AZIMUTH
    near = mi._visible(model, model.r_s, np.array([0.0]), np.array([phi_view]))
    far = mi._visible(model, model.r_s, np.array([np.pi]), np.array([phi_view]))
    behind = mi._visible(model, model.r_s, np.array([0.0]), np.array([phi_view + np.pi]))
    assert near[0] and not far[0] and not behind[0]


# --- straight-field-line angle in the model ------------------------------------


@pytest.mark.parametrize("shape", [{}, D_SHAPE, {"aspect_ratio": 1.3}], ids=["circle", "d_shape", "low_aspect"])
def test_o_points_are_evenly_spaced_in_theta_star_and_spread_on_the_low_field_side(shape):
    model = _model(m=3, n=2, r_s=0.55, width=0.16, **shape)
    o = model.o_theta(0.0)
    star = np.sort(np.mod(model.theta_star(model.r_s, o), 2 * np.pi))
    assert np.allclose(np.diff(np.r_[star, star[0] + 2 * np.pi]), 2 * np.pi / 3, atol=1e-5)
    # in the parametrisation angle the two off-midplane O-points sit beyond 120 deg: the
    # outboard (low-field) gap is wider than the inboard one
    geometric = np.sort(np.mod(o, 2 * np.pi))
    assert geometric[0] == pytest.approx(0.0, abs=1e-9)
    assert geometric[1] > 2 * np.pi / 3 + np.radians(5)


def test_theta_star_matches_the_formula_on_the_model_surface():
    model = _model(**D_SHAPE)
    theta = np.linspace(0.0, 2 * np.pi, 2001)
    R = model.major_radius + model.section(0.6, theta)[:, 0]
    reference = straight_field_line_angle(theta, model.jacobian(0.6, theta), R)
    assert np.allclose(model.theta_star(0.6, theta[:-1]), reference[:-1], atol=1e-5)
    assert np.allclose(model.theta_from_star(0.6, reference[:-1]), theta[:-1], atol=1e-5)


def test_the_width_arrow_follows_the_o_point_helical_line():
    d = vaft.diagram.magnetic_island(m=3, n=2, width=0.16, r_s=0.55, phase=1.0, projection="poloidal", **D_SHAPE)
    model, S = d.model, mi.POLOIDAL_SCALE
    (arrow,) = [item for item in d.scene.role("width") if not hasattr(item, "text")]
    for end, r in ((arrow.start, model.r_s - 0.08), (arrow.end, model.r_s + 0.08)):
        theta = model.theta_at(0.0, 0.0, 0, r=r)
        assert np.allclose(np.asarray(end) / S, model.section(r, theta), atol=1e-9)
        assert _wrap(model.xi(theta, 0.0, r=r)) == pytest.approx(0.0, abs=1e-5)
