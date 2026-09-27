"""Tokamak geometry diagrams (#1073): every surface, shift, field and angle is the formula's."""

import math

import numpy as np
import pytest

import vaft.diagram
from vaft.diagram._projection import camera
from vaft.formula.equilibrium import shafranov_shift_from_r_a_R0_beta_p_li, vacuum_toroidal_field


# --- the Shafranov-shift formula ----------------------------------------------------------


def test_the_shift_is_largest_on_axis_and_zero_at_the_edge():
    a, R0, bp, li = 0.5, 1.5, 0.6, 0.9
    assert shafranov_shift_from_r_a_R0_beta_p_li(a, a, R0, bp, li) == 0.0
    assert shafranov_shift_from_r_a_R0_beta_p_li(0.0, a, R0, bp, li) == pytest.approx(a * a / (2 * R0) * (bp + li / 2))
    r = np.linspace(0, a, 11)
    d = shafranov_shift_from_r_a_R0_beta_p_li(r, a, R0, bp, li)
    assert np.all(np.diff(d) < 0)
    # d Delta / dr = -(r / R0)(beta_p + l_i / 2), Wesson's large-aspect-ratio result
    np.testing.assert_allclose(np.gradient(d, r)[1:-1], -(r / R0 * (bp + li / 2))[1:-1], rtol=1e-6)
    with pytest.raises(ValueError):
        shafranov_shift_from_r_a_R0_beta_p_li(0.6, a, R0, bp, li)
    with pytest.raises(ValueError):
        shafranov_shift_from_r_a_R0_beta_p_li(0.1, 0.0, R0, bp, li)


# --- surfaces ---------------------------------------------------------------------------------


def test_concentric_surfaces_share_the_axis_and_shifted_ones_move_out():
    circ = vaft.diagram.flux_surfaces("circular")
    shift = vaft.diagram.flux_surfaces("shifted")
    assert np.all(circ.model["shifts"] == 0.0) and circ.model["axis_shift"] == 0.0
    m = shift.model
    np.testing.assert_allclose(m["shifts"], shafranov_shift_from_r_a_R0_beta_p_li(m["radii"], m["a"], m["R0"],
                                                                                  m["beta_p"], m["l_i"]))
    assert m["shifts"][-1] == 0.0 and m["axis_shift"] > m["shifts"][0] > 0.0
    # every drawn surface is centred where its shift says
    for surface, r, d in zip([it for it in shift.scene.role("flux_surface")], m["radii"], m["shifts"]):
        xs = np.asarray(surface.points)[:, 0]
        assert 0.5 * (xs.min() + xs.max()) == pytest.approx(d * 2.4, abs=1e-6)
        assert 0.5 * (xs.max() - xs.min()) == pytest.approx(r * 2.4, rel=1e-3)
    (axis,) = shift.scene.role("magnetic_axis")[:1]
    (geo,) = shift.scene.role("geometric_axis")[:1]
    assert axis.at[0] > geo.at[0]  # outward


def test_the_shaping_family_has_the_named_elongation_and_triangularity():
    d = vaft.diagram.shaping_family()
    for name, kappa, delta in d.model["family"]:
        b = d.model["boundaries"][name]
        width = np.ptp(b[:, 0])
        height = np.ptp(b[:, 1])
        assert height / width == pytest.approx(kappa, rel=0.02), name
        top = b[np.argmax(b[:, 1])]
        centre_x = 0.5 * (b[:, 0].min() + b[:, 0].max())
        # the top is pulled in by delta * a (in the drawing's scale)
        assert (centre_x - top[0]) / (0.5 * width) == pytest.approx(delta, abs=0.02), name


def test_the_hfs_field_is_stronger_by_the_1_over_r_ratio():
    chart = vaft.diagram.hfs_lfs_field().model
    p = chart.parameters
    assert p["B_hfs"] == pytest.approx(vacuum_toroidal_field(1.0, 3.0, 2.0))
    assert p["B_lfs"] == pytest.approx(vacuum_toroidal_field(1.0, 3.0, 4.0))
    assert p["B_hfs"] / p["B_lfs"] == pytest.approx(4.0 / 2.0)
    R, B = chart.curves["B_phi"].T
    assert np.all(np.diff(B) < 0)


# --- field lines and angles ----------------------------------------------------------------------


@pytest.mark.parametrize("q", [1, 2, 3, 5])
def test_the_winding_line_makes_q_toroidal_turns_per_poloidal_turn(q):
    m = vaft.diagram.safety_factor_winding(q).model
    line = m["field_line"]
    phi = np.unwrap(np.arctan2(line[:, 1], line[:, 0]))
    R = np.hypot(line[:, 0], line[:, 1])
    theta = np.unwrap(np.arctan2(line[:, 2], R - 3.0))
    assert (phi[-1] - phi[0]) / (2 * math.pi) == pytest.approx(q)
    assert (theta[-1] - theta[0]) / (2 * math.pi) == pytest.approx(1.0)
    # the q crossings lie on one cross-section, and the last returns to the first
    c = m["crossings"]
    np.testing.assert_allclose(np.arctan2(c[:, 1], c[:, 0]), math.atan2(math.sin(m["phi_cut"]), math.cos(m["phi_cut"])))
    np.testing.assert_allclose(c[-1], c[0], atol=1e-12)
    markers = [it for it in vaft.diagram.safety_factor_winding(q).scene.role("crossing") if hasattr(it, "kind")]
    assert len(markers) == q


def test_the_3d_cross_section_faces_the_camera():
    from vaft.diagram._tokamak_geometry import _camera_cut

    view, _, _ = camera()
    phi = _camera_cut()
    phi_hat = np.array([-math.sin(phi), math.cos(phi), 0.0])
    assert abs(phi_hat @ view) > 0.8  # the poloidal plane's normal points at the viewer, not across the line of sight
    assert vaft.diagram.safety_factor_winding(3).model["phi_cut"] == phi
    # and the torus view's filled cut is that plane
    (cut,) = vaft.diagram.tokamak_torus("3d").scene.role("cross_section")
    assert np.ptp(np.asarray(cut.points)[:, 0]) > 1.0  # seen face-on, not edge-on


def test_theta_star_lines_differ_from_geometric_rays_on_shaped_surfaces():
    d = vaft.diagram.poloidal_angle_comparison()
    lines = d.model["theta_star_lines"]
    island = d.model["island_model"]
    bent = 0
    for k, pts in lines.items():
        polar = np.unwrap(np.arctan2(pts[:, 1], pts[:, 0]))
        if np.ptp(polar[5:]) > 0.02:  # the polar angle changes along the line: it is not a ray
            bent += 1
    assert bent >= 6  # a straight-field-line coordinate line is not a ray on a D shape
    # theta* lines crowd towards the inboard side: most end on the inboard half of the boundary
    ends = np.array([pts[-1] for pts in lines.values()])
    assert np.sum(ends[:, 0] < 0) > len(ends) / 2
    # the dashed rays are equal steps of the geometric (polar) angle from the axis
    rays = [np.asarray(it.points) for it in d.scene.role("geometric_angle") if hasattr(it, "points")]
    polar = np.sort(np.mod([np.arctan2(r[-1, 1], r[-1, 0]) for r in rays], 2 * np.pi))
    np.testing.assert_allclose(np.diff(polar), 2 * np.pi / 12, atol=0.03)
    # and theta_star is the island model's, which is straight_field_line_angle's
    for r in (0.3, 0.8):
        t = np.linspace(0, 2 * np.pi, 7)
        np.testing.assert_allclose(island.theta_star(r, island.theta_from_star(r, t)), t, atol=2e-3)


@pytest.mark.parametrize("q", [1.0, 2.5, 4.0])
def test_the_unwrapped_field_line_is_straight_with_slope_q(q):
    chart = vaft.diagram.unwrapped_flux_surface(q).model
    for name, pts in chart.curves.items():
        if name.startswith("straight"):
            np.testing.assert_allclose(np.diff(pts[:, 1]) / np.diff(pts[:, 0]), q)
    dashed = [(name, pts) for name, pts in chart.curves.items() if name.startswith("parametrisation")]
    slopes = np.concatenate([np.diff(p[:, 1]) / np.diff(p[:, 0]) for _, p in dashed])
    assert np.ptp(slopes) > 0.1 * q  # not straight against the parametrisation angle
    # and the dashed curve is the same field line: its theta* is the solid line's
    from vaft.diagram._tokamak_geometry import _shaped_model

    island = _shaped_model()
    for name, pts in dashed:
        solid = chart.curves[name.replace("parametrisation", "straight")]
        np.testing.assert_allclose(island.theta_star(0.55, pts[:, 0]), solid[:, 0], atol=2e-3)


# --- common --------------------------------------------------------------------------------------

@pytest.mark.parametrize("q", [0.5, 1.0, 3.0])
def test_the_pitch_components_sum_to_the_field_line_direction(q):
    m = vaft.diagram.field_line_pitch(q).model
    B = m["B"] / np.linalg.norm(m["B"])
    assert abs(B @ m["tangent"]) == pytest.approx(1.0, abs=1e-4)


BUILDERS = [
    ("field_line_pitch", {}),
    ("tokamak_torus", {"projection": "3d"}), ("tokamak_torus", {"projection": "poloidal"}),
    ("flux_surfaces", {"shape": "circular"}), ("flux_surfaces", {"shape": "shifted"}),
    ("shaping_family", {}), ("hfs_lfs_field", {}), ("safety_factor_winding", {}), ("flux_coordinates", {}),
    ("poloidal_angle_comparison", {}), ("unwrapped_flux_surface", {}),
]


@pytest.mark.parametrize("name, kw", BUILDERS)
def test_every_tokamak_geometry_diagram_is_deterministic_and_exported(name, kw):
    fn = getattr(vaft.diagram, name)
    assert fn(**kw).tikz == fn(**kw).tikz
    assert name in vaft.diagram.__all__
    assert fn(**kw).scene.role("note") and not fn(**kw, labels=False).scene.role("note")


@pytest.mark.parametrize("fn, kw", [
    (vaft.diagram.tokamak_torus, {"projection": "top"}), (vaft.diagram.flux_surfaces, {"shape": "d"}),
    (vaft.diagram.safety_factor_winding, {"q": 0}), (vaft.diagram.safety_factor_winding, {"q": 2.5}),
    (vaft.diagram.unwrapped_flux_surface, {"q": 0.1}), (vaft.diagram.unwrapped_flux_surface, {"q": "x"}),
    (vaft.diagram.field_line_pitch, {"q": 9}),
])
def test_bad_arguments_fail(fn, kw):
    with pytest.raises(ValueError):
        fn(**kw)
