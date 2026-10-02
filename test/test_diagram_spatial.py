"""The spatial vocabulary (#1101): coordinates, geometry, meshes, mappings and topology."""

import math

import numpy as np
import pytest

import vaft.diagram
from vaft.data.cocos import cocos_spec
from vaft.diagram import _spatial as S
from vaft.diagram._gs_equilibrium import _encloses, flux_model, lcfs

NEW = ("tokamak_top_view", "cocos_orientation", "machine_and_equilibrium_geometry", "structured_rz_grid",
       "geometry_to_mesh", "logical_to_physical_mapping", "physical_to_flux_mapping")


@pytest.mark.parametrize("name", NEW)
def test_every_spatial_diagram_is_deterministic_and_exposed(name):
    fn = getattr(vaft.diagram, name)
    assert name in vaft.diagram.__all__
    assert fn().tikz == fn().tikz
    assert fn(labels=False).tikz != fn().tikz


def test_every_family_entry_is_a_builder():
    assert set(S.SPATIAL_FAMILIES) == {"coordinate", "geometry", "mesh", "mapping", "topology"}
    for names in S.SPATIAL_FAMILIES.values():
        for name in names:
            assert name in vaft.diagram.__all__, name
    filed = {n for names in S.SPATIAL_FAMILIES.values() for n in names}
    assert set(NEW) <= filed


# Sauter & Medvedev 2013, Table I: theta seen from the front (R right, Z up) and phi seen from above
THETA_CCW = {1: False, 2: True, 3: True, 4: False, 5: True, 6: False, 7: False, 8: True}


@pytest.mark.parametrize("index", sorted(THETA_CCW))
@pytest.mark.parametrize("offset", [0, 10])
def test_cocos_orientation_matches_the_sauter_table(index, offset):
    s = S.cocos_orientation_signs(index + offset)
    assert s["theta_counterclockwise"] == THETA_CCW[index]
    assert s["phi_counterclockwise_from_above"] == (index in (1, 3, 5, 7))
    assert s["phi_out_of_page"] == (not s["phi_counterclockwise_from_above"])
    assert s["psi_per_radian"] == (offset == 0)


@pytest.mark.parametrize("sigma_ip, sigma_b0", [(1, 1), (-1, 1), (1, -1), (-1, -1)])
def test_cocos_current_and_field_follow_phi(sigma_ip, sigma_b0):
    s = S.cocos_orientation_signs(11, sigma_ip=sigma_ip, sigma_b0=sigma_b0)
    assert s["ip_out_of_page"] == (s["phi_out_of_page"] == (sigma_ip == 1))
    assert s["b0_out_of_page"] == (s["phi_out_of_page"] == (sigma_b0 == 1))
    spec = cocos_spec(11)
    assert s["dpsi"] == sigma_ip * spec.sigma_bp
    assert s["q"] == sigma_ip * sigma_b0 * spec.sigma_rhotp


def test_cocos_drawing_follows_the_signs():
    d = vaft.diagram.cocos_orientation(2)
    (s,) = d.model["panels"]
    arc = d.scene.role("poloidal_angle")[0].points
    (ux, uy), (vx, vy) = np.subtract(arc[1], arc[0]), np.subtract(arc[-1], arc[0])
    assert (ux * vy - uy * vx > 0) == s["theta_counterclockwise"]
    symbols = [it.text for it in d.scene.role("phi") if it.style == "legend symbol"]
    assert symbols == (["$\\odot$"] if s["phi_out_of_page"] else ["$\\otimes$"])
    with pytest.raises(ValueError):
        S.cocos_orientation_signs(11, sigma_ip=0)


@pytest.mark.parametrize("cocos, sigma_ip", [(11, 1), (11, -1), (2, 1), (13, 1)])
def test_the_psi_arrow_points_where_psi_increases(cocos, sigma_ip):
    d = vaft.diagram.cocos_orientation(cocos, sigma_ip=sigma_ip)
    (s,) = d.model["panels"]
    (arrow,) = [it for it in d.scene.role("psi_gradient") if hasattr(it, "start")]
    centre = d.scene.role("axis")[0].at
    outward = np.hypot(*np.subtract(arrow.end, centre)) > np.hypot(*np.subtract(arrow.start, centre))
    assert outward == (s["dpsi"] > 0)
    (unit,) = [it.text for it in d.scene.role("psi_gradient") if hasattr(it, "text")]
    assert ("Wb/rad" in unit) == (cocos < 10)


def test_the_current_marker_sits_on_the_axis_and_the_title_names_the_index():
    d = vaft.diagram.cocos_orientation(12)
    (glyph,) = d.scene.role("axis")
    (s,) = d.model["panels"]
    assert glyph.style == "axis current" and glyph.text == ("$\\odot$" if s["ip_out_of_page"] else "$\\otimes$")
    centre = d.scene.role("plasma_boundary")[0].points
    xs = [p[0] for p in centre]
    assert glyph.at[0] == pytest.approx(0.5 * (min(xs) + max(xs)), abs=0.01)  # sampled circle
    assert [it.text for it in d.scene.role("title")] == ["COCOS = 12"]
    assert "(+1, -1, +1, \\mathrm{Wb})" in d.scene.role("subtitle")[0].text
    assert not d.scene.role("table")


def test_several_indices_draw_one_panel_each_at_one_scale():
    d = vaft.diagram.cocos_orientation(range(1, 9))
    assert [p["cocos"] for p in d.model["panels"]] == list(range(1, 9))
    boundaries = [np.array(it.points) for it in d.scene.role("plasma_boundary")]
    assert len(boundaries) == 8
    assert len({round(float(np.ptp(b[:, 0])), 9) for b in boundaries}) == 1
    assert len([it for it in d.scene.role("title")]) == 8
    with pytest.raises(ValueError):
        vaft.diagram.cocos_orientation([])


@pytest.mark.parametrize("cocos", [11, 12])
def test_top_view_phi_sense_follows_cocos(cocos):
    d = vaft.diagram.tokamak_top_view(cocos=cocos)
    assert d.model["phi_counterclockwise_from_above"] == (cocos_spec(cocos).sigma_rpz == 1)
    assert (d.model["phi_point"] > 0) == d.model["phi_counterclockwise_from_above"]
    assert len([it for it in d.scene.role("tf_coil") if hasattr(it, "points")]) == 12
    with pytest.raises(ValueError):
        vaft.diagram.tokamak_top_view(n_coils=2)


def test_machine_and_equilibrium_geometry_keeps_the_two_apart():
    d = vaft.diagram.machine_and_equilibrium_geometry()
    for role in ("vessel", "limiter", "coil", "passive_structure", "symmetry_axis", "separatrix", "x_point", "axis"):
        assert d.scene.role(role), role
    assert set(d.model["machine"]).isdisjoint(d.model["equilibrium"])
    assert d.model["x_point"] == flux_model("diverted")["x_point"]


def test_structured_grid_nodes_inside_are_inside_the_lcfs():
    d = vaft.diagram.structured_rz_grid(n_r=13, n_z=21)
    assert d.model["n_inside"] == len([it for it in d.scene.role("plasma_node") if hasattr(it, "kind")]) > 0
    assert len(d.scene.role("grid")) == 13 + 21
    assert d.model["dr"] == pytest.approx(1.2 / 12)
    with pytest.raises(ValueError):
        vaft.diagram.structured_rz_grid(n_r=2)


def test_the_unstructured_mesh_is_finer_where_the_spacing_says():
    mesh = S.unstructured_mesh()
    m = mesh["median_edge"]
    assert m["conductor"] < m["vacuum"] and m["plasma"] < m["vacuum"]
    for region, h in S.MESH_SPACING.items():
        assert 0.6 * h < m[region] < 1.5 * h, region
    assert all(mesh["count"][k] > 0 for k in S.MESH_SPACING)
    # triangles are classed by centroid: a plasma triangle's centroid is inside the LCFS
    boundary = lcfs(flux_model("diverted"))
    for t, region in list(zip(mesh["triangles"], mesh["regions"]))[::25]:
        assert (region == "plasma") == _encloses(boundary, mesh["nodes"][t].mean(axis=0))


def test_logical_mapping_is_the_miller_surface_and_singular_on_the_axis():
    R, Z = S.logical_to_physical(np.zeros(5), np.linspace(0, 1, 5))
    assert np.allclose(R, S._MAP_SHAPE[0]) and np.allclose(Z, 0.0)  # xi = 0 is one point
    R0, Z0 = S.logical_to_physical(0.5, 0.0)
    R1, Z1 = S.logical_to_physical(0.5, 1.0)
    assert R0 == pytest.approx(R1) and Z0 == pytest.approx(Z1, abs=1e-12)  # periodic in eta
    R, Z = S.logical_to_physical(1.0, 0.25)
    assert Z == pytest.approx(S._MAP_SHAPE[1] * S._MAP_SHAPE[2])  # top of the boundary: Z = kappa a
    with pytest.raises(ValueError):
        vaft.diagram.logical_to_physical_mapping(cell=(9, 0))


def test_physical_to_flux_mapping_collapses_the_two_halves():
    m = vaft.diagram.physical_to_flux_mapping().model
    assert np.all((m["psi_n"] >= 0) & (m["psi_n"] <= 1))
    assert m["inboard"].any() and (~m["inboard"]).any()
    # psi_N rises away from the axis on each side
    assert np.all(np.diff(m["psi_n"][m["inboard"]]) < 0) and np.all(np.diff(m["psi_n"][~m["inboard"]]) > 0)
    # the value is a function of psi_N alone: one curve
    order = np.argsort(m["psi_n"])
    assert np.all(np.diff(m["value"][order]) <= 1e-12)
    assert np.allclose(m["value"], S._profile(m["psi_n"]))


def test_the_mesh_has_no_cocircular_ties():
    """A Delaunay tie is broken differently across platforms, which would make the committed SVG stale."""
    mesh = S.unstructured_mesh()
    nodes, tris = mesh["nodes"], mesh["triangles"]
    from scipy.spatial import Delaunay

    neighbours = Delaunay(nodes).neighbors
    worst = np.inf
    for t, nb in zip(tris, neighbours):
        a, b, c = nodes[t]
        for k in nb[nb >= 0]:
            for d in nodes[tris[k]]:
                if any(np.allclose(d, p) for p in (a, b, c)):
                    continue
                m = np.array([[*(p - d), np.dot(p - d, p - d)] for p in (a, b, c)])
                scale = np.prod([np.dot(p - d, p - d) for p in (a, b, c)]) ** 0.5
                worst = min(worst, abs(np.linalg.det(m)) / scale)
    assert worst > 1e-9
    assert not mesh["nodes"].flags.writeable


@pytest.mark.parametrize("bad", [True, "11", [], [11, True], 11.0])
def test_cocos_arguments_are_validated(bad):
    with pytest.raises(ValueError):
        vaft.diagram.cocos_orientation(bad)


def test_cocos_accepts_numpy_indices():
    assert vaft.diagram.cocos_orientation(np.int64(11)).model["panels"][0]["cocos"] == 11
    assert len(vaft.diagram.cocos_orientation(np.arange(1, 5)).model["panels"]) == 4
    with pytest.raises(ValueError):
        S.cocos_orientation_signs(11, sigma_ip=True)


def test_a_grid_too_coarse_for_the_plasma_fails_explicitly():
    with pytest.raises(ValueError, match="no node inside"):
        vaft.diagram.structured_rz_grid(n_r=4, n_z=4)
    with pytest.raises(ValueError, match="pair"):
        vaft.diagram.logical_to_physical_mapping(cell=(1,))
