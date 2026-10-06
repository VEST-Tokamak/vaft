"""Geometric-approximation diagrams: drawn from vaft.formula.geometry, one concept each (#1062)."""

import numpy as np
import pytest

import vaft.diagram
from vaft.formula.geometry import (
    cylindrical_parallel_wavenumber,
    local_slab_from_cylinder,
    sheared_slab_field,
)


def test_the_map_keeps_geometry_and_ordering_on_separate_axes():
    d = vaft.diagram.geometry_ordering_map()
    columns, boxes = d.model["columns"], d.model["boxes"]
    # every representation sits in its geometry's column
    for name, geometry in (("general_torus", "toroidal"), ("lar_torus", "toroidal"),
                           ("cyl_tokamak", "cylindrical"), ("screw_pinch", "cylindrical"),
                           ("straight_slab", "slab"), ("sheared_slab", "slab"), ("curved_slab", "slab")):
        assert boxes[name][0] == pytest.approx(columns[geometry]), name
    # and in its ordering band
    def band(role):
        (strip,) = [it for it in d.scene.role(role) if getattr(it, "closed", False)]
        ys = np.asarray(strip.points)[:, 1]
        return ys.min(), ys.max()
    for names, role in ((("general_torus", "screw_pinch"), "ordering:global"),
                        (("lar_torus", "cyl_tokamak"), "ordering:large_aspect_ratio"),
                        (("straight_slab", "sheared_slab", "curved_slab"), "ordering:local")):
        y0, y1 = band(role)
        for name in names:
            assert y0 < boxes[name][1] < y1, (name, role)
    # a local slab comes from the general torus directly -- no cylinder, no large aspect ratio --
    # and the cylinder is the O(1) limit of the large-aspect-ratio torus
    assert ("general_torus", "sheared_slab") in d.model["paths"]
    assert ("general_torus", "curved_slab") in d.model["paths"]
    assert ("lar_torus", "cyl_tokamak") in d.model["paths"]
    assert not any(end.endswith("slab") and start == "lar_torus" for start, end in d.model["paths"])
    for start, end in d.model["paths"]:
        labels = [it for it in d.scene.role(f"path:{start}->{end}") if hasattr(it, "text")]
        assert labels and labels[0].text, (start, end)  # every reduction names what it keeps


@pytest.mark.parametrize("geometry", ["toroidal", "cylindrical"])
def test_the_torus_and_cylinder_share_one_field_line_pitch(geometry):
    m = vaft.diagram.field_line_geometry(geometry).model
    line = m["field_line"]
    if geometry == "toroidal":
        phi = np.unwrap(np.arctan2(line[:, 1], line[:, 0]))
        R = np.hypot(line[:, 0], line[:, 1])
        theta = np.unwrap(np.arctan2(line[:, 2], R - 2.2))
        pitch = (theta[-1] - theta[0]) / (phi[-1] - phi[0])
    else:
        theta = np.unwrap(np.arctan2(line[:, 2], line[:, 1]))
        pitch = (theta[-1] - theta[0]) / ((line[-1, 0] - line[0, 0]) / 2.2)  # d theta / d(z/R0)
    assert pitch == pytest.approx(m["pitch"], rel=1e-3)
    assert m["pitch"] == pytest.approx(vaft.diagram.field_line_geometry("toroidal").model["pitch"])
    assert m["pitch"] != pytest.approx(1.0)  # q != 1, so a q-for-1/q slip cannot pass


def test_the_slab_field_lines_follow_the_sheared_field():
    m = vaft.diagram.field_line_geometry("slab").model
    tilts = m["pitch"]
    xs = sorted(tilts)
    assert tilts[0.0] == 0.0 and tilts[xs[0]] * tilts[xs[-1]] < 0  # the tilt changes sign across x = 0
    for x, tilt in tilts.items():
        B = sheared_slab_field(x, 1.0, -3.0)
        assert tilt == pytest.approx(B[1] / B[2])


@pytest.mark.parametrize("m, n", [(2, 1), (3, 2), (5, 2)])
def test_the_mode_mapping_crosses_zero_at_the_rational_surface_with_the_slab_tangent(m, n):
    chart = vaft.diagram.mode_number_mapping(m, n).model
    p = chart.parameters
    r, k = chart.curves["k_par"].T
    assert np.count_nonzero(np.diff(k > 0)) == 1
    assert np.interp(p["r_s"], r, k) == pytest.approx(0.0, abs=1e-3 * np.max(np.abs(k)))
    assert cylindrical_parallel_wavenumber(m, n, m / n, p["R0"]) == 0.0
    k_y, L_s, k_z = local_slab_from_cylinder(m, n, p["r_s"], p["R0"], m / n, p["s_hat"])
    assert (p["k_y"], p["k_z"], p["L_s"]) == (k_y, k_z, L_s) and k_z == 0.0
    # the tangent's slope is the local slab's k_y / L_s, and the cylinder's slope at r_s
    t = chart.curves["local_slab"]
    slope = (t[-1, 1] - t[0, 1]) / (t[-1, 0] - t[0, 0])
    assert slope == pytest.approx(k_y / L_s)
    i = np.argmin(np.abs(r - p["r_s"]))
    assert slope == pytest.approx(np.gradient(k, r)[i], rel=0.02)


@pytest.mark.parametrize("m, n", [(1, 1), (1, 3), (1, 10), (2, 7), (12, 1)])
def test_the_slab_tangent_stays_on_the_chart_and_is_labelled(m, n):
    d = vaft.diagram.mode_number_mapping(m, n)
    chart = d.model
    t = chart.curves["local_slab"]
    assert t[:, 0].min() > 0.0 and t[:, 0].max() <= chart.x_range[1]
    assert np.all(np.abs(t[:, 1]) <= chart.y_range[1])
    assert [it for it in d.scene.role("local_slab") if hasattr(it, "text")]


@pytest.mark.parametrize("fn, kw", [
    (vaft.diagram.field_line_geometry, {"geometry": "helical"}),
    (vaft.diagram.mode_number_mapping, {"m": 0}),
    (vaft.diagram.mode_number_mapping, {"n": 1.5}),
    (vaft.diagram.geometry_ordering_map, {"labels": "yes"}),
])
def test_bad_arguments_fail(fn, kw):
    with pytest.raises(ValueError):
        fn(**kw)


@pytest.mark.parametrize("name", ["geometry_ordering_map", "field_line_geometry", "mode_number_mapping",
                                  "mhd_mode_geometry_map"])
def test_every_geometry_diagram_is_deterministic_and_exported(name):
    fn = getattr(vaft.diagram, name)
    assert fn().tikz == fn().tikz
    assert name in vaft.diagram.__all__
    assert fn().scene.role("note") and not fn(labels=False).scene.role("note")
