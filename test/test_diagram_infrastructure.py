"""Shared diagram infrastructure: formula-owned geometry, the chart and concept primitives."""

import inspect
import shutil

import numpy as np
import pytest

from vaft.diagram import _concept, _magnetic_island, _particle_motion
from vaft.formula.equilibrium import miller_surface, vacuum_toroidal_field

HAS_TEX = all(shutil.which(tool) for tool in ("latex", "dvisvgm"))


# --- formula-owned geometry -------------------------------------------------------


def test_vacuum_toroidal_field_is_b0_at_r0_and_falls_as_one_over_r():
    assert vacuum_toroidal_field(2.0, 1.5, 1.5) == pytest.approx(2.0)
    R = np.array([0.5, 1.0, 3.0])
    assert np.allclose(vacuum_toroidal_field(2.0, 1.5, R) * R, 3.0)
    assert vacuum_toroidal_field(-1.0, 1.0, 2.0) == pytest.approx(-0.5)  # keeps the sign of B0
    with pytest.raises(ValueError):
        vacuum_toroidal_field(1.0, 1.0, 0.0)


def test_miller_surface_reduces_to_a_circle_and_places_the_d_tips():
    theta = np.linspace(0, 2 * np.pi, 9)
    R, Z = miller_surface(0.5, theta, 2.0, 1.0, 0.0)
    assert np.allclose(np.hypot(R - 2.0, Z), 0.5)
    R, Z = miller_surface(0.5, np.pi / 2, 2.0, 1.7, 0.4, shift=0.1)
    assert R == pytest.approx(2.0 + 0.1 - 0.4 * 0.5)  # the top is pulled in by delta r
    assert Z == pytest.approx(1.7 * 0.5)
    for bad in ({"kappa": 0.0, "delta": 0.1}, {"kappa": 1.0, "delta": 1.0}):
        with pytest.raises(ValueError):
            miller_surface(0.5, 0.0, 2.0, **bad)


def test_the_island_surfaces_are_the_formula_surfaces():
    model = _magnetic_island._validate(3, 2, 0.16, 0.0, "poloidal", 0.55, 3.2, 1.7, 0.4)
    theta = np.linspace(0, 2 * np.pi, 13)
    for r in (0.3, 0.55, 1.0):
        R, Z = miller_surface(r, theta, model.major_radius, 1.7, 0.4 * r)
        assert np.allclose(model.section(r, theta), np.stack([R - model.major_radius, Z], axis=-1))


def test_the_particle_diagrams_take_the_toroidal_field_from_the_formula():
    B = _particle_motion._toroidal_field(4.0, 2.5)
    x = np.array([3.0, 4.0, 0.0])
    assert np.linalg.norm(B(x)) == pytest.approx(vacuum_toroidal_field(2.5, 4.0, 5.0))
    grad = _particle_motion._toroidal_field_gradient(4.0, 2.5, 4.0)
    assert grad == pytest.approx(-2.5 / 4.0, rel=1e-8)


def test_no_diagram_module_restates_the_toroidal_field():
    for module in (_particle_motion, _magnetic_island):
        source = inspect.getsource(module)
        assert "B0 * R0 / R" not in source and "B0 / R0" not in source


# --- chart ---------------------------------------------------------------------------


def test_the_chart_helper_is_shared():
    from vaft.diagram import _chart, _stability_space

    assert _stability_space.Chart is _chart.Chart
    assert _chart.nice_ticks(5.0) == [0.0, 1.0, 2.0, 3.0, 4.0]


# --- concept primitives -----------------------------------------------------------------


def test_a_box_is_an_outline_and_a_wrapped_label():
    b = _concept.box(1.0, 2.0, 3.0, 1.0, "Formula", role="formula")
    outline, label = b.items
    xs, ys = np.asarray(outline.points).T
    assert outline.closed and (xs.min(), xs.max(), ys.min(), ys.max()) == (-0.5, 2.5, 1.5, 2.5)
    assert label.at == (1.0, 2.0) and "text width=2.70cm" in label.style
    with pytest.raises(ValueError):
        _concept.box(0, 0, 0.2, 1.0, "too narrow")


def test_a_connector_runs_edge_to_edge_along_the_centre_line():
    a = _concept.box(0.0, 0.0, 2.0, 1.0, "A")
    b = _concept.box(4.0, 0.0, 2.0, 1.0, "B")
    arrow = _concept.connector(a, b, gap=0.0)
    assert arrow.start == pytest.approx((1.0, 0.0)) and arrow.end == pytest.approx((3.0, 0.0))
    c = _concept.box(0.0, 3.0, 2.0, 1.0, "C")
    arrow = _concept.connector(a, c, gap=0.0)
    assert arrow.start == pytest.approx((0.0, 0.5)) and arrow.end == pytest.approx((0.0, 2.5))
    with pytest.raises(ValueError):
        _concept.connector(a, _concept.box(0.5, 0.0, 2.0, 1.0, "overlap"))


def test_a_band_is_a_background_strip_with_an_optional_label():
    items = _concept.band(0, 5, 0, 2, "stage 1", role="stage")
    assert len(items) == 2 and items[0].style == "concept band"
    assert len(_concept.band(0, 5, 0, 2)) == 1
    with pytest.raises(ValueError):
        _concept.band(1, 0, 0, 2)


@pytest.mark.skipif(not HAS_TEX, reason="latex and dvisvgm are not installed")
def test_a_concept_scene_renders():
    from vaft.diagram._render import Diagram
    from vaft.diagram._scene import Scene

    a = _concept.box(0.0, 0.0, 3.0, 1.2, "vaft.formula")
    b = _concept.box(5.0, 0.0, 3.0, 1.2, "vaft.diagram")
    scene = Scene(tuple(_concept.band(-2.0, 7.0, -1.2, 1.4, "layers")) + a.items + b.items
                  + (_concept.connector(a, b),))
    svg = Diagram("concept_demo", scene).svg
    assert "<svg" in svg and "<image" not in svg
