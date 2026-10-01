"""Canonical operational-space projections (#1425): axes by quantity identity, registered boundaries only."""

import numpy as np
import pytest

import vaft.diagram
from vaft.diagram import _projections as P
from vaft.formula import boundaries as B


def test_the_hugill_axes_are_the_quantities_the_registered_boundaries_use():
    hugill = P.get_projection("hugill")
    assert hugill.x == B.get_boundary("greenwald_hugill").target
    assert hugill.y == B.get_boundary("greenwald_hugill").input("inverse_cylindrical_q")
    assert P.compatible_boundaries(hugill) == hugill.default_boundaries == ("greenwald_hugill",)
    assert P.compatible_boundaries(hugill, ("greenwald_hugill", "murakami_hugill")) == (
        "greenwald_hugill", "murakami_hugill")


@pytest.mark.parametrize("key", ["low_q", "greenwald", "murakami", "troyon"])
def test_boundaries_on_other_quantities_are_not_compatible_with_hugill(key):
    """low_q bounds q_psi, not q_cyl; greenwald and murakami bound a density, not nR/B."""
    hugill = P.get_projection("hugill")
    boundary = B.get_boundary(key)
    with pytest.raises(P.IncompatibleBoundary):
        P.placement(hugill, boundary, {name: 1.0 for name in boundary.input_names})
    assert P.compatible_boundaries(hugill, (key,)) == ()


def test_hugill_rejects_an_incompatible_boundary_by_name():
    with pytest.raises(P.IncompatibleBoundary, match="low_q"):
        vaft.diagram.hugill(boundaries=("greenwald_hugill", "low_q"))


def test_a_registered_curve_needs_its_fixed_inputs():
    with pytest.raises(P.IncompatibleBoundary, match="area_elongation"):
        P.placement(P.get_projection("hugill"), B.get_boundary("greenwald_hugill"))


def test_the_placements_say_how_each_boundary_is_drawn():
    hugill, troyon = P.get_projection("hugill"), P.get_projection("troyon")
    curve = P.placement(hugill, B.get_boundary("greenwald_hugill"), {"area_elongation": 1.0})
    assert curve == {"kind": "curve", "sweep": "inverse_cylindrical_q", "swap_axes": True}
    assert P.placement(hugill, B.get_boundary("murakami_hugill"))["kind"] == "vertical"
    assert P.placement(troyon, B.get_boundary("troyon"))["kind"] == "ratio"


def test_hugill_draws_the_registered_murakami_line_on_request():
    chart = vaft.diagram.hugill(boundaries=("greenwald_hugill", "murakami_hugill")).model
    x = B.boundary_value(B.get_boundary("murakami_hugill"))
    assert np.allclose(chart.curves["murakami"][:, 0], x)
    assert "murakami" not in vaft.diagram.hugill().model.curves


@pytest.mark.parametrize("kappa", [1.0, 1.7])
def test_the_hugill_line_is_unchanged_by_the_refactor(kappa):
    """Regression: the line through the origin with slope pi/(50 kappa_a) the formula-built chart drew."""
    g = vaft.diagram.hugill(elongation=kappa, q_limit=2.0).model.curves["greenwald"]
    assert g[0].tolist() == [0.0, 0.0] and len(g) == 201
    np.testing.assert_allclose(g[1:, 1] / g[1:, 0], np.pi / (50 * kappa), rtol=1e-12)
    assert g[-1, 1] == pytest.approx(0.7, rel=1e-12)


def test_troyon_defaults_to_the_registered_limit():
    chart = vaft.diagram.troyon().model
    assert chart.parameters["beta_N_max"] == pytest.approx(B.boundary_value(B.get_boundary("troyon")), rel=1e-12)
    assert chart.parameters["beta_N_max"] == pytest.approx(2.7646, rel=1e-4)
    assert vaft.diagram.troyon(beta_N_max=3.5).model.parameters["beta_N_max"] == 3.5


def test_the_projections_name_their_references():
    for key in P.PROJECTIONS:
        projection = P.get_projection(key)
        assert projection.references and projection.key == key
    with pytest.raises(KeyError, match="hugill"):
        P.get_projection("nope")
