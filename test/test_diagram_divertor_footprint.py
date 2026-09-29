"""Divertor heat footprint (#1209): geometry from the equilibrium, profile from vaft.formula.sol."""

import numpy as np
import pytest
from scipy.optimize import brentq

import vaft.diagram
from vaft.diagram import _divertor_footprint as df
from vaft.diagram._scene import Label
from vaft.formula.sol import eich_target_heat_flux_profile


@pytest.fixture(scope="module")
def model():
    return df.footprint_model()


def test_strike_point_and_x_point_lie_on_the_separatrix(model):
    sp, psi_x = model["spline"], model["psi_x"]
    R_x, Z_x = model["x_point"]
    scale = abs(float(sp.ev(model["R_omp"], model["Z_axis"], dx=1))) * model["lambda_q"]  # flux across one lambda_q
    assert np.hypot(sp.ev(R_x, Z_x, dx=1), sp.ev(R_x, Z_x, dy=1)) * model["lambda_q"] < 1e-4 * scale
    assert sp.ev(model["R_strike"], model["target_z"]) == pytest.approx(psi_x, abs=1e-8 * scale)
    assert model["strike_points"]["inner"] < R_x < model["strike_points"]["outer"]


@pytest.mark.parametrize("target", ["outer", "inner"])
def test_sol_surfaces_reach_the_target_one_flux_expansion_apart(target):
    # a surface lambda_q outside the separatrix at the outer midplane crosses the target at lambda_q f_x as
    # lambda_q -> 0: an independent check of f_x, found by following the flux, not by the derivative ratio
    m = df.footprint_model(lambda_q=2e-5, target=target)
    sp, z_t, R_t, d = m["spline"], m["target_z"], m["R_strike"], m["direction"]
    level = m["sol_levels"][0]
    s_1 = brentq(lambda s: float(sp.ev(R_t + d * s, z_t)) - level, 1e-9, 10 * m["lambda_q"] * m["flux_expansion"])
    assert s_1 == pytest.approx(m["lambda_q"] * m["flux_expansion"], rel=0.01)


EQUILIBRIA = [dict(), dict(elongation=1.4), dict(triangularity=0.6), dict(major_radius=0.6, aspect_ratio=2.5)]


@pytest.mark.parametrize("shape", EQUILIBRIA, ids=lambda k: ",".join(f"{a}={b}" for a, b in k.items()) or "default")
def test_geometry_holds_on_other_diverted_shapes_and_psi_conventions(shape):
    import dataclasses

    from vaft.process.equilibrium import solovev_example

    eq = solovev_example("single_null", a_parameter=0.0, **shape)
    base = df.footprint_model(eq)
    # the SOL surfaces are on the far side of the separatrix from the axis
    for level in base["sol_levels"]:
        assert np.sign(level - base["psi_x"]) != np.sign(base["psi_axis"] - base["psi_x"])
    # a flux flipped in sign and shifted by a constant is the same geometry
    flipped = dataclasses.replace(eq, psi=-np.asarray(eq.psi) + 3.0, psi_axis=-eq.psi_axis + 3.0,
                                  psi_boundary=-eq.psi_boundary + 3.0)
    other = df.footprint_model(flipped)
    assert other["R_strike"] == pytest.approx(base["R_strike"], abs=1e-9)
    assert other["flux_expansion"] == pytest.approx(base["flux_expansion"], rel=1e-9)


def test_no_flux_line_is_drawn_through_the_target():
    diagram = vaft.diagram.divertor_heat_footprint()
    plate_y = [p for p in diagram.scene.items if getattr(p, "role", "") == "target"][0].points[0][1]
    for item in diagram.scene.items:
        if getattr(item, "role", "") == "separatrix" or getattr(item, "role", "").startswith("sol_surface"):
            assert min(y for _, y in item.points) >= plate_y - 1e-9


def test_rendering_is_stable_under_last_bit_noise_in_psi():
    import dataclasses

    from vaft.process.equilibrium import solovev_example

    base = vaft.diagram.divertor_heat_footprint().tikz
    rng = np.random.default_rng(5)
    for _ in range(2):
        eq = solovev_example("single_null", a_parameter=0.0)
        eq = dataclasses.replace(eq, psi=np.asarray(eq.psi) * (1 + 1e-15 * rng.standard_normal(np.shape(eq.psi))))
        assert vaft.diagram.divertor_heat_footprint(eq).tikz == base


def test_the_profile_is_the_eich_formula_with_the_equilibrium_expansion(model):
    np.testing.assert_allclose(
        model["q"], eich_target_heat_flux_profile(model["s"], 1.0, model["lambda_q"], model["spreading"],
                                                  flux_expansion=model["flux_expansion"]))
    assert model["flux_expansion"] > 1.0
    drawn = vaft.diagram.divertor_heat_footprint().model
    assert drawn["flux_expansion"] == pytest.approx(model["flux_expansion"])


def test_scene_shows_the_named_elements():
    diagram = vaft.diagram.divertor_heat_footprint()
    roles = {getattr(i, "role", "") for i in diagram.scene.items}
    assert {"separatrix", "x_point", "strike_point", "target", "footprint", "lambda_q_f_x", "spreading",
            "sol_surface_1", "sol_crossing", "target_coordinate"} <= roles
    assert not [i for i in vaft.diagram.divertor_heat_footprint(labels=False).scene.items
                if isinstance(i, Label) and i.role not in ("axes", "ticks")]


def test_inputs_are_checked():
    from vaft.process.equilibrium import solovev_example

    with pytest.raises(ValueError, match="X-point"):
        df.footprint_model(solovev_example("limited", a_parameter=0.0))
    with pytest.raises(ValueError):
        df.footprint_model(lambda_q=0.0)
    with pytest.raises(ValueError):
        df.footprint_model(target="upper")
    with pytest.raises(ValueError, match="below"):
        df.footprint_model(target_z=0.0)
    # an explicit target higher up: legs still inside, the strike points move towards the X-point
    higher = df.footprint_model(target_z=-0.46)
    assert higher["strike_points"]["outer"] < df.footprint_model()["strike_points"]["outer"]
    # a target so low that the legs have left the vessel
    with pytest.raises(ValueError, match="outside the limiter"):
        df.footprint_model(target_z=-0.8)
