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
            "sol_surface_1"} <= roles
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
