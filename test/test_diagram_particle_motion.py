"""Single-particle motion diagrams: every drawn drift is the formula's and the orbit's."""

import numpy as np
import pytest

import vaft.diagram
from vaft.formula.particle import exb_drift_velocity


def _fit_drift(x, dt):
    t = np.arange(len(x)) * dt
    return np.polyfit(t, x, 1)[0]


# --- E x B -----------------------------------------------------------------------


@pytest.mark.parametrize("mass_ratio", [1.0, 4.0, 9.0])
def test_both_species_drift_at_the_exb_velocity(mass_ratio):
    fig = vaft.diagram.exb_drift(mass_ratio=mass_ratio).model
    v_E = exb_drift_velocity([fig.parameters["E"], 0, 0], [0, 0, fig.parameters["B"]])
    assert v_E[1] < 0 and abs(v_E[0]) < 1e-15  # along -y for E = +x, B = +z
    for name, m in (("ion", mass_ratio), ("electron", 1.0)):
        x, dt = fig.orbits[name], fig.parameters[f"{name}_dt"]
        # positions a whole number of gyroperiods apart share the gyrophase, so their
        # displacement is the drift alone
        steps = int(round(2 * np.pi * m / dt))
        k = (len(x) - 1) // steps * steps
        measured = (x[k] - x[0]) / (k * dt)
        assert np.allclose(measured[:2], v_E[:2], atol=1e-3), name
        assert np.allclose(fig.vectors[f"{name}_drift"], v_E)


def test_the_guiding_centre_arrow_starts_at_the_orbit_centre():
    fig = vaft.diagram.exb_drift().model
    for name, m in (("ion", fig.parameters["mass_ratio"]), ("electron", 1.0)):
        x = fig.orbits[name]
        dt = fig.parameters[f"{name}_dt"]
        steps = int(round(2 * np.pi * m / dt))
        v_E = fig.vectors[f"{name}_drift"]
        # the first gyroperiod's mean, moved back half a period of drift, is the starting guiding centre;
        # the leapfrog's half-step velocity offset shifts it by at most one step of gyromotion
        centre = x[:steps].mean(axis=0) - v_E * 0.5 * (steps - 1) * dt
        u = 1.0 / np.sqrt(m)
        assert np.allclose(centre[:2], fig.vectors[f"{name}_guiding_centre"][:2], atol=u * dt)


def test_equal_energy_makes_the_ion_orbit_sqrt_mass_ratio_larger():
    fig = vaft.diagram.exb_drift(mass_ratio=4.0).model
    widths = {n: np.ptp(fig.orbits[n][:, 0]) for n in ("ion", "electron")}
    assert widths["ion"] / widths["electron"] == pytest.approx(2.0, rel=0.05)


# --- curvature and grad-B ----------------------------------------------------------


def test_the_curved_field_drift_arrow_is_the_orbit_drift_and_points_up_for_an_ion():
    fig = vaft.diagram.curvature_drift().model
    x = fig.orbits["ion"]
    dt = fig.parameters["duration"] / (len(x) - 1)
    measured_vz = _fit_drift(x, dt)[2]
    assert fig.vectors["drift"][2] > 0
    assert measured_vz == pytest.approx(fig.vectors["drift"][2], rel=0.1)
    # the drawn guiding centre stays within about a Larmor radius of the orbit
    rho = fig.parameters["v_perp"] / fig.parameters["B0"]
    gc = fig.orbits["guiding_centre"]
    idx = np.linspace(0, len(x) - 1, len(gc)).astype(int)
    assert np.max(np.linalg.norm(x[idx] - gc, axis=1)) < 1.5 * rho


# --- magnetization current ------------------------------------------------------------


def test_the_net_edge_current_is_diamagnetic_and_the_interior_cancels():
    fig = vaft.diagram.magnetization_current().model
    side, rho = fig.parameters["side"], fig.parameters["rho"]
    edges = fig.vectors["bin_edges"]
    Jy = fig.vectors["J_y_of_x"]
    # B points into the page (-z): a diamagnetic current circulates counter-clockwise,
    # so it flows up (+y) along the right edge and down along the left edge
    assert fig.parameters["right_edge_current"] > 0
    assert fig.parameters["top_edge_current"] < 0
    assert Jy[edges[1:] <= rho].sum() < 0
    interior = (edges[:-1] >= 2 * rho) & (edges[1:] <= side - 2 * rho)
    assert abs(Jy[interior].sum()) < 0.1 * abs(fig.parameters["right_edge_current"])


# --- toroidal drift --------------------------------------------------------------------


@pytest.mark.parametrize("aspect_ratio", [1.5, 2.2, 4.0])
def test_toroidal_drift_directions(aspect_ratio):
    v = vaft.diagram.toroidal_drift(aspect_ratio=aspect_ratio).model.vectors
    assert v["ion_drift"][2] > 0 and v["electron_drift"][2] < 0
    assert v["E"][2] < 0  # from the ion layer (top) to the electron layer (bottom)
    assert np.dot(v["v_E"], v["R_hat"]) > 0  # outward
    assert np.allclose(v["v_E"], exb_drift_velocity(v["E"], v["B"]))


# --- common -------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["exb_drift", "curvature_drift", "magnetization_current", "toroidal_drift"])
def test_every_particle_diagram_is_deterministic_and_lazy(name):
    fn = getattr(vaft.diagram, name)
    assert fn().tikz == fn().tikz
    assert name in vaft.diagram.__all__


@pytest.mark.parametrize("fn, kw", [
    (vaft.diagram.exb_drift, {"mass_ratio": 0.5}),
    (vaft.diagram.toroidal_drift, {"aspect_ratio": 1.1}),
])
def test_invalid_parameters_fail_explicitly(fn, kw):
    with pytest.raises(ValueError):
        fn(**kw)
