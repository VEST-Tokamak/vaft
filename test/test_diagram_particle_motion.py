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
    # the lattice is much finer than rho, so the current cancels bin by bin inside --
    # not just in total, which any symmetric arrangement would give
    assert fig.parameters["lattice_spacing"] < 0.5 * rho
    interior = (edges[:-1] >= 2 * rho) & (edges[1:] <= side - 2 * rho)
    assert interior.sum() >= 10
    assert np.max(np.abs(Jy[interior])) < 0.01 * np.max(np.abs(Jy))
    assert np.max(np.abs(fig.vectors["J_x_of_y"][interior])) < 0.01 * np.max(np.abs(fig.vectors["J_x_of_y"]))


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
    (vaft.diagram.exb_drift, {"mass_ratio": 1836.0}),
    (vaft.diagram.toroidal_drift, {"aspect_ratio": 1.1}),
    (vaft.diagram.toroidal_drift, {"aspect_ratio": float("inf")}),
])
def test_invalid_parameters_fail_explicitly(fn, kw):
    with pytest.raises(ValueError):
        fn(**kw)


# --- projections: every view is drawn from the same computed model ------------------------

PROJECTIONS = {
    "exb_drift": ("perpendicular", "3d"),
    "curvature_drift": ("3d", "poloidal", "top"),
    "magnetization_current": ("perpendicular", "3d"),
    "toroidal_drift": ("3d", "poloidal", "top"),
}


@pytest.mark.parametrize("name", PROJECTIONS)
def test_every_projection_shares_one_model(name):
    fn = getattr(vaft.diagram, name)
    models = [fn(projection=p).model for p in PROJECTIONS[name]]
    for other in models[1:]:
        assert other.orbits.keys() == models[0].orbits.keys()
        for key in models[0].orbits:
            assert np.array_equal(other.orbits[key], models[0].orbits[key]), key
        for key in models[0].vectors:
            assert np.array_equal(other.vectors[key], models[0].vectors[key]), key


def test_the_parallel_velocity_leaves_the_perpendicular_exb_motion_alone():
    fig = vaft.diagram.exb_drift().model
    for name in ("ion", "electron"):
        z = fig.orbits[name][:, 2]
        dt = fig.parameters[f"{name}_dt"]
        # the helix climbs at v_par exactly: B along z does not act on v_z
        assert np.allclose(np.diff(z) / dt, fig.parameters["v_par"])


def test_the_poloidal_curvature_view_climbs_along_the_formula_drift():
    d = vaft.diagram.curvature_drift(projection="poloidal")
    v_d = d.model.vectors["drift"]
    (arrow,) = [it for it in d.scene.role("drift") if hasattr(it, "end")]
    step = np.subtract(arrow.end, arrow.start)
    # at the start R-hat is +x, so the (R, z) arrow is along (v_x, v_z) of the formula
    assert np.allclose(step / np.linalg.norm(step), np.array([v_d[0], v_d[2]]) / np.hypot(v_d[0], v_d[2]))


def test_the_poloidal_toroidal_section_draws_the_computed_directions():
    d = vaft.diagram.toroidal_drift(projection="poloidal")
    v = d.model.vectors

    def direction(role):
        (arrow,) = [it for it in d.scene.role(role) if hasattr(it, "end")]
        step = np.subtract(arrow.end, arrow.start)
        return step / np.linalg.norm(step)

    def rz(vec):
        vec = np.array([np.dot(vec, v["R_hat"]), vec[2]])
        return vec / np.linalg.norm(vec)

    for role, key in (("ion_drift", "ion_drift"), ("electron_drift", "electron_drift"), ("E", "E"), ("v_E", "v_E")):
        assert np.allclose(direction(role), rz(v[key]), atol=0.05), role
    assert "\\otimes" in d.tikz  # +phi is into the page with R right and z up


@pytest.mark.parametrize("fn", [vaft.diagram.exb_drift, vaft.diagram.curvature_drift,
                                vaft.diagram.magnetization_current, vaft.diagram.toroidal_drift])
def test_an_unknown_projection_fails_explicitly(fn):
    with pytest.raises(ValueError, match="projection"):
        fn(projection="side")


@pytest.mark.parametrize("name", PROJECTIONS)
def test_every_view_shows_the_formulas_as_their_docstrings_define_them(name):
    from vaft.diagram import _particle_motion as pm

    for projection in PROJECTIONS[name]:
        d = getattr(vaft.diagram, name)(projection=projection)
        (box,) = d.scene.role("equations")
        for function in pm._EQUATIONS[name]:
            assert pm.formula_equation(function) in box.text, (name, projection, function.__name__)
        # and the box sits above the note, not on it
        (note,) = d.scene.role("note")
        assert note.at[1] < box.at[1]
    assert not getattr(vaft.diagram, name)(labels=False).scene.role("equations")


def test_formula_equation_refuses_a_function_without_one():
    from vaft.diagram import _particle_motion as pm

    with pytest.raises(ValueError, match="documents no"):
        pm.formula_equation(lambda: None)


def _arrow_direction(diagram, role):
    arrows = [it for it in diagram.scene.role(role) if hasattr(it, "end")]
    step = np.subtract(arrows[0].end, arrows[0].start)
    return step / np.linalg.norm(step)


def test_the_3d_views_draw_the_computed_directions():
    from vaft.diagram._projection import project

    def screen(vec):
        d = project(np.asarray(vec, dtype=float)) - project(np.zeros(3))
        return d / np.linalg.norm(d)

    d = vaft.diagram.curvature_drift()
    assert np.allclose(_arrow_direction(d, "drift"), screen(d.model.vectors["drift"]), atol=1e-6)
    d = vaft.diagram.exb_drift(projection="3d")
    assert np.allclose(_arrow_direction(d, "legend_vE"), screen(d.model.vectors["ion_drift"]), atol=1e-6)
    d = vaft.diagram.toroidal_drift()
    assert np.allclose(_arrow_direction(d, "v_E"), screen(d.model.vectors["v_E"]), atol=1e-6)


def test_the_top_views_point_grad_b_inward():
    d = vaft.diagram.toroidal_drift(projection="top")
    (arrow,) = [it for it in d.scene.role("grad_B") if hasattr(it, "end")]
    assert np.linalg.norm(arrow.end) < np.linalg.norm(arrow.start)  # towards the machine axis
    d = vaft.diagram.curvature_drift(projection="top")
    (arrow,) = [it for it in d.scene.role("grad_B") if hasattr(it, "end")]
    assert np.linalg.norm(arrow.end) < np.linalg.norm(arrow.start)
