"""Canonical turbulence concept diagrams (#1112): structure from roles and the model, not pixels."""

import math

import numpy as np
import pytest

import vaft.diagram
from vaft.diagram._equations import formula_equation
from vaft.diagram._scene import Arrow, Label, Marker, Polyline
from vaft.formula import turbulence

BUILDERS = ("drift_wave_transport_cycle", "cross_phase_transport", "mixing_length_saturation",
            "exb_shear_decorrelation", "zonal_flow_feedback", "turbulence_scale_hierarchy")


@pytest.mark.parametrize("name", BUILDERS)
def test_every_turbulence_diagram_builds_a_deterministic_scene_and_strips_its_text(name):
    build = getattr(vaft.diagram, name)
    diagram = build()
    assert diagram.name == name and diagram.scene == build().scene
    bare = build(labels=False)
    assert not [it for it in bare.scene.items if isinstance(it, Label) and it.role not in ("axes", "ticks")
                and it.text.strip()]


def test_the_drift_wave_cycle_closes_through_profile_relaxation():
    diagram = vaft.diagram.drift_wave_transport_cycle()
    cycle = diagram.model["cycle"]
    assert cycle[0] == "gradient" and cycle[-1] == "radial_flux"
    assert diagram.model["feedback"] == ("radial_flux", "gradient")
    scene = diagram.scene
    assert len([it for it in scene.role("cycle") if isinstance(it, Arrow)]) == len(cycle) - 1
    relax = [it for it in scene.role("relaxation") if isinstance(it, Arrow)]
    assert len(relax) == 1 and relax[0].end[1] > relax[0].start[1]  # from the flux (bottom) back up to the gradient
    for role in ("gradient", "exb_velocity", "radial_flux"):
        assert scene.role(role), role
    assert diagram.model["equations"]["radial_flux"] == formula_equation(
        turbulence.turbulent_particle_flux_from_fluctuations)
    # owner review: growth is conditional on the drive and kinetic response, not implied by omega_*
    assert cycle.index("drift") < cycle.index("drive") < cycle.index("growth")
    # fast saturation (at fixed gradients) and slow relaxation are both drawn, and kept apart
    assert diagram.model["feedback_time_scale"].startswith("transport")
    assert scene.role("saturation") and scene.role("relaxation")


def test_cross_phase_flux_is_zero_positive_and_negative():
    diagram = vaft.diagram.cross_phase_transport()
    fluxes = diagram.model["fluxes"]
    assert fluxes["zero"] == pytest.approx(0.0, abs=1e-12)
    assert fluxes["plus"] > 0 and fluxes["minus"] == pytest.approx(-fluxes["plus"])
    assert "theta = alpha_phi - alpha_n" in diagram.model["convention"]
    # x up, y right, B = B z: z = x cross y points into the page
    assert "into the page" in diagram.model["convention"]
    assert np.allclose(np.cross([0.0, 1.0, 0.0], [1.0, 0.0, 0.0]), [0.0, 0.0, -1.0])  # page frame: up x right = in
    scene = diagram.scene
    assert len(scene.role("fluctuation_density")) >= 3 and len(scene.role("fluctuation_potential")) >= 3
    marks = sorted((it for it in scene.role("radial_flux") if not isinstance(it, Label)),
                   key=lambda it: it.at[0] if isinstance(it, Marker) else it.start[0])
    assert [type(it) for it in marks] == [Marker, Arrow, Arrow]  # left to right: 0, +pi/2, -pi/2
    outward, inward = marks[1], marks[2]
    # the flux is along x (radial), drawn up the page: vertical arrows, up for outward
    assert outward.start[0] == outward.end[0] and outward.end[1] > outward.start[1]
    assert inward.start[0] == inward.end[0] and inward.end[1] < inward.start[1]


def test_mixing_length_is_a_phenomenological_scaling_with_its_caveats():
    model = vaft.diagram.mixing_length_saturation().model
    assert model["saturated_amplitude"] == pytest.approx(model["gamma"] / model["k_perp"])
    assert model["diffusivity"] == pytest.approx(
        turbulence.mixing_length_diffusivity_from_gamma_k_perp(model["gamma"], model["k_perp"]))
    assert model["steps"][-1] == "diffusivity"
    scene = vaft.diagram.mixing_length_saturation().scene
    texts = " ".join(it.text for it in scene.items if isinstance(it, Label))
    assert "phenomenological" in texts and "Staebler" in texts and "not a predictive" in texts.lower()


def test_shear_tilts_the_eddy_and_shortens_its_radial_cut():
    diagram = vaft.diagram.exb_shear_decorrelation(shear=1.6)
    cut = diagram.model["radial_cut"]
    assert cut["sheared"] == pytest.approx(cut["unsheared"] / math.sqrt(1.0 + 1.6 ** 2))
    scene = diagram.scene
    assert scene.role("eddy") and scene.role("sheared_eddy")
    # the sheared eddy is tilted: its contour is taller than it is wide, the round one is not
    def aspect(role):
        pts = np.array(max((it for it in scene.role(role) if isinstance(it, Polyline)), key=lambda p: len(p.points))
                       .points)
        return np.ptp(pts[:, 1]) / np.ptp(pts[:, 0])
    assert aspect("eddy") == pytest.approx(1.0, rel=1e-3) and aspect("sheared_eddy") > 1.5
    # the flow is uniform on the left and changes sign across the sheared eddy
    shear_arrows = [it for it in scene.role("shear") if isinstance(it, Arrow)]
    assert {np.sign(a.end[1] - a.start[1]) for a in shear_arrows} == {-1.0, 1.0}
    assert diagram.model["strain"] == pytest.approx(1.6)
    # the fixed-y cut is geometry, kept apart from the statistical decorrelation step
    assert scene.role("radial_cut") and not scene.role("correlation_length")
    assert scene.role("decorrelation_step")
    step_text = " ".join(it.text for it in scene.role("decorrelation_step") if isinstance(it, Label))
    assert "k_x(t) = k_x(0) - Sk_yt" in step_text
    with pytest.raises(ValueError):
        vaft.diagram.exb_shear_decorrelation(shear=0.0)


def test_zonal_flow_loop_runs_turbulence_to_shear_and_back():
    diagram = vaft.diagram.zonal_flow_feedback()
    assert diagram.model["loop"] == ["gradient", "turbulence", "reynolds_stress", "zonal_flow", "shear"]
    assert diagram.model["suppression"] == ("shear", "turbulence")
    scene = diagram.scene
    assert len([it for it in scene.role("feedback_loop") if isinstance(it, Arrow)]) == 4
    # the suppression is drawn without an arrow head (an inhibition bar)
    suppression = scene.role("suppression")
    assert suppression and not [it for it in suppression if isinstance(it, Arrow)]
    # the line runs from the shear box up to the turbulence box
    line = max((it for it in suppression if isinstance(it, Polyline)), key=lambda p: np.hypot(
        *np.subtract(p.points[-1], p.points[0])))
    shear_y = np.mean([p[1] for p in next(it for it in scene.role("shear") if isinstance(it, Polyline)).points])
    turb_y = np.mean([p[1] for p in next(it for it in scene.role("turbulence") if isinstance(it, Polyline)).points])
    assert line.points[0][1] > shear_y and line.points[-1][1] < turb_y and line.points[-1][1] > line.points[0][1]
    for role in ("reynolds_stress", "zonal_flow", "shear"):
        assert scene.role(role), role


def test_scale_hierarchy_keeps_electron_below_ion_scales_and_tem_open():
    model = vaft.diagram.turbulence_scale_hierarchy().model
    lengths, bands = model["lengths"], model["bands"]
    assert lengths["rho_e"] < lengths["rho_i"] < lengths["minor"]
    # hydrogen, v_t = sqrt(T/m): the owner's check values at B = 0.1 T, T_e = 50 eV, T_i = 30 eV
    assert lengths["rho_i"] == pytest.approx(5.60e-3, rel=5e-3)
    assert lengths["rho_s"] == pytest.approx(7.22e-3, rel=5e-3)
    assert lengths["rho_e"] == pytest.approx(0.169e-3, rel=1e-2)
    # every band is a perpendicular scale below the machine: none reaches the minor radius
    assert all(hi < lengths["minor"] for lo, hi in bands.values())
    assert bands["electron_scale"][1] < bands["ion_scale"][0]
    # TEM is not confined to the ion band: it reaches below it
    assert bands["tem_extent"][0] < bands["ion_scale"][0]
    scene = vaft.diagram.turbulence_scale_hierarchy().scene
    for role in ("ion_scale", "electron_scale", "tem_extent", "equilibrium_scale"):
        assert scene.role(role), role
