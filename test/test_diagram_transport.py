"""Canonical transport concept diagrams: structure from roles and the model, not pixels."""

import numpy as np
import pytest

import vaft.diagram
from vaft.diagram._chart import Chart
from vaft.diagram._equations import formula_equation
from vaft.diagram._scene import Arrow, Label, Marker, Polyline
from vaft.formula import transport

BUILDERS = ("random_walk_diffusion", "transport_flux_gradient", "transport_reference_scales")


@pytest.mark.parametrize("name", BUILDERS)
def test_every_transport_diagram_builds_a_scene_and_strips_its_text(name):
    diagram = getattr(vaft.diagram, name)()
    assert diagram.name == name and diagram.scene.items
    bare = getattr(vaft.diagram, name)(labels=False)
    assert not [it for it in bare.scene.items if isinstance(it, Label) and it.role not in ("axes", "ticks")
                and it.text.strip()]


def test_the_einstein_and_taylor_green_kubo_views_give_one_diffusivity():
    model = vaft.diagram.random_walk_diffusion().model
    # fitted to the drawn curve's late slope, so equal to <v^2> tau_c up to the e^{-t/tau_c} tail
    assert model["diffusivity_einstein"] == pytest.approx(model["variance"] * model["correlation_time"], rel=1e-3)
    assert model["diffusivity_taylor_green_kubo"] == pytest.approx(model["diffusivity_einstein"], rel=1e-3)
    scene = vaft.diagram.random_walk_diffusion().scene
    for role in ("einstein", "taylor_green_kubo", "diffusivity", "correlation_area"):
        assert scene.role(role), role
    # the trajectories are zero-mean diffusion: inside the plot, ending on both sides
    paths = [it.points for it in scene.items if isinstance(it, Polyline) and it.role.startswith("trajectory")]
    ends = [p[-1][1] - p[0][1] for p in paths]  # displacement from the common start, in cm
    assert ends and min(ends) < 0 < max(ends)
    # both definitions point at the one diffusivity box
    assert {type(it) for it in scene.role("einstein")} >= {Arrow}
    assert {type(it) for it in scene.role("taylor_green_kubo")} >= {Arrow}


def test_flux_gradient_keeps_diffusive_and_convective_terms_apart_in_every_channel():
    diagram = vaft.diagram.transport_flux_gradient()
    scene = diagram.scene
    channels = diagram.model["channels"]
    assert channels == ["particle", "heat", "momentum"]
    for role in ("gradient", "diffusive_term", "convective_term", "radial_flux"):
        # each box is an outline and a label
        assert len(scene.role(role)) == 2 * len(channels), role
    # residual stress is its own term, neither diffusive nor convective
    assert sum(isinstance(it, Polyline) for it in scene.role("residual_stress")) == 1
    assert not any("residual" in it.text for it in scene.role("convective_term") if isinstance(it, Label))
    assert diagram.model["particle_flux"] == formula_equation(transport.particle_flux_from_D_grad_n_V_n)


def test_one_flux_is_a_line_of_d_v_pairs_not_a_point():
    diagram = vaft.diagram.transport_flux_gradient()
    ex = diagram.model["inverse_example"]
    markers = [it for it in diagram.scene.role("candidate_pair") if isinstance(it, Marker)]
    assert len(markers) >= 2
    line = next(it for it in diagram.scene.role("degenerate_line") if isinstance(it, Polyline))
    xs, ys = np.array(line.points).T
    for marker in markers:  # every candidate (D, V) pair sits on the drawn line
        assert marker.at[1] == pytest.approx(np.interp(marker.at[0], xs, ys), abs=1e-6)
    for d in (0.0, 0.7, 1.9):
        v = (ex["flux"] + d * ex["grad_n"]) / ex["n"]
        assert transport.particle_flux_from_D_grad_n_V_n(d, ex["grad_n"], ex["n"], v) == pytest.approx(ex["flux"])
    assert diagram.scene.role("inverse_problem")


def test_reference_scales_are_a_taxonomy_without_a_magnitude_axis():
    diagram = vaft.diagram.transport_reference_scales()
    model = diagram.model
    assert model["magnitude_axis"] is None
    assert model["groups"] == {"collisional": ["classical", "neoclassical"], "empirical": ["bohm"],
                               "turbulence": ["gyrobohm"]}
    assert model["equations"]["gyrobohm"] == formula_equation(transport.gyrobohm_diffusivity_from_rho_s_c_s_L_ref)
    scene = diagram.scene
    # no chart axes: the leaves sit on one row, so the figure implies no ordering
    assert not scene.role("axes")
    centres = {leaf: np.mean(np.array(scene.role(leaf)[0].points), axis=0)
               for leaf in ("classical", "neoclassical", "bohm", "gyrobohm")}
    assert len({round(c[1], 6) for c in centres.values()}) == 1
    # left to right does not read as the forbidden ladder classical < neoclassical < Bohm < gyro-Bohm
    order = sorted(centres, key=lambda leaf: centres[leaf][0])
    assert order != ["classical", "neoclassical", "bohm", "gyrobohm"]
    assert order != ["gyrobohm", "bohm", "neoclassical", "classical"]
    assert sum(isinstance(it, Arrow) for it in scene.role("taxonomy")) == 3 + 4


def test_the_neoclassical_regimes_state_their_collisionality_convention():
    diagram = vaft.diagram.neoclassical_collisionality(epsilon=0.1)
    assert isinstance(diagram.model, Chart)
    assert diagram.model.parameters["banana_plateau"] == pytest.approx(0.1 ** 1.5)
    assert any("\\hat\\nu" in it.text for it in diagram.scene.role("axes") if isinstance(it, Label))
    for regime in ("banana", "plateau", "pfirsch_schlueter"):
        assert diagram.scene.role(f"region_{regime}")
    # each boundary is a range, not an exact number
    bands = diagram.scene.role("transition_range")
    assert len(bands) == 2
    for band, boundary in zip(bands, ("banana_plateau", "plateau_pfirsch_schlueter")):
        x = np.log10(diagram.model.parameters[boundary])
        edges = sorted({p[0] for p in band.points})
        centre = diagram.model.to_cm(np.array([x, 0.0]))[0]
        assert edges[0] < centre < edges[1]
