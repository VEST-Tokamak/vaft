"""NBI diagrams (#1136): the three losses kept apart, the attenuation curves from vaft.formula.nbi."""

import numpy as np
import pytest

import vaft.diagram
from vaft.diagram import _nbi as nb
from vaft.diagram._scene import Arrow, Label, Marker
from vaft.formula.nbi import beam_birth_probability_density, shine_through_fraction


def test_losses_branch_at_the_stage_where_they_happen():
    diagram = vaft.diagram.nbi_particle_lifecycle()
    edges = dict((b, a) for a, b in diagram.model["edges"] if b in nb.LOSS_CHANNELS)
    # shine-through before ionisation, prompt loss at birth, delayed loss from the confined population
    assert edges == {"shine_through": "injection", "prompt_loss": "birth", "delayed_loss": "confined"}
    roles = {getattr(i, "role", "") for i in diagram.scene.items}
    assert {"shine_through", "prompt_loss", "delayed_loss", "thermalisation"} <= roles
    # the main path is one chain from injection to thermalisation
    chain = ["injection", "ionisation", "birth", "confined", "slowing", "thermal"]
    for a, b in zip(chain, chain[1:]):
        assert (a, b) in diagram.model["edges"]
    # thermalisation is the end of slowing down, not a product of one heating channel
    assert ("ions", "thermal") not in diagram.model["edges"]
    assert all(("slowing", c) in diagram.model["edges"] for c in ("electrons", "ions", "momentum"))
    arrows = [i for i in diagram.scene.items if isinstance(i, Arrow)]
    assert len(arrows) == len(diagram.model["edges"])


def test_attenuation_curves_are_the_formulas():
    chart = vaft.diagram.nbi_neutral_attenuation().model
    s, b = chart.curves["birth_density"].T
    alpha = nb.example_attenuation(s)
    np.testing.assert_allclose(b, beam_birth_probability_density(s, alpha))
    assert chart.parameters["shine_through_fraction"] == pytest.approx(shine_through_fraction(s, alpha))
    _, S = chart.curves["survival"].T
    assert S[0] == 1.0 and S[-1] == pytest.approx(chart.parameters["shine_through_fraction"])
    assert np.trapezoid(b, s) + S[-1] == pytest.approx(1.0, abs=1e-4)


def test_birth_markers_are_denser_where_the_birth_density_is_higher():
    diagram = vaft.diagram.nbi_neutral_attenuation()
    xs = np.sort([m.at[0] for m in diagram.scene.role("birth") if isinstance(m, Marker)])
    assert len(xs) == nb.N_BIRTH_MARKERS
    spacing = np.diff(xs)
    chart = diagram.model
    s, b = chart.curves["birth_density"].T
    peak_cm = chart.to_cm(np.array([s[np.argmax(b)], 0.0]))[0]
    # the tightest spacing sits near the peak of b, the widest at the thin exit end
    assert abs(xs[np.argmin(spacing)] - peak_cm) < 2.0
    assert np.argmax(spacing) == len(spacing) - 1


def test_labels_off():
    for build in (vaft.diagram.nbi_particle_lifecycle, vaft.diagram.nbi_neutral_attenuation):
        labelled = [i for i in build(labels=False).scene.items if isinstance(i, Label)]
        assert all(not i.text or i.role in ("axes", "ticks") for i in labelled)
        with pytest.raises(ValueError):
            build(labels=1)
