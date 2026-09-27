"""Guiding-centre invariant diagrams (#1092): the orbit is the invariants', and P_phi is visibly constant."""

import numpy as np
import pytest

import vaft.diagram
from vaft.diagram._equations import formula_equation
from vaft.diagram._guiding_center import banana_orbit
from vaft.formula.equilibrium import vacuum_toroidal_field
from vaft.formula.particle import (
    bounce_harmonic_detuning,
    guiding_center_toroidal_momentum,
    magnetic_moment,
)


def test_the_banana_conserves_p_phi_mu_and_energy():
    o = banana_orbit()
    P = guiding_center_toroidal_momentum(1.0, 1.0, o["v_par"], o["R"], 1.0, o["psi"])
    np.testing.assert_allclose(P, o["P_phi"], rtol=1e-6)
    B = vacuum_toroidal_field(1.0, 3.0, o["R"])
    v_perp = np.sqrt(np.maximum(0.012**2 - o["v_par"] ** 2, 0.0))
    mu = magnetic_moment(1.0, v_perp, B)
    np.testing.assert_allclose(mu, mu[0], rtol=1e-3)
    # finite orbit width: the guiding centre crosses surfaces, and v_par changes sign (trapped)
    assert np.ptp(o["r"]) > 0.02
    assert o["v_par"].min() < 0 < o["v_par"].max()


def test_the_decomposition_shows_the_parts_trading_at_constant_p_phi():
    d = vaft.diagram.canonical_toroidal_momentum()
    parts, changes = d.model["parts"], d.model["changes"]
    totals = [p["P_phi"] for p in parts.values()]
    np.testing.assert_allclose(totals, totals[0], rtol=1e-9)
    assert abs(parts["A"]["flux"] - parts["C"]["flux"]) > 1e-3  # psi moves
    for d_flux, d_mech in changes.values():
        assert d_flux + d_mech == pytest.approx(0.0, abs=1e-12)
    for name in ("A", "C"):
        assert d.scene.role(f"bar:flux:{name}") and d.scene.role(f"bar:mechanical:{name}")
    (box,) = d.scene.role("equations")
    assert formula_equation(guiding_center_toroidal_momentum) in box.text


@pytest.mark.parametrize("phase", [0.0, 0.2, 0.5, 0.9])
def test_the_phase_moves_point_c_along_one_orbit(phase):
    m = vaft.diagram.canonical_toroidal_momentum(phase).model
    n = len(m["orbit"]["theta"])
    assert m["points"]["C"] == round(phase * (n - 1))
    with pytest.raises(ValueError):
        vaft.diagram.canonical_toroidal_momentum(1.0)


def test_the_invariant_diagram_separates_particle_guiding_centre_surfaces_and_invariants():
    d = vaft.diagram.guiding_center_invariants()
    for role in ("particle_orbit", "guiding_center", "flux_surface", "invariant:mu", "invariant:j_parallel",
                 "invariant:p_phi", "motion:gyro", "motion:bounce", "motion:drift"):
        assert d.scene.role(role), role
    # the particle circles about the guiding centre at the drawn Larmor radius
    (particle,) = d.scene.role("particle_orbit")
    assert len(particle.points) > 50


def test_symmetry_breaking_is_secular_only_at_resonance():
    chart = vaft.diagram.toroidal_symmetry_breaking().model
    p = chart.parameters
    assert p["detuning_resonant"] == 0.0 and p["detuning_non_resonant"] != 0.0
    assert p["detuning_resonant"] == bounce_harmonic_detuning(1.0, 0.375, 0.125, l=1, n=2)
    axi = chart.curves["axisymmetric"][:, 1]
    non = chart.curves["non_resonant"][:, 1]
    res = chart.curves["resonant"][:, 1]
    assert np.ptp(axi) == 0.0
    assert abs(non[-1] - 1.0) < 2 * p["kick"] / abs(p["detuning_non_resonant"])  # bounded
    assert res[-1] - 1.0 == pytest.approx(p["kick"] * 60.0)  # linear


@pytest.mark.parametrize("name", ["guiding_center_invariants", "canonical_toroidal_momentum",
                                  "toroidal_symmetry_breaking"])
def test_every_guiding_centre_diagram_is_deterministic_and_exported(name):
    fn = getattr(vaft.diagram, name)
    assert fn().tikz == fn().tikz
    assert name in vaft.diagram.__all__
    assert fn().scene.role("note") and not fn(labels=False).scene.role("note")
