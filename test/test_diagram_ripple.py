"""Toroidicity and ripple diagrams (#1070): drawn from the ripple and particle formulas."""

import math

import numpy as np
import pytest

import vaft.diagram
from vaft.formula.equilibrium import vacuum_toroidal_field
from vaft.formula.ripple import ripple_amplitude, ripple_well_parameter


def test_the_small_pitch_orbit_bounces_and_the_large_one_passes():
    m = vaft.diagram.trapped_and_passing_orbits().model
    trapped, passing = m["orbits"]["trapped"], m["orbits"]["passing"]
    assert trapped["trapped"] and not passing["trapped"]
    # the bounce point is where the mirror says: B(theta_b) = B_out / (1 - xi^2)
    eps = m["epsilon"]
    xi = 0.5 * m["boundary_pitch"]
    B_out = vacuum_toroidal_field(1.0, 3.0, 3.0 * (1 + eps))
    B_b = vacuum_toroidal_field(1.0, 3.0, 3.0 * (1 + eps * math.cos(trapped["bounce_theta"])))
    assert B_b == pytest.approx(B_out / (1 - xi**2), rel=2e-3)
    # the banana: the two legs differ in radius by the sign of v_par
    assert np.ptp(passing["theta"]) == pytest.approx(2 * math.pi, rel=1e-3)
    assert trapped["v_par"].min() < 0 < trapped["v_par"].max()
    # the two legs sit at different radii, and the drawn bounce markers are the tips
    r = np.hypot(trapped["R"] - 3.0, trapped["Z"])
    assert np.ptp(r) > 0.1
    d = vaft.diagram.trapped_and_passing_orbits()
    tips = [((trapped["R"][i] - 3.0) * 2.4, trapped["Z"][i] * 2.4)
            for i in (int(np.argmax(trapped["theta"])), int(np.argmin(trapped["theta"])))]
    markers = [m.at for m in d.scene.role("bounce_point")]
    assert np.allclose(sorted(markers), sorted(tips))


def test_the_rippled_field_peaks_under_the_coils():
    m = vaft.diagram.toroidal_field_ripple(16).model
    phi, B = m["phi"], m["B"]
    peaks = phi[1:-1][(B[1:-1] > B[:-2]) & (B[1:-1] > B[2:])]
    coils = m["coils"][m["coils"] <= phi[-1]]
    for p in peaks:
        assert np.min(np.abs(coils - p)) < 2e-3
    assert ripple_amplitude(B.max(), B.min()) == pytest.approx(m["delta"], rel=1e-3)
    assert len(vaft.diagram.toroidal_field_ripple(12).model["coils"]) == 12


def test_wells_sit_inside_the_alpha_star_bands():
    chart = vaft.diagram.ripple_well_formation().model
    p = chart.parameters
    minima = p["minima"]
    assert len(minima) > 0
    # for the multiplicative field the local criterion is alpha* < 1 - eps cos(theta); every drawn
    # well lies inside a shaded band
    for m in minima:
        assert any(a0 - 0.01 <= m <= a1 + 0.01 for a0, a1 in p["bands"]), m
    local = ripple_well_parameter(p["epsilon"], minima, p["q"], p["delta"], p["n_tf"]) / (1 - p["epsilon"] * np.cos(minima))
    assert np.all(local < 1.02)
    # and none in the middle, where the smooth slope wins
    assert not np.any((minima > 1.2) & (minima < 1.9))


def test_the_tip_map_is_bounded_below_one_and_diffuses_above():
    chart = vaft.diagram.stochastic_ripple_orbit().model
    regular = chart.curves["regular"][:, 1]
    stochastic = chart.curves["stochastic"][:, 1]
    assert np.ptp(regular) < 1.0
    assert np.ptp(stochastic) > 5 * np.ptp(regular)


@pytest.mark.parametrize("name", ["trapped_and_passing_orbits", "toroidal_field_ripple", "ripple_well_formation",
                                  "stochastic_ripple_orbit"])
def test_every_ripple_diagram_is_deterministic_and_exported(name):
    fn = getattr(vaft.diagram, name)
    assert fn().tikz == fn().tikz
    assert name in vaft.diagram.__all__
    assert fn().scene.role("note") and not fn(labels=False).scene.role("note")


def test_bad_coil_counts_fail():
    for n in (2, 40, 16.0, True):
        with pytest.raises(ValueError):
            vaft.diagram.toroidal_field_ripple(n)
