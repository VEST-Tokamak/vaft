"""Equilibrium-aware phenomena (#1209): prescribed physics drawn on an equilibrium's own surfaces."""

import math

import numpy as np
import pytest

import vaft.diagram
from vaft.diagram import _equilibrium_geometry as eg
from vaft.diagram import _sawtooth as st
from vaft.diagram._mhd_mode import SURFACES, envelope
from vaft.diagram._scene import Label
from vaft.formula.equilibrium import flux_perturbation_from_normal_displacement
from vaft.formula.stability import kadomtsev_mixing_radius


def test_flux_freezing_is_minus_xi_dot_grad_psi():
    assert flux_perturbation_from_normal_displacement(0.01, 2.0) == pytest.approx(-0.02)
    with pytest.raises(ValueError):
        flux_perturbation_from_normal_displacement(0.01, -1.0)


def test_the_mixing_radius_is_sqrt2_r1_for_a_parabolic_inverse_q():
    r = np.linspace(0.0, 1.0, 4001)
    q = 1.0 / (1.0 + 0.25 * (1.0 - (r / 0.4) ** 2))
    assert kadomtsev_mixing_radius(r, q) == pytest.approx(math.sqrt(2) * 0.4, rel=1e-5)
    with pytest.raises(ValueError, match="no q = 1"):
        kadomtsev_mixing_radius(r, 1.2 + r)
    with pytest.raises(ValueError):
        kadomtsev_mixing_radius(r[::-1], q)


def test_the_adapter_normals_are_unit_and_outward():
    g = eg.default_equilibrium()
    assert g.q0 < 1.0 < g.q_profile[-1]
    s = g.surface(0.5)
    np.testing.assert_allclose(np.hypot(s.normal_R, s.normal_Z), 1.0, atol=1e-9)
    # a step along the normal raises psi_N
    step = 1e-3
    assert np.all(g.sfl.psi_norm(s.R + step * s.normal_R, s.Z + step * s.normal_Z) > 0.25)
    # q crosses one where rho_at_q says
    assert float(g.q(g.rho_at_q(1.0))) == pytest.approx(1.0, abs=1e-6)
    # the (rho, theta*) map lands on the surface
    R, Z = g.to_rz(np.full(8, 0.5), np.linspace(0, 2 * math.pi, 8, endpoint=False))
    np.testing.assert_allclose(g.sfl.psi_norm(R, Z), 0.25, atol=2e-3)


def test_zero_amplitude_recovers_the_equilibrium():
    m = vaft.diagram.kink_mode(amplitude=0.0).model
    for s in m["surfaces"]:
        np.testing.assert_allclose(s["R"], s["R0"])
        np.testing.assert_allclose(s["Z"], s["Z0"])


@pytest.mark.parametrize("phase", [0.0, 0.7])
def test_the_displacement_follows_the_harmonic_phase_along_the_normal(phase):
    m = vaft.diagram.kink_mode(m=2, n=1, radial_profile="global", amplitude=0.05, phase=phase).model
    for s in m["surfaces"]:
        F = envelope("global", s["rho"], m=2, rho_s=m["rho_s"])
        expected = 0.05 * m["minor_radius"] * F * np.cos(2 * s["theta_star"] + phase)
        np.testing.assert_allclose(s["xi"], expected, atol=1e-12)
        moved = np.stack([s["R"] - s["R0"], s["Z"] - s["Z0"]], -1)
        normal = np.stack(s["normal"], -1)
        cross = moved[:, 0] * normal[:, 1] - moved[:, 1] * normal[:, 0]
        np.testing.assert_allclose(cross, 0.0, atol=1e-12)  # along the normal
        np.testing.assert_allclose(np.sum(moved * normal, -1), s["xi"], atol=1e-12)


def test_the_internal_envelope_stays_inside_the_resonant_surface():
    m = vaft.diagram.kink_mode().model
    rho_s = m["rho_s"]
    for s in m["surfaces"]:
        peak = np.abs(s["xi"]).max() / (m["amplitude"] * m["minor_radius"])
        if s["rho"] < rho_s - 0.1:
            assert peak > 0.95
        if s["rho"] > rho_s + 0.1:
            assert peak < 0.01
    assert m["axis_shift"][0] > 0.0  # the core moves towards theta* = 0 (outboard) at phase 0


def test_no_surface_inversion_at_the_largest_amplitude():
    for profile in ("internal", "global"):
        m = vaft.diagram.kink_mode(m=1 if profile == "internal" else 2, n=1, radial_profile=profile,
                                   amplitude=0.15).model
        g = m["geometry"]
        previous = None
        for s in m["surfaces"]:
            level = g.sfl.psi_norm(s["R"], s["Z"])  # the displaced points, read against the unperturbed map
            if previous is not None:
                # moving outward in rho never falls back inside the previous displaced surface on average
                assert np.median(level) > previous
            previous = np.median(level)


def test_the_internal_envelope_needs_its_resonant_surface():
    with pytest.raises(ValueError, match="q = 5/1"):
        vaft.diagram.kink_mode(m=5, n=1)


def test_the_precursor_keeps_nested_topology():
    m = vaft.diagram.sawtooth(stage="precursor").model
    assert m["rho_1"] == pytest.approx(eg.default_equilibrium().rho_at_q(1.0))
    assert m["axis_shift"][0] > 0.0


def test_the_reconnection_stage_has_its_1_1_x_and_o_points():
    m = vaft.diagram.sawtooth(stage="reconnection", reconnection_fraction=0.5).model
    # the X-point is where core and outer separatrix touch, on theta* = 0; the O-point opposite
    assert m["shift"] + m["rho_c"] == pytest.approx(m["rho_o"])
    assert m["x_point"] == (pytest.approx(m["rho_o"]), 0.0)
    assert m["o_point"][0] < 0.0 and -m["rho_o"] < m["o_point"][0] < m["shift"] - m["rho_c"]
    assert m["rho_1"] <= m["rho_o"] <= m["rho_mix"]
    # island surfaces are closed crescents around the O-point, none encircles the hot core centre
    from matplotlib.path import Path

    assert m["island"]
    for line in m["island"]:
        assert not Path(line).contains_point((m["shift"], 0.0))
    assert any(Path(line).contains_point(m["o_point"]) for line in m["island"])
    # the reconnected fraction grows the island and shrinks the core
    m2 = vaft.diagram.sawtooth(stage="reconnection", reconnection_fraction=0.8).model
    assert m2["rho_c"] < m["rho_c"] and m2["rho_o"] > m["rho_o"]


def test_the_post_crash_profile_is_flat_inside_the_mixing_radius_and_conserves_energy():
    m = vaft.diagram.sawtooth(stage="post_crash").model
    rho, before, after = m["rho"], m["T_before"], m["T_after"]
    inside = rho <= m["rho_mix"]
    assert np.ptp(after[inside]) == 0.0
    np.testing.assert_allclose(after[~inside], before[~inside])
    assert np.trapezoid(after[inside] * rho[inside], rho[inside]) == pytest.approx(
        np.trapezoid(before[inside] * rho[inside], rho[inside]))
    assert m["rho_1"] < m["rho_mix"]


def test_a_sawtooth_needs_q_equal_one():
    from vaft.process.equilibrium import solovev_example

    high_q = solovev_example("limited", a_parameter=0.6)  # q > 1 everywhere
    with pytest.raises(ValueError, match="q = 1"):
        vaft.diagram.sawtooth(high_q)


def test_the_equilibrium_argument_is_checked():
    with pytest.raises(TypeError):
        vaft.diagram.kink_mode(equilibrium="39915")
    for bad in ({"radial_profile": "tearing"}, {"amplitude": 0.5}, {"m": 0}, {"harmonics": {}}):
        with pytest.raises(ValueError):
            vaft.diagram.kink_mode(**bad)
    for bad in ({"stage": "crash"}, {"reconnection_fraction": 1.0}):
        with pytest.raises(ValueError):
            vaft.diagram.sawtooth(**bad)


@pytest.mark.parametrize("fn, kwargs", [("kink_mode", {}), ("sawtooth", {"stage": "precursor"}),
                                        ("sawtooth", {"stage": "reconnection"}), ("sawtooth", {"stage": "post_crash"})])
def test_every_phenomenon_diagram_is_deterministic_and_exported(fn, kwargs):
    f = getattr(vaft.diagram, fn)
    assert f(**kwargs).tikz == f(**kwargs).tikz
    assert fn in vaft.diagram.__all__
    d = f(**kwargs)
    assert d.model["classification"] in ("synthetic_parameterization", "reduced_model")
    assert d.scene.role("note") and not f(**kwargs, labels=False).scene.role("note")
    assert sum(isinstance(i, Label) for i in f(**kwargs, labels=False).scene.items) < sum(
        isinstance(i, Label) for i in d.scene.items)
