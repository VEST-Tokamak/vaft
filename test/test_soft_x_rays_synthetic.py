"""Synthetic soft-X-ray response of a rotating island (#886)."""

from __future__ import annotations

import numpy as np
import pytest

from vaft.process.line_of_sight import Sightlines, build_line_integral_operator, project_emissivity
from vaft.process.magnetic_island import MagneticIslandSpec, island_emissivity, magnetic_island_topology
from vaft.process.soft_x_rays import synthetic_island_soft_x_rays

from test_magnetic_island import _equilibrium

OMEGA = 2.0 * np.pi * 5e3


def _profile(psi_n):
    return (1.0 - psi_n) ** 2


def _fan(phi, n=12):
    """A vertical fan through the plasma from a point above it."""
    ends = np.linspace(0.62, 1.32, n)
    return Sightlines(np.full(n, 0.95), np.full(n, 0.6), ends, np.full(n, -0.6), np.full(n, phi))


@pytest.fixture(scope="module")
def eq():
    return _equilibrium()


def test_zero_amplitude_is_the_steady_axisymmetric_signal(eq):
    chords = _fan(0.0)
    times = np.linspace(0.0, 2e-4, 5)
    result = synthetic_island_soft_x_rays(eq, MagneticIslandSpec(2, 1, 0.03, angular_frequency=OMEGA),
                                          chords, time=times, emissivity_profile=_profile,
                                          amplitude=0.0)
    assert result.brightness.shape == (5, len(chords))
    np.testing.assert_allclose(result.brightness, result.brightness[:1].repeat(5, axis=0), rtol=1e-12)
    background = np.where(result.topology.inside_boundary,
                          _profile(np.clip(result.topology.psi_n, 0, 1)), 0.0)
    expected = project_emissivity(background, build_line_integral_operator(eq.r, eq.z, chords))
    np.testing.assert_allclose(result.brightness[0], expected, rtol=1e-12)
    assert np.all(result.delta_emissivity == 0.0)


def test_a_rigidly_rotating_island_is_periodic_at_its_rotation_frequency(eq):
    chords = _fan(0.0)
    period = 2.0 * np.pi / OMEGA
    times = np.arange(64) * period / 32  # two periods
    result = synthetic_island_soft_x_rays(eq, MagneticIslandSpec(2, 1, 0.04, angular_frequency=OMEGA),
                                          chords, time=times, emissivity_profile=_profile)
    signal = result.brightness - result.brightness.mean(axis=0)
    np.testing.assert_allclose(result.brightness[32:], result.brightness[:32], rtol=1e-10)
    assert np.max(np.abs(signal)) > 1e-4 * np.max(result.brightness)
    spectrum = np.abs(np.fft.rfft(signal, axis=0)).sum(axis=1)
    freqs = np.fft.rfftfreq(times.size, d=times[1] - times[0])
    assert freqs[np.argmax(spectrum[1:]) + 1] == pytest.approx(OMEGA / (2.0 * np.pi))


def test_a_second_toroidal_plane_sees_the_island_shifted_by_n_delta_phi(eq):
    delta_phi = 2.0 * np.pi / 3.0
    spec = MagneticIslandSpec(2, 1, 0.04, angular_frequency=OMEGA)
    both = Sightlines(*(np.concatenate(pair) for pair in zip(
        (_fan(0.0).r1, _fan(0.0).z1, _fan(0.0).r2, _fan(0.0).z2, _fan(0.0).phi),
        (_fan(delta_phi).r1, _fan(delta_phi).z1, _fan(delta_phi).r2, _fan(delta_phi).z2,
         _fan(delta_phi).phi))))
    t0 = 1.3e-5
    result = synthetic_island_soft_x_rays(eq, spec, both, time=t0, emissivity_profile=_profile)
    np.testing.assert_allclose(result.planes, [0.0, delta_phi])
    helicity = result.topology.helicity
    # Independent of the code's formula: Ip and B_phi both along +phi make
    # B_Z < 0 on the outboard midplane, so phi falls as theta* rises:
    # sigma = -sign(Ip Bt).
    assert eq.ip > 0 and eq.bt0 > 0
    assert helicity == -int(np.sign(eq.ip * eq.bt0))
    # xi = m theta* - sigma n phi - omega t: the second plane at t0 is the first
    # plane at t0 + sigma n delta_phi / omega.
    shifted = synthetic_island_soft_x_rays(eq, spec, _fan(0.0),
                                           time=t0 + helicity * delta_phi / OMEGA,
                                           emissivity_profile=_profile)
    np.testing.assert_allclose(result.brightness[0, 12:], shifted.brightness[0], rtol=1e-9)
    assert not np.allclose(result.brightness[0, 12:], result.brightness[0, :12], rtol=1e-6)


def test_brightness_converges_with_the_equilibrium_grid():
    spec = MagneticIslandSpec(2, 1, 0.04, phase=0.3)
    brightness = [synthetic_island_soft_x_rays(_equilibrium(n), spec, _fan(0.0),
                                               emissivity_profile=_profile).brightness[0]
                  for n in (65, 129, 257)]
    coarse = np.max(np.abs(brightness[1] - brightness[0]) / np.abs(brightness[2]))
    fine = np.max(np.abs(brightness[2] - brightness[1]) / np.abs(brightness[2]))
    assert fine < coarse
    assert fine < 1e-2


def test_fields_can_be_dropped_for_long_series(eq):
    spec = MagneticIslandSpec(2, 1, 0.03, angular_frequency=OMEGA)
    kept = synthetic_island_soft_x_rays(eq, spec, _fan(0.0), time=[0.0, 1e-5],
                                        emissivity_profile=_profile)
    lean = synthetic_island_soft_x_rays(eq, spec, _fan(0.0), time=[0.0, 1e-5],
                                        emissivity_profile=_profile, keep_fields=False)
    assert lean.emissivity is None and lean.delta_emissivity is None
    np.testing.assert_array_equal(lean.brightness, kept.brightness)


def test_the_result_carries_per_plane_emissivity(eq):
    result = synthetic_island_soft_x_rays(eq, MagneticIslandSpec(2, 1, 0.03), _fan(0.4),
                                          time=[0.0, 1e-5], emissivity_profile=_profile,
                                          model="island", amplitude=2.0)
    assert result.emissivity.shape == (2, 1, eq.r.size, eq.z.size)
    topology = magnetic_island_topology(eq, MagneticIslandSpec(2, 1, 0.03), phi=0.4)
    eps, _ = island_emissivity(topology, profile=_profile, model="island", amplitude=2.0)
    np.testing.assert_allclose(result.emissivity[0, 0], eps)


def test_vest_sightlines_load_offline_and_cross_the_wall():
    from omas import ODS

    from vaft.machine_mapping.soft_x_rays import sxr_sightlines
    from vaft.machine_mapping.wall import wall

    chords = sxr_sightlines()
    assert len(chords) == 72
    np.testing.assert_allclose(np.unique(np.round(np.degrees(chords.phi), 6)), [0.0, 240.0])
    assert chords.labels[0] == "horizontal:1"
    assert len(sxr_sightlines(arrays=["vertical"])) == 20
    with pytest.raises(ValueError, match="known"):
        sxr_sightlines(arrays=["nonexistent"])
    ods = ODS()
    wall(ods)
    outline = (ods["wall.description_2d.0.limiter.unit.0.outline.r"],
               ods["wall.description_2d.0.limiter.unit.0.outline.z"])
    r = np.linspace(0.05, 0.85, 81)
    z = np.linspace(-1.25, 1.25, 251)
    G = build_line_integral_operator(r, z, chords, domain=outline)
    inside = np.asarray(G.sum(axis=1)).ravel()
    full = np.hypot(chords.r2 - chords.r1, chords.z2 - chords.z1)
    assert np.all(inside > 0.3) and np.all(inside < full)


def test_island_on_a_packaged_vest_equilibrium_through_vest_chords():
    """End to end on a reconstructed VEST equilibrium: the geometry comes from
    the record alone, and the chords from the machine mapping."""
    from omas import ODS

    from vaft.data.resources import sample_geqdsk
    from vaft.machine_mapping.soft_x_rays import sxr_sightlines
    from vaft.machine_mapping.wall import wall
    from vaft.process.equilibrium import as_equilibrium

    eq = as_equilibrium(sample_geqdsk())
    ods = ODS()
    wall(ods)
    outline = (ods["wall.description_2d.0.limiter.unit.0.outline.r"],
               ods["wall.description_2d.0.limiter.unit.0.outline.z"])
    period = 2.0 * np.pi / OMEGA
    result = synthetic_island_soft_x_rays(
        eq, MagneticIslandSpec(3, 1, 0.03, angular_frequency=OMEGA), sxr_sightlines(),
        time=np.arange(8) * period / 8, emissivity_profile=_profile, domain=outline)
    assert result.brightness.shape == (8, 72)
    assert np.all(np.isfinite(result.brightness))
    assert result.topology.q_s == pytest.approx(3.0, abs=1e-9)
    fluctuation = np.ptp(result.brightness, axis=0)
    assert np.count_nonzero(fluctuation > 1e-6 * result.brightness.max()) > 20
