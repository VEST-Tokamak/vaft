"""Fourier / spectral representation of closed flux surfaces (#945).

The convention is uniform arc length from the largest R, counter-clockwise,
with the phase canonicalized on the fitted series. These tests hold it to
what #945 asks: circle content exact, parity for symmetric surfaces, a
translation moving only m = 0, the asymmetric families flipping sign under a
mirror, independence from the contour's start point and sampling,
truncation convergence, a radially ordered sequence, and one definition
shared with the LCFS representation.
"""

from __future__ import annotations

import numpy as np
import pytest

from vaft.data.equilibrium import FourierSurface, MillerSurface
from vaft.process.equilibrium import (
    derive_boundary_representation,
    evaluate_fourier_surface,
    evaluate_miller,
    fit_fourier_surface,
    fit_fourier_surface_sequence,
    solovev_example,
)

THETA = np.linspace(0.0, 2*np.pi, 600, endpoint=False)
SHAPED = MillerSurface(0.22, 0.9, -0.03, 1.7, 0.32, 0.1)


def _shaped(theta=THETA):
    return evaluate_miller(SHAPED, theta)


def _asymmetric():
    r, z = _shaped()
    return r, z + 0.5*(r - 0.9)**2


def test_circle_is_a_single_harmonic():
    s = fit_fourier_surface((1.0 + 0.3*np.cos(THETA), 0.1 + 0.3*np.sin(THETA)), modes=4).surface
    np.testing.assert_allclose([s.reference_r, s.reference_z], [1.0, 0.1], atol=1e-12)
    assert s.r_cos[1] == pytest.approx(0.3, rel=1e-4) and s.z_sin[1] == pytest.approx(0.3, rel=1e-4)
    assert np.max(np.abs(np.r_[s.r_cos[2:], s.z_sin[2:], s.r_sin, s.z_cos[1:]])) < 1e-10


def test_ellipse_keeps_the_up_down_parity_and_is_dominated_by_m1():
    s = fit_fourier_surface((1.0 + 0.3*np.cos(THETA), 0.5*np.sin(THETA)), modes=8).surface
    assert np.max(np.abs(s.r_sin)) < 1e-12 and np.max(np.abs(s.z_cos[1:])) < 1e-12
    assert s.z_sin[1] > s.r_cos[1] > 0 and abs(s.r_cos[3]) < 0.1*s.r_cos[1]


def test_symmetric_surface_has_no_asymmetric_coefficients():
    s = fit_fourier_surface(_shaped(), modes=8).surface
    assert np.max(np.abs(s.r_sin)) < 1e-12 and np.max(np.abs(s.z_cos[1:])) < 1e-12


def test_translation_moves_only_the_m0_terms():
    r, z = _shaped()
    a = fit_fourier_surface((r, z), modes=8).surface
    b = fit_fourier_surface((r + 0.1, z - 0.2), modes=8).surface
    assert b.reference_r - a.reference_r == pytest.approx(0.1)
    assert b.reference_z - a.reference_z == pytest.approx(-0.2)
    for name in ("r_cos", "r_sin", "z_cos", "z_sin"):
        np.testing.assert_allclose(getattr(b, name)[1:], getattr(a, name)[1:], atol=1e-12)


def test_asymmetric_families_flip_sign_under_a_mirror():
    r, z = _asymmetric()
    up = fit_fourier_surface((r, z), modes=8).surface
    down = fit_fourier_surface((r, -z), modes=8).surface
    assert np.max(np.abs(up.r_sin)) > 1e-3                       # the asymmetry is really there
    np.testing.assert_allclose(down.r_sin, -up.r_sin, atol=1e-12)
    np.testing.assert_allclose(down.z_cos, -up.z_cos, atol=1e-12)
    np.testing.assert_allclose(down.r_cos, up.r_cos, atol=1e-12)
    np.testing.assert_allclose(down.z_sin, up.z_sin, atol=1e-12)


def test_start_point_orientation_and_sampling_do_not_change_the_representation():
    reference = fit_fourier_surface(_asymmetric(), modes=8).surface
    theta = np.linspace(0.0, 2*np.pi, 1777, endpoint=False) + 0.7
    r, z = evaluate_miller(SHAPED, theta)
    moved = fit_fourier_surface((r[::-1], (z + 0.5*(r - 0.9)**2)[::-1]), modes=8).surface
    for name in ("r_cos", "r_sin", "z_cos", "z_sin"):
        np.testing.assert_allclose(getattr(moved, name), getattr(reference, name), atol=2e-5)


def test_reconstruction_error_falls_with_mode_count():
    contour = evaluate_miller(MillerSurface(0.22, 0.9, 0.0, 1.8, 0.5, 0.2), THETA)
    errors = [fit_fourier_surface(contour, modes=m).normalized_rms_error for m in (2, 4, 8, 12)]
    assert all(later < earlier for earlier, later in zip(errors, errors[1:]))
    assert errors[-1] < 3e-3
    rejected = fit_fourier_surface(contour, modes=2)
    assert not rejected.accepted and "2 modes" in rejected.reason


def test_evaluate_inverts_the_fit():
    fit = fit_fourier_surface(_asymmetric(), modes=10)
    r, z = evaluate_fourier_surface(fit.surface, np.linspace(0, 2*np.pi, 64, endpoint=False))
    assert r.shape == (64,) and fit.accepted
    assert np.argmax(evaluate_fourier_surface(fit.surface, np.linspace(-0.01, 0.01, 2001))[0]) == pytest.approx(1000, abs=1)


def test_record_validates_its_convention():
    with pytest.raises(ValueError, match="common length"):
        FourierSurface(np.zeros(3), np.zeros(2), np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError, match="m = 0 sine"):
        FourierSurface(np.zeros(3), np.array([1.0, 0, 0]), np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError, match="angle convention"):
        FourierSurface(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), angle_convention="geometric")
    with pytest.raises(ValueError, match="at least 1"):
        fit_fourier_surface(_shaped(), modes=0)


def test_sequence_is_radially_ordered_and_continuous():
    eq = solovev_example("single_null")
    levels = [0.8, 0.2, 0.5, 0.35, 0.65, 0.9]
    sequence = fit_fourier_surface_sequence(eq, levels, modes=8)
    assert sequence.skipped == ()
    radial, r1 = sequence.coefficient("r_cos", 1)
    np.testing.assert_allclose(radial, sorted(levels))
    assert np.all(np.diff(r1) > 0)                               # minor radius grows outward
    _, asym = sequence.coefficient("r_sin", 1)
    assert np.all(np.abs(asym) > 1e-4)                           # the single null is asymmetric throughout
    assert all(f.accepted for f in sequence.fits)


def test_sequence_records_levels_without_a_closed_surface():
    sequence = fit_fourier_surface_sequence(solovev_example("limited"), [0.5, 1.4], modes=6)
    assert sequence.skipped == (1.4,) and len(sequence.fits) == 1


def test_boundary_representation_uses_the_same_definition():
    eq = solovev_example("double_null")
    rep = derive_boundary_representation(eq, fourier_modes=12)
    surface = fit_fourier_surface(eq.lcfs, modes=12).surface
    for name in ("r_cos", "r_sin", "z_cos", "z_sin"):
        np.testing.assert_array_equal(rep.fourier_coefficients[name], getattr(surface, name))
