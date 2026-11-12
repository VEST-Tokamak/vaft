"""Summing a straight-field-line spectrum back to real space (Tutorial 05, #2278).

The sign convention was fixed against GPEC's own real-space normal field
``b_n_fun`` on the Tutorial 05 VEST case (MID, n=1, 2026-11-12):
``Re sum_m b_n[m] exp(i (m theta_H - n q (theta_H - theta_PEST)))`` matched it to
2e-4, 1e-2 and 7e-3 (relative L2) at psi_N 0.546, 0.878 and 0.991, while the
three other sign choices were off by order one.  These tests pin the
convention on synthetic surfaces, where the answer is known in closed form.
"""

import numpy as np
import pytest

from vaft.process.equilibrium import pest_angle_from_jacobian_angle
from vaft.process.perturbation import spectral_field_on_mesh

THETA = np.linspace(0.0, 1.0, 257)


def _shaped_r(r0=0.4, a=0.15):
    return r0 + a * np.cos(2.0 * np.pi * THETA)


def _single(m, value, psi=(0.0, 1.0)):
    psi = np.asarray(psi, dtype=float)
    return psi, np.array([m]), np.full((psi.size, 1), value, dtype=complex)


def test_a_pest_run_is_a_plain_fourier_sum():
    psi, m, c = _single(2, 0.3 - 0.4j)
    r = np.tile(_shaped_r(), (1, 1))
    for phi in (0.0, 0.7):
        got = spectral_field_on_mesh(psi, m, c, [0.5], THETA, r, [1.7], n_tor=1, phi_rad=phi, jacobian="pest")
        expected = np.real((0.3 - 0.4j) * np.exp(1j * (2 * 2 * np.pi * THETA - phi)))
        np.testing.assert_allclose(got[0], expected, atol=1e-12)


def test_a_resonant_harmonic_sits_at_its_pest_phase_whatever_the_jacobian():
    """With m = n q the Hamada-PEST toroidal shift cancels the poloidal one exactly."""
    m, n = 3, 1
    psi, mm, c = _single(m, 1.0 + 0.5j)
    r = _shaped_r()[None, :]
    got = spectral_field_on_mesh(psi, mm, c, [0.5], THETA, r, [m / n], n_tor=n, phi_rad=0.4, jacobian="hamada")
    theta_pest = pest_angle_from_jacobian_angle(THETA, r[0], "hamada")
    expected = np.real((1.0 + 0.5j) * np.exp(1j * (m * theta_pest - n * 0.4)))
    np.testing.assert_allclose(got[0], expected, atol=1e-10)


def test_a_non_resonant_harmonic_is_shifted_toroidally_on_a_shaped_surface():
    psi, mm, c = _single(3, 1.0)
    r = _shaped_r()[None, :]
    hamada = spectral_field_on_mesh(psi, mm, c, [0.5], THETA, r, [4.5], n_tor=1, jacobian="hamada")
    pest = spectral_field_on_mesh(psi, mm, c, [0.5], THETA, r, [4.5], n_tor=1, jacobian="pest")
    assert np.max(np.abs(hamada - pest)) > 0.1


def test_the_spectrum_is_interpolated_radially():
    psi = np.array([0.0, 1.0])
    c = np.array([[0.0], [2.0]], dtype=complex)
    r = np.tile(np.full(THETA.size, 0.4), (1, 1))
    got = spectral_field_on_mesh(psi, [0], c, [0.25], THETA, r, [2.0], n_tor=1, jacobian="pest")
    np.testing.assert_allclose(got[0], 0.5, atol=1e-12)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"coefficients": np.zeros((3, 1))}, "coefficients"),
        ({"mesh_psi_norm": [1.5]}, "outside"),
        ({"q": [1.0, 2.0]}, "mesh_r"),
    ],
)
def test_inconsistent_input_is_refused(kwargs, message):
    arguments = dict(psi_norm=[0.0, 1.0], m_pol=[1], coefficients=np.zeros((2, 1)), mesh_psi_norm=[0.5],
                     mesh_theta_norm=THETA, mesh_r=np.full((1, THETA.size), 0.4), q=[2.0])
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        spectral_field_on_mesh(**arguments, n_tor=1)
