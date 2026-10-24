"""Declared toroidal Fourier conventions and their explicit conversion (#2163)."""
from __future__ import annotations

import numpy as np
import pytest

from vaft.data.fourier import (
    COIL_SECTOR_DECOMPOSITION,
    CONVENTIONS,
    GPEC_HARMONIC,
    HELICAL_HARMONIC,
    FourierConvention,
    UnresolvedConventionError,
    convert_harmonic,
    fourier_conversion,
    reconstruct_real_field,
)
from vaft.process.coils_non_axisymmetric import phased_sector_currents, toroidal_mode_decomposition

PHI = np.linspace(0.0, 2 * np.pi, 8, endpoint=False)
FINE = np.linspace(0.0, 2 * np.pi, 64, endpoint=False)


@pytest.mark.parametrize("n, phase, amplitude", [(1, 0.4, 3.0), (2, -1.1, 0.7), (3, 2.5, 1.0), (-1, 0.9, 2.0)])
def test_sector_decomposition_to_gpec_reproduces_the_phase_audit_coefficient_and_the_field(n, phase, amplitude):
    from vaft.process.perturbation import toroidal_phase_audit

    currents = phased_sector_currents(PHI, n, phase, amplitude)
    c_sector = toroidal_mode_decomposition(PHI, currents, [n])[n]
    c_gpec = convert_harmonic(c_sector, COIL_SECTOR_DECOMPOSITION, GPEC_HARMONIC)

    # Same real field under either convention.
    expected = amplitude * np.cos(n * FINE + phase)
    np.testing.assert_allclose(reconstruct_real_field(c_sector, FINE, n, COIL_SECTOR_DECOMPOSITION), expected, atol=1e-12)
    np.testing.assert_allclose(reconstruct_real_field(c_gpec, FINE, n, GPEC_HARMONIC), expected, atol=1e-12)

    # And the coefficient the phase audit extracts from that field is the converted one.
    sampled = expected[None, :, None]
    audit = toroidal_phase_audit(np.array([[c_gpec]]), sampled, FINE, n_tor=n)
    assert audit.preferred == "stored"
    assert audit.stored_relative_norm < 1e-12
    # The unconverted coefficient is the conjugate hypothesis, at half the size: it must lose.
    raw = toroidal_phase_audit(np.array([[c_sector]]), sampled, FINE, n_tor=n)
    assert raw.stored_relative_norm > 0.1


def test_gpec_and_helical_kernels_are_the_conjugate_doubled_sector_coefficient():
    conjugate, factor = fourier_conversion(COIL_SECTOR_DECOMPOSITION, GPEC_HARMONIC)
    assert conjugate and factor == 2.0
    assert fourier_conversion(GPEC_HARMONIC, GPEC_HARMONIC) == (False, 1.0)


@pytest.mark.parametrize("source", list(CONVENTIONS.values()))
@pytest.mark.parametrize("target", list(CONVENTIONS.values()))
def test_every_supported_conversion_round_trips(source, target):
    try:
        forward = convert_harmonic(0.3 - 0.8j, source, target)
    except UnresolvedConventionError:
        assert source.poloidal_angle != target.poloidal_angle
        return
    assert convert_harmonic(forward, target, source) == pytest.approx(0.3 - 0.8j)


def test_the_axisymmetric_harmonic_is_the_mean():
    for convention in (COIL_SECTOR_DECOMPOSITION, GPEC_HARMONIC):
        values = np.full(8, 2.5)
        c0 = toroidal_mode_decomposition(PHI, values, [0])[0]
        c = convert_harmonic(c0, COIL_SECTOR_DECOMPOSITION, convention)
        np.testing.assert_allclose(reconstruct_real_field(c, FINE, 0, convention), 2.5)


def test_a_helical_harmonic_is_never_converted_to_a_toroidal_one():
    with pytest.raises(UnresolvedConventionError, match="poloidal_angle"):
        convert_harmonic(1j, COIL_SECTOR_DECOMPOSITION, HELICAL_HARMONIC)


@pytest.mark.parametrize("missing", ["kernel_sign", "coefficient_scale", "toroidal_angle", "phase_origin"])
def test_an_undeclared_field_refuses_instead_of_guessing(missing):
    partial = FourierConvention(**{**GPEC_HARMONIC.__dict__, "name": "partial", missing: None})
    with pytest.raises(UnresolvedConventionError, match=missing):
        convert_harmonic(1 + 1j, partial, GPEC_HARMONIC)


def test_a_different_phase_origin_or_orientation_is_not_bridged():
    shifted = FourierConvention(**{**GPEC_HARMONIC.__dict__, "name": "shifted", "phase_origin": "port 1"})
    clockwise = FourierConvention(**{**GPEC_HARMONIC.__dict__, "name": "cw", "toroidal_angle": "clockwise"})
    for other in (shifted, clockwise):
        with pytest.raises(UnresolvedConventionError):
            convert_harmonic(1 + 1j, GPEC_HARMONIC, other)


def test_a_complex_field_is_never_conjugated():
    complex_field = FourierConvention(**{**GPEC_HARMONIC.__dict__, "name": "complex", "real_field": False})
    with pytest.raises(UnresolvedConventionError, match="complex field"):
        convert_harmonic(1 + 1j, complex_field, COIL_SECTOR_DECOMPOSITION)


def test_conversion_does_not_modify_its_input():
    held = np.array([1 + 2j, 3 - 1j])
    out = convert_harmonic(held, COIL_SECTOR_DECOMPOSITION, GPEC_HARMONIC)
    np.testing.assert_array_equal(held, [1 + 2j, 3 - 1j])
    np.testing.assert_allclose(out, 2 * np.conj(held))
