"""Electron-cyclotron resonance relations used by startup diagnostics."""

import numpy as np
import pytest

from vaft.formula.ecr import electron_cyclotron_resonance_field


def test_245_ghz_fundamental_resonance_is_about_875_mt():
    field = electron_cyclotron_resonance_field(2.45e9)
    assert field == pytest.approx(0.0875234756, rel=1e-9)


def test_harmonic_number_reduces_the_required_field():
    fundamental = electron_cyclotron_resonance_field(2.45e9)
    second = electron_cyclotron_resonance_field(2.45e9, harmonic=2)
    assert second == pytest.approx(fundamental / 2.0)


def test_frequency_arrays_are_supported():
    frequency = np.array([2.45e9, 4.90e9])
    field = electron_cyclotron_resonance_field(frequency)
    np.testing.assert_allclose(field[1], 2.0 * field[0])


@pytest.mark.parametrize("frequency", [0.0, -1.0, np.inf, np.nan])
def test_frequency_must_be_finite_and_positive(frequency):
    with pytest.raises(ValueError, match="frequency_hz"):
        electron_cyclotron_resonance_field(frequency)


@pytest.mark.parametrize("harmonic", [0, -1, 1.5, np.nan])
def test_harmonic_must_be_a_positive_integer(harmonic):
    with pytest.raises(ValueError, match="harmonic"):
        electron_cyclotron_resonance_field(2.45e9, harmonic=harmonic)
