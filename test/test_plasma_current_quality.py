"""The plasma-current Rogowski fails only on categorical evidence (#1373)."""

from __future__ import annotations

import numpy as np
import pytest

from vaft.validation.plasma_current import PlasmaCurrentQualityConfig, assess_plasma_rogowski
from vaft.validation.validity import VALIDITY_INVALID, VALIDITY_VALID

TIME = np.linspace(0.0, 1.0, 25000, endpoint=False)


def _discharge(peak: float = 220e3, tail: float = -14e3) -> np.ndarray:
    """A 20 ms discharge at 0.30 s followed by a slowly decaying induced current."""
    current = peak * np.exp(-(((TIME - 0.30) / 0.006) ** 2))
    after = TIME > 0.32
    current[after] += tail * np.exp(-(TIME[after] - 0.32) / 0.15)
    return current + 300.0 * np.random.default_rng(0).standard_normal(TIME.size)


def test_a_healthy_discharge_with_its_induced_tail_is_valid():
    quality = assess_plasma_rogowski(TIME, _discharge())
    assert quality.validity == VALIDITY_VALID
    assert quality.metrics["off_discharge_p2p"] < 20e3


def test_an_offset_record_is_judged_relative_to_its_own_reference_level():
    quality = assess_plasma_rogowski(TIME, _discharge() + 75e3)
    assert quality.validity == VALIDITY_VALID
    assert quality.metrics["reference_level"] == pytest.approx(75e3, abs=1e3)


def test_the_45781_square_wave_fails_the_whole_record():
    """Field 109 toggling between about +-100 kA for the whole second (45766-45817)."""
    square = 1e5 * np.sign(np.sin(2 * np.pi * 4.0 * TIME + 0.3))
    quality = assess_plasma_rogowski(TIME, square + _discharge(peak=80e3, tail=0.0))
    assert quality.validity == VALIDITY_INVALID
    assert quality.metrics["off_discharge_p2p"] > 150e3
    assert "outside the discharge window" in quality.reason


def test_the_limit_sits_in_the_population_gap():
    """Healthy shots reach ~53 kA off the discharge, the fault 170-260 kA (#1373 scan)."""
    limit = PlasmaCurrentQualityConfig().max_off_discharge_p2p
    assert 53e3 < limit < 170e3


def test_a_record_that_misses_the_quiet_windows_is_not_condemned():
    window = (TIME >= 0.26) & (TIME <= 0.36)
    quality = assess_plasma_rogowski(TIME[window], _discharge()[window])
    assert quality.validity == VALIDITY_VALID
    assert "not judged" in quality.reason


def test_non_finite_samples_are_ignored():
    current = _discharge()
    current[::97] = np.nan
    assert assess_plasma_rogowski(TIME, current).validity == VALIDITY_VALID
