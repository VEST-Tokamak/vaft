"""``is_signal_active`` decides by the trace's own level (issue #463).

The variance test used to divide the variance by itself, so it was 1 for any
real signal and the ``var_ratio_thresh`` parameter never governed a verdict.
Both ratios are now relative to the mean absolute level, and these tests pin
the three cases the issue reproduced plus the parameter's newly live branch.
"""

from __future__ import annotations

import numpy as np
import pytest
from omas import ODS

from vaft.omas.general import classify_shot
from vaft.process import is_signal_active


def _rng():
    return np.random.default_rng(463)


def test_a_flat_trace_with_tiny_noise_is_inactive():
    trace = 1.0 + 1e-6 * _rng().standard_normal(1000)
    assert is_signal_active(trace) is False


def test_white_noise_is_active():
    assert is_signal_active(_rng().standard_normal(1000)) is True


def test_a_ramp_is_active():
    assert is_signal_active(np.linspace(0.0, 1.0, 1000)) is True


def test_the_variance_threshold_is_no_longer_dead():
    # A slow, large-amplitude oscillation: the sample-to-sample change is
    # negligible, so only the variance ratio can call it active.
    time = np.linspace(0.0, 1.0, 200_000)
    trace = 5.0 + 2.0 * np.sin(2.0 * np.pi * time)
    assert is_signal_active(trace) is True
    assert is_signal_active(trace, var_ratio_thresh=0.5) is False


def test_an_all_zero_trace_is_inactive_and_does_not_divide_by_zero():
    with np.errstate(all="raise"):
        assert is_signal_active(np.zeros(100)) is False


@pytest.mark.parametrize("trace", [[], [1.0]])
def test_fewer_than_two_samples_is_never_active(trace):
    assert is_signal_active(np.asarray(trace)) is False


def test_the_verdict_does_not_depend_on_units():
    trace = np.linspace(0.0, 1.0, 500) + 0.05 * _rng().standard_normal(500)
    assert is_signal_active(trace) == is_signal_active(1e6 * trace) == is_signal_active(1e-6 * trace)


def _shot(pressure_active: bool, halpha_active: bool, ip_positive: bool = True) -> ODS:
    ods = ODS(consistency_check=False)
    rng = _rng()
    flat = 1.0 + 1e-6 * rng.standard_normal(500)
    pulse = np.exp(-((np.linspace(0.0, 1.0, 500) - 0.5) / 0.1) ** 2)
    ods["barometry.gauge.0.pressure.data"] = pulse if pressure_active else flat
    ods["spectrometer_uv.channel.0.processed_line.0.intensity.data"] = pulse if halpha_active else flat
    ods["magnetics.ip.0.data"] = (1.0 if ip_positive else -1.0) * pulse
    return ods


@pytest.mark.parametrize(
    "pressure_active, halpha_active, label",
    [(False, False, "Vacuum"), (True, False, "BD failure"), (True, True, "Plasma")],
)
def test_classify_shot_no_longer_swallows_a_type_error(capsys, pressure_active, halpha_active, label):
    """The classifier used to pass ``threshold=`` to a function without that
    argument; the bare ``except`` printed the TypeError and answered
    ``'Vacuum'`` for every shot."""
    assert classify_shot(_shot(pressure_active, halpha_active)) == label
    assert "Error in find_shotclass" not in capsys.readouterr().out
