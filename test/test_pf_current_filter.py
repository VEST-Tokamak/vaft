"""The PF coil-current filter as a named composition (#2162 B3.5).

``vaft.machine_mapping.pf_active._legacy_pf_filter`` designed a SciPy
``firwin`` low-pass (default **Hamming** window, tap count and cutoff from
vest.yaml), applied it forward-backward -- forward only on a record too short
for ``filtfilt`` -- and then smoothed with a 10-sample
``scipy.ndimage.uniform_filter1d``.  The frozen copy below is that function
verbatim from develop 75b154e31 (after #2135); the composition that replaces
it is held to it bit for bit, and its phase behaviour is pinned against
closed-form expectations rather than against either implementation.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage, signal

from vaft.machine_mapping.utils import resolve_vest_diagnostic
from vaft.process.signal_processing import apply_fir_filter


def _frozen_legacy_pf_filter(values, processing):
    filter_config = processing["filter"]
    taps = signal.firwin(
        int(filter_config["taps"]),
        float(filter_config["cutoff_frequency"]),
        pass_zero="lowpass",
        fs=float(filter_config["sample_rate"]),
    )
    filtered = apply_fir_filter(taps, values, short_record="causal")
    return ndimage.uniform_filter1d(filtered, size=min(int(filter_config["smoothing_window"]), values.size))


def _processing(shot=39915):
    return resolve_vest_diagnostic(shot, "pf_active")["processing"]


def _filter(values, processing):
    from vaft.process.signal_processing import pf_current_filter

    config = processing["filter"]
    return pf_current_filter(
        values,
        sample_rate=config["sample_rate"],
        cutoff_hz=config["cutoff_frequency"],
        numtaps=config["taps"],
        smoothing_window=config["smoothing_window"],
    )


def test_vest_yaml_pf_filter_values_are_the_ones_this_file_assumes():
    config = _processing()["filter"]
    assert config == {
        "sample_rate": 25000.0,
        "cutoff_frequency": 2500.0,
        "taps": 251,
        "smoothing_window": 10,
    }


@pytest.mark.parametrize("length", [20_000, 754, 753, 200, 5])
def test_pf_filter_matches_frozen_reference(length):
    """754 = 3 * 251 + 1 is the shortest record filtfilt takes; 753 and below
    take the causal fallback; 5 is shorter than the smoothing window."""
    rng = np.random.default_rng(length)
    time = np.arange(length) / 25_000.0
    values = 1e3 * np.sin(2 * np.pi * 50.0 * time) + 30.0 * rng.standard_normal(length)
    processing = _processing()
    np.testing.assert_array_equal(_filter(values, processing), _frozen_legacy_pf_filter(values, processing))


def test_the_mapper_routes_through_the_named_composition(monkeypatch):
    from vaft.machine_mapping import pf_active
    from vaft.process import signal_processing

    calls = []
    real = signal_processing.pf_current_filter

    def spy(values, **kwargs):
        calls.append(kwargs)
        return real(values, **kwargs)

    monkeypatch.setattr(pf_active, "pf_current_filter", spy)
    values = np.random.default_rng(0).standard_normal(2_000)
    processing = _processing()
    np.testing.assert_array_equal(pf_active._pf_filter(values, processing), _frozen_legacy_pf_filter(values, processing))
    assert calls == [{"sample_rate": 25000.0, "cutoff_hz": 2500.0, "numtaps": 251, "smoothing_window": 10}]
    assert not hasattr(pf_active, "_legacy_pf_filter")


def test_coefficients_are_firwin_hamming_not_the_hann_default():
    from vaft.process.signal_processing import fir_filter_coefficients

    hamming = signal.firwin(251, 2500.0, pass_zero="lowpass", fs=25000.0)
    np.testing.assert_array_equal(
        fir_filter_coefficients(2500.0, 25000.0, kind="lowpass", numtaps=251, window="hamming"), hamming
    )
    hann = fir_filter_coefficients(2500.0, 25000.0, kind="lowpass", numtaps=251)
    assert np.max(np.abs(hann - hamming)) > 1e-4  # the default would have been a different filter


def test_even_taps_are_refused_rather_than_silently_lengthened():
    from vaft.process.signal_processing import pf_current_filter

    with pytest.raises(ValueError, match="numtaps must be odd"):
        pf_current_filter(np.zeros(2_000), sample_rate=25e3, cutoff_hz=2.5e3, numtaps=250, smoothing_window=10)


def _pf(values, **overrides):
    from vaft.process.signal_processing import pf_current_filter

    kwargs = {"sample_rate": 25e3, "cutoff_hz": 2.5e3, "numtaps": 251, "smoothing_window": 10}
    kwargs.update(overrides)
    return pf_current_filter(values, **kwargs)


def test_full_record_fir_is_zero_phase_and_even_window_lags_half_a_sample():
    n = 20_000
    centre = slice(2_000, n - 2_000)
    # A ramp passes a linear-phase low-pass unchanged (zero-phase, unit DC gain);
    # the 10-sample average then reports the ramp half a sample late.
    ramp = np.arange(n, dtype=float)
    out = _pf(ramp)
    np.testing.assert_allclose(out[centre], ramp[centre] - 0.5, rtol=0, atol=1e-8)
    # With an odd window the average is centred and nothing moves.
    np.testing.assert_allclose(_pf(ramp, smoothing_window=9)[centre], ramp[centre], rtol=0, atol=1e-8)


def test_full_record_tone_phase():
    """A 200 Hz tone: zero phase through the FIR (filtfilt), then the 10-sample
    average's response exp(-i w/2) * sin(5 w)/(10 sin(w/2)) at w = 2 pi f / fs."""
    fs, f, n = 25e3, 200.0, 50_000
    t = np.arange(n) / fs
    out = _pf(np.sin(2 * np.pi * f * t))
    centre = slice(5_000, n - 5_000)
    design = np.column_stack((np.sin(2 * np.pi * f * t[centre]), np.cos(2 * np.pi * f * t[centre])))
    a, b = np.linalg.lstsq(design, out[centre], rcond=None)[0]
    w = 2 * np.pi * f / fs
    # filtfilt squares the FIR magnitude; at 200 Hz it is 1 to ~1e-6.
    fir_gain = np.abs(signal.freqz(signal.firwin(251, 2500.0, fs=fs), worN=[f], fs=fs)[1][0]) ** 2
    expected_gain = fir_gain * np.sin(5 * w) / (10 * np.sin(w / 2))
    assert abs(np.hypot(a, b) - expected_gain) < 1e-9
    assert abs(np.arctan2(b, a) - (-w / 2)) < 1e-9  # phase lag of half a sample only


def test_short_record_takes_the_causal_fallback_with_its_group_delay():
    """Below 3 * 251 + 1 = 754 samples the FIR runs forward only: an impulse
    comes out (251 - 1) / 2 = 125 samples late, i.e. 5 ms at 25 kHz."""
    impulse = np.zeros(753)
    impulse[100] = 1.0
    fir_only = _pf(impulse, smoothing_window=1)
    assert int(np.argmax(fir_only)) == 100 + 125
    long_impulse = np.zeros(754)
    long_impulse[377] = 1.0
    assert int(np.argmax(_pf(long_impulse, smoothing_window=1))) == 377  # filtfilt: no shift
