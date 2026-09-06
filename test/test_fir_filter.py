"""Linear-phase FIR filtering (#64).

`vaft.process.signal_processing` carried only Butterworth IIR filters, so a
workflow that wanted linear phase -- VEST's fluctuation spectrograms -- designed
its own `firwin` taps in a notebook. These are the properties that make the FIR
route worth having beside the IIR one.
"""

from __future__ import annotations

import numpy as np
import pytest

from vaft.process.signal_processing import (
    butterworth_bandpass,
    fir_filter,
    fir_filter_coefficients,
)


FS = 1.0e6


@pytest.fixture
def two_tone():
    """1 kHz and 100 kHz, equal duration, so a split can be checked by band."""
    time = np.arange(0.0, 0.02, 1.0 / FS)
    slow = np.sin(2.0 * np.pi * 1.0e3 * time)
    fast = 0.5 * np.sin(2.0 * np.pi * 1.0e5 * time)
    return time, slow, fast, slow + fast


def test_the_taps_are_symmetric_and_odd_which_is_what_linear_phase_means():
    taps = fir_filter_coefficients(2.0e4, FS)
    assert taps.size % 2 == 1  # Type I
    assert np.allclose(taps, taps[::-1])
    # The default is a one-millisecond impulse response at this rate.
    assert taps.size == 1001


def test_an_even_request_is_made_odd_so_a_high_pass_is_possible():
    """A Type II filter is forced to zero at Nyquist, which kills a high-pass."""
    assert fir_filter_coefficients(2.0e4, FS, numtaps=200).size == 201
    # DC gain is the tap sum: a low-pass passes it, a high-pass rejects it.
    low = fir_filter_coefficients(2.0e4, FS, numtaps=101)
    high = fir_filter_coefficients(2.0e4, FS, kind="highpass", numtaps=101)
    assert abs(low.sum()) == pytest.approx(1.0, abs=1.0e-6)
    assert abs(high.sum()) < 0.05


def test_a_split_by_band_puts_each_tone_where_it_belongs(two_tone):
    _, slow, fast, mixed = two_tone
    low = fir_filter(mixed, 2.0e4, FS, kind="lowpass")
    high = fir_filter(mixed, 2.0e4, FS, kind="highpass")

    interior = slice(2000, -2000)  # away from the filter's edge transients
    assert np.allclose(low[interior], slow[interior], atol=0.02)
    assert np.allclose(high[interior], fast[interior], atol=0.02)
    # The two halves add back up to what went in.
    assert np.max(np.abs((low + high - mixed)[interior])) < 0.02


def test_bandstop_rejects_the_band_bandpass_keeps(two_tone):
    """firwin scales each design to its own passband, so the two are not
    arithmetic complements; what they must agree on is which tone survives."""
    _, slow, fast, mixed = two_tone
    # The window method sets the transition width by the tap count alone, so a
    # 500-5000 Hz band at 1 MHz needs far more than the one-millisecond default
    # (whose transition is about 1 kHz -- as wide as the band's lower edge).
    taps = 4001
    band = fir_filter(mixed, (5.0e2, 5.0e3), FS, kind="bandpass", numtaps=taps)
    stop = fir_filter(mixed, (5.0e2, 5.0e3), FS, kind="bandstop", numtaps=taps)
    interior = slice(taps, -taps)  # 12 000 of the 20 000 samples, clear of the edges

    # The 1 kHz tone is inside the band and the 100 kHz tone is not, so each
    # filter keeps exactly one of them.
    assert np.std(band[interior]) == pytest.approx(np.std(slow[interior]), rel=0.05)
    assert np.std(stop[interior]) == pytest.approx(np.std(fast[interior]), rel=0.05)


def test_the_causal_form_delays_every_frequency_by_the_same_samples(two_tone):
    """The point of FIR: one integer delay, removable exactly."""
    _, _, _, mixed = two_tone
    numtaps = 201
    causal = fir_filter(mixed, 2.0e4, FS, numtaps=numtaps, zero_phase=False)
    phase = fir_filter(mixed, 2.0e4, FS, numtaps=numtaps, zero_phase=True)

    delay = (numtaps - 1) // 2
    aligned = np.roll(causal, -delay)
    interior = slice(2 * numtaps, -2 * numtaps)
    assert np.allclose(aligned[interior], phase[interior], atol=0.02)

    # An IIR filter's phase distortion is frequency-dependent, so no single
    # shift aligns it: that is the reason this module now offers both.
    iir = butterworth_bandpass(mixed, 5.0e2, 5.0e3, FS, zero_phase=False)
    shifts = [np.max(np.abs((np.roll(iir, -n) - phase)[interior])) for n in range(0, 60, 5)]
    assert min(shifts) > 0.05


def test_the_band_edges_are_checked_against_the_sample_rate():
    with pytest.raises(ValueError, match="kind must be one of"):
        fir_filter_coefficients(1.0e3, FS, kind="notch")
    with pytest.raises(ValueError, match="takes 2 band edge"):
        fir_filter_coefficients(1.0e3, FS, kind="bandpass")
    with pytest.raises(ValueError, match="0 < f <"):
        fir_filter_coefficients(FS, FS)  # above Nyquist
    with pytest.raises(ValueError, match="increasing"):
        fir_filter_coefficients((5.0e3, 5.0e2), FS, kind="bandpass")


def test_a_stack_of_waveforms_filters_along_the_last_axis(two_tone):
    _, _, _, mixed = two_tone
    stack = np.vstack([mixed, 2.0 * mixed])
    filtered = fir_filter(stack, 2.0e4, FS)
    assert filtered.shape == stack.shape
    assert np.allclose(filtered[1], 2.0 * filtered[0], atol=1.0e-9)
