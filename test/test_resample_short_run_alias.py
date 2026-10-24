"""A short run must not be decimated unfiltered (#2162 B3.12, a scientific correction).

``anti_alias_filter`` cannot run ``filtfilt`` over a record or finite run shorter
than ``3 * numtaps + 1`` samples -- 994 for the 331-tap filter of a 250 kHz ->
25 kHz cut -- and hands such a run back unfiltered, with a ``RuntimeWarning``
and a ``False`` in the filter mask (#893, #915).  ``resample_to_time`` then
interpolated it onto the slow grid anyway, so a 20 kHz tone in that run folds
to ``|20 - 25| = 5 kHz`` at full amplitude: a number inside the band a reader
trusts, stored beside correctly band-limited samples as their equal.  Only a
caller that asked for the mask and acted on it (Langmuir, #915) was protected.

Ground truth is analytic: the alias frequency and amplitude follow from the
sampling theorem, independent of the implementation.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from vaft.process.signal_processing import resample_to_time

FS_FAST = 250_000.0
FS_SLOW = 25_000.0
SHORT = 900  # < 994 = 3 * 331 + 1, the shortest run the 250 -> 25 kHz filter accepts


def _short_tone(n=SHORT, frequency=20_000.0):
    source = np.arange(n) / FS_FAST
    target = np.arange(0.0, source[-1], 1.0 / FS_SLOW)
    return source, np.sin(2.0 * np.pi * frequency * source), target


def _decimate(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return resample_to_time(*args, **kwargs)


def test_the_short_run_is_long_enough_to_show_the_alias():
    source, _, target = _short_tone()
    assert target.size >= 80  # 3.6 ms of slow grid: a 5 kHz alias is 18 cycles


@pytest.mark.xfail(strict=True, reason="B3.12: short runs are decimated unfiltered (fixed in the next commit)")
def test_a_short_record_is_not_decimated_into_an_alias():
    source, tone, target = _short_tone()
    out = _decimate(source, tone, target)
    # No target sample may carry the folded tone as if it were data.
    assert not np.isfinite(out).any()


@pytest.mark.xfail(strict=True, reason="B3.12: short runs are decimated unfiltered (fixed in the next commit)")
def test_a_short_finite_run_beside_a_long_one_is_blanked_and_the_long_one_kept():
    source = np.arange(25_000) / FS_FAST  # 0.1 s
    signal = np.full(source.size, np.nan)
    signal[1_000:6_000] = np.sin(2 * np.pi * 1_000.0 * source[1_000:6_000])  # filterable, in band
    signal[10_000:10_900] = np.sin(2 * np.pi * 20_000.0 * source[10_000:10_900])  # short, out of band
    target = np.arange(0.0, source[-1], 1.0 / FS_SLOW)
    out = _decimate(source, signal, target)
    short = (target >= source[10_000]) & (target <= source[10_899])
    assert not np.isfinite(out[short]).any()
    long_interior = (target >= source[2_000]) & (target <= source[5_000])
    assert np.isfinite(out[long_interior]).all()
    expected = np.sin(2 * np.pi * 1_000.0 * target[long_interior])
    assert np.max(np.abs(out[long_interior] - expected)) < 1e-3


def _amplitude(values, time, frequency):
    design = np.column_stack((np.sin(2 * np.pi * frequency * time), np.cos(2 * np.pi * frequency * time)))
    return float(np.hypot(*np.linalg.lstsq(design, values, rcond=None)[0]))


def test_passed_through_unfiltered_the_tone_folds_to_5khz_at_full_amplitude():
    """The defect itself: the stored samples are a 5 kHz sinusoid of amplitude 1."""
    source, tone, target = _short_tone()
    out = _decimate(source, tone, target)
    assert np.isfinite(out).all()
    assert abs(_amplitude(out, target, 5_000.0) - 1.0) < 1e-6
