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


def test_a_short_record_is_not_decimated_into_an_alias():
    source, tone, target = _short_tone()
    out = _decimate(source, tone, target)
    # No target sample may carry the folded tone as if it were data.
    assert not np.isfinite(out).any()


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
    # Two passes of a Hamming FIR: passband ripple ~0.02 dB each, ~0.5 % in all.
    assert np.max(np.abs(out[long_interior] - expected)) < 5e-3


def _amplitude(values, time, frequency):
    design = np.column_stack((np.sin(2 * np.pi * frequency * time), np.cos(2 * np.pi * frequency * time)))
    return float(np.hypot(*np.linalg.lstsq(design, values, rcond=None)[0]))


def test_kept_unfiltered_the_tone_folds_to_5khz_at_full_amplitude():
    """The defect itself, still reachable by name: with ``unfiltered="keep"``
    the stored samples are a 5 kHz sinusoid of amplitude 1."""
    source, tone, target = _short_tone()
    out = _decimate(source, tone, target, unfiltered="keep")
    assert np.isfinite(out).all()
    assert abs(_amplitude(out, target, 5_000.0) - 1.0) < 1e-6


def test_a_short_chirp_is_blanked_and_kept_unfiltered_it_folds():
    """Linear chirp 1 -> 100 kHz over a 900-sample run: everything above the
    12.5 kHz target Nyquist would fold back. Kept, the target samples equal
    np.interp of the raw chirp, so they carry the folded sweep; by default none
    is stored."""
    source = np.arange(SHORT) / FS_FAST
    rate = (100_000.0 - 1_000.0) / source[-1]
    chirp = np.sin(2 * np.pi * (1_000.0 * source + 0.5 * rate * source**2))
    target = np.arange(0.0, source[-1], 1.0 / FS_SLOW)
    kept = _decimate(source, chirp, target, unfiltered="keep")
    np.testing.assert_array_equal(kept, np.interp(target, source, chirp))
    assert np.std(kept[target > 0.5 * source[-1]]) > 0.3  # the fold is not small
    assert not np.isfinite(_decimate(source, chirp, target)).any()

def test_a_rate_cut_without_short_runs_is_unchanged_by_the_option():
    source = np.arange(25_000) / FS_FAST
    tone = np.sin(2 * np.pi * 1_000.0 * source) + np.sin(2 * np.pi * 40_000.0 * source)
    target = np.arange(0.0, source[-1], 1.0 / FS_SLOW)
    np.testing.assert_array_equal(
        resample_to_time(source, tone, target), resample_to_time(source, tone, target, unfiltered="keep")
    )


def test_no_filter_no_change():
    """Equal-rate alignment and upsampling never filter, so nothing is blanked:
    the result is still bit-for-bit np.interp, short record or not."""
    source = np.arange(50) / FS_SLOW
    values = np.sin(2 * np.pi * 3_000.0 * source)
    for target in (source + 0.3 / FS_SLOW, np.arange(0.0, source[-1], 0.25 / FS_SLOW)):
        np.testing.assert_array_equal(resample_to_time(source, values, target), np.interp(target, source, values))


def test_the_mask_and_the_nans_agree():
    source = np.arange(25_000) / FS_FAST
    signal = np.full(source.size, np.nan)
    signal[1_000:6_000] = 1.0
    signal[10_000:10_900] = 1.0
    target = np.arange(0.0, source[-1], 1.0 / FS_SLOW)
    out, mask = _decimate(source, signal, target, return_filter_mask=True)
    kept, _ = _decimate(source, signal, target, return_filter_mask=True, unfiltered="keep")
    fraction = resample_to_time(source, mask.astype(float), target, anti_alias=False)
    np.testing.assert_array_equal(np.isnan(out), np.isnan(kept) | (fraction < 1.0))
    np.testing.assert_array_equal(out[fraction >= 1.0], kept[fraction >= 1.0])


def test_unknown_option_is_refused():
    source, tone, target = _short_tone()
    with pytest.raises(ValueError, match="unfiltered must be"):
        resample_to_time(source, tone, target, unfiltered="drop")


# --- per caller -------------------------------------------------------------
# Every production caller that takes no filter mask now gets the NaN default;
# the one opt-out must be a caller that reads the mask itself.

import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _resample_calls():
    for path in sorted((_ROOT / "vaft").rglob("*.py")) + sorted((_ROOT / "workflow").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
                if name == "resample_to_time":
                    yield path.relative_to(_ROOT).as_posix(), node


def _keyword(node, name):
    for keyword in node.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def test_only_a_caller_that_reads_the_mask_keeps_unfiltered_samples():
    keeps = []
    for path, node in _resample_calls():
        value = _keyword(node, "unfiltered")
        if value is None:
            continue
        assert isinstance(value, ast.Constant), f"{path}:{node.lineno}: unfiltered must be a literal"
        if value.value == "keep":
            mask = _keyword(node, "return_filter_mask")
            assert isinstance(mask, ast.Constant) and mask.value is True, (
                f"{path}:{node.lineno}: unfiltered='keep' without return_filter_mask=True "
                "stores possibly aliased samples with nothing to flag them"
            )
            keeps.append(path)
    assert keeps == ["vaft/machine_mapping/langmuir_probes.py"]


def test_the_exposed_callers_take_the_default():
    """The B3.12 callers: none of them opts out, so none can store an alias."""
    exposed = {
        "vaft/machine_mapping/barometry.py",
        "vaft/machine_mapping/ec_launchers.py",
        "vaft/machine_mapping/spectrometer_uv.py",
        "vaft/omas/process_wrapper.py",
        "vaft/process/fluctuation.py",
        "vaft/plot/mirnov.py",
        "vaft/process/signal_processing.py",
    }
    seen = set()
    for path, node in _resample_calls():
        if path in exposed:
            seen.add(path)
            assert _keyword(node, "unfiltered") is None, f"{path}:{node.lineno}"
    assert seen == exposed


def test_barometry_does_not_store_an_alias_from_a_short_record(monkeypatch):
    from omas import ODS

    from vaft.machine_mapping import barometry

    n = SHORT
    source = np.arange(n) / FS_FAST
    volts = 5.0 + 0.5 * np.sin(2 * np.pi * 20_000.0 * source)
    monkeypatch.setattr(barometry, "_safe_vest_load", lambda shot, field, raw_source=None: (source, volts))
    ods = ODS()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        barometry.vfit_barometry_dynamic(
            ods, 41672, 0.0, source[-1], 1.0 / FS_SLOW, target_time=np.arange(0.0, source[-1], 1.0 / FS_SLOW)
        )
    data = np.asarray(ods["barometry.gauge.0.pressure.data"])
    assert data.size >= 80
    assert not np.isfinite(data).any()
