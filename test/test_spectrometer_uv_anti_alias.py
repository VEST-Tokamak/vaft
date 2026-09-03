"""The filterscope mapper must not alias the fast DAQ onto the policy grid.

Issue #425, motivated by #409.  The versatile filterscope (channel 2, fields
138-144) is acquired at 250 kHz while ``spectrometer_uv``'s policy grid is
25 kHz, so writing it out is a 10x decimation.  The slow-DAQ lines on channels
0 and 1 (fields 101 and 214) are already at the grid rate, and the assertion
that *those* are untouched is what makes the change safe for the packaged
samples and the compact reference fixture, which read channel 0 only.
"""

from __future__ import annotations

import gzip
import json

import numpy as np
import pytest
from omas import ODS

from vaft.machine_mapping.spectrometer_uv import SIGNALS, vfit_filterscope

SHOT = 43017
FS_FAST = 250_000.0
FS_SLOW = 25_000.0
T0 = 0.26
SPAN = 0.1
TARGET_TIME = np.arange(T0, T0 + SPAN, 1.0 / FS_SLOW)

FAST_FIELDS = {field for field, channel, *_ in SIGNALS if channel == 2}
SLOW_FIELDS = {field for field, channel, *_ in SIGNALS if channel != 2}


def _write_dump(path, payload_by_field):
    """Write a raw archive whose entries carry explicit ``t0``/``dt``.

    The self-describing form keeps the shot-era trigger correction and the
    mapper's 0.24/0.26 legacy shift out of the picture, so the test measures
    the resampling and nothing else.
    """
    payload = {"shot": SHOT, "fields": {}}
    for field, (dt, values) in payload_by_field.items():
        payload["fields"][str(field)] = {
            "data": np.asarray(values, dtype=float).tolist(),
            "type": "slow" if dt > 1e-5 else "fast",
            "t0": T0,
            "dt": dt,
        }
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _amplitude_at(values, frequency, fs):
    spectrum = np.abs(np.fft.rfft(values)) * 2.0 / values.size
    freqs = np.fft.rfftfreq(values.size, 1.0 / fs)
    return float(spectrum[np.argmin(np.abs(freqs - frequency))])


@pytest.fixture
def raw_dump(tmp_path):
    # Built exactly as vaft.database.raw reconstructs it (t0 + arange * dt), so
    # the bit-for-bit assertion below compares the same floats the mapper saw.
    fast_time = T0 + np.arange(int(SPAN * FS_FAST)) * (1.0 / FS_FAST)
    slow_time = T0 + np.arange(int(SPAN * FS_SLOW)) * (1.0 / FS_SLOW)
    # 40 kHz folds to |40 - 2*25| = 10 kHz on the target grid; 1 kHz must survive.
    fast_values = np.sin(2 * np.pi * 40_000.0 * fast_time) + np.sin(
        2 * np.pi * 1_000.0 * fast_time
    )
    slow_values = np.sin(2 * np.pi * 1_000.0 * slow_time) + 0.25 * np.cos(
        2 * np.pi * 4_000.0 * slow_time
    )
    payload = {field: (1.0 / FS_FAST, fast_values) for field in FAST_FIELDS}
    payload.update({field: (1.0 / FS_SLOW, slow_values) for field in SLOW_FIELDS})
    path = tmp_path / "raw.json.gz"
    _write_dump(path, payload)
    return path, fast_time, fast_values, slow_time, slow_values


def _map(raw_path):
    ods = ODS()
    vfit_filterscope(
        ods, SHOT, T0, T0 + SPAN, 1.0 / FS_SLOW,
        raw_source=str(raw_path), target_time=TARGET_TIME,
    )
    return ods


def test_fast_channel_alias_is_rejected(raw_dump):
    raw_path, _, _, _, _ = raw_dump
    ods = _map(raw_path)
    stored = np.asarray(
        ods["spectrometer_uv.channel.2.processed_line.0.intensity.data"]
    )
    assert stored.size == TARGET_TIME.size
    interior = stored[400:-400]
    assert _amplitude_at(interior, 10_000.0, FS_SLOW) < 1e-2
    # The physically relevant low-frequency content is preserved (the mapper
    # negates the raw trace, which does not change the spectrum's magnitude).
    assert _amplitude_at(interior, 1_000.0, FS_SLOW) == pytest.approx(1.0, rel=5e-2)


def test_bare_interpolation_would_have_aliased(raw_dump):
    _, fast_time, fast_values, _, _ = raw_dump
    aliased = -np.interp(TARGET_TIME, fast_time, fast_values)
    assert _amplitude_at(aliased[400:-400], 10_000.0, FS_SLOW) > 0.5


def test_slow_channels_are_bit_for_bit_unchanged(raw_dump):
    raw_path, _, _, slow_time, slow_values = raw_dump
    ods = _map(raw_path)
    expected = -np.interp(TARGET_TIME, slow_time, slow_values)
    for path in (
        "spectrometer_uv.channel.0.processed_line.0.intensity.data",
        "spectrometer_uv.channel.1.processed_line.0.intensity.data",
    ):
        assert np.array_equal(np.asarray(ods[path]), expected), path


def test_every_declared_line_is_written(raw_dump):
    raw_path, _, _, _, _ = raw_dump
    ods = _map(raw_path)
    for _, channel, line, label, wavelength in SIGNALS:
        prefix = f"spectrometer_uv.channel.{channel}.processed_line.{line}"
        assert ods[f"{prefix}.label"] == label
        assert ods[f"{prefix}.wavelength_central"] == pytest.approx(wavelength)
        assert np.asarray(ods[f"{prefix}.intensity.data"]).size == TARGET_TIME.size
    assert np.allclose(np.asarray(ods["spectrometer_uv.time"]), TARGET_TIME)
