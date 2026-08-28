"""Behavioral tests for `vaft.machine_mapping.magnetics.vfit_plasma_current`.

These exercise the real function (not just config resolution, which is
covered by `test_vest_yaml_boundaries.py`) against synthetic, offline raw
dumps -- no live SQL/HSDS access.
"""

import gzip
import json

import numpy as np
import pytest

from vaft.machine_mapping.magnetics import (
    _apply_fl10_windowed_compensation,
    vfit_plasma_current,
)

PLASMA_CURRENT_FIELD = 109
FLUX_REFERENCE_FIELD = 25


def _write_raw_dump(path, shot, fields):
    payload = {
        "shot": shot,
        "fields": {
            str(field): {
                "data": values.tolist() if hasattr(values, "tolist") else values,
                "type": "slow",
            }
            for field, values in fields.items()
        },
    }
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _quadratic_waveform(n: int) -> np.ndarray:
    """A pure quadratic so a local linear baseline fit is sensitive to
    *where* the fitting window sits -- distinct baseline windows must yield
    distinct fitted (and therefore subtracted) baselines."""
    index = np.arange(n, dtype=float)
    return 1.0e-4 * (index - 5000.0) ** 2


def _write_synthetic_shot(tmp_path, shot: int, n: int = 9000):
    raw_ip = _quadratic_waveform(n)
    raw_flux = np.zeros(n, dtype=float)
    source = tmp_path / f"raw_{shot}.json.gz"
    _write_raw_dump(source, shot, {PLASMA_CURRENT_FIELD: raw_ip, FLUX_REFERENCE_FIELD: raw_flux})
    return source


def test_baseline_window_boundary_43761_changes_the_fitted_baseline(tmp_path):
    """Shots 43760 and 43761 fall in different `processing.baseline` eras
    (analysis window 7250:8750 vs 6000:7251). Feeding both the same
    quadratic raw waveform must produce different processed output, since
    a linear fit through different windows of a quadratic differs."""
    source_pre = _write_synthetic_shot(tmp_path, 43760)
    source_post = _write_synthetic_shot(tmp_path, 43761)

    time_pre, ip_pre = vfit_plasma_current(43760, raw_source=source_pre)
    time_post, ip_post = vfit_plasma_current(43761, raw_source=source_post)

    np.testing.assert_allclose(time_pre, time_post)
    assert not np.allclose(ip_pre, ip_post)


@pytest.mark.parametrize(
    ("shot", "analysis_start", "analysis_end", "lookback"),
    [
        (43760, 7250, 8750, 500),
        (43761, 6000, 7251, 500),
    ],
)
def test_baseline_subtraction_matches_the_documented_algorithm(
    tmp_path, shot, analysis_start, analysis_end, lookback
):
    """Pin the exact baseline-subtraction + sign-convention algorithm
    (raw calibration -> linear baseline fit over `xBase` -> subtract ->
    subtract FL10 reference -> apply sign) so a future refactor of
    `vfit_plasma_current` cannot silently change it."""
    n = 9000
    raw_ip = _quadratic_waveform(n)
    raw_flux = np.zeros(n, dtype=float)
    source = _write_synthetic_shot(tmp_path, shot, n=n)

    time, ip = vfit_plasma_current(shot, raw_source=source)

    calibration_factor = 1.0e-5  # shot >= 42851 era
    calibrated_ip = raw_ip / calibration_factor
    x_time = np.arange(analysis_start, analysis_end)
    x_base = np.arange(x_time[0] - lookback, x_time[0] + 1)
    x_base = x_base[(x_base >= 0) & (x_base < n)]

    ip_shot = calibrated_ip - np.polyval(
        np.polyfit(time[x_base], calibrated_ip[x_base], 1), time
    )
    ip_ref = raw_flux * 11.0 / 5.0e-4  # flux_gain / mutual_inductance (post-17455 era)
    ip_ref = ip_ref - np.polyval(np.polyfit(time[x_base], ip_ref[x_base], 1), time)
    expected = (ip_shot - ip_ref) * -1.0  # sign convention flips at shot 20259

    np.testing.assert_allclose(ip, expected)


def test_fl10_disabled_mode_ignores_the_flux_reference_entirely(tmp_path):
    """Shot >= 47117 (#195): Ip == Ip_shot after sign only. The mapper must
    not even require a flux-reference field to be present."""
    shot = 47117
    n = 9000
    raw_ip = _quadratic_waveform(n)
    source = tmp_path / "raw.json.gz"
    _write_raw_dump(source, shot, {PLASMA_CURRENT_FIELD: raw_ip})

    time, ip = vfit_plasma_current(shot, raw_source=source)

    calibration_factor = 1.0e-5
    calibrated_ip = raw_ip / calibration_factor
    analysis_start, lookback = 6000, 500
    x_time = np.arange(analysis_start, 7251)
    x_base = np.arange(x_time[0] - lookback, x_time[0] + 1)
    x_base = x_base[(x_base >= 0) & (x_base < n)]
    ip_shot = calibrated_ip - np.polyval(
        np.polyfit(time[x_base], calibrated_ip[x_base], 1), time
    )
    expected = ip_shot * -1.0  # sign convention (shot >= 20259)

    np.testing.assert_allclose(ip, expected)


def test_fl10_windowed_mode_subtracts_only_inside_the_compensation_window(tmp_path):
    """Shots 46403-47116 (#195): compensation must not leak outside
    0.26 <= t <= 0.36 s. Compare against a zero-FL10-reference control,
    which is mathematically equivalent to no compensation at all (mode
    'disabled') for the same underlying Ip_shot."""
    shot = 46403
    n = 12000  # 12000 * 4e-5 s = 0.48 s, covers the 0.26-0.36 s window
    raw_ip = np.zeros(n, dtype=float)  # flat Ip_shot after baseline removal
    zero_fl10 = np.zeros(n, dtype=float)
    nonzero_fl10 = np.sin(np.linspace(0.0, 40.0, n)) + 5.0

    source_zero = tmp_path / "raw_zero.json.gz"
    _write_raw_dump(source_zero, shot, {PLASMA_CURRENT_FIELD: raw_ip, FLUX_REFERENCE_FIELD: zero_fl10})
    source_nonzero = tmp_path / "raw_nonzero.json.gz"
    _write_raw_dump(source_nonzero, shot, {PLASMA_CURRENT_FIELD: raw_ip, FLUX_REFERENCE_FIELD: nonzero_fl10})

    time, ip_zero_ref = vfit_plasma_current(shot, raw_source=source_zero)
    _, ip_nonzero_ref = vfit_plasma_current(shot, raw_source=source_nonzero)

    mask = (time >= 0.26) & (time <= 0.36)
    assert mask.any() and (~mask).any()

    np.testing.assert_allclose(ip_nonzero_ref[~mask], ip_zero_ref[~mask])
    assert not np.allclose(ip_nonzero_ref[mask], ip_zero_ref[mask])


def test_fl10_degenerate_offset_uses_the_documented_vaft_convention(tmp_path):
    """Pin the resolved interpretation of the legacy MATLAB
    `ipRef = ipRef - polyval(polyfit(time2(1), ipRef(175), 1), time2)`
    expression: VAFT treats it as `ip_ref -= ip_ref[174]` (0-based), a
    documented compatibility convention (#195), not a proof of exact MATLAB
    numerical equivalence. Uses decimate_factor=1 and smooth_span=1 so the
    result is exactly analytically predictable."""
    shot = 46403
    n = 300
    dt = 4e-5
    raw_fl10 = np.arange(n, dtype=float)  # distinctive, monotonic values
    source = tmp_path / "raw.json.gz"
    _write_raw_dump(source, shot, {FLUX_REFERENCE_FIELD: raw_fl10})

    time = np.arange(n, dtype=float) * dt
    ip_shot = np.zeros(n, dtype=float)
    reference_config = {
        "mutual_inductance": 1.0,
        "fl10": {
            "field": FLUX_REFERENCE_FIELD,
            "time_offset_s": 0.0,
            "decimate_factor": 1,
            "gain_numerator": 1.0,
            "smooth_span": 1,
            "subtract_window": [0.0, 1.0],
            "reference_offset_index": 175,
        },
    }

    compensated = _apply_fl10_windowed_compensation(
        shot, time, ip_shot, reference_config, source
    )

    expected_offset = raw_fl10[174]  # 1-based index 175 -> 0-based 174
    expected = ip_shot - (raw_fl10 - expected_offset)
    np.testing.assert_allclose(compensated, expected)
