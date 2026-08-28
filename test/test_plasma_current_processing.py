"""Behavioral tests for `vaft.machine_mapping.magnetics.vfit_plasma_current`.

These exercise the real function (not just config resolution, which is
covered by `test_vest_yaml_boundaries.py`) against synthetic, offline raw
dumps -- no live SQL/HSDS access. Additional eras (FL10 windowed
compensation, disabled mode) land with their own PRs once the Python
dispatch logic for `processing.reference.mode` exists.
"""

import gzip
import json

import numpy as np
import pytest

from vaft.machine_mapping.magnetics import vfit_plasma_current

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
