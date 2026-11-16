"""Declared sample rates checked against the acquired timebase (audit gaps G9/G10)."""

from __future__ import annotations

import gzip
import json

import numpy as np
import pytest

from vaft.process.acquisition import (
    AcquisitionInconsistencyError,
    check_declared_sample_rate,
    summarize_sample_rate_checks,
)
from vaft.process.magnetics import (
    VestMagneticsProcessingConfig,
    mirnov_preprocess_signal,
    vest_equilibrium_magnetics_detailed,
)


def _time(rate, n=1000):
    return np.arange(n) / float(rate)


def test_a_matching_rate_is_consistent():
    check = check_declared_sample_rate(_time(25_000.0), 25_000.0, cutoff_hz=250.0)
    assert check.status == "consistent"
    assert check.observed_hz == pytest.approx(25_000.0)
    assert check.fatal is False


def test_a_different_rate_is_a_deviation():
    check = check_declared_sample_rate(_time(20_000.0), 25_000.0)
    assert check.status == "deviation"
    assert check.relative_deviation == pytest.approx(0.2)
    assert "differs from the declared" in check.reason


def test_irregular_sampling_is_a_deviation():
    time = _time(25_000.0)
    time[500:] += 1.0e-3  # a gap
    check = check_declared_sample_rate(time, 25_000.0)
    assert check.status == "deviation"
    assert "irregular" in check.reason


@pytest.mark.parametrize("declared", [0.0, -1.0, float("nan"), None, "fast", True])
def test_an_unusable_declaration_is_invalid(declared):
    assert check_declared_sample_rate(_time(25_000.0), declared).status == "invalid"


def test_a_non_increasing_timebase_is_invalid_but_not_fatal():
    time = _time(25_000.0)
    time[10] = time[9]
    check = check_declared_sample_rate(time, 25_000.0)
    assert (check.status, check.fatal) == ("invalid", False)


def test_a_non_increasing_timebase_still_meets_the_nyquist_test():
    time = _time(4_000.0)
    time[10] = time[9]
    check = check_declared_sample_rate(time, 250_000.0, cutoff_hz=2_500.0)
    assert (check.status, check.fatal) == ("invalid", True)


@pytest.mark.parametrize("cutoff", [float("nan"), -5.0, [250.0, float("nan")]])
def test_an_unusable_cutoff_is_invalid_and_serializable(cutoff):
    check = check_declared_sample_rate(_time(25_000.0), 25_000.0, cutoff_hz=cutoff)
    assert (check.status, check.fatal) == ("invalid", False)
    json.dumps(check.as_dict(), allow_nan=False)


@pytest.mark.parametrize("time", [np.array([]), np.array([0.1]), np.array([np.nan, np.nan])])
def test_too_short_a_record_is_not_verifiable(time):
    assert check_declared_sample_rate(time, 25_000.0).status == "not_verifiable"


def test_a_cutoff_above_the_observed_nyquist_is_fatal():
    # Declared 250 kHz, acquired at 4 kHz: a 2.5 kHz low pass cannot exist.
    check = check_declared_sample_rate(_time(4_000.0), 250_000.0, cutoff_hz=[100.0, 2_500.0])
    assert (check.status, check.fatal) == ("invalid", True)
    assert check.cutoff_hz == 2_500.0
    assert "Nyquist" in check.reason


def test_the_summary_counts_and_keeps_only_flagged_channels():
    checks = {
        "a": check_declared_sample_rate(_time(25_000.0), 25_000.0),
        "b": check_declared_sample_rate(_time(20_000.0), 25_000.0),
    }
    summary = summarize_sample_rate_checks(checks)
    assert summary["status"] == "deviation"
    assert summary["counts"] == {"consistent": 1, "deviation": 1, "invalid": 0, "not_verifiable": 0}
    assert list(summary["channels"]) == ["b"]
    assert summary["observed_hz_range"] == [pytest.approx(20_000.0), pytest.approx(25_000.0)]
    json.dumps(summary, allow_nan=False)


# --- the Mirnov band-pass no longer skips silently ----------------------------------------


def test_mirnov_records_a_skipped_band_edge_and_keeps_its_output():
    rng = np.random.default_rng(0)
    data = rng.normal(size=4000)
    with pytest.warns(RuntimeWarning, match="low pass not applied"):
        plain = mirnov_preprocess_signal(data, sample_rate=25_000.0)
    report: dict = {}
    reported = mirnov_preprocess_signal(data, sample_rate=25_000.0, report=report)
    np.testing.assert_array_equal(plain, reported)
    assert report["high_pass"]["status"] == "applied"
    assert report["low_pass"]["status"] == "skipped"  # 90 kHz above the 12.5 kHz Nyquist
    assert "outside (0, 12500)" in report["low_pass"]["reason"]
    none: dict = {}
    mirnov_preprocess_signal(data, sample_rate=25_000.0, high_pass_cutoff=None, report=none)
    assert none["high_pass"]["status"] == "not_requested"
    empty: dict = {}
    mirnov_preprocess_signal(np.array([]), report=empty)
    assert empty["low_pass"]["status"] == "not_applied"


# --- the equilibrium probe chain ---------------------------------------------------------


_PROBE = {"field_code": 1, "calibration": 1.0, "kind": "b_field_pol_probe", "name": "probe_1"}
_CONFIG = VestMagneticsProcessingConfig(window_override=(0, 500, 100))


def _loader(rate, n):
    def load(_shot, _field):
        time = 0.26 + np.arange(n) / rate
        return time, np.sin(2 * np.pi * 50.0 * time)

    return load


def test_the_probe_chain_records_its_acquisition_and_keeps_its_output():
    loader = _loader(200_000.0, 20_000)
    plain = vest_equilibrium_magnetics_detailed(39915, [_PROBE], loader, config=_CONFIG)
    report: dict = {}
    reported = vest_equilibrium_magnetics_detailed(
        39915, [_PROBE], loader, config=_CONFIG, acquisition_report=report
    )
    np.testing.assert_array_equal(plain.probes[0], reported.probes[0])
    assert report["probe_1"].status == "deviation"  # declared 250 kHz, acquired at 200 kHz
    assert report["probe_1"].observed_hz == pytest.approx(200_000.0)


def test_the_probe_chain_refuses_a_cutoff_the_record_cannot_carry():
    with pytest.raises(AcquisitionInconsistencyError, match="observed Nyquist"):
        vest_equilibrium_magnetics_detailed(39915, [_PROBE], _loader(4_000.0, 4_000), config=_CONFIG)


def test_a_fatal_acquisition_loses_only_its_component(tmp_path, monkeypatch):
    """The stage leaves magnetics out with the reason and keeps its siblings."""
    from vaft.omas import vest_upstream

    shot = 43017
    raw = tmp_path / "raw.json.gz"
    with gzip.open(raw, "wt", encoding="utf-8") as handle:
        json.dump({"shot": shot, "fields": {"12": {"data": np.linspace(1.0, 2.0, 200).tolist(), "type": "slow"}}}, handle)
    static_path = tmp_path / "static.json.gz"
    static, manifest = vest_upstream.build_static_ods(vest_upstream.machine_era_for_shot(shot).name)
    vest_upstream.write_stage_product(static, manifest, output=static_path, metadata=tmp_path / "static.json")

    def refuse(*_args, **_kwargs):
        raise AcquisitionInconsistencyError("probe low pass cannot be applied: synthetic")

    monkeypatch.setattr(vest_upstream, "vfit_magnetics_dynamic", refuse)
    _ods, diagnostics = vest_upstream.build_diagnostics_ods(
        shot=shot, raw_source=raw, static_ods=static_path, tstart=0.0, tend=0.005, dt=4e-5
    )
    assert diagnostics["channel_status"]["magnetics"]["status"] == "unavailable"
    assert "synthetic" in diagnostics["channel_status"]["magnetics"]["reason"]
    assert diagnostics["channel_status"]["barometry"]["status"] == "success"
