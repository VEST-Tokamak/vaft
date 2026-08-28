"""Flux-loop terminal-voltage persistence in the magnetics IDS (issue #209)."""

from __future__ import annotations

import gzip
import json
from unittest.mock import patch

import numpy as np
import pytest
from omas import ODS, load_omas_json

from vaft.compat import cumtrapz_compat
from vaft.database.raw import SLOW_DT
from vaft.machine_mapping.magnetics import (
    vest_equilibrium_magnetics_channel_definitions,
    vfit_magnetics_dynamic,
    vfit_magnetics_static,
)
from vaft.omas import save as save_ods
from vaft.process.magnetics import (
    vest_flux_loop_flux_from_voltage,
    vest_flux_loop_legacy,
    vest_flux_loop_voltage,
)

SHOT = 39915  # pre-41660 era: MD window indices 6000..8500 (0.24 s .. 0.34 s)
TSTART = 0.26
TEND = 0.34
DT = 4e-5
SAMPLES = 25_000
LIMITER_SHUNT_FIELD = 216

FLUX_LOOP_CHANNELS = [
    channel
    for channel in vest_equilibrium_magnetics_channel_definitions()
    if channel["kind"] == "flux_loop"
]
# Map two loops and leave the third (index 2) without raw data.
PRESENT_LOOPS = (0, 1, 3, 4, 5, 6, 7, 8, 9, 10)
MISSING_LOOP = 2


def _write_raw_dump(path, shot: int, fields: dict[int, np.ndarray]) -> None:
    payload = {
        "shot": shot,
        "fields": {
            str(field): {"data": np.asarray(data, dtype=float).tolist(), "type": "slow"}
            for field, data in fields.items()
        },
    }
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _native_time() -> np.ndarray:
    return np.arange(SAMPLES) * SLOW_DT


def _raw_waveform(index: int) -> np.ndarray:
    """A distinct, smooth waveform per flux loop."""
    time = _native_time()
    return 0.05 * (index + 1) * np.sin(2 * np.pi * 40.0 * time) + 0.01 * (index + 1)


@pytest.fixture(scope="module")
def mapped(tmp_path_factory):
    """Map a synthetic shot with one flux loop deliberately unavailable."""
    tmp_path = tmp_path_factory.mktemp("flux_loop_voltage")
    fields = {
        int(FLUX_LOOP_CHANNELS[index]["field_code"]): _raw_waveform(index)
        for index in PRESENT_LOOPS
    }
    fields[LIMITER_SHUNT_FIELD] = 0.25 + np.where(_native_time() >= 0.26, 0.5, 0.0)
    raw = tmp_path / "raw.json.gz"
    _write_raw_dump(raw, SHOT, fields)

    ods = ODS(consistency_check=False)
    vfit_magnetics_static(ods)
    with (
        patch(
            "vaft.machine_mapping.magnetics.vfit_plasma_current",
            return_value=(np.array([TSTART, TEND]), np.zeros(2)),
        ),
        patch("vaft.machine_mapping.magnetics._map_diamagnetic_flux"),
    ):
        vfit_magnetics_dynamic(ods, SHOT, TSTART, TEND, DT, raw_source=raw)
    return ods, tmp_path


def test_voltage_is_the_calibrated_pre_integration_signal(mapped):
    """`voltage.data` is raw / calibration, not a derivative of processed flux."""
    ods, _ = mapped
    native_time = _native_time()
    window = (native_time >= TSTART) & (native_time < TEND)

    for index in PRESENT_LOOPS:
        calibration = float(FLUX_LOOP_CHANNELS[index]["calibration"])
        expected = _raw_waveform(index)[window] / calibration
        np.testing.assert_allclose(
            np.asarray(ods[f"magnetics.flux_loop.{index}.voltage.data"]), expected
        )
        assert ods[f"magnetics.flux_loop.{index}.voltage.validity"] == 0


def test_voltage_uses_the_native_acquisition_timebase(mapped):
    """Native sampling is preserved; only the analysis window is applied."""
    ods, _ = mapped
    native_time = _native_time()
    expected_time = native_time[(native_time >= TSTART) & (native_time < TEND)]

    voltage_time = np.asarray(ods["magnetics.flux_loop.0.voltage.time"])
    np.testing.assert_allclose(voltage_time, expected_time)
    assert np.allclose(np.diff(voltage_time), SLOW_DT)
    assert voltage_time[0] >= TSTART
    assert voltage_time[-1] < TEND
    # Flux keeps the canonical processed grid; voltage is not forced onto it.
    np.testing.assert_allclose(
        np.asarray(ods["magnetics.flux_loop.0.flux.time"]), np.asarray(ods["magnetics.time"])
    )


def test_integrating_stored_voltage_reproduces_stored_flux(mapped):
    """flux = -integral(voltage dt) - 2*pi*baseline, up to the linear baseline."""
    ods, _ = mapped
    for index in (0, 9):
        voltage_time = np.asarray(ods[f"magnetics.flux_loop.{index}.voltage.time"])
        voltage = np.asarray(ods[f"magnetics.flux_loop.{index}.voltage.data"])
        flux = np.interp(
            voltage_time,
            np.asarray(ods[f"magnetics.flux_loop.{index}.flux.time"]),
            np.asarray(ods[f"magnetics.flux_loop.{index}.flux.data"]),
        )
        integrated = -cumtrapz_compat(voltage, x=voltage_time, initial=0)
        residual = flux - integrated
        # The only difference is the removed linear baseline (plus the
        # integration constant from starting at tstart instead of t=0).
        fit = np.polyval(np.polyfit(voltage_time, residual, 1), voltage_time)
        assert np.max(np.abs(residual - fit)) < 1e-9 * max(1.0, np.max(np.abs(flux)))


def test_flux_processing_is_unchanged_by_the_voltage_split(mapped):
    """Mapped flux still equals the legacy per-radian processing times 2*pi."""
    ods, _ = mapped
    native_time = _native_time()
    magnetics_time = np.asarray(ods["magnetics.time"])
    # Process-layer MD window for this shot era.
    source_time = np.linspace(0.0, 0.99996, SAMPLES)[6000:8501]

    for index in (0, 4):
        calibration = float(FLUX_LOOP_CHANNELS[index]["calibration"])
        legacy = vest_flux_loop_legacy(
            native_time,
            _raw_waveform(index),
            calibration,
            flux_loop_number=index + 1,
        )
        processed = np.interp(source_time, native_time, legacy)
        expected = np.interp(magnetics_time, source_time, processed) * 2 * np.pi
        np.testing.assert_allclose(
            np.asarray(ods[f"magnetics.flux_loop.{index}.flux.data"]), expected
        )


def test_legacy_flux_is_the_documented_voltage_composition():
    """`vest_flux_loop_legacy` == integrate(calibrate(raw)), formula unchanged."""
    time = np.linspace(0.0, 0.1, 2_000)
    raw = np.sin(2 * np.pi * 50.0 * time)
    calibration = -0.0909

    voltage = vest_flux_loop_voltage(raw, calibration)
    np.testing.assert_allclose(voltage, raw / calibration)

    composed = vest_flux_loop_flux_from_voltage(time, voltage, flux_loop_number=1)
    np.testing.assert_array_equal(
        composed, vest_flux_loop_legacy(time, raw, calibration, flux_loop_number=1)
    )
    # Too short for the configured baseline windows, so the baseline is zero
    # and the raw integration formula is exposed directly.
    np.testing.assert_allclose(
        composed, -cumtrapz_compat(raw / calibration, x=time, initial=0) / (2 * np.pi)
    )


def test_missing_channel_keeps_indexing_and_marks_voltage_invalid(mapped):
    """A dropped raw channel yields an empty, invalid voltage without shifting loops."""
    ods, _ = mapped
    assert len(ods["magnetics.flux_loop"]) == len(FLUX_LOOP_CHANNELS)

    assert ods[f"magnetics.flux_loop.{MISSING_LOOP}.voltage.validity"] == -2
    assert np.asarray(ods[f"magnetics.flux_loop.{MISSING_LOOP}.voltage.data"]).size == 0
    assert np.asarray(ods[f"magnetics.flux_loop.{MISSING_LOOP}.voltage.time"]).size == 0
    assert "flux" not in ods[f"magnetics.flux_loop.{MISSING_LOOP}"]

    # Loops after the gap keep their own identity and their own waveform.
    for index in (3, 10):
        assert ods[f"magnetics.flux_loop.{index}.identifier"]
        assert np.asarray(ods[f"magnetics.flux_loop.{index}.flux.data"]).size > 0
        calibration = float(FLUX_LOOP_CHANNELS[index]["calibration"])
        native_time = _native_time()
        window = (native_time >= TSTART) & (native_time < TEND)
        np.testing.assert_allclose(
            np.asarray(ods[f"magnetics.flux_loop.{index}.voltage.data"]),
            _raw_waveform(index)[window] / calibration,
        )


def test_voltage_validity_is_quantity_specific(mapped):
    """Voltage validity tracks acquisition only, and never gates flux."""
    ods, _ = mapped
    for index in range(len(FLUX_LOOP_CHANNELS)):
        has_flux = "flux" in ods[f"magnetics.flux_loop.{index}"]
        validity = ods[f"magnetics.flux_loop.{index}.voltage.validity"]
        assert validity == (0 if index in PRESENT_LOOPS else -2)
        assert has_flux == (index in PRESENT_LOOPS)
        if has_flux:
            # Flux validity stays unset: signal-quality classification is #189,
            # and a valid acquired voltage does not imply a valid flux.
            assert "validity" not in ods[f"magnetics.flux_loop.{index}.flux"]


def test_mirnov_and_shunt_voltage_mappings_are_unaffected(mapped):
    """The new flux-loop node reuses, and does not disturb, the voltage helpers."""
    ods, _ = mapped
    native_time = _native_time()
    limiter = np.where(native_time >= 0.26, 0.5, 0.0)
    np.testing.assert_allclose(np.asarray(ods["magnetics.shunt.0.voltage.data"]), limiter)
    assert ods["magnetics.shunt.0.voltage.validity"] == 0
    for index in (1, 2):
        assert ods[f"magnetics.shunt.{index}.voltage.validity"] == -2

    # No raw Mirnov channels in this dump: probes stay explicitly unavailable.
    assert ods["magnetics.b_field_pol_probe.0.voltage.validity"] == -2
    assert np.asarray(ods["magnetics.b_field_pol_probe.0.voltage.data"]).size == 0


def test_flux_loop_voltage_round_trips_through_omas(mapped):
    ods, tmp_path = mapped
    output = tmp_path / "diagnostics.json"
    save_ods(ods, output)
    reloaded = load_omas_json(str(output), consistency_check=True)
    np.testing.assert_allclose(
        np.asarray(reloaded["magnetics.flux_loop.0.voltage.data"]),
        np.asarray(ods["magnetics.flux_loop.0.voltage.data"]),
    )
    assert reloaded["magnetics.flux_loop.0.voltage.validity"] == 0
