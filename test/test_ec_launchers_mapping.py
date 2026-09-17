"""VEST 6 kW 2.45 GHz ECH power mapping into `ec_launchers` (issue #165).

The forward (field 27) and reflected (field 28) detector voltages go through
the legacy log-detector calibration, whose exponent explodes on negative-going
pickup spikes. The mapper masks voltages outside the configured valid input
range to NaN rather than publishing tens of kilowatts -- or 1e17 kW -- from a
6 kW source.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest
from omas import ODS

from vaft.database.raw import RawSignalUnavailableError
from vaft.machine_mapping import ec_launchers as ec_module
from vaft.machine_mapping.ec_launchers import ec_launchers, vest_ec_power
from vaft.machine_mapping.utils import resolve_vest_diagnostic


SAMPLES = Path(__file__).resolve().parents[1] / "vaft" / "data" / "samples"
SLOW_DT = 4e-5


def _raw_dump(shot: int) -> Path:
    path = SAMPLES / str(shot) / "source" / f"vest_{shot}_daq_raw.json.gz"
    if not path.exists():
        pytest.skip(f"archived raw dump for shot {shot} is not packaged")
    return path


def _window(time, values, start, end):
    selected = (time >= start) & (time < end)
    return values[selected]


def _synthetic_loader(forward, reflected):
    time = np.arange(len(forward), dtype=float) * SLOW_DT
    waveforms = {27: np.asarray(forward, dtype=float), 28: np.asarray(reflected, dtype=float)}

    def load(shot, field, raw_source=None):
        if field not in waveforms:
            return None
        return time, waveforms[field]

    return load


def test_valid_input_range_is_configuration_not_a_code_literal():
    for key in ("ech_6kw_forward", "ech_6kw_reflected"):
        config = resolve_vest_diagnostic(39915, key)
        low, high = config["processing"]["valid_input_range"]
        assert (low, high) == (0.5, 2.5)
        assert config["calibration"]["type"] == "logarithmic_power"
    assert resolve_vest_diagnostic(39915, "ech_6kw_forward")["source"]["field"] == 27
    assert resolve_vest_diagnostic(39915, "ech_6kw_reflected")["source"]["field"] == 28


def test_calibration_masks_out_of_range_voltages_and_nets_forward_minus_reflected(monkeypatch):
    forward = [1.01745, 0.4, 2.6, 1.01745, 1.01745 - 0.2763]
    reflected = [2.13, 2.13, 2.13, 0.3, 1.01745]
    monkeypatch.setattr(ec_module, "_safe_vest_load", _synthetic_loader(forward, reflected))

    power = vest_ec_power(39915)

    assert set(power) >= {"time", "forward", "reflected", "net"}
    np.testing.assert_allclose(power["time"], np.arange(5) * SLOW_DT)
    assert power["forward"][0] == pytest.approx(1000.0)
    assert np.isnan(power["forward"][1])  # 0.4 V: below the valid range
    assert np.isnan(power["forward"][2])  # 2.6 V: above the valid range
    assert power["forward"][4] == pytest.approx(10_000.0)
    assert np.isnan(power["reflected"][3])  # 0.3 V
    assert power["reflected"][4] == pytest.approx(1000.0)
    # net = forward - reflected, and a masked side makes the net unknown.
    assert power["net"][0] == pytest.approx(1000.0 - power["reflected"][0])
    assert np.isnan(power["net"][1])
    assert np.isnan(power["net"][2])
    assert np.isnan(power["net"][3])
    assert power["net"][4] == pytest.approx(9000.0)


def test_absent_forward_field_is_unavailable_not_zero(monkeypatch):
    monkeypatch.setattr(ec_module, "_safe_vest_load", lambda *args, **kwargs: None)

    with pytest.raises(RawSignalUnavailableError) as caught:
        ec_launchers({}, 29000, 0.0, 1.0, SLOW_DT)

    assert caught.value.shot == 29000
    assert caught.value.field == 27


def test_absent_reflected_field_is_unavailable_not_zero(monkeypatch):
    loader = _synthetic_loader([2.13] * 10, [2.13] * 10)

    def forward_only(shot, field, raw_source=None):
        return None if field == 28 else loader(shot, field, raw_source)

    monkeypatch.setattr(ec_module, "_safe_vest_load", forward_only)

    with pytest.raises(RawSignalUnavailableError) as caught:
        vest_ec_power(39915)

    assert caught.value.field == 28


def test_shot_39915_carries_a_real_ec_pulse():
    raw = _raw_dump(39915)
    power = vest_ec_power(39915, raw_source=raw)
    time, net = power["time"], power["net"]

    pulse = _window(time, net, 0.30, 0.34)
    assert 1_000.0 < np.nanmedian(pulse) < 10_000.0
    before = _window(time, net, 0.0, 0.05)
    assert np.nanmax(np.abs(before)) < 50.0
    for key in ("forward", "reflected", "net"):
        assert np.nanmax(power[key]) < 70_000.0

    ods = ODS(consistency_check=False)
    ec_launchers(ods, 39915, 0.0, 1.0, SLOW_DT, raw_source=raw)

    assert ods["ec_launchers.beam.0.identifier"] == "5ML10"
    assert ods["ec_launchers.beam.0.name"] == "ECH 6 kW 2.45 GHz"
    assert ods["ec_launchers.ids_properties.homogeneous_time"] == 0
    mapped_time = np.asarray(ods["ec_launchers.beam.0.power_launched.time"])
    mapped = np.asarray(ods["ec_launchers.beam.0.power_launched.data"])
    frequency = np.asarray(ods["ec_launchers.beam.0.frequency.data"])
    assert mapped.shape == mapped_time.shape == frequency.shape
    np.testing.assert_allclose(np.diff(mapped_time), SLOW_DT, rtol=1e-6)
    assert mapped_time[0] >= 0.0 and mapped_time[-1] < 1.0
    np.testing.assert_allclose(frequency, 2.45e9)
    np.testing.assert_allclose(
        np.asarray(ods["ec_launchers.beam.0.frequency.time"]), mapped_time
    )
    assert np.nanmax(mapped) < 70_000.0
    assert 1_000.0 < np.nanmedian(_window(mapped_time, mapped, 0.30, 0.34)) < 10_000.0
    comment = ods["ec_launchers.ids_properties.comment"]
    for fragment in ("27", "28", "0.5", "2.5", "reflected"):
        assert fragment in comment


def test_window_restricts_the_mapped_grid():
    raw = _raw_dump(39915)
    ods = ODS(consistency_check=False)
    ec_launchers(ods, 39915, 0.26, 0.36, SLOW_DT, raw_source=raw)

    mapped_time = np.asarray(ods["ec_launchers.beam.0.power_launched.time"])
    assert mapped_time[0] == pytest.approx(0.26)
    assert mapped_time[-1] < 0.36
    assert mapped_time.size == 2500


@pytest.mark.parametrize("shot", [41524, 41672])
def test_ec_off_shots_report_no_launched_power(shot):
    power = vest_ec_power(shot, raw_source=_raw_dump(shot))
    time, net = power["time"], power["net"]

    assert np.nanmedian(np.abs(_window(time, net, 0.26, 0.36))) < 50.0
    # Pickup spikes inside the valid range (41672 at 631.6 ms dips field 27 to
    # ~0.72 V, ~12 kW) are not hidden by the mask; they must stay rare.
    finite = net[np.isfinite(net)]
    assert np.count_nonzero(finite > 1_000.0) / finite.size < 1e-3


def _write_raw_dump(path, shot, fields):
    payload = {
        "shot": shot,
        "fields": {
            str(field): {"data": values, "type": "slow"} for field, values in fields.items()
        },
    }
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


def test_diagnostics_stage_maps_ec_launchers_on_the_ec_power_policy(tmp_path):
    from vaft.omas.vest_upstream import (
        build_diagnostics_ods,
        build_static_ods,
        machine_era_for_shot,
        write_stage_product,
    )

    shot = 43017
    samples = 2000
    forward = np.full(samples, 2.13)
    forward[500:1500] = 1.01745  # 1 kW forward
    reflected = np.full(samples, 2.13)
    raw = tmp_path / "raw.json.gz"
    _write_raw_dump(raw, shot, {27: forward.tolist(), 28: reflected.tolist()})
    static_path = tmp_path / "static.json.gz"
    static, manifest = build_static_ods(machine_era_for_shot(shot).name)
    write_stage_product(
        static, manifest, output=static_path, metadata=tmp_path / "static-manifest.json"
    )

    ods, diagnostics_manifest = build_diagnostics_ods(
        shot=shot, raw_source=raw, static_ods=static_path, tstart=0.0, tend=0.005, dt=SLOW_DT
    )

    assert diagnostics_manifest["channel_status"]["ec_launchers"]["status"] == "success"
    assert "ec_launchers" in ods
    component = diagnostics_manifest["time_grid"]["components"]["ec_power"]
    assert component["policy"] == "full_discharge"
    # Half-open window clipped to the source: 2000 samples end at 0.07996 s.
    assert component["sample_count"] == samples - 1
    mapped = np.asarray(ods["ec_launchers.beam.0.power_launched.data"])
    assert mapped.size == component["sample_count"]
    # 1 kW forward against the ~0.09 W EC-off reflected baseline.
    assert np.nanmax(mapped) == pytest.approx(1000.0, rel=1e-3)
    assert abs(mapped[0]) < 1.0



def test_a_spike_is_masked_before_resampling_not_smeared_into_power(monkeypatch):
    """A -5 V spike interpolated or filtered onto a coarser grid lands in the
    valid 0.5-2.5 V range on its neighbours and calibrates to kilowatts that
    were never there; masking the native samples first keeps it a gap."""
    samples = 2000
    forward = np.full(samples, 2.13)  # EC off: ~0.1 W
    forward[1000] = -5.0
    reflected = np.full(samples, 2.13)
    monkeypatch.setattr(ec_module, "_safe_vest_load", _synthetic_loader(forward, reflected))

    coarse = np.arange(0, samples - 1, 7, dtype=float) * SLOW_DT + 0.5 * SLOW_DT
    power = vest_ec_power(39915, time=coarse)

    assert np.nanmax(power["forward"]) < 10.0
    assert np.isfinite(power["forward"]).mean() > 0.9
