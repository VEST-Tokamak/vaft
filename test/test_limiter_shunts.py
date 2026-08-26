"""Regression coverage for VEST limiter-current monitor mapping (issue #134)."""

from __future__ import annotations

import gzip
import json

import numpy as np
from omas import ODS, load_omas_json

from vaft.database.raw import SLOW_DT
from vaft.machine_mapping.magnetics import (
    LIMITER_SHUNT_CHANNELS,
    LIMITER_SHUNT_RESISTANCE,
    vfit_limiter_shunts_dynamic,
    vfit_magnetics_static,
)
from vaft.omas import save as save_ods


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


def test_limiter_shunts_use_database_order_and_effective_resistance():
    ods = ODS(consistency_check=False)
    vfit_magnetics_static(ods)

    assert len(ods["magnetics.shunt"]) == 3
    assert [ods[f"magnetics.shunt.{index}.identifier"] for index in range(3)] == [
        channel["identifier"] for channel in LIMITER_SHUNT_CHANNELS
    ]
    assert [ods[f"magnetics.shunt.{index}.name"] for index in range(3)] == [
        channel["name"] for channel in LIMITER_SHUNT_CHANNELS
    ]
    for index in range(3):
        assert ods[f"magnetics.shunt.{index}.resistance"] == LIMITER_SHUNT_RESISTANCE
        assert f"magnetics.shunt.{index}.position" not in ods
        assert f"magnetics.shunt.{index}.current" not in ods


def test_limiter_shunt_voltage_is_baseline_corrected_and_round_trips(tmp_path):
    shot = 39915
    time = np.arange(8_000) * SLOW_DT
    baseline = 0.25
    limiter_signal = np.where(time >= 0.26, 0.5, 0.0)
    raw = tmp_path / "raw.json.gz"
    _write_raw_dump(raw, shot, {216: baseline + limiter_signal})

    ods = ODS(consistency_check=False)
    vfit_magnetics_static(ods)
    ods["magnetics.time"] = time
    vfit_limiter_shunts_dynamic(ods, shot, raw_source=raw)

    np.testing.assert_allclose(ods["magnetics.shunt.0.voltage.time"], time)
    np.testing.assert_allclose(ods["magnetics.shunt.0.voltage.data"], limiter_signal)
    np.testing.assert_allclose(
        ods["magnetics.shunt.0.voltage.data"] / ods["magnetics.shunt.0.resistance"],
        limiter_signal / 0.1,
    )
    assert ods["magnetics.shunt.0.voltage.validity"] == 0

    for index in (1, 2):
        assert ods[f"magnetics.shunt.{index}.voltage.validity"] == -2
        assert np.asarray(ods[f"magnetics.shunt.{index}.voltage.time"]).size == 0
        assert np.asarray(ods[f"magnetics.shunt.{index}.voltage.data"]).size == 0

    output = tmp_path / "diagnostics.json"
    save_ods(ods, output)
    reloaded = load_omas_json(str(output), consistency_check=True)
    np.testing.assert_allclose(reloaded["magnetics.shunt.0.voltage.data"], limiter_signal)
