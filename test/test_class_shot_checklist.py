"""The class-shot checklist reads a product without touching it, and says why (#1543)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from omas import ODS

SCRIPT = Path(__file__).resolve().parents[1] / "workflow" / "magnetics_quality" / "class_shot_checklist.py"


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location("class_shot_checklist", SCRIPT)
    loaded = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = loaded
    try:
        spec.loader.exec_module(loaded)
    except Exception:
        del sys.modules[spec.name]
        raise
    yield loaded
    sys.modules.pop(spec.name, None)


def _discharge(time: np.ndarray, *, peak: float, tail: float) -> np.ndarray:
    """A 20 ms discharge at 0.30 s that leaves ``tail`` amperes behind."""
    shape = np.exp(-(((time - 0.30) / 0.006) ** 2)) * peak
    return shape + tail * (time > 0.32)


def _ods(*, peak: float = 200e3, tail: float = 0.0, clamp_samples: int = 0) -> ODS:
    ods = ODS(consistency_check=False)
    time = np.linspace(0.26, 0.36, 2501)
    ods["magnetics.time"] = time
    ods["magnetics.ip.0.data"] = _discharge(time, peak=peak, tail=tail)
    ods["magnetics.ip.0.time"] = time
    ods["spectrometer_uv.time"] = time
    # OMAS arrays grow in order: the fast filterscope is channel 2.
    for channel in (0, 1, 2):
        ods[f"spectrometer_uv.channel.{channel}.name"] = f"channel {channel}"
    line = np.sin(np.linspace(0.0, 30.0, time.size)) + 2.0
    if clamp_samples:
        line[-clamp_samples:] = line[-clamp_samples - 1]
    ods["spectrometer_uv.channel.2.processed_line.0.intensity.data"] = line
    return ods


def test_plasma_current_row_reports_peak_and_tail(module):
    row = module.plasma_current_row(_ods(peak=200e3, tail=-15e3))
    assert row["available"]
    assert row["peak_kA"] == pytest.approx(200.0, rel=1e-3)
    assert row["t_peak_s"] == pytest.approx(0.30, abs=1e-4)
    assert row["tail_kA"] == pytest.approx(-15.0, rel=1e-3)
    assert row["head_kA"] == pytest.approx(0.0, abs=1e-3)


def test_reading_a_product_does_not_create_paths(module):
    ods = _ods()
    before = set(ods.flat())
    module.rogowski_row(ods)
    module.plasma_current_row(ods)
    assert set(ods.flat()) == before
    assert "magnetics.rogowski_coil" not in ods
    assert "magnetics.diamagnetic_flux" not in ods


def test_missing_ip_is_reported_not_raised(module):
    assert module.plasma_current_row(ODS(consistency_check=False)) == {"available": False}


def test_clamped_filterscope_tail_is_measured_in_milliseconds(module):
    # 50 samples of 40 us is 2 ms; the run counts the last measured sample too.
    row = module.filterscope_row(_ods(clamp_samples=50))
    assert row["lines"] == 1
    assert row["clamped_tail_ms"] == pytest.approx(2.04, abs=0.05)
    assert module.filterscope_row(_ods())["clamped_tail_ms"] == 0.0


def test_flags_name_each_condition(module):
    row = {
        "ip": module.plasma_current_row(_ods(peak=1.6e6, tail=900e3)),
        "rogowski": {"plasma_rogowski_validity": -2, "tf_rogowski_validity": 0, "diamagnetic_flux": False},
        "magnetics": {"families": {"side": {"declared": 16, "condemned": 8}, "inboard": {"declared": 27, "condemned": 2}}},
        "pf": {"worst": {"coil": "PF1", "drift_A": -120.0}},
        "filterscope": {"clamped_tail_ms": 20.0},
    }
    flags = module.flags_for(row)
    assert flags == [
        "ip_peak_implausible",
        "ip_tail_residual",
        "plasma_rogowski_invalid",
        "no_diamagnetic_flux",
        "side_half_condemned",
        "pf_drift_PF1",
        "filterscope_tail_clamped",
    ]


def test_a_clean_shot_raises_no_flag(module):
    row = {
        "ip": module.plasma_current_row(_ods()),
        "rogowski": {"plasma_rogowski_validity": 0, "tf_rogowski_validity": 0, "diamagnetic_flux": True},
        "magnetics": {"families": {"outboard": {"declared": 21, "condemned": 1}}},
        "filterscope": {"clamped_tail_ms": 0.0},
    }
    assert module.flags_for(row) == []


def test_shot_ranges_parse_in_order_without_repeats(module):
    assert module.parse_shots("48940, 48224-48226,48225") == [48940, 48224, 48225, 48226]


def test_an_absent_product_is_a_row_not_an_error(module, tmp_path):
    assert module.check_shot(1, filedb=tmp_path, raw=False) == {"shot": 1, "status": "absent"}
