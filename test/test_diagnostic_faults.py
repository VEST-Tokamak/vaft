"""Recorded faults of diagnostics without an IMAS validity node (#1543)."""

from __future__ import annotations

import pytest
import yaml

from vaft.machine_mapping.diagnostic_faults import known_diagnostic_faults
from vaft.machine_mapping.utils import VestConfigurationError


def _labels(shot: int) -> list[tuple[str, str, str]]:
    return [(f["ids"], f["label"], f["kind"]) for f in known_diagnostic_faults(shot)]


@pytest.mark.parametrize(
    ("shot", "expected"),
    [
        (46676, []),
        (46677, [("spectrometer_uv", "H-beta_4861", "no_signal"), ("spectrometer_uv", "OV_629", "no_signal")]),
        (46992, [("spectrometer_uv", "H-beta_4861", "no_signal"), ("spectrometer_uv", "OV_629", "no_signal")]),
        (47415, [("spectrometer_uv", "H-gamma_4340", "railed"), ("spectrometer_uv", "OV_629", "no_signal")]),
        (47616, [("spectrometer_uv", "H-gamma_4340", "railed"), ("spectrometer_uv", "H-beta_4861", "no_signal"),
                 ("spectrometer_uv", "OV_629", "no_signal")]),
        (47615, [("spectrometer_uv", "H-gamma_4340", "railed"), ("spectrometer_uv", "OV_629", "no_signal")]),
        (48224, [("spectrometer_uv", "H-gamma_4340", "railed"), ("spectrometer_uv", "H-beta_4861", "no_signal"),
                 ("spectrometer_uv", "OV_629", "no_signal")]),
    ],
)
def test_records_at_each_boundary(shot, expected):
    assert _labels(shot) == expected


def test_h_gamma_rails_from_46993():
    assert ("spectrometer_uv", "H-gamma_4340", "railed") not in _labels(46992)
    assert ("spectrometer_uv", "H-gamma_4340", "railed") in _labels(46993)


def test_the_gauge_change_is_a_calibration_not_a_fault_record():
    """46993 is the IKR 251 replacement, resolved in barometry_main (#1543)."""
    assert all(f["ids"] != "barometry" for f in known_diagnostic_faults(48224))


def _write(tmp_path, entries):
    path = tmp_path / "vest.yaml"
    path.write_text(yaml.safe_dump({"diagnostic_faults": entries}))
    return str(path)


BASE = {"ids": "spectrometer_uv", "node": "channel.2.processed_line.2", "label": "H-gamma_4340",
        "field": 138, "from_shot": 1, "kind": "railed", "reason": "test"}


def test_an_entry_naming_the_wrong_field_is_refused(tmp_path):
    with pytest.raises(VestConfigurationError, match="field 138"):
        known_diagnostic_faults(10, info_file=_write(tmp_path, [BASE | {"field": 141}]))


def test_an_entry_naming_the_wrong_label_is_refused(tmp_path):
    with pytest.raises(VestConfigurationError, match="H-gamma_4340"):
        known_diagnostic_faults(10, info_file=_write(tmp_path, [BASE | {"label": "H-beta_4861"}]))


def test_an_unknown_kind_or_ids_is_refused(tmp_path):
    with pytest.raises(VestConfigurationError, match="kind"):
        known_diagnostic_faults(10, info_file=_write(tmp_path, [BASE | {"kind": "broken"}]))
    with pytest.raises(VestConfigurationError, match="no record check"):
        known_diagnostic_faults(10, info_file=_write(tmp_path, [BASE | {"ids": "tf"}]))


def test_a_barometry_entry_must_name_the_main_gauge_field(tmp_path):
    entry = {"ids": "barometry", "node": "gauge.0", "label": "PKR-251 Main Gauge", "field": 13,
             "from_shot": 46993, "kind": "calibration_unverified", "reason": "test"}
    with pytest.raises(VestConfigurationError, match="field 12"):
        known_diagnostic_faults(47000, info_file=_write(tmp_path, [entry]))
