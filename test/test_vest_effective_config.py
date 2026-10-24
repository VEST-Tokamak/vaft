"""Effective VEST configuration: resolution, overrides, provenance and identity (#2159)."""

from __future__ import annotations

import hashlib
import math
from dataclasses import replace
from types import MappingProxyType

import pytest
import yaml

from vaft.machine_mapping.effective_config import (
    FieldClass,
    canonical_json,
    classify_config_path,
    configuration_fingerprint,
    impa_shot_era,
    resolve_vest_processing_config,
    vest_configuration_record,
)
from vaft.machine_mapping.impa import impa_expected_fields, resolve_impa_config
from vaft.machine_mapping.magnetics import (
    _equilibrium_magnetics_window_for_shot,
    _plasma_processing_for_shot,
    equilibrium_magnetics_processing_config,
)
from vaft.machine_mapping.provenance import vest_processing_provenance
from vaft.machine_mapping.utils import VestConfigurationError, package_data_path, resolve_vest_diagnostic
from vaft.omas.vest_upstream import magnetics_processing_for_shot, resolve_magnetics_processing
from vaft.process.magnetics import VestMagneticsProcessingConfig


def _strip_tables(value):
    """A resolved diagnostic without its still-unresolved nested tables."""
    if isinstance(value, dict):
        return {
            key: _strip_tables(item)
            for key, item in value.items()
            if key not in ("revisions", "era_only_keys")
        }
    return value


# --- the effective configuration is what the mappers resolve --------------------------


@pytest.mark.parametrize("shot", [17454, 17455, 41446, 43761, 46402, 46403, 47116, 47117])
def test_plasma_current_matches_the_mapper_resolution(shot):
    effective = resolve_vest_processing_config(shot, "plasma_current").effective_dict()
    config, reference, baseline, sign = _plasma_processing_for_shot(shot)
    assert effective["processing"]["reference"] == reference
    assert effective["processing"]["baseline"] == baseline
    assert effective["processing"]["sign"] == sign
    assert effective["calibration"] == config["calibration"]


@pytest.mark.parametrize("shot", [39437, 39438, 41446, 43684, 43685, 46403, 46404])
def test_equilibrium_magnetics_window_matches_the_mapper_resolution(shot):
    effective = resolve_vest_processing_config(shot, "equilibrium_magnetics").effective_dict()
    assert effective["processing"]["window"] == _equilibrium_magnetics_window_for_shot(shot)


@pytest.mark.parametrize("name", ["pf_active", "tf", "diamagnetic_flux", "barometry_main", "core_profiles"])
@pytest.mark.parametrize("shot", [39915, 45896, 48372])
def test_top_level_values_match_resolve_vest_diagnostic(name, shot):
    effective = resolve_vest_processing_config(shot, name).effective_dict()
    assert effective == _strip_tables(resolve_vest_diagnostic(shot, name))


def test_impa_effective_is_the_resolved_impa_block():
    assert resolve_vest_processing_config(39915, "impa").effective_dict() == resolve_impa_config(39915)


# --- boundary regressions --------------------------------------------------------------


def _key(resolved, path):
    return resolved.provenance["keys"][path]


def test_boundary_39437_39438_wiring():
    before = resolve_vest_processing_config(39437, "equilibrium_magnetics")
    after = resolve_vest_processing_config(39438, "equilibrium_magnetics")
    assert before.effective["processing"]["wiring"]["layout"] != after.effective["processing"]["wiring"]["layout"]
    assert _key(before, "processing.wiring.layout")["origin"] == "revision"
    assert _key(after, "processing.wiring.layout")["origin"] == "base"
    assert before.fingerprint != after.fingerprint


def test_boundary_46402_46403_plasma_current_fl10():
    before = resolve_vest_processing_config(46402, "plasma_current")
    after = resolve_vest_processing_config(46403, "plasma_current")
    assert before.effective["processing"]["reference"]["mode"] == "subtract"
    assert after.effective["processing"]["reference"]["mode"] == "subtract_fl10_windowed"
    assert "fl10" not in before.effective["processing"]["reference"]
    record = _key(after, "processing.reference.fl10.field")
    assert record["origin"] == "revision"
    assert record["revision_bounds"] == {"from_shot": 46403, "to_shot": 47116}


def test_boundary_46403_46404_daq():
    before = resolve_vest_processing_config(46403, "equilibrium_magnetics")
    after = resolve_vest_processing_config(46404, "equilibrium_magnetics")
    assert before.effective["processing"]["window"]["daq_mode"] == "legacy"
    assert after.effective["processing"]["window"]["daq_mode"] == "native_daq"
    record = _key(after, "processing.window.daq_mode")
    assert record["class"] == "machine_fact"
    assert record["revision_bounds"] == {"from_shot": 46404, "to_shot": None}
    assert resolve_magnetics_processing(46404).effective["daq_mode"] == "native_daq"
    assert resolve_magnetics_processing(46403).effective["daq_mode"] == "legacy"


@pytest.mark.parametrize("pair", [(45895, 45896), (48371, 48372)])
def test_boundary_pf_calibration_eras(pair):
    before, after = (resolve_vest_processing_config(shot, "pf_active") for shot in pair)
    assert before.effective["processing"]["coil_gains"] != after.effective["processing"]["coil_gains"]
    assert before.fingerprint != after.fingerprint
    assert _key(after, "processing.coil_gains.0")["class"] == "calibration"
    assert _key(after, "processing.coil_gains.0")["revision_bounds"]["from_shot"] == pair[1]


# --- fingerprint and source identity ---------------------------------------------------


def test_fingerprint_is_shared_within_an_era_and_deterministic():
    one = resolve_vest_processing_config(46500, "equilibrium_magnetics")
    two = resolve_vest_processing_config(46501, "equilibrium_magnetics")
    assert one.fingerprint == two.fingerprint == configuration_fingerprint(one.effective_dict())
    assert one.source_sha256 == hashlib.sha256(open(package_data_path("vest.yaml"), "rb").read()).hexdigest()


_BLOCK_STYLE = """\
# a comment
0:
  diagnostics:
    probe:
      source: {field: 7}
      processing:
        cutoff: 2.5e+3
        taps: 251
        window: [1, 2]
      revisions:
        - {from_shot: 10, processing: {cutoff: 1000.0}}
"""

_FLOW_STYLE = """\
0: {diagnostics: {probe: {processing: {window: [1, 2], taps: 251.0, cutoff: 2500.0},
  revisions: [{processing: {cutoff: 1.0e+3}, from_shot: 10}], source: {field: 7}}}}
"""


def test_fingerprint_ignores_yaml_formatting(tmp_path):
    block, flow = tmp_path / "block.yaml", tmp_path / "flow.yaml"
    block.write_text(_BLOCK_STYLE, encoding="utf-8")
    flow.write_text(_FLOW_STYLE, encoding="utf-8")
    for shot in (5, 10):
        a = resolve_vest_processing_config(shot, "probe", info_file=str(block))
        b = resolve_vest_processing_config(shot, "probe", info_file=str(flow))
        assert a.fingerprint == b.fingerprint
        assert a.source_sha256 != b.source_sha256
    assert (
        resolve_vest_processing_config(5, "probe", info_file=str(block)).fingerprint
        != resolve_vest_processing_config(10, "probe", info_file=str(block)).fingerprint
    )
    assert resolve_vest_processing_config(5, "probe", info_file=str(block)).provenance["source"]["path"] == str(block)


def test_canonical_json_rules():
    assert canonical_json({"b": 1.0, "a": [2, 2.5]}) == '{"a":[2,2.5],"b":1}'
    assert canonical_json({0: True}) == '{"0":true}'
    assert canonical_json([math.nan, math.inf, -math.inf]) == '["NaN","Infinity","-Infinity"]'
    assert canonical_json((1, 2)) == canonical_json([1, 2])
    with pytest.raises(VestConfigurationError, match="collide"):
        canonical_json({0: 1, "0": 2})
    with pytest.raises(VestConfigurationError, match="not a configuration value"):
        canonical_json({"a": object()})


def test_resolved_configuration_is_read_only():
    resolved = resolve_vest_processing_config(39915, "plasma_current")
    assert isinstance(resolved.effective, MappingProxyType)
    with pytest.raises(TypeError):
        resolved.effective["processing"]["reference"]["mode"] = "x"  # type: ignore[index]
    copy = resolved.effective_dict()
    copy["processing"]["reference"]["mode"] = "x"
    assert resolve_vest_processing_config(39915, "plasma_current").effective["processing"]["reference"]["mode"] != "x"


def test_a_missing_explicit_file_raises(tmp_path):
    with pytest.raises(VestConfigurationError, match="does not exist"):
        resolve_vest_processing_config(39915, "tf", info_file=str(tmp_path / "nope.yaml"))
    with pytest.raises(VestConfigurationError, match="does not exist"):
        vest_processing_provenance(39915, info_file=str(tmp_path / "nope.yaml"))


# --- field classes and overrides -------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (("processing", "window", "daq_mode"), FieldClass.MACHINE_FACT),
        (("source", "field"), FieldClass.MACHINE_FACT),
        (("processing", "wiring", "layout"), FieldClass.MACHINE_FACT),
        (("calibration", "factor"), FieldClass.CALIBRATION),
        (("processing", "coil_gains", 3), FieldClass.CALIBRATION),
        (("processing", "window", "index_start"), FieldClass.ANALYSIS),
        (("processing", "saturation_repair", "tolerance"), FieldClass.PROCESSING),
    ],
)
def test_field_classes(path, expected):
    assert classify_config_path(path) is expected


def test_an_unknown_override_key_is_refused():
    with pytest.raises(VestConfigurationError, match="names no configured key"):
        resolve_vest_processing_config(39915, "plasma_current", overrides={"processing": {"baseline": {"lookbak": 1}}})


def test_an_override_may_not_change_a_type():
    with pytest.raises(VestConfigurationError, match="is a string, the configured value a number"):
        resolve_vest_processing_config(39915, "plasma_current", overrides={"processing": {"baseline": {"lookback": "ten"}}})
    with pytest.raises(VestConfigurationError, match="is a number, the configured value a mapping"):
        resolve_vest_processing_config(39915, "plasma_current", overrides={"processing": {"baseline": 3}})


def test_machine_facts_cannot_be_overridden_but_may_be_restated():
    with pytest.raises(VestConfigurationError, match="machine fact"):
        resolve_vest_processing_config(
            46404, "equilibrium_magnetics", overrides={"processing": {"window": {"daq_mode": "legacy"}}}
        )
    with pytest.raises(VestConfigurationError, match="machine fact"):
        resolve_vest_processing_config(39915, "plasma_current", overrides={"source": {"field": 1}})
    restated = resolve_vest_processing_config(
        46404, "equilibrium_magnetics", overrides={"processing": {"window": {"daq_mode": "native_daq"}}}
    )
    assert restated.fingerprint == resolve_vest_processing_config(46404, "equilibrium_magnetics").fingerprint
    assert restated.provenance["overrides"]["accepted"] == ()


def test_calibration_overrides_need_a_recorded_reason():
    override = {"calibration": {"factor": 1.0e-5}}
    with pytest.raises(VestConfigurationError, match="calibration_override_reason"):
        resolve_vest_processing_config(39915, "plasma_current", overrides=override)
    accepted = resolve_vest_processing_config(
        39915, "plasma_current", overrides=override, calibration_override_reason="bench re-calibration"
    )
    assert accepted.effective["calibration"]["factor"] == 1.0e-5
    assert accepted.provenance["overrides"]["calibration_override_reason"] == "bench re-calibration"
    assert accepted.provenance["keys"]["calibration.factor"]["overridden"] is True


def test_numeric_strings_compare_by_value():
    """PyYAML reads the PF gains (``1.0e4``) as strings; an override compares by number."""
    configured = resolve_vest_processing_config(39915, "pf_active").effective["processing"]["coil_gains"]
    assert isinstance(configured[0], str)
    restated = resolve_vest_processing_config(
        39915, "pf_active", overrides={"processing": {"coil_gains": {0: float(configured[0])}}}
    )
    assert restated.provenance["overrides"]["accepted"] == ()
    with pytest.raises(VestConfigurationError, match="calibration_override_reason"):
        resolve_vest_processing_config(39915, "pf_active", overrides={"processing": {"coil_gains": {0: 1.0}}})


def test_a_processing_override_is_applied_and_recorded():
    resolved = resolve_vest_processing_config(
        43761,
        "plasma_current",
        overrides={"processing": {"baseline": {"lookback": 10}}},
        override_source="campaign.yaml",
    )
    assert resolved.effective["processing"]["baseline"]["lookback"] == 10
    record = resolved.provenance["keys"]["processing.baseline.lookback"]
    assert record == {**record, "origin": "override", "overridden": True, "source": "campaign.yaml"}
    untouched = resolved.provenance["keys"]["processing.baseline.analysis_start"]
    assert untouched["overridden"] is False and untouched["origin"] == "revision"
    assert resolved.provenance["overrides"]["accepted"][0]["path"] == "processing.baseline.lookback"


# --- IMPA shot eras ----------------------------------------------------------------------


def _calibration(*eras):
    return {"gain": -0.1, "shot_era_overrides": list(eras)}


def test_impa_era_bounds_are_inclusive():
    calibration = _calibration({"min_shot": 10, "max_shot": 20, "gain": -0.2})
    for shot, expected in ((9, None), (10, 0), (20, 0), (21, None)):
        _era, provenance = impa_shot_era(calibration, shot)
        assert provenance["revision_index"] == expected


def test_impa_overlapping_eras_are_refused():
    calibration = _calibration({"min_shot": 10, "max_shot": 20}, {"min_shot": 20, "max_shot": 30})
    with pytest.raises(VestConfigurationError, match="overlap"):
        impa_shot_era(calibration, 5)


def test_impa_era_bounds_are_explicit():
    with pytest.raises(VestConfigurationError, match="max_shot is required"):
        impa_shot_era(_calibration({"min_shot": 10, "gain": -0.2}), 15)
    era, _ = impa_shot_era(_calibration({"max_shot": 20, "gain": -0.2}), 3)
    assert era["gain"] == -0.2  # a missing min_shot is open-ended, as the old 0 default
    with pytest.raises(VestConfigurationError, match="unknown keys 'gian'"):
        impa_shot_era(_calibration({"min_shot": 1, "max_shot": 2, "gian": 1}), 1)


def test_packaged_impa_era_is_unchanged():
    assert len(impa_expected_fields(35380)) == 7
    assert len(impa_expected_fields(35391)) == 8
    resolved = resolve_vest_processing_config(35380, "impa")
    assert resolved.provenance["revisions"][-1]["revision_bounds"] == {"from_shot": 35370, "to_shot": 35390}


# --- magnetics processing overrides ------------------------------------------------------


def test_an_unknown_magnetics_key_is_a_configuration_error():
    with pytest.raises(VestConfigurationError, match="unknown magnetics processing key"):
        magnetics_processing_for_shot(47946, {"lowpass_cutof": 1000.0})


@pytest.mark.parametrize(("key", "value"), [("daq_mode", "legacy"), ("fast_sample_rate", 125_000.0)])
def test_the_magnetics_acquisition_cannot_be_overridden(key, value):
    with pytest.raises(VestConfigurationError, match="cannot be overridden"):
        magnetics_processing_for_shot(47946, {key: value})


def test_a_magnetics_calibration_override_needs_a_reason():
    with pytest.raises(VestConfigurationError, match="calibration_override_reason"):
        magnetics_processing_for_shot(47946, {"calibration_mode": "multiply"})
    resolution = resolve_magnetics_processing(
        47946, {"calibration_mode": "multiply", "calibration_override_reason": "sign study"}
    )
    assert resolution.config.calibration_mode == "multiply"
    assert resolution.provenance["overrides"]["calibration_override_reason"] == "sign study"
    assert resolution.provenance["keys"]["calibration_mode"]["origin"] == "override"


def test_an_invalid_magnetics_filter_is_a_configuration_error():
    with pytest.raises(VestConfigurationError, match="Nyquist"):
        magnetics_processing_for_shot(47946, {"lowpass_cutoff": 125_000.0})


def test_magnetics_resolution_records_every_shot():
    resolution = resolve_magnetics_processing(46404)
    assert resolution.overridden is False
    assert resolution.config == equilibrium_magnetics_processing_config(46404)
    assert resolution.provenance["keys"]["window_override"]["origin"] == "vest.yaml"
    assert resolution.provenance["keys"]["lowpass_cutoff"]["origin"] == "default"
    assert resolution.fingerprint == configuration_fingerprint(resolution.effective)


@pytest.mark.parametrize(
    "changes",
    [
        {"lowpass_cutoff": 125_000.0},
        {"lowpass_cutoff": 0.0},
        {"fast_sample_rate": -1.0},
        {"lowpass_taps": 0},
        {"sample_count": 1},
        {"time_end": 0.0},
        {"lowpass_taps": True},
        {"calibration_mode": "add"},
        {"window_override": (10, 5, 1)},
        {"window_override": (1, 2)},
        {"flux_baseline_window": (0.3, 0.2)},
        {"flux_baseline_samples": 0},
    ],
)
def test_processing_config_validation(changes):
    with pytest.raises(ValueError):
        replace(VestMagneticsProcessingConfig(), **changes)


def test_processing_config_accepts_every_era():
    for shot in (36000, 39438, 41446, 43685, 46404, 48900):
        equilibrium_magnetics_processing_config(shot)  # __post_init__ passes
    VestMagneticsProcessingConfig(window_override=(0, 0, 1750))


# --- provenance and manifest records -----------------------------------------------------


def test_processing_provenance_covers_tf_impa_and_time_policies():
    record = vest_processing_provenance(35380)
    assert record["source"]["packaged"] is True
    assert record["impa"]["era"]["revision_bounds"] == {"from_shot": 35370, "to_shot": 35390}
    assert record["impa"]["gain"] == pytest.approx(-2.0 / 15.0)
    assert record["tf"]["fingerprint"] == resolve_vest_processing_config(35380, "tf").fingerprint
    assert "impa" in record["time_policies"]["diagnostics_time_policies"]["components"]
    assert record["time_policies"]["plasma_timing"]["fingerprint"]


def test_configuration_record_reports_an_era_gap_instead_of_failing(tmp_path):
    path = tmp_path / "vest.yaml"
    path.write_text(
        yaml.safe_dump(
            {0: {"diagnostics": {"ok": {"a": 1}, "bad": {"calibration": {"type": "nope"}}}}}
        ),
        encoding="utf-8",
    )
    record = vest_configuration_record(1, info_file=str(path), extra={"time_policies": {"x": 1}})
    assert record["components"]["ok"]["fingerprint"] == configuration_fingerprint({"a": 1})
    assert "Unsupported VEST calibration type" in record["components"]["bad"]["error"]
    assert record["components"]["time_policies"]["effective"] == {"x": 1}
    assert record["source"]["path"] == str(path)
    assert record["fingerprint"] == vest_configuration_record(1, info_file=str(path), extra={"time_policies": {"x": 1}})["fingerprint"]
