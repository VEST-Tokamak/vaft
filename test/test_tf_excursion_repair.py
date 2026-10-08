"""TF-current acquisition excursions are repaired, healthy records untouched (#1543)."""

from __future__ import annotations

import gzip
import json

import numpy as np
import pytest

from vaft.machine_mapping.tf import repair_tf_excursions, vfit_tf_dynamic
from vaft.machine_mapping.utils import resolve_vest_diagnostic

TIME = np.arange(0.0, 1.0, 4e-5)


@pytest.fixture(scope="module")
def config():
    return resolve_vest_diagnostic(48245, "tf")["processing"]["excursion_repair"]


def _tf(seed: int = 0, plateau: float = 12000.0, noise: float = 1500.0) -> tuple[np.ndarray, np.ndarray]:
    """A TF current: ramp to the plateau by 0.15 s, slow decay, ~13 % rms raw noise."""
    true = plateau * np.clip(TIME / 0.15, 0, 1) * np.exp(-np.clip(TIME - 0.2, 0, None) / 0.8)
    return true, true + noise * np.random.default_rng(seed).standard_normal(TIME.size)


def test_the_repair_is_enabled_for_vest(config):
    assert config["enabled"] is True
    assert config["window"] == [0.25, 0.40]


@pytest.mark.parametrize("plateau", [12000.0, 9000.0, 5000.0, 3000.0])
def test_healthy_records_are_left_exactly_as_they_were_at_any_plateau(config, plateau):
    """The raw noise is a fixed size; a plateau-fraction trigger alone flagged
    18/20 healthy 9 kA records (cold review of #1616)."""
    for seed in range(40):
        _true, measured = _tf(seed, plateau=plateau)
        repaired, intervals = repair_tf_excursions(TIME, measured, config)
        assert intervals == [], (plateau, seed)
        np.testing.assert_array_equal(repaired, measured)


@pytest.mark.parametrize("plateau", [12000.0, 5000.0, 3000.0])
def test_a_real_excursion_is_caught_at_any_plateau(config, plateau):
    for seed in range(5):
        _true, measured = _tf(seed, plateau=plateau)
        measured[(TIME > 0.303) & (TIME < 0.326)] -= 2 * plateau
        _repaired, intervals = repair_tf_excursions(TIME, measured, config)
        assert len(intervals) == 1, (plateau, seed)


@pytest.mark.parametrize("shift", [-25000.0, +13000.0])
def test_the_48245_excursion_is_bridged_to_within_one_percent(config, shift):
    """The trace leaves the plateau from ~0.303 s to ~0.326 s, as on 48238-48269."""
    true, measured = _tf()
    excursion = (TIME > 0.303) & (TIME < 0.326)
    measured[excursion] += shift
    repaired, intervals = repair_tf_excursions(TIME, measured, config)
    assert len(intervals) == 1
    start, end = intervals[0]
    assert start <= 0.303 and end >= 0.326 and end - start < 0.030
    span = (TIME > 0.30) & (TIME < 0.33)
    assert abs(np.mean(repaired[span] - true[span])) < 0.01 * 12000.0
    outside = ~((TIME >= start) & (TIME <= end))
    np.testing.assert_array_equal(repaired[outside], measured[outside])


def test_a_shot_with_the_tf_off_is_not_touched(config):
    _true, measured = _tf(plateau=10.0, noise=50.0)
    measured[(TIME > 0.30) & (TIME < 0.32)] += 500.0
    repaired, intervals = repair_tf_excursions(TIME, measured, config)
    assert intervals == []
    np.testing.assert_array_equal(repaired, measured)


def test_a_disabled_repair_returns_the_record_unchanged(config):
    _true, measured = _tf()
    measured[(TIME > 0.303) & (TIME < 0.326)] -= 25000.0
    repaired, intervals = repair_tf_excursions(TIME, measured, config | {"enabled": False})
    assert intervals == []
    np.testing.assert_array_equal(repaired, measured)


def test_a_burst_whose_median_dips_back_mid_way_is_repaired_as_one(config):
    """48625: the 1 ms median returns near the trend inside the burst for ~2 ms."""
    true, measured = _tf()
    measured[(TIME > 0.280) & (TIME < 0.2840)] += 120000.0
    measured[(TIME > 0.2862) & (TIME < 0.302)] += 120000.0
    repaired, intervals = repair_tf_excursions(TIME, measured, config)
    assert len(intervals) == 1
    span = (TIME > 0.280) & (TIME < 0.302)
    assert np.max(np.abs(repaired[span] - true[span])) < 0.5 * 12000.0


def _two_ms_error(repaired, true, plateau):
    from scipy.ndimage import uniform_filter1d

    span = (TIME > 0.27) & (TIME < 0.36)
    return float(np.max(np.abs(uniform_filter1d(repaired - true, 50)[span]))) / plateau


def test_a_burst_of_spikes_is_repaired(config):
    """48625: +130 kA spikes on a 5 kA plateau, which a median ignores but the
    low-pass passes into BTOR as a 6x shift."""
    true, measured = _tf(plateau=5000.0)
    burst = (TIME > 0.28) & (TIME < 0.30)
    measured[burst] += 100000.0 * (np.random.default_rng(2).random(burst.sum()) < 0.3)
    repaired, intervals = repair_tf_excursions(TIME, measured, config)
    assert len(intervals) == 1
    # 1500 A raw noise alone moves a 2 ms mean by up to ~15 % of 5 kA.
    assert _two_ms_error(repaired, true, 5000.0) < 0.2
    assert _two_ms_error(measured, true, 5000.0) > 5.0


def test_a_slow_recovery_and_a_shifted_offset_are_bridged_from_before_the_onset(config):
    """46525: -150 kA, still recovering at 0.40 s, and the sensor offset is
    left shifted; a fit through the post-event record would sit ~4 kA low."""
    true, measured = _tf(plateau=6500.0)
    measured[(TIME > 0.295) & (TIME < 0.33)] -= 150000.0
    measured[TIME >= 0.33] -= 4000.0
    repaired, intervals = repair_tf_excursions(TIME, measured, config)
    assert intervals and intervals[0][0] <= 0.295 and intervals[-1][1] >= 0.36
    assert _two_ms_error(repaired, true, 6500.0) < 0.2
    assert _two_ms_error(measured, true, 6500.0) > 5.0


# --------------------------------------------------------------------------- #
# The repair is recorded in the stage manifest (cold review 0.8.0 delta-absorb-16 F2)
# --------------------------------------------------------------------------- #

SHOT = 48245
TF_FIELD = 1          # vest.yaml diagnostics.tf.source.field
TF_FACTOR = -3.0e4    # vest.yaml diagnostics.tf.calibration: amperes = volts * factor


def _write_tf_raw_dump(path, current_amperes):
    """A slow-DAQ archive holding only the TF coil channel, in raw volts."""
    payload = {
        "shot": SHOT,
        "fields": {str(TF_FIELD): {"data": (np.asarray(current_amperes) / TF_FACTOR).tolist(), "type": "slow"}},
    }
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


def test_the_repaired_intervals_reach_the_diagnostics_manifest(tmp_path):
    """`vfit_tf_current_detailed` returns the intervals it replaced and the
    stage rewrote `tf.coil.0.current.data` from them, yet the manifest's
    `quality_summary.repaired` was a literal `[]`: a false provenance statement
    on every shot whose BTOR is a fitted line (48238-48269, 46250, 46500, ...)."""
    from vaft.omas.vest_upstream import (
        build_diagnostics_ods,
        build_static_ods,
        machine_era_for_shot,
        write_stage_product,
    )

    _true, measured = _tf()
    measured[(TIME > 0.303) & (TIME < 0.326)] -= 25000.0
    raw = tmp_path / "raw.json.gz"
    _write_tf_raw_dump(raw, measured)
    static, static_manifest = build_static_ods(machine_era_for_shot(SHOT).name)
    static_path = tmp_path / "static.json.gz"
    write_stage_product(static, static_manifest, output=static_path, metadata=tmp_path / "static-manifest.json")

    ods, manifest = build_diagnostics_ods(shot=SHOT, raw_source=raw, static_ods=static_path)

    assert manifest["channel_status"]["tf"]["status"] == "success"
    repaired = manifest["quality_summary"]["repaired"]
    assert len(repaired) == 1
    (entry,) = repaired
    assert entry["component"] == "tf" and entry["signal"] == "tf.coil.0.current"
    assert entry["start"] <= 0.303 and entry["end"] >= 0.326 and entry["end"] - entry["start"] < 0.030
    assert "straight line" in entry["method"] and "#1543" in entry["method"]
    assert manifest["channel_status"]["tf"]["repaired"] == repaired
    # The IDS says the same thing where a reader of the product looks for it.
    assert f"{entry['start']:.4f}-{entry['end']:.4f} s" in str(ods["tf.ids_properties.comment"])
    json.dumps(manifest)  # the manifest is written as JSON; the entries must survive that


def test_a_healthy_record_reports_no_repair(tmp_path):
    _true, measured = _tf()
    raw = tmp_path / "raw.json.gz"
    _write_tf_raw_dump(raw, measured)
    report = {}

    vfit_tf_dynamic({}, SHOT, 0.0, 1.0, 4e-5, raw_source=raw, report=report)

    assert report == {"repaired": []}
