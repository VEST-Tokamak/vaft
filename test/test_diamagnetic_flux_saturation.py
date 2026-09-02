"""Diamagnetic-Rogowski acquisition-limit saturation (issue #285).

The channel is a signed 16-bit ADC spanning +/-5 V, so it rails at exactly
``-5.0`` and at ``5 * 32767 / 32768`` -- two asymmetric limits, both of which
are hit on every packaged reference shot. The raw voltage is integrated three
times before it reaches ``magnetics.diamagnetic_flux``, so the samples must be
detected and reconstructed before the first integration, not after.
"""

import copy
import gzip
import json
from unittest.mock import patch

import numpy as np
import pytest

import vaft.machine_mapping.magnetics as magnetics
from vaft.machine_mapping.utils import resolve_vest_diagnostic
from vaft.process.signal_processing import (
    SignalRepairError,
    detect_clipped_samples,
    repair_clipped_interval,
)

NEGATIVE_RAIL = -5.0
POSITIVE_RAIL = 5.0 * 32767 / 32768
RAILS = [NEGATIVE_RAIL, POSITIVE_RAIL]
RAIL_TOLERANCE = 3.0e-4

SAMPLE_SOURCE = "vaft/data/samples/{shot}/source/vest_{shot}_daq_raw.json.gz"

# Measured directly from the packaged raw dumps at the exact rail values.
PACKAGED_SATURATION = {39915: 47, 41524: 108, 41672: 109}


# --------------------------------------------------------------------------
# The shared detection primitive
# --------------------------------------------------------------------------


def test_detection_unions_both_asymmetric_rails():
    values = np.array([0.0, NEGATIVE_RAIL, 1.0, POSITIVE_RAIL, -1.0])
    mask = detect_clipped_samples(values, clip_values=RAILS, tolerance=RAIL_TOLERANCE)
    np.testing.assert_array_equal(mask, [False, True, False, True, False])


def test_detection_is_a_no_op_on_an_unsaturated_trace():
    values = np.linspace(-4.5, 4.5, 101)
    mask = detect_clipped_samples(values, clip_values=RAILS, tolerance=RAIL_TOLERANCE)
    assert not mask.any()


def test_detection_accepts_a_scalar_limit():
    values = np.array([0.0, -5000.0, 10.0])
    mask = detect_clipped_samples(values, clip_values=-5000.0, tolerance=10.0)
    np.testing.assert_array_equal(mask, [False, True, False])


@pytest.mark.parametrize("bad", [[], np.nan])
def test_detection_rejects_unusable_limits(bad):
    with pytest.raises(SignalRepairError):
        detect_clipped_samples(np.zeros(4), clip_values=bad, tolerance=1.0)


def test_detection_rejects_a_non_positive_tolerance():
    with pytest.raises(SignalRepairError):
        detect_clipped_samples(np.zeros(4), clip_values=1.0, tolerance=0.0)


# --------------------------------------------------------------------------
# Repair with two interleaved rails
# --------------------------------------------------------------------------


def _interleaved_rail_waveform():
    """A hard oscillation that clips on both rails within a few samples."""
    time = np.linspace(0.0, 1.0, 401)
    truth = 9.0 * np.sin(2 * np.pi * 5.0 * time)
    clipped = np.clip(truth, NEGATIVE_RAIL, POSITIVE_RAIL)
    return time, truth, clipped


def test_repair_handles_both_rails_in_one_pass():
    time, truth, clipped = _interleaved_rail_waveform()
    repaired, mask = repair_clipped_interval(
        time,
        clipped,
        clip_value=RAILS,
        tolerance=RAIL_TOLERANCE,
        return_mask=True,
    )
    assert mask.sum() > 0
    assert (clipped[mask] > 0).any() and (clipped[mask] < 0).any()
    # Reconstruction must move the railed samples away from the limits, and
    # towards the truth rather than anywhere at all.
    assert np.abs(repaired[mask] - truth[mask]).max() < np.abs(clipped[mask] - truth[mask]).max()


def test_repair_preserves_every_unsaturated_sample_exactly():
    time, _, clipped = _interleaved_rail_waveform()
    repaired, mask = repair_clipped_interval(
        time, clipped, clip_value=RAILS, tolerance=RAIL_TOLERANCE, return_mask=True
    )
    np.testing.assert_array_equal(repaired[~mask], clipped[~mask])


def test_repair_returns_a_bare_array_without_return_mask():
    time, _, clipped = _interleaved_rail_waveform()
    repaired = repair_clipped_interval(
        time, clipped, clip_value=RAILS, tolerance=RAIL_TOLERANCE
    )
    assert isinstance(repaired, np.ndarray)


def test_scalar_clip_value_still_behaves_as_before():
    """The PF6 caller passes a scalar; multi-level support must not change it."""
    time = np.linspace(0.0, 1.0, 101)
    values = np.full(101, -1000.0)
    values[40:45] = -5000.0
    scalar = repair_clipped_interval(time, values, clip_value=-5000.0, tolerance=10.0)
    sequence = repair_clipped_interval(time, values, clip_value=[-5000.0], tolerance=10.0)
    np.testing.assert_allclose(scalar, sequence)
    assert np.all(scalar[40:45] > -5000.0)


def test_clean_trace_returns_an_all_false_mask():
    time = np.linspace(0.0, 1.0, 51)
    values = np.zeros(51)
    repaired, mask = repair_clipped_interval(
        time, values, clip_value=RAILS, tolerance=RAIL_TOLERANCE, return_mask=True
    )
    assert not mask.any()
    np.testing.assert_array_equal(repaired, values)


# --------------------------------------------------------------------------
# Shot-era configuration moved into vest.yaml
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("shot", "field"),
    [(1, 246), (37504, 246), (37505, 4), (38451, 4), (38452, 257), (41672, 257)],
)
def test_yaml_reproduces_the_retired_field_code_ternary(shot, field):
    """`246 if shot < 37505 else 4 if shot < 38452 else 257`, now in vest.yaml."""
    config = resolve_vest_diagnostic(shot, "diamagnetic_flux")
    assert int(config["source"]["field"]) == field


def test_yaml_carries_both_asymmetric_acquisition_rails():
    limits = resolve_vest_diagnostic(41672, "diamagnetic_flux")["processing"][
        "saturation_repair"
    ]
    assert sorted(limits["values"]) == pytest.approx([NEGATIVE_RAIL, POSITIVE_RAIL])
    assert limits["tolerance"] == pytest.approx(RAIL_TOLERANCE)


def test_yaml_preserves_the_tf_circuit_constants():
    processing = resolve_vest_diagnostic(41672, "diamagnetic_flux")["processing"]
    assert processing["tf_circuit"] == {
        "turns": 24,
        "inductance": pytest.approx(9.3e-4),
        "resistance": pytest.approx(0.0279),
        "capacitance": pytest.approx(120.0),
    }
    assert processing["rogowski_shunt"] == pytest.approx(8.12e-3)


# --------------------------------------------------------------------------
# Packaged reference shots
# --------------------------------------------------------------------------


def _flux(shot, *, repair: bool):
    """Map the packaged shot with saturation repair enabled or disabled."""
    config = magnetics._diamagnetic_config(shot)
    if not repair:
        config = copy.deepcopy(config)
        # An unreachable limit disables detection without touching the chain.
        config["processing"]["saturation_repair"] = {"values": [1e12], "tolerance": 1e-9}
    with patch.object(magnetics, "_diamagnetic_config", return_value=config):
        ods = {}
        magnetics.diamagnetic_flux_rogowski_coil_from_raw_database(
            ods, shot, raw_source=SAMPLE_SOURCE
        )
    return np.asarray(ods["magnetics"]["diamagnetic_flux"][0]["data"], dtype=float)


@pytest.mark.parametrize("shot", sorted(PACKAGED_SATURATION))
def test_packaged_shots_saturate_on_both_rails(shot):
    report = magnetics.diamagnetic_saturation_report(shot, raw_source=SAMPLE_SOURCE)
    assert report["field"] == 257
    assert report["n_saturated"] == PACKAGED_SATURATION[shot]
    assert report["n_intervals"] > 1
    assert report["first_time"] < 0.2 < report["last_time"]


@pytest.mark.parametrize("shot", sorted(PACKAGED_SATURATION))
def test_packaged_saturation_falls_outside_the_plasma_window(shot):
    """The plasma window is confined to 0.30-0.36 s; the rails are hit outside it."""
    report = magnetics.diamagnetic_saturation_report(
        shot, raw_source=SAMPLE_SOURCE, plasma_start=0.30, plasma_end=0.36
    )
    assert report["n_saturated_in_window"] == 0


@pytest.mark.parametrize("shot", sorted(PACKAGED_SATURATION))
def test_repair_does_not_move_the_packaged_waveforms(shot):
    """Regression lock on the quantification asked for by issue #285.

    Because every saturated sample lies outside the plasma window, the
    reference subtraction and the two-point baseline removal cancel the
    clipping-induced offset exactly. Repair is therefore a no-op *here* --
    and a future change that starts moving these waveforms is a real finding,
    not a rounding difference.
    """
    original = _flux(shot, repair=False)
    corrected = _flux(shot, repair=True)
    assert np.abs(original).max() > 1e-4  # the waveform is not trivially zero
    np.testing.assert_allclose(corrected, original, atol=1e-12)


@pytest.mark.parametrize("shot", sorted(PACKAGED_SATURATION))
def test_method_name_records_the_saturation_outcome(shot):
    ods = {}
    magnetics.diamagnetic_flux_rogowski_coil_from_raw_database(
        ods, shot, raw_source=SAMPLE_SOURCE
    )
    method_name = ods["magnetics"]["diamagnetic_flux"][0]["method_name"]
    assert "field 257" in method_name
    assert f"{PACKAGED_SATURATION[shot]}/25000" in method_name
    assert "0 inside the plasma window" in method_name


# --------------------------------------------------------------------------
# Synthetic shots: the cases the packaged data does not exercise
# --------------------------------------------------------------------------


SYNTHETIC_SHOT = 41672
FIELD = 257
HALPHA_FIELD = 101
IP_FIELD = 109
N_SAMPLES = 25000
DT = 4e-5


def _packaged_fields(shot):
    with gzip.open(SAMPLE_SOURCE.format(shot=shot), "rt", encoding="utf-8") as handle:
        return json.load(handle)["fields"]


def _dump_with_diamagnetic(tmp_path, values):
    """A raw dump reusing a packaged shot, with field 257 replaced."""
    fields = _packaged_fields(SYNTHETIC_SHOT)
    fields[str(FIELD)] = {"data": np.asarray(values, dtype=float).tolist(), "type": "slow"}
    path = tmp_path / f"vest_{SYNTHETIC_SHOT}_daq_raw.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump({"shot": SYNTHETIC_SHOT, "fields": fields}, handle)
    return str(path)


def test_saturation_inside_the_plasma_window_warns_and_changes_the_flux(tmp_path):
    """Where the packaged shots are silent, the correction must be live."""
    clean = np.asarray(_packaged_fields(SYNTHETIC_SHOT)[str(FIELD)]["data"], dtype=float)
    time = np.arange(clean.size) * DT
    window = (time >= 0.32) & (time <= 0.33)
    railed = clean.copy()
    railed[window] = np.where(clean[window] >= 0.0, POSITIVE_RAIL, NEGATIVE_RAIL)
    source = _dump_with_diamagnetic(tmp_path, railed)

    report = magnetics.diamagnetic_saturation_report(
        SYNTHETIC_SHOT, raw_source=source, plasma_start=0.30, plasma_end=0.36
    )
    assert report["n_saturated_in_window"] > 100

    with pytest.warns(RuntimeWarning, match="inside the plasma window"):
        _, flux, detail = magnetics.vest_diamagnetic_flux_detailed(
            SYNTHETIC_SHOT, 0.30, 0.36, raw_source=source
        )
    assert detail["repaired"] is True
    _, baseline = magnetics.vest_diamagnetic_flux(
        SYNTHETIC_SHOT, 0.30, 0.36, raw_source=SAMPLE_SOURCE
    )
    assert np.abs(flux - baseline).max() > 1e-6


def test_saturation_running_to_the_end_of_the_record_is_not_extrapolated(tmp_path):
    clean = np.asarray(_packaged_fields(SYNTHETIC_SHOT)[str(FIELD)]["data"], dtype=float)
    railed = clean.copy()
    railed[-50:] = NEGATIVE_RAIL
    source = _dump_with_diamagnetic(tmp_path, railed)
    with pytest.raises(SignalRepairError, match="reaches the start or end"):
        magnetics.vest_diamagnetic_flux(SYNTHETIC_SHOT, 0.30, 0.36, raw_source=source)


def test_an_unsaturated_channel_reports_cleanly(tmp_path):
    source = _dump_with_diamagnetic(tmp_path, np.zeros(N_SAMPLES))
    report = magnetics.diamagnetic_saturation_report(SYNTHETIC_SHOT, raw_source=source)
    assert report["n_saturated"] == 0
    assert report["repaired"] is False
    assert report["reason"] == "no sample reached the acquisition limit"
    assert "no acquisition-limit saturation detected" in magnetics._diamagnetic_method_name(report)
