"""Registry/config regression tests for the outboard fluctuation-Mirnov array (issue #155)."""

import numpy as np

from vaft.machine_mapping.magnetics import (
    FLUCTUATION_MIRNOV_FIRST_SHOT,
    fluctuation_mirnov_channel_definitions,
)
from vaft.plot.mirnov import _known_gain_by_identifier, _normalise_channels


def _defs_by_field():
    return {int(c["field"]): c for c in fluctuation_mirnov_channel_definitions()}


def test_all_thirty_raw_fields_present_exactly_once():
    defs = fluctuation_mirnov_channel_definitions()
    fields = [int(c["field"]) for c in defs]
    assert sorted(fields) == list(range(286, 316))
    assert len(fields) == len(set(fields)) == 30


def test_config_order_is_not_controlled_by_raw_field_number():
    """Canonical order is toroidal group -> L1/L2 -> z descending, not field order."""
    defs = fluctuation_mirnov_channel_definitions()
    fields = [int(c["field"]) for c in defs]
    assert fields != sorted(fields)

    # Within each contiguous run of 5, z must descend from +0.4 to -0.4.
    for start in range(0, 30, 5):
        group = defs[start : start + 5]
        z_values = [float(c["z"]) for c in group]
        assert z_values == [0.4, 0.2, 0.0, -0.2, -0.4]


def test_canonical_identifiers_angles_z_and_gains_match_the_issue_table():
    by_field = _defs_by_field()

    def check(field, identifier, angle_deg, z, gain):
        c = by_field[field]
        assert c["identifier"] == identifier
        assert float(c["toroidal_angle_deg"]) == angle_deg
        assert float(c["z"]) == z
        assert float(c["gain"]) == gain

    check(286, "OutMirnov_45_L1-01", 45, 0.4, -0.00105)
    check(288, "OutMirnov_45_L1-05", 45, -0.4, 0.00105)
    check(292, "OutMirnov_135_L1-01", 135, 0.4, 0.00096)
    check(297, "OutMirnov_135_L2-05", 135, -0.4, -0.0023)
    check(298, "OutMirnov_225_L1-01", 225, 0.4, 0.00105)
    check(303, "OutMirnov_225_L2-05", 225, -0.4, 0.00245)


def test_45deg_uses_s_to_l1_d_to_l2_with_reversed_source_sheet_vertical_order():
    """45deg S->L1, D->L2; source sheet numbers bottom-to-top, so DB -05..-01
    corresponds to z descending from +0.4 (-01) to -0.4 (-05)."""
    by_field = _defs_by_field()
    assert by_field[286]["identifier"] == "OutMirnov_45_L1-01"
    assert by_field[286]["z"] == 0.4
    assert by_field[288]["identifier"] == "OutMirnov_45_L1-05"
    assert by_field[288]["z"] == -0.4
    assert by_field[289]["identifier"] == "OutMirnov_45_L2-01"
    assert by_field[289]["z"] == 0.4
    assert by_field[291]["identifier"] == "OutMirnov_45_L2-05"
    assert by_field[291]["z"] == -0.4


def test_field_289_gain_is_estimated_with_recorded_provenance():
    by_field = _defs_by_field()
    channel = by_field[289]
    assert channel["identifier"] == "OutMirnov_45_L2-01"
    assert float(channel["gain"]) == 0.00105
    assert channel["calibration_provenance"] == "estimated"
    # No other channel should carry this provenance marker.
    others = [c for f, c in by_field.items() if f != 289]
    assert all(c.get("calibration_provenance") != "estimated" for c in others)


def test_raw_typo_field_301_is_read_while_identifier_is_normalized():
    by_field = _defs_by_field()
    channel = by_field[301]
    assert channel["raw_label"] == "OutMirno_225_L2-01"
    assert channel["identifier"] == "OutMirnov_225_L2-01"


def test_no_layer_field_is_encoded_separately():
    for channel in fluctuation_mirnov_channel_definitions():
        assert "layer" not in channel


def test_channels_are_resolved_by_identifier_not_fixed_index():
    """vaft.plot.mirnov resolves probes by identifier string, matching the
    identifier convention _populate_fluctuation_mirnov_static writes."""

    class _FakeODS(dict):
        pass

    ods = {
        "magnetics": {
            "b_field_pol_probe": {
                str(i): {"identifier": f"probe-{i}", "name": f"probe-{i}"}
                for i in range(70)
            }
        }
    }
    ods["magnetics"]["b_field_pol_probe"]["68"] = {
        "identifier": "OutMirnov_45_L1-01",
        "name": "OutMirnov_45_L1-01",
    }
    resolved = _normalise_channels(ods, "b_field_pol_probe", "OutMirnov_45_L1-01")
    assert resolved == [68]


def test_known_gain_lookup_covers_all_thirty_fluctuation_identifiers_and_toroidal_reference():
    gains = _known_gain_by_identifier()
    for channel in fluctuation_mirnov_channel_definitions():
        assert channel["identifier"] in gains
        assert gains[channel["identifier"]] == float(channel["gain"])
    assert "OutMirnov_530_Bz:phase_reference" in gains


def test_fluctuation_mirnov_first_shot_boundary_is_44156():
    assert FLUCTUATION_MIRNOV_FIRST_SHOT == 44156


def test_synthetic_toroidal_mode_recovered_from_45_135_225_channels():
    """Sanity check that fields at 45/135/225deg carry a coherent phase
    progression a toroidal-mode-number fit can recover -- reuses the existing
    toroidal_phase_fit_at_time algorithm, no new physics."""
    from vaft.process.magnetics import toroidal_phase_fit_at_time

    sample_rate = 100_000.0
    time = np.arange(4096, dtype=float) / sample_rate
    angles = np.radians([45.0, 135.0, 225.0])
    expected_n = 2
    frequency = 10_000.0
    phases = 0.3 - expected_n * angles
    signals = np.vstack([np.sin(2.0 * np.pi * frequency * time + phase) for phase in phases])

    result = toroidal_phase_fit_at_time(
        time,
        signals,
        angles,
        center_time=time[len(time) // 2],
        sample_rate=sample_rate,
        window_size=512,
        frequencies=[frequency],
        candidate_n=range(0, 5),
    )
    assert len(result.modes) == 1
    assert result.modes[0].n == expected_n
    assert result.modes[0].rms_error < 0.05
