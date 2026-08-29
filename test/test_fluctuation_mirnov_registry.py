"""Registry/config regression tests for the outboard fluctuation-Mirnov array (issue #155)."""

from unittest.mock import patch

import numpy as np
import pytest

import vaft.machine_mapping.magnetics as mapping_magnetics
from vaft.machine_mapping.magnetics import (
    FLUCTUATION_MIRNOV_FIRST_SHOT,
    OUTBOARD_MIRNOV_MAJOR_RADIUS,
    fluctuation_mirnov_channel_definitions,
    fluctuation_mirnov_gain_by_identifier,
    fluctuation_mirnov_probe_indices,
    select_fluctuation_mirnov_channels,
)
from vaft.machine_mapping.utils import VestConfigurationError, get_path, resolve_data_root
from vaft.plot.mirnov import _normalise_channels


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
    gains = fluctuation_mirnov_gain_by_identifier()
    for channel in fluctuation_mirnov_channel_definitions():
        assert channel["identifier"] in gains
        assert gains[channel["identifier"]] == float(channel["gain"])
    assert "OutMirnov_530_Bz:phase_reference" in gains


def test_fluctuation_mirnov_first_shot_boundary_is_44156():
    assert FLUCTUATION_MIRNOV_FIRST_SHOT == 44156


NATIVE_SAMPLE_RATE = 250_000.0
INJECTED_N = 2
INJECTED_FREQUENCY = 10_000.0
INJECTED_PHASE_OFFSET = 0.3


def _synthetic_fluctuation_raw_loader():
    """A raw source that emits a known toroidal phase progression.

    The per-channel toroidal angle is read back out of the configuration under
    test, so the injected geometry and the mapped geometry cannot silently
    drift apart.
    """
    by_field = {int(c["field"]): c for c in fluctuation_mirnov_channel_definitions()}
    time = np.arange(0.2, 0.4, 1.0 / NATIVE_SAMPLE_RATE)

    def _load(shot, field, raw_source=None):
        channel = by_field.get(int(field))
        if channel is None:
            return None
        phase = INJECTED_PHASE_OFFSET - INJECTED_N * np.radians(channel["toroidal_angle_deg"])
        data = np.sin(2.0 * np.pi * INJECTED_FREQUENCY * time + phase)
        return time, data

    return _load


def _map_synthetic_fluctuation_shot(shot=44740, tstart=0.26, tend=0.36, loader=None):
    """Run the real mapper over a synthetic raw source, equilibrium stages stubbed."""
    from vaft.machine_mapping.magnetics import vfit_magnetics_dynamic, vfit_magnetics_static

    def _fake_equilibrium_magnetics(*args, **kwargs):
        del args, kwargs
        time = np.array([tstart, 0.5 * (tstart + tend), tend])
        return time, [np.array([1.0, 2.0, 3.0])], [np.array([4.0, 5.0, 6.0])]

    payload = {}
    vfit_magnetics_static(payload)
    with (
        patch(
            "vaft.machine_mapping.magnetics.vfit_equilibrium_magnetics",
            side_effect=_fake_equilibrium_magnetics,
        ),
        patch(
            "vaft.machine_mapping.magnetics._safe_vest_load",
            side_effect=loader if loader is not None else _synthetic_fluctuation_raw_loader(),
        ),
        patch(
            "vaft.machine_mapping.magnetics.vfit_plasma_current",
            return_value=(np.array([tstart, tend]), np.zeros(2)),
        ),
        patch("vaft.machine_mapping.magnetics._map_diamagnetic_flux"),
    ):
        vfit_magnetics_dynamic(payload, shot, tstart, tend, 4e-5)
    return payload


def test_toroidal_mode_recovered_end_to_end_through_the_mapper():
    """Synthetic raw source -> mapper -> ODS -> semantic discovery -> phase fit.

    This is issue #155 requirements 7, 8 and 10 in one path: the probes are
    found by geometry rather than by index, the voltage that comes back is at
    the native acquisition rate, and the injected toroidal mode number
    survives the round trip.
    """
    from vaft.process.magnetics import toroidal_phase_fit_at_time

    payload = _map_synthetic_fluctuation_shot()

    # Discover the midplane L1 probes semantically -- no index literals.
    selected = select_fluctuation_mirnov_channels(sub_array="L1", z_range=(-0.05, 0.05))
    assert [c["toroidal_angle_deg"] for c in selected] == [45.0, 135.0, 225.0]

    indices = fluctuation_mirnov_probe_indices(payload)
    assert len(indices) == 30

    signals = []
    angles = []
    for channel in selected:
        index = indices[channel["identifier"]]
        base = f"magnetics.b_field_pol_probe.{index}"
        assert get_path(payload, f"{base}.voltage.validity") == 0
        time = np.asarray(get_path(payload, f"{base}.voltage.time"))
        signals.append(np.asarray(get_path(payload, f"{base}.voltage.data")))
        angles.append(float(get_path(payload, f"{base}.toroidal_angle")))

    # Cropped to the analysis window, but never resampled onto the 25 kHz
    # equilibrium grid (issue #136).
    assert time.size == round((0.36 - 0.26) * NATIVE_SAMPLE_RATE)
    assert np.isclose(np.median(np.diff(time)), 1.0 / NATIVE_SAMPLE_RATE)
    assert np.allclose(angles, np.radians([45.0, 135.0, 225.0]))

    result = toroidal_phase_fit_at_time(
        time,
        np.vstack(signals),
        np.asarray(angles),
        center_time=float(time[time.size // 2]),
        sample_rate=NATIVE_SAMPLE_RATE,
        window_size=512,
        frequencies=[INJECTED_FREQUENCY],
        candidate_n=range(0, 5),
    )
    assert len(result.modes) == 1
    assert result.modes[0].n == INJECTED_N
    assert result.modes[0].rms_error < 0.05


def test_missing_fluctuation_channel_stays_empty_and_invalid():
    """Requirement 9, on a fluctuation probe: unavailable is not zero-filled."""
    base_loader = _synthetic_fluctuation_raw_loader()

    def _loader_missing_one(shot, field, raw_source=None):
        if int(field) == 301:  # OutMirno_225_L2-01, the typo'd raw label
            return None
        return base_loader(shot, field, raw_source)

    payload = _map_synthetic_fluctuation_shot(loader=_loader_missing_one)
    indices = fluctuation_mirnov_probe_indices(payload)

    missing = f"magnetics.b_field_pol_probe.{indices['OutMirnov_225_L2-01']}"
    assert get_path(payload, f"{missing}.voltage.validity") == -2
    assert np.asarray(get_path(payload, f"{missing}.voltage.data")).size == 0
    assert np.asarray(get_path(payload, f"{missing}.voltage.time")).size == 0

    # Its neighbours are untouched.
    present = f"magnetics.b_field_pol_probe.{indices['OutMirnov_225_L2-02']}"
    assert get_path(payload, f"{present}.voltage.validity") == 0
    assert np.asarray(get_path(payload, f"{present}.voltage.data")).size > 0


# --- Step 1: the inventory is vest.yaml configuration, not a geometry asset ---


def test_inventory_lives_in_vest_yaml_and_the_geometry_asset_is_gone():
    """The 30-channel table is mapping configuration, so it belongs in vest.yaml.

    The old `vaft/data/geometry/FluctuationMirnov.yaml` must not linger: two
    copies of a calibration table is exactly how the two silently diverge.
    """
    assert not (resolve_data_root() / "geometry" / "FluctuationMirnov.yaml").exists()

    config = mapping_magnetics._fluctuation_mirnov_config()
    assert len(config["channels"]) == 30
    assert config["first_operational_shot"] == FLUCTUATION_MIRNOV_FIRST_SHOT
    assert config["geometry"]["major_radius"] == OUTBOARD_MIRNOV_MAJOR_RADIUS


def test_array_level_defaults_are_merged_into_every_channel():
    """`role` and `preserve_native_voltage` are live config, not documentation."""
    for channel in fluctuation_mirnov_channel_definitions():
        assert channel["role"] == "fluctuation"
        assert channel["preserve_native_voltage"] is True
        assert channel["unit"] == "V"


# --- Step 1: loader hardening ---


def _config_with(channels, **overrides):
    config = {
        "source": {"raw_unit": "V", "daq_type": "fast"},
        "role": "fluctuation",
        "preserve_native_voltage": True,
        "first_operational_shot": 44156,
        "geometry": {"major_radius": 0.796},
        "channels": channels,
    }
    config.update(overrides)
    return config


def _loaded_from(config):
    with patch.object(mapping_magnetics, "_fluctuation_mirnov_config", return_value=config):
        mapping_magnetics._load_fluctuation_mirnov_channels.cache_clear()
        try:
            return mapping_magnetics._load_fluctuation_mirnov_channels()
        finally:
            mapping_magnetics._load_fluctuation_mirnov_channels.cache_clear()


def test_mutating_a_returned_channel_cannot_corrupt_the_cache():
    first = fluctuation_mirnov_channel_definitions()
    first[0]["gain"] = 12345.0
    first[0]["identifier"] = "clobbered"
    second = fluctuation_mirnov_channel_definitions()
    assert second[0]["identifier"] == "OutMirnov_45_L1-01"
    assert second[0]["gain"] == -0.00105


def test_malformed_identifier_is_rejected():
    config = _config_with(
        [{"field": 286, "raw_label": "x", "identifier": "OutMirnov_45_L3-01",
          "toroidal_angle_deg": 45, "z": 0.4, "gain": 0.001}]
    )
    with pytest.raises(VestConfigurationError, match="naming schema"):
        _loaded_from(config)


def test_identifier_angle_disagreeing_with_toroidal_angle_deg_is_rejected():
    config = _config_with(
        [{"field": 286, "raw_label": "x", "identifier": "OutMirnov_45_L1-01",
          "toroidal_angle_deg": 135, "z": 0.4, "gain": 0.001}]
    )
    with pytest.raises(VestConfigurationError, match="disagrees"):
        _loaded_from(config)


def test_duplicate_identifier_is_rejected():
    entry = {"field": 286, "raw_label": "x", "identifier": "OutMirnov_45_L1-01",
             "toroidal_angle_deg": 45, "z": 0.4, "gain": 0.001}
    config = _config_with([entry, dict(entry, field=304)])
    with pytest.raises(VestConfigurationError, match="duplicate identifier"):
        _loaded_from(config)


def test_duplicate_raw_field_is_rejected():
    config = _config_with([
        {"field": 286, "raw_label": "x", "identifier": "OutMirnov_45_L1-01",
         "toroidal_angle_deg": 45, "z": 0.4, "gain": 0.001},
        {"field": 286, "raw_label": "y", "identifier": "OutMirnov_45_L1-02",
         "toroidal_angle_deg": 45, "z": 0.2, "gain": 0.001},
    ])
    with pytest.raises(VestConfigurationError, match="already mapped"):
        _loaded_from(config)


# --- Step 2: semantic selection and ODS discovery ---


def test_select_by_toroidal_angle_and_sub_array():
    for angle in (45, 135, 225):
        assert len(select_fluctuation_mirnov_channels(toroidal_angle_deg=angle)) == 10
        for sub_array in ("L1", "L2"):
            selected = select_fluctuation_mirnov_channels(
                toroidal_angle_deg=angle, sub_array=sub_array
            )
            assert len(selected) == 5
            # Canonical order is z descending, never raw field-number order.
            assert [c["z"] for c in selected] == [0.4, 0.2, 0.0, -0.2, -0.4]
    assert len(select_fluctuation_mirnov_channels(toroidal_angle_deg=[45, 225])) == 20


def test_select_by_z_range_and_role():
    midplane = select_fluctuation_mirnov_channels(z_range=(-0.1, 0.1))
    assert {c["z"] for c in midplane} == {0.0}
    assert len(midplane) == 6  # three toroidal angles x two sub-arrays
    assert select_fluctuation_mirnov_channels(role="equilibrium") == ()
    assert len(select_fluctuation_mirnov_channels(role=None)) == 30


def test_unknown_selection_criteria_raise_rather_than_returning_nothing():
    """An empty tuple would read as "no probes installed there" -- a different
    and far more misleading answer than "you asked for something impossible"."""
    with pytest.raises(ValueError, match="sub-array"):
        select_fluctuation_mirnov_channels(sub_array="L3")
    with pytest.raises(ValueError, match="toroidal angle"):
        select_fluctuation_mirnov_channels(toroidal_angle_deg=90)


def test_probe_indices_resolve_semantically_and_ignore_unrelated_probes():
    ods = {"magnetics": {"b_field_pol_probe": {
        **{str(i): {"identifier": f"equilibrium-{i}"} for i in range(4)},
        "4": {"identifier": "OutMirnov_135_L2-03"},
        "5": {"identifier": "OutMirnov_45_L1-01"},
    }}}
    assert fluctuation_mirnov_probe_indices(ods) == {
        "OutMirnov_135_L2-03": 4,
        "OutMirnov_45_L1-01": 5,
    }


def test_probe_indices_reject_a_duplicated_identifier():
    ods = {"magnetics": {"b_field_pol_probe": {
        "0": {"identifier": "OutMirnov_45_L1-01"},
        "1": {"identifier": "OutMirnov_45_L1-01"},
    }}}
    with pytest.raises(VestConfigurationError, match="appears at both"):
        fluctuation_mirnov_probe_indices(ods)


def test_probe_indices_on_an_ods_without_probes_is_empty_not_an_error():
    assert fluctuation_mirnov_probe_indices({}) == {}


# --- Review follow-ups on the public API surface ---


def test_mutating_the_gain_table_cannot_corrupt_the_cache():
    """Same hazard the channel definitions already guard, on the gain lookup."""
    gains = fluctuation_mirnov_gain_by_identifier()
    expected = len(gains)
    gains.pop("OutMirnov_45_L1-01")
    gains["OutMirnov_135_L2-03"] = 999.0
    refetched = fluctuation_mirnov_gain_by_identifier()
    assert len(refetched) == expected
    assert refetched["OutMirnov_45_L1-01"] == -0.00105
    assert refetched["OutMirnov_135_L2-03"] == -0.0025


def test_numpy_scalars_are_accepted_as_a_single_toroidal_angle():
    """Mode-analysis code hands over NumPy scalars, not just Python ints."""
    for angle in (45, 45.0, np.int64(45), np.float32(45), np.array([45, 135])[0]):
        assert len(select_fluctuation_mirnov_channels(toroidal_angle_deg=angle)) == 10
    assert len(select_fluctuation_mirnov_channels(toroidal_angle_deg=np.array([45, 135]))) == 20


def test_reversed_z_range_raises_rather_than_selecting_nothing():
    """Canonical order is z descending, so writing the bounds that way is an easy slip."""
    with pytest.raises(ValueError, match="reversed"):
        select_fluctuation_mirnov_channels(z_range=(0.4, -0.4))


def test_a_revision_without_from_shot_is_rejected():
    """shot=0 must stay the base inventory: the import-time constants and the
    shot-less gain lookup both depend on nothing shadowing it."""
    config = {0: {"diagnostics": {"fluctuation_mirnov": {
        "channels": [], "revisions": [{"to_shot": 50000, "gain": 1.0}],
    }}}}
    with patch.object(mapping_magnetics, "load_yaml", return_value=config):
        mapping_magnetics._fluctuation_mirnov_config.cache_clear()
        try:
            with pytest.raises(VestConfigurationError, match="from_shot is required"):
                mapping_magnetics._fluctuation_mirnov_config()
        finally:
            mapping_magnetics._fluctuation_mirnov_config.cache_clear()
