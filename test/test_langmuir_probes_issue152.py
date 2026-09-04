import logging
from unittest.mock import patch

import numpy as np
import pytest

import vaft.machine_mapping.langmuir_probes as lp
from vaft.database.raw import RawSignalUnavailableError
from vaft.process.langmuir import (
    calibrate_current,
    calibrate_voltage,
    electron_density,
    probe_surface_area,
    process_triple_probe,
    remove_offset,
    solve_electron_temperature,
)

MID_SHOT_EARLY = 42000  # era: vd3=38.0, tip_length=1.5mm, tip_radius=0.05mm
MID_SHOT_MID = 42300  # era: vd3=48.0
MID_SHOT_GAP = 42140  # documented gap
MID_SHOT_SINGLE_GAP = 42641  # documented single-shot gap
UPPER_SHOT_PRE_INSTALL = 42000  # before UPPER_PROBE_FIRST_SHOT
UPPER_SHOT_POST_INSTALL = 42800  # after install, era: vd3=64.0


def _synthetic_vd2(vd3: float, te: float) -> float:
    rhs = 0.5 * (1.0 - np.exp(-vd3 / te))
    return -te * np.log(1.0 - rhs)


def _make_fake_loader(voltage_field, current_field, *, present_fields=None, n=2000, te=5.0, vd3=48.0,
                       gain=22.0, divisor=100.0, current_true=0.02):
    present_fields = present_fields if present_fields is not None else {voltage_field, current_field}
    t = np.linspace(0.0, 0.1, n)
    vd2_true = _synthetic_vd2(vd3, te)
    vd2_raw = np.full(n, vd2_true / gain)
    i_raw = np.full(n, current_true / divisor)
    vd2_raw[:500] = 0.0
    i_raw[:500] = 0.0

    def fake_load(shot, field, sample_opt=False):
        if field not in present_fields:
            return None
        if field == voltage_field:
            return t, vd2_raw
        if field == current_field:
            return t, i_raw
        return None

    return fake_load


# --- process.langmuir unit tests -------------------------------------------------


def test_remove_offset_subtracts_baseline_mean():
    data = np.concatenate([np.full(500, 3.0), np.full(10, 3.0) + 1.0])
    result = remove_offset(data, n_baseline_samples=500)
    np.testing.assert_allclose(result[:500], 0.0)
    np.testing.assert_allclose(result[500:], 1.0)


def test_remove_offset_requires_enough_samples():
    with pytest.raises(ValueError):
        remove_offset(np.zeros(10), n_baseline_samples=500)


def test_calibration_factors():
    np.testing.assert_allclose(calibrate_voltage(np.array([1.0, 2.0]), gain=22.0), [22.0, 44.0])
    np.testing.assert_allclose(calibrate_current(np.array([100.0, 200.0]), divisor=100.0), [1.0, 2.0])
    with pytest.raises(ValueError):
        calibrate_current(np.array([1.0]), divisor=0.0)


def test_probe_surface_area():
    area = probe_surface_area(tip_radius_m=0.15e-3, tip_length_m=2.0e-3)
    assert area == pytest.approx(2 * np.pi * 0.15e-3 * 2.0e-3)
    with pytest.raises(ValueError):
        probe_surface_area(tip_radius_m=0.0, tip_length_m=1.0)


def test_solve_electron_temperature_round_trips_a_self_consistent_signal():
    te_true = 5.0
    vd3 = 48.0
    vd2 = _synthetic_vd2(vd3, te_true)
    te, ok = solve_electron_temperature(np.array([vd2]), np.array([vd3]))
    assert ok[0]
    assert te[0] == pytest.approx(te_true, rel=1e-4)


def test_solve_electron_temperature_flags_nonphysical_zero_signal():
    te, ok = solve_electron_temperature(np.array([0.0]), np.array([48.0]))
    assert not ok[0]
    assert np.isnan(te[0])


def test_electron_density_requires_matching_shapes_and_positive_mass():
    with pytest.raises(ValueError):
        electron_density(
            np.array([1.0, 2.0]), np.array([1.0]), np.array([1.0]),
            tip_radius_m=1e-4, tip_length_m=1e-3, ion_mass_kg=1.67e-27,
        )
    with pytest.raises(ValueError):
        electron_density(
            np.array([1.0]), np.array([1.0]), np.array([1.0]),
            tip_radius_m=1e-4, tip_length_m=1e-3, ion_mass_kg=0.0,
        )


def test_electron_density_is_si_and_flags_nonphysical_results():
    te_true = 5.0
    vd3 = 48.0
    vd2 = _synthetic_vd2(vd3, te_true)
    n_e, valid = electron_density(
        np.array([vd2]), np.array([te_true]), np.array([0.02]),
        tip_radius_m=0.15e-3, tip_length_m=2.0e-3, ion_mass_kg=1.67262192369e-27,
    )
    assert valid[0]
    assert n_e[0] > 0
    assert np.isfinite(n_e[0])

    # A negative/zero current with this Vd2/Te sign combination yields a
    # nonphysical (non-positive) density and must be flagged, not clipped.
    n_e_bad, valid_bad = electron_density(
        np.array([vd2]), np.array([te_true]), np.array([-0.02]),
        tip_radius_m=0.15e-3, tip_length_m=2.0e-3, ion_mass_kg=1.67262192369e-27,
    )
    assert not valid_bad[0]
    assert np.isnan(n_e_bad[0])


def test_process_triple_probe_rejects_misaligned_time_coordinates():
    t = np.linspace(0, 0.1, 100)
    t_other = np.linspace(0, 0.1, 101)
    with pytest.raises(ValueError, match="time coordinates"):
        process_triple_probe(
            t, np.zeros(100), t_other, np.zeros(101), 48.0,
            tip_radius_m=1e-4, tip_length_m=1e-3, ion_mass_kg=1.67e-27,
        )


def test_process_triple_probe_recovers_known_te_end_to_end():
    n = 2000
    t = np.linspace(0, 0.1, n)
    te_true = 5.0
    vd3 = 48.0
    vd2_true = _synthetic_vd2(vd3, te_true)
    vd2_raw = np.full(n, vd2_true / 22.0)
    i_raw = np.full(n, 0.02 / 100.0)
    vd2_raw[:500] = 0.0
    i_raw[:500] = 0.0

    result = process_triple_probe(
        t, vd2_raw, t, i_raw, vd3,
        tip_radius_m=0.15e-3, tip_length_m=2.0e-3, ion_mass_kg=1.67262192369e-27,
    )
    assert result["solver_ok"][1000]
    assert result["te"][1000] == pytest.approx(te_true, rel=1e-3)


# --- machine_mapping.langmuir_probes: shot-era configuration ---------------------


@pytest.mark.parametrize(
    ("assembly_key", "shot", "expected_vd3"),
    [
        ("mid", 42000, 38.0),
        ("mid", 42145, 48.0),
        ("mid", 42640, 48.0),
        ("mid", 42642, 48.0),
        ("mid", 43828, 48.0),
        ("mid", 43829, 57.5),
        ("upper", 42675, 64.0),
        ("upper", 43665, 64.0),
        ("upper", 43666, 73.4),
        ("upper", 44065, 73.4),
        ("upper", 44066, 102.5),
        ("upper", 44280, 102.5),
        ("upper", 46311, 164.5),
        ("upper", 46312, 188.0),
        ("upper", 47257, 188.0),
        ("upper", 47258, 89.8),
    ],
)
def test_shot_era_boundaries_resolve_the_documented_vd3(assembly_key, shot, expected_vd3):
    config = lp.resolve_langmuir_probe_config(assembly_key, shot)
    era = lp._resolve_era(config, shot, assembly_key=assembly_key)
    assert era["vd3"] == pytest.approx(expected_vd3)


@pytest.mark.parametrize("shot", [MID_SHOT_GAP, MID_SHOT_SINGLE_GAP])
def test_documented_mid_probe_gaps_raise_clearly(shot):
    config = lp.resolve_langmuir_probe_config("mid", shot)
    with pytest.raises(lp.LangmuirProbeConfigError, match="unresolved era gap"):
        lp._resolve_era(config, shot, assembly_key="mid")


def test_documented_upper_probe_gap_raises_clearly():
    config = lp.resolve_langmuir_probe_config("upper", 42140)
    with pytest.raises(lp.LangmuirProbeConfigError, match="unresolved era gap"):
        lp._resolve_era(config, 42140, assembly_key="upper")


# --- machine_mapping.langmuir_probes: mapper behavior -----------------------------


def test_upper_probe_absent_before_installation_shot():
    # Both mid and upper fields report data present -- the point is that the
    # structural 42675 gate must still suppress embedded.1 on its own, not
    # merely because the upper signal happens to be missing.
    fake_load = _make_fake_loader(99, 100, present_fields={99, 100, 259, 260}, vd3=38.0)

    def combined_loader(shot, field, sample_opt=False):
        if field in (99, 100):
            return fake_load(shot, field, sample_opt=sample_opt)
        if field in (259, 260):
            return np.linspace(0, 0.1, 2000), np.zeros(2000)
        return None

    with patch.object(lp.raw_db, "vest_load", side_effect=combined_loader):
        ods = {}
        lp.langmuir_probes(ods, UPPER_SHOT_PRE_INSTALL, 0.0, 0.1, 1e-3)
    embedded = ods["langmuir_probes"]["embedded"]
    assert len(embedded) == 1
    assert embedded[0] is not None


def test_upper_probe_present_after_installation_shot():
    fake_load = _make_fake_loader(259, 260, vd3=64.0)
    with patch.object(lp.raw_db, "vest_load", side_effect=fake_load):
        ods = {}
        lp.langmuir_probes(ods, UPPER_SHOT_POST_INSTALL, 0.0, 0.1, 1e-3)
    embedded = ods["langmuir_probes"]["embedded"]
    # The mid probe is absent from this fake loader, so the upper probe fills
    # the first embedded slot: IMAS arrays of structures must stay contiguous.
    assert len(embedded) == 1
    assert embedded[0]["name"] == "Upper triple Langmuir probe"


def test_upper_probe_alone_writes_a_contiguous_omas_embedded_array():
    # Regression (release review): with the mid probe not operated, writing
    # the upper probe to its nominal slot 1 left embedded.0 unfilled, which a
    # real OMAS ODS rejects with an IndexError.
    omas = pytest.importorskip("omas")
    fake_load = _make_fake_loader(259, 260, vd3=64.0)
    with patch.object(lp.raw_db, "vest_load", side_effect=fake_load):
        ods = omas.ODS()
        lp.langmuir_probes(ods, UPPER_SHOT_POST_INSTALL, 0.0, 0.1, 1e-3)
    assert len(ods["langmuir_probes.embedded"]) == 1
    assert ods["langmuir_probes.embedded.0.name"] == "Upper triple Langmuir probe"


def test_probe_not_operated_this_shot_is_silently_absent_not_an_error():
    def fake_load(shot, field, sample_opt=False):
        return None

    with patch.object(lp.raw_db, "vest_load", side_effect=fake_load):
        ods = {}
        # No exception, even though 42140 is also a documented mid-probe gap:
        # presence is checked (and fails) before the era lookup ever runs.
        lp.langmuir_probes(ods, MID_SHOT_MID, 0.0, 0.1, 1e-3)
    assert "embedded" not in ods.get("langmuir_probes", {})


def test_operated_probe_in_a_documented_gap_shot_raises():
    fake_load = _make_fake_loader(99, 100, vd3=48.0)
    with patch.object(lp.raw_db, "vest_load", side_effect=fake_load):
        ods = {}
        with pytest.raises(lp.LangmuirProbeConfigError, match="unresolved era gap"):
            lp.langmuir_probes(ods, MID_SHOT_GAP, 0.0, 0.1, 1e-3)


def test_mid_probe_processes_signals_correctly_and_channel_isolated():
    fake_load = _make_fake_loader(99, 100, vd3=38.0)
    with patch.object(lp.raw_db, "vest_load", side_effect=fake_load):
        ods = {}
        lp.langmuir_probes(ods, MID_SHOT_EARLY, 0.0, 0.1, 1e-3)
    assert set(ods.keys()) == {"langmuir_probes"}
    mid = ods["langmuir_probes"]["embedded"][0]
    assert mid["position"]["z"] == pytest.approx(lp.MID_Z_M)
    assert "position" not in mid or "r" not in mid["position"]
    assert np.nanmax(mid["t_e"]["data"]) == pytest.approx(5.0, rel=1e-2)
    assert mid["surface_area"] == pytest.approx(probe_surface_area(tip_radius_m=0.05e-3, tip_length_m=1.5e-3))


def test_position_is_independent_of_signal_processing():
    fake_load = _make_fake_loader(99, 100, vd3=38.0)
    with patch.object(lp.raw_db, "vest_load", side_effect=fake_load):
        ods_without_position = {}
        lp.langmuir_probes(ods_without_position, MID_SHOT_EARLY, 0.0, 0.1, 1e-3)
        ods_with_position = {}
        lp.langmuir_probes(ods_with_position, MID_SHOT_EARLY, 0.0, 0.1, 1e-3, mid_r=0.7)

    mid_without = ods_without_position["langmuir_probes"]["embedded"][0]
    mid_with = ods_with_position["langmuir_probes"]["embedded"][0]
    assert "r" not in mid_without["position"]
    assert mid_with["position"]["r"] == pytest.approx(0.7)
    np.testing.assert_allclose(mid_without["t_e"]["data"], mid_with["t_e"]["data"])
    np.testing.assert_allclose(mid_without["n_e"]["data"], mid_with["n_e"]["data"])


def test_time_coordinate_mismatch_between_voltage_and_current_raises():
    t = np.linspace(0, 0.1, 2000)
    t_other = np.linspace(0, 0.1, 1999)

    def fake_load(shot, field, sample_opt=False):
        if field == 99:
            return t, np.ones(2000)
        if field == 100:
            return t_other, np.ones(1999)
        return None

    with patch.object(lp.raw_db, "vest_load", side_effect=fake_load):
        ods = {}
        with pytest.raises(ValueError, match="time coordinates"):
            lp.langmuir_probes(ods, MID_SHOT_EARLY, 0.0, 0.1, 1e-3)


def test_solver_failure_flags_invalid_samples_instead_of_a_plausible_value():
    n = 2000
    t = np.linspace(0, 0.1, n)
    vd2_raw = np.zeros(n)  # Vd2 = 0 for the whole shot: no self-consistent Te
    i_raw = np.zeros(n)

    def fake_load(shot, field, sample_opt=False):
        if field == 99:
            return t, vd2_raw
        if field == 100:
            return t, i_raw
        return None

    with patch.object(lp.raw_db, "vest_load", side_effect=fake_load):
        ods = {}
        lp.langmuir_probes(ods, MID_SHOT_EARLY, 0.0, 0.1, 1e-3)
    mid = ods["langmuir_probes"]["embedded"][0]
    assert np.all(mid["t_e"]["validity_timed"] < 0)
    assert np.all(np.isnan(mid["t_e"]["data"]))


def test_live_and_archived_raw_source_are_equivalent():
    fake_load = _make_fake_loader(99, 100, vd3=38.0)
    results = {}
    for raw_source, label in ((None, "live"), ("archive.json.gz", "archived")):
        with patch.object(lp.raw_db, "vest_load", side_effect=fake_load):
            ods = {}
            lp.langmuir_probes(ods, MID_SHOT_EARLY, 0.0, 0.1, 1e-3, raw_source=raw_source)
        results[label] = ods["langmuir_probes"]["embedded"][0]
    np.testing.assert_allclose(results["live"]["t_e"]["data"], results["archived"]["t_e"]["data"])
    np.testing.assert_allclose(results["live"]["n_e"]["data"], results["archived"]["n_e"]["data"])


def test_missing_raw_signal_after_presence_check_still_raises(monkeypatch):
    # Presence check sees a real signal, but require_signal then finds it
    # malformed -- this is a genuine data problem, not "not operated".
    def fake_load(shot, field, sample_opt=False):
        if field == 99:
            return np.array([0.0, 1.0]), np.array([0.0, 1.0, 2.0])  # mismatched lengths
        if field == 100:
            return np.array([0.0, 1.0]), np.array([0.0, 1.0])
        return None

    with patch.object(lp.raw_db, "vest_load", side_effect=fake_load):
        ods = {}
        with pytest.raises(RawSignalUnavailableError):
            lp.langmuir_probes(ods, MID_SHOT_EARLY, 0.0, 0.1, 1e-3)


# --- apply_langmuir_probe_measured_positions --------------------------------------


def _write_position_csv(tmp_path):
    csv_path = tmp_path / "positions.csv"
    csv_path.write_text(
        "shot,mid TP position[m],upper TP position[m]\n"
        f"{MID_SHOT_EARLY},0.70,0.55\n"
        f"{UPPER_SHOT_POST_INSTALL},0.71,0.56\n"
    , encoding="utf-8")
    return csv_path


def test_apply_measured_positions_updates_only_position_r(tmp_path):
    csv_path = _write_position_csv(tmp_path)
    fake_load = _make_fake_loader(99, 100, vd3=38.0)
    with patch.object(lp.raw_db, "vest_load", side_effect=fake_load):
        ods = {}
        lp.langmuir_probes(ods, MID_SHOT_EARLY, 0.0, 0.1, 1e-3)
        te_before = np.array(ods["langmuir_probes"]["embedded"][0]["t_e"]["data"])
        lp.apply_langmuir_probe_measured_positions(ods, MID_SHOT_EARLY, csv_path=csv_path)

    mid = ods["langmuir_probes"]["embedded"][0]
    assert mid["position"]["r"] == pytest.approx(0.70)
    np.testing.assert_allclose(np.array(mid["t_e"]["data"]), te_before)


def test_apply_measured_positions_leaves_shots_outside_csv_untouched(tmp_path):
    csv_path = _write_position_csv(tmp_path)
    fake_load = _make_fake_loader(99, 100, vd3=48.0)
    with patch.object(lp.raw_db, "vest_load", side_effect=fake_load):
        ods = {}
        lp.langmuir_probes(ods, MID_SHOT_MID, 0.0, 0.1, 1e-3)
        lp.apply_langmuir_probe_measured_positions(ods, MID_SHOT_MID, csv_path=csv_path)

    assert "r" not in ods["langmuir_probes"]["embedded"][0]["position"]


def test_apply_measured_positions_missing_csv_does_not_raise(caplog, tmp_path):
    fake_load = _make_fake_loader(99, 100, vd3=38.0)
    with patch.object(lp.raw_db, "vest_load", side_effect=fake_load):
        ods = {}
        lp.langmuir_probes(ods, MID_SHOT_EARLY, 0.0, 0.1, 1e-3)
        with caplog.at_level(logging.INFO):
            lp.apply_langmuir_probe_measured_positions(
                ods, MID_SHOT_EARLY, csv_path=tmp_path / "does_not_exist.csv"
            )
    assert "r" not in ods["langmuir_probes"]["embedded"][0]["position"]


def test_apply_measured_positions_no_csv_path_does_not_raise():
    # Non-blocking by design even with no csv_path at all.
    lp.apply_langmuir_probe_measured_positions({}, MID_SHOT_EARLY, csv_path=None)


def test_measured_positions_follow_probe_identity_not_slot_number():
    """A skipped assembly must not shift another probe's measured radius.

    embedded[] is filled contiguously, so when the mid probe is not
    operated the upper probe occupies embedded[0]. Keying the measured
    position CSV off the slot number then wrote the *mid* probe's radius
    onto the upper probe (release review, 0.6.0): shot 42699 records
    mid=0.79 m, upper=0.59 m.
    """
    omas = pytest.importorskip("omas")

    shot = 42699
    fake_load = _make_fake_loader(259, 260, vd3=64.0)  # upper only; mid absent
    with patch.object(lp.raw_db, "vest_load", side_effect=fake_load):
        ods = omas.ODS()
        lp.vfit_langmuir_probes_dynamic(ods, shot, 0.0, 0.1, 1e-3)

    lp.apply_langmuir_probe_measured_positions(ods, shot)

    assert len(ods["langmuir_probes.embedded"]) == 1
    assert ods["langmuir_probes.embedded.0.identifier"] == "langmuir_probes:upper"
    assert float(ods["langmuir_probes.embedded.0.position.r"]) == pytest.approx(0.59)


def test_measured_positions_are_correct_when_both_probes_are_present():
    omas = pytest.importorskip("omas")

    shot = 42699
    t = np.linspace(0.0, 0.1, 2000)
    vd2 = np.full(2000, _synthetic_vd2(48.0, 5.0) / 22.0)
    current = np.full(2000, 0.02 / 100.0)
    vd2[:500] = 0.0
    current[:500] = 0.0

    def fake_load(_shot, field, sample_opt=False):
        if field in (99, 259):
            return t, vd2
        if field in (100, 260):
            return t, current
        return None

    with patch.object(lp.raw_db, "vest_load", side_effect=fake_load):
        ods = omas.ODS()
        lp.vfit_langmuir_probes_dynamic(ods, shot, 0.0, 0.1, 1e-3)

    lp.apply_langmuir_probe_measured_positions(ods, shot)

    by_identifier = {
        ods[f"langmuir_probes.embedded.{index}.identifier"]: float(
            ods[f"langmuir_probes.embedded.{index}.position.r"]
        )
        for index in range(len(ods["langmuir_probes.embedded"]))
    }
    assert by_identifier == {
        "langmuir_probes:mid": pytest.approx(0.79),
        "langmuir_probes:upper": pytest.approx(0.59),
    }
