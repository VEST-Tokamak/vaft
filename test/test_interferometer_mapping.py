"""Tests for the VEST interferometer mapper (issue #153)."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.io as sio
from omas import ODS

from vaft.data.resources import data_path, require_repository_sample
from vaft.machine_mapping.interferometer import (
    interferometer,
    interferometer_94ghz,
    interferometer_282ghz,
)

SHOT = 47230


def _write_94ghz_mat(path, shot=SHOT, n_time=50, n_channels=5, shot_override=None):
    time = 0.29 + np.arange(n_time) * 4e-6
    rng = np.random.default_rng(0)
    line_den = 1e18 * rng.random((n_time, n_channels))
    sio.savemat(
        str(path),
        {
            "line_den": line_den,
            "time": time.reshape(-1, 1),
            "shotNum": np.array([[shot_override if shot_override is not None else shot]], dtype=np.uint16),
            "trigger_time": np.array([[0]], dtype=np.uint8),
        },
    )
    return time, line_den


def _write_282ghz_mat(path, shot=SHOT, n_time=60):
    time = 0.29 + np.arange(n_time) * 8e-6
    rng = np.random.default_rng(1)
    line_den = 1e18 * rng.random((n_time, 1))
    sio.savemat(
        str(path),
        {
            "line_den": line_den,
            "time": time.reshape(-1, 1),
            "shotNum": np.array([[shot]], dtype=np.uint16),
            "trigger_time": np.array([[0]], dtype=np.uint8),
        },
    )
    return time, line_den


# --- file/array validation -------------------------------------------------


def test_shot_mismatch_raises(tmp_path):
    mat = tmp_path / "mismatch.mat"
    _write_94ghz_mat(mat, shot_override=47231)
    ods = ODS()
    with pytest.raises(ValueError, match="shotNum"):
        interferometer_94ghz(ods, SHOT, mat_file=str(mat))


def test_non_monotonic_time_raises(tmp_path):
    mat = tmp_path / "nonmono.mat"
    time, line_den = _write_94ghz_mat(mat)
    data = sio.loadmat(str(mat))
    time_arr = data["time"].reshape(-1)
    time_arr[10] = time_arr[9]
    clean = {k: v for k, v in data.items() if not k.startswith("__")}
    sio.savemat(str(mat), {**clean, "time": time_arr.reshape(-1, 1)})
    ods = ODS()
    with pytest.raises(ValueError, match="monotonic"):
        interferometer_94ghz(ods, SHOT, mat_file=str(mat))


def test_non_finite_time_raises(tmp_path):
    mat = tmp_path / "nonfinite.mat"
    _write_94ghz_mat(mat)
    data = sio.loadmat(str(mat))
    time_arr = data["time"].reshape(-1).astype(float)
    time_arr[5] = np.nan
    clean = {k: v for k, v in data.items() if not k.startswith("__")}
    sio.savemat(str(mat), {**clean, "time": time_arr.reshape(-1, 1)})
    ods = ODS()
    with pytest.raises(ValueError, match="non-finite"):
        interferometer_94ghz(ods, SHOT, mat_file=str(mat))


def test_94ghz_wrong_column_count_raises(tmp_path):
    mat = tmp_path / "wrongcols.mat"
    _write_94ghz_mat(mat, n_channels=4)
    ods = ODS()
    with pytest.raises(ValueError, match="5"):
        interferometer_94ghz(ods, SHOT, mat_file=str(mat))


def test_282ghz_wrong_column_count_raises(tmp_path):
    mat = tmp_path / "wrongcols282.mat"
    _write_94ghz_mat(mat, n_channels=5)  # 5 columns, but 282GHz expects 1
    ods = ODS()
    with pytest.raises(ValueError, match="1"):
        interferometer_282ghz(ods, SHOT, mat_file=str(mat))


def test_trigger_time_does_not_affect_mapped_time(tmp_path):
    mat_a = tmp_path / "a.mat"
    mat_b = tmp_path / "b.mat"
    time, line_den = _write_94ghz_mat(mat_a)
    data = sio.loadmat(str(mat_a))
    clean = {k: v for k, v in data.items() if not k.startswith("__")}
    sio.savemat(str(mat_b), {**clean, "trigger_time": np.array([[7]], dtype=np.uint8)})

    ods_a, ods_b = ODS(), ODS()
    interferometer_94ghz(ods_a, SHOT, mat_file=str(mat_a))
    interferometer_94ghz(ods_b, SHOT, mat_file=str(mat_b))
    np.testing.assert_allclose(ods_a["interferometer.time"], ods_b["interferometer.time"])


# --- channel mapping ---------------------------------------------------


def test_94ghz_columns_map_to_physical_channels_and_z(tmp_path):
    mat = tmp_path / "shot.mat"
    time, line_den = _write_94ghz_mat(mat)
    ods = ODS()
    interferometer_94ghz(ods, SHOT, mat_file=str(mat))

    expected_z = [0.000, -0.075, -0.150, -0.225, -0.300]
    identifiers = set()
    for index, z in enumerate(expected_z):
        prefix = f"interferometer.channel.{index}"
        assert ods[f"{prefix}.line_of_sight.first_point.z"] == pytest.approx(z)
        np.testing.assert_allclose(ods[f"{prefix}.n_e_line.data"], line_den[:, index])
        identifiers.add(ods[f"{prefix}.identifier"])
    assert len(identifiers) == 5


def test_282ghz_identifier(tmp_path):
    mat = tmp_path / "shot.mat"
    _write_282ghz_mat(mat)
    ods = ODS()
    interferometer_282ghz(ods, SHOT, mat_file=str(mat))
    assert ods["interferometer.channel.0.identifier"] == "282GHz_V_ch01"


# --- geometry ------------------------------------------------------------


def test_94ghz_los_has_three_non_negative_points(tmp_path):
    mat = tmp_path / "shot.mat"
    _write_94ghz_mat(mat)
    ods = ODS()
    interferometer_94ghz(ods, SHOT, mat_file=str(mat))
    for index in range(5):
        los = ods[f"interferometer.channel.{index}.line_of_sight"]
        assert set(los.keys()) == {"first_point", "second_point", "third_point"}
        for point in ("first_point", "second_point", "third_point"):
            assert los[point]["r"] >= 0.0


def test_282ghz_los_has_two_points_no_third(tmp_path):
    mat = tmp_path / "shot.mat"
    _write_282ghz_mat(mat)
    ods = ODS()
    interferometer_282ghz(ods, SHOT, mat_file=str(mat))
    los = ods["interferometer.channel.0.line_of_sight"]
    assert set(los.keys()) == {"first_point", "second_point"}
    assert los["first_point"]["r"] >= 0.0
    assert los["second_point"]["r"] >= 0.0


def test_94ghz_path_length_is_1_4_m():
    from vaft.machine_mapping.interferometer import _interferometer_config

    config = _interferometer_config()["horizontal_94ghz"]
    assert config["path_length_m"] == pytest.approx(1.4)
    assert config["path_length_m"] == pytest.approx(2 * (config["launch_r_m"] - config["mirror_r_m"]))


def test_282ghz_path_length_is_2_4_m():
    from vaft.machine_mapping.interferometer import _interferometer_config

    config = _interferometer_config()["vertical_282ghz"]
    assert config["path_length_m"] == pytest.approx(2.4)


# --- physical quantities ---------------------------------------------------


def test_n_e_line_maps_directly_without_scaling(tmp_path):
    mat = tmp_path / "shot.mat"
    time, line_den = _write_94ghz_mat(mat)
    ods = ODS()
    interferometer_94ghz(ods, SHOT, mat_file=str(mat))
    for index in range(5):
        np.testing.assert_array_equal(
            ods[f"interferometer.channel.{index}.n_e_line.data"], line_den[:, index]
        )


def test_line_average_density_uses_configured_path_length(tmp_path):
    mat = tmp_path / "shot.mat"
    time, line_den = _write_94ghz_mat(mat)
    ods = ODS()
    interferometer_94ghz(ods, SHOT, mat_file=str(mat), compute_line_average=True)
    for index in range(5):
        np.testing.assert_allclose(
            ods[f"interferometer.channel.{index}.n_e_line_average.data"],
            line_den[:, index] / 1.4,
        )


def test_line_average_absent_by_default(tmp_path):
    mat = tmp_path / "shot.mat"
    _write_94ghz_mat(mat)
    ods = ODS()
    interferometer_94ghz(ods, SHOT, mat_file=str(mat))
    assert "n_e_line_average" not in ods["interferometer.channel.0"]


# --- validity and fringe metadata ------------------------------------------


def test_validity_and_fringe_fields_remain_unset(tmp_path):
    mat_h = tmp_path / "h.mat"
    mat_v = tmp_path / "v.mat"
    _write_94ghz_mat(mat_h)
    _write_282ghz_mat(mat_v)
    ods = ODS()
    interferometer(ods, SHOT, mat_file_94ghz=str(mat_h), mat_file_282ghz=str(mat_v))
    for index in range(6):
        channel = ods[f"interferometer.channel.{index}"]
        assert "validity" not in channel["n_e_line"]
        assert "validity_timed" not in channel["n_e_line"]
        assert "fringe_jump_correction" not in channel["wavelength"][0]
        assert "phase_corrected" not in channel["wavelength"][0]


# --- time semantics ----------------------------------------------------


def test_single_system_builders_are_homogeneous_time(tmp_path):
    mat_h = tmp_path / "h.mat"
    mat_v = tmp_path / "v.mat"
    time_h, _ = _write_94ghz_mat(mat_h)
    time_v, _ = _write_282ghz_mat(mat_v)

    ods_h = ODS()
    interferometer_94ghz(ods_h, SHOT, mat_file=str(mat_h))
    assert ods_h["interferometer.ids_properties.homogeneous_time"] == 1
    np.testing.assert_allclose(ods_h["interferometer.time"], time_h)

    ods_v = ODS()
    interferometer_282ghz(ods_v, SHOT, mat_file=str(mat_v))
    assert ods_v["interferometer.ids_properties.homogeneous_time"] == 1
    np.testing.assert_allclose(ods_v["interferometer.time"], time_v)


def test_merged_fallback_is_heterogeneous_with_independent_timebases(tmp_path):
    mat_h = tmp_path / "h.mat"
    mat_v = tmp_path / "v.mat"
    time_h, _ = _write_94ghz_mat(mat_h, n_time=50)
    time_v, _ = _write_282ghz_mat(mat_v, n_time=60)  # deliberately different length/spacing

    ods = ODS()
    interferometer(ods, SHOT, mat_file_94ghz=str(mat_h), mat_file_282ghz=str(mat_v))

    assert ods["interferometer.ids_properties.homogeneous_time"] == 0
    assert len(ods["interferometer.channel"]) == 6
    for index in range(5):
        np.testing.assert_allclose(
            ods[f"interferometer.channel.{index}.n_e_line.time"], time_h
        )
    np.testing.assert_allclose(ods["interferometer.channel.5.n_e_line.time"], time_v)
    # No artificial common timebase: the two systems' time arrays differ in length.
    assert len(time_h) != len(time_v)


# --- schema round-trip ---------------------------------------------------


def test_all_populated_paths_resolve_under_imas_3_41(tmp_path):
    mat_h = tmp_path / "h.mat"
    mat_v = tmp_path / "v.mat"
    _write_94ghz_mat(mat_h)
    _write_282ghz_mat(mat_v)
    ods = ODS()
    interferometer(
        ods, SHOT, mat_file_94ghz=str(mat_h), mat_file_282ghz=str(mat_v), compute_line_average=True
    )
    paths = ods.paths()
    assert len(paths) > 0  # constructing paths() already validates against the DD schema


# --- real archived samples ------------------------------------------------


def test_real_archived_94ghz_sample():
    sample = require_repository_sample(data_path("legacy/47230_056789_LID_1_100.mat"))
    ods = ODS()
    interferometer_94ghz(ods, 47230, mat_file=str(sample))
    assert len(ods["interferometer.channel"]) == 5
    assert ods["interferometer.time"][0] == pytest.approx(0.29)


def test_real_archived_282ghz_sample():
    sample = require_repository_sample(data_path("legacy/47230_ALL_LID_1_100.mat"))
    ods = ODS()
    interferometer_282ghz(ods, 47230, mat_file=str(sample))
    assert len(ods["interferometer.channel"]) == 1
    assert ods["interferometer.time"][0] == pytest.approx(0.29)
