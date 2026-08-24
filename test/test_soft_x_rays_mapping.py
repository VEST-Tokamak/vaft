import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omas import ODS

import vaft.machine_mapping.soft_x_rays as sxr_mapping
from vaft.machine_mapping.soft_x_rays import (
    build_time_axis,
    default_channel_map,
    load_sxr_geometry_table,
    resolve_sxr_geometry_table,
    save_soft_x_rays_ods,
    soft_x_rays,
    soft_x_rays_from_digitizer_csv,
    resolve_sxr_time_alignment,
)


def _write_tiny_geometry(root: Path) -> None:
    root.mkdir()
    (root / "lowermid").mkdir()
    np.savetxt(root / "rgrid.csv", np.linspace(0, 800, 5), delimiter=",")
    np.savetxt(root / "zgrid.csv", np.linspace(-1200, 1200, 5), delimiter=",")
    los = np.zeros((5, 5))
    np.fill_diagonal(los, 1.0)
    np.savetxt(root / "lowermid" / "twofilter_horizontalLOS_ch_1_.csv", los, delimiter=",")


def test_build_time_axis_uses_seconds():
    got = build_time_axis(3, sample_rate=10.0, time_offset=0.2)
    np.testing.assert_allclose(got, [0.2, 0.3, 0.4])


def test_default_channel_map_uses_canonical_wiring_and_filters():
    top = default_channel_map("17592", 40)
    assert (top[0]["array"], top[0]["array_channel"]) == ("vertical", 1)
    assert (top[19]["array"], top[19]["array_channel"]) == ("vertical", 20)
    assert (top[20]["array"], top[20]["array_channel"]) == ("horizontal", 1)
    assert (top[39]["array"], top[39]["array_channel"]) == ("horizontal", 20)

    mapping = default_channel_map("22577", 64)
    assert (mapping[0]["array"], mapping[0]["filter_material"], mapping[0]["array_channel"]) == (
        "bottom", "Be", 1,
    )
    assert (mapping[16]["array"], mapping[16]["filter_material"], mapping[16]["array_channel"]) == (
        "bottom", "Al", 1,
    )
    assert (mapping[32]["array"], mapping[32]["filter_material"], mapping[32]["array_channel"]) == (
        "lowermid", "Be", 16,
    )
    assert (mapping[63]["array"], mapping[63]["filter_material"], mapping[63]["array_channel"]) == (
        "lowermid", "Al", 1,
    )
    assert all(item["filter_thickness_m"] == pytest.approx(0.2e-6) for item in mapping)


def test_packaged_soft_x_ray_geometry_table_is_available():
    table = resolve_sxr_geometry_table()
    assert table is not None
    assert table.name == "line_of_sight_endpoints.csv"
    table_frame = pd.read_csv(table)
    assert list(table_frame.columns) == [
        "daq_label",
        "array",
        "channel",
        "first_r",
        "first_z",
        "second_r",
        "second_z",
        "phi",
    ]
    assert "source" not in table_frame.columns

    geometry = load_sxr_geometry_table(table)
    assert len(geometry) == 72
    assert ("lowermid", 1) in geometry
    assert geometry[("horizontal", 1)]["daq_label"] == "17592"
    assert geometry[("lowermid", 1)]["daq_label"] == "22577"
    np.testing.assert_allclose(geometry[("horizontal", 1)]["phi"], 0.0)
    np.testing.assert_allclose(geometry[("lowermid", 1)]["phi"], 2.0 * np.pi / 3.0)




def test_soft_x_rays_uses_packaged_phi_for_default_daq_mapping(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    np.savetxt(data_root / "digitizer_22577_12345.csv", np.array([[1.0, 2.0, 3.0]]), delimiter=",")
    ods = soft_x_rays_from_digitizer_csv(
        12345,
        22577,
        data_root=data_root,
        sample_rate=1.0,
        time_offset=0.0,
    )
    np.testing.assert_allclose(
        ods["soft_x_rays.channel.0.line_of_sight.first_point.phi"],
        2.0 * np.pi / 3.0,
    )
    assert ods["soft_x_rays.channel.0.filter_window.0.material.index"] == 10
    assert ods["soft_x_rays.channel.0.filter_window.0.thickness"] == pytest.approx(0.2e-6)
    assert "etendue" not in ods["soft_x_rays.channel.0"].keys()


def test_soft_x_rays_maps_digitizer_csv_to_omas_brightness(tmp_path):
    geometry_root = tmp_path / "geometry"
    _write_tiny_geometry(geometry_root)

    data_root = tmp_path / "data"
    data_root.mkdir()
    # CSV convention: channel rows, time samples across columns.
    values = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 20.0, 30.0, 40.0],
        ]
    )
    np.savetxt(data_root / "digitizer_test_12345.csv", values, delimiter=",")

    channel_map = [
        {
            "source_column": 0,
            "array": "lowermid",
            "array_channel": 1,
            "name": "LM Ch 1",
            "identifier": "test:lowermid:1",
        },
        {"source_column": 1, "name": "Generic Ch 2", "identifier": "test:generic:2"},
    ]

    ods = soft_x_rays_from_digitizer_csv(
        12345,
        "test",
        data_root=data_root,
        geometry_root=geometry_root,
        channel_map=channel_map,
        sample_rate=2.0,
        time_offset=0.0,
        baseline_range=(0, 1),
    )

    np.testing.assert_allclose(ods["soft_x_rays.time"], [0.0, 0.5, 1.0, 1.5])
    np.testing.assert_allclose(
        ods["soft_x_rays.channel.0.brightness.data"][0],
        [0.0, 1.0, 2.0, 3.0],
    )
    np.testing.assert_allclose(
        ods["soft_x_rays.channel.1.brightness.data"][0],
        [0.0, 10.0, 20.0, 30.0],
    )
    assert ods["soft_x_rays.channel.0.name"] == "LM Ch 1"
    assert np.isfinite(ods["soft_x_rays.channel.0.line_of_sight.first_point.r"])
    assert "line_of_sight" not in ods["soft_x_rays.channel.1"]


def test_soft_x_rays_populates_existing_ods(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    np.savetxt(data_root / "digitizer_test_12345.csv", np.array([[1.0, 2.0, 3.0]]), delimiter=",")
    ods = ODS()
    soft_x_rays(
        ods,
        12345,
        "test",
        data_root=data_root,
        channel_map=[{"source_column": 0, "name": "Only Ch"}],
        sample_rate=1.0,
        time_offset=0.0,
    )
    assert ods["soft_x_rays.channel.0.identifier"] == "Only Ch"
    assert ods["soft_x_rays.channel.0.brightness.data"].shape == (1, 3)


def test_save_soft_x_rays_ods_accepts_consistency_check_true(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    np.savetxt(data_root / "digitizer_test_12345.csv", np.array([[1.0, 2.0, 3.0]]), delimiter=",")
    output = tmp_path / "sxr.pkl"
    ods = save_soft_x_rays_ods(
        output,
        12345,
        "test",
        data_root=data_root,
        channel_map=[{"source_column": 0, "name": "Only Ch"}],
        sample_rate=1.0,
        time_offset=0.0,
        consistency_check=True,
    )
    assert output.exists()
    assert ods.consistency_check is True
    assert ods["soft_x_rays.channel.0.brightness.data"].shape == (1, 3)


def test_trigger_settings_prefers_explicit_sxr_over_hxr(tmp_path):
    settings = tmp_path / "trigger-settings.yaml"
    settings.write_text(
        "shots:\n"
        "  47370:\n"
        "    SXR:\n"
        "      start_time_ms: 287\n"
        "    HXR:\n"
        "      start_time_ms: 285\n",
        encoding="utf-8",
    )
    alignment = resolve_sxr_time_alignment(47370, trigger_settings_path=settings)
    assert alignment.source == "sxr_trigger"
    assert alignment.offset_seconds == pytest.approx(0.287)


def test_trigger_settings_uses_migrated_early_sxr_record():
    alignment = resolve_sxr_time_alignment(39350)
    assert alignment.source == "sxr_trigger"
    assert alignment.offset_seconds == pytest.approx(0.300)


def test_trigger_settings_uses_455xx_hxr_fallback():
    alignment = resolve_sxr_time_alignment(45539)
    assert alignment.source == "hxr_fallback"
    assert alignment.offset_seconds == pytest.approx(0.285)
    assert "Inferred SXR start" in alignment.detail


def test_trigger_settings_missing_shot_keeps_archive_axis():
    with pytest.warns(RuntimeWarning, match="trigger-relative"):
        alignment = resolve_sxr_time_alignment(12345)
    assert alignment.source == "trigger_relative"
    assert alignment.offset_seconds == 0.0


def test_mapper_uses_sample_v3_rate_and_trigger_settings_by_default(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    np.savetxt(data_root / "digitizer_22577_45539.csv", np.array([[1.0, 2.0, 3.0]]), delimiter=",")
    ods = soft_x_rays_from_digitizer_csv(45539, 22577, data_root=data_root)
    np.testing.assert_allclose(
        ods["soft_x_rays.time"],
        0.285 + np.arange(3) / (125e6 / 128.0),
    )


def test_mapper_uses_17592_sample_v3_rate_by_default(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    np.savetxt(data_root / "digitizer_17592_45539.csv", np.array([[1.0, 2.0, 3.0]]), delimiter=",")
    ods = soft_x_rays_from_digitizer_csv(45539, 17592, data_root=data_root)
    np.testing.assert_allclose(
        ods["soft_x_rays.time"],
        0.285 + np.arange(3) / (125e6 / 32.0),
    )


def test_aluminum_filter_uses_private_material_identifier(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    np.savetxt(
        data_root / "digitizer_22577_45539.csv",
        np.arange(34, dtype=float).reshape(17, 2),
        delimiter=",",
    )
    ods = soft_x_rays_from_digitizer_csv(45539, 22577, data_root=data_root)
    assert ods["soft_x_rays.channel.16.filter_window.0.material.index"] == -1
    assert ods["soft_x_rays.channel.16.filter_window.0.material.name"] == "Al"


def test_explicit_time_override_beats_trigger_settings(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    np.savetxt(data_root / "digitizer_22577_45539.csv", np.array([[1.0, 2.0]]), delimiter=",")
    ods = soft_x_rays_from_digitizer_csv(
        45539,
        22577,
        data_root=data_root,
        time_offset=0.1,
    )
    np.testing.assert_allclose(ods["soft_x_rays.time"], 0.1 + np.arange(2) / (125e6 / 128.0))
    assert "explicit" in ods["soft_x_rays.ids_properties.comment"]


def test_shot_level_mapper_merges_available_daqs_without_duplicate_channels(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    np.savetxt(
        data_root / "digitizer_17592_12345.csv",
        np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]),
        delimiter=",",
    )
    np.savetxt(
        data_root / "digitizer_22577_12345.csv",
        np.array([[100.0, 200.0, 300.0], [1000.0, 2000.0, 3000.0]]),
        delimiter=",",
    )

    ods = soft_x_rays_from_digitizer_csv(12345, data_root=data_root, time_offset=0.0)

    assert len(ods["soft_x_rays.channel"]) == 4
    assert [ods[f"soft_x_rays.channel.{index}.identifier"] for index in range(4)] == [
        "17592:vertical:none:1",
        "17592:vertical:none:2",
        "22577:bottom:Be:1",
        "22577:bottom:Be:2",
    ]
    np.testing.assert_allclose(
        ods["soft_x_rays.channel.0.brightness.time"],
        np.arange(3) / (125e6 / 32.0),
    )
    np.testing.assert_allclose(
        ods["soft_x_rays.channel.2.brightness.time"],
        np.arange(3) / (125e6 / 128.0),
    )
    assert ods["soft_x_rays.ids_properties.homogeneous_time"] == 0
    assert "digitizer_17592_12345.csv" in ods["soft_x_rays.ids_properties.source"]
    assert "digitizer_22577_12345.csv" in ods["soft_x_rays.ids_properties.source"]

    soft_x_rays(ods, 12345, data_root=data_root, time_offset=0.0)
    assert len(ods["soft_x_rays.channel"]) == 4


def test_shot_level_mapper_supports_one_available_daq(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    np.savetxt(data_root / "digitizer_17592_12345.csv", np.array([[1.0, 2.0, 3.0]]), delimiter=",")

    ods = soft_x_rays_from_digitizer_csv(12345, data_root=data_root, time_offset=0.0)

    assert len(ods["soft_x_rays.channel"]) == 1
    assert ods["soft_x_rays.channel.0.identifier"] == "17592:vertical:none:1"
    assert ods["soft_x_rays.ids_properties.homogeneous_time"] == 1


def test_explicit_digitizer_file_can_infer_its_daq_label(tmp_path):
    digitizer_file = tmp_path / "digitizer_22577_12345.csv"
    np.savetxt(digitizer_file, np.array([[1.0, 2.0, 3.0]]), delimiter=",")

    ods = soft_x_rays_from_digitizer_csv(12345, digitizer_file=digitizer_file, time_offset=0.0)

    assert ods["soft_x_rays.channel.0.identifier"] == "22577:bottom:Be:1"


def test_sxr_mapper_does_not_use_raw_database_trigger_correction():
    source = inspect.getsource(sxr_mapping)
    assert "from vaft.database import raw" not in source
    assert "_daq_trigger_time_correction" not in source
