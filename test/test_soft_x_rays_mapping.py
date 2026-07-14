from pathlib import Path

import numpy as np
import pandas as pd
from omas import ODS

from vaft.machine_mapping.soft_x_rays import (
    build_time_axis,
    default_channel_map,
    load_sxr_geometry_table,
    resolve_sxr_geometry_table,
    save_soft_x_rays_ods,
    soft_x_rays,
    soft_x_rays_from_digitizer_csv,
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


def test_default_channel_map_keeps_unassigned_digitizer_channels():
    mapping = default_channel_map("22577", 34)
    assert mapping[0]["array"] == "lowermid"
    assert mapping[0]["daq_label"] == "22577"
    assert mapping[16]["array"] == "bottom"
    assert mapping[32]["array"] is None
    assert mapping[33]["source_column"] == 33


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
        ods["soft_x_rays.channel.0.brightness.data"][:, 0],
        [0.0, 1.0, 2.0, 3.0],
    )
    np.testing.assert_allclose(
        ods["soft_x_rays.channel.1.brightness.data"][:, 0],
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
    assert ods["soft_x_rays.channel.0.brightness.data"].shape == (3, 1)


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
    assert ods["soft_x_rays.channel.0.brightness.data"].shape == (3, 1)
