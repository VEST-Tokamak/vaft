"""Numerical machine-space contract, independent of successful rendering."""
import json

import numpy as np
import pytest

from vaft.plot.machine_geometry import MachineGeometry, machine_geometry_registry, project_machine_geometry


def test_cartesian_convention_and_unknown_phi():
    point = MachineGeometry("test", "point", [2], [3], [np.pi / 2])
    np.testing.assert_allclose(point.xyz, [[0, 2, 3]], atol=1e-14)
    top = project_machine_geometry(point, "top")
    np.testing.assert_allclose([top.r[0], top.z[0]], [0, 2], atol=1e-14)
    unknown = MachineGeometry("test", "point", [2], [3])
    assert project_machine_geometry(unknown, "3d") is None
    assert project_machine_geometry(unknown, "camera") is None
    np.testing.assert_equal(project_machine_geometry(unknown, "rz").r, [2])
    assert not point.r.flags.writeable


def test_los_is_cartesian_segment_not_linear_radius():
    los = MachineGeometry("test", "line_of_sight", [1, 1], [0, 0], [0, np.pi])
    rz = project_machine_geometry(los, "rz", samples_per_segment=3)
    np.testing.assert_allclose(rz.r, [1, 0, 1], atol=1e-14)
    assert "line_of_sight" in rz.label
    axis = MachineGeometry("test", "directed_axis", [1], [2], [0], direction_xyz=[0, 1, 0])
    result = project_machine_geometry(axis, "3d", axis_length=0.5, samples_per_segment=2)
    np.testing.assert_allclose(np.column_stack((result.x, result.y, result.z)), [[1, 0, 2], [1, .5, 2]])
    with pytest.raises(ValueError, match="unit vector"):
        MachineGeometry("test", "directed_axis", [1], [2], [0], direction_xyz=[0, 2, 0])


def test_camera_cm_conversion_and_invalid_gap():
    class Calibration:
        def project(self, xyz_cm):
            np.testing.assert_allclose(xyz_cm, [[100, 0, 0], [150, 0, 0], [200, 0, 0]])
            return np.array([[10, 20], [30, 40], [50, 60]]), np.array([True, False, True])
    record = MachineGeometry("test", "trajectory", [1, 2], [0, 0], [0, 0])
    result = project_machine_geometry(record, "camera", projection=Calibration(), samples_per_segment=3)
    np.testing.assert_allclose(result.r[[0, 2]], [10, 50])
    assert np.isnan(result.r[1]) and np.isnan(result.z[1])


def test_stored_coil_segments_preserved_including_gap_and_second_conductor():
    from vaft.ods_access import set_path
    data = {}
    for conductor in range(2):
        base = f"coils_non_axisymmetric.coil.0.conductor.{conductor}.elements"
        for axis, starts, ends in (("r", [1, 3], [2, 4]), ("z", [5, 7], [6, 8]), ("phi", [.1, .3], [.2, .4])):
            set_path(data, f"{base}.start_points.{axis}", starts)
            set_path(data, f"{base}.end_points.{axis}", ends)
    records = machine_geometry_registry(data)
    assert len(records) == 2
    for record in records:
        np.testing.assert_equal(record.r, [1, 2, np.nan, 3, 4])
        np.testing.assert_equal(record.phi, [.1, .2, np.nan, .3, .4])


def test_small_fixture_sources_and_reflected_chord():
    from vaft.data import unified_diagnostics_fixture, unified_diagnostics_manifest
    data = unified_diagnostics_fixture()
    manifest = unified_diagnostics_manifest()
    before = set(data.flat())
    records = machine_geometry_registry(data, manifest=manifest)
    assert set(data.flat()) == before
    assert {record.family for record in records} == {"thomson_scattering", "charge_exchange", "langmuir_probes", "interferometer", "soft_x_rays"}
    chords = [record for record in records if record.family == "interferometer"]
    assert [record.r.size for record in chords] == [3, 2]
    for record in records:
        provenance = json.loads(record.provenance_json)
        assert provenance["physical_discharge"] is False
        assert provenance["geometry_reference"]["source_shot"] == 39915
        assert provenance["sources"]
        if record.family == "charge_exchange":
            assert record.semantic == "point" and record.phi is None
    assert machine_geometry_registry({}, families=("interferometer",)) == ()
    with pytest.raises(ValueError, match="unsupported"):
        machine_geometry_registry({}, families=("gas_injection",))


def test_native_imas_and_omas_coordinate_equivalence():
    imas = pytest.importorskip("imas")
    from omas import ODS
    from vaft.imas.access import IDSEntry
    native = imas.IDSFactory(version="3.41.0").new("interferometer")
    native.channel.resize(1)
    ods = ODS(consistency_check=False)
    for endpoint, values in (("first_point", (1, .2, .3)), ("second_point", (2, .4, .5)), ("third_point", (3, .6, .7))):
        for coordinate, value in zip(("r", "z", "phi"), values):
            setattr(getattr(native.channel[0].line_of_sight, endpoint), coordinate, value)
            ods[f"interferometer.channel.0.line_of_sight.{endpoint}.{coordinate}"] = value
    left, = machine_geometry_registry(ods)
    right, = machine_geometry_registry(IDSEntry({"interferometer": native}))
    np.testing.assert_array_equal(left.xyz, right.xyz)
    for view in ("rz", "top", "3d"):
        a = project_machine_geometry(left, view)
        b = project_machine_geometry(right, view)
        for coordinate in (("x", "y", "z") if view == "3d" else ("r", "z")):
            np.testing.assert_array_equal(getattr(a, coordinate), getattr(b, coordinate))
