"""Numerical machine-space contract, independent of successful rendering."""
import json

import numpy as np
import pytest

from vaft.plot.machine_geometry import MachineGeometry, machine_geometry_registry, machine_geometry_view, project_machine_geometry


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
    unknown_los = MachineGeometry("test", "line_of_sight", [1, 2], [0, 1])
    rz_vertices = project_machine_geometry(unknown_los, "rz")
    assert rz_vertices.kind == "points"
    assert "phi unknown" in rz_vertices.label


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
    assert {record.family for record in records} == {"thomson_scattering", "charge_exchange", "langmuir_probes", "interferometer", "soft_x_rays", "coils_non_axisymmetric", "ec_launchers", "nbi"}
    chords = [record for record in records if record.family == "interferometer"]
    assert [record.r.size for record in chords] == [3, 2]
    assert [next(iter(json.loads(record.provenance_json)["sources"])) for record in chords] == [
        "interferometer_94ghz", "interferometer_282ghz"
    ]
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


def test_fixture_derived_paths_and_sources_are_numerically_consistent():
    import hashlib
    from pathlib import Path
    from vaft.data import unified_diagnostics_fixture, unified_diagnostics_manifest
    from vaft.machine_mapping.ec_launchers import resolve_ec_launcher_geometry
    from vaft.machine_mapping.coils_non_axisymmetric_geometry import load_vest_3d_coil_config
    data = unified_diagnostics_fixture()
    manifest = unified_diagnostics_manifest()
    records = machine_geometry_registry(data, manifest=manifest)
    laser, = (record for record in records if record.family == "thomson_scattering" and record.semantic == "trajectory")
    sites = [record for record in records if record.family == "thomson_scattering" and "scattering site" in record.label]
    assert len(sites) == 5
    origin, end = laser.xyz[:, :2]
    along = end - origin
    for site in sites:
        offset = site.xyz[0, :2] - origin
        fraction = float(np.dot(offset, along) / np.dot(along, along))
        assert 0 <= fraction <= 1
        np.testing.assert_allclose(offset, fraction * along, atol=1e-12)
        assert json.loads(site.provenance_json)["value_kind"].startswith("derived")

    coil, = (record for record in records if record.family == "coils_non_axisymmetric")
    original = load_vest_3d_coil_config(coil_sets=["MID"])["MID"].filaments[0].points_xyz
    # Each stored element includes both endpoints; no simplified or fabricated loop.
    np.testing.assert_allclose(coil.xyz[::2], original[:-1], atol=1e-12)
    np.testing.assert_allclose(coil.xyz[1::2], original[1:], atol=1e-12)
    assert json.loads(coil.provenance_json)["sources"]["coil_filament_model"]["model_reference_shot"] == 48226

    ec = resolve_ec_launcher_geometry(39915)
    ec_axis, = (record for record in records if record.family == "ec_launchers" and record.semantic == "directed_axis")
    np.testing.assert_allclose([ec_axis.r[0], ec_axis.z[0], ec_axis.phi[0]], [ec["r"], ec["z"], ec["phi"]])
    kr, kphi, kz = ec["direction"]
    phi = ec["phi"]
    np.testing.assert_allclose(ec_axis.direction_xyz, [kr * np.cos(phi) - kphi * np.sin(phi), kr * np.sin(phi) + kphi * np.cos(phi), kz])
    assert "provisional" in json.loads(ec_axis.provenance_json)["value_kind"]

    nbi_axis, = (record for record in records if record.family == "nbi" and record.semantic == "directed_axis")
    nbi_source, = (record for record in records if record.family == "nbi" and record.semantic == "point")
    np.testing.assert_allclose(nbi_axis.xyz[0], nbi_source.xyz[0])
    assert nbi_axis.direction_xyz[2] == 0
    nbi_provenance = json.loads(nbi_axis.provenance_json)
    assert "source_shot" not in nbi_provenance["sources"]["nbi_model"]
    assert "not as-built" in nbi_provenance["value_kind"]
    # The model axis reaches the stated tangency radius and its velocity is
    # clockwise. This checks direction independently of any renderer output.
    xy = nbi_source.xyz[0, :2]
    velocity = nbi_axis.direction_xyz[:2]
    along = -np.dot(xy, velocity)
    foot = xy + along * velocity
    np.testing.assert_allclose(np.linalg.norm(foot), .22129, atol=1e-10)
    assert xy[0] * velocity[1] - xy[1] * velocity[0] < 0
    assert manifest["camera_calibration_reference"]["camera_shot"] == 39915
    from vaft.omas.process_wrapper import camera_projection_for
    camera = camera_projection_for(39915)
    site = sites[0]
    pixels, valid = camera.project(site.xyz * 100.0)
    view = project_machine_geometry(site, "camera", projection=camera)
    assert valid[0]
    np.testing.assert_allclose([view.r[0], view.z[0]], pixels[0])
    data_root = Path(__file__).parents[1] / "vaft/data"
    for source in manifest["geometry_sources"].values():
        assert hashlib.sha256((data_root / source["source_artifact"]).read_bytes()).hexdigest() == source["source_sha256"]


def test_four_views_share_family_selection_and_composite_notice():
    from vaft.data import unified_diagnostics_fixture, unified_diagnostics_manifest
    from vaft.omas.process_wrapper import camera_projection_for
    data = unified_diagnostics_fixture()
    manifest = unified_diagnostics_manifest()
    selected = ("thomson_scattering", "interferometer", "coils_non_axisymmetric", "nbi")
    camera = camera_projection_for(39915)
    for view in ("rz", "top", "3d", "camera"):
        model = machine_geometry_view(data, view, families=selected, manifest=manifest,
                                      projection=camera if view == "camera" else None)
        assert "Cross-shot composite" in model.title
        assert "shot 39915" in model.title
        assert model.layers
        if view == "3d":
            assert {layer.group.split("/")[0] for layer in model.layers} == set(selected)
        elif view == "camera":
            assert any("Thomson derived laser / sites" in layer.label for layer in model.layers)
            assert any("Interferometer LOS" in layer.label for layer in model.layers)
        else:
            assert any("NBI model geometry" in layer.label for layer in model.layers)
    with pytest.raises(ValueError, match="calibrated"):
        machine_geometry_view(data, "camera", families=selected, manifest=manifest)
    assert not machine_geometry_view(data, "top", families=("langmuir_probes",),
                                     manifest=manifest).layers


def test_incomplete_reflection_is_omitted_not_downgraded_to_two_point_los():
    from vaft.ods_access import set_path
    data = {}
    for endpoint in ("first_point", "second_point", "third_point"):
        base = f"interferometer.channel.0.line_of_sight.{endpoint}"
        for coordinate, value in (("r", 1.0), ("z", 0.0)):
            set_path(data, f"{base}.{coordinate}", value)
        if endpoint != "third_point":
            set_path(data, f"{base}.phi", 0.5)
    assert machine_geometry_registry(data, families=("interferometer",)) == ()


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


def test_stored_ec_ids_keeps_real_phi_and_steering_without_time_guess():
    from omas import ODS
    from vaft.machine_mapping.ec_launchers import (
        ec_launchers_geometry, ec_launchers_static, resolve_ec_launcher_geometry,
    )
    source = resolve_ec_launcher_geometry(39915)
    data = ODS(consistency_check=False)
    ec_launchers_static(data)
    ec_launchers_geometry(data, source, [.1, .2, .3])
    point, axis = machine_geometry_registry(data, families=("ec_launchers",))
    assert point.semantic == "point" and axis.semantic == "directed_axis"
    np.testing.assert_allclose(point.xyz[0], axis.xyz[0])
    np.testing.assert_allclose(axis.direction_xyz, [-np.cos(source["phi"]), -np.sin(source["phi"]), 0])
    # A moving launcher cannot be collapsed onto an arbitrary first sample.
    data["ec_launchers.beam.0.launching_position.phi"] = [source["phi"], source["phi"] + .1, source["phi"]]
    remaining, = machine_geometry_registry(data, families=("ec_launchers",))
    assert remaining.semantic == "point" and remaining.phi is None
    assert project_machine_geometry(remaining, "top") is None
