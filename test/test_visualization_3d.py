"""PyVista/VTK export and K3D rendering of the 3-D scenes (issue #1087).

The reference scene is the #223 VEST non-axisymmetric coil set: its
Cartesian vertices must be the stored ``(r, phi, z)`` through the #718
convention, survive conversion with their polyline connectivity and subsystem
identity, and round-trip through ``.vtm`` and ``.vtp`` files.  Each library
is optional, so each half skips on its own.
"""

from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import pytest
from omas import ODS

import vaft
import vaft.data
import vaft.omas
import vaft.plot
import vaft.visualization
from vaft.machine_mapping.conventions import cylindrical_to_cartesian
from vaft.plot.models import Geometry3DLayer, Geometry3DLayers


@pytest.fixture(scope="module")
def coils():
    from vaft.machine_mapping.coils_non_axisymmetric import coils_non_axisymmetric

    ods = ODS()
    coils_non_axisymmetric(ods)
    return ods


@pytest.fixture(scope="module")
def sample():
    with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return vaft.omas.load(str(vaft.data.data_path("samples/39915/omas.json.gz")))


def test_the_coil_scene_is_the_stored_cylindrical_geometry(coils):
    scene = vaft.plot.extract("coil_3d_geometry3d", coils)
    assert len(scene.layers) == 18
    for index, layer in enumerate(scene.layers):
        base = f"coils_non_axisymmetric.coil.{index}.conductor.0.elements"
        r = np.append(coils[f"{base}.start_points.r"], coils[f"{base}.end_points.r"][-1])
        phi = np.append(coils[f"{base}.start_points.phi"], coils[f"{base}.end_points.phi"][-1])
        z = np.append(coils[f"{base}.start_points.z"], coils[f"{base}.end_points.z"][-1])
        assert np.allclose(np.column_stack([layer.x, layer.y, layer.z]), np.column_stack(cylindrical_to_cartesian(r, phi, z)))
        name = coils[f"coils_non_axisymmetric.coil.{index}.name"]
        assert layer.group == f"coils_non_axisymmetric/{name.rsplit(' sector', 1)[0]}/{name}"


def test_the_machine_scene_places_only_what_the_data_state(sample):
    scene = vaft.plot.extract("machine_geometry3d", sample)
    roots = {layer.group.split("/")[0] for layer in scene.layers}
    assert roots == {"machine", "equilibrium", "diagnostics"}
    probes = [layer for layer in scene.layers if layer.group == "diagnostics/magnetics.b_field_pol_probe"]
    assert len(probes) == 1 and probes[0].kind == "points"
    stated = sum(
        1 for index in range(len(sample["magnetics.b_field_pol_probe"]))
        if all(np.isfinite(np.atleast_1d(sample.get(f"magnetics.b_field_pol_probe.{index}.position.{axis}", np.nan))).all()
               for axis in ("r", "phi", "z"))
    )
    assert probes[0].x.size == stated
    # The wall is a wireframe: four toroidal cuts and two rings per limiter unit.
    wall = [layer for layer in scene.layers if layer.group.startswith("machine/wall/")]
    assert len(wall) == 6 and sum(bool(layer.label) for layer in wall) == 1


class TestPyVista:
    @pytest.fixture(autouse=True)
    def _pyvista(self):
        self.pv = pytest.importorskip("pyvista")

    def test_polylines_keep_their_order_and_the_tree_follows_group(self, coils):
        scene = vaft.plot.extract("coil_3d_geometry3d", coils)
        blocks = vaft.visualization.to_pyvista(scene)
        assert list(blocks.keys()) == ["coils_non_axisymmetric"]
        assert list(blocks["coils_non_axisymmetric"].keys()) == ["UP", "MID", "LOW"]
        total = 0
        for layer in scene.layers:
            _, set_name, coil = layer.group.split("/")
            mesh = blocks["coils_non_axisymmetric"][set_name][coil]
            assert isinstance(mesh, self.pv.PolyData)
            assert np.allclose(mesh.points, np.column_stack([layer.x, layer.y, layer.z]))
            assert list(mesh.lines) == [layer.x.size, *range(layer.x.size)]
            assert str(mesh.field_data["vaft_group"][0]) == layer.group
            total += mesh.n_points
        assert total == sum(layer.x.size for layer in scene.layers)
        assert str(blocks.field_data["vaft_title"][0]) == scene.title

    def test_a_nan_splits_a_polyline_and_points_become_vertices(self):
        layers = Geometry3DLayers((
            Geometry3DLayer(x=[0.0, 1.0, np.nan, 2.0, 3.0], y=[0.0] * 5, z=[0.0] * 5, label="split"),
            Geometry3DLayer(x=[0.0, 1.0, np.nan], y=[1.0] * 3, z=[0.0] * 3, kind="points"),
        ))
        blocks = vaft.visualization.to_pyvista(layers)
        line, points = blocks["layers"]["layer 0"], blocks["layers"]["layer 1"]
        assert line.n_points == 4 and list(line.lines) == [2, 0, 1, 2, 2, 3]
        assert points.n_points == 2 and points.n_verts == 2 and points.n_lines == 0

    @pytest.mark.parametrize("suffix", [".vtm", ".vtp"])
    def test_the_files_read_back_with_connectivity_and_identity(self, tmp_path, sample, coils, suffix):
        for source, name in ((coils, "coil_3d_geometry3d"), (sample, "machine_geometry3d")):
            scene = vaft.plot.extract(name, source)
            path = vaft.visualization.write_vtk(scene, tmp_path / f"{name}{suffix}")
            records = vaft.visualization.read_vtk_blocks(path)
            assert len(records) == len(scene.layers)
            by_group = {record["group"]: record for record in records.values()}
            for layer in scene.layers:
                record = by_group[layer.group]
                assert record["kind"] == layer.kind and record["label"] == layer.label
                assert np.allclose(record["points"], np.column_stack([layer.x, layer.y, layer.z]))
                if layer.kind == "polyline":
                    assert [list(cell) for cell in record["lines"]] == [list(range(layer.x.size))]
                else:
                    assert len(record["verts"]) == layer.x.size

    def test_an_unknown_suffix_is_refused(self, tmp_path, coils):
        with pytest.raises(ValueError, match=r"\.vtm, \.vtp"):
            vaft.visualization.write_vtk(vaft.plot.extract("coil_3d_geometry3d", coils), tmp_path / "coils.stl")


class TestK3D:
    @pytest.fixture(autouse=True)
    def _k3d(self):
        self.k3d = pytest.importorskip("k3d")

    def test_one_named_object_per_layer(self, sample):
        scene = vaft.plot.extract("machine_geometry3d", sample)
        plot = vaft.visualization.to_k3d(scene)
        assert [obj.name for obj in plot.objects] == [layer.group for layer in scene.layers]
        kinds = {type(obj).__name__ for obj in plot.objects}
        assert kinds == {"Line", "Points"}
        assert plot.axes == ["x [m]", "y [m]", "z [m]"]

    def test_the_explorer_drives_existing_physics_in_place(self, coils):
        from vaft.process.coils_non_axisymmetric import toroidal_mode_decomposition

        result = vaft.visualization.coil_phase_explorer(coils, coil_set="MID", show=False)
        plot, state = result.figure, result.state
        objects = list(plot.objects)
        mid = [obj for obj in objects if obj.name.startswith("coils_non_axisymmetric/MID/")]
        assert len(mid) == 6
        before = [obj.color for obj in mid]
        coefficient = toroidal_mode_decomposition(result.computed["phi_sector"], result.computed["currents"], [1])[1]
        assert np.isclose(2 * abs(coefficient), 1.0e3) and np.isclose(np.angle(coefficient), 0.0, atol=1e-9)
        # The strongest sector sits at phi = -phase for n = 1.
        state.set("phase_deg", 90)
        peak = result.computed["phi_sector"][np.argmax(result.computed["currents"])]
        assert np.isclose(np.cos(peak + np.pi / 2), 1.0, atol=0.2)
        assert list(plot.objects) == objects  # updated in place, not rebuilt
        assert [obj.color for obj in mid] != before
        # The n = 1 pattern gives an n = 1 field on the probe ring.
        field = result.computed["component"]
        angle = np.arctan2(result.computed["probe_xyz"][:, 1], result.computed["probe_xyz"][:, 0])
        spectrum = {n: abs(np.mean(field * np.exp(-1j * n * angle))) for n in range(4)}
        assert max(spectrum, key=spectrum.get) == 1
        state.set("n", 2)
        field = result.computed["component"]
        spectrum = {n: abs(np.mean(field * np.exp(-1j * n * angle))) for n in range(4)}
        assert max(spectrum, key=spectrum.get) == 2

    def test_an_unknown_coil_set_is_named(self, coils):
        with pytest.raises(ValueError, match="UP, MID, LOW"):
            vaft.visualization.coil_phase_explorer(coils, coil_set="MIDDLE", show=False)
