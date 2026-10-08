"""The 3-D adapters' contract without their optional libraries (issue #1087).

Importing VAFT, ``vaft.plot``, and the adapters ``vaft.plot.pyvista`` and
``vaft.plot.k3d`` themselves, must not import PyVista or K3D; a missing library is named together with the extra that installs it;
the Cartesian convention is the IMAS toroidal angle of #718; and the coil
phasing the K3D explorer drives is the inverse of the existing toroidal mode
decomposition.  The rendering itself is covered in test_plot_3d_adapters.py.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

import numpy as np
import pytest

from vaft.machine_mapping.conventions import (
    clock_angle_to_toroidal_angle,
    cylindrical_to_cartesian,
    port_toroidal_angle,
)
from vaft.plot.models import Geometry3DLayer, Geometry3DLayers
from vaft.process.coils_non_axisymmetric import phased_sector_currents, toroidal_mode_decomposition

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_importing_vaft_and_the_3d_adapters_imports_no_3d_library():
    code = (
        "import sys, vaft, vaft.plot, vaft.plot.pyvista as pv_adapter, vaft.plot.k3d as k3d_adapter\n"
        "assert pathlib_root in vaft.__file__, vaft.__file__\n"
        "print(sorted(m for m in ('pyvista', 'vtk', 'k3d') if m in sys.modules))\n"
        "print(sorted(pv_adapter.__all__ + k3d_adapter.__all__))\n"
    ).replace("pathlib_root", repr(str(ROOT)))
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    loaded, names = result.stdout.strip().splitlines()
    assert loaded == "[]"
    assert names == str(sorted(["coil_phase_explorer", "read_vtk_blocks", "to_k3d", "to_pyvista", "write_vtk"]))


@pytest.mark.parametrize("module, name, extra", [("pyvista", "to_pyvista", "vaft[vtk]"), ("k3d", "to_k3d", "vaft[jupyter3d]")])
def test_a_missing_library_names_its_extra(monkeypatch, module, name, extra):
    import importlib

    adapter = importlib.import_module(f"vaft.plot.{module}")

    monkeypatch.setitem(sys.modules, module, None)  # makes `import <module>` raise
    scene = Geometry3DLayers((Geometry3DLayer(x=[0, 1], y=[0, 1], z=[0, 1]),))
    with pytest.raises(ImportError, match=re.escape(f"pip install {extra}")):
        getattr(adapter, name)(scene)


def test_a_non_scene_is_refused_with_the_way_to_build_one():
    from vaft.plot._scene3d import as_layers

    with pytest.raises(TypeError, match=r"vaft\.plot\.extract\('machine_geometry3d', ods\)"):
        as_layers({"x": [0.0]})


def test_the_cartesian_axes_follow_the_imas_toroidal_angle():
    # 3 o'clock is phi = 270 degrees (#718): the -y axis, counter-clockwise from +x.
    x, y, z = cylindrical_to_cartesian(0.5, port_toroidal_angle(3), 0.2)
    assert np.allclose([x, y, z], [0.0, -0.5, 0.2])
    x, y, _ = cylindrical_to_cartesian(1.0, clock_angle_to_toroidal_angle(45.0), 0.0)
    assert np.allclose([x, y], [np.cos(np.deg2rad(315.0)), np.sin(np.deg2rad(315.0))])
    r, phi, z = np.array([0.4, 0.6]), np.array([0.0, np.pi / 2]), np.array([0.0, 1.0])
    x, y, height = cylindrical_to_cartesian(r, phi, z)
    assert np.allclose(np.hypot(x, y), r) and np.allclose(np.arctan2(y, x), phi) and np.allclose(height, z)
    assert cylindrical_to_cartesian([0.5, 0.7], 0.0, 0.0)[0].shape == (2,)


def test_the_3d_layer_group_is_a_normalised_path():
    layer = Geometry3DLayer(x=[0.0], y=[0.0], z=[0.0], group="/machine/wall/")
    assert layer.group == "machine/wall"
    assert Geometry3DLayer(x=[0.0], y=[0.0], z=[0.0]).group == ""
    dataset = Geometry3DLayers((layer,)).to_xarray()
    assert list(dataset["group"].values) == ["machine/wall"]


@pytest.mark.parametrize("n, phase", [(1, 0.0), (1, 1.2), (2, -0.7), (3, 2.5)])
def test_phased_sector_currents_invert_the_mode_decomposition(n, phase):
    # Six sectors (VEST's rows) resolve n = 1, 2 without aliasing; n = 3 needs eight.
    phi = np.deg2rad(np.arange(15.0, 360.0, 60.0) if n < 3 else np.arange(0.0, 360.0, 45.0))
    currents = phased_sector_currents(phi, n, phase, 800.0)
    coefficient = toroidal_mode_decomposition(phi, currents, [n])[n]
    assert np.isclose(2 * abs(coefficient), 800.0)
    assert np.isclose(np.angle(coefficient), np.angle(np.exp(1j * phase)))


def test_phased_sector_currents_refuse_no_sectors():
    with pytest.raises(ValueError, match="non-empty one-dimensional"):
        phased_sector_currents([], 1, 0.0, 1.0)
