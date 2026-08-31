"""VEST 3D coil machine mapping into ``coils_non_axisymmetric``.

Static geometry and run-specific excitation are separate layers; these tests
pin the identifier scheme, the turn count, the cylindrical conversion, the
``homogeneous_time`` transitions, and the identifier-addressed excitation.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from omas import ODS

from vaft.machine_mapping.coil_geometry_3d import (
    CoilExcitation,
    load_vest_3d_coil_config,
)
from vaft.machine_mapping.coils_non_axisymmetric import (
    apply_coil_excitation,
    coils_non_axisymmetric,
)
from vaft.machine_mapping.utils import VestConfigurationError


@pytest.fixture(scope="module")
def geometry_ods():
    ods = ODS()
    coils_non_axisymmetric(ods)
    return ods


def _identifiers(ods):
    return [
        ods[f"coils_non_axisymmetric.coil.{i}.identifier"]
        for i in range(len(ods["coils_non_axisymmetric.coil"]))
    ]


def test_static_mapping_layout(geometry_ods):
    identifiers = _identifiers(geometry_ods)
    assert len(identifiers) == 18
    assert len(set(identifiers)) == 18
    for prefix in ("VEST_3D_UP", "VEST_3D_MID", "VEST_3D_LOW"):
        members = [name for name in identifiers if name.startswith(prefix + "_")]
        assert members == [f"{prefix}_{k:02d}" for k in range(1, 7)]
    assert (
        geometry_ods["coils_non_axisymmetric.ids_properties.homogeneous_time"] == 2
    )
    for index in range(18):
        assert geometry_ods[f"coils_non_axisymmetric.coil.{index}.turns"] == 20.0


def test_elements_roundtrip_to_source_geometry(geometry_ods):
    config = load_vest_3d_coil_config()
    identifiers = _identifiers(geometry_ods)
    index = identifiers.index("VEST_3D_MID_01")
    filament = config["MID"].filaments[0]

    elements = geometry_ods[
        f"coils_non_axisymmetric.coil.{index}.conductor.0.elements"
    ]
    npts = filament.points_xyz.shape[0]
    assert elements["types"].size == npts - 1
    assert np.all(elements["types"] == 1)

    radius = elements["start_points.r"]
    phi = elements["start_points.phi"]
    height = elements["start_points.z"]
    xyz = np.column_stack(
        [radius * np.cos(phi), radius * np.sin(phi), height]
    )
    np.testing.assert_allclose(xyz, filament.points_xyz[:-1], atol=1e-12)

    # Segments chain: end points are the next start points, closing the loop.
    np.testing.assert_allclose(
        elements["end_points.r"][:-1], radius[1:], atol=1e-12
    )
    np.testing.assert_allclose(
        elements["end_points.z"][-1], filament.points_xyz[-1][2], atol=1e-12
    )


def test_phi_is_continuous(geometry_ods):
    for index in range(18):
        phi = geometry_ods[
            f"coils_non_axisymmetric.coil.{index}.conductor.0.elements.start_points.phi"
        ]
        assert np.abs(np.diff(phi)).max() < np.pi / 2.0


def test_subset_selection():
    ods = ODS()
    coils_non_axisymmetric(ods, options={"coil_sets": ["MID"]})
    identifiers = _identifiers(ods)
    assert len(identifiers) == 6
    assert all(name.startswith("VEST_3D_MID_") for name in identifiers)


def test_provenance_fragment(geometry_ods):
    parameters = geometry_ods["coils_non_axisymmetric.code.parameters"]
    for token in ("VEST_3D_MID", 'turns="20"', "vest_MID.dat", "Gwang-geun Seo"):
        assert token in parameters
    assert geometry_ods["coils_non_axisymmetric.code.name"] == "vaft"


def test_excitation_targets_identifiers():
    ods = ODS()
    coils_non_axisymmetric(ods)
    currents = (200.0, 200.0, 0.0, -200.0, -200.0, 0.0)
    apply_coil_excitation(
        ods, [CoilExcitation("MID", currents)], time_s=0.3
    )
    identifiers = _identifiers(ods)
    for sector, expected in enumerate(currents):
        index = identifiers.index(f"VEST_3D_MID_{sector + 1:02d}")
        base = f"coils_non_axisymmetric.coil.{index}.current"
        np.testing.assert_allclose(ods[f"{base}.data"], [expected])
        np.testing.assert_allclose(ods[f"{base}.time"], [0.3])
    # Non-excited coils carry no current node.
    up_index = identifiers.index("VEST_3D_UP_01")
    assert "data" not in ods[f"coils_non_axisymmetric.coil.{up_index}.current"]
    assert ods["coils_non_axisymmetric.ids_properties.homogeneous_time"] == 1


def test_excitation_requires_geometry():
    with pytest.raises(VestConfigurationError, match="geometry"):
        apply_coil_excitation(ODS(), [CoilExcitation("MID", (0.0,) * 6)])


def test_excitation_rejects_wrong_sector_count():
    ods = ODS()
    coils_non_axisymmetric(ods)
    with pytest.raises(VestConfigurationError, match="sectors"):
        apply_coil_excitation(ods, [CoilExcitation("MID", (1.0, 2.0))])


def test_excitation_rejects_unmapped_set():
    ods = ODS()
    coils_non_axisymmetric(ods, options={"coil_sets": ["UP"]})
    with pytest.raises(VestConfigurationError, match="not present"):
        apply_coil_excitation(ods, [CoilExcitation("MID", (0.0,) * 6)])


def test_ods_save_load_roundtrip(tmp_path, geometry_ods):
    from omas import load_omas_json, save_omas_json

    path = tmp_path / "coils.json"
    save_omas_json(geometry_ods, str(path))
    reloaded = load_omas_json(str(path))
    assert _identifiers(reloaded) == _identifiers(geometry_ods)


def test_geometry3d_plot_renders(geometry_ods):
    import matplotlib.pyplot as plt

    import vaft.omas as vomas

    figure, axes = vomas.plot_coils_non_axisymmetric_geometry3d(geometry_ods)
    assert type(axes).__name__ == "Axes3D"
    assert len(axes.lines) == 18
    plt.close(figure)

    figure, axes = vomas.plot_coils_non_axisymmetric_geometry_topview(geometry_ods)
    assert len(axes.lines) == 18
    plt.close(figure)
