"""Ideal-GPEC output mapping into ``mhd_linear`` and the stage builder.

Synthetic fixtures shaped like the real ideal-GPEC netCDF layout (see
``gpec_nc_fixtures``) exercise the mapping hermetically: complex plasma
response fields, the derived (total minus plasma) vacuum field, the dense
toroidal-mode grid, shot/time injection, provenance, and the
``build_gpec_ideal_ods`` stage contract.
"""

from __future__ import annotations

import numpy as np
import pytest
from omas import ODS

from gpec_nc_fixtures import write_control_nc, write_cylindrical_nc
from vaft.code.gpec import _runtime as gpec_runtime
from vaft.machine_mapping.gpec_ideal import gpec_ideal
from vaft.omas.vest_upstream import build_gpec_ideal_ods


REFERENCE_COIL_IN = """&COIL_CONTROL
 coil_num=1
 coil_name(1)="MID"
 coil_cur(1,1)=200.0
 coil_cur(1,2)=200.0
 coil_cur(1,3)=0.0
 coil_cur(1,4)=-200.0
 coil_cur(1,5)=-200.0
 coil_cur(1,6)=0.0
/
"""


@pytest.fixture()
def run_dir(tmp_path):
    write_control_nc(tmp_path)
    write_cylindrical_nc(tmp_path)
    return tmp_path


def test_maps_plasma_and_derived_vacuum_fields(run_dir):
    ods = ODS(consistency_check=False)
    data = write_cylindrical_nc(run_dir)  # regenerate to hold expected arrays
    extras = gpec_ideal(ods, str(run_dir), {"time_s": 0.3})

    entry = ods["mhd_linear.time_slice.0.toroidal_mode.0"]
    assert entry["n_tor"] == 1
    assert entry["energy_perturbed"] == pytest.approx(1.5)

    np.testing.assert_allclose(entry["plasma.grid.dim1"], data["R"])
    np.testing.assert_allclose(entry["plasma.grid.dim2"], data["z"])
    plasma_r = (
        entry["plasma.b_field_perturbed.coordinate1.real"]
        + 1j * entry["plasma.b_field_perturbed.coordinate1.imaginary"]
    )
    # Native (z, R) arrays land as (dim1, dim2) = (R, z).
    np.testing.assert_allclose(plasma_r, data["b_plasma"].T)

    vacuum_r = (
        entry["vacuum.b_field_perturbed.coordinate1.real"]
        + 1j * entry["vacuum.b_field_perturbed.coordinate1.imaginary"]
    )
    np.testing.assert_allclose(vacuum_r, (data["b_total"] - data["b_plasma"]).T)

    assert extras[1]["energy_perturbed"] == pytest.approx(1.5)
    assert (run_dir / "gpec_ideal_native_n1.json").exists()


def test_time_injection_overrides_zero_attrs(run_dir):
    ods = ODS(consistency_check=False)
    gpec_ideal(ods, str(run_dir), {"time_s": 0.3})
    assert ods["mhd_linear.ids_properties.homogeneous_time"] == 1
    np.testing.assert_allclose(ods["mhd_linear.time"], [0.3])
    assert ods["mhd_linear.time_slice.0.time"] == pytest.approx(0.3)


def test_dense_mode_grid_positions(run_dir):
    ods = ODS(consistency_check=False)
    gpec_ideal(ods, str(run_dir), {"modes": [1, 2]})
    modes = ods["mhd_linear.time_slice.0.toroidal_mode"]
    assert len(modes) == 2
    assert modes[0]["n_tor"] == 1
    assert modes[1]["n_tor"] == 2  # padded entry: n_tor only
    assert "energy_perturbed" not in modes[1]


def test_include_vacuum_false_omits_vacuum(run_dir):
    ods = ODS(consistency_check=False)
    gpec_ideal(ods, str(run_dir), {"include_vacuum": False})
    entry = ods["mhd_linear.time_slice.0.toroidal_mode.0"]
    assert "b_field_perturbed" not in entry["vacuum"] if "vacuum" in entry else True
    assert "real" in entry["plasma.b_field_perturbed.coordinate1"]


def test_code_parameters_provenance(run_dir):
    ods = ODS(consistency_check=False)
    gpec_ideal(ods, str(run_dir))
    parameters = ods["mhd_linear.code.parameters"]
    for token in (
        'solver name="gpec"',
        'derivation="energy_vacuum+energy_surface+energy_plasma"',
        'derivation="total_minus_plasma"',
        "<jacobian>hamada</jacobian>",
    ):
        assert token in parameters
    assert ods["mhd_linear.code.name"] == "GPEC"
    assert ods["mhd_linear.code.version"] == "v1.5.5-test"
    assert ods["mhd_linear.code.output_flag"][0] == 0


def test_control_only_run_still_maps_energy(tmp_path):
    write_control_nc(tmp_path)
    ods = ODS(consistency_check=False)
    gpec_ideal(ods, str(tmp_path))
    entry = ods["mhd_linear.time_slice.0.toroidal_mode.0"]
    assert entry["energy_perturbed"] == pytest.approx(1.5)
    assert "grid" not in entry["plasma"] if "plasma" in entry else True


def _make_cell(root, time_ms, mode):
    run_dir = gpec_runtime.module_dir(root, time_ms, "gpec", mode)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_control_nc(run_dir, n=mode)
    write_cylindrical_nc(run_dir, n=mode)
    (run_dir / "coil.in").write_text(REFERENCE_COIL_IN, encoding="utf-8")
    return run_dir


def test_build_gpec_ideal_ods(tmp_path):
    _make_cell(tmp_path, 300, 1)
    ods, manifest = build_gpec_ideal_ods(
        shot=48226, time_values=[300], workdir=tmp_path, modes=[1]
    )

    assert manifest["stage"] == "gpec_ideal"
    assert manifest["status"] == "success"
    cell = manifest["modules_modes"]["t=300/gpec/n=1"]
    assert cell["status"] == "success"
    hashed = manifest["input"]
    assert "t=300/gpec/n=1/gpec_control_output_n1.nc" in hashed
    assert "t=300/gpec/n=1/coil.in" in hashed

    # Field and cause travel together: the run's excitation reaches the
    # canonical coil geometry, matched by identifier with turns preserved.
    identifiers = [
        ods[f"coils_non_axisymmetric.coil.{i}.identifier"]
        for i in range(len(ods["coils_non_axisymmetric.coil"]))
    ]
    assert len(identifiers) == 18
    mid01 = identifiers.index("VEST_3D_MID_01")
    np.testing.assert_allclose(
        ods[f"coils_non_axisymmetric.coil.{mid01}.current.data"], [200.0]
    )
    assert ods[f"coils_non_axisymmetric.coil.{mid01}.turns"] == 20.0

    assert ods["mhd_linear.time_slice.0.toroidal_mode.0.n_tor"] == 1
    np.testing.assert_allclose(ods["mhd_linear.time"], [0.3])


def test_build_gpec_ideal_ods_records_missing_cells(tmp_path):
    ods, manifest = build_gpec_ideal_ods(
        shot=48226, time_values=[300], workdir=tmp_path, modes=[1]
    )
    assert manifest["status"] == "empty"
    assert manifest["modules_modes"]["t=300/gpec/n=1"]["status"] == "missing"
    # The dense grid still exists with a padded, flagged entry.
    assert ods["mhd_linear.time_slice.0.toroidal_mode.0.n_tor"] == 1
    assert ods["mhd_linear.code.output_flag"][0] == -1


def test_build_gpec_ideal_ods_mode_workdirs(tmp_path):
    root = tmp_path / "cellroot"
    _make_cell(root, 300, 1)
    ods, manifest = build_gpec_ideal_ods(
        shot=48226,
        time_values=[300],
        mode_workdirs={1: root},
        modes=[1],
    )
    assert manifest["status"] == "success"


def test_build_gpec_ideal_ods_requires_a_workdir():
    with pytest.raises(ValueError, match="workdir"):
        build_gpec_ideal_ods(shot=48226, time_values=[300], modes=[1])
