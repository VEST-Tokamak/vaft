"""Ideal-GPEC native output parsing and DCON-prerequisite validation.

No real GPEC binary is required: the tests use synthetic fixtures shaped like
the real ideal-GPEC netCDF conventions (see ``gpec_nc_fixtures``), so the
parsing logic is exercised hermetically against the source-verified layout.
An optional integration test reads the real shot-48226 reference bundle when
``VAFT_GPEC_REFERENCE_DIR`` points at it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from gpec_nc_fixtures import write_control_nc, write_cylindrical_nc
from vaft.code import gpec
from vaft.code.gpec import GpecIdealResult, read_gpec_netcdf, validate_dcon_result

from external_code_stubs import write_launchable_stub


def test_reads_control_and_cylindrical(tmp_path):
    control_data = write_control_nc(tmp_path)
    cylindrical_data = write_cylindrical_nc(tmp_path)

    result = read_gpec_netcdf(tmp_path)
    assert result.n_tor == 1
    control = result.control
    assert control.machine == "VEST"
    assert control.jacobian == "hamada"
    assert control.helicity == -1
    assert control.energy_total == pytest.approx(1.5)
    np.testing.assert_allclose(control.b_n, control_data["b_n"])
    np.testing.assert_allclose(control.b_n_fun, control_data["b_n_fun"])
    assert control.coil_names == ("MID",)
    np.testing.assert_allclose(control.Phi_coil, control_data["phi_coil"])
    assert "W_e_eigenvalue" in control.extras

    cylindrical = result.cylindrical
    np.testing.assert_allclose(cylindrical.R, cylindrical_data["R"])
    np.testing.assert_allclose(cylindrical.b_r_plasma, cylindrical_data["b_plasma"])
    np.testing.assert_allclose(cylindrical.b_r, cylindrical_data["b_total"])
    assert cylindrical.units["b_r"] == "Tesla"
    assert cylindrical.b_r.shape == (cylindrical.z.size, cylindrical.R.size)


def test_n_tor_comes_from_the_attribute_not_the_filename(tmp_path):
    write_control_nc(tmp_path, n=2, filename_mode=1)
    with pytest.warns(UserWarning, match="trusting the attribute"):
        result = GpecIdealResult.from_netcdf(
            tmp_path, control_path=tmp_path / "gpec_control_output_n1.nc"
        )
    assert result.n_tor == 2


def test_missing_n_attribute_is_an_error(tmp_path):
    import xarray as xr

    xr.Dataset(attrs={"machine": "VEST"}).to_netcdf(
        tmp_path / "gpec_control_output_n1.nc"
    )
    with pytest.raises(ValueError, match="'n' global attribute"):
        read_gpec_netcdf(tmp_path)


def test_missing_cylindrical_is_tolerated(tmp_path):
    write_control_nc(tmp_path)
    result = read_gpec_netcdf(tmp_path)
    assert result.cylindrical is None
    assert "cylindrical" not in result.source_paths


def test_mode_mismatch_between_files_is_an_error(tmp_path):
    write_control_nc(tmp_path, n=1)
    write_cylindrical_nc(tmp_path, n=2)
    with pytest.raises(ValueError, match="cylindrical file is"):
        GpecIdealResult.from_netcdf(
            tmp_path,
            cylindrical_path=tmp_path / "gpec_cylindrical_output_n2.nc",
        )


def test_multiple_control_files_need_an_explicit_mode(tmp_path):
    write_control_nc(tmp_path, n=1)
    write_control_nc(tmp_path, n=2)
    with pytest.raises(ValueError, match="pass mode="):
        read_gpec_netcdf(tmp_path)
    assert read_gpec_netcdf(tmp_path, mode=2).n_tor == 2


def test_empty_directory_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="not a completed"):
        read_gpec_netcdf(tmp_path)


def test_write_json_sidecar(tmp_path):
    control_data = write_control_nc(tmp_path)
    write_cylindrical_nc(tmp_path)
    result = read_gpec_netcdf(tmp_path)

    sidecar = result.write_json(tmp_path / "gpec_ideal_n1.json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["control"]["n_tor"] == 1
    np.testing.assert_allclose(
        payload["control"]["b_n"]["real"], control_data["b_n"].real
    )
    assert payload["control"]["coil_names"] == ["MID"]
    assert "W_e_eigenvalue" in payload["control"]["extras"]
    # Cylindrical bulk arrays live in the source .nc, not the sidecar.
    assert "cylindrical" in payload["source_paths"]


def test_validate_dcon_result_lists_missing_products(tmp_path):
    problems = validate_dcon_result(tmp_path, 1)
    assert [p for p in problems if "euler.bin" in p]
    assert [p for p in problems if "psi_in.bin" in p]

    (tmp_path / "euler.bin").write_bytes(b"")
    (tmp_path / "psi_in.bin").write_bytes(b"")
    assert validate_dcon_result(tmp_path, 1) == []


def test_validate_dcon_result_verifies_nc_content(tmp_path):
    (tmp_path / "euler.bin").write_bytes(b"")
    (tmp_path / "psi_in.bin").write_bytes(b"")
    problems = validate_dcon_result(tmp_path, 1, verify_outputs=True)
    assert problems and "dcon_output_n1.nc" in problems[0]

    import xarray as xr

    xr.Dataset(
        {"W_t_eigenvalue": (("mode", "i"), [[1.0, 0.0]])},
        coords={"i": [0, 1], "mode": [1]},
    ).to_netcdf(tmp_path / "dcon_output_n1.nc")
    assert validate_dcon_result(tmp_path, 1, verify_outputs=True) == []


GFILE_TEXT = "  EFITD   01/01/2024   #  48226  300ms        3  65  65\n 1.0 2.0 3.0\n"


def _stub_installation(tmp_path):
    executable = write_launchable_stub(tmp_path / "gpec/bin/gpec")
    return tmp_path / "gpec"


@pytest.fixture()
def case(tmp_path):
    geqdsk = tmp_path / "g048226.00300"
    geqdsk.write_text(GFILE_TEXT, encoding="utf-8")
    return gpec.GPECCaseInputs(
        shot=48226, time_ms=300, geqdsk=geqdsk, workdir=tmp_path / "run"
    )


def test_run_without_dcon_result_is_skipped_with_guidance(monkeypatch, tmp_path, case):
    monkeypatch.setenv(gpec.GPEC_HOME_ENV, str(_stub_installation(tmp_path)))
    result = gpec.run_gpec_suite_case(
        case, gpec.GPECSuiteConfig(modules=("gpec",), modes=(1,))
    )
    (record,) = result.records
    assert record.status == "skipped"
    assert "invalid DCON result" in record.reason
    assert "euler.bin" in record.reason
    assert "dcon_workdir" in record.reason


def test_strict_run_without_dcon_result_raises(monkeypatch, tmp_path, case):
    monkeypatch.setenv(gpec.GPEC_HOME_ENV, str(_stub_installation(tmp_path)))
    with pytest.raises(RuntimeError, match="invalid DCON result"):
        gpec.run_gpec_suite_case(
            case,
            gpec.GPECSuiteConfig(modules=("gpec",), modes=(1,), run_mode="strict"),
        )


@pytest.mark.integration
def test_reference_bundle_parses():
    reference = os.environ.get("VAFT_GPEC_REFERENCE_DIR")
    if not reference:
        pytest.skip("VAFT_GPEC_REFERENCE_DIR not set")
    result = read_gpec_netcdf(Path(reference) / "04_ideal_gpec_result")
    control = result.control
    assert result.n_tor == 1
    assert control.machine == "VEST"
    assert control.coil_names == ("12inch_20turn",)
    assert control.energy_total == pytest.approx(1.2068106340712506)
    assert result.cylindrical.R.size == 129
