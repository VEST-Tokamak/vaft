"""Unit tests for parsing GPEC-suite `.nc` output into the `mhd_linear` IDS.

No real DCON/RDCON/STRIDE binary is required: these use synthetic `.nc`
fixtures with the exact variable names/shapes the parser expects (and, for
the malformed case, without them) so the parsing logic is exercised
hermetically.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest
import xarray as xr
from omas import ODS

from vaft.machine_mapping import mhd_linear


def _write_dcon_output(path, *, n: int, w_t: float) -> None:
    ds = xr.Dataset(
        {"W_t_eigenvalue": (("i", "mode"), [[w_t]])},
        coords={"i": [0], "mode": [1]},
    )
    ds.to_netcdf(path / f"dcon_output_n{n}.nc")


def _write_resistive_output(module: str, path, *, n: int, delta_prime: float) -> None:
    # Shaped like real RDCON/STRIDE output: a [1,1,1]-style scalar wrapper.
    ds = xr.Dataset({"Delta_prime": (("a", "b", "c"), [[[delta_prime]]])})
    ds.to_netcdf(path / f"{module}_output_n{n}.nc")


def test_dcon_module_writes_energy_perturbed_and_n_tor(tmp_path):
    _write_dcon_output(tmp_path, n=1, w_t=-0.42)
    ods = ODS()

    extras = mhd_linear(ods, str(tmp_path), {"module": "dcon", "time_slice": 0})

    # `toroidal_mode` is an AOS: array position is insertion order, not the
    # physical mode number, which is only ever recovered via `n_tor`.
    assert ods["mhd_linear"]["time_slice"][0]["toroidal_mode"][0]["n_tor"] == 1
    assert ods["mhd_linear"]["time_slice"][0]["toroidal_mode"][0]["energy_perturbed"] == pytest.approx(-0.42)
    assert extras == {1: {"module": "dcon", "variable": "W_t_eigenvalue", "value": pytest.approx(-0.42)}}


def test_rdcon_module_has_no_ids_slot_for_delta_prime_but_classifies_ballooning_type(tmp_path):
    _write_resistive_output("rdcon", tmp_path, n=2, delta_prime=1.23)
    ods = ODS()

    extras = mhd_linear(ods, str(tmp_path), {"module": "rdcon", "time_slice": 0})

    assert ods["mhd_linear"]["time_slice"][0]["toroidal_mode"][0]["n_tor"] == 2
    assert ods["mhd_linear"]["time_slice"][0]["toroidal_mode"][0]["ballooning_type"]["name"] == "Tearing"
    assert "energy_perturbed" not in ods["mhd_linear"]["time_slice"][0]["toroidal_mode"][0]
    assert extras[2]["value"] == pytest.approx(1.23)


def test_stride_module_extracts_delta_prime_the_same_way_as_rdcon(tmp_path):
    _write_resistive_output("stride", tmp_path, n=1, delta_prime=-0.05)
    ods = ODS()

    extras = mhd_linear(ods, str(tmp_path), {"module": "stride", "time_slice": 0})

    assert extras[1]["value"] == pytest.approx(-0.05)
    assert ods["mhd_linear"]["time_slice"][0]["toroidal_mode"][0]["ballooning_type"]["name"] == "Tearing"


def test_missing_variable_is_skipped_not_fatal(tmp_path):
    # Present file, wrong variable -- must not raise.
    ds = xr.Dataset({"some_other_variable": (("i",), [1.0])})
    ds.to_netcdf(tmp_path / "dcon_output_n1.nc")
    ods = ODS()

    extras = mhd_linear(ods, str(tmp_path), {"module": "dcon", "time_slice": 0})

    assert extras == {}
    assert len(ods["mhd_linear"]) == 0


def test_no_matching_files_is_skipped_not_fatal(tmp_path):
    ods = ODS()

    extras = mhd_linear(ods, str(tmp_path), {"module": "rdcon", "time_slice": 0})

    assert extras == {}


def test_unsupported_module_raises():
    ods = ODS()
    with pytest.raises(ValueError, match="gpec"):
        mhd_linear(ods, "/nonexistent", {"module": "gpec"})


def _write_single_step_solutions_bin(path, *, psi: float, real: float, imaginary: float) -> None:
    """One perturbation block, one step: [psi, _, _, real, imaginary, _, _]."""
    vec7 = [psi, 0.0, 0.0, real, imaginary, 0.0, 0.0]
    with open(path / "solutions.bin", "wb") as f:
        payload = struct.pack("<7f", *vec7)
        f.write(struct.pack("<i", len(payload)))
        f.write(payload)
        f.write(struct.pack("<i", len(payload)))  # trailing length (discarded)
        f.write(struct.pack("<i", 0))  # zero-length record ends this block's steps


def test_dcon_solutions_bin_appends_displacement_after_nc_derived_modes(tmp_path):
    """solutions.bin populates plasma.grid/displacement -- previously crashed on
    every real invocation (wrong ODS path, non-sequential AOS indexing)."""
    _write_dcon_output(tmp_path, n=1, w_t=-0.42)
    _write_single_step_solutions_bin(tmp_path, psi=0.5, real=1.5, imaginary=-2.5)
    ods = ODS()

    mhd_linear(ods, str(tmp_path), {"module": "dcon", "time_slice": 0})

    modes = ods["mhd_linear"]["time_slice"][0]["toroidal_mode"]
    assert len(modes) == 2
    assert modes[0]["energy_perturbed"] == pytest.approx(-0.42)
    displacement = modes[1]["plasma"]["displacement_perpendicular"]
    assert displacement["real"] == pytest.approx([1.5])
    assert displacement["imaginary"] == pytest.approx([-2.5])
    assert modes[1]["plasma"]["grid"]["dim1"] == pytest.approx([0.5])


def test_defaults_to_dcon_module_when_unspecified(tmp_path):
    _write_dcon_output(tmp_path, n=1, w_t=0.1)
    ods = ODS()

    extras = mhd_linear(ods, str(tmp_path))

    assert extras[1]["module"] == "dcon"
