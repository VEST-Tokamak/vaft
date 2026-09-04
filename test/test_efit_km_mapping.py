"""Current-EFIT k-/m-file parsing and final ODS assembly."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from omas import ODS, load_omas_json, save_omas_json
from scipy.io import netcdf_file

from vaft.code.efit import EFITConfig, collect_efit_outputs
from vaft.data import read_keqdsk, read_meqdsk
from vaft.data.resources import data_path
from vaft.database._summary import extract_efit_magnetic_reliability


def _write_current_efit_mfile(path: Path) -> None:
    with netcdf_file(path, "w") as dataset:
        dataset.history = b"synthetic current-EFIT write_m.F90 fixture"
        dataset.createDimension("dim_time", 1)
        dataset.createDimension("magpri", 2)
        dataset.createDimension("nsilop", 2)
        dataset.createDimension("nfsum", 2)
        dataset.createDimension("kxiter", 2)

        def variable(name, data, dimensions):
            item = dataset.createVariable(name, "f", dimensions)
            item.long_name = f"fixture {name}".encode()
            item[:] = data

        for prefix, dimensions, values in (
            (("expmpi", "cmpr2", "sigmpi", "fwtmp2", "saimpi"), ("dim_time", "magpri"),
             ([1.0, 2.0], [1.1, 2.1], [0.1, 0.2], [10.0, 20.0], [0.01, 0.02])),
            (("silopt", "csilop", "sigsil", "fwtsi", "saisil"), ("dim_time", "nsilop"),
             ([3.0, 4.0], [3.1, 4.1], [0.3, 0.4], [30.0, 40.0], [0.03, 0.04])),
            (("fccurt", "ccbrsp", "sigfcc", "fwtfc", "chifcc"), ("dim_time", "nfsum"),
             ([5.0, 6.0], [5.1, 6.1], [0.5, 0.6], [50.0, 60.0], [0.05, 0.06])),
        ):
            for name, data in zip(prefix, values):
                variable(name, [data], dimensions)
        for name, data in (
            ("plasma", [70_000.0]), ("cpasma", [69_500.0]),
            ("sigpasma", [500.0]), ("fwtpasma", [4.0]), ("chipasma", [0.25]),
            ("diamag", [0.007]), ("cdflux", [0.0065]),
            ("sigdia", [0.0005]), ("fwtdia", [8.0]), ("chidflux", [0.5]),
        ):
            variable(name, data, ("dim_time",))
        variable("cerror", [[0.1, 0.001]], ("dim_time", "kxiter"))
        variable("chifin", [1.25], ("dim_time",))
        variable("unsupported_current_efit_value", [42.0], ("dim_time",))


def test_keqdsk_preserves_all_namelists_and_fortran_repetition(tmp_path):
    path = tmp_path / "k039915.00319"
    path.write_text(
        "&IN1\n PLASMA=70000., EXPMP2=2*1.5, FWTCUR=1.0\n/\n"
        "&INWANT\n RZERO=0.4\n/\n",
        encoding="utf-8",
    )

    item = read_keqdsk(path)
    ods = item.to_omas()

    assert item["in1"]["expmp2"] == [1.5, 1.5]
    assert ods["equilibrium.code.parameters.time_slice.0.in1.fwtcur"] == 1.0
    assert ods["equilibrium.code.parameters.time_slice.0.inwant.rzero"] == 0.4


def test_meqdsk_maps_only_current_efit_output_names(tmp_path):
    path = tmp_path / "m039915.00319.nc"
    _write_current_efit_mfile(path)

    ods = read_meqdsk(path).to_omas()

    np.testing.assert_allclose(
        ods["equilibrium.time_slice.0.constraints.bpol_probe.:.reconstructed"],
        [1.1, 2.1],
    )
    assert ods["equilibrium.time_slice.0.constraints.ip.weight"] == 4.0
    assert ods["equilibrium.time_slice.0.constraints.diamagnetic_flux.weight"] == 8.0
    assert ods["equilibrium.time_slice.0.convergence.iterations_n"] == 2
    assert ods["equilibrium.time_slice.0.constraints.chi_squared_reduced"] == 1.25
    assert ods["equilibrium.code.parameters.time_slice.0.meqdsk.variables.unsupported_current_efit_value.data"] == 42.0

    rows = extract_efit_magnetic_reliability(ods, 39915)
    assert {row["measurement_type"] for row in rows} == {
        "bpol_probe",
        "flux_loop",
        "pf_current",
        "ip",
        "diamagnetic_flux",
    }
    assert all(np.isfinite(row["measured"]) for row in rows)
    assert all(np.isfinite(row["reconstructed"]) for row in rows)


def test_collection_merges_constraints_kfile_and_nc_mfile_with_mfile_precedence(tmp_path):
    for directory in ("gfile", "kfile", "mfile"):
        (tmp_path / directory).mkdir()
    (tmp_path / "gfile" / "g039915.00319").write_text(
        data_path("efit/g039915.00319").read_text(encoding="utf-8"), encoding="utf-8"
    )
    kfile = tmp_path / "kfile" / "k039915.00319"
    kfile.write_text("&IN1\n PLASMA=70000., FWTCUR=1.0, DFLUX=7.0, FWTDLC=1.0\n/\n", encoding="utf-8")
    _write_current_efit_mfile(tmp_path / "mfile" / "m039915.00319.nc")

    constraints = ODS()
    constraints["equilibrium.time"] = [0.319]
    root = "equilibrium.time_slice.0.constraints.ip"
    constraints[f"{root}.measured"] = 70_100.0
    constraints[f"{root}.measured_error_upper"] = 1_000.0
    constraints[f"{root}.weight"] = 1.0
    constraints[f"{root}.source"] = "submitted diagnostics"

    result = collect_efit_outputs(
        tmp_path,
        EFITConfig(shot=39915),
        constraints_ods=constraints,
    )

    assert len(result.slice_statuses) == 1
    assert result.slice_statuses[0].usable
    assert len(result.meqdsk) == 1
    ods = result.ods
    assert ods["equilibrium.time_slice.0.constraints.ip.measured"] == 70_000.0
    assert ods["equilibrium.time_slice.0.constraints.ip.measured_error_upper"] == 500.0
    assert ods["equilibrium.time_slice.0.constraints.ip.weight"] == 4.0
    assert ods["equilibrium.time_slice.0.constraints.ip.source"] == "submitted diagnostics"
    assert ods["equilibrium.code.parameters.time_slice.0.in1.fwtcur"] == 1.0
    assert result.mapping_diagnostics[0]["differences"]

    output = tmp_path / "final_equilibrium.json"
    save_omas_json(ods, str(output))
    restored = load_omas_json(str(output), consistency_check=True)
    assert restored["equilibrium.time_slice.0.constraints.ip.weight"] == 4.0
    assert (
        restored[
            "equilibrium.code.parameters.time_slice.0.meqdsk.variables.fwtdia.data"
        ]
        == 8.0
    )


def test_mfile_kfile_input_names_are_not_accepted_as_output_aliases(tmp_path):
    path = tmp_path / "m039915.00319.nc"
    with netcdf_file(path, "w") as dataset:
        dataset.createDimension("dim_time", 1)
        for name, value in (("fwtcur", 9.0), ("fwtdlc", 11.0)):
            variable = dataset.createVariable(name, "f", ("dim_time",))
            variable[:] = [value]

    ods = read_meqdsk(path).to_omas()

    assert "equilibrium.time_slice.0.constraints.ip.weight" not in ods
    assert "equilibrium.time_slice.0.constraints.diamagnetic_flux.weight" not in ods
    assert ods["equilibrium.code.parameters.time_slice.0.meqdsk.variables.fwtcur.data"] == 9.0
