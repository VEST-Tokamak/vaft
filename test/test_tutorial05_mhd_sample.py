"""The pieces Tutorial 05 runs on (#2278), tested without any solver.

* ``vaft.code.stage_status``: the four facts of a run-or-load stage stay apart.
* ``GPECSuiteConfig.mpsi`` / ``DCONOptions.termbycross_flag`` reach the prepared namelists.
* ``vaft.data.resources.mhd_stability_sample``: the precomputed CHEASE / DCON /
  RDCON / GPEC products load as IMAS-compatible ODS and still carry the
  quantities they were accepted on.
* ``vaft.omas.compute_gpec_resonant_table`` reproduces the build's recorded
  total and vacuum resonant fields from the mapped IDS alone.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from vaft.code import gpec
from vaft.code.stage_status import StageStatus, executable_ready, format_stage_table
from vaft.data.resources import data_path, mhd_stability_manifest, mhd_stability_sample

SAMPLE_DIR = data_path("samples/39915/mhd_stability")


# --------------------------------------------------------------- stage status
def test_a_stage_keeps_its_four_facts_apart():
    stage = StageStatus("GPEC")
    stage.execution = "failed"
    stage.source = "precomputed sample"
    stage.validation = "passed"
    stage.reason = "timeout"
    assert stage.fell_back
    line = format_stage_table([stage]).splitlines()[1]
    assert "failed" in line and "precomputed sample" in line and "passed" in line and "timeout" in line


def test_a_stage_refuses_a_word_outside_its_vocabulary():
    stage = StageStatus("DCON")
    with pytest.raises(ValueError):
        stage.execution = "succeeded"
    with pytest.raises(ValueError):
        stage.source = "sample"


def test_an_offline_sample_is_not_a_fallback():
    stage = StageStatus("RDCON", source="precomputed sample")
    assert not stage.fell_back


def test_found_is_not_ready(tmp_path):
    assert executable_ready(None) == (False, "not found")
    plain = tmp_path / "dcon"
    plain.write_text("")
    plain.chmod(0o644)
    assert executable_ready(plain)[0] is False
    plain.chmod(0o755)
    assert executable_ready(plain) == (True, str(plain))
    assert executable_ready(tmp_path)[0] is False


# --------------------------------------------------------------- adapter options
@pytest.fixture()
def case(tmp_path):
    geqdsk = tmp_path / "g039915.00319"
    shutil.copy(data_path("efit/g039915.00319"), geqdsk)
    return gpec.GPECCaseInputs(shot=39915, time_ms=319, geqdsk=geqdsk, workdir=tmp_path / "run")


def _namelist_value(path: Path, key: str) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.split("!")[0].strip()
        if stripped.replace(" ", "").lower().startswith(f"{key}="):
            return stripped.split("=", 1)[1].strip()
    raise AssertionError(f"{key} not in {path}")


def test_mpsi_and_termbycross_reach_the_prepared_namelists(case):
    gpec.prepare_gpec_suite_case(
        case,
        gpec.GPECSuiteConfig(modules=("dcon",), modes=(1,), mpsi=512,
                             dcon=gpec.DCONOptions(termbycross_flag=False, con_flag=False)),
    )
    run_dir = case.workdir / "00319" / "dcon" / "nn=1"
    assert _namelist_value(run_dir / "equil.in", "mpsi") == "512"
    assert _namelist_value(run_dir / "dcon.in", "termbycross_flag").lower().strip(".") in ("f", "false")
    assert _namelist_value(run_dir / "dcon.in", "con_flag").lower().strip(".") in ("f", "false")


def test_unset_options_leave_the_template_alone(case):
    gpec.prepare_gpec_suite_case(case, gpec.GPECSuiteConfig(modules=("dcon",), modes=(1,)))
    run_dir = case.workdir / "00319" / "dcon" / "nn=1"
    template = data_path("gpec")
    assert _namelist_value(run_dir / "equil.in", "mpsi") == _namelist_value(template / "equil.in", "mpsi")
    assert _namelist_value(run_dir / "dcon.in", "termbycross_flag") == \
        _namelist_value(template / "dcon.in", "termbycross_flag")


# --------------------------------------------------------------- the sample
@pytest.fixture(scope="module")
def manifest():
    if not SAMPLE_DIR.is_dir():
        pytest.skip("the Tutorial 05 MHD sample is repository-only")
    return mhd_stability_manifest()


def test_the_manifest_is_complete_and_matches_the_files(manifest):
    import hashlib

    assert manifest["status"] == "validated"
    assert manifest["parent_manifest"] == "../manifest.yaml"
    assert manifest["inputs"]["source_sample_sha256"] == hashlib.sha256(
        data_path("samples/39915/omas.json.gz").read_bytes()).hexdigest()
    assert manifest["inputs"]["source_time_ms"] == manifest["time_ms"]
    for key in ("gpec", "chease", "configuration", "script_sha256"):
        assert manifest["inputs"][key], key
    expected = {f"chease_{c}" for c in ("stable", "high_ip")} \
        | {f"dcon_{c}_mpsi{m}" for c in ("stable", "high_ip") for m in (256, 512)} \
        | {f"rdcon_stable_mpsi{m}" for m in (256, 512, 1024)} | {"gpec_UP", "gpec_LOW"} \
        | {f"gpec_MID_n{n}" for n in (1, 2, 3)}
    assert set(manifest["products"]) == expected
    on_disk = {path.name for path in SAMPLE_DIR.iterdir()} - {"manifest.yaml"}
    assert on_disk == {record["path"] for record in manifest["products"].values()}
    for record in manifest["products"].values():
        assert record["validation"] == "passed"
        assert hashlib.sha256((SAMPLE_DIR / record["path"]).read_bytes()).hexdigest() == record["sha256"]


def test_dcon_verdicts_survive_serialisation(manifest):
    for case, negative in (("stable", False), ("high_ip", True)):
        dominant = set()
        for mpsi in (256, 512):
            ods = mhd_stability_sample(f"dcon_{case}_mpsi{mpsi}")
            mode = ods["mhd_linear.time_slice.0.toroidal_mode.0"]
            check = manifest["products"][f"dcon_{case}_mpsi{mpsi}"]["check"]
            assert (float(mode["energy_perturbed"]) < 0) is negative
            assert float(mode["energy_perturbed"]) == pytest.approx(check["W_t"])
            assert int(mode["m_pol_dominant"]) == check["m_pol_dominant"]
            assert np.asarray(mode["plasma.displacement_perpendicular.real"]).ndim == 2
            dominant.add(int(mode["m_pol_dominant"]))
        if negative:
            assert dominant == {3}


def test_rdcon_delta_prime_comes_back_from_ntms(manifest):
    from vaft.machine_mapping.mhd_linear import extract_rdcon_stability

    for mpsi in (256, 512, 1024):
        [row] = extract_rdcon_stability(mhd_stability_sample(f"rdcon_stable_mpsi{mpsi}"))
        recorded = manifest["products"][f"rdcon_stable_mpsi{mpsi}"]["check"]["surfaces"]
        assert [s["m"] for s in row["surfaces"]] == [s["m"] for s in recorded]
        np.testing.assert_allclose([s["delta_prime"].real for s in row["surfaces"]],
                                   [s["delta_prime_real"] for s in recorded], rtol=1e-12)


def test_gpec_resonant_table_is_recovered_from_the_mapped_ids(manifest):
    import vaft

    for name, surfaces in (("UP", 6), ("LOW", 6), ("MID_n1", 6), ("MID_n2", 12), ("MID_n3", 18)):
        table = vaft.omas.compute_gpec_resonant_table(mhd_stability_sample(f"gpec_{name}"))
        check = manifest["products"][f"gpec_{name}"]["check"]
        total = [abs(row["Phi_res"]) for row in table["rows"]]
        vacuum = [abs(row["Phi_res_vacuum"]) for row in table["rows"]]
        assert table["n_tor"] == check["n_tor"]
        assert len(total) == check["n_rational_surfaces"] == surfaces
        # The derived jump reproduces GPEC's own; 1 % leaves room for a weak near-axis surface.
        np.testing.assert_allclose(total, check["abs_phi_res_native"], rtol=1e-2)
        np.testing.assert_allclose(vacuum, check["abs_phi_res_vacuum"], rtol=1e-9)
        assert max(t / v for t, v in zip(total, vacuum)) < 1.0


def test_the_delta_prime_overview_renders_from_a_sample(manifest):
    import matplotlib

    matplotlib.use("Agg")
    import vaft

    figure, axes = vaft.omas.plot_ntms_overview_delta_prime(mhd_stability_sample("rdcon_stable_mpsi512"))
    assert len(np.ravel(axes)) == 2
    import matplotlib.pyplot as plt

    plt.close(figure)


def test_every_gpec_product_carries_its_flux_surface_mesh(manifest):
    """The real-space figures need GPEC's own R(psi_N, theta), z(psi_N, theta)."""
    for key in ("gpec_UP", "gpec_LOW", "gpec_MID_n1", "gpec_MID_n2", "gpec_MID_n3"):
        mesh = mhd_stability_sample(key)["mhd_linear.time_slice.0.toroidal_mode.0.plasma.coordinate_system"]
        psi = np.asarray(mesh["grid.dim1"], dtype=float)
        theta = np.asarray(mesh["grid.dim2"], dtype=float)
        check = manifest["products"][key]["check"]
        assert np.asarray(mesh["r"]).shape == (psi.size, theta.size) == (check["mesh_radial_points"],
                                                                          check["mesh_poloidal_angles"])
        assert theta[0] == 0.0 and theta[-1] == pytest.approx(1.0)
        assert np.all(np.diff(psi) > 0)
