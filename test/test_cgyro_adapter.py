"""CGYRO adapter: input translation, output parsing, run verdicts (#1354).

No GACODE is needed. The launcher is replaced by a fake that writes CGYRO-shaped files in
the layouts ``f2py/pygacode/cgyro/data.py`` reads, and the translation is held against
the TGLF projection it is built from. The real-run checks (reg01, the PROFILE_MODEL=2
oracle) live in ``test_cgyro_oracle.py`` against fixtures captured on tdst.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vaft.code.gacode import cgyro
from vaft.code.gacode.cgyro import (
    CGYROConfig,
    CGYROExecutionError,
    CgyroOutputs,
    collect_cgyro_outputs,
)
from vaft.code.gacode.tglf.inputs import TGLFInput, TGLFNormalisation


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def tglf_local(**overrides) -> TGLFInput:
    """A three-species (e, H+, C6+) local input with VEST-like numbers."""
    values = dict(
        rho=0.7, rmin_loc=0.7, rmaj_loc=1.49, drmajdx_loc=-0.12, zmaj_loc=0.1,
        dzmajdx_loc=0.005, q_loc=2.3, q_prime_loc=(2.3 / 0.7) ** 2 * 0.92,
        p_prime_loc=-0.01, kappa_loc=1.69, s_kappa_loc=0.017, delta_loc=0.19,
        s_delta_loc=0.13, zeta_loc=0.0, s_zeta_loc=0.0,
        zs=np.array([-1.0, 1.0, 6.0]),
        mass=np.array([2.72443705e-4, 0.50392, 6.0055]),
        as_=np.array([1.0, 0.8, 1.0 / 30.0]),
        taus=np.array([1.0, 1.0, 1.0]),
        rlns=np.array([4.9, 4.9, 4.9]),
        rlts=np.array([1.5, 1.4, 1.4]),
        betae=1.0e-3, xnue=0.69, zeff=2.0, debye=0.0099,
        sign_bt=1.0, sign_it=-1.0,
        names=("e", "H+", "C6+"),
        normalisation=TGLFNormalisation(
            electron_density=1.0e19, electron_temperature=50.0 * 1.602176634e-19,
            sound_speed=4.9e4, gyroradius=1.0e-3, minor_radius=0.3, b_unit=0.5,
        ),
        provenance={"vexb_shear": {"kind": "unavailable", "reason": "no w0"}},
    )
    values.update(overrides)
    return TGLFInput(**values)


def write_run(
    directory: Path,
    *,
    n_n: int = 1,
    n_species: int = 3,
    n_field: int = 1,
    n_radial: int = 4,
    n_theta: int = 8,
    steps: int = 5,
    gamma: float = 0.25,
    omega: float = 0.4,
    ion_direction: str = ">",
    exit_message: str | None = "Linear converged",
    error: str | None = None,
    b_gs2: float = 0.8,
    flux: bool = False,
    hiprec: int = 0,
) -> Path:
    """Write the files a CGYRO run leaves, in CGYRO's layouts."""
    directory.mkdir(parents=True, exist_ok=True)
    m_box = 1
    sizes = [n_n, n_species, n_field, n_radial, n_theta, 8, 16, m_box, 12.5, 4, 1]
    p = np.arange(n_radial) - n_radial // 2
    theta = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    thetab = np.concatenate([theta + 2 * np.pi * k for k in p])
    ky = 0.3 * np.arange(1, n_n + 1) if n_n == 1 else 0.1 * np.arange(n_n)
    grids = np.concatenate([
        sizes, p, theta, np.linspace(0, 8, 8), np.linspace(-1, 1, 16), thetab, ky,
        np.zeros(n_n), np.zeros(n_radial),
    ])
    (directory / "out.cgyro.grids").write_text("\n".join(f"{v:.10g}" for v in grids))

    scalars = [0.7, 1.49, 2.3, 0.92, -0.12, 1.69, 0.017, 0.19, 0.13, 0.0, 0.0, 0.1, 0.005]
    shape = [0.0] * 22
    norms = [0.0] * 19
    norms[2] = 1.0e-3        # betae_unit
    norms[10] = b_gs2        # b_gs2
    species = []
    for z, mass, dens in ((1.0, 0.50392, 0.8), (6.0, 6.0055, 1 / 30), (-1.0, 2.72e-4, 1.0))[:n_species]:
        species += [z, mass, dens, 1.0, 4.9, 1.5, 0.1]
    tail = [0.0] * (2 * n_species) + [0.0, 2.0, hiprec]
    equilibrium = scalars + shape + norms + species + tail
    (directory / "out.cgyro.equilibrium").write_text("\n".join(f"{v:.10g}" for v in equilibrium))

    t = np.linspace(1.0, 5.0, steps)
    (directory / "out.cgyro.time").write_text(
        "\n".join(f"{x:.6e} {1e-3:.6e} {1e-3:.6e} 0.0" for x in t)
    )
    real = np.float64 if hiprec else np.float32
    cplx = np.complex128 if hiprec else np.complex64
    freq = np.zeros((2, n_n, steps))
    freq[0] = omega
    freq[1] = gamma
    freq.astype(real).flatten(order="F").tofile(directory / "bin.cgyro.freq")
    spatial = n_theta * n_radial
    phi = np.exp(-thetab**2)[:, None] * np.ones((1, steps))
    phi.astype(cplx).flatten(order="F").tofile(directory / "bin.cgyro.phib")
    if flux:
        data = np.ones((n_species, 3, n_field, n_n, steps))
        data[:, 1] *= 2.0
        data.astype(real).flatten(order="F").tofile(directory / "bin.cgyro.ky_flux")
    del spatial

    info = [
        "INFO: (CGYRO) Single-mode linear analysis",
        f"INFO: (CGYRO) Ion direction: omega {ion_direction} 0",
    ]
    if error:
        info.append(f"ERROR: (CGYRO) {error}")
    if exit_message:
        info.append(f"EXIT: (CGYRO) {exit_message}")
    (directory / "out.cgyro.info").write_text("\n".join(info) + "\n")
    (directory / "out.cgyro.version").write_text(
        "26-Oct-01 12:00:00 [b493397 [2026-08-20]][TDST_GNU][0.0]\n"
    )
    return directory


# --------------------------------------------------------------------------
# the translation from TGLF's local input
# --------------------------------------------------------------------------


def test_the_translation_renames_and_moves_electrons_last():
    local = cgyro.cgyro_input_from_tglf(tglf_local())
    assert list(local.species["Z"]) == [1.0, 6.0, -1.0]
    assert local.names == ("H+", "C6+", "e")
    assert local.species["DLNTDR"][-1] == pytest.approx(1.5)  # the electrons' a/L_T
    assert local.species["MASS"][-1] == pytest.approx(2.72443705e-4)


def test_the_shear_is_recovered_from_tglfs_q_prime():
    local = cgyro.cgyro_input_from_tglf(tglf_local())
    assert local.geometry["S"] == pytest.approx(0.92)
    assert local.geometry["SHIFT"] == pytest.approx(-0.12)


def test_collisionality_and_beta_are_the_same_numbers_tglf_was_given():
    tglf = tglf_local()
    local = cgyro.cgyro_input_from_tglf(tglf)
    assert local.nu_ee == tglf.xnue
    assert local.betae_unit == tglf.betae
    assert local.lambda_star == tglf.debye


def test_the_field_orientation_is_gacodes_ipccw_and_btccw():
    local = cgyro.cgyro_input_from_tglf(tglf_local(sign_bt=-1.0, sign_it=1.0))
    parameters = cgyro.cgyro_parameters(local)
    assert parameters["BTCCW"] == -1.0 and parameters["IPCCW"] == 1.0
    assert parameters["Q"] > 0.0


def test_z_eff_is_not_written_because_cgyro_recomputes_it():
    local = cgyro.cgyro_input_from_tglf(tglf_local())
    parameters = cgyro.cgyro_parameters(local)
    assert "Z_EFF" not in parameters
    assert local.z_eff == pytest.approx(0.8 + 36.0 / 30.0)
    assert local.provenance["z_eff"]["tglf_value"] == 2.0


def test_an_input_without_one_electron_species_is_refused():
    with pytest.raises(cgyro.LocalConversionError, match="one kinetic electron"):
        cgyro.cgyro_input_from_tglf(tglf_local(zs=np.array([1.0, 1.0, 6.0])))


def test_the_configuration_reaches_the_file(tmp_path):
    local = cgyro.cgyro_input_from_tglf(tglf_local())
    config = CGYROConfig(field_model="em-aperp", ky=0.6, n_theta=48,
                         extra_parameters={"e_max": 10})
    staged = cgyro.stage_cgyro_case(local, tmp_path / "case", config,
                                    state_key={"shot": 39915})
    text = staged.input_cgyro.read_text()
    for line in ("N_FIELD=2", "KY=0.6", "N_THETA=48", "E_MAX=10", "PROFILE_MODEL=1",
                 "N_SPECIES=3", "Z_3=-1.0"):
        assert line in text.splitlines()
    assert staged.provenance["state_key"] == {"shot": 39915}
    assert staged.provenance["input_sha256"] == cgyro.input_sha256(staged.input_cgyro)
    assert staged.provenance["resolution"]["n_theta"] == 48


def test_the_configuration_refuses_what_cgyro_cannot_run():
    with pytest.raises(ValueError, match="field_model"):
        CGYROConfig(field_model="em")
    with pytest.raises(ValueError, match="nonlinear"):
        CGYROConfig(nonlinear=True, n_toroidal=1)
    with pytest.raises(ValueError, match="ky"):
        CGYROConfig(ky=0.0)


def test_the_formalism_is_recorded_per_1353():
    record = cgyro.formalism(CGYROConfig(field_model="em-aperp", nonlinear=True, n_toroidal=8))
    assert record["spatial_domain"] == "local"
    assert record["distribution_formulation"] == "delta_f"
    assert record["field_model"] == "em-aperp"
    assert record["regime"] == "nonlinear"


# --------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------


def test_a_converged_linear_run_parses(tmp_path):
    run = collect_cgyro_outputs(write_run(tmp_path / "run"))
    assert run.grid["n_species"] == 3 and run.grid["ky"][0] == pytest.approx(0.3)
    assert run.final_growth_rate[0] == pytest.approx(0.25)
    assert run.final_frequency[0] == pytest.approx(0.4)
    assert run.equilibrium["b_gs2"] == pytest.approx(0.8)
    assert run.equilibrium["species"]["z"].tolist() == [1.0, 6.0, -1.0]
    assert run.converged and run.solved
    assert run.version["commit"] == "b493397"
    assert run.version["revision"] == "b493397 [2026-08-20]"
    assert run.ballooning["phi"].shape == (32,)


def test_the_ion_negative_frequency_follows_cgyros_own_statement(tmp_path):
    ion_positive = collect_cgyro_outputs(write_run(tmp_path / "a", ion_direction=">"))
    ion_negative = collect_cgyro_outputs(write_run(tmp_path / "b", ion_direction="<"))
    # omega = +0.4 is the ion direction in the first run, the electron one in the second
    assert ion_positive.frequency_ion_negative[0] == pytest.approx(-0.4)
    assert ion_negative.frequency_ion_negative[0] == pytest.approx(0.4)


def test_without_an_ion_direction_no_sign_is_invented(tmp_path):
    directory = write_run(tmp_path / "run")
    (directory / "out.cgyro.info").write_text("EXIT: (CGYRO) Linear converged\n")
    assert collect_cgyro_outputs(directory).frequency_ion_negative is None


def test_max_time_is_executed_but_not_qualified(tmp_path):
    run = collect_cgyro_outputs(write_run(tmp_path / "run",
                                          exit_message="Linear terminated at max time"))
    assert run.solved and not run.converged


def test_an_error_line_means_not_solved_whatever_else(tmp_path):
    run = collect_cgyro_outputs(write_run(tmp_path / "run", error="Only one electron species allowed"))
    assert run.errors and not run.solved


def test_no_exit_line_means_the_kernel_never_finished(tmp_path):
    run = collect_cgyro_outputs(write_run(tmp_path / "run", exit_message=None))
    assert run.exit_message is None and not run.solved


def test_a_truncated_binary_record_is_read_to_its_last_whole_step(tmp_path):
    directory = write_run(tmp_path / "run", steps=5)
    path = directory / "bin.cgyro.freq"
    path.write_bytes(path.read_bytes()[:-4])
    run = collect_cgyro_outputs(directory)
    assert run.frequency.shape == (1, 4)


def test_double_precision_is_taken_from_the_equilibrium_flag(tmp_path):
    run = collect_cgyro_outputs(write_run(tmp_path / "run", hiprec=1))
    assert run.final_growth_rate[0] == pytest.approx(0.25)


def test_nonlinear_fluxes_and_their_time_average(tmp_path):
    run = collect_cgyro_outputs(write_run(
        tmp_path / "run", n_n=4, n_field=2, flux=True, exit_message="Normal"))
    assert run.flux.shape == (3, 3, 2, 4, 5)
    average = run.time_average_flux((2.0, 5.0))
    assert average.shape == (3, 3)
    assert average[0, 1] == pytest.approx(2.0 * 2 * 4)  # energy: fields x modes summed
    assert run.time_average_flux((10.0, 11.0)) is None


def test_an_empty_directory_has_no_outputs(tmp_path):
    assert collect_cgyro_outputs(tmp_path) is None


def test_the_container_round_trips_through_json(tmp_path):
    run = collect_cgyro_outputs(write_run(tmp_path / "run"))
    back = CgyroOutputs.read_json(run.write_json(tmp_path / "out.json"))
    assert back.final_growth_rate[0] == pytest.approx(0.25)
    assert np.allclose(back.ballooning["phi"], run.ballooning["phi"])
    assert back.equilibrium["species"]["z"].tolist() == [1.0, 6.0, -1.0]
    assert back.converged and back.ion_direction == 1


def test_a_newer_schema_is_refused():
    with pytest.raises(ValueError, match="schema"):
        CgyroOutputs.from_dict({"schema": cgyro.SCHEMA, "schema_version": 99})


# --------------------------------------------------------------------------
# the runner, with the launcher faked
# --------------------------------------------------------------------------


@pytest.fixture
def fake_launcher(monkeypatch, tmp_path):
    from vaft.code.gacode import _runtime
    from vaft.code.gacode.cgyro import runner

    calls = {}

    def fake_run(executable, arguments, *, cwd, log_path, config=None, code="neo"):
        calls["arguments"] = list(arguments)
        calls["cwd"] = cwd
        workdir = Path(cwd) / arguments[1]
        assert not (workdir / "bin.cgyro.restart").exists() or config.restart
        write_run(workdir, **calls.get("write", {}))
        Path(log_path).write_text("ran\n")
        return _runtime.GACODERun(calls.get("returncode", 0), Path(log_path), "completed", 1.0)

    monkeypatch.setattr(runner, "run_gacode", fake_run)
    monkeypatch.setattr(runner, "require_gacode_executable", lambda c, code: Path("/x/cgyro"))
    monkeypatch.setattr(runner, "gacode_platform", lambda c: "TEST")
    monkeypatch.setattr(runner, "gacode_revision", lambda c: "b493397deadbeef")
    return calls


def test_run_passes_the_launcher_flags_cgyro_takes(fake_launcher, tmp_path):
    local = cgyro.cgyro_input_from_tglf(tglf_local())
    staged = cgyro.stage_cgyro_case(local, tmp_path / "case", CGYROConfig(n_mpi=8, n_omp=2))
    result = cgyro.run_cgyro(staged, CGYROConfig(n_mpi=8, n_omp=2), state_key={"shot": 1})
    assert fake_launcher["arguments"] == ["-e", "case", "-n", "8", "-nomp", "2"]
    assert result.executed and result.ok and result.qualified
    assert result.provenance["gacode_commit"] == "b493397deadbeef"
    assert result.provenance["formalism"]["solver_version"] == "b493397deadbeef"
    assert result.provenance["state_key"] == {"shot": 1}
    assert result.provenance["input_sha256"] == staged.provenance["input_sha256"]


def test_a_stale_restart_file_is_removed_unless_restarting(fake_launcher, tmp_path):
    local = cgyro.cgyro_input_from_tglf(tglf_local())
    staged = cgyro.stage_cgyro_case(local, tmp_path / "case")
    (tmp_path / "case" / "bin.cgyro.restart").write_bytes(b"old")
    cgyro.run_cgyro(staged)
    assert not (tmp_path / "case" / "bin.cgyro.restart").exists()


def test_a_restart_keeps_the_restart_file(fake_launcher, tmp_path):
    local = cgyro.cgyro_input_from_tglf(tglf_local())
    config = CGYROConfig(restart=True)
    staged = cgyro.stage_cgyro_case(local, tmp_path / "case", config)
    (tmp_path / "case" / "bin.cgyro.restart").write_bytes(b"old")
    cgyro.run_cgyro(staged, config)
    assert (tmp_path / "case" / "bin.cgyro.restart").exists()


def test_an_input_error_with_exit_zero_raises_and_says_why(fake_launcher, tmp_path):
    fake_launcher["write"] = {"error": "No electron species specified", "exit_message": None}
    local = cgyro.cgyro_input_from_tglf(tglf_local())
    staged = cgyro.stage_cgyro_case(local, tmp_path / "case")
    with pytest.raises(CGYROExecutionError, match="No electron species"):
        cgyro.run_cgyro(staged)
    result = cgyro.run_cgyro(staged, check=False)
    assert not result.executed and not result.ok


def test_an_unconverged_run_is_ok_but_not_qualified(fake_launcher, tmp_path):
    fake_launcher["write"] = {"exit_message": "Linear terminated at max time"}
    local = cgyro.cgyro_input_from_tglf(tglf_local())
    result = cgyro.run_cgyro(cgyro.stage_cgyro_case(local, tmp_path / "case"))
    assert result.ok and not result.qualified


def test_a_timeout_comes_back_as_a_result(monkeypatch, tmp_path):
    from vaft.code.gacode import _runtime
    from vaft.code.gacode.cgyro import runner

    def stopped(executable, arguments, *, cwd, log_path, config=None, code="neo"):
        Path(log_path).write_text("CGYRO stopped after 60 s (timeout)\n")
        return _runtime.GACODERun(None, Path(log_path), "timeout", 60.0)

    monkeypatch.setattr(runner, "run_gacode", stopped)
    monkeypatch.setattr(runner, "require_gacode_executable", lambda c, code: Path("/x/cgyro"))
    monkeypatch.setattr(runner, "gacode_platform", lambda c: "TEST")
    monkeypatch.setattr(runner, "gacode_revision", lambda c: None)
    local = cgyro.cgyro_input_from_tglf(tglf_local())
    result = cgyro.run_cgyro(cgyro.stage_cgyro_case(local, tmp_path / "case"), check=False)
    assert result.timed_out and not result.ok
    with pytest.raises(CGYROExecutionError, match="timeout"):
        cgyro.run_cgyro(cgyro.stage_cgyro_case(local, tmp_path / "case"))


def test_importing_the_suite_does_not_need_gacode():
    import vaft.code.gacode as gacode

    assert "cgyro" in gacode.SUPPORTED_CODES
    assert gacode.cgyro is cgyro
