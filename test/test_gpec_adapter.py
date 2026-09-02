"""Unit tests for how the GPEC-suite adapter locates a GPEC installation.

The adapter must be usable on a machine with no GPEC build: preparing a case is
pure file templating and has to work regardless, while running a module needs an
installation and must say so clearly instead of silently pointing at a path that
only exists on the author's laptop.
"""

from pathlib import Path

import pytest

from vaft.code import gpec


GFILE_TEXT = "  EFITD   01/01/2024   #  39915  325ms        3  65  65\n 1.0 2.0 3.0\n"


@pytest.fixture()
def no_gpec_env(monkeypatch):
    monkeypatch.delenv(gpec.GPEC_HOME_ENV, raising=False)


@pytest.fixture()
def case(tmp_path):
    geqdsk = tmp_path / "g039915.00325"
    geqdsk.write_text(GFILE_TEXT)
    return gpec.GPECCaseInputs(
        shot=39915,
        time_ms=325,
        geqdsk=geqdsk,
        workdir=tmp_path / "run",
    )


def test_gpec_home_falls_back_to_environment(monkeypatch, tmp_path):
    executable = tmp_path / "gpec/bin/dcon"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    monkeypatch.setenv(gpec.GPEC_HOME_ENV, str(tmp_path / "gpec"))
    config = gpec.GPECSuiteConfig()

    assert gpec._gpec_home(config) == tmp_path / "gpec"
    assert gpec._executable(config, "dcon") == executable


def test_config_gpec_home_overrides_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(gpec.GPEC_HOME_ENV, str(tmp_path / "from_env"))
    config = gpec.GPECSuiteConfig(gpec_home=tmp_path / "from_config")

    assert gpec._gpec_home(config) == tmp_path / "from_config"


def test_executable_dir_overrides_gpec_home(no_gpec_env, tmp_path):
    config = gpec.GPECSuiteConfig(
        gpec_home=tmp_path / "gpec",
        executable_dir=tmp_path / "custom_bin",
    )

    assert gpec._executable(config, "dcon") == tmp_path / "custom_bin" / "dcon"


def test_unconfigured_installation_resolves_to_nothing(no_gpec_env):
    config = gpec.GPECSuiteConfig()

    assert gpec._gpec_home(config) is None
    assert gpec._executable_dir(config) is None
    assert gpec._executable(config, "dcon") is None


def test_gpec_env_omits_gpechome_when_unconfigured(no_gpec_env):
    config = gpec.GPECSuiteConfig()

    assert gpec.GPEC_HOME_ENV not in gpec._gpec_env(config)


def test_prepare_works_without_a_gpec_installation(no_gpec_env, case):
    result = gpec.prepare_gpec_suite_case(
        case,
        gpec.GPECSuiteConfig(modules=("dcon",), modes=(1,)),
    )

    assert result.ok
    run_dir = case.workdir / "00325" / "dcon" / "nn=1"
    for name in ("equil.in", "vac.in", "dcon.in", "match.in", case.geqdsk.name):
        assert (run_dir / name).exists(), f"{name} was not materialized"

    # nn is templated per mode, and the site-specific default is gone from coil.in.
    assert "nn=1" in (run_dir / "dcon.in").read_text()
    packaged_coil = Path(gpec.__file__).resolve().parents[2] / "data" / "gpec" / "coil.in"
    assert 'data_dir=""' in packaged_coil.read_text()


def test_run_without_installation_is_skipped_with_a_clear_reason(no_gpec_env, case):
    result = gpec.run_gpec_suite_case(
        case,
        gpec.GPECSuiteConfig(modules=("dcon",), modes=(1,), run_mode="auto"),
    )

    assert result.ok  # a skip is not a failure
    (record,) = result.records
    assert record.status == "skipped"
    assert gpec.GPEC_HOME_ENV in record.reason


def test_run_without_installation_raises_in_strict_mode(no_gpec_env, case):
    with pytest.raises(FileNotFoundError, match=gpec.GPEC_HOME_ENV):
        gpec.run_gpec_suite_case(
            case,
            gpec.GPECSuiteConfig(modules=("dcon",), modes=(1,), run_mode="strict"),
        )


def test_run_if_available_keeps_successful_dcon_when_optional_match_is_missing(
    monkeypatch,
    tmp_path,
    case,
):
    dcon = tmp_path / "gpec/bin/dcon"
    dcon.parent.mkdir(parents=True)
    dcon.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    dcon.chmod(0o755)
    monkeypatch.setenv(gpec.GPEC_HOME_ENV, str(tmp_path / "gpec"))

    result = gpec.run_gpec_suite_case(
        case,
        gpec.GPECSuiteConfig(modules=("dcon",), modes=(1,), run_mode="auto"),
    )

    (record,) = result.records
    assert record.status == "completed"
    assert record.returncode == 0
    assert record.commands == (str(dcon),)


def test_run_reuses_a_completed_solver_cell_without_rerunning(monkeypatch, tmp_path, case):
    dcon = tmp_path / "gpec/bin/dcon"
    dcon.parent.mkdir(parents=True)
    dcon.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    dcon.chmod(0o755)
    monkeypatch.setenv(gpec.GPEC_HOME_ENV, str(tmp_path / "gpec"))

    run_dir = case.workdir / "00325" / "dcon" / "nn=1"
    run_dir.mkdir(parents=True)
    for filename in gpec._solvers.SOLVERS["dcon"].output_patterns(1):
        (run_dir / filename).write_text("existing", encoding="utf-8")

    result = gpec.run_gpec_suite_case(
        case,
        gpec.GPECSuiteConfig(modules=("dcon",), modes=(1,), run_mode="auto"),
    )

    (record,) = result.records
    assert record.status == "completed"
    assert record.reason == "reused existing solver outputs"


def test_prepare_writes_rdcon_and_rmatch_without_a_gpec_installation(no_gpec_env, case):
    result = gpec.prepare_gpec_suite_case(
        case,
        gpec.GPECSuiteConfig(modules=("rdcon",), modes=(1,)),
    )

    assert result.ok
    run_dir = case.workdir / "00325" / "rdcon" / "nn=1"
    for name in ("equil.in", "vac.in", "rdcon.in", "rmatch.in", case.geqdsk.name):
        assert (run_dir / name).exists(), f"{name} was not materialized"
    assert "nn=1" in (run_dir / "rdcon.in").read_text()
    # rmatch.in is copied verbatim (no `nn` key) and must not reference DCON's
    # directory -- it reads vmat.bin from its own run directory.
    assert 'vmat_filename="vmat.bin"' in (run_dir / "rmatch.in").read_text()


def test_prepare_writes_stride_without_a_gpec_installation(no_gpec_env, case):
    result = gpec.prepare_gpec_suite_case(
        case,
        gpec.GPECSuiteConfig(modules=("stride",), modes=(2,)),
    )

    assert result.ok
    run_dir = case.workdir / "00325" / "stride" / "nn=2"
    for name in ("equil.in", "vac.in", "stride.in", case.geqdsk.name):
        assert (run_dir / name).exists(), f"{name} was not materialized"
    assert "nn=2" in (run_dir / "stride.in").read_text()


def test_prepare_ideal_gpec_uses_a_separate_dcon_work_tree(no_gpec_env, tmp_path, case):
    """FileDB stores ideal-GPEC and DCON under distinct code/mode roots."""
    coil = tmp_path / "coil.in"
    coil.write_text("&coil /\n", encoding="utf-8")
    case.dcon_workdir = tmp_path / "dcon-work"
    case.coil_in = coil

    result = gpec.prepare_gpec_suite_case(
        case,
        gpec.GPECSuiteConfig(modules=("gpec",), modes=(1,)),
    )

    assert result.ok
    gpec_in = case.workdir / "00325" / "gpec" / "nn=1" / "gpec.in"
    expected_dcon = case.dcon_workdir / "00325" / "dcon" / "nn=1"
    assert str(expected_dcon.resolve()) in gpec_in.read_text(encoding="utf-8")


def test_run_if_available_chains_rmatch_after_a_successful_rdcon(monkeypatch, tmp_path, case):
    rdcon = tmp_path / "gpec/bin/rdcon"
    rmatch = tmp_path / "gpec/bin/rmatch"
    rdcon.parent.mkdir(parents=True)
    rdcon.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    rdcon.chmod(0o755)
    rmatch.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    rmatch.chmod(0o755)
    monkeypatch.setenv(gpec.GPEC_HOME_ENV, str(tmp_path / "gpec"))

    result = gpec.run_gpec_suite_case(
        case,
        gpec.GPECSuiteConfig(modules=("rdcon",), modes=(1,), run_mode="auto"),
    )

    (record,) = result.records
    assert record.status == "completed"
    assert record.commands == (str(rdcon), str(rmatch))


def test_run_if_available_fails_when_rmatch_exits_nonzero(monkeypatch, tmp_path, case):
    rdcon = tmp_path / "gpec/bin/rdcon"
    rmatch = tmp_path / "gpec/bin/rmatch"
    rdcon.parent.mkdir(parents=True)
    rdcon.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    rdcon.chmod(0o755)
    rmatch.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    rmatch.chmod(0o755)
    monkeypatch.setenv(gpec.GPEC_HOME_ENV, str(tmp_path / "gpec"))

    result = gpec.run_gpec_suite_case(
        case,
        gpec.GPECSuiteConfig(modules=("rdcon",), modes=(1,), run_mode="auto"),
    )

    (record,) = result.records
    assert record.status == "failed"
    assert record.returncode == 1


def test_run_if_available_skips_stride_cleanly_when_missing(no_gpec_env, case):
    result = gpec.run_gpec_suite_case(
        case,
        gpec.GPECSuiteConfig(modules=("stride",), modes=(1,), run_mode="auto"),
    )

    (record,) = result.records
    assert result.ok  # a skip is not a failure
    assert record.status == "skipped"


def test_verify_outputs_fails_a_completed_run_missing_the_expected_variable(monkeypatch, tmp_path, case):
    """``verify_outputs`` catches a solver that exits 0 without writing real physics content."""
    dcon = tmp_path / "gpec/bin/dcon"
    dcon.parent.mkdir(parents=True)
    dcon.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    dcon.chmod(0o755)
    monkeypatch.setenv(gpec.GPEC_HOME_ENV, str(tmp_path / "gpec"))

    result = gpec.run_gpec_suite_case(
        case,
        gpec.GPECSuiteConfig(modules=("dcon",), modes=(1,), run_mode="auto", verify_outputs=True),
    )

    (record,) = result.records
    assert record.status == "failed"
    assert "dcon_output_n1.nc" in record.reason


def test_solver_companion_executables_takes_no_config_argument():
    """`Solver.companion_executables` no longer accepts a `config` -- none of
    the four solvers ever consulted it, so it was dead protocol surface."""
    from vaft.code.gpec._solvers import SOLVERS

    for solver in SOLVERS.values():
        solver.companion_executables()  # must not require a config argument
        with pytest.raises(TypeError):
            solver.companion_executables(gpec.GPECSuiteConfig())


def test_runtime_module_dir_resolves_from_workdir_and_time_ms_without_a_geqdsk(tmp_path):
    """`_runtime.module_dir` no longer needs a `GPECCaseInputs` (and therefore no
    placeholder GEQDSK path) when `time_ms` is already known -- callers like
    `build_mhd_linear_ods` that only have a run manifest's time label can call
    it directly."""
    from vaft.code.gpec import _runtime as rt

    run_dir = rt.module_dir(tmp_path, 325, "dcon", 1)

    assert run_dir == tmp_path / "00325" / "dcon" / "nn=1"


def test_runtime_module_dir_falls_back_to_geqdsk_only_when_time_ms_is_none(tmp_path):
    from vaft.code.gpec import _runtime as rt

    geqdsk = tmp_path / "g039915.00325"

    run_dir = rt.module_dir(tmp_path, None, "dcon", 1, geqdsk=geqdsk)

    assert run_dir == tmp_path / "00325" / "dcon" / "nn=1"


def test_timeout_with_truncated_outputs_is_not_reported_as_success(monkeypatch, tmp_path, case):
    """A GPEC run killed mid-write must not be archived as a completed run.

    GPEC is known to hang after its outputs materialize, so a timeout with
    the three core netCDFs present is treated as success. But a process
    killed *while* writing leaves those files present and truncated, so the
    carve-out has to apply the same ``verify_outputs`` check the normal
    completion path applies (release review, 0.6.0).
    """
    import subprocess

    executable = tmp_path / "gpec/bin/gpec"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv(gpec.GPEC_HOME_ENV, str(tmp_path / "gpec"))

    # verify_outputs stays at its shipped default (False): a timed-out run
    # must not be called successful regardless of that flag.
    config = gpec.GPECSuiteConfig(modules=("gpec",), modes=(1,), run_mode="auto")
    gpec.prepare_gpec_suite_case(case, config)
    run_dir = gpec._module_dir(case.workdir, case.time_ms, "gpec", 1, geqdsk=case.geqdsk)

    # ideal-GPEC consumes a completed same-mode DCON result; give it one so
    # the run reaches the executable instead of skipping on the DCON gate.
    dcon_dir = gpec._module_dir(case.workdir, case.time_ms, "dcon", 1, geqdsk=case.geqdsk)
    dcon_dir.mkdir(parents=True, exist_ok=True)
    (dcon_dir / "euler.bin").write_bytes(b"")
    (dcon_dir / "psi_in.bin").write_bytes(b"")
    _write_valid_dcon_netcdf(dcon_dir, n=1)

    # The three core outputs exist but hold no readable physics content.
    for name in (
        "gpec_control_output_n1.nc",
        "gpec_profile_output_n1.nc",
        "gpec_cylindrical_output_n1.nc",
    ):
        (run_dir / name).write_bytes(b"truncated")

    def _timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="gpec", timeout=1)

    monkeypatch.setattr(gpec.rt, "run_subprocess", _timeout)

    result = gpec.run_gpec_suite_case(case, config)
    (record,) = result.records
    assert record.status == "failed"
    assert record.returncode is None
    assert "failed verification" in record.reason


def _write_valid_dcon_netcdf(path, *, n):
    import xarray as xr

    xr.Dataset(
        {"W_t_eigenvalue": (("mode", "i"), [[-0.3, 0.0]])},
        coords={"i": [0, 1], "mode": [1]},
        attrs={"mlow": -2, "mhigh": 0, "mpert": 3, "mband": 0, "n": n},
    ).to_netcdf(path / f"dcon_output_n{n}.nc")


def test_truncated_netcdf_does_not_pass_the_output_check(tmp_path):
    """Header-only checking let a truncated classic netCDF read as success.

    xarray opens lazily and a truncated classic file returns silent zeros
    for its missing tail, so the check must force a read and hold the file
    to the size its own header declares.
    """
    import numpy as np
    import xarray as xr

    from vaft.code.gpec._solvers import _check_nc_variable

    name = "dcon_output_n1.nc"

    def _write(fmt):
        xr.Dataset(
            {"W_t_eigenvalue": (("m",), np.arange(20000, dtype=float))}
        ).to_netcdf(tmp_path / name, format=fmt)

    for fmt in ("NETCDF3_CLASSIC", "NETCDF4"):
        _write(fmt)
        assert _check_nc_variable(tmp_path, name, "W_t_eigenvalue")[0], fmt
        full = (tmp_path / name).stat().st_size

        for fraction in (0.3, 0.95):
            _write(fmt)
            with open(tmp_path / name, "r+b") as handle:
                handle.truncate(int(full * fraction))
            ok, reason = _check_nc_variable(tmp_path, name, "W_t_eigenvalue")
            assert not ok, f"{fmt} truncated to {fraction:.0%} passed the check"
            assert reason
