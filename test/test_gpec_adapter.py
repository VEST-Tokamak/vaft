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
