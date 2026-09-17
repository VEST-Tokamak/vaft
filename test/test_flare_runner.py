"""Driving the FLARE executable.

FLARE is wrapped rather than imported: it ships an f2py extension beside the
library, but that binds a caller to one FLARE build and one Python, where the
shell driver needs only ``$FLAREHOME``. These tests cover the wrapper's own
behaviour with a stub driver, and run the real one when this machine has it.
"""

from __future__ import annotations

import os
import stat

import pytest

from vaft.compat import IS_WINDOWS
from vaft.code.flare import (
    FLARE_TASKS,
    FlareConfig,
    FlareResult,
    flare_executable,
    run_flare,
)


@pytest.fixture
def flare_driver(tmp_path):
    """A platform-native driver that echoes how it was called."""
    binary = tmp_path / "bin"
    binary.mkdir()
    driver = binary / "flare"
    if IS_WINDOWS:
        driver = driver.with_name(driver.name + ".cmd")
        driver.write_text(
            "@echo off\r\n"
            "echo args: %*\r\n"
            "echo cwd: %CD%\r\n"
            "if defined FLARE_TEST_MARKER (echo marker: %FLARE_TEST_MARKER%) else (echo marker: unset)\r\n"
            "for %%a in (%*) do if \"%%~a\"==\"boom\" (echo failed 1>&2 & exit /b 3)\r\n"
            "exit /b 0\r\n",
            encoding="ascii",
        )
    else:
        driver.write_text(
            "#!/usr/bin/env bash\n"
            'echo "args: $@"\n'
            'echo "cwd: $PWD"\n'
            'echo "marker: ${FLARE_TEST_MARKER:-unset}"\n'
            'for a in "$@"; do [ "$a" == "boom" ] && { echo "failed" >&2; exit 3; }; done\n'
            "exit 0\n"
        )
        driver.chmod(driver.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return driver


@pytest.fixture
def flare_home(flare_driver):
    return flare_driver.parent.parent


# --------------------------------------------------------------------------
# Locating the driver
# --------------------------------------------------------------------------


def test_the_driver_is_found_under_the_installation_root(flare_home, flare_driver):
    assert flare_executable(flare_home) == flare_driver


def test_an_unconfigured_home_says_which_variable_to_set(monkeypatch):
    monkeypatch.delenv("FLAREHOME", raising=False)
    with pytest.raises(RuntimeError, match=r"\$FLAREHOME"):
        flare_executable()


def test_the_environment_supplies_the_root_when_the_argument_does_not(
    flare_home, flare_driver, monkeypatch
):
    monkeypatch.setenv("FLAREHOME", str(flare_home))
    assert flare_executable() == flare_driver


def test_a_build_tree_is_refused_rather_than_run(tmp_path):
    """CMake leaves `build/flare` without its executable bit; only
    `cmake --install` produces the `bin/flare` this resolves. Pointing
    $FLAREHOME at a build tree must not silently find something else."""
    (tmp_path / "flare").write_text("#!/usr/bin/env bash\nexit 0\n")  # no bin/, no +x
    # Either separator: the message names the resolved path, which is
    # `bin\flare` on Windows and `bin/flare` everywhere else.
    with pytest.raises(FileNotFoundError, match=r"bin[\\/]flare"):
        flare_executable(tmp_path)


def test_a_driver_without_its_executable_bit_is_refused(tmp_path):
    binary = tmp_path / "bin"
    binary.mkdir()
    (binary / "flare").write_text("#!/usr/bin/env bash\nexit 0\n")
    with pytest.raises(PermissionError):
        flare_executable(tmp_path)


# --------------------------------------------------------------------------
# Running it
# --------------------------------------------------------------------------


def test_the_task_reaches_the_driver_and_the_result_records_the_command(
    flare_home, flare_driver, tmp_path
):
    workdir = tmp_path / "case"
    workdir.mkdir()
    result = run_flare("poincare_plot", FlareConfig(workdir=workdir), home=flare_home)
    assert isinstance(result, FlareResult)
    assert result.ok and result.returncode == 0
    assert result.task == "poincare_plot"
    assert "args: poincare_plot" in result.stdout
    assert result.command[0] == str(flare_driver)
    assert result.command[-1] == "poincare_plot"


def test_the_working_directory_is_what_selects_a_case(flare_home, tmp_path):
    """FLARE reads its control file from the working directory, not from an
    argument, so a wrapper that ignored workdir would silently run whichever
    case the process happened to be sitting in."""
    workdir = tmp_path / "case"
    workdir.mkdir()
    result = run_flare("run", FlareConfig(workdir=workdir), home=flare_home)
    assert f"cwd: {workdir}" in result.stdout
    assert result.workdir == workdir


def test_the_process_count_is_passed_as_flares_own_flag(flare_home):
    workers = 3
    result = run_flare("run", FlareConfig(processes=workers), home=flare_home)
    assert f"args: -n {workers} run" in result.stdout
    # Unset, the flag does not appear at all. Read the args line rather than
    # the whole stream: the stub also echoes the working directory, and a path
    # is free to carry the flag as a substring.
    plain = run_flare("run", home=flare_home).stdout
    args_line = next(line for line in plain.splitlines() if line.startswith("args:"))
    assert args_line == "args: run"


@pytest.mark.parametrize("bad", [0, -1])
def test_a_non_positive_process_count_is_refused(flare_home, bad):
    with pytest.raises(ValueError, match="at least 1"):
        run_flare("run", FlareConfig(processes=bad), home=flare_home)


def test_extra_arguments_follow_the_task(flare_home):
    result = run_flare("geqdsk", home=flare_home, extra_args=("g000001.00100", "plot"))
    assert "args: geqdsk g000001.00100 plot" in result.stdout


def test_the_environment_overlay_reaches_the_driver(flare_home):
    result = run_flare("run", FlareConfig(env={"FLARE_TEST_MARKER": "set"}), home=flare_home)
    assert "marker: set" in result.stdout
    assert "marker: unset" in run_flare("run", home=flare_home).stdout


def test_a_failing_run_keeps_its_streams_and_is_not_ok(flare_home):
    result = run_flare("run", FlareConfig(args=("boom",)), home=flare_home)
    assert result.returncode == 3
    assert not result.ok
    assert "failed" in result.stderr


@pytest.mark.parametrize("bad", ["not_a_task", "--help", "", "RUN"])
def test_an_unknown_task_is_refused_with_the_list(flare_home, bad):
    with pytest.raises(ValueError, match="is not a FLARE subcommand"):
        run_flare(bad, home=flare_home)


def test_the_task_list_is_flares_own(flare_home):
    assert "run" in FLARE_TASKS and "poincare_plot" in FLARE_TASKS
    assert len(set(FLARE_TASKS)) == len(FLARE_TASKS)


# --------------------------------------------------------------------------
# The real driver, when this machine has one
# --------------------------------------------------------------------------


def _installed_flare():
    home = os.environ.get("FLAREHOME")
    if not home:
        return None
    try:
        return flare_executable(home)
    except (RuntimeError, FileNotFoundError, PermissionError):
        return None


@pytest.mark.skipif(_installed_flare() is None, reason="no FLARE installation configured")
def test_the_installed_driver_answers(tmp_path):
    """Measured on a real installation: an unknown task exits 1, but **no
    arguments at all exits 0** after printing usage. A caller whose command
    line lost its task -- an empty variable -- would read that as success."""
    import subprocess

    executable = str(_installed_flare())
    assert subprocess.run([executable], cwd=tmp_path, capture_output=True).returncode == 0
    assert subprocess.run([executable, "bogus"], cwd=tmp_path, capture_output=True).returncode == 1
    # The wrapper never lets the empty case through.
    with pytest.raises(ValueError):
        run_flare("")
