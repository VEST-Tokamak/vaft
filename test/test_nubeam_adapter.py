"""NUBEAM adapter: path budget, inputf rewriting, staging and collection.

Every test here runs without NUBEAM installed. The one integration test skips
unless ``$NUBEAMHOME`` is set.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from vaft.code import nubeam
from vaft.compat import short_temporary_directory
from vaft.code.nubeam.config import (
    NUBEAM_LONGEST_OUTPUT_SUFFIX,
    NUBEAM_PATH_BUFFER_CHARS,
)
from vaft.code.nubeam.inputs import _apply_particle_count


@pytest.fixture(autouse=True)
def clear_nubeam_home(monkeypatch):
    monkeypatch.delenv("NUBEAMHOME", raising=False)


def test_import_smoke_without_external_runner():
    assert nubeam.NUBEAMConfig().runid == "NUBEAM"
    assert nubeam.find_nubeam_executable() is None


def test_explicit_executable_that_does_not_exist_is_reported(tmp_path):
    config = nubeam.NUBEAMConfig(executable=str(tmp_path / "absent"))
    with pytest.raises(FileNotFoundError):
        nubeam.find_nubeam_executable(config)


# --------------------------------------------------------------------------
# The fixed-width path budget
# --------------------------------------------------------------------------


def test_budget_leaves_room_for_the_longest_filename():
    """The budget is exactly what ``character*140 zfile`` can still hold."""
    runid = "FUSMA_NUBEAM"
    budget = nubeam.workdir_budget(runid)
    longest = budget + 1 + len(runid) + len(NUBEAM_LONGEST_OUTPUT_SUFFIX)
    assert longest == NUBEAM_PATH_BUFFER_CHARS


def test_budget_shrinks_as_the_run_id_grows():
    short = nubeam.workdir_budget("ab")
    long = nubeam.workdir_budget("ab" + "c" * 10)
    assert short - long == 10


def test_workdir_within_budget_is_accepted():
    config = nubeam.NUBEAMConfig(runid="RUN")
    nubeam.check_workdir_length(Path("/tmp/nubeam"), config)


def test_overlong_workdir_names_the_real_cause():
    config = nubeam.NUBEAMConfig(runid="RUN")
    workdir = Path("/tmp") / ("x" * config.workdir_budget)

    with pytest.raises(nubeam.NUBEAMInputError) as error:
        nubeam.check_workdir_length(workdir, config)

    message = str(error.value)
    # The failure NUBEAM itself produces names the input state, not the path,
    # so this message has to say what actually went wrong.
    assert "140" in message
    assert "truncated" in message
    assert str(workdir) in message


# --------------------------------------------------------------------------
# inputf rewriting -- the GNU/BSD sed incompatibility, done in Python
# --------------------------------------------------------------------------

INPUTF = (
    "0.0 0.005\t\t\t\t\t! init/end timeq\n"
    "chease_g026537.031000\t\t\t! EQDSK file\n"
    "mdescr_VEST_190307.dat\t\t! mdeescr file\n"
    "sconfig_VEST_190307.dat\t\t\t! sconfig file\n"
    "FUSMA_NUBEAM.cdf\t\t! final plasma state file\n"
    "FUSMA_NUBEAM\t! runID\n"
    "1\t\t\t\t! fprofil\n"
    "profiles\t\t\t\t! profile file:\n"
)


def test_only_the_equilibrium_line_changes():
    result = nubeam.rewrite_inputf_equilibrium(INPUTF, "equilibrium.gfile")

    before = INPUTF.splitlines()
    after = result.splitlines()
    assert len(before) == len(after)
    assert "equilibrium.gfile" in after[1]
    assert "chease_g026537.031000" not in result
    for index, (original, rewritten) in enumerate(zip(before, after)):
        if index != 1:
            assert original == rewritten


@pytest.mark.parametrize("terminator", ["\n", "\r\n"])
def test_line_terminators_are_preserved(terminator):
    text = INPUTF.replace("\n", terminator)
    result = nubeam.rewrite_inputf_equilibrium(text, "equilibrium.gfile")
    assert result.count(terminator) == text.count(terminator)
    if terminator == "\r\n":
        assert "\r\n" in result.splitlines(keepends=True)[1]


def test_a_missing_final_newline_is_not_invented():
    text = INPUTF.rstrip("\n")
    result = nubeam.rewrite_inputf_equilibrium(text, "equilibrium.gfile")
    assert not result.endswith("\n")


def test_a_truncated_inputf_is_rejected():
    with pytest.raises(nubeam.NUBEAMInputError):
        nubeam.rewrite_inputf_equilibrium("0.0 0.005\n", "equilibrium.gfile")


def test_positional_fields_are_read_back():
    assert nubeam.inputf_state_filename(INPUTF) == "FUSMA_NUBEAM.cdf"
    assert nubeam.inputf_runid(INPUTF) == "FUSMA_NUBEAM"


# --------------------------------------------------------------------------
# Namelist particle-count override
# --------------------------------------------------------------------------

NAMELIST = (
    " &NBI_INIT\n"
    "  nseed = 1094088621\n"
    "  nptcls = 20000\n"
    "      !! number of Monte Carlo particles per beam ion specie\n"
    "  nptclf = 20000\n"
    "  ndep0 = 4000\n"
    " /\n"
)


def test_particle_count_override_keeps_the_documentation(tmp_path):
    namelist = tmp_path / "nubeam_init.dat"
    namelist.write_text(NAMELIST, encoding="utf-8")

    _apply_particle_count(namelist, 1000)
    updated = namelist.read_text(encoding="utf-8")

    assert "nptcls = 1000" in updated
    assert "nptclf = 1000" in updated
    # ndep0 is a separate knob and must not be swept along.
    assert "ndep0 = 4000" in updated
    # The shipped namelists document every knob inline; a namelist writer
    # round-trip would discard all of it.
    assert "!! number of Monte Carlo particles" in updated


def test_particle_count_below_the_nubeam_minimum_is_rejected(tmp_path):
    namelist = tmp_path / "nubeam_init.dat"
    namelist.write_text(NAMELIST, encoding="utf-8")
    with pytest.raises(nubeam.NUBEAMInputError):
        _apply_particle_count(namelist, 99)


# --------------------------------------------------------------------------
# Staging
# --------------------------------------------------------------------------


def _case_directory(root: Path) -> Path:
    source = root / "case"
    source.mkdir()
    (source / "inputf").write_text(INPUTF, encoding="utf-8")
    (source / "profiles").write_text("101\n", encoding="utf-8")
    for name in (
        "nubeam_init.dat",
        "nubeam_step.dat",
    ):
        (source / name).write_text(NAMELIST, encoding="utf-8")
    for name in ("nubeam_init_files.dat", "nubeam_step_files.dat"):
        (source / name).write_text(" &NUBEAM_FILES\n /\n", encoding="utf-8")
    (source / "mdescr_VEST_190307.dat").write_text(" &mdescr\n /\n", encoding="utf-8")
    (source / "sconfig_VEST_190307.dat").write_text(" &sconfig\n /\n", encoding="utf-8")
    return source


def test_staging_rewrites_inputf_and_reports_the_state(tmp_path):
    source = _case_directory(tmp_path)
    gfile = tmp_path / "g020000.015100"
    gfile.write_text("EQDSK\n", encoding="utf-8")
    config = nubeam.NUBEAMConfig(runid="FUSMA_NUBEAM")

    # Not tmp_path: pytest nests it deeply enough (117 characters on macOS)
    # that NUBEAM's 140-character buffer would not survive the run. This is
    # the pattern callers are meant to use for exactly that reason.
    with short_temporary_directory(max_length=config.workdir_budget) as scratch:
        run = scratch / "run"
        inputs = nubeam.prepare_nubeam_inputs(
            source, gfile=gfile, workdir=run, config=config
        )

        assert inputs.runid == "FUSMA_NUBEAM"
        assert inputs.plasma_state == run / "FUSMA_NUBEAM.cdf"
        assert (run / "equilibrium.gfile").read_text() == "EQDSK\n"
        staged = (run / "inputf").read_text(encoding="utf-8")
        assert "equilibrium.gfile" in staged.splitlines()[1]
        assert (run / "mdescr_VEST_190307.dat").is_file()


def test_a_deep_pytest_tmp_path_is_refused_rather_than_truncated(tmp_path):
    """The guard fires on a realistically deep path, not only a synthetic one."""
    source = _case_directory(tmp_path)
    gfile = tmp_path / "g"
    gfile.write_text("EQDSK\n", encoding="utf-8")
    config = nubeam.NUBEAMConfig(runid="FUSMA_NUBEAM")

    deep = tmp_path / ("nested" * 20) / "run"
    with pytest.raises(nubeam.NUBEAMInputError):
        nubeam.prepare_nubeam_inputs(
            source, gfile=gfile, workdir=deep, config=config
        )
    # Nothing was created before the refusal.
    assert not deep.exists()


def test_staging_reports_a_missing_input(tmp_path):
    source = _case_directory(tmp_path)
    (source / "profiles").unlink()
    gfile = tmp_path / "g"
    gfile.write_text("EQDSK\n", encoding="utf-8")

    with pytest.raises(nubeam.NUBEAMInputError) as error:
        nubeam.prepare_nubeam_inputs(source, gfile=gfile, workdir=tmp_path / "run")
    assert "profiles" in str(error.value)


def test_staging_refuses_a_machine_description_free_directory(tmp_path):
    source = _case_directory(tmp_path)
    (source / "mdescr_VEST_190307.dat").unlink()
    (source / "sconfig_VEST_190307.dat").unlink()
    gfile = tmp_path / "g"
    gfile.write_text("EQDSK\n", encoding="utf-8")

    with pytest.raises(nubeam.NUBEAMInputError) as error:
        nubeam.prepare_nubeam_inputs(source, gfile=gfile, workdir=tmp_path / "run")
    assert "mdescr" in str(error.value)


# --------------------------------------------------------------------------
# Collection
# --------------------------------------------------------------------------


def test_collecting_an_empty_directory_is_not_an_error(tmp_path):
    result = nubeam.collect_nubeam_outputs(tmp_path)
    assert result.outputs_native is not None
    assert result.outputs_native.profiles == {}
    assert result.outputs_native.birth is None


def test_collecting_counts_interpolation_warnings(tmp_path):
    (tmp_path / "step.log").write_text(
        "noise\n"
        + "  ?xpprof: x arguments for interpolation are out of bounds.\n" * 3,
        encoding="utf-8",
    )
    result = nubeam.collect_nubeam_outputs(tmp_path)
    assert result.outputs_native.interpolation_warnings == 3


def test_collecting_a_missing_directory_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        nubeam.collect_nubeam_outputs(tmp_path / "absent")


# --------------------------------------------------------------------------
# Integration
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("NUBEAMHOME"),
    reason="NUBEAM integration test requires NUBEAMHOME",
)
def test_installed_nubeam_resolves():
    # The autouse fixture clears NUBEAMHOME, so read it from the real
    # environment the skipif consulted.
    executable = nubeam.find_nubeam_executable(
        nubeam.NUBEAMConfig(
            executable=str(
                Path(os.environ["NUBEAMHOME"]) / "bin" / "nubeam_comp_exec"
            )
        )
    )
    assert executable is not None and executable.is_file()
