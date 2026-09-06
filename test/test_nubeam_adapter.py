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


# --------------------------------------------------------------------------
# LOST_ORBIT decoding
# --------------------------------------------------------------------------

LOST_FIELDS = nubeam.LOST_PARTICLE_FIELDS


def _write_xplasma_out(path: Path, columns: dict[str, list[float]]) -> Path:
    """Write the three LOST_ORBIT variables the way xplasma packs them.

    Columns are laid end to end in ``r8work``; ``iwork`` carries the marker
    count and one 1-based start offset per column. The trailing zero after the
    offsets is xplasma's, and is what makes the reference reader drop a value.
    """
    import numpy as np
    import xarray as xr

    count = len(next(iter(columns.values())))
    flat: list[float] = []
    offsets: list[int] = []
    for name in LOST_FIELDS:
        offsets.append(len(flat) + 1)  # 1-based
        flat.extend(columns[name])
    iwork = [40, count, *offsets, 0]

    xr.Dataset(
        {
            "LOST_ORBIT_type": ((), np.int32(900)),
            "LOST_ORBIT_iwork": (("ni",), np.asarray(iwork, dtype="int32")),
            "LOST_ORBIT_r8work": (("nr",), np.asarray(flat, dtype=float)),
        }
    ).to_netcdf(path)
    return path


def _lost_columns(count: int = 3) -> dict[str, list[float]]:
    values = {
        "time": [1e-5, 2e-5, 3e-5],
        "beam": [1.0, 1.0, 1.0],
        "efrac": [1.0, 1.0, 2.0],
        "ptcl": [5.0e14, 4.0e14, 3.0e14],
        "rlost": [0.51, 0.60, 0.69],
        "zlost": [-0.01, 0.20, 0.44],
        "energy": [9.6, 9.8, 10.0],
        "vfrac": [0.2, 0.5, 0.9],
        "lstype": [1.0, 1.0, 2.0],
        "spec": [1.0, 1.0, 1.0],
    }
    return {k: v[:count] for k, v in values.items()}


def test_lost_particles_decode_every_column_to_full_length(tmp_path):
    """The reference reader truncates the last column; this must not."""
    _write_xplasma_out(tmp_path / "RUN_xplasma_out.cdf", _lost_columns())
    (tmp_path / "nubeam_comp_exec.RUNID").write_text("RUN\n", encoding="utf-8")

    result = nubeam.collect_nubeam_outputs(tmp_path)
    lost = result.outputs_native.lost

    assert lost is not None
    assert lost.count == 3
    assert sorted(lost.columns) == sorted(LOST_FIELDS)
    # `spec` is the column the reference reader returns one short.
    assert all(len(lost.columns[name]) == 3 for name in LOST_FIELDS)
    assert list(lost.columns["spec"]) == [1.0, 1.0, 1.0]


def test_lost_particles_keep_native_metres_and_kev(tmp_path):
    _write_xplasma_out(tmp_path / "RUN_xplasma_out.cdf", _lost_columns())
    (tmp_path / "nubeam_comp_exec.RUNID").write_text("RUN\n", encoding="utf-8")

    lost = nubeam.collect_nubeam_outputs(tmp_path).outputs_native.lost

    # Metres, unlike the birth file's centimetres. No conversion here.
    assert list(lost.columns["rlost"]) == [0.51, 0.60, 0.69]
    assert list(lost.columns["energy"]) == [9.6, 9.8, 10.0]


def test_lost_particles_split_prompt_from_orbit(tmp_path):
    _write_xplasma_out(tmp_path / "RUN_xplasma_out.cdf", _lost_columns())
    (tmp_path / "nubeam_comp_exec.RUNID").write_text("RUN\n", encoding="utf-8")

    lost = nubeam.collect_nubeam_outputs(tmp_path).outputs_native.lost

    assert lost.channel_counts() == {"prompt": 2, "orbit": 1}
    assert list(lost.prompt) == [True, True, False]


def test_a_run_that_lost_nothing_is_an_empty_record_not_a_missing_one(tmp_path):
    import numpy as np
    import xarray as xr

    xr.Dataset(
        {
            "LOST_ORBIT_iwork": (
                ("ni",),
                np.asarray([40, 0, *([1] * len(LOST_FIELDS)), 0], dtype="int32"),
            ),
            "LOST_ORBIT_r8work": (("nr",), np.zeros(1)),
        }
    ).to_netcdf(tmp_path / "RUN_xplasma_out.cdf")
    (tmp_path / "nubeam_comp_exec.RUNID").write_text("RUN\n", encoding="utf-8")

    lost = nubeam.collect_nubeam_outputs(tmp_path).outputs_native.lost

    assert lost is not None and lost.count == 0
    assert lost.channel_counts() == {"prompt": 0, "orbit": 0}


def test_an_xplasma_file_without_the_record_yields_none(tmp_path):
    import numpy as np
    import xarray as xr

    xr.Dataset({"something_else": (("n",), np.zeros(3))}).to_netcdf(
        tmp_path / "RUN_xplasma_out.cdf"
    )
    (tmp_path / "nubeam_comp_exec.RUNID").write_text("RUN\n", encoding="utf-8")

    assert nubeam.collect_nubeam_outputs(tmp_path).outputs_native.lost is None


def test_no_xplasma_file_is_not_an_error(tmp_path):
    assert nubeam.collect_nubeam_outputs(tmp_path).outputs_native.lost is None


# --------------------------------------------------------------------------
# Power balance, parsed from the step log
# --------------------------------------------------------------------------

STEP_LOG = """\
 ... rough power balance ...

 H beam ion:
    +injected power (W):   2.000D+05
    +OH -> beam ions:      0.000D+00
    -electron heating:     1.138D+05
    -ion heating:          5.713D+03
    -"internal" cx loss:   1.325D+02
    -shine-through:        3.852D+04
    -bad orbit loss:       2.309D+04
    -ripple loss:          0.000D+00
    -d/dt(f.i. energy):    1.970D+04
       ->residual:        -1.468D+03

 NUBEAM 2d output sampled at [R,Z] = [ 4.934D+01, 9.045D+00] cm
"""


def test_power_balance_reads_fortran_exponents():
    (balance,) = nubeam.parse_power_balance(STEP_LOG)
    assert balance.species == "H beam ion"
    # 2.000D+05, which Python will not parse as written.
    assert balance.injected == pytest.approx(2.0e5)


def test_power_balance_signs_sources_and_sinks_as_the_log_does():
    (balance,) = nubeam.parse_power_balance(STEP_LOG)
    assert balance.entries["injected power (W)"] > 0
    assert balance.entries["electron heating"] < 0
    assert balance.sinks()["electron heating"] == pytest.approx(1.138e5)


def test_power_balance_keeps_the_residual_separate_from_the_channels():
    """`->residual:` must not be read as a channel called '>residual'."""
    (balance,) = nubeam.parse_power_balance(STEP_LOG)
    assert balance.residual == pytest.approx(-1468.0)
    assert not any("residual" in name for name in balance.entries)


def test_power_balance_fractions_match_the_run():
    (balance,) = nubeam.parse_power_balance(STEP_LOG)
    fractions = balance.fractions()
    assert fractions["electron heating"] == pytest.approx(0.569, abs=5e-4)
    assert fractions["shine-through"] == pytest.approx(0.1926, abs=5e-4)
    assert fractions["bad orbit loss"] == pytest.approx(0.11545, abs=5e-4)


def test_power_balance_strips_the_quotes_nubeam_writes():
    (balance,) = nubeam.parse_power_balance(STEP_LOG)
    assert "internal cx loss" in balance.entries
    assert not any('"' in name for name in balance.entries)


def test_power_balance_stops_at_the_end_of_the_block():
    (balance,) = nubeam.parse_power_balance(STEP_LOG)
    assert not any("sampled at" in name for name in balance.entries)


def test_a_log_without_a_balance_yields_nothing():
    assert nubeam.parse_power_balance("nubeam STEP completed:  normal exit.\n") == ()


def test_profiles_exclude_plasma_state_bookkeeping(tmp_path):
    """`ps_partial_update` is a flag, not something a caller can plot."""
    import numpy as np
    import xarray as xr

    xr.Dataset(
        {
            "pbe": (("rho",), np.linspace(10.0, 1.0, 8)),
            "ps_partial_update": ((), np.int32(1)),
            "frac_full": (("one",), np.array([1.0])),
            "version_id": (("c",), np.array(["2.055"], dtype="U8")),
        }
    ).to_netcdf(tmp_path / "state_changes.cdf")

    profiles = nubeam.collect_nubeam_outputs(tmp_path).outputs_native.profiles

    assert "pbe" in profiles
    assert "ps_partial_update" not in profiles  # a scalar flag
    assert "frac_full" not in profiles  # one point is not a profile
    assert "version_id" not in profiles  # a character array
