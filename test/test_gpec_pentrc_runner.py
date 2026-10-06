"""Running PENTRC, and the one convention that decides whether the answer means anything.

``vaft.code.pentrc`` reads what PENTRC produced.  This is the half that produces
it, and the thing worth testing is not the subprocess call -- it is the
``pentrc.in`` that goes beside the result:

- ``jac_in`` has to be the Jacobian the displacement file was decomposed in.
  The file states it in its own header; the legacy runner wrote ``"hamada"``
  from a config file, which is right only while two *other* namelists agree.
- ``peq_file`` is left empty so PENTRC names it from the mode it read off
  ``euler.bin``, which is the one file that cannot be the wrong mode.
- Every method flag is written, not only the requested ones, so the namelist
  beside a torque states the whole selection.  PENTRC's own code default is
  ``fgar_flag=.true.``, so a partial write is a silent extra calculation.

Every test here uses stubs and text fixtures; none needs a GPEC installation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vaft.code import gpec, pentrc
from vaft.code.gpec import PENTRCOptions

import stat

from external_code_stubs import (
    IS_WINDOWS,
    RecordingBackend,
    write_launchable_stub,
    write_unlaunchable_file,
)

KIN_TEXT = "psi_n n_i n_e T_i T_e omega_EXB\n0.1 1e19 1e19 500 500 0.0\n1.0 1e18 1e18 50 50 0.0\n"


def _options(**overrides) -> PENTRCOptions:
    defaults = dict(
        methods=("fgar", "tgar"),
        main_ion="deuterium",
        impurity="carbon",
        collision_operator="harmonic",
    )
    defaults.update(overrides)
    return PENTRCOptions(**defaults)


def _jac_out_line(jac_out: str) -> str:
    """The header line byte for byte, including the field width that truncates.

    GPEC writes ``(1/,1x,a13,a8)`` with ``jac_out`` declared ``CHARACTER(16)``
    (``gpec/gpout.f:5762``, ``gpec/gpglobal.f``), so the word reaches the file as
    the **leftmost eight characters** of a blank-padded sixteen: ``hamada``
    arrives as ``'hamada  '`` and ``equal_arc`` as ``'equal_ar'``.  Reproduced
    here rather than approximated, because the truncation is the thing under
    test; checked against a real file from a 1.5.5 run.
    """
    return " " + "jac_out = ".rjust(13) + jac_out.ljust(16)[:8]


def _xclebsch(path: Path, jac_out: str = "hamada") -> Path:
    """A ``gpec_xclebsch_n<n>.out`` header shaped as GPEC writes it."""
    path.write_text(
        " GPEC_XCLEBSCH: Clebsch Components of the displacement.\n"
        " v1.5.5-378-gtest\n"
        "\n"
        f"{_jac_out_line(jac_out)}\n"
        "\n"
        "      mstep =   4332      mpert =  48   mthsurf = 512\n"
        "\n"
        "                     psi    m   real(derxi^psi)\n"
        "   1.000000000000000E-02    1     0.00000000E+000\n",
        encoding="utf-8",
    )
    return path


#: The one ``gpec.in`` key PENTRC's conventions are read from.
#:
#: A completed cell has the whole namelist; what the runner reads out of it is
#: ``tmag_out``, which is the toroidal angle the displacement was written in.
GPEC_IN_TEXT = "&GPEC_OUTPUT\n    jsurf_out=0\n    tmag_out=1\n/\n"


@pytest.fixture()
def cell(tmp_path) -> Path:
    """A completed ideal-GPEC cell for n = 1, with its kinetic profiles beside it."""
    directory = tmp_path / "00325" / "gpec" / "nn=1"
    directory.mkdir(parents=True)
    (directory / "euler.bin").write_bytes(b"euler")
    (directory / "gpec.in").write_text(GPEC_IN_TEXT, encoding="utf-8")
    _xclebsch(directory / "gpec_xclebsch_n1.out")
    (tmp_path / "profiles.kin").write_text(KIN_TEXT, encoding="utf-8")
    return directory


def _pent(path: Path, group: str = "pent_input") -> dict[str, str]:
    from vaft.code.gpec._runtime import read_namelist_group

    return read_namelist_group(path, group)


# --------------------------------------------------------------------------
# The Jacobian
# --------------------------------------------------------------------------


def test_the_jacobian_comes_from_the_displacement_files_own_header(tmp_path):
    assert gpec.peq_jacobian(_xclebsch(tmp_path / "x.out", "pest")) == "pest"


def test_an_empty_jac_out_becomes_default_rather_than_an_empty_string(tmp_path):
    """Both mean "DCON's own jac_type"; only one says so.

    PENTRC treats ``""`` and ``"default"`` identically
    (``pentrc/inputs.f90:631-632``), so writing ``"default"`` loses nothing and
    makes the namelist readable by a person.
    """
    assert gpec.peq_jacobian(_xclebsch(tmp_path / "x.out", "")) == "default"


def test_a_file_with_no_jac_out_line_is_refused_rather_than_guessed(tmp_path):
    """Guessing is the legacy defect this function exists to remove."""
    bare = tmp_path / "x.out"
    bare.write_text(" GPEC_XCLEBSCH:\n 1.0 1 0.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no 'jac_out ="):
        gpec.peq_jacobian(bare)


def test_only_the_header_is_read(tmp_path):
    """A production displacement file is megabytes; the header is four lines.

    Asserted by putting a second, contradictory ``jac_out`` deep in the table:
    a reader that scanned the whole file would find it.
    """
    path = _xclebsch(tmp_path / "x.out", "hamada")
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n" * 40 + " jac_out = pest\n")
    assert gpec.peq_jacobian(path) == "hamada"


def test_the_written_namelist_carries_the_files_jacobian_not_a_constant(cell):
    _xclebsch(cell / "gpec_xclebsch_n1.out", "boozer")
    path = gpec.prepare_pentrc_run(
        cell, mode=1, options=_options(), kinetic_file=cell.parents[2] / "profiles.kin"
    )
    assert _pent(path)["jac_in"] == "boozer"


# --------------------------------------------------------------------------
# What else reaches pentrc.in
# --------------------------------------------------------------------------


def test_peq_file_is_left_for_pentrc_to_name(cell):
    """So it cannot be pointed at another mode's displacement.

    PENTRC fills an empty ``peq_file`` with ``gpec_xclebsch_n<nn>.out`` using
    the ``nn`` it read from ``euler.bin``
    (``pentrc/pentrc_interface.f90:237-239``) -- the mode of the run, not of the
    namelist.
    """
    path = gpec.prepare_pentrc_run(
        cell, mode=1, options=_options(), kinetic_file=cell.parents[2] / "profiles.kin"
    )
    assert _pent(path)["peq_file"] == ""


def test_the_requested_methods_are_on_and_every_other_one_is_written_off(cell):
    path = gpec.prepare_pentrc_run(
        cell, mode=1, options=_options(methods=("fgar",)), kinetic_file=cell.parents[2] / "profiles.kin"
    )
    values = _pent(path, "pent_output")
    assert values["fgar_flag"] == "t"
    # PENTRC's own default for this one is `.true.`, so leaving it unwritten
    # would add a calculation nobody asked for.
    assert values["tgar_flag"] == "f"
    for method in pentrc.TORQUE_METHODS:
        assert values[f"{method}_flag"] == ("t" if method == "fgar" else "f"), method


def test_every_method_vaft_can_read_has_a_namelist_flag_to_ask_for_it(cell):
    """The reader's vocabulary and the writer's are the same eighteen.

    ``pentrc.in``'s ``&PENT_OUTPUT`` declares one ``<method>_flag`` per method
    (``pentrc/pentrc_interface.f90:156-158``), and ``TORQUE_METHODS`` is the
    reader's list of the same.  If the two ever drift, this asks for a flag the
    template has not got and ``write_template`` refuses it here rather than
    PENTRC ignoring it at run time.
    """
    path = gpec.prepare_pentrc_run(
        cell,
        mode=1,
        options=_options(methods=tuple(pentrc.TORQUE_METHODS)),
        kinetic_file=cell.parents[2] / "profiles.kin",
    )
    values = _pent(path, "pent_output")
    assert all(values[f"{method}_flag"] == "t" for method in pentrc.TORQUE_METHODS)


def _template_without(tmp_path, *methods: str) -> Path:
    """A ``templates_dir`` whose ``pentrc.in`` lacks the named ``<method>_flag`` lines."""
    templates = tmp_path / "templates"
    templates.mkdir(exist_ok=True)
    packaged = gpec._runtime.package_vest_dir() / "pentrc.in"
    kept = "\n".join(
        line for line in packaged.read_text(encoding="utf-8").splitlines()
        if not any(line.strip().startswith(f"{m}_flag") for m in methods)
    )
    (templates / "pentrc.in").write_text(kept + "\n", encoding="utf-8")
    return templates


#: The nine flags GPEC's own ``input/pentrc.in`` does not declare.
GPEC_OMITS = ("twmm", "pwmm", "ttmm", "ptmm", "tkmm", "pkmm", "frmm", "trmm", "prmm")


def test_an_unselected_method_the_template_does_not_declare_is_left_out(cell, tmp_path):
    """GPEC's own ``input/pentrc.in`` declares 9 of the 18 flags.

    The missing nine default to ``.false.`` inside PENTRC
    (``pentrc/pentrc_interface.f90:63-80``), so a torque run that does not
    select them loses nothing by not writing them; it used to die on a bare
    ``KeyError`` (cold review 0.8.0 delta-squash F2).
    """
    path = gpec.prepare_pentrc_run(
        cell, mode=1, options=_options(methods=("fgar", "tgar")),
        kinetic_file=cell.parents[2] / "profiles.kin",
        config=gpec.GPECSuiteConfig(templates_dir=_template_without(tmp_path, *GPEC_OMITS)),
    )
    values = _pent(path, "pent_output")
    assert values["fgar_flag"] == "t" and values["tgar_flag"] == "t"
    assert values["pgar_flag"] == "f"
    assert not any(f"{m}_flag" in values for m in GPEC_OMITS)


def test_a_selected_method_the_template_does_not_declare_is_refused_by_name(cell, tmp_path):
    """Asked for, it has to land: a ``ValueError`` naming the template and the key."""
    templates = _template_without(tmp_path, *GPEC_OMITS)
    with pytest.raises(ValueError, match="twmm_flag") as refused:
        gpec.prepare_pentrc_run(
            cell, mode=1, options=_options(methods=("twmm",)),
            kinetic_file=cell.parents[2] / "profiles.kin",
            config=gpec.GPECSuiteConfig(templates_dir=templates),
        )
    assert isinstance(refused.value, KeyError)
    assert str(templates / "pentrc.in") in str(refused.value)
    assert not (cell / "pentrc.in").exists()


def test_the_flag_pentrc_turns_on_by_itself_is_required_even_when_unselected(cell, tmp_path):
    """``fgar_flag=.true.`` is PENTRC's code default, so an absent line runs FGAR.

    A template that cannot say ``fgar_flag=f`` cannot express a run without it,
    and leaving the key out would be the silent extra calculation this module
    writes every flag to prevent.
    """
    with pytest.raises(ValueError, match="fgar_flag"):
        gpec.prepare_pentrc_run(
            cell, mode=1, options=_options(methods=("tgar",)),
            kinetic_file=cell.parents[2] / "profiles.kin",
            config=gpec.GPECSuiteConfig(templates_dir=_template_without(tmp_path, "fgar")),
        )


def test_the_species_are_integers_because_the_namelist_keys_are(cell):
    """``mi``/``zi``/``mimp``/``zimp`` are Fortran integers.

    A namelist read of ``2.0`` into an integer is an error, not a rounding
    (``pentrc/pentrc_interface.f90:99-103``), so a float here would stop PENTRC
    before it started.
    """
    path = gpec.prepare_pentrc_run(
        cell,
        mode=1,
        options=_options(main_ion="deuterium", impurity="carbon"),
        kinetic_file=cell.parents[2] / "profiles.kin",
    )
    values = _pent(path)
    assert (values["mi"], values["zi"], values["mimp"], values["zimp"]) == ("2", "1", "12", "6")
    assert "." not in values["mi"] + values["mimp"]


def test_the_grids_and_the_scan_factors_reach_the_file(cell):
    path = gpec.prepare_pentrc_run(
        cell,
        mode=1,
        options=_options(
            grids=("dynamic", "input"),
            psi_limits=(0.05, 0.99),
            artificial_factors={"wefac": 0.0},
        ),
        kinetic_file=cell.parents[2] / "profiles.kin",
    )
    output = _pent(path, "pent_output")
    assert (output["dynamic_grid"], output["equil_grid"], output["input_grid"]) == ("t", "f", "t")
    control = _pent(path, "pent_control")
    assert float(control["wefac"]) == 0.0
    assert [float(value) for value in control["psilims"].split(",")] == [0.05, 0.99]


def test_the_default_data_dir_is_the_installation_not_a_tree_depth(cell):
    """``"default"`` is ``$GPECHOME/pentrc``.

    The runs this was ported from carried ``"../../../pentrc"``, which is a claim
    about how deep the run directory sits and breaks the moment the tree is laid
    out differently -- silently, since the two methods that read it are off.
    """
    path = gpec.prepare_pentrc_run(
        cell, mode=1, options=_options(), kinetic_file=cell.parents[2] / "profiles.kin"
    )
    assert _pent(path)["data_dir"] == "default"


def test_the_kinetic_file_is_staged_into_the_cell_and_named_relatively(cell):
    """A namelist pointing outside its own directory stops being reproducible.

    The run's inputs then depend on a path that may not survive the tree being
    moved or archived, which is exactly what provenance is supposed to remove.
    """
    source = cell.parents[2] / "profiles.kin"
    path = gpec.prepare_pentrc_run(cell, mode=1, options=_options(), kinetic_file=source)
    assert (cell / "profiles.kin").read_text(encoding="utf-8") == KIN_TEXT
    assert _pent(path)["kinetic_file"] == "profiles.kin"


def test_a_kin_already_in_the_cell_is_not_copied_over_itself(cell):
    inside = cell / "in_place.kin"
    inside.write_text(KIN_TEXT, encoding="utf-8")
    path = gpec.prepare_pentrc_run(cell, mode=1, options=_options(), kinetic_file=inside)
    assert _pent(path)["kinetic_file"] == "in_place.kin"
    assert inside.read_text(encoding="utf-8") == KIN_TEXT


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


def test_a_missing_displacement_is_refused_before_anything_is_written(cell):
    (cell / "gpec_xclebsch_n1.out").unlink()
    with pytest.raises(FileNotFoundError, match="xclebsch_flag and ascii_flag"):
        gpec.prepare_pentrc_run(
            cell, mode=1, options=_options(), kinetic_file=cell.parents[2] / "profiles.kin"
        )
    assert not (cell / "pentrc.in").exists()


def test_a_missing_kin_is_refused(cell):
    with pytest.raises(FileNotFoundError, match="kinetic profile file"):
        gpec.prepare_pentrc_run(
            cell, mode=1, options=_options(), kinetic_file=cell / "absent.kin"
        )


def test_the_prerequisites_name_the_step_that_writes_each_one(cell):
    (cell / "euler.bin").unlink()
    (cell / "gpec_xclebsch_n1.out").unlink()
    problems = gpec.validate_pentrc_inputs(cell, 1)
    assert len(problems) == 3
    assert any("pentrc.in" in reason for reason in problems)
    assert any("stage_dcon_products" in reason for reason in problems)
    assert any("ascii_flag" in reason for reason in problems)


def test_the_mode_decides_which_displacement_is_required(cell):
    assert "gpec_xclebsch_n2.out" in "".join(gpec.validate_pentrc_inputs(cell, 2))


def test_output_name_follows_pentrcs_own_convention():
    assert gpec.pentrc_output_name(3) == "pentrc_output_n3.nc"


# --------------------------------------------------------------------------
# Running it
# --------------------------------------------------------------------------


def _config(tmp_path, *, installed: bool = True, **overrides) -> gpec.GPECSuiteConfig:
    home = tmp_path / "gpec_home"
    if installed:
        write_launchable_stub(home / "bin" / "pentrc")
    else:
        (home / "bin").mkdir(parents=True, exist_ok=True)
    return gpec.GPECSuiteConfig(gpec_home=home, **overrides)


def test_prepare_only_writes_the_namelist_and_launches_nothing(cell, tmp_path):
    record = gpec.run_pentrc(
        cell,
        mode=1,
        options=_options(),
        kinetic_file=cell.parents[2] / "profiles.kin",
        config=_config(tmp_path, run_mode="prepare_only"),
    )
    assert record.module == "pentrc"
    assert record.status == "prepared"
    assert record.commands == ()
    assert (cell / "pentrc.in").is_file()


@pytest.mark.parametrize("installed, run_mode", [(False, "prepare_only"), (True, "run_if_available")])
def test_a_missing_displacement_is_skipped_not_raised_under_the_lenient_modes(
    cell, tmp_path, installed, run_mode
):
    """``skipped`` when an input is missing, as the docstring says -- in every mode but strict.

    ``prepare_pentrc_run`` needs the displacement for ``jac_in`` and raised
    ``FileNotFoundError`` before the ``prepare_only`` return and before the
    input check, after staging the kin file into the cell (cold review 0.8.0
    delta-squash F4).
    """
    (cell / "gpec_xclebsch_n1.out").unlink()
    record = gpec.run_pentrc(
        cell, mode=1, options=_options(),
        kinetic_file=cell.parents[2] / "profiles.kin",
        config=_config(tmp_path, installed=installed, run_mode=run_mode),
    )
    assert record.status == "skipped"
    assert "gpec_xclebsch_n1.out" in record.reason
    assert not (cell / "pentrc.in").exists()
    assert not (cell / "profiles.kin").exists(), "nothing is staged into a cell that cannot run"


def test_strict_still_raises_for_a_missing_displacement(cell, tmp_path):
    (cell / "gpec_xclebsch_n1.out").unlink()
    with pytest.raises(FileNotFoundError, match="xclebsch_flag and ascii_flag"):
        gpec.run_pentrc(
            cell, mode=1, options=_options(),
            kinetic_file=cell.parents[2] / "profiles.kin",
            config=_config(tmp_path, run_mode="strict"),
        )


def test_an_uninstalled_pentrc_is_skipped_with_the_reason(cell, tmp_path):
    record = gpec.run_pentrc(
        cell,
        mode=1,
        options=_options(),
        kinetic_file=cell.parents[2] / "profiles.kin",
        config=_config(tmp_path, installed=False),
    )
    assert record.status == "skipped"
    assert "pentrc" in record.reason


def test_a_pentrc_without_an_execute_bit_is_skipped_as_unlaunchable(cell, tmp_path):
    home = tmp_path / "gpec_home"
    write_unlaunchable_file(home / "bin" / "pentrc")
    record = gpec.run_pentrc(
        cell,
        mode=1,
        options=_options(),
        kinetic_file=cell.parents[2] / "profiles.kin",
        config=gpec.GPECSuiteConfig(gpec_home=home),
    )
    assert record.status == "skipped"


def test_strict_raises_for_a_missing_executable(cell, tmp_path):
    with pytest.raises(FileNotFoundError):
        gpec.run_pentrc(
            cell,
            mode=1,
            options=_options(),
            kinetic_file=cell.parents[2] / "profiles.kin",
            config=_config(tmp_path, installed=False, run_mode="strict"),
        )


def test_exit_zero_with_no_output_is_a_failure_not_a_quiet_success(cell, tmp_path):
    """The stub exits 0 and writes nothing, which is the case that used to pass.

    ``PENTRCOptions`` refuses both output forms off, so there is no legitimate
    way for a successful run to leave no netCDF behind.
    """
    record = gpec.run_pentrc(
        cell,
        mode=1,
        options=_options(),
        kinetic_file=cell.parents[2] / "profiles.kin",
        config=_config(tmp_path),
    )
    assert record.status == "failed"
    assert "wrote no pentrc_output_n1.nc" in record.reason
    assert record.returncode == 0


def test_a_nonzero_exit_is_reported_with_its_code(cell, tmp_path):
    home = tmp_path / "gpec_home"
    write_launchable_stub(home / "bin" / "pentrc", exit_code=3)
    record = gpec.run_pentrc(
        cell,
        mode=1,
        options=_options(),
        kinetic_file=cell.parents[2] / "profiles.kin",
        config=gpec.GPECSuiteConfig(gpec_home=home),
    )
    assert record.status == "failed"
    assert record.returncode == 3
    assert "exited 3" in record.reason


def _writing_stub(path: Path, *, mode: int = 1, log: str = "") -> Path:
    """A stand-in for ``pentrc`` that writes the netCDF a real run writes.

    Written by the *program*, because that is the only thing whose output may be
    reported as the run's: ``run_pentrc`` deletes the expected file before
    launching, so a file this script does not create is not there afterwards.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    body = f"#!/bin/sh\nprintf 'CDF' > pentrc_output_n{int(mode)}.nc\n"
    if log:
        body += f"cat <<'EOF'\n{log}\nEOF\n"
    target.write_text(body + "exit 0\n", encoding="utf-8")
    target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return target


@pytest.mark.skipif(IS_WINDOWS, reason="the stub is a POSIX shell script")
def test_a_run_that_writes_its_output_completes_and_reports_it(cell, tmp_path):
    home = tmp_path / "gpec_home"
    _writing_stub(home / "bin" / "pentrc")
    record = gpec.run_pentrc(
        cell,
        mode=1,
        options=_options(),
        kinetic_file=cell.parents[2] / "profiles.kin",
        config=gpec.GPECSuiteConfig(gpec_home=home),
    )
    assert record.status == "completed", record.reason
    assert record.outputs == (cell / "pentrc_output_n1.nc",)
    assert record.logs and record.logs[0].name == "pentrc.log"
    assert record.reason == ""


@pytest.mark.skipif(IS_WINDOWS, reason="the stub is a POSIX shell script")
def test_a_stale_output_is_not_reported_as_this_runs_product(cell, tmp_path):
    """The netCDF is deleted before the launch, so its presence means one thing.

    PENTRC's own failure exit is ``STOP "ERROR: ..."``, which gfortran gives
    status **0**. With an earlier run's output still in the cell, a run that
    stopped that way reported ``completed`` with a file it never wrote --
    the same numbers, one shot or one mode or one ``nl`` out of date.
    """
    stale = cell / "pentrc_output_n1.nc"
    stale.write_bytes(b"CDF-from-an-earlier-run")
    home = tmp_path / "gpec_home"
    write_launchable_stub(home / "bin" / "pentrc")  # writes nothing

    record = gpec.run_pentrc(
        cell,
        mode=1,
        options=_options(),
        kinetic_file=cell.parents[2] / "profiles.kin",
        config=gpec.GPECSuiteConfig(gpec_home=home),
    )
    assert record.status == "failed"
    assert "wrote no pentrc_output_n1.nc" in record.reason
    assert record.outputs == ()
    assert not stale.exists()


@pytest.mark.skipif(IS_WINDOWS, reason="the stub is a POSIX shell script")
def test_a_fortran_error_stop_is_a_failure_despite_exit_zero(cell, tmp_path):
    """The log is the only place such a run says it stopped.

    ``STOP "ERROR: ..."`` prints the string and exits 0 under gfortran -- only an
    integer stop-code sets the status -- and PENTRC uses that form for every
    quadrature it gives up on (``pentrc/energy.f90:225,280``,
    ``pentrc/pitch.f90:279``). So a run that stopped there, having already
    written an output in an earlier pass, looked exactly like success.
    """
    home = tmp_path / "gpec_home"
    _writing_stub(
        home / "bin" / "pentrc",
        log="ERROR: xintgrl_lsode - too many steps in x required.",
    )
    record = gpec.run_pentrc(
        cell,
        mode=1,
        options=_options(),
        kinetic_file=cell.parents[2] / "profiles.kin",
        config=gpec.GPECSuiteConfig(gpec_home=home),
    )
    assert record.status == "failed"
    assert record.returncode == 0
    assert "xintgrl_lsode" in record.reason
    assert "exits 0" in record.reason
    # And the file it did write is not offered as a result.
    assert record.outputs == ()


@pytest.mark.skipif(IS_WINDOWS, reason="the stub is a POSIX shell script")
def test_lsode_non_convergence_is_counted_onto_a_completed_record(cell, tmp_path):
    """Not a verdict: LSODE can complain and recover.

    PENTRC checks only ``istate == -1`` (``pentrc/energy.f90:224``), so
    ``istate == -5`` -- a repeated corrector failure -- leaves the run going with
    a result nothing vouched for. Silence about it is what hid it; calling it a
    failure would be this function deciding a physics question.
    """
    home = tmp_path / "gpec_home"
    _writing_stub(
        home / "bin" / "pentrc",
        log=(
            " LSODE- Energy-  At T (=R1) and step size H (=R2), the\n"
            "       corrector convergence failed repeatedly\n"
            " LSODE- Energy-  At T (=R1) and step size H (=R2), the\n"
            "       corrector convergence failed repeatedly"
        ),
    )
    record = gpec.run_pentrc(
        cell,
        mode=1,
        options=_options(),
        kinetic_file=cell.parents[2] / "profiles.kin",
        config=gpec.GPECSuiteConfig(gpec_home=home),
    )
    assert record.status == "completed"
    assert record.ok
    assert "non-convergence on 2 line(s)" in record.reason
    assert "istate==-1" in record.reason


def test_the_log_scan_streams_rather_than_reads(tmp_path):
    """A diverging run's log is hundreds of megabytes of one repeated line.

    Counting stops at the scan limit and the caller says "at least", so the
    report stays a sentence whatever the log does.
    """
    log = tmp_path / "pentrc.log"
    limit = gpec._pentrc.PENTRC_NONCONVERGENCE_SCAN_LIMIT  # a tuning knob, not API
    log.write_text(
        "       corrector convergence failed repeatedly\n" * (limit + 500),
        encoding="utf-8",
    )
    fatal, count = gpec.scan_pentrc_log(log)
    assert fatal is None
    assert count == limit


def test_the_log_scan_finds_the_first_fatal_line_only(tmp_path):
    log = tmp_path / "pentrc.log"
    log.write_text(
        " PENTRC START\n"
        "ERROR: lambdaintgrl_lsode - too many steps in lambda required.\n"
        "ERROR: a later one\n",
        encoding="utf-8",
    )
    fatal, count = gpec.scan_pentrc_log(log)
    assert fatal == "ERROR: lambdaintgrl_lsode - too many steps in lambda required."
    # The fatal line says "too many steps" too, and is not also counted as a
    # non-convergence warning: it is the thing that stopped the run, not a
    # complaint the run carried on past.
    assert count == 0


def test_a_missing_log_scans_clean(tmp_path):
    assert gpec.scan_pentrc_log(tmp_path / "absent.log") == (None, 0)


def test_pentrc_is_not_a_suite_module(cell):
    """Adding the name to ``SUPPORTED_MODULES`` would give it a directory.

    It has to run in the ideal-GPEC cell -- the only one holding ``euler.bin``,
    the displacement and the ``.kin`` at once -- so ``modules=(..., "pentrc")``
    must stay an error rather than become a half-working layout.
    """
    assert "pentrc" not in gpec.SUPPORTED_MODULES
    assert "pentrc" not in gpec.DEFAULT_MODULES
    with pytest.raises(ValueError, match="Unsupported GPEC suite module"):
        gpec.prepare_gpec_suite_case(
            gpec.GPECCaseInputs(shot=1, time_ms=1, geqdsk=cell / "euler.bin", workdir=cell),
            gpec.GPECSuiteConfig(modules=("pentrc",)),
        )


# --------------------------------------------------------------------------
# PENTRCOptions
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"methods": ()}, "methods is empty"),
        ({"methods": ("fgar", "nope")}, "unknown PENTRC method"),
        ({"main_ion": "D"}, "main_ion must be one of"),
        ({"impurity": "C"}, "impurity must be one of"),
        ({"collision_operator": "bgk"}, "collision_operator must be one of"),
        ({"moment": "particle"}, "moment must be"),
        ({"bounce_harmonics": -1}, "cannot be negative"),
        ({"grids": ("lsode",)}, "unknown PENTRC grid"),
        ({"grids": ()}, "grids is empty"),
        ({"artificial_factors": {"wxfac": 2.0}}, "no scan factor"),
        ({"psi_limits": (0.9, 0.1)}, "must be increasing"),
        ({"output_ascii": False, "output_netcdf": False}, "write nothing"),
    ],
)
def test_options_refuse_what_would_produce_no_usable_run(overrides, match):
    with pytest.raises(ValueError, match=match):
        _options(**overrides)


def test_the_species_vocabulary_and_the_namelist_agree():
    options = _options(main_ion="helium", impurity="tungsten")
    assert options.species_namelist == {"mi": 4, "zi": 2, "mimp": 184, "zimp": 74}


def test_the_grid_names_are_the_readers_grid_names_where_they_overlap():
    """``equil`` and ``input`` are spelled the same on both sides.

    ``dynamic`` is the exception and deliberately so: ``pentrc.in`` calls the
    switch ``dynamic_grid`` and the output calls the grid ``lsode``
    (``TORQUE_GRIDS``), because one names the method and the other the solver
    that produced it.  Pinned so the mismatch is a documented fact rather than a
    surprise at read time.
    """
    assert set(PENTRCOptions.GRID_FLAGS) - {"dynamic"} <= set(pentrc.TORQUE_GRIDS)
    assert "dynamic" not in pentrc.TORQUE_GRIDS
    assert "lsode" in pentrc.TORQUE_GRIDS


def test_the_packaged_template_ships_every_method_off():
    """So a selection is stated in full rather than added to GPEC's own default.

    GPEC's ``input/pentrc.in`` ships ``fgar_flag = t``; a caller who asked for
    ``tgar`` alone against that template would get both.
    """
    from vaft.code.gpec._runtime import package_vest_dir

    values = _pent(package_vest_dir() / "pentrc.in", "pent_output")
    for method in pentrc.TORQUE_METHODS:
        assert values[f"{method}_flag"] == "f", method


# --------------------------------------------------------------------------
# A run that will not converge
# --------------------------------------------------------------------------


def _timing_out(seconds: float):
    """Stand in for ``rt.run_subprocess``, writing a log and then timing out.

    The log matters: the record reports its size, and the reason a timeout is
    worth reporting at all is that PENTRC's failure mode is *volume* -- LSODE's
    energy corrector failing repeatedly wrote 223 MB of one repeated message in
    the 300 s it was given, on a real case.
    """
    import subprocess

    def run(executable_path, cwd, log_path, *, config):
        Path(log_path).write_text(
            " LSODE- Energy-  At T (=R1) and step size H (=R2), the\n"
            "       corrector convergence failed repeatedly\n" * 64,
            encoding="utf-8",
        )
        raise subprocess.TimeoutExpired([str(executable_path)], seconds)

    return run


def test_a_pentrc_that_does_not_finish_is_a_failed_record_not_an_exception(
    cell, tmp_path, monkeypatch
):
    from vaft.code.gpec import _pentrc, _runtime

    monkeypatch.setattr(_runtime, "run_subprocess", _timing_out(300.0))
    record = _pentrc.run_pentrc(
        cell,
        mode=1,
        options=_options(),
        kinetic_file=cell.parents[2] / "profiles.kin",
        config=_config(tmp_path, timeout=300.0),
    )
    assert record.status == "failed"
    assert record.returncode is None
    assert "300 s" in record.reason
    # The log is named and its size reported, because that is the evidence.
    assert record.logs and record.logs[0].name == "pentrc.log"
    assert str(record.logs[0].stat().st_size) in record.reason
    # No output claimed: PENTRC writes its netCDF at the end, so a timed-out run
    # has nothing to salvage.
    assert record.outputs == ()


def test_a_pentrc_memory_stop_says_so(cell, tmp_path, monkeypatch):
    """#1460: a memory stop is not "did not finish within N s"."""
    from vaft.code.execution import ExecutionResult
    from vaft.code.gpec import _pentrc, _runtime

    def run(executable_path, cwd, log_path, *, config):
        Path(log_path).write_text("partial\n", encoding="utf-8")
        execution = ExecutionResult(returncode=None, timed_out=True, runtime_status="memory_limit",
                                    peak_rss_mb=9000.0, elapsed_s=42.0)
        raise _runtime.GPECLimitStop([str(executable_path)], 300.0, execution,
                                     "pentrc was stopped by the memory limit at 9000 MiB resident after 42 s")

    monkeypatch.setattr(_runtime, "run_subprocess", run)
    record = _pentrc.run_pentrc(
        cell,
        mode=1,
        options=_options(),
        kinetic_file=cell.parents[2] / "profiles.kin",
        config=_config(tmp_path, timeout=300.0),
    )
    assert record.status == "failed"
    assert record.reason.startswith("pentrc was stopped by the memory limit")
    assert "did not finish within" not in record.reason and "its log is 8 bytes" in record.reason


def test_strict_still_raises_when_pentrc_does_not_finish(cell, tmp_path, monkeypatch):
    """Because a scheduler-driven scan wants the walltime failure to stop it."""
    import subprocess

    from vaft.code.gpec import _pentrc, _runtime

    monkeypatch.setattr(_runtime, "run_subprocess", _timing_out(5.0))
    with pytest.raises(subprocess.TimeoutExpired):
        _pentrc.run_pentrc(
            cell,
            mode=1,
            options=_options(),
            kinetic_file=cell.parents[2] / "profiles.kin",
            config=_config(tmp_path, run_mode="strict", timeout=5.0),
        )


# --------------------------------------------------------------------------
# The three halves of the displacement's convention
# --------------------------------------------------------------------------


def test_a_truncated_jacobian_name_is_resolved_not_passed_on(cell):
    """``equal_arc`` reaches the file as ``equal_ar``, and PENTRC stops on that.

    The header field is ``a8``, so the one name in ``JACOBIAN_NAMES`` longer than
    eight characters cannot survive it. PENTRC's own ``SELECT CASE`` has no
    ``equal_ar`` branch (``pentrc/inputs.f90:404-418``), so writing the word back
    verbatim would stop the run with a message about a Jacobian nobody asked for.
    """
    _xclebsch(cell / "gpec_xclebsch_n1.out", jac_out="equal_arc")
    assert "equal_ar" in (cell / "gpec_xclebsch_n1.out").read_text(encoding="utf-8")
    assert gpec.peq_jacobian(cell / "gpec_xclebsch_n1.out") == "equal_arc"


@pytest.mark.parametrize("name", ["hamada", "pest", "boozer", "park", "polar", "other"])
def test_every_other_jacobian_name_survives_the_header_unchanged(cell, name):
    """Eight characters or fewer, so there is nothing to resolve."""
    _xclebsch(cell / "gpec_xclebsch_n1.out", jac_out=name)
    assert gpec.peq_jacobian(cell / "gpec_xclebsch_n1.out") == name


def test_a_word_that_is_no_jacobian_is_refused(cell):
    _xclebsch(cell / "gpec_xclebsch_n1.out", jac_out="chebyshe")
    with pytest.raises(ValueError, match="not a Jacobian"):
        gpec.peq_jacobian(cell / "gpec_xclebsch_n1.out")


def test_the_toroidal_angle_comes_from_the_runs_own_gpec_in(cell, tmp_path):
    """``tmag_in`` is ``gpec.in``'s ``tmag_out``, because that is what wrote the file.

    ``gpout_xclebsch`` transforms through ``gpeq_bcoordsout`` with no explicit
    angle, and that routine takes ``tout = tmag_out`` (``gpec/gpeq.f:826-829``).
    A ``pentrc.in`` that assumed the common value would transform the harmonics
    under the wrong angle and say nothing.
    """
    assert gpec.peq_toroidal_angle(cell) == 1
    (cell / "gpec.in").write_text(
        "&GPEC_OUTPUT\n    jsurf_out=1\n    tmag_out=0\n/\n", encoding="utf-8"
    )
    assert gpec.peq_toroidal_angle(cell) == 0

    written = gpec.prepare_pentrc_run(
        cell, mode=1, options=_options(),
        kinetic_file=cell.parents[2] / "profiles.kin",
    )
    pent = _pent(written)
    assert pent["tmag_in"] == "0"
    # And `jsurf_in` is 0 even though this gpec.in asks for area-weighted
    # *control* output: the displacement is written with jout hard-coded to 0.
    assert pent["jsurf_in"] == "0"


def test_a_cell_with_no_gpec_in_cannot_say_which_angle_the_file_is_in(cell):
    (cell / "gpec.in").unlink()
    with pytest.raises(FileNotFoundError, match="tmag_in cannot be derived"):
        gpec.peq_toroidal_angle(cell)


def test_an_absent_tmag_out_falls_back_to_gpecs_own_code_default(cell):
    """Absent from the namelist means GPEC's declaration applies, which is 1."""
    (cell / "gpec.in").write_text("&GPEC_OUTPUT\n    jsurf_out=0\n/\n", encoding="utf-8")
    assert gpec.peq_toroidal_angle(cell) == gpec._pentrc.PEQ_TMAG_DEFAULT == 1


def test_a_namelist_without_the_requested_group_is_refused_not_read_from_the_top(tmp_path):
    """``read_namelist_group`` used to return the *first* group when the asked one was absent.

    Reached through hand-edited namelists read with the exported reader: a
    ``tmag_out`` placed in ``&GPEC_INPUT`` came back as ``&GPEC_OUTPUT``'s
    (cold review 0.8.0 delta-squash F5).
    """
    from vaft.code.gpec._runtime import read_namelist_group

    path = tmp_path / "gpec.in"
    path.write_text("&GPEC_INPUT\n  tmag_in=0\n/\n&GPEC_OUTPUT\n  tmag_out=0\n/\n", encoding="utf-8")
    assert read_namelist_group(path, "gpec_output") == {"tmag_out": "0"}
    with pytest.raises(ValueError, match=r"no &xyz namelist group") as refused:
        read_namelist_group(path, "xyz")
    assert "gpec_input" in str(refused.value) and "gpec_output" in str(refused.value)


def test_a_tmag_out_in_the_wrong_group_does_not_pass_for_the_output_angle(cell):
    (cell / "gpec.in").write_text("&GPEC_INPUT\n    tmag_out=0\n/\n", encoding="utf-8")
    with pytest.raises(ValueError, match="GPEC_OUTPUT|gpec_output"):
        gpec.peq_toroidal_angle(cell)


def test_an_unbounded_run_is_refused_rather_than_launched(cell, tmp_path):
    """``timeout=None`` is not "as long as it takes" for this module.

    PENTRC does not stop when an integration will not converge -- it prints --
    so an unbounded run reports nothing and grows a log without limit. The suite's
    other modules exit on their own; this one has no other brake.
    """
    home = tmp_path / "gpec_home"
    write_launchable_stub(home / "bin" / "pentrc")
    with pytest.raises(ValueError, match="needs a timeout"):
        gpec.run_pentrc(
            cell,
            mode=1,
            options=_options(),
            kinetic_file=cell.parents[2] / "profiles.kin",
            config=gpec.GPECSuiteConfig(gpec_home=home, timeout=None),
        )


@pytest.mark.skipif(IS_WINDOWS, reason="the stub is a POSIX shell script")
def test_an_unbounded_run_is_allowed_when_something_else_bounds_it(cell, tmp_path):
    """A scheduler with its own walltime is the case the opt-in exists for."""
    home = tmp_path / "gpec_home"
    _writing_stub(home / "bin" / "pentrc")
    record = gpec.run_pentrc(
        cell,
        mode=1,
        options=_options(),
        kinetic_file=cell.parents[2] / "profiles.kin",
        config=gpec.GPECSuiteConfig(gpec_home=home, timeout=None),
        allow_unbounded_runtime=True,
    )
    assert record.status == "completed", record.reason


def test_preparing_only_needs_no_timeout(cell, tmp_path):
    """Nothing is launched, so there is nothing to bound."""
    record = gpec.run_pentrc(
        cell,
        mode=1,
        options=_options(),
        kinetic_file=cell.parents[2] / "profiles.kin",
        config=gpec.GPECSuiteConfig(
            gpec_home=tmp_path / "gpec_home", timeout=None, run_mode="prepare_only"
        ),
    )
    assert record.status == "prepared"
