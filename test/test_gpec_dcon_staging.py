"""Ideal GPEC reads its DCON products out of its own directory.

GPEC *writes* the vacuum handshake files into its working directory and *reads*
them back from ``dcon_dir``: ``ahg2msc_gpec.out`` and one
``ahg2msc_gpecflx_<psi>.out`` per rational surface.  Driven the usual way --
DCON and GPEC in one directory -- the two coincide and nothing shows.  Given a
separate directory per module, GPEC creates an empty file under ``dcon_dir``
and stops at its first read with ``Fortran runtime error: End of file``, an
hour into a run, naming a file it wrote itself.

The per-module layout is kept and DCON's products are brought to the cell
instead.  Every test here stubs the solvers: the executable is a script that
writes the files a real run would, so what is under test is the *state of the
cell at the moment GPEC starts*, which is the thing that was wrong.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from vaft.code import gpec
from vaft.code.gpec import DCON_PRODUCTS_FOR_GPEC, stage_dcon_products

from external_code_stubs import IS_WINDOWS, write_launchable_stub

GFILE_TEXT = "  EFITD   01/01/2024   #  39915  325ms        3  65  65\n 1.0 2.0 3.0\n"
#: What a real ideal-GPEC run writes into its working directory and reads back
#: out of ``dcon_dir``; the flux one is named for a rational surface, so the
#: set is not known until the solve finds them.
HANDSHAKE_FILES = ("ahg2msc_gpec.out", "ahg2msc_gpecflx_0.5428.out")


@pytest.fixture()
def case(tmp_path):
    geqdsk = tmp_path / "g039915.00325"
    geqdsk.write_text(GFILE_TEXT, encoding="utf-8")
    return gpec.GPECCaseInputs(
        shot=39915, time_ms=325, geqdsk=geqdsk, workdir=tmp_path / "run"
    )


def _dcon_cell(case, mode: int = 1) -> Path:
    cell = gpec._module_dir(case.workdir, case.time_ms, "dcon", mode, geqdsk=case.geqdsk)
    cell.mkdir(parents=True, exist_ok=True)
    return cell


#: What a prepared DCON cell's ``equil.in`` says, reduced to the two lines the
#: staging reads.  A real one is 25 keys long; this is the group name, the
#: equilibrium it names, and nothing else, so a test that breaks here is about
#: the staging rather than about the template.
EQUIL_IN_TEXT = """&EQUIL_CONTROL
    eq_type='efit'
    eq_filename="{name}"
/
"""


def _valid_dcon_netcdf(path: Path, *, n: int) -> None:
    """The minimum ``dcon_output_n<n>.nc`` ``validate_dcon_result`` accepts."""
    import xarray as xr

    xr.Dataset(
        {"W_t_eigenvalue": (("mode", "i"), [[-0.3, 0.0]])},
        coords={"i": [0, 1], "mode": [1]},
        attrs={"mlow": -2, "mhigh": 0, "mpert": 3, "mband": 0, "n": n},
    ).to_netcdf(path / f"dcon_output_n{n}.nc")


def _equilibrium_inputs(cell: Path, case) -> None:
    """``equil.in`` and the g-file it names, in a cell built by hand."""
    (cell / "equil.in").write_text(
        EQUIL_IN_TEXT.format(name=case.geqdsk.name), encoding="utf-8"
    )
    (cell / case.geqdsk.name).write_text(GFILE_TEXT, encoding="utf-8")


def _complete_dcon(
    case, mode: int = 1, *, optional: bool = False, equilibrium: bool = True
) -> Path:
    """A DCON cell holding what ideal GPEC needs from it.

    ``equil.in`` and the g-file it names included: GPEC opens ``equil.in`` in its
    own working directory and stops the program without it, so a cell without one
    is not a cell ideal GPEC can run in.
    """
    cell = _dcon_cell(case, mode)
    (cell / "euler.bin").write_bytes(b"euler")
    (cell / "psi_in.bin").write_bytes(b"psi")
    if equilibrium:
        (cell / "equil.in").write_text(
            EQUIL_IN_TEXT.format(name=case.geqdsk.name), encoding="utf-8"
        )
        (cell / case.geqdsk.name).write_text(GFILE_TEXT, encoding="utf-8")
    if optional:
        (cell / "vacuum.bin").write_bytes(b"vac")
        (cell / "globalsol.bin").write_bytes(b"gal")
    return cell


# --- the staging itself ----------------------------------------------------


def test_the_required_products_are_brought_to_the_cell(case, tmp_path):
    dcon = _complete_dcon(case)
    run_dir = tmp_path / "gpec_cell"
    staged = stage_dcon_products(dcon, run_dir)

    assert [p.name for p in staged] == [
        "euler.bin", "psi_in.bin", "equil.in", case.geqdsk.name
    ]
    assert (run_dir / "euler.bin").read_bytes() == b"euler"
    assert (run_dir / "psi_in.bin").read_bytes() == b"psi"


def test_the_optional_products_are_brought_when_dcon_made_them(case, tmp_path):
    """``vacuum.bin`` is deprecated and ``globalsol.bin`` needs ``gal_flag``.

    Absent, neither is an error; present, both are staged, because ``gpec.in``
    names them and GPEC joins each onto ``dcon_dir``.
    """
    required = [name for name, needed in DCON_PRODUCTS_FOR_GPEC if needed]
    optional = [name for name, needed in DCON_PRODUCTS_FOR_GPEC if not needed]
    assert required == ["euler.bin", "psi_in.bin"]
    assert optional == ["vacuum.bin", "globalsol.bin"]

    equilibrium = ["equil.in", case.geqdsk.name]
    without = stage_dcon_products(_complete_dcon(case), tmp_path / "a")
    assert [p.name for p in without] == required + equilibrium

    with_optional = stage_dcon_products(
        _complete_dcon(case, optional=True), tmp_path / "b"
    )
    assert [p.name for p in with_optional] == required + optional + equilibrium


def test_a_missing_required_product_names_it(case, tmp_path):
    dcon = _dcon_cell(case)
    (dcon / "euler.bin").write_bytes(b"euler")  # psi_in.bin absent
    with pytest.raises(FileNotFoundError, match="psi_in.bin"):
        stage_dcon_products(dcon, tmp_path / "gpec_cell")


def test_a_missing_equil_in_is_refused_before_gpec_starts(case, tmp_path):
    """Named here rather than by GPEC four minutes in.

    GPEC's own report is ``PROGRAM STOP => Can't open input file equil.in``, from
    a process whose DCON had already succeeded and whose cell holds every binary
    it needs -- so the file it asks for looks like something the suite never had
    rather than something it did not bring over.
    """
    dcon = _complete_dcon(case, equilibrium=False)
    with pytest.raises(FileNotFoundError, match="equil.in"):
        stage_dcon_products(dcon, tmp_path / "gpec_cell")


def test_the_equilibrium_the_namelist_names_is_staged_with_it(case, tmp_path):
    """Because the name in it is relative to whichever directory reads it.

    The DCON cell holds the g-file under a bare filename and ``equil.in`` names
    it that way, so copying the namelist alone would leave GPEC resolving it
    against its own empty cell.
    """
    dcon = _complete_dcon(case)
    run_dir = tmp_path / "gpec_cell"
    stage_dcon_products(dcon, run_dir)

    assert (run_dir / "equil.in").read_bytes() == (dcon / "equil.in").read_bytes()
    assert (run_dir / case.geqdsk.name).read_text(encoding="utf-8") == GFILE_TEXT


def test_an_absolute_equilibrium_name_is_left_alone(case, tmp_path):
    """Nothing to bring over: it resolves the same from either directory."""
    dcon = _complete_dcon(case)
    elsewhere = tmp_path / "shared" / "g039915.00325"
    elsewhere.parent.mkdir(parents=True, exist_ok=True)
    elsewhere.write_text(GFILE_TEXT, encoding="utf-8")
    (dcon / "equil.in").write_text(
        EQUIL_IN_TEXT.format(name=str(elsewhere)), encoding="utf-8"
    )
    (dcon / case.geqdsk.name).unlink()

    run_dir = tmp_path / "gpec_cell"
    staged = stage_dcon_products(dcon, run_dir)
    assert [p.name for p in staged] == ["euler.bin", "psi_in.bin", "equil.in"]
    assert not (run_dir / case.geqdsk.name).exists()


def test_a_relative_equilibrium_that_is_not_there_names_itself(case, tmp_path):
    dcon = _complete_dcon(case)
    (dcon / case.geqdsk.name).unlink()
    with pytest.raises(FileNotFoundError, match=case.geqdsk.name):
        stage_dcon_products(dcon, tmp_path / "gpec_cell")


def test_staging_into_the_directory_the_products_are_already_in_does_nothing(case):
    """GPEC's own layout: one directory, so there is nothing to bring over."""
    dcon = _complete_dcon(case)
    assert stage_dcon_products(dcon, dcon) == ()
    assert (dcon / "euler.bin").read_bytes() == b"euler"


def test_a_stale_product_is_replaced_rather_than_kept(case, tmp_path):
    """A resumed scan may point at a DCON cell that has since been re-solved.

    A stale ``euler.bin`` is a wrong answer rather than a failure, so staging
    is unconditional instead of skipping what is already there.
    """
    run_dir = tmp_path / "gpec_cell"
    run_dir.mkdir()
    (run_dir / "euler.bin").write_bytes(b"from the previous solve")
    stage_dcon_products(_complete_dcon(case), run_dir)
    assert (run_dir / "euler.bin").read_bytes() == b"euler"


@pytest.mark.skipif(IS_WINDOWS, reason="hard links are the POSIX path here")
def test_a_large_product_is_linked_rather_than_copied(case, tmp_path):
    """``euler.bin`` is the largest thing either code writes, once per mode."""
    dcon = _complete_dcon(case)
    run_dir = tmp_path / "gpec_cell"
    stage_dcon_products(dcon, run_dir)
    assert (run_dir / "euler.bin").stat().st_ino == (dcon / "euler.bin").stat().st_ino


def test_a_copy_is_used_when_the_filesystem_refuses_a_link(case, tmp_path, monkeypatch):
    """A different device, or a filesystem with no links, still works."""
    def _no_links(*args, **kwargs):
        raise OSError("cross-device link")

    monkeypatch.setattr(os, "link", _no_links)
    run_dir = tmp_path / "gpec_cell"
    stage_dcon_products(_complete_dcon(case), run_dir)
    assert (run_dir / "euler.bin").read_bytes() == b"euler"


# --- what the prepared namelist says ---------------------------------------


def test_gpec_in_points_at_its_own_cell(case, tmp_path):
    config = gpec.GPECSuiteConfig(modules=("gpec",), modes=(1,), run_mode="prepare_only")
    gpec.prepare_gpec_suite_case(case, config)
    run_dir = gpec._module_dir(case.workdir, case.time_ms, "gpec", 1, geqdsk=case.geqdsk)
    text = (run_dir / "gpec.in").read_text(encoding="utf-8")
    line = next(l for l in text.splitlines() if l.strip().startswith("dcon_dir"))
    assert str(run_dir.resolve()) in line
    # Not DCON's cell: that is the whole change.
    dcon_dir = gpec._module_dir(case.workdir, case.time_ms, "dcon", 1, geqdsk=case.geqdsk)
    assert str(dcon_dir.resolve()) not in line


# --- the handshake, end to end with a stubbed solver -----------------------


def _handshake_stub(path: Path) -> Path:
    """A stand-in for ``gpec`` that behaves the way the real one does.

    It writes the handshake files into its *working* directory and then reads
    each of them back from the ``dcon_dir`` its ``gpec.in`` names -- which is
    exactly the round trip that failed when the two were different
    directories.  It also checks that the DCON products it needs are beside it.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    script = """#!/bin/sh
set -e
dcon_dir=$(sed -n 's/^ *dcon_dir *= *"\\(.*\\)".*/\\1/p' gpec.in | head -1)
for f in euler.bin psi_in.bin; do
  [ -f "$dcon_dir/$f" ] || { echo "missing $dcon_dir/$f" >&2; exit 3; }
done
for f in ahg2msc_gpec.out ahg2msc_gpecflx_0.5428.out; do
  echo handshake > "$f"
  [ -s "$dcon_dir/$f" ] || { echo "cannot read back $dcon_dir/$f" >&2; exit 4; }
done
for n in gpec_control_output_n1.nc gpec_profile_output_n1.nc \\
         gpec_cylindrical_output_n1.nc gpec_response_n1.out \\
         gpec_bnormal_pest_n1.out; do
  echo out > "$n"
done
"""
    target.write_text(script, encoding="utf-8")
    target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return target


@pytest.mark.skipif(IS_WINDOWS, reason="the stub is a POSIX shell script")
def test_the_handshake_files_land_where_gpec_reads_them(case, monkeypatch, tmp_path):
    """The regression, with the stub standing in for the solver.

    Before the staging change this stub exits 4: it writes the handshake files
    into its working directory and cannot read them back from a ``dcon_dir``
    pointing somewhere else.
    """
    home = tmp_path / "gpec_home"
    _handshake_stub(home / "bin" / "gpec")
    monkeypatch.setenv(gpec.GPEC_HOME_ENV, str(home))

    _complete_dcon(case)
    config = gpec.GPECSuiteConfig(modules=("gpec",), modes=(1,), run_mode="strict")
    result = gpec.run_gpec_suite_case(case, config)

    (record,) = result.records
    assert record.status == "completed", record.reason
    assert record.returncode == 0
    run_dir = gpec._module_dir(case.workdir, case.time_ms, "gpec", 1, geqdsk=case.geqdsk)
    for name in HANDSHAKE_FILES:
        assert (run_dir / name).is_file(), name
    # And the DCON products were brought over, which is what let it read them.
    assert (run_dir / "euler.bin").read_bytes() == b"euler"


@pytest.mark.skipif(IS_WINDOWS, reason="the stub is a POSIX shell script")
def test_dcon_products_from_a_separate_work_tree_are_staged_too(case, monkeypatch, tmp_path):
    """``dcon_workdir`` points at another tree; the products still come here."""
    home = tmp_path / "gpec_home"
    _handshake_stub(home / "bin" / "gpec")
    monkeypatch.setenv(gpec.GPEC_HOME_ENV, str(home))

    elsewhere = tmp_path / "dcon_tree"
    dcon_cell = gpec._module_dir(elsewhere, case.time_ms, "dcon", 1, geqdsk=case.geqdsk)
    dcon_cell.mkdir(parents=True)
    (dcon_cell / "euler.bin").write_bytes(b"euler")
    (dcon_cell / "psi_in.bin").write_bytes(b"psi")
    _equilibrium_inputs(dcon_cell, case)

    inputs = gpec.GPECCaseInputs(
        shot=case.shot, time_ms=case.time_ms, geqdsk=case.geqdsk,
        workdir=case.workdir, dcon_workdir=elsewhere,
    )
    config = gpec.GPECSuiteConfig(modules=("gpec",), modes=(1,), run_mode="strict")
    result = gpec.run_gpec_suite_case(inputs, config)

    (record,) = result.records
    assert record.status == "completed", record.reason
    run_dir = gpec._module_dir(case.workdir, case.time_ms, "gpec", 1, geqdsk=case.geqdsk)
    assert (run_dir / "euler.bin").read_bytes() == b"euler"


@pytest.mark.skipif(IS_WINDOWS, reason="the stub is a POSIX shell script")
def test_two_modes_get_their_own_products(case, monkeypatch, tmp_path):
    """One DCON cell per mode, one GPEC cell per mode, no crossing over."""
    home = tmp_path / "gpec_home"
    _handshake_stub(home / "bin" / "gpec")
    monkeypatch.setenv(gpec.GPEC_HOME_ENV, str(home))

    for mode, payload in ((1, b"euler-n1"), (2, b"euler-n2")):
        cell = _dcon_cell(case, mode)
        (cell / "euler.bin").write_bytes(payload)
        (cell / "psi_in.bin").write_bytes(b"psi")
        _equilibrium_inputs(cell, case)

    # The stub only writes n=1 outputs, so only that mode can complete; the
    # point here is which euler.bin each cell received.
    config = gpec.GPECSuiteConfig(modules=("gpec",), modes=(1, 2), run_mode="auto")
    gpec.run_gpec_suite_case(case, config)
    for mode, payload in ((1, b"euler-n1"), (2, b"euler-n2")):
        run_dir = gpec._module_dir(case.workdir, case.time_ms, "gpec", mode, geqdsk=case.geqdsk)
        assert (run_dir / "euler.bin").read_bytes() == payload, mode


def test_an_incomplete_dcon_result_is_still_refused_before_staging(case, monkeypatch, tmp_path):
    """The prerequisite check comes first, so the reason names DCON."""
    executable = write_launchable_stub(tmp_path / "gpec_home" / "bin" / "gpec")
    monkeypatch.setenv(gpec.GPEC_HOME_ENV, str(executable.parent.parent))

    _dcon_cell(case)  # empty
    config = gpec.GPECSuiteConfig(modules=("gpec",), modes=(1,), run_mode="auto")
    result = gpec.run_gpec_suite_case(case, config)
    (record,) = result.records
    assert record.status == "skipped"
    assert "invalid DCON result" in record.reason
    run_dir = gpec._module_dir(case.workdir, case.time_ms, "gpec", 1, geqdsk=case.geqdsk)
    assert not (run_dir / "euler.bin").exists()


def test_a_dcon_cell_without_equil_in_is_skipped_rather_than_raising(
    case, monkeypatch, tmp_path
):
    """Under ``run_mode="auto"`` every other missing prerequisite here is a skip.

    ``equil.in`` is an *input*, so ``validate_dcon_result`` -- which judges whether
    DCON finished -- does not look for it, and a cell prepared by something other
    than this suite may not have one. Staging it is therefore the first step that
    can fail on an otherwise complete DCON result, and it must fail the same way
    the others do.
    """
    home = tmp_path / "gpec_home"
    _handshake_stub(home / "bin" / "gpec")
    monkeypatch.setenv(gpec.GPEC_HOME_ENV, str(home))

    dcon = _complete_dcon(case, equilibrium=False)
    _valid_dcon_netcdf(dcon, n=1)
    config = gpec.GPECSuiteConfig(modules=("gpec",), modes=(1,), run_mode="auto")
    result = gpec.run_gpec_suite_case(case, config)

    (record,) = result.records
    assert record.status == "skipped"
    assert "equil.in" in record.reason


def test_strict_raises_for_a_dcon_cell_without_equil_in(case, monkeypatch, tmp_path):
    home = tmp_path / "gpec_home"
    _handshake_stub(home / "bin" / "gpec")
    monkeypatch.setenv(gpec.GPEC_HOME_ENV, str(home))

    dcon = _complete_dcon(case, equilibrium=False)
    _valid_dcon_netcdf(dcon, n=1)
    config = gpec.GPECSuiteConfig(
        modules=("gpec",), modes=(1,), run_mode="strict", gpec_home=home
    )
    with pytest.raises(FileNotFoundError, match="equil.in"):
        gpec.run_gpec_suite_case(case, config)


def test_a_relative_name_that_resolves_onto_the_source_stages_nothing(case, tmp_path):
    """And above all does not delete the equilibrium.

    ``equil.in`` names the g-file relative to whichever cell reads it, and a
    relative name can resolve to the same file from both: ``../../<g-file>`` in a
    GPEC cell beside the DCON cell is the DCON cell's own. Unlinking the
    destination first then deleted the equilibrium, and the link that followed
    failed with its source already gone -- a staging step destroying what it
    stages.
    """
    dcon = _complete_dcon(case)
    run_dir = gpec._module_dir(case.workdir, case.time_ms, "gpec", 1, geqdsk=case.geqdsk)
    run_dir.mkdir(parents=True, exist_ok=True)
    # One level above both cells, which is what makes the same relative name
    # resolve onto one file from either -- the per-module layout's own shape.
    shared = dcon.parents[1] / "shared.geqdsk"
    shared.write_text(GFILE_TEXT, encoding="utf-8")
    relative = os.path.relpath(shared, dcon)
    assert relative == os.path.relpath(shared, run_dir)
    (dcon / "equil.in").write_text(EQUIL_IN_TEXT.format(name=relative), encoding="utf-8")

    staged = stage_dcon_products(dcon, run_dir)

    assert shared.read_text(encoding="utf-8") == GFILE_TEXT
    assert [p.name for p in staged] == ["euler.bin", "psi_in.bin", "equil.in"]
