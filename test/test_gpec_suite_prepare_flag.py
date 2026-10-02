"""Running a prepared case without re-preparing it.

``run_gpec_suite_case`` prepared unconditionally, and preparation rewrites every
namelist it owns from the template. So a caller that prepared a case, *added to*
it, and then called this function had its additions to those namelists reverted --
silently, because the files it staged beside them survive and the solver is happy
either way.

That is not hypothetical. A kinetic case is exactly this shape: prepare, then set
``dcon.in``'s ``kin_flag=t`` and stage the ``.kin`` the flag refers to. Reverted,
it solves the ideal problem under a kinetic label with every file present.
A downstream adapter already hit this and worked around it by calling a private
function rather than this entry point; ``prepare=False`` is the public way.

The tests below use a stub solver: what is under test is **the state of the
namelists at the moment the solver starts**, which is the thing that was wrong.
"""

from __future__ import annotations

import re

import pytest

from vaft.code import gpec

from external_code_stubs import write_launchable_stub

GFILE_TEXT = "  EFITD   01/01/2024   #  39915  325ms        3  65  65\n 1.0 2.0 3.0\n"


@pytest.fixture()
def case(tmp_path):
    geqdsk = tmp_path / "g039915.00325"
    geqdsk.write_text(GFILE_TEXT, encoding="utf-8")
    return gpec.GPECCaseInputs(
        shot=39915, time_ms=325, geqdsk=geqdsk, workdir=tmp_path / "run"
    )


def _installed(tmp_path, programs=("dcon",)) -> gpec.GPECSuiteConfig:
    home = tmp_path / "gpec_home"
    for program in programs:
        write_launchable_stub(home / "bin" / program)
    return gpec.GPECSuiteConfig(modules=programs, modes=(1,), gpec_home=home)


def _dcon_cell(case, config):
    return gpec._module_dir(case.workdir, case.time_ms, "dcon", 1, geqdsk=case.geqdsk)


def _kin_flag(path) -> str:
    return gpec.read_namelist_group(path, "dcon_control")["kin_flag"]


def _set_kin_flag(path) -> None:
    """Patch ``kin_flag`` to ``t`` whatever spacing the template used.

    The packaged ``dcon.in`` writes ``kin_flag = f``; a literal string replace on
    ``kin_flag=f`` matches nothing and the test then passes for the wrong reason,
    which is how the first version of this file "passed".
    """
    text = path.read_text(encoding="utf-8")
    patched, count = re.subn(
        r"(?m)^(\s*kin_flag\s*=\s*)f\b", r"\1t", text, count=1
    )
    assert count == 1, f"no kin_flag assignment in {path}"
    path.write_text(patched, encoding="utf-8")


def test_preparing_again_reverts_a_staged_namelist_change(case, tmp_path):
    """The defect, pinned: this is what ``prepare=True`` does to a kinetic case.

    Asserted rather than described, because it is the reason the argument exists
    and because a future change that made preparation idempotent-with-respect-to
    patches would make this test fail loudly instead of making the argument
    quietly pointless.
    """
    config = _installed(tmp_path)
    gpec.prepare_gpec_suite_case(case, config)
    cell = _dcon_cell(case, config)
    assert _kin_flag(cell / "dcon.in") == "f"

    # What a kinetic case does after preparing.
    _set_kin_flag(cell / "dcon.in")
    (cell / "profiles.kin").write_text("psi\n0\n", encoding="utf-8")
    assert _kin_flag(cell / "dcon.in") == "t"

    gpec.run_gpec_suite_case(case, config)
    assert _kin_flag(cell / "dcon.in") == "f", "prepare=True should revert it"
    # And the file it referred to is still there, which is what makes it quiet.
    assert (cell / "profiles.kin").is_file()


def test_prepare_false_leaves_the_staged_change_alone(case, tmp_path):
    config = _installed(tmp_path)
    gpec.prepare_gpec_suite_case(case, config)
    cell = _dcon_cell(case, config)
    _set_kin_flag(cell / "dcon.in")

    gpec.run_gpec_suite_case(case, config, prepare=False)
    assert _kin_flag(cell / "dcon.in") == "t"


def test_prepare_false_writes_nothing_at_all(case, tmp_path):
    """Not "writes the same thing": writes nothing.

    Checked by contents rather than by mtime, because a rewrite with identical
    bytes is still a rewrite and would still have reverted a patch.
    """
    config = _installed(tmp_path)
    gpec.prepare_gpec_suite_case(case, config)
    cell = _dcon_cell(case, config)
    before = {path.name: path.read_bytes() for path in sorted(cell.iterdir()) if path.is_file()}
    for name in before:
        if name.endswith(".in"):
            (cell / name).write_bytes(before[name] + b"! touched by the caller\n")
    after_patch = {path.name: path.read_bytes() for path in sorted(cell.iterdir()) if path.is_file()}

    gpec.run_gpec_suite_case(case, config, prepare=False)
    still = {
        path.name: path.read_bytes()
        for path in sorted(cell.iterdir())
        if path.is_file() and path.name in after_patch
    }
    assert still == after_patch


def test_an_unprepared_cell_is_named_rather_than_created(case, tmp_path):
    """Creating it would launch the solver in an empty directory.

    The Fortran error that follows names a namelist nobody wrote, which is the
    least useful place for the report to appear.
    """
    config = _installed(tmp_path)
    result = gpec.run_gpec_suite_case(case, config, prepare=False)
    (record,) = result.records
    assert record.status == "skipped"
    assert "has not been prepared" in record.reason
    assert not _dcon_cell(case, config).exists()


def test_strict_raises_for_an_unprepared_cell(case, tmp_path):
    config = _installed(tmp_path)
    config = gpec.GPECSuiteConfig(
        modules=config.modules, modes=config.modes,
        gpec_home=config.gpec_home, run_mode="strict",
    )
    with pytest.raises(FileNotFoundError, match="has not been prepared"):
        gpec.run_gpec_suite_case(case, config, prepare=False)


def test_an_unsupported_module_is_still_refused_with_prepare_false(case, tmp_path):
    """Normalization happens inside ``prepare_gpec_suite_case``.

    Skipping it would let the name reach ``SOLVERS[module]`` as a ``KeyError``
    instead of the suite's own message -- which is how ``modules=("pentrc",)``
    would have surfaced.
    """
    config = gpec.GPECSuiteConfig(modules=("pentrc",), modes=(1,))
    with pytest.raises(ValueError, match="Unsupported GPEC suite module"):
        gpec.run_gpec_suite_case(case, config, prepare=False)


def test_prepare_defaults_to_true(case, tmp_path):
    """So no existing caller changes behaviour."""
    config = _installed(tmp_path)
    result = gpec.run_gpec_suite_case(case, config)
    (record,) = result.records
    assert record.status != "skipped"
    assert _dcon_cell(case, config).is_dir()
