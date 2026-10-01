"""GPEC's resonant-field penetration thresholds, as options rather than a patched file.

Two thresholds, two models, two output columns, and they are reached by three
namelist keys GPEC ships in no template at all -- they exist as code defaults
(``gpec/gpec.f:159-162``).  The tests here pin the four things that made this
worth an option instead of a template edit:

1. ``singthresh_flag`` is a *shorthand*: GPEC forces both sub-flags true under
   it (``gpec/gpec.f:274-279``), so asking for it is asking for both.
2. Callen's threshold is built from the coil vacuum field, so it is refused
   without one rather than left to come back as a column of zeros.
3. The SLAYER Prandtl number is required when SLAYER runs and refused when it
   does not, because GPEC stamps it into the output as ``Pr`` either way.
4. A threshold run needs kinetic profiles *staged for the ideal-GPEC cell*,
   which is not the same thing as a kinetic DCON.

Every test writes namelists and stubs; none needs a GPEC installation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vaft.code import gpec
from vaft.code.gpec import IdealGPECOptions

from external_code_stubs import write_launchable_stub

GFILE_TEXT = "  EFITD   01/01/2024   #  39915  325ms        3  65  65\n 1.0 2.0 3.0\n"

#: The three keys this feature adds to the packaged ``gpec.in``.
NEW_KEYS = (
    "singthresh_callen_flag",
    "singthresh_slayer_flag",
    "singthresh_slayer_inpr",
)


def _namelist(path: Path) -> dict[str, str]:
    from vaft.code.gpec._runtime import read_namelist_group

    return read_namelist_group(path, "gpec_output")


@pytest.fixture()
def case(tmp_path):
    geqdsk = tmp_path / "g039915.00325"
    geqdsk.write_text(GFILE_TEXT, encoding="utf-8")
    return gpec.GPECCaseInputs(
        shot=39915, time_ms=325, geqdsk=geqdsk, workdir=tmp_path / "run"
    )


def _prepared_gpec_cell(case, options: IdealGPECOptions, mode: int = 1) -> Path:
    config = gpec.GPECSuiteConfig(modules=("gpec",), modes=(mode,), gpec=options)
    gpec.prepare_gpec_suite_case(case, config)
    return gpec._module_dir(case.workdir, case.time_ms, "gpec", mode, geqdsk=case.geqdsk)


# --------------------------------------------------------------------------
# The packaged template
# --------------------------------------------------------------------------


def test_the_packaged_template_carries_every_threshold_key():
    """Without these three lines the options have nothing to patch.

    ``write_template`` refuses a key the template does not have, by design, so
    the template is the enabling half of this feature rather than a convenience.
    """
    from vaft.code.gpec._runtime import package_vest_dir

    text = (package_vest_dir() / "gpec.in").read_text(encoding="utf-8")
    for key in ("singthresh_flag", *NEW_KEYS):
        assert f"{key}=" in text, key


def test_the_packaged_template_leaves_every_threshold_off():
    """A case that asks for nothing computes no threshold.

    Stated as a test because the alternative -- inheriting a template that
    happens to have one on -- is how the DIII-D ideal example comes with
    ``singthresh_flag=t``.
    """
    from vaft.code.gpec._runtime import package_vest_dir

    values = _namelist(package_vest_dir() / "gpec.in")
    assert values["singthresh_flag"] == "f"
    assert values["singthresh_callen_flag"] == "f"
    assert values["singthresh_slayer_flag"] == "f"


# --------------------------------------------------------------------------
# What reaches gpec.in
# --------------------------------------------------------------------------


def test_the_default_case_writes_every_threshold_off(case):
    cell = _prepared_gpec_cell(case, IdealGPECOptions())
    values = _namelist(cell / "gpec.in")
    assert values["singthresh_flag"] == "f"


def test_a_default_case_does_not_touch_the_two_sub_flags(case):
    """They are patched only when a threshold is asked for.

    A caller's own ``templates_dir`` may legitimately not carry them -- GPEC
    ships them in no namelist -- so patching unconditionally would refuse a
    template that worked before this feature existed.  The test asserts the
    consequence on a template that *is* missing them.
    """
    from vaft.code.gpec._runtime import package_vest_dir

    templates = case.workdir.parent / "templates"
    templates.mkdir(parents=True, exist_ok=True)
    for source in package_vest_dir().glob("*"):
        if source.is_file():
            (templates / source.name).write_bytes(source.read_bytes())
    stripped = "\n".join(
        line
        for line in (templates / "gpec.in").read_text(encoding="utf-8").splitlines()
        if not any(line.strip().startswith(key) for key in NEW_KEYS)
    )
    (templates / "gpec.in").write_text(stripped + "\n", encoding="utf-8")

    config = gpec.GPECSuiteConfig(
        modules=("gpec",), modes=(1,), templates_dir=templates, gpec=IdealGPECOptions()
    )
    gpec.prepare_gpec_suite_case(case, config)  # does not raise
    cell = gpec._module_dir(case.workdir, case.time_ms, "gpec", 1, geqdsk=case.geqdsk)
    assert "singthresh_callen_flag" not in (cell / "gpec.in").read_text(encoding="utf-8")


def test_singthresh_flag_writes_both_sub_flags_as_gpec_would_force_them(case):
    """The shorthand is expanded here, so the namelist says what GPEC will do.

    GPEC prints "setting callen flag and slayer flag to true" and does it
    (``gpec/gpec.f:274-279``).  Writing ``singthresh_callen_flag=f`` beside
    ``singthresh_flag=t`` would be a namelist that contradicts the run.
    """
    cell = _prepared_gpec_cell(
        case, IdealGPECOptions(singthresh_flag=True, singthresh_slayer_inpr=7.287)
    )
    values = _namelist(cell / "gpec.in")
    assert values["singthresh_flag"] == "t"
    assert values["singthresh_callen_flag"] == "t"
    assert values["singthresh_slayer_flag"] == "t"
    assert float(values["singthresh_slayer_inpr"]) == pytest.approx(7.287)


def test_callen_alone_leaves_slayer_off_and_needs_no_prandtl_number(case):
    cell = _prepared_gpec_cell(case, IdealGPECOptions(singthresh_callen_flag=True))
    values = _namelist(cell / "gpec.in")
    assert values["singthresh_flag"] == "f"
    assert values["singthresh_callen_flag"] == "t"
    assert values["singthresh_slayer_flag"] == "f"
    # Untouched, so it keeps GPEC's own default -- which SLAYER never reads here.
    assert float(values["singthresh_slayer_inpr"]) == pytest.approx(5.0)


def test_slayer_alone_leaves_callen_off(case):
    cell = _prepared_gpec_cell(
        case, IdealGPECOptions(singthresh_slayer_flag=True, singthresh_slayer_inpr=5.0)
    )
    values = _namelist(cell / "gpec.in")
    assert values["singthresh_callen_flag"] == "f"
    assert values["singthresh_slayer_flag"] == "t"


def test_an_options_change_alone_moves_the_prandtl_number(case):
    """The number in the file is the number that was asked for, to its digits."""
    cell = _prepared_gpec_cell(
        case, IdealGPECOptions(singthresh_slayer_flag=True, singthresh_slayer_inpr=7.28701557)
    )
    assert float(_namelist(cell / "gpec.in")["singthresh_slayer_inpr"]) == pytest.approx(
        7.28701557, abs=0.0
    )


# --------------------------------------------------------------------------
# What is refused, and where
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "options",
    [
        {"singthresh_flag": True, "coil_flag": False},
        {"singthresh_callen_flag": True, "coil_flag": False},
    ],
)
def test_callen_without_a_coil_is_refused_at_construction(options):
    if options.get("singthresh_flag"):
        options["singthresh_slayer_inpr"] = 5.0
    with pytest.raises(ValueError, match="coil vacuum field"):
        IdealGPECOptions(**options)


def test_slayer_without_a_coil_is_allowed():
    """Only Callen's model divides by the coil vacuum flux.

    Asserted rather than left implicit, because the two thresholds are easy to
    treat as one feature: SLAYER's layer calculation takes the local profiles
    and the resonant field, and GPEC gates only the Callen branch on
    ``coil_flag`` (``gpec/gpec.f:280-291``).
    """
    options = IdealGPECOptions(
        coil_flag=False, singthresh_slayer_flag=True, singthresh_slayer_inpr=5.0
    )
    assert options.wants_slayer_threshold
    assert not options.wants_callen_threshold


def test_slayer_without_a_prandtl_number_is_refused():
    with pytest.raises(ValueError, match="singthresh_slayer_inpr"):
        IdealGPECOptions(singthresh_slayer_flag=True)


def test_a_prandtl_number_without_slayer_is_refused():
    """Because GPEC writes it into the output whatever the flags say.

    ``gpec/gpout.f:1851-1852`` puts ``Pr`` on the profile file as a global
    attribute from inside the ``msing>0`` block, with no threshold flag in
    sight -- so a value set on a run that computed nothing would sit in the
    result looking like the Prandtl number one was computed at.
    """
    with pytest.raises(ValueError, match="no SLAYER threshold is requested"):
        IdealGPECOptions(singthresh_slayer_inpr=5.0)


def test_the_three_wants_properties_agree_with_gpecs_own_forcing():
    assert IdealGPECOptions().wants_any_threshold is False
    both = IdealGPECOptions(singthresh_flag=True, singthresh_slayer_inpr=5.0)
    assert (both.wants_callen_threshold, both.wants_slayer_threshold) == (True, True)
    callen = IdealGPECOptions(singthresh_callen_flag=True)
    assert (callen.wants_callen_threshold, callen.wants_slayer_threshold) == (True, False)


# --------------------------------------------------------------------------
# The kinetic profiles a threshold needs
# --------------------------------------------------------------------------


def test_a_directory_with_no_pentrc_in_cannot_compute_a_threshold(tmp_path):
    problems = gpec.validate_threshold_inputs(tmp_path)
    assert len(problems) == 1
    assert "pentrc.in" in problems[0]


def test_a_pentrc_in_naming_a_missing_kin_is_reported_with_both_names(tmp_path):
    (tmp_path / "pentrc.in").write_text(
        '&PENT_INPUT\n    kinetic_file = "gone.kin"\n/\n', encoding="utf-8"
    )
    problems = gpec.validate_threshold_inputs(tmp_path)
    assert len(problems) == 1
    assert "gone.kin" in problems[0]
    assert str(tmp_path / "gone.kin") in problems[0]


def test_a_pentrc_in_naming_no_kin_at_all_is_reported(tmp_path):
    (tmp_path / "pentrc.in").write_text(
        '&PENT_INPUT\n    kinetic_file = ""\n/\n', encoding="utf-8"
    )
    assert "names no kinetic_file" in gpec.validate_threshold_inputs(tmp_path)[0]


def test_a_staged_pair_is_accepted(tmp_path):
    (tmp_path / "pentrc.in").write_text(
        '&PENT_INPUT\n    kinetic_file = "p.kin"\n/\n', encoding="utf-8"
    )
    (tmp_path / "p.kin").write_text("psi ne\n0 1\n", encoding="utf-8")
    assert gpec.validate_threshold_inputs(tmp_path) == []


def test_kinetic_profiles_are_not_the_same_thing_as_a_kinetic_dcon(tmp_path):
    """The prerequisite is the two files, and nothing asks about ``kin_flag``.

    GPEC's own ``docs/examples/DIIID_ideal_example`` runs ``kin_flag=f`` with
    ``singthresh_flag=t``, a ``.kin`` and a ``pentrc.in``: the profiles feed the
    threshold models, not the Euler-Lagrange equation.  A check that keyed on a
    kinetic DCON would refuse that example.
    """
    (tmp_path / "pentrc.in").write_text(
        '&PENT_INPUT\n    kinetic_file = "p.kin"\n/\n', encoding="utf-8"
    )
    (tmp_path / "p.kin").write_text("psi ne\n0 1\n", encoding="utf-8")
    (tmp_path / "dcon.in").write_text("&DCON_CONTROL\n    kin_flag=f\n/\n", encoding="utf-8")
    assert gpec.validate_threshold_inputs(tmp_path) == []


# --------------------------------------------------------------------------
# The check happens before GPEC is launched
# --------------------------------------------------------------------------


def _installation(tmp_path, *, programs=("dcon", "gpec")) -> Path:
    home = tmp_path / "gpec_home"
    for program in programs:
        write_launchable_stub(home / "bin" / program)
    return home


def _complete_dcon(case, mode: int = 1) -> Path:
    cell = gpec._module_dir(case.workdir, case.time_ms, "dcon", mode, geqdsk=case.geqdsk)
    cell.mkdir(parents=True, exist_ok=True)
    (cell / "euler.bin").write_bytes(b"euler")
    (cell / "psi_in.bin").write_bytes(b"psi")
    # Staged into the GPEC cell with the binaries, because GPEC opens `equil.in`
    # in its own working directory; see test_gpec_dcon_staging.py.
    (cell / "equil.in").write_text(
        f"&EQUIL_CONTROL\n    eq_filename=\"{case.geqdsk.name}\"\n/\n", encoding="utf-8"
    )
    (cell / case.geqdsk.name).write_text(GFILE_TEXT, encoding="utf-8")
    return cell


def test_a_threshold_run_without_profiles_is_skipped_rather_than_solved(case, tmp_path):
    """An hour of solve followed by a column of exact zeros is the failure avoided.

    Zero is also what a threshold larger than any attainable field looks like,
    so a run that silently produced one would be indistinguishable from a
    physics result.
    """
    options = IdealGPECOptions(singthresh_flag=True, singthresh_slayer_inpr=5.0)
    config = gpec.GPECSuiteConfig(
        modules=("gpec",),
        modes=(1,),
        gpec=options,
        gpec_home=_installation(tmp_path),
    )
    gpec.prepare_gpec_suite_case(case, config)
    _complete_dcon(case)
    result = gpec.run_gpec_suite_case(case, config)
    (record,) = result.records
    assert record.status == "skipped"
    assert "penetration threshold was requested" in record.reason
    assert "exact zeros" in record.reason


def test_strict_raises_instead_of_skipping(case, tmp_path):
    options = IdealGPECOptions(singthresh_callen_flag=True)
    config = gpec.GPECSuiteConfig(
        modules=("gpec",),
        modes=(1,),
        gpec=options,
        gpec_home=_installation(tmp_path),
        run_mode="strict",
    )
    gpec.prepare_gpec_suite_case(case, config)
    _complete_dcon(case)
    with pytest.raises(RuntimeError, match="penetration threshold"):
        gpec.run_gpec_suite_case(case, config)


def test_a_case_that_asked_for_no_threshold_is_not_checked(case, tmp_path):
    """The check is scoped to runs that requested one.

    Otherwise every ideal case in the suite would start demanding a
    ``pentrc.in`` it has no use for.
    """
    config = gpec.GPECSuiteConfig(
        modules=("gpec",),
        modes=(1,),
        gpec=IdealGPECOptions(),
        gpec_home=_installation(tmp_path),
    )
    gpec.prepare_gpec_suite_case(case, config)
    _complete_dcon(case)
    result = gpec.run_gpec_suite_case(case, config)
    (record,) = result.records
    assert record.status != "skipped"
    assert "pentrc.in" not in record.reason
