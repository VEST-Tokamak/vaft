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

import shutil
from pathlib import Path

import pytest

from vaft.code import gpec
from vaft.code.gpec import IdealGPECOptions

from external_code_stubs import write_launchable_stub

#: An inverse Prandtl number with enough digits to catch a reformat.
#:
#: Arbitrary, and deliberately so: a value carried over from a real discharge
#: would make this fixture a measurement of that discharge.
AWKWARD_INPR = 3.14159265


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
        case, IdealGPECOptions(singthresh_flag=True, singthresh_slayer_inpr=AWKWARD_INPR)
    )
    values = _namelist(cell / "gpec.in")
    assert values["singthresh_flag"] == "t"
    assert values["singthresh_callen_flag"] == "t"
    assert values["singthresh_slayer_flag"] == "t"
    assert float(values["singthresh_slayer_inpr"]) == pytest.approx(AWKWARD_INPR)


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
        case, IdealGPECOptions(singthresh_slayer_flag=True, singthresh_slayer_inpr=AWKWARD_INPR)
    )
    assert float(_namelist(cell / "gpec.in")["singthresh_slayer_inpr"]) == pytest.approx(
        AWKWARD_INPR, abs=0.0
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


#: An ideal DCON namelist, reduced to the two keys the check reads.
#:
#: ``con_flag=f`` is **not** the packaged default; the packaged ``dcon.in`` and
#: GPEC's own ``input/dcon.in`` both ship ``t``, which is why a caller who asks
#: for nothing cannot get a threshold.
IDEAL_DCON_IN = "&DCON_CONTROL\n    con_flag=f\n    kin_flag=f\n/\n"


def _profiles(directory, *, kinetic_file: str = "p.kin", present: bool = True) -> None:
    (directory / "pentrc.in").write_text(
        f'&PENT_INPUT\n    kinetic_file = "{kinetic_file}"\n/\n', encoding="utf-8"
    )
    if present and kinetic_file:
        (directory / kinetic_file).write_text("psi ne\n0 1\n", encoding="utf-8")


def test_a_directory_with_no_pentrc_in_cannot_compute_a_threshold(tmp_path):
    (tmp_path / "dcon.in").write_text(IDEAL_DCON_IN, encoding="utf-8")
    problems = gpec.validate_threshold_inputs(tmp_path)
    assert len(problems) == 1
    assert "pentrc.in" in problems[0]


def test_a_pentrc_in_naming_a_missing_kin_is_reported_with_both_names(tmp_path):
    (tmp_path / "dcon.in").write_text(IDEAL_DCON_IN, encoding="utf-8")
    _profiles(tmp_path, kinetic_file="gone.kin", present=False)
    problems = gpec.validate_threshold_inputs(tmp_path)
    assert len(problems) == 1
    assert "gone.kin" in problems[0]
    assert str(tmp_path / "gone.kin") in problems[0]


def test_a_pentrc_in_naming_no_kin_at_all_is_reported(tmp_path):
    (tmp_path / "dcon.in").write_text(IDEAL_DCON_IN, encoding="utf-8")
    _profiles(tmp_path, kinetic_file="")
    assert "names no kinetic_file" in gpec.validate_threshold_inputs(tmp_path)[0]


def test_a_staged_pair_beside_an_ideal_dcon_is_accepted(tmp_path):
    """GPEC's own single-directory layout: ``dcon_dir`` defaults to ``run_dir``."""
    (tmp_path / "dcon.in").write_text(IDEAL_DCON_IN, encoding="utf-8")
    _profiles(tmp_path)
    assert gpec.validate_threshold_inputs(tmp_path) == []


def test_the_dcon_namelist_can_live_in_another_cell(tmp_path):
    """Which is this adapter's own layout, where ``dcon.in`` is never staged over."""
    gpec_cell, dcon_cell = tmp_path / "gpec", tmp_path / "dcon"
    gpec_cell.mkdir()
    dcon_cell.mkdir()
    _profiles(gpec_cell)
    (dcon_cell / "dcon.in").write_text(IDEAL_DCON_IN, encoding="utf-8")
    assert gpec.validate_threshold_inputs(gpec_cell, dcon_cell) == []
    # And without it the keys are reported unchecked rather than passed over.
    unchecked = gpec.validate_threshold_inputs(gpec_cell)
    assert len(unchecked) == 1
    assert "could not be checked" in unchecked[0]


def test_kinetic_profiles_are_not_the_same_thing_as_a_kinetic_dcon(tmp_path):
    """The profiles are the prerequisite; a kinetic DCON is a *blocker*.

    GPEC's own ``docs/examples/DIIID_ideal_example`` runs ``kin_flag=f`` with
    ``singthresh_flag=t``, a ``.kin`` and a ``pentrc.in``: the profiles feed the
    threshold models, not the Euler-Lagrange equation. The first version of this
    check read that as "``kin_flag`` is irrelevant", which is the opposite of
    true -- with ``kin_flag=t`` GPEC counts ``msing = 0`` and writes no
    rational-surface block at all, so there is nothing for a per-surface
    threshold to attach to.
    """
    _profiles(tmp_path)
    (tmp_path / "dcon.in").write_text(
        "&DCON_CONTROL\n    con_flag=f\n    kin_flag=t\n/\n", encoding="utf-8"
    )
    (problem,) = gpec.validate_threshold_inputs(tmp_path)
    assert "kin_flag=t" in problem
    assert "msing=0" in problem


def test_the_packaged_con_flag_is_itself_a_blocker(tmp_path):
    """The default path, and the reason this check exists at all.

    ``con_flag=t`` is what the packaged ``dcon.in`` ships, so a caller who sets
    ``singthresh_flag`` and nothing else gets a run GPEC refuses to compute a
    threshold in -- and the refusal is a warning in the log followed by a column
    of exact zeros, which is also what an unattainable threshold looks like.
    """
    _profiles(tmp_path)
    (tmp_path / "dcon.in").write_text(
        "&DCON_CONTROL\n    con_flag=t\n    kin_flag=f\n/\n", encoding="utf-8"
    )
    (problem,) = gpec.validate_threshold_inputs(tmp_path)
    assert "con_flag=t" in problem
    assert "DCONOptions(con_flag=False)" in problem


def test_the_packaged_dcon_namelist_still_ships_the_blocking_value(tmp_path):
    """Read off the template, not asserted from memory.

    If a future template drops ``con_flag`` or ships ``f``, the two tests above
    stop describing the default and this one says so.
    """
    from vaft.code.gpec._runtime import package_vest_dir, read_namelist_group

    control = read_namelist_group(package_vest_dir() / "dcon.in", "dcon_control")
    assert control["con_flag"] == "t"


def test_a_threshold_run_can_turn_con_flag_off(case, tmp_path):
    """And the prepared namelist says so, which is what the check then reads."""
    config = gpec.GPECSuiteConfig(
        modules=("dcon",), modes=(1,),
        dcon=gpec.DCONOptions(con_flag=False),
        gpec_home=tmp_path / "gpec_home",
    )
    gpec.prepare_gpec_suite_case(case, config)
    cell = gpec._module_dir(case.workdir, case.time_ms, "dcon", 1, geqdsk=case.geqdsk)
    from vaft.code.gpec._runtime import read_namelist_group

    assert read_namelist_group(cell / "dcon.in", "dcon_control")["con_flag"] == "f"


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


# --------------------------------------------------------------------------
# PENTRC's input, and the per-surface Prandtl profile
# --------------------------------------------------------------------------


def _prepared_gpec_in(case, tmp_path, **options):
    config = gpec.GPECSuiteConfig(
        modules=("gpec",), modes=(1,), gpec_home=tmp_path / "gpec_home",
        gpec=gpec.IdealGPECOptions(**options),
    )
    gpec.prepare_gpec_suite_case(case, config)
    cell = gpec._module_dir(case.workdir, case.time_ms, "gpec", 1, geqdsk=case.geqdsk)
    from vaft.code.gpec._runtime import read_namelist_group

    return read_namelist_group(cell / "gpec.in", "gpec_output"), cell


def test_the_packaged_run_writes_nothing_pentrc_can_read(case, tmp_path):
    """Which is why ``ascii_flag`` had to be exposed at all.

    PENTRC reads the ASCII ``gpec_xclebsch_n<n>.out``; GPEC writes that file only
    with ``ascii_flag`` *and* ``xclebsch_flag`` on, and the packaged template
    ships the second without the first. So the default run computes the
    displacement, writes it to netCDF, and leaves the torque unreachable.
    """
    assert gpec.IdealGPECOptions().writes_pentrc_input is False
    written, _ = _prepared_gpec_in(case, tmp_path)
    assert (written["ascii_flag"], written["xclebsch_flag"]) == ("f", "t")


def test_asking_for_pentrcs_input_writes_both_flags(case, tmp_path):
    options = gpec.IdealGPECOptions(ascii_flag=True)
    assert options.writes_pentrc_input is True
    written, _ = _prepared_gpec_in(case, tmp_path, ascii_flag=True)
    assert (written["ascii_flag"], written["xclebsch_flag"]) == ("t", "t")


def test_the_displacement_can_be_turned_off_without_turning_ascii_off(case, tmp_path):
    """Both directions, because either flag alone writes nothing PENTRC opens."""
    options = gpec.IdealGPECOptions(ascii_flag=True, xclebsch_flag=False)
    assert options.writes_pentrc_input is False
    written, _ = _prepared_gpec_in(case, tmp_path, ascii_flag=True, xclebsch_flag=False)
    assert (written["ascii_flag"], written["xclebsch_flag"]) == ("t", "f")


def test_a_per_surface_prandtl_profile_needs_a_template_that_declares_it(case, tmp_path):
    """The packaged one does not, and that is deliberate.

    Only GPEC from ``28f6df64`` onwards declares
    ``singthresh_slayer_inpr_prof``, and a Fortran namelist READ stops the program
    on a name it does not declare whatever the flags say -- so shipping the key
    would break every run on an older build. The refusal names the way out.
    """
    config = gpec.GPECSuiteConfig(
        modules=("gpec",), modes=(1,), gpec_home=tmp_path / "gpec_home",
        gpec=gpec.IdealGPECOptions(
            singthresh_slayer_flag=True,
            singthresh_slayer_inpr=5.0,
            singthresh_slayer_inpr_prof=(3.0, 5.0, 8.0),
        ),
    )
    with pytest.raises(ValueError, match="does not declare"):
        gpec.prepare_gpec_suite_case(case, config)


def test_a_template_that_declares_it_gets_the_profile_comma_separated(case, tmp_path):
    """Written the way GPEC reads a list: one assignment, comma separated."""
    templates = tmp_path / "templates"
    templates.mkdir()
    packaged = gpec._runtime.package_vest_dir()
    for name in ("gpec.in", "coil.in", "equil.in", "dcon.in", "vac.in", "match.in"):
        shutil.copy2(packaged / name, templates / name)
    patched = (templates / "gpec.in").read_text(encoding="utf-8").replace(
        "   singthresh_slayer_inpr=5.0",
        "   singthresh_slayer_inpr=5.0\n   singthresh_slayer_inpr_prof=-1.0",
        1,
    )
    (templates / "gpec.in").write_text(patched, encoding="utf-8")

    config = gpec.GPECSuiteConfig(
        modules=("gpec",), modes=(1,), gpec_home=tmp_path / "gpec_home",
        templates_dir=templates,
        gpec=gpec.IdealGPECOptions(
            singthresh_slayer_flag=True,
            singthresh_slayer_inpr=5.0,
            singthresh_slayer_inpr_prof=(3.0, 5.0, 8.0),
        ),
    )
    gpec.prepare_gpec_suite_case(case, config)
    cell = gpec._module_dir(case.workdir, case.time_ms, "gpec", 1, geqdsk=case.geqdsk)
    line = [
        text for text in (cell / "gpec.in").read_text(encoding="utf-8").splitlines()
        if "singthresh_slayer_inpr_prof" in text
    ]
    assert line and "3.0, 5.0, 8.0" in line[0]


def test_a_template_without_singthresh_flag_still_prepares(case, tmp_path):
    """A caller's own ``gpec.in`` is entitled not to carry a key GPEC defaults.

    Restating ``singthresh_flag=f`` on a run that asked for no threshold must not
    refuse such a template -- but a template shipping ``t`` must still be
    overridden, or a threshold would run behind a caller who never asked.
    """
    templates = tmp_path / "templates"
    templates.mkdir()
    packaged = gpec._runtime.package_vest_dir()
    for name in ("gpec.in", "coil.in", "equil.in", "dcon.in", "vac.in", "match.in"):
        shutil.copy2(packaged / name, templates / name)
    stripped = "\n".join(
        line for line in (templates / "gpec.in").read_text(encoding="utf-8").splitlines()
        if "singthresh" not in line
    )
    (templates / "gpec.in").write_text(stripped + "\n", encoding="utf-8")

    config = gpec.GPECSuiteConfig(
        modules=("gpec",), modes=(1,), gpec_home=tmp_path / "gpec_home",
        templates_dir=templates,
    )
    gpec.prepare_gpec_suite_case(case, config)  # no KeyError
    cell = gpec._module_dir(case.workdir, case.time_ms, "gpec", 1, geqdsk=case.geqdsk)
    assert "singthresh" not in (cell / "gpec.in").read_text(encoding="utf-8")


def test_a_template_shipping_singthresh_flag_on_is_turned_off_when_unasked(case, tmp_path):
    templates = tmp_path / "templates"
    templates.mkdir()
    packaged = gpec._runtime.package_vest_dir()
    for name in ("gpec.in", "coil.in", "equil.in", "dcon.in", "vac.in", "match.in"):
        shutil.copy2(packaged / name, templates / name)
    enabled = (templates / "gpec.in").read_text(encoding="utf-8").replace(
        "singthresh_flag=f", "singthresh_flag=t", 1
    )
    (templates / "gpec.in").write_text(enabled, encoding="utf-8")

    config = gpec.GPECSuiteConfig(
        modules=("gpec",), modes=(1,), gpec_home=tmp_path / "gpec_home",
        templates_dir=templates,
    )
    gpec.prepare_gpec_suite_case(case, config)
    cell = gpec._module_dir(case.workdir, case.time_ms, "gpec", 1, geqdsk=case.geqdsk)
    from vaft.code.gpec._runtime import read_namelist_group

    assert read_namelist_group(cell / "gpec.in", "gpec_output")["singthresh_flag"] == "f"


# --------------------------------------------------------------------------
# The pentrc.in GPEC's own threshold models read
# --------------------------------------------------------------------------


def _pentrc_options():
    return gpec.PENTRCOptions(
        methods=("fgar", "tgar"),
        main_ion="deuterium",
        impurity="carbon",
        collision_operator="harmonic",
    )


def test_the_threshold_input_can_be_written_before_gpec_runs(tmp_path):
    """Which is the point: no displacement exists yet.

    GPEC reads this file itself when a threshold flag is on, so it has to be
    there *before* the run -- and ``prepare_pentrc_run`` cannot write it, because
    it derives ``jac_in`` and ``tmag_in`` from products the run has not made.
    """
    cell = tmp_path / "cell"
    cell.mkdir()
    kin = tmp_path / "profiles.kin"
    kin.write_text("psi n_e\n0.1 1e19\n", encoding="utf-8")

    written = gpec.write_threshold_pentrc_input(
        cell, options=_pentrc_options(), kinetic_file=kin
    )
    assert written == cell / "pentrc.in"
    assert not list(cell.glob("gpec_xclebsch_*"))
    assert gpec.validate_threshold_inputs(cell, cell) == [
        reason
        for reason in gpec.validate_threshold_inputs(cell, cell)
        if "dcon.in" in reason
    ], "the profiles half of the check should be satisfied"


def test_the_threshold_input_carries_the_species_the_threshold_is_built_from(tmp_path):
    """Callen's critical width is a gyroradius, so ``mi`` and ``zi`` are in it."""
    from vaft.code.gpec._runtime import read_namelist_group

    cell = tmp_path / "cell"
    cell.mkdir()
    kin = tmp_path / "profiles.kin"
    kin.write_text("psi n_e\n0.1 1e19\n", encoding="utf-8")
    gpec.write_threshold_pentrc_input(
        cell, options=_pentrc_options(), kinetic_file=kin
    )
    pent = read_namelist_group(cell / "pentrc.in", "pent_input")
    assert (pent["mi"], pent["zi"], pent["mimp"], pent["zimp"]) == ("2", "1", "12", "6")
    assert pent["kinetic_file"] == "profiles.kin"
    assert (cell / "profiles.kin").is_file()


def test_the_threshold_input_claims_no_torque_method(tmp_path):
    """GPEC's threshold path runs none of them, so none is written on."""
    from vaft.code.pentrc import TORQUE_METHODS
    from vaft.code.gpec._runtime import read_namelist_group

    cell = tmp_path / "cell"
    cell.mkdir()
    kin = tmp_path / "profiles.kin"
    kin.write_text("psi n_e\n0.1 1e19\n", encoding="utf-8")
    gpec.write_threshold_pentrc_input(
        cell, options=_pentrc_options(), kinetic_file=kin
    )
    methods = read_namelist_group(cell / "pentrc.in", "pent_output")
    assert {methods[f"{method}_flag"] for method in TORQUE_METHODS} == {"f"}


def test_the_threshold_input_survives_the_torque_run_that_overwrites_it(tmp_path):
    """Same file name, different contents, and the first one is the evidence.

    Without the copy, a case that computed a threshold and then a torque has no
    record of which species or which profiles the *threshold* used.
    """
    from vaft.code.gpec._runtime import read_namelist_group

    cell = tmp_path / "00325" / "gpec" / "nn=1"
    cell.mkdir(parents=True)
    (cell / "euler.bin").write_bytes(b"euler")
    (cell / "gpec.in").write_text(
        "&GPEC_OUTPUT\n    tmag_out=1\n/\n", encoding="utf-8"
    )
    (cell / "gpec_xclebsch_n1.out").write_text(
        " GPEC_XCLEBSCH\n v1\n\n    jac_out = hamada  \n\n", encoding="utf-8"
    )
    kin = tmp_path / "profiles.kin"
    kin.write_text("psi n_e\n0.1 1e19\n", encoding="utf-8")

    gpec.write_threshold_pentrc_input(cell, options=_pentrc_options(), kinetic_file=kin)
    threshold_copy = (cell / gpec.THRESHOLD_PENTRC_INPUT).read_bytes()

    gpec.prepare_pentrc_run(cell, mode=1, options=_pentrc_options(), kinetic_file=kin)
    torque = read_namelist_group(cell / "pentrc.in", "pent_output")
    assert torque["fgar_flag"] == "t", "the torque run asked for fgar"
    # The threshold input is still there, and so is a copy of what was replaced.
    assert (cell / gpec.THRESHOLD_PENTRC_INPUT).read_bytes() == threshold_copy
    assert (cell / gpec.PRIOR_PENTRC_INPUT).read_bytes() == threshold_copy
    assert read_namelist_group(cell / gpec.THRESHOLD_PENTRC_INPUT, "pent_output")[
        "fgar_flag"
    ] == "f"


def test_a_second_torque_run_does_not_overwrite_the_first_record(tmp_path):
    cell = tmp_path / "00325" / "gpec" / "nn=1"
    cell.mkdir(parents=True)
    (cell / "euler.bin").write_bytes(b"euler")
    (cell / "gpec.in").write_text("&GPEC_OUTPUT\n    tmag_out=1\n/\n", encoding="utf-8")
    (cell / "gpec_xclebsch_n1.out").write_text(
        " GPEC_XCLEBSCH\n v1\n\n    jac_out = hamada  \n\n", encoding="utf-8"
    )
    kin = tmp_path / "profiles.kin"
    kin.write_text("psi n_e\n0.1 1e19\n", encoding="utf-8")

    gpec.write_threshold_pentrc_input(cell, options=_pentrc_options(), kinetic_file=kin)
    first = (cell / "pentrc.in").read_bytes()
    gpec.prepare_pentrc_run(cell, mode=1, options=_pentrc_options(), kinetic_file=kin)
    gpec.prepare_pentrc_run(
        cell, mode=1,
        options=gpec.PENTRCOptions(
            methods=("tgar",), main_ion="deuterium", impurity="carbon",
            collision_operator="harmonic",
        ),
        kinetic_file=kin,
    )
    assert (cell / gpec.PRIOR_PENTRC_INPUT).read_bytes() == first
