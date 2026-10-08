"""Named EFIT configurations (#891 working setting) and their pipeline wiring."""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import replace
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest
from omas import save_omas_json

from test_efit_config import _constraints_ods
from vaft.code.efit import PRESETS, EFITScientificConfig, apply_sigma_floor, efit_preset, generate_kfile
from vaft.code.efit.config import routine_profile_config, routine_scientific_config
from vaft.code.efit.presets import DEFAULT_PRESET, PRESET_RECORD, PSI_ONLY_SAICON

#: k-files written on develop before the 2026-10-01 default change, from the
#: `_constraints_ods` fixture with INPUT_DIR=/GOLDEN_INPUT_DIR: the routine
#: default and the statistical_891 preset of #1339.
GOLDEN = Path(__file__).resolve().parent / "data" / "efit_kfile_golden"

REPOSITORY = Path(__file__).resolve().parents[1]
PIPELINE1 = REPOSITORY / "workflow" / "automatic_pipeline_1_routine_data_processing"


def _kfile(tmp_path, ods, **kwargs):
    generate_kfile(ods, 39915, save_dir=str(tmp_path), **kwargs)
    return next((tmp_path / "kfile").iterdir()).read_text(encoding="utf-8")


def _key(text, name):
    match = re.search(rf"(?m)^\s*{name}\s*=\s*([^,\n]+)", text)
    assert match, f"{name} not in the k-file"
    return match.group(1).strip()


def _golden_kfile(tmp_path, **kwargs):
    # PurePosixPath: the golden INPUT_DIR is written with a forward slash on every OS.
    generate_kfile(_constraints_ods(PurePosixPath("/GOLDEN_INPUT_DIR")), 39915, save_dir=str(tmp_path), **kwargs)
    return next((tmp_path / "kfile").iterdir()).read_text(encoding="utf-8")


def test_the_package_exports_what_the_profiles_page_names():
    """`vaft.code.efit.DEFAULT_PRESET` and the routine builders, not only their modules."""
    import vaft.code.efit as efit

    assert efit.DEFAULT_PRESET is DEFAULT_PRESET
    assert efit.routine_scientific_config is routine_scientific_config
    for name in ("DEFAULT_PRESET", "routine_profile_config", "routine_numerics_config",
                 "routine_constraint_config", "routine_scientific_config", "preset_of"):
        assert name in efit.__all__


def test_the_default_is_the_working_setting_byte_for_byte(tmp_path):
    """2026-10-01: the defaults are statistical_891, whose k-file is unchanged from #1339."""
    assert DEFAULT_PRESET == "statistical_891"
    assert efit_preset(DEFAULT_PRESET).scientific == EFITScientificConfig()
    golden = (GOLDEN / "statistical_891.k").read_text(encoding="utf-8")
    assert _golden_kfile(tmp_path / "default") == golden
    assert _golden_kfile(tmp_path / "preset", config=efit_preset("statistical_891").scientific) == golden


def test_the_routine_preset_is_the_old_default_byte_for_byte(tmp_path):
    """The legacy configuration stays reachable exactly, by name."""
    routine = efit_preset("routine")
    assert routine.scientific == routine_scientific_config()
    assert routine.sigma_floor == 0.0
    golden = (GOLDEN / "routine.k").read_text(encoding="utf-8")
    assert _golden_kfile(tmp_path / "preset", config=routine.scientific) == golden
    assert _golden_kfile(tmp_path / "builder", config=routine_scientific_config(
        profile=routine_profile_config(kppcur=2, kffcur=2))) == golden


def test_a_positional_basis_swaps_only_the_basis_on_the_default(tmp_path, monkeypatch):
    """One spelling, one meaning: ``generate_kfile(ods, shot, 2, 2)`` is
    ``EFITConfig(npprime=2, nffprime=2)`` is the default with a (2,2) basis.

    It used to select the whole routine configuration (legacy weights, EFIT
    termination, no floor) while ``EFITConfig(npprime=..)`` swapped only the
    basis on the statistical default (cold review 0.8.0 delta-absorb-11b F1).
    """
    from vaft.code.efit import EFITConfig, EFITProfileConfig, kfile as kfile_module

    monkeypatch.setattr(kfile_module, "_POSITIONAL_BASIS_WARNED", False)
    with pytest.warns(DeprecationWarning, match="routine_scientific_config.*'routine'"):
        positional = _golden_kfile(tmp_path / "positional", npprime=2, nffprime=2)
    typed = _golden_kfile(tmp_path / "typed",
                          config=EFITScientificConfig(profile=EFITProfileConfig(kppcur=2, kffcur=2)))
    through_efit_config = _golden_kfile(tmp_path / "efit_config",
                                        config=EFITConfig(shot=39915, npprime=2, nffprime=2))
    assert positional == typed == through_efit_config
    # The statistical set, not the routine one: psi-only exit and SERROR 0 are in it.
    assert positional != (GOLDEN / "routine.k").read_text(encoding="utf-8")
    assert "SAICON" in positional and _key(positional, "KFFCUR") == "2"
    # The warning is one-time, and a single argument leaves the other at the default (KFFCUR 1).
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        one_sided = _golden_kfile(tmp_path / "one_sided", npprime=3)
    assert _key(one_sided, "KPPCUR") == "3" and _key(one_sided, "KFFCUR") == "1"


def test_the_writer_applies_the_configured_sigma_floor(tmp_path):
    """The floor is part of the configuration: no prepare step is needed for it."""
    ods = _constraints_ods(tmp_path)
    ods["equilibrium.time_slice.0.constraints.bpol_probe.0.measured_error_upper"] = 1e-9
    default = _kfile(tmp_path / "default", ods)
    unfloored = _kfile(tmp_path / "unfloored", ods, config=EFITScientificConfig(
        constraints=replace(EFITScientificConfig().constraints, sigma_floor=0.0)))
    assert default != unfloored


def test_statistical_891_writes_the_working_setting(tmp_path):
    preset = efit_preset("statistical_891")
    ods, _ = preset.prepare_constraints(_constraints_ods(tmp_path))
    text = _kfile(tmp_path, ods, config=preset.scientific)
    assert int(_key(text, "KPPCUR")) == 2 and int(_key(text, "KFFCUR")) == 1
    assert float(_key(text, "ERRMIN")) == pytest.approx(1.0e-4)
    assert float(_key(text, "SAICON")) == pytest.approx(PSI_ONLY_SAICON)
    assert int(_key(text, "NXITER")) == 1
    assert int(_key(text, "MXITER")) == -514
    # Statistical sigma: EFIT's own relative and bit floors are off.
    assert float(_key(text, "SERROR")) == 0.0 and float(_key(text, "VBIT")) == 1.0
    scales = preset.scientific.constraints.uncertainty_scales
    assert scales["bpol_probe"] == pytest.approx(1 / 3.62)
    assert scales["flux_loop"] == pytest.approx(1 / 2.15)
    assert scales["plasma_current"] == pytest.approx(0.25)
    assert scales["diamagnetic_flux"] == pytest.approx(1 / 16)
    assert scales["pf_current"] == 1.0
    assert preset.scientific.constraints.use_diamagnetic_flux  # fitted, not held inactive
    # The diamagnetic sigma is the stored one widened x16 (fixture: 0.0002 Wb -> mWb).
    assert float(_key(text, "SIGDLC")) == pytest.approx(0.0002 * 1000 * 16)


def test_the_sigma_floor_raises_small_sigma_to_two_percent_of_the_family_median(tmp_path):
    ods = _constraints_ods(tmp_path)
    root = "equilibrium.time_slice.0.constraints.bpol_probe"
    for j, (measured, error, weight) in enumerate(((0.01, 1e-5, 1.0), (0.03, 0.01, 1.0), (5.0, 1e-9, 0.0))):
        ods[f"{root}.{j}.measured"], ods[f"{root}.{j}.measured_error_upper"] = measured, error
        ods[f"{root}.{j}.weight"] = weight
    changes = apply_sigma_floor(ods, 0.02, ("bpol_probe",))
    floor = 0.02 * 0.02  # median |m| of the two weighted channels
    assert changes == [{"slice": 0, "family": "bpol_probe", "floor": pytest.approx(floor), "raised": 1, "fitted": 2}]
    assert ods[f"{root}.0.measured_error_upper"] == pytest.approx(floor)   # raised
    assert ods[f"{root}.1.measured_error_upper"] == pytest.approx(0.01)    # already above
    assert ods[f"{root}.2.measured_error_upper"] == pytest.approx(1e-9)    # unweighted: untouched


def test_preparing_constraints_leaves_the_product_alone(tmp_path):
    ods = _constraints_ods(tmp_path)
    before = float(ods["equilibrium.time_slice.0.constraints.bpol_probe.0.measured_error_upper"])
    efit_preset("statistical_891").prepare_constraints(ods)
    assert float(ods["equilibrium.time_slice.0.constraints.bpol_probe.0.measured_error_upper"]) == before


def test_an_unknown_preset_names_the_known_ones():
    with pytest.raises(ValueError, match="statistical_891"):
        efit_preset("statistical")
    assert set(PRESETS) >= {"routine", "statistical_891"}


def _run_generate_kfile(tmp_path, *extra):
    constraints = tmp_path / "constraints.json"
    save_omas_json(_constraints_ods(tmp_path), str(constraints))
    manifest = tmp_path / "efit" / "manifest" / "kfiles.txt"
    # Run the script as a path-run would, but importing this checkout's vaft:
    # an editable install of another checkout sits on sys.meta_path, ahead of
    # sys.path, so its finder is dropped as well as the path put first.
    shim = ("import runpy, sys; "
            "sys.meta_path[:] = [f for f in sys.meta_path if '__editable__' not in type(f).__module__]; "
            "sys.path.insert(0, sys.argv[1]); import vaft; "
            "assert vaft.__file__.startswith(sys.argv[1]), vaft.__file__; "
            "script = sys.argv[2]; sys.argv = sys.argv[2:]; runpy.run_path(script, run_name='__main__')")
    return manifest, subprocess.run(
        [sys.executable, "-c", shim, str(REPOSITORY), str(PIPELINE1 / "generate_kfile.py"), "--shot", "39915",
         "--constraints-ods", str(constraints), "--output", str(manifest), *extra],
        capture_output=True, text=True, cwd=REPOSITORY,
    )


def test_generate_kfile_records_the_preset_beside_its_manifest(tmp_path):
    manifest, result = _run_generate_kfile(tmp_path, "--preset", "statistical_891")
    assert result.returncode == 0, result.stderr[-2000:]
    record = json.loads((manifest.parent / PRESET_RECORD).read_text())
    assert record["name"] == "statistical_891"
    assert record["scientific_sha256"] == efit_preset("statistical_891").scientific.sha256
    kfile = Path(manifest.read_text().split()[0]).read_text()
    assert int(_key(kfile, "KFFCUR")) == 1


def test_generate_kfile_refuses_a_preset_with_a_basis_override(tmp_path):
    _, result = _run_generate_kfile(tmp_path, "--preset", "statistical_891", "--npprime", "2")
    assert result.returncode != 0 and "--preset carries its own" in result.stderr


def test_a_legacy_basis_run_removes_a_stale_preset_record(tmp_path):
    manifest, first = _run_generate_kfile(tmp_path, "--preset", "statistical_891")
    assert first.returncode == 0, first.stderr[-2000:]
    _, second = _run_generate_kfile(tmp_path, "--npprime", "2", "--nffprime", "2")
    assert second.returncode == 0, second.stderr[-2000:]
    assert not (manifest.parent / PRESET_RECORD).exists()


def test_a_run_naming_nothing_records_the_default(tmp_path):
    manifest, result = _run_generate_kfile(tmp_path)
    assert result.returncode == 0, result.stderr[-2000:]
    record = json.loads((manifest.parent / PRESET_RECORD).read_text())
    assert record["name"] == DEFAULT_PRESET


def test_a_rerun_lists_only_its_own_kfiles_and_keeps_the_earlier_ones_aside(tmp_path):
    """k-files of an earlier run must not enter the new manifest (#1786).

    An earlier run left a k-file at an instant this run does not produce (and,
    in production, from another Green table). Globbing ``kfile/`` listed it
    with the new ones, and EFIT ran both.
    """
    manifest, first = _run_generate_kfile(tmp_path, "--preset", "statistical_891")
    assert first.returncode == 0, first.stderr[-2000:]
    first_run = sorted(Path(line).name for line in manifest.read_text().split())
    kfile_dir = manifest.parent.parent / "kfile"
    stale = kfile_dir / "k039915.00001"  # sorts before every real instant, as the 09-03 files did
    stale.write_text(" &IN1\n TABLE_DIR = '/old/table/'\n /\n", encoding="utf-8")
    stale_g = manifest.parent.parent / "gfile" / "g039915.00001"  # that earlier run's reconstruction
    stale_g.parent.mkdir(parents=True, exist_ok=True)
    stale_g.write_text("an earlier configuration's g-file", encoding="utf-8")

    _, second = _run_generate_kfile(tmp_path, "--preset", "statistical_891")
    assert second.returncode == 0, second.stderr[-2000:]
    listed = [Path(line) for line in manifest.read_text().split()]
    assert sorted(path.name for path in listed) == first_run  # this run's instants, nothing else
    assert all(path.parent == kfile_dir for path in listed)
    assert sorted(p.name for p in kfile_dir.glob("k039915.*")) == first_run
    superseded = list((kfile_dir / "superseded").glob("*/k039915.*"))
    assert stale.name in {p.name for p in superseded}  # kept aside, not deleted
    assert len(superseded) == len(first_run) + 1
    assert not stale_g.exists()  # the collection cannot read it as this run's
    assert (stale_g.parent / "superseded").is_dir() and list((stale_g.parent / "superseded").glob("*/g039915.00001"))


def _generate_kfile_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("pipeline1_generate_kfile", PIPELINE1 / "generate_kfile.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _plant_earlier_run(efit_dir, generation):
    """A stale k-, g-file and run-directory file of an earlier run, marked by generation."""
    for relative in ("kfile/k039915.00001", "gfile/g039915.00001", "m039915.00001"):
        path = efit_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"generation {generation}", encoding="utf-8")


def _stamps(directory):
    return sorted(p.name for p in (directory / "superseded").glob("*") if p.is_dir())


def test_superseding_three_times_keeps_one_generation_by_default(tmp_path):
    """``superseded/<stamp>/`` is bounded: the newest generation stays, older ones go.

    Every re-run moved 22-28 MB of a shot's g/k/m/a-files aside and nothing
    pruned them, ~100 GB per full regeneration pass (cold review 0.8.0
    delta-absorb-18 F3).
    """
    module = _generate_kfile_module()
    efit_dir = tmp_path / "39915" / "efit"
    for generation in (1, 2, 3):
        _plant_earlier_run(efit_dir, generation)
        module.supersede_earlier_run(efit_dir, 39915)

    for directory, name in ((efit_dir / "kfile", "k039915.00001"), (efit_dir / "gfile", "g039915.00001"), (efit_dir, "m039915.00001")):
        stamps = _stamps(directory)
        assert len(stamps) == 1, stamps  # the most recent superseded tree, never deleted
        assert (directory / "superseded" / stamps[0] / name).read_text(encoding="utf-8") == "generation 3"
        assert not (directory / name).exists()


def test_keep_superseded_bounds_the_generations_kept(tmp_path):
    module = _generate_kfile_module()
    efit_dir = tmp_path / "39915" / "efit"
    for generation in (1, 2, 3):
        _plant_earlier_run(efit_dir, generation)
        module.supersede_earlier_run(efit_dir, 39915, keep=2)
    stamps = _stamps(efit_dir / "kfile")
    assert len(stamps) == 2
    kept = [(efit_dir / "kfile" / "superseded" / s / "k039915.00001").read_text(encoding="utf-8") for s in stamps]
    assert kept == ["generation 2", "generation 3"]  # the oldest went, in time order
    (efit_dir / "kfile" / "superseded" / "notes.txt").write_text("not a stamp", encoding="utf-8")
    assert module.prune_superseded(efit_dir, keep=0) == stamps  # reports what it removed
    assert _stamps(efit_dir / "kfile") == []
    assert (efit_dir / "kfile" / "superseded" / "notes.txt").exists()  # only stamp directories are pruned
    with pytest.raises(ValueError):
        module.prune_superseded(efit_dir, keep=-1)


def test_the_script_prunes_older_superseded_generations_and_logs_them(tmp_path):
    """Three runs through the CLI on a FileDB-shaped tree leave one superseded stamp per kind."""
    manifest, first = _run_generate_kfile(tmp_path, "--preset", "statistical_891")
    assert first.returncode == 0, first.stderr[-2000:]
    efit_dir = manifest.parent.parent
    (efit_dir / "gfile").mkdir(exist_ok=True)
    (efit_dir / "gfile" / "g039915.00001").write_text("run 1", encoding="utf-8")
    _, second = _run_generate_kfile(tmp_path, "--preset", "statistical_891")
    assert second.returncode == 0, second.stderr[-2000:]
    assert "Pruned" not in second.stderr  # one generation: nothing older to prune
    (efit_dir / "gfile" / "g039915.00001").write_text("run 2", encoding="utf-8")
    _, third = _run_generate_kfile(tmp_path, "--preset", "statistical_891")
    assert third.returncode == 0, third.stderr[-2000:]

    kfile_stamps, gfile_stamps = _stamps(efit_dir / "kfile"), _stamps(efit_dir / "gfile")
    assert len(kfile_stamps) == 1 and len(gfile_stamps) == 1
    assert (efit_dir / "gfile" / "superseded" / gfile_stamps[0] / "g039915.00001").read_text(encoding="utf-8") == "run 2"
    assert "Pruned 1 older superseded generation(s) of shot 39915 (keeping 1)" in third.stderr
    _, refused = _run_generate_kfile(tmp_path, "--keep-superseded", "-1")
    assert refused.returncode != 0 and "must be >= 0" in refused.stderr


def test_the_efit_collection_payload_of_a_pre_record_product_is_unchanged():
    """No record (a product from before the k-file stage wrote one): the payload is what it was."""
    efit_collection_parameters = _generate_efit_ods_module().efit_collection_parameters
    common = dict(status="success", slice_statuses=[], mapping_diagnostics=[], artifact_hashes={},
                  artifact_manifest={})
    pre_record = json.loads(efit_collection_parameters(**common))["efit_collection"]
    assert "efit_preset" not in pre_record
    record = efit_preset("statistical_891").record()
    with_preset = json.loads(efit_collection_parameters(**common, efit_preset=record))["efit_collection"]
    assert with_preset["efit_preset"]["name"] == "statistical_891"


def _generate_efit_ods_module():
    sys.path.insert(0, str(PIPELINE1))
    try:
        import generate_efit_ods
    finally:
        sys.path.remove(str(PIPELINE1))
    return generate_efit_ods


def test_a_collection_without_a_record_stamps_unrecorded_not_the_default_sha(tmp_path):
    """A re-collect over k-files nobody recorded (pre-switch, --config or a legacy
    basis) used to stamp every slice status with today's default sha while the
    collection said nothing (cold review 0.8.0 delta-absorb-11b F4)."""
    from vaft.code.efit import collect_efit_outputs, resolved_efit_configuration

    module = _generate_efit_ods_module()
    config, configuration = module._collection_config(None, tmp_path, 39915)
    assert configuration["scientific"] is None
    assert configuration["scientific_sha256"] == "unrecorded"
    assert configuration["execution"]["shot"] == 39915
    result = collect_efit_outputs(tmp_path, config, configuration=configuration)
    assert result.configuration["scientific_sha256"] == "unrecorded"
    assert EFITScientificConfig().sha256 not in json.dumps(result.configuration)


@pytest.mark.parametrize("name", ["statistical_891", "routine"])
def test_a_collection_with_a_record_stamps_the_recorded_configuration(tmp_path, name):
    from vaft.code.efit import resolved_efit_configuration

    module = _generate_efit_ods_module()
    record = {**efit_preset(name).record(), "sigma_floor_changes": []}
    config, configuration = module._collection_config(record, tmp_path, 39915)
    assert configuration is None
    assert resolved_efit_configuration(config)["scientific_sha256"] == record["scientific_sha256"]


@pytest.mark.parametrize("preset, kffcur", [(None, 1), ("statistical_891", 1), ("routine", 2)])
def test_the_kinetic_base_kfile_follows_the_preset(tmp_path, monkeypatch, preset, kffcur):
    from vaft.code.efit import kinetic

    # The pressure points need a real equilibrium; only the base k-file is under test here.
    monkeypatch.setattr(kinetic, "kinetic_pressure_points", lambda *a, **k: None)
    config = kinetic.KineticEFITConfig(workdir=tmp_path, shot=39915, time_ms=319.0, efit_preset=preset)
    inputs = kinetic.prepare_kinetic_efit_inputs(_constraints_ods(tmp_path), None, config)
    assert int(_key(inputs.base_kfile_text, "KFFCUR")) == kffcur
    assert ("SAICON" in inputs.base_kfile_text) == (preset != "routine")


#: `EFITScientificConfig().sha256` on develop before 2026-10-01: the routine
#: configuration keeps it, so records that name it still match.
ROUTINE_SHA256 = "9bbb69a6a858bede8184c72e3ad4436d952215ca6f71710ed2fa6433cb6812ee"


def test_the_routine_hash_did_not_move():
    assert routine_scientific_config().sha256 == ROUTINE_SHA256
    assert "sigma_floor" not in routine_scientific_config().to_dict()["constraints"]
    assert "sigma_floor" in EFITScientificConfig().to_dict()["constraints"]


def test_from_dict_fills_what_the_payload_lacks_from_the_default():
    """A partial payload is the default with those keys changed, like every other "nothing named" path.

    It used to be filled from the routine values, so ``--config '{"profile":
    {"kppcur": 3}}'`` silently selected legacy weights, EFIT termination and
    no floor (cold review 0.8.0 delta-absorb-11b F2).
    """
    from vaft.code.efit import EFITProfileConfig

    assert EFITScientificConfig.from_dict({}) == EFITScientificConfig()
    partial = EFITScientificConfig.from_dict({"profile": {"kppcur": 3}})
    assert partial == EFITScientificConfig(profile=EFITProfileConfig(kppcur=3))
    assert partial.constraints.sigma_floor == 0.02 and partial.numerics.max_iterations == 514
    assert partial.constraints.uncertainty_mode == "standard_deviation"
    # Present keys win, including inside a section.
    mixed = EFITScientificConfig.from_dict({"numerics": {"max_iterations": 7}})
    assert mixed.numerics.max_iterations == 7 and mixed.numerics.error_minimum == 1e-4


def test_from_dict_is_the_inverse_of_to_dict_for_both_configurations():
    """`to_dict` omits a zero floor (the routine hash depends on that), so a
    constraints section without one reads back as no floor -- and a record
    written before the floor was part of the configuration keeps its hash."""
    routine = routine_scientific_config()
    assert EFITScientificConfig.from_dict(routine.to_dict()) == routine
    assert EFITScientificConfig.from_dict(EFITScientificConfig().to_dict()) == EFITScientificConfig()
    payload = EFITScientificConfig().to_dict()
    payload["constraints"].pop("sigma_floor")
    payload["constraints"].pop("sigma_floor_families")
    replayed = EFITScientificConfig.from_dict(payload)
    assert replayed.constraints.sigma_floor == 0.0
    assert replayed.constraints.uncertainty_mode == "standard_deviation"
    assert replayed.to_dict() == payload


def test_a_payload_that_resolves_to_a_preset_names_it():
    """What a `--config` stage should record instead of nothing."""
    from vaft.code.efit import EFITProfileConfig
    from vaft.code.efit.presets import preset_of

    assert preset_of(EFITScientificConfig.from_dict({})).name == DEFAULT_PRESET
    assert preset_of(EFITScientificConfig.from_dict(routine_scientific_config().to_dict())).name == "routine"
    assert preset_of(EFITScientificConfig(profile=EFITProfileConfig(kppcur=3))) is None


@pytest.mark.parametrize("driver", [
    "workflow/efit_numerics/baseline_termination.py",
    "workflow/efit_temporal/run_cadence_study.py",
    "workflow/efit_tables/ab_efit_table.py",
])
def test_routine_study_drivers_pass_the_whole_configuration(driver):
    """`EFITConfig(npprime=..)` swaps only the basis on top of today's defaults."""
    source = (REPOSITORY / driver).read_text(encoding="utf-8")
    assert "npprime=scientific" not in source
    assert "numerics=scientific.numerics" in source and "constraints=scientific.constraints" in source
