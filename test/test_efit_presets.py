"""Named EFIT configurations (#891 working setting) and their pipeline wiring."""

from __future__ import annotations

import json
import re
from dataclasses import replace
import subprocess
import sys
from pathlib import Path

import pytest
from omas import save_omas_json

from test_efit_config import _constraints_ods
from vaft.code.efit import PRESETS, EFITScientificConfig, apply_sigma_floor, efit_preset, generate_kfile
from vaft.code.efit.config import routine_scientific_config
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
    generate_kfile(_constraints_ods(Path("/GOLDEN_INPUT_DIR")), 39915, save_dir=str(tmp_path), **kwargs)
    return next((tmp_path / "kfile").iterdir()).read_text(encoding="utf-8")


def test_the_default_is_the_working_setting_byte_for_byte(tmp_path):
    """2026-10-01: the defaults are statistical_891, whose k-file is unchanged from #1339."""
    assert DEFAULT_PRESET == "statistical_891"
    assert efit_preset(DEFAULT_PRESET).scientific == EFITScientificConfig()
    golden = (GOLDEN / "statistical_891.k").read_text(encoding="utf-8")
    assert _golden_kfile(tmp_path / "default") == golden
    assert _golden_kfile(tmp_path / "preset", config=efit_preset("statistical_891").scientific) == golden


def test_the_routine_preset_is_the_old_default_byte_for_byte(tmp_path):
    """The legacy configuration stays reachable exactly, by name or by a positional basis."""
    routine = efit_preset("routine")
    assert routine.scientific == routine_scientific_config()
    assert routine.sigma_floor == 0.0
    golden = (GOLDEN / "routine.k").read_text(encoding="utf-8")
    assert _golden_kfile(tmp_path / "preset", config=routine.scientific) == golden
    assert _golden_kfile(tmp_path / "positional", npprime=2, nffprime=2) == golden


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


def test_the_efit_collection_records_a_preset_only_when_one_was_used():
    sys.path.insert(0, str(PIPELINE1))
    try:
        from generate_efit_ods import efit_collection_parameters
    finally:
        sys.path.remove(str(PIPELINE1))
    common = dict(status="success", slice_statuses=[], mapping_diagnostics=[], artifact_hashes={},
                  artifact_manifest={})
    routine = json.loads(efit_collection_parameters(**common))["efit_collection"]
    assert "efit_preset" not in routine
    record = efit_preset("statistical_891").record()
    with_preset = json.loads(efit_collection_parameters(**common, efit_preset=record))["efit_collection"]
    assert with_preset["efit_preset"]["name"] == "statistical_891"


@pytest.mark.parametrize("preset, kffcur", [(None, 1), ("statistical_891", 1), ("routine", 2)])
def test_the_kinetic_base_kfile_follows_the_preset(tmp_path, monkeypatch, preset, kffcur):
    from vaft.code.efit import kinetic

    # The pressure points need a real equilibrium; only the base k-file is under test here.
    monkeypatch.setattr(kinetic, "kinetic_pressure_points", lambda *a, **k: None)
    config = kinetic.KineticEFITConfig(workdir=tmp_path, shot=39915, time_ms=319.0, efit_preset=preset)
    inputs = kinetic.prepare_kinetic_efit_inputs(_constraints_ods(tmp_path), None, config)
    assert int(_key(inputs.base_kfile_text, "KFFCUR")) == kffcur
    assert ("SAICON" in inputs.base_kfile_text) == (preset != "routine")
