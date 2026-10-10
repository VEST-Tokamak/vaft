"""The plasma formalism of each GPEC-suite run (#1734, Phase C1 of #1723).

Derived from the prepared namelists, never entered by hand; the classification is
the Phase A audit's (``docs/_guide/Plasma_models.md`` section 3, #1725).
"""

from __future__ import annotations

import dataclasses
import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from vaft.code.formalism import PlasmaFormalism
from vaft.code.gpec import GPECModuleRun
from vaft.code.gpec._formalism import resolve_plasma_formalism
from vaft.data.resources import data_path

TEMPLATES = Path(data_path("gpec"))


def _cell(tmp_path: Path, name: str, **overrides: str) -> Path:
    """A cell holding the packaged namelist, with ``key = value`` lines overridden."""
    cell = tmp_path / name.split(".")[0]
    cell.mkdir(parents=True, exist_ok=True)
    text = (TEMPLATES / name).read_text(encoding="utf-8")
    for key, value in overrides.items():
        text, count = re.subn(rf"(?im)^(\s*{key}\s*=\s*)[^!\n]*", rf"\g<1>{value} ", text)
        assert count == 1, f"{key} not in {name}"
    (cell / name).write_text(text, encoding="utf-8")
    return cell


def test_dcon_ideal_is_an_ideal_mhd_fluid_calculation(tmp_path):
    record = resolve_plasma_formalism("dcon", _cell(tmp_path, "dcon.in"))
    assert (record.scientific_operation, record.bulk_description, record.fluid_model) == (
        "ideal_stability", "fluid", "ideal_mhd")
    assert record.kinetic_equation is None and record.kinetic_population is None
    assert record.regime == "static" and record.solver == "dcon"
    # Numerical and boundary choices stay out of the axes.
    assert record.extensions["free_boundary"] is True and record.extensions["integrate_through_layers"] is True


@pytest.mark.parametrize(
    ("ion", "electron", "population"),
    [("t", "f", ("thermal_ions",)), ("f", "t", ("electrons",)), ("t", "t", ("thermal_ions", "electrons"))],
)
def test_dcon_kinetic_is_hybrid_and_names_only_its_populations(tmp_path, ion, electron, population):
    cell = _cell(tmp_path, "dcon.in", kin_flag="t", ion_flag=ion, electron_flag=electron)
    shutil.copy(TEMPLATES / "pentrc.in", cell / "pentrc.in")
    record = resolve_plasma_formalism("dcon", cell)
    assert (record.bulk_description, record.kinetic_equation, record.kinetic_coupling) == (
        "hybrid", "drift_kinetic", "energy")
    assert record.orbit_representation == "bounce_averaged" and record.distribution_formulation == "delta_f"
    assert record.kinetic_population == population  # never "all"
    assert record.extensions["collision_operator"] == "harmonic"
    assert record.extensions["passing_particles"] is True and record.extensions["trapped_particles"] is True


def test_dcon_kinetic_defaults_follow_upstream_when_the_namelist_omits_them(tmp_path):
    cell = tmp_path / "dcon"
    cell.mkdir()
    (cell / "dcon.in").write_text("&dcon_control\n    kin_flag = t\n/\n", encoding="utf-8")
    record = resolve_plasma_formalism("dcon", cell)
    # dcon_mod.f: passing_flag=F, trapped_flag=T, ion_flag=T, electron_flag=F.
    assert record.kinetic_population == ("thermal_ions",)
    assert record.extensions["passing_particles"] is False and record.extensions["trapped_particles"] is True


def test_a_missing_or_unreadable_namelist_gives_no_record(tmp_path):
    assert resolve_plasma_formalism("dcon", tmp_path) is None
    cell = tmp_path / "bad"
    cell.mkdir()
    (cell / "dcon.in").write_text("&dcon_control\n    kin_flag = maybe\n/\n", encoding="utf-8")
    assert resolve_plasma_formalism("dcon", cell) is None  # never fall back to "ideal"


def test_several_assignments_on_one_line_are_all_read(tmp_path):
    cell = tmp_path / "dcon"
    cell.mkdir()
    (cell / "dcon.in").write_text("&dcon_control\n    kin_flag=.T., electron_flag=t, ion_flag=.f.\n/\n",
                                  encoding="utf-8")
    record = resolve_plasma_formalism("dcon", cell)
    assert record.bulk_description == "hybrid" and record.kinetic_population == ("electrons",)


def test_kin_flag_without_a_species_is_ideal(tmp_path):
    record = resolve_plasma_formalism("dcon", _cell(tmp_path, "dcon.in", kin_flag="t", ion_flag="f", electron_flag="f"))
    assert record.bulk_description == "fluid" and record.extensions["kin_flag_without_species"] is True


def test_kin_flag_alone_names_no_kinetic_limit(tmp_path):
    cell = _cell(tmp_path, "dcon.in", kin_flag="t")
    text = json.dumps(resolve_plasma_formalism("dcon", cell).as_dict()).lower()
    assert "kruskal" not in text and "oberman" not in text


def test_rdcon_is_resistive_stability_and_profiles_never_make_it_kinetic(tmp_path):
    cell = _cell(tmp_path, "rdcon.in")
    shutil.copy(TEMPLATES / "rmatch.in", cell / "rmatch.in")
    template = resolve_plasma_formalism("rdcon", cell)
    profiles = resolve_plasma_formalism("rdcon", cell, config=SimpleNamespace(rdcon=SimpleNamespace(has_kinetic_profiles=True)))
    for record in (template, profiles):
        assert record.scientific_operation == "resistive_stability" and record.bulk_description == "fluid"
        assert record.kinetic_equation is None
    assert template.extensions["resistivity"] == "template_scalar"
    assert profiles.extensions["resistivity"] == "profile_informed_spitzer"
    # Without RMATCH's solution only the ideal outer region was solved.
    assert template.fluid_model == "ideal_mhd" and template.extensions["inner_layer"].startswith("not solved")


def test_rdcon_inner_layer_evidence_is_delta_out_not_globalsol(tmp_path):
    """match_flag writes delta.out (rmatch/match.f:814); globalsol.bin is RPEC only (match.f:1463)."""
    cell = _cell(tmp_path, "rdcon.in")
    (cell / "delta.out").write_text("", encoding="utf-8")
    solved = resolve_plasma_formalism("rdcon", cell)
    assert solved.fluid_model == "resistive_mhd" and solved.extensions["inner_layer"].startswith("rmatch match_flag")
    (cell / "globalsol.bin").write_bytes(b"")
    rpec = resolve_plasma_formalism("rdcon", cell)
    assert rpec.scientific_operation == "perturbed_equilibrium" and rpec.regime == "static"


def test_a_per_surface_eta_is_read_from_rmatch_in_without_the_config(tmp_path):
    cell = _cell(tmp_path, "rdcon.in")
    (cell / "rmatch.in").write_text("&match_input\n    eta=1e-8, 2e-8, 3e-8\n/\n", encoding="utf-8")
    assert resolve_plasma_formalism("rdcon", cell).extensions["resistivity"] == "per_surface_array"


def test_stride_is_the_ideal_outer_region_and_not_rdcon(tmp_path):
    stride = resolve_plasma_formalism("stride", _cell(tmp_path, "stride.in"))
    assert (stride.fluid_model, stride.extensions["resistivity"]) == ("ideal_mhd", "none")
    assert stride.solver == "stride" and stride != resolve_plasma_formalism("rdcon", _cell(tmp_path, "rdcon.in"))


def test_gpec_inherits_ideal_or_kinetic_from_its_dcon_cell(tmp_path):
    gpec = _cell(tmp_path, "gpec.in")
    ideal_dcon = _cell(tmp_path / "ideal", "dcon.in")
    kinetic_dcon = _cell(tmp_path / "kinetic", "dcon.in", kin_flag="t")
    ideal = resolve_plasma_formalism("gpec", gpec, dcon_dir=ideal_dcon)
    kinetic = resolve_plasma_formalism("gpec", gpec, dcon_dir=kinetic_dcon)
    assert ideal.scientific_operation == kinetic.scientific_operation == "perturbed_equilibrium"
    assert ideal.bulk_description == "fluid" and kinetic.bulk_description == "hybrid"
    assert kinetic.kinetic_coupling == "energy" and "inherited" in kinetic.extensions["kinetic_source"]
    # Without the DCON cell nothing says whether GPEC inherited a kinetic response.
    assert resolve_plasma_formalism("gpec", gpec) is None
    assert resolve_plasma_formalism("gpec", gpec, dcon_dir=tmp_path / "nowhere") is None


def test_gpec_thresholds_do_not_change_the_governing_formalism(tmp_path):
    text = (TEMPLATES / "gpec.in").read_text(encoding="utf-8")
    keys = re.findall(r"(?im)^\s*(singthresh\w*)\s*=", text)
    overrides = {key: "t" for key in keys if key.lower().endswith("_flag")}
    if not overrides:
        pytest.skip("packaged gpec.in exposes no singthresh flag")
    record = resolve_plasma_formalism("gpec", _cell(tmp_path, "gpec.in", **overrides), dcon_dir=_cell(tmp_path / "d", "dcon.in"))
    assert record.bulk_description == "fluid" and record.extensions["auxiliary_threshold_models"]


def test_pentrc_is_a_one_species_closure(tmp_path):
    cell = _cell(tmp_path, "pentrc.in", electron="t", tgar_flag="t", nutype='"krook"')
    record = resolve_plasma_formalism("pentrc", cell)
    assert (record.scientific_operation, record.bulk_description, record.kinetic_coupling) == (
        "toroidal_torque", "hybrid", "closure")
    assert record.kinetic_population == ("electrons",)
    assert record.extensions["methods"] == ("tgar",)  # frozen: lists read back as tuples
    assert record.extensions["collision_operator"] == "krook"


def test_match_and_rmatch_have_no_record_of_their_own(tmp_path):
    for module in ("match", "rmatch"):
        assert resolve_plasma_formalism(module, tmp_path) is None


def test_the_record_serializes_through_the_run_manifest(tmp_path):
    formalism = resolve_plasma_formalism("dcon", _cell(tmp_path, "dcon.in", kin_flag="t"))
    run = GPECModuleRun("dcon", 1, tmp_path, plasma_formalism=formalism.as_dict())
    manifest = json.loads(json.dumps(dataclasses.asdict(run), default=str))
    assert manifest["plasma_formalism"]["schema_version"] == 1
    assert PlasmaFormalism.from_dict(manifest["plasma_formalism"]) == formalism
    assert run.formalism == formalism and GPECModuleRun("dcon", 1, tmp_path).formalism is None


def test_the_suite_attaches_a_formalism_to_every_prepared_cell(tmp_path, monkeypatch):
    from vaft.code import gpec

    gfile = Path(data_path("efit/g039915.00319"))
    inputs = gpec.GPECCaseInputs(shot=39915, time_ms=319, geqdsk=gfile, workdir=tmp_path)
    config = gpec.GPECSuiteConfig(modules=("dcon", "rdcon", "stride"), modes=(1,), run_mode="prepare_only")
    result = gpec.run_gpec_suite_case(inputs, config)
    by_module = {record.module: record.formalism for record in result.records}
    assert by_module["dcon"].fluid_model == "ideal_mhd"
    assert by_module["rdcon"].scientific_operation == "resistive_stability"
    assert by_module["stride"].solver == "stride"


def test_the_record_travels_with_mhd_linear(tmp_path):
    """Phase A deferred this to #1734: the ODS provenance carries the formalism too."""
    pytest.importorskip("omas")
    from omas import ODS

    from vaft.machine_mapping.mhd_linear import extract_dcon_stability, extract_rdcon_stability, mhd_linear

    data = Path(__file__).resolve().parent / "data" / "gpec"
    dcon = tmp_path / "dcon"
    shutil.copytree(data / "dcon_edge_792" / "full_edge", dcon)
    rdcon = tmp_path / "rdcon"
    rdcon.mkdir()
    shutil.copy(data / "rdcon_39915_319_n1" / "rdcon_output_n1.nc", rdcon / "rdcon_output_n1.nc")
    shutil.copy(TEMPLATES / "rdcon.in", rdcon / "rdcon.in")
    bare = tmp_path / "bare"
    bare.mkdir()
    shutil.copy(data / "rdcon_39915_319_n1" / "rdcon_output_n1.nc", bare / "rdcon_output_n1.nc")
    ods = ODS(consistency_check=False)
    mhd_linear(ods, str(dcon), {"module": "dcon", "modes": [1]})
    mhd_linear(ods, str(rdcon), {"module": "rdcon", "modes": [1]})
    [dcon_row] = extract_dcon_stability(ods)
    [rdcon_row] = extract_rdcon_stability(ods)
    assert PlasmaFormalism.from_dict(dcon_row["plasma_formalism"]) == resolve_plasma_formalism("dcon", dcon)
    assert PlasmaFormalism.from_dict(rdcon_row["plasma_formalism"]).scientific_operation == "resistive_stability"
    # Mapped from a directory without its namelist: no record, not an "ideal" guess.
    other = ODS(consistency_check=False)
    mhd_linear(other, str(bare), {"module": "rdcon", "modes": [1]})
    [bare_row] = extract_rdcon_stability(other)
    assert bare_row["plasma_formalism"] is None
