"""Cross-backend consistency of plasma-formalism provenance (#1738, Phase D of #1723).

Taxonomy fixtures, not solver results: every test asks whether a shared field
means the same thing everywhere, never whether two solvers agree numerically.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from vaft.code.formalism import UNKNOWN, PlasmaFormalism
from vaft.validation.formalism_consistency import (
    COMPARISON_KEYS,
    FUTURE_FORMULATIONS,
    LAYERS,
    QUERIES,
    REFERENCE_FORMULATIONS,
    assert_comparable,
    backend_consistency,
    comparison_key,
    group_by_formalism,
    imported_formalism_findings,
    matches,
    mode_consistency,
    query,
    validate_formalism,
)
from vaft.validation.model_contracts import ADAPTER_MODES, adapter_mode

REF = REFERENCE_FORMULATIONS


def _mode(key):
    return adapter_mode(key).formalism


def _registry():
    return [m.formalism for m in ADAPTER_MODES if m.formalism is not None]


# -- the three layers (§3) and the generic rules (§4-§5) --

def test_every_reference_registry_and_future_formulation_is_valid():
    for name, record in {**REF, **FUTURE_FORMULATIONS}.items():
        assert validate_formalism(record) == (), name
        assert validate_formalism(record.as_dict()) == (), name
    for record in _registry():
        assert validate_formalism(record) == (), record.solver


def test_schema_problems_are_all_reported_and_semantics_wait_for_a_clean_schema():
    findings = validate_formalism({"schema_version": 99, "colour": "blue", "scientific_operation": "magic",
                                   "bulk_description": "fluid"})
    assert {f.layer for f in findings} == {"schema"}
    assert {f.field for f in findings} == {"schema_version", "colour", "scientific_operation"}
    # a fluid record with no fluid model is semantically wrong, but only reported once the schema is clean
    assert not any(f.layer == "semantic" for f in findings)
    assert [f.layer for f in validate_formalism({"scientific_operation": "ideal_stability",
                                                 "bulk_description": "fluid"})] == ["semantic"]


@pytest.mark.parametrize("fields, field, phrase", [
    (dict(bulk_description="hybrid", fluid_model="ideal_mhd", kinetic_coupling="pressure"),
     "kinetic_equation", "kinetic_equation unset"),
    (dict(bulk_description="kinetic", kinetic_equation="gyrokinetic", orbit_representation="full_orbit"),
     "orbit_representation", "removes the gyro-phase"),
    (dict(bulk_description="fluid", fluid_model="ideal_mhd", distribution_formulation="delta_f"),
     "distribution_formulation", "no meaning without a kinetic equation"),
    (dict(bulk_description="fluid", fluid_model="resistive_mhd", kinetic_equation="drift_kinetic"),
     "kinetic_equation", "bulk_description='hybrid'"),
    (dict(bulk_description="kinetic", kinetic_equation="fokker_planck"),
     "kinetic_equation", "test-species"),
])
def test_contradictions_name_the_field_and_explain_the_physics(fields, field, phrase):
    findings = validate_formalism({"scientific_operation": "ideal_stability", **fields})
    assert findings and findings[0].layer == "semantic" and findings[0].field == field
    assert phrase in findings[0].message
    with pytest.raises(ValueError, match=re.escape(phrase)):
        PlasmaFormalism(scientific_operation="ideal_stability", **fields)


def test_layers_are_named_and_findings_print_their_layer():
    assert LAYERS == ("schema", "semantic", "backend")
    finding = validate_formalism({"scientific_operation": "ideal_stability", "bulk_description": "fluid"})[0]
    assert str(finding).startswith("[semantic] fluid_model:")


def test_unsupported_formulations_are_valid():
    """Valid but not run by any VAFT backend: never rejected by generic validation (§5, §23)."""
    names = set(FUTURE_FORMULATIONS)
    assert {"global_deltaf_gyrokinetic_pic", "full_f_gyrokinetic_across_separatrix",
            "mhd_with_gyrokinetic_energetic_particles", "mars_k_kinetic_mhd"} <= names
    solvers = {m.formalism.distribution_formulation for m in ADAPTER_MODES if m.formalism}
    assert "full_f" not in solvers                       # no backend runs full-f ...
    assert FUTURE_FORMULATIONS["full_f_gyrokinetic_across_separatrix"].distribution_formulation == "full_f"


# -- independent axes (§6, §12) --

def test_delta_f_full_f_local_global_regime_and_field_model_vary_independently():
    base = REF["local_deltaf_continuum_gyrokinetic_linear"]
    for change in (dict(distribution_formulation="full_f"), dict(spatial_domain="radially_global"),
                   dict(regime="nonlinear"), dict(field_model="electromagnetic"),
                   dict(distribution_formulation="full_f", spatial_domain="local"),
                   dict(regime="linear", field_model="electromagnetic")):
        varied = replace(base, **change)
        assert validate_formalism(varied) == ()
        for name, value in change.items():
            others = {k: getattr(base, k) for k in ("distribution_formulation", "spatial_domain", "regime",
                                                   "field_model") if k not in change}
            assert all(getattr(varied, k) == v for k, v in others.items()), name


def test_numerical_representation_never_determines_the_formalism():
    gk = REF["local_deltaf_continuum_gyrokinetic_linear"]
    for representation in ("continuum", "pic", "monte_carlo"):
        varied = replace(gk, extensions={"numerical_representation": representation})
        assert comparison_key(varied) == comparison_key(gk)
        assert varied.bulk_description == "kinetic" and varied.orbit_representation == "gyrocenter"
    # marker particles do not make a model "particle", nor Monte Carlo a guiding-centre one
    assert FUTURE_FORMULATIONS["global_deltaf_gyrokinetic_pic"].bulk_description == "kinetic"
    assert FUTURE_FORMULATIONS["ascot5_full_orbit"].orbit_representation == "full_orbit"


def test_orbit_representation_is_not_a_kinetic_equation():
    gc = REF["guiding_center_test_particle"]
    assert gc.orbit_representation == "guiding_center" and gc.kinetic_equation is None
    assert not matches(gc, kinetic_equation="drift_kinetic")
    assert REF["full_orbit_test_particle"].bulk_description == "particle"   # full orbit != full kinetic


# -- pairwise non-conflation (§19) --

@pytest.mark.parametrize("a, b", [
    ("gpec:dcon_ideal", "gpec:dcon_kinetic"),
    ("gpec:rdcon_rmatch", "gpec:dcon_kinetic"),
    ("gacode:neo", "gpec:gpec_kinetic"),
    ("gacode:cgyro_linear", "gacode:tglf_linear"),
    ("gacode:cgyro_nonlinear", "gacode:tglf"),
    ("nubeam:fast_ions", "gacode:cgyro_nonlinear"),
])
def test_easily_conflated_modes_have_different_keys(a, b):
    assert comparison_key(_mode(a)) != comparison_key(_mode(b))
    with pytest.raises(ValueError, match="must not be aggregated"):
        assert_comparable([_mode(a), _mode(b)])


def test_neo_is_not_an_orbit_code_and_pic_gyrokinetics_is_not_orbit_following():
    assert comparison_key(_mode("gacode:neo")) != comparison_key(REF["guiding_center_test_particle"])
    pic = FUTURE_FORMULATIONS["global_deltaf_gyrokinetic_pic"]
    assert pic.bulk_description != REF["full_orbit_test_particle"].bulk_description


def test_rdcon_with_kinetic_profiles_stays_fluid():
    rdcon = _mode("gpec:rdcon_rmatch")
    assert rdcon.bulk_description == "fluid" and rdcon.kinetic_equation is None
    assert rdcon not in query(_registry(), "hybrid_kinetic_mhd")


def test_a_surrogate_is_not_the_model_it_emulates():
    tglf = _mode("gacode:tglf")
    surrogate = replace(tglf, extensions={**tglf.extensions, "realization": "learned_surrogate"})
    assert comparison_key(tglf) != comparison_key(surrogate)
    assert dict(comparison_key(surrogate))["realization"] == "learned_surrogate"
    assert query([tglf, surrogate], "reduced_from_gyrokinetics") == [tglf, surrogate]


def test_reduced_ancestry_is_not_the_equation_solved():
    tglf = _mode("gacode:tglf")
    assert tglf.derived_from == "gyrokinetic" and tglf.kinetic_equation is None
    assert tglf not in query(_registry(), "first_principles_gyrokinetic")


def test_selected_energetic_populations_are_not_a_kinetic_bulk():
    nubeam = _mode("nubeam:fast_ions")
    mhd_ep = FUTURE_FORMULATIONS["mhd_with_gyrokinetic_energetic_particles"]
    assert nubeam.bulk_description == "particle" and mhd_ep.bulk_description == "hybrid"
    energetic = query([*_registry(), *FUTURE_FORMULATIONS.values()], "energetic_particles_kinetic")
    assert nubeam in energetic and mhd_ep in energetic
    assert _mode("gacode:cgyro_nonlinear") not in energetic


# -- queries (§20) --

def test_cross_backend_queries_return_the_intended_modes():
    by_query = {name: {m.mode for m in ADAPTER_MODES if m.formalism and matches(m.formalism, **criteria)}
                for name, criteria in QUERIES.items()}
    assert by_query["drift_kinetic"] == {"dcon_kinetic", "gpec_kinetic", "ntv_closure", "neo"}
    assert by_query["first_principles_gyrokinetic"] == {"cgyro_linear", "cgyro_nonlinear"}
    assert by_query["hybrid_kinetic_mhd"] == {"dcon_kinetic", "gpec_kinetic", "ntv_closure"}
    assert by_query["guiding_center_orbits"] == {"fast_ions"}
    assert by_query["reduced_from_gyrokinetics"] == {"tglf_linear", "tglf", "tglf_nn"}
    assert by_query["energetic_particles_kinetic"] == {"fast_ions"}
    with pytest.raises(KeyError, match="known"):
        query([], "best_model")


# -- unknown versus not applicable (§14) --

def test_unknown_and_not_applicable_survive_serialization_and_never_group_together():
    na = REF["drift_kinetic_neoclassical"]
    unknown = replace(na, distribution_formulation=UNKNOWN)
    none = replace(na, distribution_formulation=None)
    for record in (unknown, none):
        assert PlasmaFormalism.from_dict(record.as_dict()) == record
    assert len(group_by_formalism([unknown, none, unknown])) == 2
    assert not matches(unknown, distribution_formulation=None)
    assert matches(none, distribution_formulation=None)


def test_grouping_keeps_guiding_centre_and_full_orbit_apart():
    groups = group_by_formalism([FUTURE_FORMULATIONS["ascot5_full_orbit"], FUTURE_FORMULATIONS["ascot5_guiding_center"]])
    assert len(groups) == 2
    assert "orbit_representation" in COMPARISON_KEYS
    assert_comparable([_mode("gacode:cgyro_linear"), _mode("gacode:cgyro_linear")])


# -- backend consistency (§15) and imported provenance (§16) --

def _cell(tmp_path: Path, name: str, **overrides: str) -> Path:
    from vaft.data.resources import data_path

    cell = tmp_path / name.split(".")[0]
    cell.mkdir(parents=True, exist_ok=True)
    text = (Path(data_path("gpec")) / name).read_text(encoding="utf-8")
    for key, value in overrides.items():
        text, count = re.subn(rf"(?im)^(\s*{key}\s*=\s*)[^!\n]*", rf"\g<1>{value} ", text)
        assert count == 1
    (cell / name).write_text(text, encoding="utf-8")
    return cell


def test_a_dcon_record_must_agree_with_its_prepared_kin_flag(tmp_path):
    from vaft.code.gpec._formalism import resolve_plasma_formalism

    ideal = resolve_plasma_formalism("dcon", _cell(tmp_path / "i", "dcon.in", kin_flag="f"))
    kinetic = resolve_plasma_formalism("dcon", _cell(tmp_path / "k", "dcon.in", kin_flag="t"))
    assert backend_consistency(ideal, ideal) == ()
    claimed = backend_consistency(kinetic, ideal)      # a kinetic record for a kin_flag=f input
    assert {f.field for f in claimed} >= {"bulk_description", "kinetic_equation"}
    assert all(f.layer == "backend" for f in claimed)
    assert mode_consistency(ideal, adapter_mode("gpec:dcon_ideal")) == ()
    assert {f.field for f in mode_consistency(ideal, adapter_mode("gpec:dcon_kinetic"))} >= {"bulk_description"}


def test_an_unsettled_input_never_accepts_a_recorded_formalism():
    record = _mode("gpec:dcon_ideal")
    assert backend_consistency(None, None) == ()
    assert backend_consistency(record, None)[0].field == "*"
    assert backend_consistency(None, record)[0].field == "*"


def test_a_cgyro_record_follows_its_configured_regime_and_fields():
    from vaft.code.gacode.cgyro import CGYROConfig, plasma_formalism

    linear = plasma_formalism(CGYROConfig())
    assert linear.regime == "linear"
    assert mode_consistency(linear, adapter_mode("gacode:cgyro_linear")) == ()
    assert {f.field for f in mode_consistency(linear, adapter_mode("gacode:cgyro_nonlinear"))} >= {"regime"}


def test_imported_records_use_the_generic_rules_and_a_declared_evidence_level():
    legacy = {"scientific_operation": "turbulent_transport", "bulk_description": "kinetic",
              "kinetic_equation": "gyrokinetic", "distribution_formulation": UNKNOWN}
    assert imported_formalism_findings(legacy, "legacy_ambiguous") == ()
    with pytest.raises(ValueError, match="evidence"):
        imported_formalism_findings(legacy, "trust_me")


# -- design guards (§21-§22, §24) --

def test_no_ranking_no_selection_and_no_machine_names():
    import vaft.validation.formalism_consistency as module

    source = Path(module.__file__).read_text(encoding="utf-8").lower()
    for forbidden in ("fidelity_rank", "best_model", "select_solver", "vest", "kstar", "iter "):
        assert forbidden not in source, forbidden


def test_importing_the_module_builds_no_record_and_loads_no_plotting():
    code = ("import sys, vaft.validation.formalism_consistency as m\n"
            "assert m.REFERENCE_FORMULATIONS._data is None\n"
            "assert not any(n.startswith(('matplotlib', 'vaft.plot', 'vaft.database')) for n in sys.modules)\n")
    subprocess.run([sys.executable, "-c", code], check=True)
