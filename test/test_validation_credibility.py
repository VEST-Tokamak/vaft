"""The #1639 credibility taxonomy and the applicability evaluation (Lane AP).

Pure Python and NumPy/pandas on synthetic states: no ODS is read and no solver
runs.  The import-isolation tests run in a clean interpreter, because what they
measure is an import side effect.
"""

from __future__ import annotations

import math
import subprocess
import sys

import pandas as pd
import pytest

from vaft.validation import ValidationStatus
from vaft.validation.applicability import (
    APPLICABILITY_STATUSES,
    ApproximationContract,
    OrderingAssumption,
    as_evidence,
    evaluate_contract,
    evaluate_population,
    ordering_margin,
    successive_discrepancy,
)
from vaft.validation.credibility import (
    AXES,
    CATEGORY_AXES,
    CHECK_AXES,
    CRITERIA_AXES,
    Evidence,
    compose,
    evidence_from_efit_criteria,
    evidence_from_equilibrium_report,
)
from vaft.validation.model import CATEGORIES
from vaft.validation.registry import CHECKS

# A toy contract.  Its quantities and thresholds are illustrative, not #1627's.
IDEAL = ApproximationContract(
    name="toy_ideal_mhd",
    physical_model="ideal single-fluid MHD",
    assumptions=(
        OrderingAssumption("lundquist_number", "large", "a", meaning="resistivity perturbative"),
        OrderingAssumption("ion_skin_depth_over_a", "small", "a", meaning="Hall terms small"),
        OrderingAssumption("rho_i_over_LTi", "small", "L_Ti", scope="profile"),
    ),
    references=("illustrative",),
)


def _in_subprocess(source: str) -> str:
    result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# margins
# ---------------------------------------------------------------------------

def test_margin_counts_decades_on_the_permitted_side():
    assert ordering_margin(1e-3, "small") == pytest.approx(3.0)
    assert ordering_margin(1e6, "large") == pytest.approx(6.0)
    assert ordering_margin(10.0, "small") == pytest.approx(-1.0)
    assert ordering_margin(0.1, "large") == pytest.approx(-1.0)
    assert ordering_margin(0.05, "small", threshold=0.1) == pytest.approx(math.log10(2.0))


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), 0.0, -1.0, "abc"])
def test_margin_is_nan_where_no_ordering_statement_can_be_made(value):
    assert math.isnan(ordering_margin(value, "small"))


def test_a_threshold_other_than_order_unity_must_name_its_source():
    with pytest.raises(ValueError, match="threshold_source"):
        OrderingAssumption("x", "small", "a", threshold=0.3)
    assumption = OrderingAssumption("x", "small", "a", threshold=0.3, threshold_source="Ref. (2)")
    assert assumption.threshold == 0.3


def test_an_ordering_must_name_its_scale_and_a_known_scope():
    with pytest.raises(ValueError, match="scale"):
        OrderingAssumption("x", "small", "")
    with pytest.raises(ValueError, match="scope"):
        OrderingAssumption("x", "small", "a", scope="somewhere")
    with pytest.raises(ValueError, match="ordering"):
        OrderingAssumption("x", "tiny", "a")


# ---------------------------------------------------------------------------
# contracts
# ---------------------------------------------------------------------------

def test_every_assumption_satisfied_is_supported_and_names_the_limiting_one():
    result = evaluate_contract(IDEAL, {"lundquist_number": 1e5, "ion_skin_depth_over_a": 0.2,
                                       "rho_i_over_LTi": 0.01})
    assert result.status == "SUPPORTED"
    assert result.limiting == "ion_skin_depth_over_a"
    assert [item.status for item in result.assumptions] == ["SUPPORTED"] * 3


def test_a_missing_quantity_is_unassessed_and_never_a_violation():
    result = evaluate_contract(IDEAL, {"lundquist_number": 1e5, "ion_skin_depth_over_a": 0.2})
    assert result.status == "UNASSESSED"
    assert result.unassessed == ("rho_i_over_LTi",)
    assert result.assumptions[2].reason == "not provided"
    assert result.limiting == "ion_skin_depth_over_a"


def test_a_violation_is_decided_even_when_another_quantity_is_missing():
    result = evaluate_contract(IDEAL, {"lundquist_number": 1e5, "ion_skin_depth_over_a": 1.5})
    assert result.status == "OUTSIDE"
    assert result.limiting == "ion_skin_depth_over_a"
    assert "violates" in result.assumptions[1].reason


def test_the_threshold_itself_is_not_scale_separation():
    result = evaluate_contract(IDEAL, {"lundquist_number": 1.0, "ion_skin_depth_over_a": 0.1,
                                       "rho_i_over_LTi": 0.1})
    assert result.status == "OUTSIDE"
    assert result.limiting == "lundquist_number"


def test_a_state_outside_the_contract_scope_is_not_applicable():
    contract = ApproximationContract("flat_only", "toy", (OrderingAssumption("x", "small", "a"),),
                                     applies_to={"phase": ("flat",)})
    assert evaluate_contract(contract, {"phase": "ramp_up", "x": 1e-3}).status == "NOT_APPLICABLE"
    assert evaluate_contract(contract, {"phase": "flat", "x": 1e-3}).status == "SUPPORTED"


def test_a_state_without_the_scope_field_is_unassessed_not_out_of_scope():
    contract = ApproximationContract("flat_only", "toy", (OrderingAssumption("x", "small", "a"),),
                                     applies_to={"phase": ("flat",)})
    for state in ({"x": 1e-3}, {"x": 1e-3, "phase": float("nan")}, {"x": 1e-3, "phase": None}):
        result = evaluate_contract(contract, state)
        assert result.status == "UNASSESSED"
        assert "phase not provided" in result.reason
        assert result.assumptions[0].margin == pytest.approx(3.0)  # the margin is still reported
        assert as_evidence(result).status is ValidationStatus.NOT_AVAILABLE


@pytest.mark.parametrize("value", [
    pytest.param(["flat", "ramp"], id="list"),
    pytest.param(("flat",), id="one-element-tuple"),
    pytest.param("ndarray-str", id="ndarray-str"),
    pytest.param("ndarray-float", id="ndarray-float"),
    pytest.param("ndarray-one", id="ndarray-one-element"),
])
def test_a_non_scalar_scope_value_is_unassessed_not_an_exception_or_silently_out_of_scope(value):
    # cold review 0.8.0 delta-absorb-17 species-docs F7: an array raised out of evaluate_population,
    # a list became NOT_APPLICABLE without a word
    import numpy as np

    arrays = {"ndarray-str": np.array(["flat", "ramp"]), "ndarray-float": np.array([1.0, 2.0]),
              "ndarray-one": np.array(["flat"])}
    value = arrays.get(value, value) if isinstance(value, str) else value
    contract = ApproximationContract("flat_only", "toy", (OrderingAssumption("x", "small", "a"),),
                                     applies_to={"phase": ("flat",)})
    result = evaluate_contract(contract, {"phase": value, "x": 1e-3})
    assert result.status == "UNASSESSED" and "not one scalar value" in result.reason
    table = pd.DataFrame({"phase": ["flat", value, "ramp"], "x": [1e-3] * 3}, index=[7, 8, 9])
    frame = evaluate_population(contract, table)
    assert list(frame.index) == [7, 8, 9]
    assert list(frame["status"]) == ["SUPPORTED", "UNASSESSED", "NOT_APPLICABLE"]
    # a 0-d array is one value
    assert evaluate_contract(contract, {"phase": np.array("flat"), "x": 1e-3}).status == "SUPPORTED"


def test_a_scope_given_as_a_bare_string_is_refused():
    with pytest.raises(TypeError, match="collection"):
        ApproximationContract("c", "toy", (OrderingAssumption("x", "small", "a"),), applies_to={"phase": "flat"})


def test_a_missing_value_is_not_provided_whatever_its_missing_type():
    import numpy as np

    for raw in (None, float("nan"), np.float32("nan"), pd.NA):
        result = evaluate_contract(IDEAL, {"lundquist_number": raw})
        assert result.assumptions[0].reason == "not provided"
    assert "admits no ordering" in evaluate_contract(IDEAL, {"lundquist_number": -3.0}).assumptions[0].reason


def test_a_contract_rejects_duplicate_or_absent_assumptions():
    with pytest.raises(ValueError):
        ApproximationContract("empty", "toy", ())
    with pytest.raises(ValueError):
        ApproximationContract("dup", "toy", (OrderingAssumption("x", "small", "a"),
                                             OrderingAssumption("x", "large", "a")))


def test_population_keeps_every_row_and_every_individual_check():
    table = pd.DataFrame({
        "shot": [1, 2, 3],
        "lundquist_number": [1e5, 1e5, float("nan")],
        "ion_skin_depth_over_a": [0.2, 2.0, 0.2],
        "rho_i_over_LTi": [0.01, 0.01, 0.01],
    }, index=[10, 11, 12])
    frame = evaluate_population(IDEAL, table, keep=("shot",))
    assert list(frame.index) == [10, 11, 12]
    assert list(frame["status"]) == ["SUPPORTED", "OUTSIDE", "UNASSESSED"]
    assert list(frame["shot"]) == [1, 2, 3]
    assert frame.loc[12, "lundquist_number_status"] == "UNASSESSED"
    assert frame.loc[11, "ion_skin_depth_over_a_margin"] == pytest.approx(-math.log10(2.0))
    assert frame.attrs["contract"] == "toy_ideal_mhd"


def test_the_status_vocabulary_is_lane_vs():
    assert APPLICABILITY_STATUSES == ("SUPPORTED", "OUTSIDE", "UNASSESSED", "NOT_APPLICABLE")
    try:
        from vaft.plot import operational_space
    except ImportError:  # pragma: no cover - plotting extras absent
        pytest.skip("vaft.plot unavailable")
    if not hasattr(operational_space, "APPLICABILITY_STATUSES"):
        pytest.skip("#1664 not merged yet")
    assert operational_space.APPLICABILITY_STATUSES == APPLICABILITY_STATUSES


def test_a_contract_result_becomes_applicability_evidence():
    supported = as_evidence(evaluate_contract(IDEAL, {"lundquist_number": 1e5, "ion_skin_depth_over_a": 0.2,
                                                       "rho_i_over_LTi": 0.01}))
    assert supported.axis == "applicability"
    assert supported.status is ValidationStatus.PASS
    assert supported.metrics["lundquist_number"]["margin"] == pytest.approx(5.0)
    outside = as_evidence(evaluate_contract(IDEAL, {"ion_skin_depth_over_a": 3.0}))
    assert outside.status is ValidationStatus.FAIL
    unassessed = as_evidence(evaluate_contract(IDEAL, {}))
    assert unassessed.status is ValidationStatus.NOT_AVAILABLE
    contract = ApproximationContract("flat_only", "toy", (OrderingAssumption("x", "small", "a"),),
                                     applies_to={"phase": ("flat",)})
    assert as_evidence(evaluate_contract(contract, {"phase": "vacuum"})) is None


def test_successive_discrepancy_measures_convergence_toward_the_fullest_model():
    steps = successive_discrepancy([1.0, 1.5, 1.6, 1.61])
    assert steps == pytest.approx([0.5 / 1.61, 0.1 / 1.61, 0.01 / 1.61])
    assert successive_discrepancy([1.0]) == []
    assert all(math.isnan(x) for x in successive_discrepancy([1.0, 2.0], reference=0.0))


# ---------------------------------------------------------------------------
# the taxonomy
# ---------------------------------------------------------------------------

def test_every_category_and_registered_check_lands_on_a_known_axis():
    assert set(CATEGORY_AXES) == set(CATEGORIES)
    assert set(CATEGORY_AXES.values()) <= set(AXES)
    assert set(CHECK_AXES) <= set(CHECKS)
    assert set(CHECK_AXES.values()) <= set(AXES)
    assert set(CRITERIA_AXES.values()) <= set(AXES)


def test_evidence_rejects_an_unknown_axis_cost_or_role():
    with pytest.raises(ValueError):
        Evidence("credibility", "k", "pass")
    with pytest.raises(ValueError):
        Evidence("numerical", "k", "pass", cost="free")
    with pytest.raises(ValueError):
        Evidence("numerical", "k", "pass", role="witness")
    assert Evidence("numerical", "k", "warn").status is ValidationStatus.WARN


def test_compose_keeps_axes_apart_and_missing_evidence_is_not_failure():
    composed = compose([
        Evidence("inference", "a", "pass"),
        Evidence("numerical", "b", "pass"),
        Evidence("numerical", "c", "not_available"),
        Evidence("independent_validation", "d", "not_available"),
    ])
    assert composed == {
        "inference": ValidationStatus.PASS,
        "numerical": ValidationStatus.INDETERMINATE,
        "independent_validation": ValidationStatus.NOT_AVAILABLE,
    }
    assert list(composed) == [axis for axis in AXES if axis in composed]
    assert "applicability" not in composed


def _entry(status, **slice_fields):
    """One check's report entry, in the shape ``validate_equilibrium``'s ``_collect`` builds."""
    one = {"time_slice": 0, "time": 0.3, "status": status, **slice_fields}
    return {"status": status, "counts": {status: 1}, "slices": [one]}


def _report(*, dia_fitted: bool, dia_entry: dict | None = None) -> dict:
    fit = {"global": _entry("warn", value=3.1)}
    if dia_fitted:
        fit["diamagnetic_flux"] = _entry("pass", value=0.4, fit_role="fitted", sigma_from_weight=1.0)
    if dia_entry is not None:
        fit["diamagnetic_flux"] = dia_entry
    return {
        "status": "warn",
        "verification": {"convergence": _entry("pass", error=1e-5)},
        "diagnostic_fit": fit,
        "physical_validity": {
            "virial_conditioning": _entry("warn", reason="denominator near zero"),
            "diamagnetic_flux": _entry("pass", value=0.02),
            "q_profile": _entry("pass"),
            "virial_identity": _entry("pass", value=0.01),
        },
        "independent_validation": {
            "thomson_pressure": _entry("not_available", reason="no TS"),
            "virial_measured_mu_i": _entry("pass", value=0.1),
        },
    }


def test_the_equilibrium_report_projects_onto_the_axes():
    evidence = {item.key: item for item in evidence_from_equilibrium_report(_report(dia_fitted=False))}
    assert evidence["verification.convergence"].axis == "numerical"
    assert (evidence["diagnostic_fit.global"].axis, evidence["diagnostic_fit.global"].role) == (
        "inference", "used_for_inference")
    assert evidence["physical_validity.virial_conditioning"].axis == "inference"
    assert evidence["physical_validity.virial_conditioning"].reasons == ("denominator near zero",)
    assert evidence["physical_validity.q_profile"].axis == "inference"
    assert evidence["physical_validity.virial_identity"].axis == "numerical"
    dia = evidence["physical_validity.diamagnetic_flux"]
    assert (dia.axis, dia.role) == ("independent_validation", "independent_validation")
    assert evidence["independent_validation.thomson_pressure"].status is ValidationStatus.NOT_AVAILABLE
    assert evidence["diagnostic_fit.global"].metrics["counts"] == {"warn": 1}
    assert evidence["diagnostic_fit.global"].metrics["slices"][0]["value"] == 3.1


def test_a_fitted_diamagnetic_flux_takes_every_check_against_it_off_v():
    evidence = {item.key: item for item in evidence_from_equilibrium_report(_report(dia_fitted=True))}
    for key in ("physical_validity.diamagnetic_flux", "independent_validation.virial_measured_mu_i"):
        assert (evidence[key].axis, evidence[key].role) == ("inference", "used_for_inference"), key
    assert evidence["independent_validation.thomson_pressure"].axis == "independent_validation"


def _dia_axis(report):
    item = {e.key: e for e in evidence_from_equilibrium_report(report)}["physical_validity.diamagnetic_flux"]
    return item.axis, item.role


def test_an_ungraded_but_fitted_diamagnetic_flux_is_still_inference():
    # cold review 0.8.0 delta-absorb-17 species-docs F3: the fit decision is the weight, not the grade
    ungraded = _entry("not_available", reason="uncertainty model is 'unknown' (#891)",
                      fit_role="fitted", sigma_from_weight=1.0, measured=1.4e-3, chi_squared=7e-16)
    assert _dia_axis(_report(dia_fitted=False, dia_entry=ungraded)) == ("inference", "used_for_inference")
    older = _entry("not_available", reason="uncertainty model is 'unknown' (#891)", sigma_from_weight=0.5)
    assert _dia_axis(_report(dia_fitted=False, dia_entry=older)) == ("inference", "used_for_inference")


@pytest.mark.parametrize("entry", [
    _entry("not_available", reason="no reconstructed diamagnetic_flux constraint on this slice", fit_role="absent"),
    _entry("not_available", reason="uncertainty model is 'unknown' (#891)", fit_role="prescribed",
           sigma_from_weight=float("nan"), measured=1.4e-3),
    _entry("pass", value=0.4, fit_role="prescribed", sigma_from_weight=float("nan")),
], ids=["absent", "weight-zero-ungraded", "weight-zero-graded"])
def test_an_unfitted_diamagnetic_flux_stays_independent_validation(entry):
    assert _dia_axis(_report(dia_fitted=False, dia_entry=entry)) == ("independent_validation", "independent_validation")


def test_the_packaged_sample_s_fitted_flux_never_lands_on_the_independent_axis():
    from vaft.omas import sample_ods
    from vaft.validation import validate_equilibrium

    report = validate_equilibrium(sample_ods(), checks=("diagnostic_fit", "physical_validity"), time_slice=0)
    dia = report["diagnostic_fit"]["diamagnetic_flux"]
    assert dia["slices"][0]["fit_role"] == "fitted" and dia["slices"][0]["sigma_from_weight"] == pytest.approx(1.0)
    assert _dia_axis(report) == ("inference", "used_for_inference")


def test_fitted_data_is_never_independent_validation():
    evidence = {item.key: item for item in evidence_from_equilibrium_report(
        _report(dia_fitted=False), used_for_inference=("independent_validation.thomson_pressure",))}
    thomson = evidence["independent_validation.thomson_pressure"]
    assert (thomson.axis, thomson.role) == ("inference", "used_for_inference")


@pytest.mark.parametrize("keys, error", [
    ("physical_validity.diamagnetic_flux", TypeError),
    (("diamagnetic_flux",), ValueError),
])
def test_a_fitted_key_that_cannot_mean_anything_is_refused(keys, error):
    with pytest.raises(error):
        evidence_from_equilibrium_report(_report(dia_fitted=False), used_for_inference=keys)
    with pytest.raises(error):
        evidence_from_efit_criteria(_criteria_evaluation(), used_for_inference=keys)


def test_the_real_report_round_trips_through_the_adapter():
    import json

    from vaft.omas import sample_ods
    from vaft.validation import validate_equilibrium

    report = validate_equilibrium(sample_ods(), checks=("verification", "diagnostic_fit"), time_slice=0)
    evidence = evidence_from_equilibrium_report(report)
    assert evidence, "the sample report produced no evidence"
    assert {item.key.split(".", 1)[0] for item in evidence} == {"verification", "diagnostic_fit"}
    for item in evidence:
        category, name = item.key.split(".", 1)
        assert item.status is ValidationStatus(report[category][name]["status"])
    json.dumps([item.as_dict() for item in evidence], allow_nan=False)


def _criteria_evaluation() -> dict:
    return {
        "verdicts": {
            "admissible": {"status": "pass", "reasons": []},
            "measurement": {"status": "pass", "families": {}, "reasons": []},
            "virial": {"status": "indeterminate", "reasons": ["pair_13 denominator 0.01"]},
            "grad_shafranov": {"status": "pass", "whole": 0.004, "reasons": []},
            "thomson": {"status": "fail", "log_ratio": -0.9, "reasons": ["ln(sum p_e / sum p_recon) -0.9"]},
        },
        "good": True,
        "physically_consistent": False,
        "criteria_version": 2,
    }


def test_criteria_v2_keeps_thomson_alone_on_v():
    evidence = {item.key: item for item in evidence_from_efit_criteria(_criteria_evaluation())}
    assert evidence["efit_criteria.thomson"].axis == "independent_validation"
    assert evidence["efit_criteria.thomson"].metrics == {"log_ratio": -0.9}
    # virial compares two beta_p of the same g-file: consistency, not validation.
    assert evidence["efit_criteria.virial"].axis == "numerical"
    assert [key for key, item in evidence.items() if item.axis == "independent_validation"] == [
        "efit_criteria.thomson"]
    composed = compose(evidence.values())
    assert composed["inference"] is ValidationStatus.PASS
    assert composed["numerical"] is ValidationStatus.INDETERMINATE  # virial undecided
    assert composed["independent_validation"] is ValidationStatus.FAIL


@pytest.mark.parametrize("version", [1, 2.7, float("nan")])
def test_criteria_of_another_version_are_refused(version):
    evaluation = dict(_criteria_evaluation(), criteria_version=version)
    with pytest.raises(ValueError, match="version"):
        evidence_from_efit_criteria(evaluation)


def test_evidence_is_json_ready_and_takes_one_reason_whole():
    import json

    item = Evidence("applicability", "k", "pass", {"margin": float("nan"), "nested": [float("inf"), 1.0]},
                    reasons="one reason")
    assert item.reasons == ("one reason",)
    assert json.loads(json.dumps(item.as_dict(), allow_nan=False))["metrics"] == {"margin": None,
                                                                                  "nested": [None, 1.0]}


# ---------------------------------------------------------------------------
# performance contract (#1639 s15)
# ---------------------------------------------------------------------------

def test_the_new_modules_import_no_plotting_database_or_solver_layer():
    leaked = _in_subprocess(
        "import sys, vaft.validation.credibility, vaft.validation.applicability\n"
        "print(','.join(sorted(m for m in sys.modules if m.startswith("
        "('matplotlib', 'vaft.database', 'vaft.plot', 'vaft.code', 'pandas', 'omas')))))"
    )
    assert leaked == "", f"importing the credibility layer pulled in: {leaked}"


def test_the_default_processing_path_does_not_load_the_assessment_layer():
    loaded = _in_subprocess(
        "import sys, vaft.process\n"
        "print(','.join(sorted(m for m in sys.modules if m in "
        "('vaft.validation.credibility', 'vaft.validation.applicability'))))"
    )
    assert loaded == ""
