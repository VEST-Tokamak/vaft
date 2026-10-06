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


def _report() -> dict:
    return {
        "status": "warn",
        "verification": {"convergence": {"status": "pass", "error": 1e-5}},
        "diagnostic_fit": {"global": {"status": "warn", "value": 3.1}},
        "physical_validity": {
            "virial_conditioning": {"status": "warn", "reason": "denominator near zero"},
            "diamagnetic_flux": {"status": "pass", "value": 0.02},
            "q_profile": {"status": "pass"},
        },
        "independent_validation": {"thomson_pressure": {"status": "not_available", "reason": "no TS"}},
    }


def test_the_equilibrium_report_projects_onto_the_axes():
    evidence = {item.key: item for item in evidence_from_equilibrium_report(_report())}
    assert evidence["verification.convergence"].axis == "numerical"
    assert evidence["diagnostic_fit.global"].axis == "inference"
    assert evidence["physical_validity.virial_conditioning"].axis == "inference"
    assert evidence["physical_validity.virial_conditioning"].reasons == ("denominator near zero",)
    assert evidence["physical_validity.q_profile"].axis == "numerical"
    dia = evidence["physical_validity.diamagnetic_flux"]
    assert (dia.axis, dia.role) == ("independent_validation", "independent_validation")
    assert evidence["independent_validation.thomson_pressure"].status is ValidationStatus.NOT_AVAILABLE
    assert evidence["diagnostic_fit.global"].metrics == {"value": 3.1}


def test_fitted_data_is_never_independent_validation():
    evidence = {item.key: item for item in evidence_from_equilibrium_report(
        _report(), used_for_inference=("physical_validity.diamagnetic_flux",))}
    dia = evidence["physical_validity.diamagnetic_flux"]
    assert (dia.axis, dia.role) == ("inference", "used_for_inference")


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


def test_criteria_v2_keeps_thomson_off_the_fit_axes():
    evidence = {item.key: item for item in evidence_from_efit_criteria(_criteria_evaluation())}
    assert evidence["efit_criteria.thomson"].axis == "independent_validation"
    assert evidence["efit_criteria.thomson"].metrics == {"log_ratio": -0.9}
    composed = compose(evidence.values())
    assert composed["inference"] is ValidationStatus.PASS
    assert composed["numerical"] is ValidationStatus.PASS
    assert composed["independent_validation"] is ValidationStatus.FAIL


def test_criteria_of_another_version_are_refused():
    evaluation = dict(_criteria_evaluation(), criteria_version=1)
    with pytest.raises(ValueError, match="version"):
        evidence_from_efit_criteria(evaluation)


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
