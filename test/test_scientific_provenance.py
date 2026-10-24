"""Scientific evidence and transformation provenance (#2165, Cases A, C, D and the multi-stage chain of #2160).

Six different claims stay apart here: a declared convention, a compatible
representation, an available transformation, an executed one, a checked one,
and an independently validated result.
"""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from vaft.data.representation import (
    Compatibility,
    EvidenceLevel,
    Representation,
    TransformationRecord,
    assess_compatibility,
    cocos_transformation_record,
)
from vaft.validation.credibility import (
    AXES,
    CHECK_AXES,
    Evidence,
    phase_verdict_evidence,
    transformation_evidence,
)
from vaft.validation.model import ValidationStatus as S

KEY = "verification.cocos_transform_invariants"


@pytest.fixture(scope="module")
def sample():
    from vaft.data.resources import sample_geqdsk
    from vaft.process.equilibrium import as_equilibrium

    return as_equilibrium(sample_geqdsk(), convention=1)


@pytest.fixture(scope="module")
def converted(sample):
    from vaft.process.equilibrium import convert_cocos

    return convert_cocos(sample, 11)


# --- the registry and the axis map ------------------------------------------


@pytest.mark.parametrize("key", [KEY, "verification.harmonic_phase_convention"])
def test_transformation_checks_are_registered_and_sit_on_the_t_axis(key):
    import importlib

    from vaft.validation.registry import describe

    spec = describe(key)
    module, name = spec.provider.rsplit(".", 1)
    assert callable(getattr(importlib.import_module(module), name))
    assert CHECK_AXES[key] == "transformation"


# --- declared, available, executed, checked ----------------------------------


def test_an_available_transformation_is_not_evidence():
    assessment = assess_compatibility(Representation("psi", "Wb/rad", cocos=1), Representation("psi", "Wb", cocos=11))
    assert assessment.status is Compatibility.REQUIRES_TRANSFORM
    (reference,) = assessment.transformations
    record = TransformationRecord.from_reference(reference, assessment.source, assessment.target)
    assert not record.executed
    assert transformation_evidence(record) == ()


def test_a_compatible_assessment_produces_no_evidence_at_all():
    assessment = assess_compatibility(Representation("psi", "Wb", cocos=11), Representation("psi", "Wb", cocos=11))
    assert assessment.status is Compatibility.COMPATIBLE and not assessment.transformations


def test_an_executed_but_unchecked_transformation_is_not_available():
    record = cocos_transformation_record(1, 11)
    (evidence,) = transformation_evidence(record)
    assert (evidence.axis, evidence.status) == ("transformation", S.NOT_AVAILABLE)
    assert evidence.status is not S.INDETERMINATE


def test_a_checked_cocos_conversion_passes_on_t(sample, converted):
    from vaft.validation.cocos import check_cocos_transform

    report = check_cocos_transform(sample, converted)
    assert report.status is S.PASS, report.issues
    record = cocos_transformation_record(1, 11, checks=(KEY,))
    (evidence,) = transformation_evidence(record, report)
    assert (evidence.axis, evidence.key, evidence.status) == ("transformation", KEY, S.PASS)


@pytest.mark.parametrize("field, factor", [("psi_boundary", 2.0), ("ip", -1.5), ("q", 1.01)])
def test_a_wrong_factor_fails_the_invariant_check(sample, converted, field, factor):
    from vaft.validation.cocos import check_cocos_transform

    broken = replace(converted, **{field: np.asarray(getattr(converted, field)) * factor})
    report = check_cocos_transform(sample, broken)
    assert report.status is S.FAIL
    (evidence,) = transformation_evidence(cocos_transformation_record(1, 11, checks=(KEY,)), report)
    assert evidence.status is S.FAIL


@pytest.mark.parametrize("field, factor", [("psi", 7.0), ("pprime", 7.0), ("ffprime", 1 / 7)])
def test_the_flux_grid_and_its_derivatives_are_checked_too(sample, converted, field, factor):
    from vaft.validation.cocos import check_cocos_transform

    broken = replace(converted, **{field: np.asarray(getattr(converted, field)) * factor})
    assert check_cocos_transform(sample, broken).status is S.FAIL


@pytest.mark.parametrize("change", [
    {"ip": float("nan")}, {"bt0": float("nan")}, {"q": "nan"}, {"ip": 0.0},
])
def test_what_could_not_be_checked_is_never_a_pass(sample, converted, change):
    from vaft.validation.cocos import check_cocos_transform

    def spoil(e):
        values = {k: (np.full_like(np.asarray(getattr(e, k), dtype=float), np.nan) if v == "nan" else v)
                  for k, v in change.items()}
        return replace(e, **values)

    report = check_cocos_transform(spoil(sample), spoil(converted))
    assert report.status is not S.PASS


def test_a_check_named_without_its_report_is_refused():
    with pytest.raises(ValueError, match="no report"):
        transformation_evidence(cocos_transformation_record(1, 11, checks=(KEY,)))


def test_an_undeclared_convention_cannot_be_checked(sample, converted):
    from vaft.validation.cocos import check_cocos_transform

    with pytest.raises(ValueError):
        check_cocos_transform(replace(sample, convention=replace(sample.convention, cocos=None)), converted)


def test_a_faithful_transform_does_not_validate_the_source(sample, converted):
    from vaft.machine_mapping.conventions import unresolved_orientation_assumptions
    from vaft.validation.cocos import check_cocos_transform

    open_items = unresolved_orientation_assumptions()
    assert open_items and all("#288" in item for item in open_items)
    record = cocos_transformation_record(1, 11, checks=(KEY,), unresolved=open_items)
    (evidence,) = transformation_evidence(record, check_cocos_transform(sample, converted))
    assert evidence.status is S.PASS            # the conversion was faithful (T) ...
    assert evidence.axis != "source_evidence"   # ... and says nothing on E
    assert any("#288" in reason for reason in evidence.reasons)


# --- Case C: phase-audit verdicts on T ----------------------------------------------


def _verdict(stored_norm, conjugate_norm):
    from vaft.process.perturbation import ToroidalPhaseAudit
    from vaft.validation.perturbation_phase import GPEC_COIL_FIELD, phase_convention_verdict

    ratio = max(stored_norm, conjugate_norm) / min(stored_norm, conjugate_norm)
    audit = ToroidalPhaseAudit(
        n_tor=1, preferred="stored" if stored_norm < conjugate_norm else "conjugate",
        stored_relative_norm=stored_norm, conjugate_relative_norm=conjugate_norm,
        stored_component_relative_norm=(stored_norm,), conjugate_component_relative_norm=(conjugate_norm,),
        separation_ratio=ratio, stored_norm=1.0, measured_norm=1.0, sample_count=1, phi_sample_count=16)
    return phase_convention_verdict(audit, criteria=GPEC_COIL_FIELD)


@pytest.mark.parametrize("stored, conjugate, declared, status", [
    (1e-8, 1.0, "gpec", S.PASS),
    (1.0, 1e-8, "gpec", S.FAIL),
    # The sector decomposition uses the opposite kernel: the audit's "conjugate" confirms it.
    (1.0, 1e-8, "sector", S.PASS),
    (1e-8, 1.0, "sector", S.FAIL),
    (0.5, 0.6, "gpec", S.INDETERMINATE),
    (0.5, 0.6, "sector", S.INDETERMINATE),
    (1e-8, 1.0, "undeclared", S.INDETERMINATE),
])
def test_phase_verdicts_map_onto_t_against_the_declared_kernel(stored, conjugate, declared, status):
    from vaft.data.fourier import COIL_SECTOR_DECOMPOSITION, GPEC_HARMONIC, FourierConvention

    conventions = {"gpec": GPEC_HARMONIC, "sector": COIL_SECTOR_DECOMPOSITION,
                   "undeclared": FourierConvention("undeclared", None, 2.0)}
    evidence = phase_verdict_evidence(_verdict(stored, conjugate), conventions[declared])
    assert (evidence.axis, evidence.status, evidence.cost) == ("transformation", status, "moderate")


# --- the multi-stage chain -------------------------------------------------------------------


def _chain(sample, converted):
    from vaft.data.fourier import GPEC_HARMONIC
    from vaft.machine_mapping.conventions import unresolved_orientation_assumptions
    from vaft.validation.applicability import evaluate_contract
    from vaft.validation.cocos import check_cocos_transform
    from vaft.validation.context import ScientificContext
    from vaft.validation.orderings import contract

    efit = ScientificContext(
        "efit", "vaft.code.efit",
        representation=Representation("psi", "Wb/rad", cocos=1, evidence=EvidenceLevel.INFERRED),
        evidence=(
            Evidence("inference", "diagnostic_fit.bpol_probe", S.PASS, role="used_for_inference"),
            Evidence("numerical", "verification.convergence", S.PASS),
        ),
        unresolved=unresolved_orientation_assumptions(),
    )
    record = cocos_transformation_record(1, 11, checks=(KEY,))
    chease = ScientificContext(
        "chease", "vaft.code.chease",
        representation=Representation("psi", "Wb", cocos=11),
        transformations=(record,),
        evidence=(*transformation_evidence(record, check_cocos_transform(sample, converted)),
                  Evidence("numerical", "chease.convergence", S.PASS)),
        upstream=(efit,),
    )
    gpec = ScientificContext(
        "gpec", "vaft.code.gpec",
        applicability=(evaluate_contract(contract("ideal_single_fluid_mhd"), {}),),
        evidence=(phase_verdict_evidence(_verdict(0.5, 0.6), GPEC_HARMONIC),
                  Evidence("numerical", "gpec.convergence", S.PASS)),
        upstream=(chease,),
    )
    return efit, chease, gpec


def test_an_unresolved_upstream_polarity_survives_every_successful_stage(sample, converted):
    efit, chease, gpec = _chain(sample, converted)
    assert gpec.credibility()["numerical"] is S.PASS
    assert all(any("#288" in item for item in ctx.unresolved_assumptions()) for ctx in (efit, chease, gpec))


def test_axes_stay_separate_and_absent_axes_never_pass(sample, converted):
    _, _, gpec = _chain(sample, converted)
    composed = gpec.credibility()
    assert set(composed) <= set(AXES)
    # The T axis holds a pass (COCOS) and an indeterminate (phase): indeterminate wins.
    assert composed["transformation"] is S.INDETERMINATE
    # An evaluation with no ordering inputs is UNASSESSED: not available, not a pass.
    assert composed["applicability"] is S.NOT_AVAILABLE
    assert gpec.applicability[0].status == "UNASSESSED"
    assert "independent_validation" not in composed and "source_evidence" not in composed
    assert gpec.credibility(include_upstream=False) == {
        "transformation": S.INDETERMINATE, "applicability": S.NOT_AVAILABLE, "numerical": S.PASS}


def test_fitted_evidence_is_never_independent_validation_downstream(sample, converted):
    from vaft.validation.context import ScientificContext

    _, _, gpec = _chain(sample, converted)
    mislabelled = ScientificContext("thomson", evidence=(
        Evidence("independent_validation", "independent_validation.thomson", S.PASS, role="used_for_inference"),
    ), upstream=(gpec,))
    fitted = [e for e in mislabelled.all_evidence() if e.role == "used_for_inference"]
    assert len(fitted) == 2 and all(e.axis == "inference" for e in fitted)
    assert "independent_validation" not in mislabelled.credibility()
    independent = ScientificContext("thomson", evidence=(
        Evidence("independent_validation", "independent_validation.thomson", S.PASS, role="independent_validation"),
    ))
    assert independent.credibility() == {"independent_validation": S.PASS}


def test_the_chain_records_transformations_and_serializes(sample, converted):
    efit, _, gpec = _chain(sample, converted)
    assert [r.kind for r in gpec.all_transformations()] == ["cocos"]
    payload = json.loads(json.dumps(gpec.as_dict(), allow_nan=False))
    assert payload["schema_version"] == 1
    assert payload["upstream"][0]["transformations"][0]["provider"] == "vaft.process.equilibrium.convert_cocos"
    assert any("#288" in item for item in payload["unresolved_assumptions"])
    shared = type(gpec)("summary", upstream=(efit, efit))
    assert len(shared.chain()) == 2


def test_records_serialize_as_strict_json():
    record = cocos_transformation_record(1, 11, parameters={"eps": float("nan"), "k": np.float32(2.5)})
    payload = json.loads(json.dumps(record.as_dict(), allow_nan=False))
    assert payload["parameters"] == {"source_cocos": 1, "target_cocos": 11, "eps": None, "k": 2.5}


def test_a_context_accepts_none_for_empty_fields():
    from vaft.validation.context import ScientificContext

    assert ScientificContext("x", evidence=None, upstream=None).credibility() == {}


def test_numerical_convergence_is_not_applicability(sample, converted):
    _, _, gpec = _chain(sample, converted)
    composed = gpec.credibility()
    assert composed["numerical"] is S.PASS and composed["applicability"] is not S.PASS
