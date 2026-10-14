"""Scientific evidence is bound and cannot promote fitted/missing metrics."""
from dataclasses import replace
import copy
import numpy as np
import pytest
from vaft.process import ml
from vaft.validation.ml import MetricCriterion, build_ml_report, ml_report_evidence
from vaft.validation.model import ValidationStatus

PROTOCOL = {"name":"synthetic-contract", "revision":"v1", "population":"held-out synthetic groups", "scorer":"vaft numerical evaluation"}
CRITERIA = (MetricCriterion("mse", "independent_validation", "[a.u.]^2", "ml.evaluate_model", "synthetic linear regression contract", "v1", lower=0., upper=1.),)


def example(partition="test"):
    data = ml.build_dataset(np.arange(16.).reshape(8,2), np.repeat(["a","b","c","d"],2),
        targets=np.arange(8.), spec=ml.DatasetSpec(name="report",task="regress"))
    data, split = ml.import_data_roles(data, {"a":"train","b":"select","c":"calibrate","d":"science"},
        {"train":["train","preprocess","evaluate"],"select":["select","evaluate"],
         "calibrate":["calibrate","evaluate"],"science":["evaluate"]},
        role_to_partition={"train":"train","select":"validation","calibrate":"calibration","science":"test"})
    model = ml.train_model(data, split, ml.ModelSpec(architecture="ridge",backend="numpy",task="regress")).artifact
    evaluation = ml.evaluate_model(model, data, split, partition)
    report = build_ml_report(model, evaluation, data, split, CRITERIA, protocol=PROTOCOL)
    return model,evaluation,data,split,report


def evidence(objects, **kwargs):
    model,evaluation,data,split,report = objects
    return ml_report_evidence(report,model,evaluation,data,split,CRITERIA,
        protocol=kwargs.pop("protocol",PROTOCOL), expected_report_sha256=report["report_sha256"], **kwargs)


def test_independent_fitted_and_missing_evidence_are_distinct():
    independent = evidence(example())[0]
    assert independent.axis == "independent_validation" and independent.role == "independent_validation"
    assert independent.status == ValidationStatus.PASS
    for partition in ("train","validation","calibration"):
        fitted = evidence(example(partition))[0]
        assert fitted.axis == "inference" and fitted.role == "used_for_inference"
    model,evaluation,data,split,_ = example()
    criterion = (MetricCriterion("missing_physics", "applicability", "[-]", "test provider", "explicit test criterion", "v1", upper=1.),)
    report = build_ml_report(model,evaluation,data,split,criterion,protocol=PROTOCOL)
    result = ml_report_evidence(report,model,evaluation,data,split,criterion,protocol=PROTOCOL,
        expected_report_sha256=report["report_sha256"])
    assert result[0].status == ValidationStatus.NOT_AVAILABLE


def test_hash_metric_protocol_and_model_contract_mismatch():
    objects = example()
    model,evaluation,data,split,report = objects
    altered = copy.deepcopy(report)
    altered["metrics"]["mse"] = 0.
    with pytest.raises(ValueError, match="hash"):
        evidence((model,evaluation,data,split,altered))
    with pytest.raises(ValueError, match="identity/protocol"):
        evidence(objects,protocol={**PROTOCOL,"revision":"v2"})
    changed = model.replace(normalization={**model.normalization,"input_mean":np.zeros(2)})
    with pytest.raises(ValueError, match="model contract"):
        evidence((changed,evaluation,data,split,report))
    bad = replace(evaluation,n_samples=evaluation.n_samples+1)
    with pytest.raises(ValueError, match="population"):
        build_ml_report(model,bad,data,split,CRITERIA,protocol=PROTOCOL)


def test_legacy_history_and_nonfinite_metrics_never_become_independent_pass():
    model,evaluation,data,split,_ = example()
    legacy = model.replace(training={k:v for k,v in model.training.items() if k != "data_uses"})
    evaluation = ml.evaluate_model(legacy,data,split)
    report = build_ml_report(legacy,evaluation,data,split,CRITERIA,protocol=PROTOCOL)
    assert evidence((legacy,evaluation,data,split,report))[0].status == ValidationStatus.NOT_AVAILABLE
    evaluation = ml.evaluate_model(model,data,split)
    criterion=(replace(CRITERIA[0],axis="numerical"),)
    bad = replace(evaluation,metrics={"mse":float("nan")})
    report = build_ml_report(model,bad,data,split,criterion,protocol=PROTOCOL)
    assert ml_report_evidence(report,model,bad,data,split,criterion,protocol=PROTOCOL,
        expected_report_sha256=report["report_sha256"])[0].status == ValidationStatus.INDETERMINATE


def test_criteria_must_be_explicit_and_finite():
    with pytest.raises(ValueError, match="finite"):
        replace(CRITERIA[0],upper=float("inf"))
    with pytest.raises(ValueError, match="explicit"):
        replace(CRITERIA[0],source="")
    with pytest.raises(ValueError, match="lower"):
        replace(CRITERIA[0],lower=2.)


def test_changed_contract_cannot_build_a_new_report_from_old_evaluation():
    model,evaluation,data,split,_ = example()
    changed=model.replace(normalization={**model.normalization,"input_mean":np.zeros(2)})
    with pytest.raises(ValueError,match="model contract"):
        build_ml_report(changed,evaluation,data,split,CRITERIA,protocol=PROTOCOL)


def test_empty_population_cannot_grade_supplied_metrics():
    model,_,data,split,_ = example()
    empty=ml.build_dataset(data.inputs,data.groups,targets=data.targets,spec=data.spec,available=np.zeros(data.n_samples,bool))
    evaluation=ml.evaluate_model(model,empty,split)
    evaluation=replace(evaluation,metrics={"mse":1e10})
    criteria=(replace(CRITERIA[0],axis="numerical"),)
    report=build_ml_report(model,evaluation,empty,split,criteria,protocol=PROTOCOL)
    assert ml_report_evidence(report,model,evaluation,empty,split,criteria,protocol=PROTOCOL,
        expected_report_sha256=report["report_sha256"])[0].status == ValidationStatus.NOT_AVAILABLE
