"""Data roles survive round trip and constrain observed lifecycle operations."""
from dataclasses import replace
import numpy as np
import pytest
from vaft.process import ml


def declared():
    data = ml.build_dataset(np.arange(12.).reshape(6,2), ["a","a","b","c","d","d"],
        targets=np.arange(6.), spec=ml.DatasetSpec(name="roles", task="regress"))
    assignments = {"a":"fit", "b":"tune", "c":"cal", "d":"science", "absent":"science"}
    allowed = {"fit":["train","preprocess","evaluate","infer"], "tune":["select","evaluate","infer"],
        "cal":["calibrate","evaluate","infer"], "science":["evaluate","infer"]}
    data, split = ml.import_data_roles(data, assignments, allowed,
        role_to_partition={"fit":"train","tune":"validation","cal":"calibration","science":"test"},
        provenance={"split_revision":"owner-v2", "targets":{"origin":"reconstructed","method":"EFIT"}})
    return data, split, assignments


def test_exact_import_and_training_receipts(tmp_path):
    data, split, assignments = declared()
    assert dict(data.spec.metadata["data_use_policy"]["group_roles"]) == assignments
    assert split.assignment == {"a":"train","b":"validation","c":"calibration","d":"test"}
    ml.save_dataset(data, tmp_path / "data.npz")
    back = ml.load_dataset(tmp_path / "data.npz")
    assert back.spec.to_dict() == data.spec.to_dict() and back.fingerprint == data.fingerprint
    result = ml.train_model(back, split, ml.ModelSpec(architecture="ridge", backend="numpy", task="regress"))
    assert [r["operation"] for r in result.artifact.training["data_uses"]] == ["train","preprocess","evaluate"]
    evaluation = ml.evaluate_model(result.artifact, back, split)
    assert evaluation.provenance["data_use"]["group_roles"] == {"d":"science"}
    for operation in ("train","preprocess","calibrate","select"):
        with pytest.raises(ml.ModelContractError, match="forbidden"):
            ml.record_data_use(data, split, operation, groups=["d"])
    with pytest.raises(ml.ModelContractError, match="split differs"):
        ml.train_model(data, replace(split, assignment={**split.assignment,"d":"train"}),
                       ml.ModelSpec(architecture="ridge", backend="numpy", task="regress"))


def test_missing_groups_digest_and_external_preprocessing_rejected():
    data, split, _ = declared()
    with pytest.raises(ml.ModelContractError, match="outside"):
        ml.record_data_use(data, split, "evaluate", groups=["unknown"])
    policy = dict(data.spec.metadata["data_use_policy"])
    policy["assignment_sha256"] = "bad"
    bad = replace(data, spec=replace(data.spec, metadata={"data_use_policy":policy}))
    with pytest.raises(ml.ModelContractError, match="digest"):
        ml.record_data_use(bad, split, "evaluate")
    bad = replace(data, spec=replace(data.spec, metadata={**data.spec.metadata,"fitted_transform_groups":["c"]}))
    with pytest.raises(ml.ModelContractError, match="forbidden"):
        ml.train_model(bad, split, ml.ModelSpec(architecture="ridge", backend="numpy", task="regress"))


def test_legacy_is_opt_in_and_structured_source_stays_lazy():
    plain = ml.build_dataset(np.zeros((3,1)), ["a","b","c"], spec=ml.DatasetSpec(name="plain",task="test"))
    partition = ml.split_groups(plain)
    assert ml.record_data_use(plain, partition, "evaluate") is None
    rows = [ml.StructuredSample({"x":np.array([i],np.float32)}, {"y":np.array([i],np.float32)}) for i in range(3)]
    data = ml.build_structured_dataset(rows, ["a","b","c"], input_specs={"x":ml.TensorSpec((1,))},
        target_specs={"y":ml.TensorSpec((1,))}, spec=plain.spec)
    imported, split = ml.import_data_roles(data, {"a":"train","b":"validate","c":"science"},
        {"train":["train","preprocess"], "validate":["select"], "science":["evaluate"]},
        role_to_partition={"train":"train","validate":"validation","science":"test"})
    assert imported.source is rows and imported.fingerprint != data.fingerprint
    assert ml.record_data_use(imported.subset([2]), split, "evaluate")["groups"] == ["c"]


def test_calibration_and_inference_receipts_and_policy_mutation():
    data, split, _ = declared()
    model = ml.train_model(data, split, ml.ModelSpec(architecture="pca_autoencoder", backend="numpy",
        task="ae", hyperparameters={"n_components":1})).artifact
    calibration = ml.calibrate_threshold(model, data, split, partition="calibration")
    assert calibration.to_dict()["usage"]["groups"] == ["c"]
    with pytest.raises(ml.ModelContractError, match="forbidden"):
        ml.calibrate_threshold(model, data, split, partition="test")
    out = ml.predict(model, data)
    assert out.provenance["data_use"]["operation"] == "infer"
    data.spec.metadata["data_use_policy"]["allowed_operations"]["science"].append("train")
    with pytest.raises(ml.ModelContractError, match="policy digest"):
        ml.record_data_use(data, split, "train", groups=["d"])


def test_pep610_revision_is_owned_by_installed_vaft(monkeypatch):
    from vaft.process.ml import _provenance as p
    import json
    monkeypatch.setattr(p, "_git", lambda *args: None)
    class Installed:
        def locate_file(self, name):
            return p._PACKAGE_ROOT / name
        def read_text(self, name):
            return json.dumps({"vcs_info":{"vcs":"git","commit_id":"a"*40}})
    monkeypatch.setattr("importlib.metadata.distribution", lambda _: Installed())
    assert p.vaft_revision() == ("a"*40, None)
    monkeypatch.setattr(Installed, "locate_file", lambda self, name: p._PACKAGE_ROOT / "other" / name)
    assert p.vaft_revision() == (None, None)
