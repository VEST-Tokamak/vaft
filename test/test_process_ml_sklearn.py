"""The scikit-learn backend: a one-class SVM trained in sklearn, stored as ONNX (#973 section 15)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from vaft.process import ml

sklearn = pytest.importorskip("sklearn")
pytest.importorskip("skl2onnx")
ort = pytest.importorskip("onnxruntime")


def _data():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((400, 4))
    ds = ml.build_dataset(x, np.repeat(np.arange(20), 20), spec=ml.DatasetSpec(name="o", task="t", group_key="shot"))
    return ds, ml.split_groups(ds, ml.SplitSpec(seed=3))


def _spec(**hp):
    return ml.ModelSpec(architecture="one_class_svm", task="toy_ocsvm", backend="sklearn",
                        hyperparameters={"nu": 0.1, **hp})


def test_ocsvm_scores_match_sklearn_with_larger_meaning_more_anomalous():
    from sklearn.svm import OneClassSVM

    ds, split = _data()
    result = ml.train_model(ds, split, _spec())
    model = result.artifact
    assert model.kind == "score" and model.outputs == ("score",)
    fitted = ds.inputs[result.train_indices]
    xn = (fitted - model.normalization["input_mean"]) / model.normalization["input_std"]
    reference = -OneClassSVM(nu=0.1, gamma="scale").fit(xn.astype(np.float32)).decision_function(
        ((ds.inputs - model.normalization["input_mean"]) / model.normalization["input_std"]).astype(np.float32)
    )
    np.testing.assert_allclose(ml.predict(model, ds).outputs["score"], reference, atol=1e-3)
    far = ml.predict(model, ds.inputs[:1] + 20.0).outputs["score"][0]
    assert far > np.max(ml.predict(model, ds).outputs["score"])
    assert result.train_scores is None  # in-sample SVM scores are not leave-self-out
    assert model.training["history"]["onnx_parity_max_abs_diff"][0] < 1e-3
    assert "skl2onnx" in model.training["runtimes"]


def test_ocsvm_artifact_is_onnx_and_round_trips(tmp_path):
    ds, split = _data()
    model = ml.train_model(ds, split, _spec()).artifact
    ml.save_model_artifact(model, tmp_path / "m", version="0.1.0")
    manifest = json.loads((tmp_path / "m" / "manifest.json").read_text())
    assert manifest["inference_format"] == "onnx" and "weights.onnx" in manifest["files"]
    ort.InferenceSession(str(tmp_path / "m" / "weights.onnx"), providers=["CPUExecutionProvider"])
    loaded = ml.load_model_artifact(tmp_path / "m")
    np.testing.assert_array_equal(ml.predict(model, ds).outputs["score"], ml.predict(loaded, ds).outputs["score"])
    record = ml.export_model(loaded, tmp_path / "export")
    assert record["format"] == "onnx" and (tmp_path / "export" / "model.onnx").is_file()
