"""The PyTorch backend and the two #669 reference archetypes, in miniature.

(a) the IRE shape: an engineered-feature table in two blocks, one
autoencoder per block, a threshold calibrated on held-out shots;
(b) the SXR shape: paired (channel x time) inputs to a 2-D target, split by
simulation case, whose prediction keeps its 2-D shape.
"""

from __future__ import annotations

import numpy as np
import pytest

from vaft.process import ml

torch = pytest.importorskip("torch")

_FAST = ml.TrainingConfig(epochs=60, batch_size=32, learning_rate=5e-3, seed=11, early_stopping_patience=None)


def _low_rank(n_groups=16, per_group=40, width=8, seed=0):
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((n_groups * per_group, 2))
    x = latent @ rng.standard_normal((2, width)) + 0.01 * rng.standard_normal((n_groups * per_group, width))
    return x, np.repeat(np.arange(n_groups), per_group)


def test_cpu_training_is_deterministic_for_one_seed():
    x, g = _low_rank()
    ds = ml.build_dataset(x, g, spec=ml.DatasetSpec(name="d", task="t"))
    split = ml.split_groups(ds)
    spec = ml.ModelSpec(architecture="autoencoder", task="det", hyperparameters={"hidden_dims": [8, 2]})
    a = ml.train_model(ds, split, spec, _FAST).artifact
    b = ml.train_model(ds, split, spec, _FAST).artifact
    c = ml.train_model(ds, split, spec, ml.TrainingConfig(**{**_FAST.to_dict(), "seed": 12})).artifact
    assert a.state_sha256 == b.state_sha256 != c.state_sha256
    assert a.training["runtimes"]["torch"] == torch.__version__


def test_autoencoder_learns_a_low_rank_signal_and_scores_an_outlier():
    x, g = _low_rank()
    ds = ml.build_dataset(x, g, spec=ml.DatasetSpec(name="d", task="t"))
    split = ml.split_groups(ds)
    # GELU through a width-2 bottleneck: with ReLU the 2-unit code dies on some
    # seeds and the reconstruction plateaus near 0.1-0.8 of the input variance.
    spec = ml.ModelSpec(
        architecture="autoencoder", task="det", hyperparameters={"hidden_dims": [16, 2], "activation": "gelu"}
    )
    config = ml.TrainingConfig(epochs=300, learning_rate=3e-3, seed=11, early_stopping_patience=30)
    model = ml.train_model(ds, split, spec, config).artifact
    test_score = ml.evaluate_model(model, ds, split, "test").metrics["score_median"]
    assert test_score < 0.01  # normalised units; the input variance is 1
    outlier = x[:1] + 10 * x.std(axis=0)
    assert ml.predict(model, outlier).outputs["score"][0] > 50 * test_score


def test_early_stopping_keeps_the_best_validation_epoch():
    x, g = _low_rank()
    ds = ml.build_dataset(x, g, spec=ml.DatasetSpec(name="d", task="t"))
    split = ml.split_groups(ds)
    spec = ml.ModelSpec(architecture="autoencoder", task="det", hyperparameters={"hidden_dims": [8, 2]})
    config = ml.TrainingConfig(epochs=200, learning_rate=5e-3, early_stopping_patience=3, seed=1)
    result = ml.train_model(ds, split, spec, config)
    history = result.history["validation_loss"]
    assert result.best_epoch == int(np.argmin(history))
    assert len(history) <= 200 and len(history) - 1 - result.best_epoch <= 3
    assert result.artifact.metrics["validation_loss_best"] == min(history)


def test_ire_shaped_two_block_detector_with_calibrated_thresholds(tmp_path):
    x, shots = _low_rank(width=6)
    fspec = ml.FeatureSpec(
        feature_names=("ip_mean", "ip_std", "dipdt_max", "ha_mean", "ha_std", "dhadt_max"),
        blocks={"stage1": ("ip_mean", "ip_std", "dipdt_max"), "stage2": ("ha_mean", "ha_std", "dhadt_max")},
        version="v4_spike_coincidence",
    )
    table = ml.build_dataset(x, shots + 40000, spec=ml.DatasetSpec(name="ire", task="ire", group_key="shot"),
                             feature_spec=fspec)
    split = ml.split_groups(table, ml.SplitSpec(seed=20011001))
    models = {}
    for block in ("stage1", "stage2"):
        data = table.select_block(block)
        spec = ml.ModelSpec(architecture="autoencoder", task=f"ire_{block}", hyperparameters={"hidden_dims": [8, 2]})
        model = ml.train_model(data, split, spec, _FAST).artifact
        cal = ml.calibrate_threshold(model, data, split, quantile=0.9, name=block)
        models[block] = ml.save_model_artifact(model.with_calibration(cal), tmp_path / block, version="0.1.0")
    stage1 = ml.predict(ml.load_model_artifact(tmp_path / "stage1"), table.select_block("stage1"))
    assert stage1.outputs["stage1_exceeds"].dtype == bool
    assert stage1.model_identity.manifest_sha256 == models["stage1"].identity.manifest_sha256
    with pytest.raises(ml.ModelContractError, match="columns"):
        ml.predict(models["stage1"], table.select_block("stage2"))


def test_sxr_shaped_paired_reconstruction_keeps_its_2d_target():
    rng = np.random.default_rng(2)
    n_cases, crops = 24, 12
    x = rng.standard_normal((n_cases * crops, 8, 6))           # channels x time
    w = rng.standard_normal((48, 20)) / 7
    y = np.tanh(x.reshape(len(x), -1) @ w).reshape(len(x), 5, 4)  # 2-D emissivity
    ds = ml.build_dataset(x, np.repeat(np.arange(n_cases), crops), targets=y,
                          spec=ml.DatasetSpec(name="sxr", task="sxr_recon", group_key="case"))
    split = ml.split_groups(ds)  # by case, before any crop is drawn
    spec = ml.ModelSpec(architecture="mlp", task="sxr_recon", hyperparameters={"hidden": [64]})
    model = ml.train_model(ds, split, spec, ml.TrainingConfig(epochs=150, learning_rate=3e-3, seed=0)).artifact
    result = ml.predict(model, ds)
    assert result.outputs["prediction"].shape == (len(x), 5, 4)
    assert ml.evaluate_model(model, ds, split, "test").metrics["r2"] > 0.8


def test_torch_artifact_round_trips(tmp_path):
    x, g = _low_rank()
    ds = ml.build_dataset(x, g, spec=ml.DatasetSpec(name="d", task="t"))
    split = ml.split_groups(ds)
    spec = ml.ModelSpec(architecture="autoencoder", task="rt", hyperparameters={"hidden_dims": [8, 2]})
    model = ml.save_model_artifact(ml.train_model(ds, split, spec, _FAST).artifact, tmp_path / "m", version="0.1.0")
    assert (tmp_path / "m" / "weights.pt").is_file()
    loaded = ml.load_model_artifact(tmp_path / "m")
    np.testing.assert_array_equal(ml.predict(model, ds).outputs["score"], ml.predict(loaded, ds).outputs["score"])


def test_onnx_export_matches_the_torch_model(tmp_path):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    rng = np.random.default_rng(0)
    x = rng.standard_normal((120, 4, 5))
    y = x.reshape(120, -1)[:, :6].reshape(120, 3, 2)
    ds = ml.build_dataset(x, np.repeat(np.arange(12), 10), targets=y, spec=ml.DatasetSpec(name="e", task="t"))
    split = ml.split_groups(ds)
    model = ml.train_model(ds, split, ml.ModelSpec(architecture="mlp", task="e"), ml.TrainingConfig(epochs=5)).artifact
    record = ml.export_model(model, tmp_path / "onnx")
    assert record["parity_max_abs_diff"] <= record["parity_tolerance"]
    session = ort.InferenceSession(str(tmp_path / "onnx" / "model.onnx"), providers=["CPUExecutionProvider"])
    norm = np.load(tmp_path / "onnx" / "normalization.npz")
    xn = ((x - norm["input_mean"]) / norm["input_std"]).astype(np.float32)
    out = session.run(None, {"input": xn})[0] * norm["target_std"] + norm["target_mean"]
    np.testing.assert_allclose(out, ml.predict(model, ds).outputs["prediction"], atol=1e-4)
