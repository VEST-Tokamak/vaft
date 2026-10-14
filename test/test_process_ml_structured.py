"""Structured masks, train-only statistics, streaming and checkpoint semantics."""
from dataclasses import replace
import pickle

import numpy as np
import pytest

from vaft.process import ml

torch = pytest.importorskip("torch")


def architecture(spec, inputs, outputs):
    class SetNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(2, 1)
        def forward(self, x):
            mask = x["valid"] & x["tokens__valid"]
            pooled = (x["tokens"] * mask[..., None]).sum(1) / mask.sum(1).clamp(min=1)[:, None]
            return {"value": self.head(pooled)}
    return SetNet()


ml.register_architecture("contract_set", architecture, kind="supervised")


def dataset():
    rows = []
    for i in range(12):
        x = np.full((1 + i % 3, 2), float(i), np.float32)
        valid = np.ones(len(x), bool)
        valid[-1] = i % 2 == 0
        x[~valid] = np.nan
        rows.append(ml.StructuredSample({"tokens": x, "valid": valid},
                    {"value": np.array([i * .5], np.float32)}, {"value": np.array([i % 4 != 0])}))
    return ml.build_structured_dataset(rows, np.repeat(["train", "val", "test"], 4),
        input_specs={"tokens": ml.TensorSpec((None, 2), mask="valid", unit="[A]"),
                     "valid": ml.TensorSpec((None,), "bool", normalize=False)},
        target_specs={"value": ml.TensorSpec((1,), unit="[Wb]")},
        spec=ml.DatasetSpec(name="set", task="regress", group_key="shot"),
        sample_meta={"machine": np.repeat("DIII-D", 12), "time": np.arange(12)})


def split(data):
    return ml.split_groups(data, ml.SplitSpec(fixed={"train": "train", "val": "validation", "test": "test"}))


SPEC = ml.ModelSpec(architecture="contract_set", task="set_test")
CFG = ml.TrainingConfig(epochs=4, batch_size=2, early_stopping_patience=None, optimizer="adamw",
                        scheduler="onecycle", max_grad_norm=1.0)


def test_masked_moments_and_artifact_inference(tmp_path):
    data = dataset()
    result = ml.train_model(data, split(data), SPEC, CFG)
    expected = np.concatenate([r.inputs["tokens"][r.inputs["valid"]] for r in data.source[:4]]).mean(0)
    np.testing.assert_allclose(result.artifact.normalization["input.tokens.mean"], expected)
    assert set(result.history) >= {"train_loss", "train_value", "validation_loss", "learning_rate"}
    saved = ml.save_model_artifact(result.artifact, tmp_path / "model", version="0.1.0")
    back = ml.load_model_artifact(tmp_path / "model")
    np.testing.assert_array_equal(ml.predict(saved, data).outputs["value"], ml.predict(back, data).outputs["value"])
    evaluation = ml.evaluate_model(back, data, split(data), "test")
    assert evaluation.n_samples == 4 and np.isfinite(evaluation.metrics["value/mse"])
    changed = ml.Split(spec=split(data).spec, assignment={"train": "test", "test": "train", "val": "validation"})
    with pytest.raises(ml.ModelContractError, match="fitted on"):
        ml.evaluate_model(back, data, changed, "test")


def test_padding_and_missing_observations_are_not_measurements():
    from vaft.process.ml._backends._torch_engine import StructuredSamples, StructuredCollator
    from vaft.process.ml.structured import fit_moments
    data = dataset()
    samples = StructuredSamples(data, [0, 1, 2], fit_moments(data, [0, 1, 2, 3], CFG))
    inputs, targets, masks = StructuredCollator(data.input_specs)([samples[i] for i in range(3)])
    net = architecture(SPEC, {}, {})
    with torch.no_grad():
        before = net(inputs)["value"]
        inputs["tokens"][~(inputs["valid"] & inputs["tokens__valid"])] = 1e8
        after = net(inputs)["value"]
    torch.testing.assert_close(before, after)
    assert inputs["tokens__valid"].sum() == 6
    assert torch.isfinite(inputs["tokens"]).all()


def test_exact_epoch_resume_and_changed_split_rejection(tmp_path):
    data = dataset()
    full = ml.train_model(data, split(data), SPEC, CFG)
    checkpoint = str(tmp_path / "training.pt")
    partial = ml.train_model(data, split(data), SPEC, replace(CFG, checkpoint_path=checkpoint, max_epochs_this_run=2))
    assert len(partial.history["train_loss"]) == 2
    resumed = ml.train_model(data, split(data), SPEC, replace(CFG, resume_from=checkpoint))
    assert full.history == resumed.history
    assert full.artifact.state_sha256 == resumed.artifact.state_sha256
    with pytest.raises(ml.ModelContractError, match="resume checkpoint differs"):
        ml.train_model(data, split(data), SPEC, replace(CFG, resume_from=checkpoint, seed=8))


def test_picklable_lazy_contract_and_worker_reproducibility():
    data = dataset()
    back = pickle.loads(pickle.dumps(data))
    assert back.schema() == data.schema()
    config = replace(CFG, epochs=1, scheduler=None, loader_workers=1)
    a = ml.train_model(data, split(data), SPEC, config)
    b = ml.train_model(back, split(back), SPEC, config)
    assert a.artifact.state_sha256 == b.artifact.state_sha256


def test_structured_backends_and_invalid_values_rejected():
    data = dataset()
    with pytest.raises(ml.ModelContractError, match="only by the torch"):
        ml.train_model(data, split(data), ml.ModelSpec(architecture="ridge", backend="numpy", task="bad"), CFG)
    bad = list(data.source)
    bad[0] = ml.StructuredSample({"tokens": np.full((1, 2), np.nan), "valid": np.ones(1, bool)},
                                 bad[0].targets, bad[0].target_masks)
    with pytest.raises(ml.ModelContractError, match="nonfinite"):
        ml.build_structured_dataset(bad, data.groups, input_specs=data.input_specs, target_specs=data.target_specs, spec=data.spec)
