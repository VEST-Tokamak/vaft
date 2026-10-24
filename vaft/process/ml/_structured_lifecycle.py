"""Structured dispatch behind the existing training/evaluation/inference APIs."""
from __future__ import annotations

import numpy as np

from ._backends import get_backend
from ._provenance import software_provenance, sha256_file
from ._types import ModelArtifact, ModelContractError, TrainingResult, EvaluationResult, InferenceResult
from .structured import fit_moments


def train_structured(dataset, split, spec, config, *, name=None, train_mask=None, train_mask_record=None):
    if spec.backend != "torch":
        raise ModelContractError("structured observations are supported only by the torch backend")
    if config.augmentation:
        raise ModelContractError("dense augmentation is not a structured transform; apply source transforms before collation")
    if not dataset.fingerprint:
        raise ModelContractError("build structured data with build_structured_dataset before training")
    backend = get_backend(spec.backend)
    if backend.kind(spec.architecture) != "supervised":
        raise ModelContractError("structured training currently supports supervised registered architectures")
    fitted = set(str(g) for g in dataset.spec.metadata.get("fitted_transform_groups", ()))
    if fitted - set(split.groups(config.train_partition)):
        raise ModelContractError("preprocessing was fitted on groups outside the training partition")
    train = split.mask(dataset, config.train_partition) & dataset.available_mask
    initial = train.copy()
    if train_mask is not None:
        mask = np.asarray(train_mask)
        if mask.dtype != bool or mask.shape != train.shape:
            raise ModelContractError("train_mask must be boolean with one entry per sample")
        train &= mask
    if not train.any():
        raise ModelContractError("no available training samples")
    val = (split.mask(dataset, config.validation_partition) & dataset.available_mask
           if config.validation_partition is not None else np.zeros(dataset.n_samples, bool))
    if (train & val).any():
        raise ModelContractError("training and validation partitions must be distinct")
    from .usage import _fit_uses
    data_uses = _fit_uses(dataset, split, config, np.flatnonzero(train), np.flatnonzero(val))
    moments = fit_moments(dataset, np.flatnonzero(train), config)
    context = {"dataset_fingerprint": dataset.fingerprint, "split": split.to_dict(), "train_indices": np.flatnonzero(train).tolist()}
    state, history, best_epoch, resources = backend.fit_structured(spec, config, dataset,
        np.flatnonzero(train), np.flatnonzero(val), moments, context)
    record = {**software_provenance(backend.runtime_versions()), **context,
        "dataset": {"spec": dataset.spec.to_dict(), "groups": split.to_dict()["groups"]},
        "seed": config.seed, "config": config.to_dict(), "history": history, "best_epoch": best_epoch,
        "epochs_run": len(history["train_loss"]), "resources": resources,
        "train_mask": None if train_mask is None else {"n_train_partition": int(initial.sum()),
            "n_kept": int(train.sum()), "record": dict(train_mask_record or {})}}
    if data_uses:
        record["data_uses"] = data_uses
    if config.resume_from or config.warm_start_from:
        path = config.resume_from or config.warm_start_from
        record["initial_state"] = {"mode": "resume" if config.resume_from else "warm_start", "sha256": sha256_file(path)}
    model = ModelArtifact(name=name or spec.task, model_spec=spec, state=state, input_shape=(), target_shape=(),
        kind="supervised", outputs=tuple(dataset.target_specs), input_names=tuple(dataset.input_specs),
        tensor_schema=dataset.schema(), normalization=moments, training=record,
        metrics={f"{key}_best": float(values[best_epoch]) for key, values in history.items() if values})
    return TrainingResult(artifact=model, split=split, history={k: tuple(v) for k, v in history.items()},
                          best_epoch=best_epoch, train_indices=np.flatnonzero(train))


def predict_structured(model, data):
    if model.calibration:
        raise ModelContractError("structured calibration is not qualified; apply calibration in the domain adapter")
    if not model.tensor_schema or model.tensor_schema != data.schema():
        raise ModelContractError("structured dataset tensor schema differs from model schema")
    backend = get_backend(model.model_spec.backend)
    mask = data.available_mask
    outputs = {k: np.full((data.n_samples, *s.shape), np.nan) for k, s in data.target_specs.items()}
    if mask.any():
        values = backend.predict_structured(model.model_spec, model.state, data.subset(np.flatnonzero(mask)),
            model.normalization, int(model.training.get("config", {}).get("batch_size", 64)))
        for name in outputs:
            outputs[name][mask] = values[name]
    if data.available is not None:
        outputs["available"] = mask.copy()
    return InferenceResult(outputs=outputs, model_identity=model.resolved_identity(), groups=data.groups,
        sample_meta=data.sample_meta, provenance={"input": "structured_dataset", "dataset_fingerprint": data.fingerprint})


def evaluate_structured(model, data, split, partition, metrics):
    from .training import _check_holdout
    _check_holdout(model, split, partition, allow_train_partition=True)
    selection = split.mask(data, partition) & data.available_mask
    indices = np.flatnonzero(selection)
    if not len(indices):
        return EvaluationResult(partition=partition, metrics={}, per_group={}, n_samples=0, model_identity=model.resolved_identity())
    subset = data.subset(indices)
    outputs = dict(predict_structured(model, subset).outputs)
    def measured(local_indices):
        result = {}
        for name in data.target_specs:
            residuals = []
            for j in local_indices:
                sample = subset.sample(j)
                valid = sample.target_masks[name]
                residuals.extend((outputs[name][j] - sample.targets[name])[valid].ravel().tolist())
            if residuals:
                a = np.asarray(residuals)
                result[f"{name}/mse"] = float(np.mean(a**2))
                result[f"{name}/mae"] = float(np.mean(np.abs(a)))
        if metrics is not None:
            result.update({k: float(v) for k, v in metrics({k: v[local_indices] for k, v in outputs.items()},
                          subset.subset(local_indices)).items()})
        return result
    labels = subset.group_labels
    return EvaluationResult(partition=partition, metrics=measured(np.arange(len(indices))),
        per_group={g: measured(np.flatnonzero(labels == g)) for g in sorted(set(labels))},
        n_samples=len(indices), model_identity=model.resolved_identity())
