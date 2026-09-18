"""Training, evaluation and threshold calibration (#669 sections 6-8)."""

from __future__ import annotations

import numpy as np

from ._backends import get_backend
from ._provenance import software_provenance
from ._types import (
    CalibrationResult,
    DatasetArtifact,
    EvaluationResult,
    FeatureDataset,
    ModelArtifact,
    ModelContractError,
    ModelSpec,
    Split,
    TrainingConfig,
    TrainingResult,
)

__all__ = ["calibrate_threshold", "evaluate_model", "train_model"]

#: A standard deviation below this is treated as a constant column.
_STD_FLOOR = 1.0e-12


def _moments(values: np.ndarray, enabled: bool) -> tuple[np.ndarray, np.ndarray]:
    shape = values.shape[1:]
    if not enabled:
        return np.zeros(shape), np.ones(shape)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    return mean, np.where(std > _STD_FLOOR, std, 1.0)


def _input_names(dataset: DatasetArtifact) -> tuple[str, ...]:
    if isinstance(dataset, FeatureDataset):
        return dataset.feature_spec.feature_names
    return dataset.spec.input_names


def _dataset_record(dataset: DatasetArtifact, split: Split) -> dict:
    record = {
        "fingerprint": dataset.fingerprint,
        "spec": dataset.spec.to_dict(),
        "n_samples": {p: int(split.mask(dataset, p).sum()) for p in split.partitions},
    }
    if isinstance(dataset, FeatureDataset):
        record["feature_spec"] = dataset.feature_spec.to_dict()
    return record


def train_model(dataset, split, model_spec, config=None, *, name=None):
    """Train a model on the train partition and package it with its provenance.

    The one training entry point for every backend and architecture.  What
    comes back is a :class:`ModelArtifact`, not a framework object: its
    trained state is opaque bytes and everything that identifies it -- data,
    split, configuration, software -- travels with it.

    Parameters
    ----------
    dataset : DatasetArtifact
        Fingerprinted training data from :func:`build_dataset` [-].
    split : Split
        Group assignment from :func:`split_groups`, made before any windowing [-].
    model_spec : ModelSpec
        Backend, architecture, task and hyperparameters [-].
    config : TrainingConfig or None, optional
        Optimisation settings; ``None`` uses the defaults [-].
    name : str or None, optional
        Model name used when publishing; ``model_spec.task`` when ``None`` [-].

    Returns
    -------
    TrainingResult
        The artifact, the per-epoch loss history, the split and the epoch whose
        weights were kept [-].

    Raises
    ------
    ModelContractError
        When the dataset has no fingerprint, a group is missing from the split,
        the train partition is empty, or a supervised architecture has no
        targets.
    BackendUnavailableError
        When the backend's framework is not installed.

    Processing steps
    ----------------
    1. Select the train and validation partitions through the split's group
       assignment.
    2. Compute input (and target) normalisation moments per element on the
       train partition only.
    3. Fit the backend on the normalised arrays; an unsupervised architecture
       learns to reproduce its input.  The validation loss drives early
       stopping and the best epoch's weights are kept.
    4. Record dataset fingerprint and spec, split groups, model spec,
       configuration, software revision and runtimes, and the loss history.

    Defaults
    --------
    The ``TrainingConfig`` defaults (Adam, learning rate 1e-3, batch 64,
    patience 10) are numerical convenience, not tuned for any task.

    Convention
    ----------
    Normalisation is a per-element z-score, ``(x - mean) / std``, with a
    constant element given ``std = 1``; the reconstruction ``score`` of an
    unsupervised model is measured in these normalised units.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [vaft669] VEST-Tokamak/vaft#669, sections 6 and 7: training as a VAFT
       processing stage, dataset provenance as part of model identity.
    """
    config = config or TrainingConfig()
    if not isinstance(model_spec, ModelSpec):
        raise ModelContractError("model_spec must be a ModelSpec")
    if not dataset.fingerprint:
        raise ModelContractError("dataset has no fingerprint; build it with build_dataset")
    backend = get_backend(model_spec.backend)
    supervised = backend.is_supervised(model_spec.architecture)
    if supervised and dataset.targets is None:
        raise ModelContractError(f"architecture {model_spec.architecture!r} needs targets")
    train_mask = split.mask(dataset, config.train_partition)
    if not train_mask.any():
        raise ModelContractError(f"partition {config.train_partition!r} is empty")
    val_mask = (
        split.mask(dataset, config.validation_partition)
        if config.validation_partition is not None
        else np.zeros(dataset.n_samples, dtype=bool)
    )
    x = np.asarray(dataset.inputs, dtype=np.float64)
    x_mean, x_std = _moments(x[train_mask], config.normalize_inputs)
    xn = (x - x_mean) / x_std
    normalization = {"input_mean": x_mean, "input_std": x_std}
    if supervised:
        y = np.asarray(dataset.targets, dtype=np.float64)
        y_mean, y_std = _moments(y[train_mask], config.normalize_targets)
        yn = (y - y_mean) / y_std
        normalization.update(target_mean=y_mean, target_std=y_std)
        outputs = ("prediction",)
        target_shape = dataset.target_shape
    else:
        yn = xn
        outputs = ("reconstruction", "score")
        target_shape = None
    state, history, best_epoch = backend.fit(
        model_spec,
        config,
        xn[train_mask],
        yn[train_mask],
        xn[val_mask] if val_mask.any() else None,
        yn[val_mask] if val_mask.any() else None,
    )
    history = {k: tuple(float(v) for v in vs) for k, vs in history.items()}
    pick = best_epoch if best_epoch is not None else -1
    metrics = {f"{k}_best": vs[pick] for k, vs in history.items() if vs}
    training = {
        **software_provenance(backend.runtime_versions()),
        "dataset_fingerprint": dataset.fingerprint,
        "dataset": _dataset_record(dataset, split),
        "split": split.to_dict(),
        "seed": config.seed,
        "config": config.to_dict(),
        "best_epoch": best_epoch,
        "epochs_run": len(history.get("train_loss", ())),
        "history": {k: list(v) for k, v in history.items()},
    }
    artifact = ModelArtifact(
        name=name or model_spec.task,
        model_spec=model_spec,
        state=state,
        input_shape=dataset.input_shape,
        target_shape=target_shape,
        supervised=supervised,
        outputs=outputs,
        normalization=normalization,
        input_names=_input_names(dataset),
        training=training,
        metrics=metrics,
    )
    return TrainingResult(artifact=artifact, history=history, split=split, best_epoch=best_epoch)


def _metrics(artifact: ModelArtifact, outputs, dataset: DatasetArtifact, mask: np.ndarray) -> dict:
    if not mask.any():
        return {}
    if artifact.supervised:
        prediction = outputs["prediction"][mask]
        target = np.asarray(dataset.targets, dtype=np.float64)[mask]
        residual = prediction - target
        total = np.sum((target - target.mean(axis=0)) ** 2)
        return {
            "mse": float(np.mean(residual**2)),
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "mae": float(np.mean(np.abs(residual))),
            "r2": float(1.0 - np.sum(residual**2) / total) if total > 0 else float("nan"),
        }
    score = outputs["score"][mask]
    residual = outputs["reconstruction"][mask] - np.asarray(dataset.inputs, dtype=np.float64)[mask]
    return {
        "score_mean": float(np.mean(score)),
        "score_median": float(np.median(score)),
        "score_p95": float(np.percentile(score, 95)),
        "reconstruction_mse": float(np.mean(residual**2)),
    }


def evaluate_model(model, dataset, split, partition="test"):
    """Standard ML metrics of a model on one partition, overall and per group.

    This is training evaluation.  Whether a model is scientifically valid --
    campaign robustness, negative controls, cross-diagnostic timing -- is a
    task-specific verdict for ``vaft.validation``, not a number here.

    Parameters
    ----------
    model : ModelArtifact
        Trained or loaded model [-].
    dataset : DatasetArtifact
        Data with the model's input shape [-].
    split : Split
        Group assignment selecting the partition [-].
    partition : str, optional
        Partition to evaluate [-].

    Returns
    -------
    EvaluationResult
        Aggregate and per-group metrics: ``mse``, ``rmse``, ``mae`` and ``r2``
        in target units for a supervised model; ``score_mean``,
        ``score_median``, ``score_p95`` in normalised units and
        ``reconstruction_mse`` in input units for an unsupervised one [any].

    Applicability
    -------------
    Machine-independent.
    """
    from .inference import _forward

    outputs = _forward(model, np.asarray(dataset.inputs))
    mask = split.mask(dataset, partition)
    labels = dataset.group_labels
    per_group = {
        g: _metrics(model, outputs, dataset, mask & (labels == g)) for g in split.groups(partition)
        if (mask & (labels == g)).any()
    }
    return EvaluationResult(
        partition=partition,
        metrics=_metrics(model, outputs, dataset, mask),
        per_group=per_group,
        n_samples=int(mask.sum()),
        model_identity=model.resolved_identity(),
    )


def calibrate_threshold(model, dataset, split, *, partition="validation", output="score", quantile=0.99, name=None):
    """Set a decision threshold on a per-sample model output from one partition.

    The second half of a two-stage detector: a score is only an event
    candidate once a threshold, fixed on data the network did not train on,
    has been applied.  Attach the result with
    ``model.with_calibration(result)``; :func:`predict` then reports
    ``<name>_exceeds`` for every sample.

    Parameters
    ----------
    model : ModelArtifact
        Trained or loaded model [-].
    dataset : DatasetArtifact
        Data with the model's input shape [-].
    split : Split
        Group assignment selecting the partition [-].
    partition : str, optional
        Partition the quantile is taken over [-].
    output : str, optional
        Per-sample scalar output to threshold [-].
    quantile : float, optional
        Quantile of that output on the partition, in ``(0, 1)`` [-].
    name : str or None, optional
        Calibration name; ``output`` when ``None`` [-].

    Returns
    -------
    CalibrationResult
        Threshold with the partition, quantile, sample count and groups it was
        set on [any].

    Raises
    ------
    ModelContractError
        When the output is not a per-sample scalar, the partition is empty,
        or ``quantile`` is outside ``(0, 1)``.

    Defaults
    --------
    ``quantile = 0.99`` is numerical convenience; a detector's quantile is a
    task decision (the VEST IRE prototype used 0.85-0.90).

    Applicability
    -------------
    Machine-independent.
    """
    from .inference import _forward

    if not 0.0 < quantile < 1.0:
        raise ModelContractError(f"quantile must be in (0, 1); got {quantile}")
    outputs = _forward(model, np.asarray(dataset.inputs))
    if output not in outputs or outputs[output].ndim != 1:
        raise ModelContractError(f"output {output!r} is not a per-sample scalar of this model")
    mask = split.mask(dataset, partition)
    if not mask.any():
        raise ModelContractError(f"partition {partition!r} is empty")
    return CalibrationResult(
        name=name or output,
        output=output,
        threshold=float(np.quantile(outputs[output][mask], quantile)),
        method="quantile",
        quantile=float(quantile),
        partition=partition,
        n_samples=int(mask.sum()),
        groups=tuple(g for g in split.groups(partition) if (mask & (dataset.group_labels == g)).any()),
    )

