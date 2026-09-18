"""Training, evaluation and threshold calibration (#669 sections 6-8)."""

from __future__ import annotations

import numpy as np

from ._backends import AUGMENTATIONS, KIND_OUTPUTS, get_backend
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

#: A standard deviation below this is treated as a constant element.
_STD_FLOOR = 1.0e-12


def _moments(values: np.ndarray, enabled: bool, axes: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std over the sample axis and ``axes``, broadcastable to one sample."""
    shape = values.shape[1:]
    if not enabled:
        return np.zeros(shape), np.ones(shape)
    if any(a >= values.ndim for a in axes):
        raise ModelContractError(f"normalisation axes {axes} do not exist for samples of shape {shape}")
    pooled = (0, *axes)
    mean = values.mean(axis=pooled, keepdims=True)[0]
    std = values.std(axis=pooled, keepdims=True)[0]
    return mean, np.where(std > _STD_FLOOR, std, 1.0)


def _input_names(dataset: DatasetArtifact) -> tuple[str, ...]:
    if isinstance(dataset, FeatureDataset):
        return dataset.feature_spec.feature_names
    return dataset.spec.input_names


def _dataset_record(dataset: DatasetArtifact, split: Split) -> dict:
    available = dataset.available_mask
    record = {
        "fingerprint": dataset.fingerprint,
        "spec": dataset.spec.to_dict(),
        "n_samples": {p: int(split.mask(dataset, p).sum()) for p in split.partitions},
        "n_unavailable": int((~available).sum()),
    }
    if isinstance(dataset, FeatureDataset):
        record["feature_spec"] = dataset.feature_spec.to_dict()
    return record


def _augmenter(name, normalization, supervised):
    """Wrap a physical-units augmentation so the backend can apply it to normalised arrays."""
    if name is None:
        return None
    if name not in AUGMENTATIONS:
        raise ModelContractError(f"augmentation {name!r} is not registered; known: {sorted(AUGMENTATIONS)}")
    function = AUGMENTATIONS[name]
    xm, xs = normalization["input_mean"], normalization["input_std"]

    def augment(x_norm, y_norm, rng):
        x = x_norm * xs + xm
        y = None
        if supervised:
            y = y_norm * normalization["target_std"] + normalization["target_mean"]
        x_out, y_out = function(x, y, rng)
        x_out = np.asarray(x_out, dtype=np.float64)
        if x_out.shape != x.shape:
            raise ModelContractError(f"augmentation {name!r} changed the input shape {x.shape} -> {x_out.shape}")
        if not supervised:
            return (x_out - xm) / xs, None
        y_out = np.asarray(y_out, dtype=np.float64)
        if y_out.shape != y.shape:
            raise ModelContractError(f"augmentation {name!r} changed the target shape {y.shape} -> {y_out.shape}")
        return (x_out - xm) / xs, (y_out - normalization["target_mean"]) / normalization["target_std"]

    return augment


def train_model(dataset, split, model_spec, config=None, *, name=None, train_mask=None, train_mask_record=None):
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
        Optimisation, normalisation and augmentation settings; ``None`` uses
        the defaults [-].
    name : str or None, optional
        Model name used when publishing; ``model_spec.task`` when ``None`` [-].
    train_mask : array_like of bool or None, optional
        Per-sample mask further restricting the fitted samples within the
        train partition -- training-set cleaning; ``None`` keeps them all [-].
    train_mask_record : mapping or None, optional
        How ``train_mask`` was made (method, threshold, source model), stored
        in the provenance next to the kept and removed counts [-].

    Returns
    -------
    TrainingResult
        The artifact, the per-epoch history, the split, the epoch whose
        weights were kept, the fitted sample indices and, for a score model
        that can leave a sample out (k-NN), the leave-self-out training
        scores [-].

    Raises
    ------
    ModelContractError
        When the dataset has no fingerprint, a group is missing from the split,
        no sample is left to fit, a supervised architecture has no targets, or
        a named augmentation or loss is not registered.
    BackendUnavailableError
        When the backend's framework is not installed.

    Processing steps
    ----------------
    1. Select the fitted samples: train partition, available, and
       ``train_mask``; select the available validation samples.
    2. Compute input (and target) normalisation moments on the fitted samples
       only, per element or pooled over ``config.normalize_axes``.
    3. Fit the backend on the normalised arrays; a reconstruction architecture
       learns to reproduce its input.  A registered augmentation is applied to
       the fitted samples only, once per epoch.  The validation loss drives
       early stopping and the best epoch's weights are kept.
    4. Record dataset fingerprint and spec, split groups, the train-mask
       record, model spec, configuration, software revision and runtimes, and
       the loss history.

    Defaults
    --------
    The ``TrainingConfig`` defaults (Adam, learning rate 1e-3, batch 64,
    patience 10, per-element normalisation) are numerical convenience, not
    tuned for any task.

    Convention
    ----------
    Normalisation is a z-score, ``(x - mean) / std``, with a constant element
    given ``std = 1``; the reconstruction ``score`` of an autoencoder is
    measured in these normalised units.  A score model's ``score`` is larger
    for more anomalous samples.

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
    kind = backend.kind(model_spec.architecture)
    supervised = kind == "supervised"
    if supervised and dataset.targets is None:
        raise ModelContractError(f"architecture {model_spec.architecture!r} needs targets")
    available = dataset.available_mask
    partition_mask = split.mask(dataset, config.train_partition) & available
    fit_mask = partition_mask.copy()
    if train_mask is not None:
        extra = np.asarray(train_mask)
        if extra.dtype != bool or extra.shape != (dataset.n_samples,):
            raise ModelContractError(f"train_mask must be a boolean array of shape ({dataset.n_samples},)")
        fit_mask &= extra
    if not fit_mask.any():
        raise ModelContractError(f"no available sample of partition {config.train_partition!r} is left to fit")
    val_mask = (
        split.mask(dataset, config.validation_partition) & available
        if config.validation_partition is not None
        else np.zeros(dataset.n_samples, dtype=bool)
    )
    x = np.asarray(dataset.inputs, dtype=np.float64)
    x_mean, x_std = _moments(x[fit_mask], config.normalize_inputs, config.normalize_axes)
    xn = (x - x_mean) / x_std
    normalization = {"input_mean": x_mean, "input_std": x_std}
    yn = None
    if supervised:
        y = np.asarray(dataset.targets, dtype=np.float64)
        y_mean, y_std = _moments(y[fit_mask], config.normalize_targets, config.target_normalize_axes)
        yn = (y - y_mean) / y_std
        normalization.update(target_mean=y_mean, target_std=y_std)
        output_shape = dataset.target_shape
    else:
        output_shape = dataset.input_shape
    has_val = bool(val_mask.any())
    state, history, best_epoch, train_scores = backend.fit(
        model_spec,
        config,
        xn[fit_mask],
        yn[fit_mask] if supervised else None,
        xn[val_mask] if has_val else None,
        yn[val_mask] if supervised and has_val else None,
        output_shape=output_shape,
        normalization=normalization,
        augment=_augmenter(config.augmentation, normalization, supervised),
    )
    history = {k: tuple(float(v) for v in vs) for k, vs in history.items()}
    pick = best_epoch if best_epoch is not None else -1
    metrics = {f"{k}_best": vs[pick] for k, vs in history.items() if vs}
    mask_record = None
    if train_mask is not None:
        mask_record = {
            "n_train_partition": int(partition_mask.sum()),
            "n_kept": int(fit_mask.sum()),
            "n_removed": int(partition_mask.sum() - fit_mask.sum()),
            "record": dict(train_mask_record or {}),
        }
    training = {
        **software_provenance(backend.runtime_versions()),
        "dataset_fingerprint": dataset.fingerprint,
        "dataset": _dataset_record(dataset, split),
        "split": split.to_dict(),
        "seed": config.seed,
        "config": config.to_dict(),
        "train_mask": mask_record,
        "loss": str(model_spec.hyperparameters.get("loss", "mse")) if backend.iterative else None,
        "best_epoch": best_epoch,
        "epochs_run": len(history.get("train_loss", ())),
        "history": {k: list(v) for k, v in history.items()},
    }
    artifact = ModelArtifact(
        name=name or model_spec.task,
        model_spec=model_spec,
        state=state,
        input_shape=dataset.input_shape,
        target_shape=dataset.target_shape if supervised else None,
        kind=kind,
        outputs=KIND_OUTPUTS[kind],
        normalization=normalization,
        input_names=_input_names(dataset),
        training=training,
        metrics=metrics,
    )
    return TrainingResult(
        artifact=artifact,
        history=history,
        split=split,
        best_epoch=best_epoch,
        train_indices=np.flatnonzero(fit_mask),
        train_scores=None if train_scores is None else np.asarray(train_scores, dtype=float),
    )


def _metrics(artifact: ModelArtifact, outputs, dataset: DatasetArtifact, mask: np.ndarray) -> dict:
    if not mask.any():
        return {}
    if artifact.kind == "supervised":
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
    result = {
        "score_mean": float(np.mean(score)),
        "score_median": float(np.median(score)),
        "score_p95": float(np.percentile(score, 95)),
    }
    if artifact.kind == "reconstruction":
        residual = outputs["reconstruction"][mask] - np.asarray(dataset.inputs, dtype=np.float64)[mask]
        result["reconstruction_mse"] = float(np.mean(residual**2))
    return result


def evaluate_model(model, dataset, split, partition="test", *, metrics=None):
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
    metrics : callable or None, optional
        Task metrics, ``metrics(outputs, subset)`` returning a mapping of names
        to floats, where ``outputs`` are the model outputs and ``subset`` the
        :class:`DatasetArtifact` of the evaluated samples; evaluated overall
        and per group, and merged into the built-in metrics [-].

    Returns
    -------
    EvaluationResult
        Aggregate and per-group metrics over the *available* samples:
        ``mse``, ``rmse``, ``mae`` and ``r2`` in target units for a supervised
        model; ``score_mean``, ``score_median``, ``score_p95`` (normalised
        units for an autoencoder, the detector's own for a score model) and,
        for an autoencoder, ``reconstruction_mse`` in input units [any].

    Applicability
    -------------
    Machine-independent.
    """
    from .inference import _forward

    outputs = _forward(model, np.asarray(dataset.inputs), dataset.available)
    mask = split.mask(dataset, partition) & dataset.available_mask
    labels = dataset.group_labels

    def measured(selection):
        values = _metrics(model, outputs, dataset, selection)
        if metrics is not None and selection.any():
            idx = np.flatnonzero(selection)
            extra = metrics({k: v[idx] for k, v in outputs.items()}, dataset.subset(idx))
            values.update({str(k): float(v) for k, v in dict(extra).items()})
        return values

    per_group = {
        g: measured(mask & (labels == g)) for g in split.groups(partition) if (mask & (labels == g)).any()
    }
    return EvaluationResult(
        partition=partition,
        metrics=measured(mask),
        per_group=per_group,
        n_samples=int(mask.sum()),
        model_identity=model.resolved_identity(),
    )


def calibrate_threshold(
    model, dataset, split, *, partition="validation", output="score", quantile=0.99, name=None,
    mask=None, population=None,
):
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
    mask : array_like of bool or None, optional
        Further restricts the population within the partition -- a second
        stage calibrated only on the first stage's candidates [-].
    population : str or None, optional
        Description of ``mask`` stored with the result, e.g.
        ``"stage1_exceeds"`` [-].

    Returns
    -------
    CalibrationResult
        Threshold with the partition, population, quantile, sample count and
        groups it was set on [any].

    Raises
    ------
    ModelContractError
        When the output is not a per-sample scalar, the population is empty,
        ``quantile`` is outside ``(0, 1)``, ``mask`` is given without a
        ``population`` description, or a score model is calibrated on the
        partition it was fitted on -- a k-NN scores its own training samples
        against themselves, so that threshold would be too low; use
        ``TrainingResult.train_scores`` for a training-population statistic.

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
    fitted_on = model.training.get("config", {}).get("train_partition")
    if model.kind == "score" and partition == fitted_on:
        raise ModelContractError(
            f"a score model scores the samples it was fitted on in-sample; calibrate on another partition "
            f"than {partition!r}, or use TrainingResult.train_scores"
        )
    outputs = _forward(model, np.asarray(dataset.inputs), dataset.available)
    if output not in outputs or outputs[output].ndim != 1:
        raise ModelContractError(f"output {output!r} is not a per-sample scalar of this model")
    selected = split.mask(dataset, partition) & dataset.available_mask
    if mask is not None:
        extra = np.asarray(mask)
        if extra.dtype != bool or extra.shape != (dataset.n_samples,):
            raise ModelContractError(f"mask must be a boolean array of shape ({dataset.n_samples},)")
        if not population:
            raise ModelContractError("a calibration mask needs a population description")
        selected &= extra
    if not selected.any():
        raise ModelContractError(f"no available sample of partition {partition!r} is in the population")
    return CalibrationResult(
        name=name or output,
        output=output,
        threshold=float(np.quantile(outputs[output][selected], quantile)),
        method="quantile",
        quantile=float(quantile),
        partition=partition,
        n_samples=int(selected.sum()),
        groups=tuple(g for g in split.groups(partition) if (selected & (dataset.group_labels == g)).any()),
        population=population,
    )
