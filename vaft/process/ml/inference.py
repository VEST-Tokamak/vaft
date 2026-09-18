"""Inference that always records which exact model produced the result."""

from __future__ import annotations

import hashlib

import numpy as np

from ._backends import get_backend
from ._types import DatasetArtifact, FeatureDataset, InferenceResult, ModelContractError

__all__ = ["predict"]


def _forward(model, x: np.ndarray) -> dict[str, np.ndarray]:
    """Normalise, run the backend, undo the normalisation, apply calibrations."""
    x = np.asarray(x, dtype=np.float64)
    if tuple(x.shape[1:]) != tuple(model.input_shape):
        raise ModelContractError(
            f"model {model.name!r} expects inputs of shape (n, {', '.join(map(str, model.input_shape))}); "
            f"got {x.shape}"
        )
    norm = model.normalization
    xn = (x - norm["input_mean"]) / norm["input_std"]
    backend = get_backend(model.model_spec.backend)
    shape = model.target_shape if model.supervised else model.input_shape
    out = backend.predict(model.model_spec, model.state, xn, shape)
    if model.supervised:
        outputs = {"prediction": out * norm["target_std"] + norm["target_mean"]}
    else:
        axes = tuple(range(1, out.ndim))
        outputs = {
            "reconstruction": out * norm["input_std"] + norm["input_mean"],
            "score": np.mean((out - xn) ** 2, axis=axes) if len(x) else np.zeros(0),
        }
    for calibration in model.calibration.values():
        outputs[f"{calibration.name}_exceeds"] = outputs[calibration.output] > calibration.threshold
    return outputs


def predict(model, data):
    """Run a model on new data and stamp the result with its exact identity.

    Accepts a dataset (checked against the model's feature schema) or a bare
    array.  The result records the model's immutable identity -- name, exact
    version, manifest and weight hashes -- never an alias such as
    ``production``, so a stored result keeps meaning what it meant when it was
    produced.

    Parameters
    ----------
    model : ModelArtifact
        Trained, loaded or resolved model [-].
    data : DatasetArtifact or array_like
        Inputs of shape ``(n, *model.input_shape)`` [any].

    Returns
    -------
    InferenceResult
        Outputs aligned with the samples (``prediction``; or
        ``reconstruction`` and a per-sample ``score``; plus ``<name>_exceeds``
        for every calibration), the model identity, and the groups and
        per-sample metadata of a dataset input [any].

    Raises
    ------
    ModelContractError
        When the input shape, or a dataset's column names, differ from the
        model's.

    Output semantics
    ----------------
    ``prediction`` and ``reconstruction`` are in the units of the targets and
    inputs; ``score`` is the mean squared reconstruction error in the model's
    normalised units.  A score or an exceedance is a model output, not a
    physical event: an ``IRE-like candidate``, never a reconnection onset
    (#669 section 16).

    Applicability
    -------------
    Machine-independent.
    """
    if isinstance(data, DatasetArtifact):
        names = data.feature_spec.feature_names if isinstance(data, FeatureDataset) else data.spec.input_names
        if names and model.input_names and tuple(names) != tuple(model.input_names):
            raise ModelContractError(
                f"dataset columns {list(names)[:4]}... differ from the columns model "
                f"{model.name!r} was trained on {list(model.input_names)[:4]}..."
            )
        x = np.asarray(data.inputs)
        groups, meta = data.groups, dict(data.sample_meta)
        provenance = {"input": "dataset", "dataset_fingerprint": data.fingerprint}
    else:
        x = np.asarray(data)
        groups, meta = None, {}
        digest = hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
        provenance = {"input": "array", "input_sha256": digest}
    outputs = _forward(model, x)
    return InferenceResult(
        outputs=outputs,
        model_identity=model.resolved_identity(),
        groups=groups,
        sample_meta=meta,
        provenance=provenance,
    )
