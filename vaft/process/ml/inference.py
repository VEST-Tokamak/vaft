"""Inference that always records which exact model produced the result."""

from __future__ import annotations

import hashlib

import numpy as np

from ._backends import get_backend
from ._types import DatasetArtifact, FeatureDataset, InferenceResult, ModelContractError

__all__ = ["predict"]


def _forward(model, x: np.ndarray, available: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Normalise, run the backend, undo the normalisation, apply calibrations.

    Unavailable samples are never shown to the network: their outputs are NaN
    and every ``<name>_exceeds`` is False, and an ``available`` output says so.
    """
    x = np.asarray(x, dtype=np.float64)
    if tuple(x.shape[1:]) != tuple(model.input_shape):
        raise ModelContractError(
            f"model {model.name!r} expects inputs of shape (n, {', '.join(map(str, model.input_shape))}); "
            f"got {x.shape}"
        )
    ok = np.ones(len(x), dtype=bool) if available is None else np.asarray(available, dtype=bool)
    norm = model.normalization
    xn = (x[ok] - norm["input_mean"]) / norm["input_std"]
    backend = get_backend(model.model_spec.backend)
    shape = model.target_shape if model.supervised else model.input_shape

    def full(values: np.ndarray) -> np.ndarray:
        out = np.full((len(x), *values.shape[1:]), np.nan)
        out[ok] = values
        return out

    out = backend.predict(model.model_spec, model.state, xn, shape) if ok.any() else None
    if model.kind == "supervised":
        values = out if out is not None else np.zeros((0, *shape))
        outputs = {"prediction": full(values * norm["target_std"] + norm["target_mean"])}
    elif model.kind == "reconstruction":
        values = out if out is not None else np.zeros((0, *shape))
        axes = tuple(range(1, values.ndim))
        outputs = {
            "reconstruction": full(values * norm["input_std"] + norm["input_mean"]),
            "score": full(np.mean((values - xn) ** 2, axis=axes) if len(values) else np.zeros(0)),
        }
    else:
        outputs = {"score": full(np.asarray(out if out is not None else np.zeros(0), dtype=np.float64))}
    for calibration in model.calibration.values():
        with np.errstate(invalid="ignore"):
            outputs[f"{calibration.name}_exceeds"] = ok & (outputs[calibration.output] > calibration.threshold)
    if available is not None:
        outputs["available"] = ok.copy()
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
        Outputs aligned with the samples (``prediction``; ``reconstruction``
        and a per-sample ``score``; or ``score`` alone; plus
        ``<name>_exceeds`` for every calibration and, for a dataset with an
        availability mask, ``available``), the model identity, and the groups
        and per-sample metadata of a dataset input [any].

    Raises
    ------
    ModelContractError
        When the input shape, or a dataset's column names, differ from the
        model's.

    Output semantics
    ----------------
    ``prediction`` and ``reconstruction`` are in the units of the targets and
    inputs; an autoencoder's ``score`` is the mean squared reconstruction
    error in the model's normalised units, and a score model's is the
    detector's own anomaly measure (larger is more anomalous).  An
    unavailable sample has NaN outputs and exceeds no threshold: missing
    data is not "no anomaly".  A score or an exceedance is a model output, not a
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
        available = data.available
        groups, meta = data.groups, dict(data.sample_meta)
        provenance = {"input": "dataset", "dataset_fingerprint": data.fingerprint}
    else:
        x = np.asarray(data)
        available = None
        groups, meta = None, {}
        digest = hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
        provenance = {"input": "array", "input_sha256": digest}
    outputs = _forward(model, x, available)
    return InferenceResult(
        outputs=outputs,
        model_identity=model.resolved_identity(),
        groups=groups,
        sample_meta=meta,
        provenance=provenance,
    )
