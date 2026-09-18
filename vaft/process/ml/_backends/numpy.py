"""Dependency-free NumPy backend.

Closed-form or memory-based models with exact, framework-free solutions --
ridge regression for paired targets, a PCA autoencoder for reconstruction,
and a k-nearest-neighbour distance detector -- so the whole lifecycle (train,
save, resolve, verify, infer) runs and is tested without an ML framework.
"""

from __future__ import annotations

import io

import numpy as np

from .._types import ModelContractError
from . import Backend, mse

#: Query rows per block in the k-NN distance computation, bounding memory.
_KNN_BLOCK = 1024


def _pack(**arrays) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def _unpack(state: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(state), allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def _flat(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(len(x), -1)


def _kth_distance(queries: np.ndarray, reference: np.ndarray, k: int, exclude_self: bool) -> np.ndarray:
    """Euclidean distance from each query to its k-th nearest reference row.

    With ``exclude_self`` the queries *are* the reference rows, and each row's
    own zero distance is skipped -- the k-th neighbour among the *others*.
    """
    scores = np.empty(len(queries))
    ref_sq = np.einsum("ij,ij->i", reference, reference)
    for start in range(0, len(queries), _KNN_BLOCK):
        block = queries[start:start + _KNN_BLOCK]
        sq = np.einsum("ij,ij->i", block, block)[:, None] + ref_sq[None, :] - 2.0 * block @ reference.T
        np.maximum(sq, 0.0, out=sq)
        if exclude_self:
            rows = np.arange(len(block))
            sq[rows, start + rows] = np.inf
        scores[start:start + len(block)] = np.sqrt(np.partition(sq, k - 1, axis=1)[:, k - 1])
    return scores


class NumpyBackend(Backend):
    name = "numpy"
    inference_format = "numpy-npz"
    weights_suffix = ".npz"
    builtin = {"ridge": "supervised", "pca_autoencoder": "reconstruction", "knn_distance": "score"}

    def fit(self, spec, config, x_train, y_train, x_val, y_val, *, output_shape, normalization, augment=None):
        self.kind(spec.architecture)
        if augment is not None:
            raise ModelContractError("the numpy backend trains in closed form; augmentation needs an iterative backend")
        if "loss" in spec.hyperparameters:
            raise ModelContractError("the numpy backend has no configurable loss")
        hp = dict(spec.hyperparameters)
        X = _flat(x_train)
        train_scores = None
        if spec.architecture == "ridge":
            alpha = float(hp.get("alpha", 1.0e-3))
            Y = _flat(y_train)
            design = np.hstack([X, np.ones((len(X), 1))])
            penalty = np.sqrt(alpha) * np.eye(design.shape[1])
            penalty[-1, -1] = 0.0  # the intercept is not shrunk
            coef, *_ = np.linalg.lstsq(
                np.vstack([design, penalty]),
                np.vstack([Y, np.zeros((design.shape[1], Y.shape[1]))]),
                rcond=None,
            )
            state = _pack(weight=coef[:-1], bias=coef[-1])
        elif spec.architecture == "pca_autoencoder":
            n_components = int(hp.get("n_components", min(8, X.shape[1])))
            if not 1 <= n_components <= min(X.shape):
                raise ModelContractError(
                    f"n_components={n_components} must be in [1, {min(X.shape)}] for {X.shape} training data"
                )
            mean = X.mean(axis=0)
            _, _, vt = np.linalg.svd(X - mean, full_matrices=False)
            state = _pack(mean=mean, components=vt[:n_components])
        else:  # knn_distance
            k = int(hp.get("k", 5))
            if not 1 <= k < len(X):
                raise ModelContractError(f"k={k} must be in [1, {len(X) - 1}] for {len(X)} training samples")
            state = _pack(reference=X, k=np.asarray(k))
            train_scores = _kth_distance(X, X, k, exclude_self=True)
        history: dict[str, list[float]] = {}
        if spec.architecture != "knn_distance":
            target = y_train if y_train is not None else x_train
            history["train_loss"] = [mse(self.predict(spec, state, x_train, output_shape), target)]
            if x_val is not None and len(x_val):
                val_target = y_val if y_val is not None else x_val
                history["validation_loss"] = [mse(self.predict(spec, state, x_val, output_shape), val_target)]
        return state, history, 0, train_scores

    def predict(self, spec, state, x, output_shape):
        arrays = _unpack(state)
        X = _flat(x)
        if spec.architecture == "ridge":
            out = X @ arrays["weight"] + arrays["bias"]
        elif spec.architecture == "pca_autoencoder":
            latent = (X - arrays["mean"]) @ arrays["components"].T
            out = latent @ arrays["components"] + arrays["mean"]
        else:
            return _kth_distance(X, arrays["reference"], int(arrays["k"]), exclude_self=False)
        return out.reshape((len(X), *output_shape))


BACKEND = NumpyBackend()
