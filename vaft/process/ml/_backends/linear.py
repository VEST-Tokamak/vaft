"""Closed-form NumPy reference backend.

Two architectures with exact, dependency-free solutions -- ridge regression
for paired targets and a PCA autoencoder for reconstruction -- so the whole
lifecycle (train, save, resolve, verify, infer) runs and is tested without an
ML framework installed.
"""

from __future__ import annotations

import io

import numpy as np

from .._types import ModelContractError
from . import Backend, mse


def _pack(**arrays) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def _unpack(state: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(state), allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def _flat(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(len(x), -1)


class LinearBackend(Backend):
    name = "linear"
    inference_format = "numpy-npz"
    weights_suffix = ".npz"
    architectures = {"ridge": True, "pca_autoencoder": False}

    def fit(self, spec, config, x_train, y_train, x_val, y_val):
        self.is_supervised(spec.architecture)
        hp = dict(spec.hyperparameters)
        X = _flat(x_train)
        if spec.architecture == "ridge":
            alpha = float(hp.get("alpha", 1.0e-3))
            Y = _flat(y_train)
            design = np.hstack([X, np.ones((len(X), 1))])
            penalty = np.sqrt(alpha) * np.eye(design.shape[1])
            penalty[-1, -1] = 0.0  # the intercept is not shrunk
            lhs = np.vstack([design, penalty])
            rhs = np.vstack([Y, np.zeros((design.shape[1], Y.shape[1]))])
            coef, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
            state = _pack(weight=coef[:-1], bias=coef[-1])
        else:
            n_components = int(hp.get("n_components", min(8, X.shape[1])))
            if not 1 <= n_components <= min(X.shape):
                raise ModelContractError(
                    f"n_components={n_components} must be in [1, {min(X.shape)}] for {X.shape} training data"
                )
            mean = X.mean(axis=0)
            _, _, vt = np.linalg.svd(X - mean, full_matrices=False)
            state = _pack(mean=mean, components=vt[:n_components])
        output_shape = y_train.shape[1:]
        history = {"train_loss": [mse(self.predict(spec, state, x_train, output_shape), y_train)]}
        if x_val is not None and len(x_val):
            history["validation_loss"] = [mse(self.predict(spec, state, x_val, output_shape), y_val)]
        return state, history, 0

    def predict(self, spec, state, x, output_shape):
        arrays = _unpack(state)
        X = _flat(x)
        if spec.architecture == "ridge":
            out = X @ arrays["weight"] + arrays["bias"]
        else:
            latent = (X - arrays["mean"]) @ arrays["components"].T
            out = latent @ arrays["components"] + arrays["mean"]
        return out.reshape((len(X), *output_shape))


BACKEND = LinearBackend()
