"""scikit-learn backend, stored as ONNX.

Training uses scikit-learn; the trained estimator is converted to ONNX with
``skl2onnx`` and accepted only after ONNX Runtime reproduces the estimator on
a probe batch.  The stored state is the ONNX graph, so loading a published
model never unpickles anything, and inference needs ``onnxruntime`` alone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .._types import BackendUnavailableError, ModelContractError
from . import Backend

_INSTALL_HINT = "pip install 'vaft[ml]'"

#: ONNX opsets written; ai.onnx.ml carries the SVM operators.
_OPSETS = {"": 17, "ai.onnx.ml": 3}


def _require(*names):
    modules = []
    for name in names:
        try:
            modules.append(__import__(name))
        except ImportError as exc:
            raise BackendUnavailableError(
                f"the 'sklearn' ML backend needs {name}, which is not installed ({_INSTALL_HINT})"
            ) from exc
    return modules


def _session(state: bytes):
    (onnxruntime,) = _require("onnxruntime")
    return onnxruntime.InferenceSession(state, providers=["CPUExecutionProvider"])


def _onnx_scores(state: bytes, x: np.ndarray) -> np.ndarray:
    session = _session(state)
    feed = {session.get_inputs()[0].name: np.asarray(x, dtype=np.float32).reshape(len(x), -1)}
    names = [o.name for o in session.get_outputs()]
    if "scores" not in names:
        raise ModelContractError(f"ONNX graph has no 'scores' output; outputs {names}")
    return session.run(["scores"], feed)[0].reshape(len(x)).astype(np.float64)


class SklearnBackend(Backend):
    name = "sklearn"
    inference_format = "onnx"
    weights_suffix = ".onnx"
    builtin = {"one_class_svm": "score"}

    def runtime_versions(self):
        sklearn, skl2onnx, onnxruntime = _require("sklearn", "skl2onnx", "onnxruntime")
        return {
            "scikit-learn": sklearn.__version__,
            "skl2onnx": skl2onnx.__version__,
            "onnxruntime": onnxruntime.__version__,
        }

    def fit(self, spec, config, x_train, y_train, x_val, y_val, *, output_shape, normalization, augment=None):
        self.kind(spec.architecture)
        if augment is not None:
            raise ModelContractError("the sklearn backend does not train per epoch; augmentation is not applied")
        if "loss" in spec.hyperparameters:
            raise ModelContractError("the sklearn backend has no configurable loss")
        _require("sklearn", "skl2onnx", "onnxruntime")
        from skl2onnx import to_onnx
        from sklearn.svm import OneClassSVM

        hp = dict(spec.hyperparameters)
        X = np.asarray(x_train, dtype=np.float32).reshape(len(x_train), -1)
        estimator = OneClassSVM(
            nu=float(hp.get("nu", 0.1)),
            kernel=str(hp.get("kernel", "rbf")),
            gamma=hp.get("gamma", "scale"),
        ).fit(X)
        state = to_onnx(estimator, X[:1], target_opset=_OPSETS).SerializeToString()
        # Parity: the graph that will be published must reproduce the estimator.
        probe = X[: min(len(X), 256)]
        reference = estimator.decision_function(probe)
        exported = _onnx_scores(state, probe)
        tolerance = 1.0e-4 * max(1.0, float(np.max(np.abs(reference))))
        worst = float(np.max(np.abs(exported - reference)))
        if not worst <= tolerance:
            raise ModelContractError(f"ONNX OneClassSVM disagrees with scikit-learn by {worst:.3g} > {tolerance:.3g}")
        # No training scores: an SVM's scores on its own training set are
        # in-sample (support vectors sit on the boundary), not leave-self-out.
        history = {"onnx_parity_max_abs_diff": [worst]}
        return state, history, None, None

    def export(self, spec, state, input_shape, output_shape, path, fmt):
        if fmt != "onnx":
            return super().export(spec, state, input_shape, output_shape, path, fmt)
        # The trained state already is the parity-checked ONNX graph.
        Path(path).write_bytes(state)
        return {"format": "onnx", "opsets": dict(_OPSETS), "parity": "checked at training", **self.runtime_versions()}

    def predict(self, spec, state, x, output_shape):
        # decision_function is positive inside the learned support; the
        # contract's score is larger-is-more-anomalous, hence the sign.
        return -_onnx_scores(state, x)


BACKEND = SklearnBackend()
