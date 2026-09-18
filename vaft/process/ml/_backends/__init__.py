"""Training/inference backends behind the backend-neutral contract.

A backend turns normalised NumPy arrays into opaque ``bytes`` of trained state
and back into predictions.  It never sees a :class:`DatasetArtifact`, a split
or provenance -- those stay in the framework-free layer -- and nothing it
defines appears in a public signature.  Backends are looked up by name, so
``import vaft.process.ml`` never imports an ML framework.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Mapping

import numpy as np

from .._types import BackendUnavailableError, ModelContractError, ModelSpec, TrainingConfig

_REGISTRY = {
    "linear": ".linear",
    "torch": ".torch",
}


class Backend:
    """Interface every backend implements."""

    name: str = ""
    inference_format: str = ""
    weights_suffix: str = ""
    #: architecture name -> whether it learns a paired target
    architectures: Mapping[str, bool] = {}

    def runtime_versions(self) -> dict[str, str]:
        return {}

    def is_supervised(self, architecture: str) -> bool:
        if architecture not in self.architectures:
            raise ModelContractError(
                f"backend {self.name!r} has no architecture {architecture!r}; "
                f"known: {sorted(self.architectures)}"
            )
        return self.architectures[architecture]

    def fit(
        self,
        spec: ModelSpec,
        config: TrainingConfig,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None,
        y_val: np.ndarray | None,
    ) -> tuple[bytes, dict[str, list[float]], int | None]:
        raise NotImplementedError

    def predict(self, spec: ModelSpec, state: bytes, x: np.ndarray, output_shape: tuple[int, ...]) -> np.ndarray:
        raise NotImplementedError

    def export(self, spec: ModelSpec, state: bytes, input_shape, output_shape, path, fmt: str) -> dict[str, Any]:
        raise ModelContractError(f"backend {self.name!r} cannot export to {fmt!r}")


def get_backend(name: str) -> Backend:
    if name not in _REGISTRY:
        raise ModelContractError(f"unknown ML backend {name!r}; known: {sorted(_REGISTRY)}")
    module = import_module(_REGISTRY[name], __name__)
    return module.BACKEND


def mse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((prediction - target) ** 2)) if prediction.size else float("nan")


__all__ = ["Backend", "BackendUnavailableError", "get_backend", "mse"]
