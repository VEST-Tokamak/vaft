"""Training/inference backends behind the backend-neutral contract.

A backend turns normalised NumPy arrays into opaque ``bytes`` of trained state
and back into predictions.  It never sees a :class:`DatasetArtifact`, a split
or provenance -- those stay in the framework-free layer -- and nothing it
defines appears in a public signature.  Backends are looked up by name, so
``import vaft.process.ml`` never imports an ML framework.

Every architecture declares what it learns (its *kind*):

``"supervised"``
    maps inputs to paired targets; output ``prediction``.
``"reconstruction"``
    reproduces its input; outputs ``reconstruction`` and a per-sample
    ``score`` (mean squared normalised residual).
``"score"``
    returns a per-sample anomaly ``score`` directly; larger is more anomalous.

Task code can add architectures, losses and augmentations to a backend by
name (:func:`register_architecture`, :func:`register_loss`,
:func:`register_augmentation`).  The registries are per process: a task
module registers what it defines when it is imported, and loading a model
whose architecture nobody registered fails naming it.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable, Mapping

import numpy as np

from .._types import BackendUnavailableError, ModelContractError, ModelSpec, TrainingConfig

_REGISTRY = {
    "numpy": ".numpy",
    "sklearn": ".sklearn",
    "torch": ".torch",
}

KINDS = ("supervised", "reconstruction", "score")

#: Outputs each kind produces, in order.
KIND_OUTPUTS = {
    "supervised": ("prediction",),
    "reconstruction": ("reconstruction", "score"),
    "score": ("score",),
}


@dataclass(frozen=True)
class Architecture:
    """A named network shape: its kind and, for registered ones, its factory."""

    name: str
    kind: str
    factory: Callable[..., Any] | None = None


#: Architectures registered by task code, per backend: {backend: {name: Architecture}}.
_REGISTERED: dict[str, dict[str, Architecture]] = {}
#: Losses registered by task code: {name: factory(spec, normalization) -> loss(pred, target, inputs)}.
LOSSES: dict[str, Callable[..., Any]] = {}
#: Augmentations: {name: fn(inputs, targets, rng) -> (inputs, targets)}, in physical units.
AUGMENTATIONS: dict[str, Callable[..., Any]] = {}


class Backend:
    """Interface every backend implements."""

    name: str = ""
    inference_format: str = ""
    weights_suffix: str = ""
    #: built-in architecture name -> kind
    builtin: Mapping[str, str] = {}
    #: whether the backend trains iteratively (and so can augment per epoch)
    iterative: bool = False
    #: whether task code may register architectures with this backend
    extensible: bool = False
    #: kinds a registered architecture of this backend may declare
    trainable_kinds: tuple[str, ...] = ()

    def runtime_versions(self) -> dict[str, str]:
        return {}

    def architecture(self, name: str) -> Architecture:
        registered = _REGISTERED.get(self.name, {})
        if name in registered:
            return registered[name]
        if name in self.builtin:
            return Architecture(name=name, kind=self.builtin[name])
        known = sorted({*self.builtin, *registered})
        raise ModelContractError(
            f"backend {self.name!r} has no architecture {name!r}; known: {known}. "
            "A task-defined architecture must be registered (register_architecture) "
            "before a model that uses it is trained or loaded."
        )

    def kind(self, architecture: str) -> str:
        return self.architecture(architecture).kind

    def fit(
        self,
        spec: ModelSpec,
        config: TrainingConfig,
        x_train: np.ndarray,
        y_train: np.ndarray | None,
        x_val: np.ndarray | None,
        y_val: np.ndarray | None,
        *,
        output_shape: tuple[int, ...],
        normalization: Mapping[str, np.ndarray],
        augment: Callable | None = None,
    ) -> tuple[bytes, dict[str, list[float]], int | None, np.ndarray | None]:
        """Train; return (state, history, best_epoch, train_scores or None)."""
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


def _check_name(what: str, name: str) -> None:
    if not isinstance(name, str) or not name or not name.replace("_", "").isalnum():
        raise ModelContractError(f"{what} name {name!r} must be a non-empty identifier")


def register_architecture(name, factory, *, kind, backend="torch"):
    """Make a task-defined network available to a backend by name.

    Task code (the SXR reconstruction transformer, a sequence model) defines
    its network next to its science and registers it here; a
    :class:`ModelSpec` then selects it with ``architecture=name``.  The
    factory is called only when a model is built, so registering imports no
    ML framework.

    Parameters
    ----------
    name : str
        Architecture name a ``ModelSpec`` refers to [-].
    factory : callable
        ``factory(spec, input_shape, output_shape)`` returning a network that
        maps a batch of shape ``(n, *input_shape)`` to ``(n, *output_shape)``
        without flattening it first [-].
    kind : {"supervised", "reconstruction", "score"}
        What the network learns [-].
    backend : str, optional
        Backend the factory builds for; only ``"torch"`` accepts registrations [-].

    Returns
    -------
    None
        The architecture is registered for this process [-].

    Raises
    ------
    ModelContractError
        On an unknown kind, a backend that does not accept registrations, a
        name that shadows a built-in, or a second registration of one name
        with a different factory.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    The registry is per process.  A published model records its
    architecture's name, not its code; loading it needs the same task module
    imported, and the code version is traced through the VAFT revision in the
    manifest.
    """
    if kind not in KINDS:
        raise ModelContractError(f"kind must be one of {KINDS}; got {kind!r}")
    _check_name("architecture", name)
    target = get_backend(backend)
    if not target.extensible:
        raise ModelContractError(f"backend {backend!r} does not accept registered architectures")
    if kind not in target.trainable_kinds:
        raise ModelContractError(f"backend {backend!r} trains {target.trainable_kinds}, not {kind!r}")
    if name in target.builtin:
        raise ModelContractError(f"{name!r} is a built-in {backend} architecture")
    registered = _REGISTERED.setdefault(backend, {})
    existing = registered.get(name)
    if existing is not None and (existing.factory is not factory or existing.kind != kind):
        raise ModelContractError(f"architecture {name!r} is already registered for {backend!r}")
    registered[name] = Architecture(name=name, kind=kind, factory=factory)


def register_loss(name, factory):
    """Make a task-defined training loss available by name.

    Selected with ``ModelSpec(hyperparameters={"loss": name})``.  The factory
    receives the model spec and the train-partition normalisation, so a
    physics loss -- a line-of-sight residual through a projector acting on
    emissivity in physical units -- can undo the normalisation the network
    works in.

    Parameters
    ----------
    name : str
        Loss name [-].
    factory : callable
        ``factory(spec, normalization)`` returning ``loss(prediction, target,
        inputs)``, a differentiable scalar in the backend's tensor type.
        ``prediction``, ``target`` and ``inputs`` are normalised and keep
        their sample shapes; ``normalization`` maps ``input_mean``,
        ``input_std`` and, for a supervised model, ``target_mean`` and
        ``target_std`` to NumPy arrays that broadcast against one sample [-].

    Returns
    -------
    None
        The loss is registered for this process [-].

    Raises
    ------
    ModelContractError
        On a name that is already registered with a different factory, or
        that shadows the built-in ``"mse"``.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Only the iterative ``torch`` backend trains with a loss; the closed-form
    and scikit-learn backends refuse a ``loss`` hyperparameter.
    """
    _check_name("loss", name)
    if name == "mse":
        raise ModelContractError("'mse' is the built-in loss")
    if LOSSES.get(name, factory) is not factory:
        raise ModelContractError(f"loss {name!r} is already registered")
    LOSSES[name] = factory


def register_augmentation(name, function):
    """Make a train-time augmentation available by name.

    Selected with ``TrainingConfig(augmentation=name)``.  It is applied to the
    **train partition only**, once per epoch, after the group split -- so an
    augmented copy of a sample can never reach validation or test.

    Parameters
    ----------
    name : str
        Augmentation name [-].
    function : callable
        ``function(inputs, targets, rng)`` returning ``(inputs, targets)`` of
        the same shapes, in physical (un-normalised) units; ``targets`` is
        ``None`` for an unsupervised model and ``rng`` is a seeded
        ``numpy.random.Generator`` [-].

    Returns
    -------
    None
        The augmentation is registered for this process [-].

    Raises
    ------
    ModelContractError
        On a name already registered with a different function.

    Applicability
    -------------
    Machine-independent.
    """
    _check_name("augmentation", name)
    if AUGMENTATIONS.get(name, function) is not function:
        raise ModelContractError(f"augmentation {name!r} is already registered")
    AUGMENTATIONS[name] = function


__all__ = [
    "AUGMENTATIONS",
    "Architecture",
    "Backend",
    "BackendUnavailableError",
    "KINDS",
    "KIND_OUTPUTS",
    "LOSSES",
    "get_backend",
    "mse",
    "register_architecture",
    "register_augmentation",
    "register_loss",
]
