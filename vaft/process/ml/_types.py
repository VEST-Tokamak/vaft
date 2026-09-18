"""Backend-neutral records of the ML lifecycle (#669 section 3).

Nothing here imports an ML framework.  Arrays are NumPy arrays; a model's
trained state is opaque ``bytes`` that only its backend interprets, so no
public record carries a ``torch.Tensor`` or ``nn.Module``.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

__all__ = [
    "MLError",
    "ModelContractError",
    "ModelResolutionError",
    "BackendUnavailableError",
    "DatasetSpec",
    "DatasetArtifact",
    "FeatureSpec",
    "FeatureDataset",
    "SplitSpec",
    "Split",
    "ModelSpec",
    "TrainingConfig",
    "TrainingResult",
    "ModelIdentity",
    "CalibrationResult",
    "ModelArtifact",
    "ModelBundle",
    "EvaluationResult",
    "InferenceResult",
]


class MLError(RuntimeError):
    """Base class of every error ``vaft.process.ml`` raises on purpose."""


class ModelContractError(MLError, ValueError):
    """Data, split or model do not satisfy the contract they are used under."""


class ModelResolutionError(MLError):
    """A published model could not be resolved to verified local files."""


class BackendUnavailableError(MLError, ImportError):
    """The requested training/inference backend is not installed."""


def _frozen_mapping(value: Mapping | None) -> Mapping:
    return MappingProxyType(dict(value or {}))


def _jsonable(value: Any) -> Any:
    """Plain JSON types for a manifest; refuses what would not round-trip."""
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if dataclasses.is_dataclass(value):
        return _jsonable(value.to_dict())
    raise TypeError(f"{type(value).__name__} is not JSON-serialisable in a manifest")


def canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class DatasetSpec:
    """What a dataset is: its task, its independent unit and its columns.

    ``group_key`` names the independent scientific unit -- ``"shot"`` for a
    VEST event detector, ``"case"`` for a simulation-trained reconstruction --
    that every split is made over (#669 section 6).
    """

    name: str
    task: str
    group_key: str = "group"
    input_names: tuple[str, ...] = ()
    target_names: tuple[str, ...] = ()
    description: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "input_names", tuple(self.input_names))
        object.__setattr__(self, "target_names", tuple(self.target_names))
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "task": self.task,
            "group_key": self.group_key,
            "input_names": list(self.input_names),
            "target_names": list(self.target_names),
            "description": self.description,
            "metadata": _jsonable(self.metadata),
        }


@dataclass(frozen=True, kw_only=True)
class FeatureSpec:
    """Engineered-feature schema: column names, named blocks and a version.

    ``blocks`` maps a block name (``"stage1"``) to the feature names it holds,
    so a multi-stage model can train one network per block of one table.
    """

    feature_names: tuple[str, ...]
    blocks: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    version: str = ""

    def __post_init__(self):
        names = tuple(self.feature_names)
        if len(set(names)) != len(names):
            raise ModelContractError("FeatureSpec.feature_names contains duplicates")
        blocks = {key: tuple(value) for key, value in dict(self.blocks).items()}
        for key, members in blocks.items():
            unknown = sorted(set(members) - set(names))
            if unknown:
                raise ModelContractError(f"block {key!r} names unknown features {unknown}")
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "blocks", MappingProxyType(blocks))

    def block_indices(self, block: str) -> np.ndarray:
        if block not in self.blocks:
            raise ModelContractError(f"no feature block {block!r}; have {sorted(self.blocks)}")
        position = {name: i for i, name in enumerate(self.feature_names)}
        return np.asarray([position[name] for name in self.blocks[block]], dtype=int)

    def to_dict(self) -> dict:
        return {
            "feature_names": list(self.feature_names),
            "blocks": {k: list(v) for k, v in self.blocks.items()},
            "version": self.version,
        }


@dataclass(frozen=True, kw_only=True, eq=False)
class DatasetArtifact:
    """Samples, optional paired targets and the group each sample belongs to.

    ``inputs`` has shape ``(n, *input_shape)``; ``targets`` is ``None`` for an
    unsupervised dataset or ``(n, *target_shape)`` for paired supervision.
    ``groups`` holds the independent unit of every sample; ``sample_meta``
    carries per-sample arrays (time, window start) that travel with subsets.
    ``available`` marks samples whose inputs exist at all -- an absent
    diagnostic is *unavailable*, which is neither a zero nor a normal sample;
    ``None`` means every sample is available.
    ``fingerprint`` is the SHA-256 :func:`dataset_fingerprint` returns.
    """

    spec: DatasetSpec
    inputs: np.ndarray
    groups: np.ndarray
    targets: np.ndarray | None = None
    sample_meta: Mapping[str, np.ndarray] = field(default_factory=dict)
    available: np.ndarray | None = None
    fingerprint: str = ""

    def __post_init__(self):
        n = len(self.inputs)
        if len(self.groups) != n:
            raise ModelContractError(f"{len(self.groups)} groups for {n} samples")
        if self.targets is not None and len(self.targets) != n:
            raise ModelContractError(f"{len(self.targets)} targets for {n} samples")
        if self.available is not None:
            available = np.asarray(self.available)
            if available.dtype != bool or available.shape != (n,):
                raise ModelContractError(f"available must be a boolean array of shape ({n},)")
            object.__setattr__(self, "available", available)
        for key, value in dict(self.sample_meta).items():
            if len(value) != n:
                raise ModelContractError(f"sample_meta[{key!r}] has {len(value)} rows for {n} samples")
        object.__setattr__(self, "sample_meta", _frozen_mapping(self.sample_meta))

    @property
    def n_samples(self) -> int:
        return int(len(self.inputs))

    @property
    def input_shape(self) -> tuple[int, ...]:
        return tuple(int(s) for s in self.inputs.shape[1:])

    @property
    def target_shape(self) -> tuple[int, ...] | None:
        return None if self.targets is None else tuple(int(s) for s in self.targets.shape[1:])

    @property
    def group_labels(self) -> np.ndarray:
        """Groups as strings, the form a :class:`Split` assigns."""
        return np.asarray([str(g) for g in self.groups])

    @property
    def available_mask(self) -> np.ndarray:
        """``available``, with ``None`` read as every sample available."""
        return np.ones(self.n_samples, dtype=bool) if self.available is None else self.available

    def subset(self, indices) -> "DatasetArtifact":
        """The samples at ``indices``, with their groups, metadata and availability.

        The subset's ``fingerprint`` is empty: it is a view for evaluation, not a
        dataset a model should record as its training data.
        """
        idx = np.asarray(indices)
        return dataclasses.replace(
            self,
            inputs=self.inputs[idx],
            groups=np.asarray(self.groups)[idx],
            targets=None if self.targets is None else self.targets[idx],
            sample_meta={k: np.asarray(v)[idx] for k, v in self.sample_meta.items()},
            available=None if self.available is None else self.available[idx],
            fingerprint="",
        )

    def replace(self, **changes) -> "DatasetArtifact":
        return dataclasses.replace(self, **changes)


@dataclass(frozen=True, kw_only=True, eq=False)
class FeatureDataset(DatasetArtifact):
    """A 2-D engineered-feature table whose columns follow a :class:`FeatureSpec`."""

    feature_spec: FeatureSpec

    def __post_init__(self):
        super().__post_init__()
        if self.inputs.ndim != 2 or self.inputs.shape[1] != len(self.feature_spec.feature_names):
            raise ModelContractError(
                f"a FeatureDataset needs inputs of shape (n, {len(self.feature_spec.feature_names)}); "
                f"got {self.inputs.shape}"
            )

    def select_block(self, block: str) -> "FeatureDataset":
        """The columns of one feature block, as a fingerprinted dataset of its own."""
        from .dataset import dataset_fingerprint

        columns = self.feature_spec.block_indices(block)
        names = self.feature_spec.blocks[block]
        sub = dataclasses.replace(
            self,
            inputs=self.inputs[:, columns],
            feature_spec=FeatureSpec(
                feature_names=names, blocks={block: names}, version=self.feature_spec.version
            ),
            spec=dataclasses.replace(self.spec, input_names=names),
        )
        return dataclasses.replace(sub, fingerprint=dataset_fingerprint(sub))


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class SplitSpec:
    """How groups are assigned to partitions.

    ``fractions`` are shares of *groups*, not of samples.  ``fixed`` pins named
    groups to a partition (a hand-picked test campaign) before the seeded
    assignment distributes the rest.

    ``method`` chooses that assignment.  ``"permutation"`` permutes the sorted
    groups and cuts at the fractions: exact shares, but adding one group can
    move others.  ``"hash"`` places each group by a hash of ``(seed, group)``
    alone: shares are only expected values, but a group never changes
    partition when others are added -- what a growing shot database needs.
    """

    fractions: Mapping[str, float] = field(
        default_factory=lambda: {"train": 0.70, "validation": 0.15, "test": 0.15}
    )
    seed: int = 0
    fixed: Mapping[str, str] = field(default_factory=dict)
    method: str = "permutation"

    def __post_init__(self):
        fractions = {str(k): float(v) for k, v in dict(self.fractions).items()}
        if not fractions or any(v < 0 for v in fractions.values()):
            raise ModelContractError("SplitSpec.fractions must be non-empty and non-negative")
        total = sum(fractions.values())
        if not np.isclose(total, 1.0):
            raise ModelContractError(f"SplitSpec.fractions sum to {total}, not 1")
        fixed = {str(k): str(v) for k, v in dict(self.fixed).items()}
        unknown = sorted(set(fixed.values()) - set(fractions))
        if unknown:
            raise ModelContractError(f"SplitSpec.fixed names unknown partitions {unknown}")
        if self.method not in ("permutation", "hash"):
            raise ModelContractError(f"SplitSpec.method must be 'permutation' or 'hash'; got {self.method!r}")
        object.__setattr__(self, "fractions", MappingProxyType(fractions))
        object.__setattr__(self, "fixed", MappingProxyType(fixed))

    def to_dict(self) -> dict:
        return {
            "fractions": dict(self.fractions),
            "seed": self.seed,
            "fixed": dict(self.fixed),
            "method": self.method,
        }


@dataclass(frozen=True, kw_only=True)
class Split:
    """A group-to-partition assignment, independent of any sample ordering.

    Because it assigns groups and not sample indices, one split applies
    unchanged to the record-level dataset and to every windowed or augmented
    dataset derived from it.  That is what makes "split before windowing"
    hold by construction.
    """

    spec: SplitSpec
    assignment: Mapping[str, str]

    def __post_init__(self):
        object.__setattr__(
            self, "assignment", MappingProxyType({str(k): str(v) for k, v in dict(self.assignment).items()})
        )

    @property
    def partitions(self) -> tuple[str, ...]:
        return tuple(self.spec.fractions)

    def groups(self, partition: str) -> tuple[str, ...]:
        self._check_partition(partition)
        return tuple(sorted(g for g, p in self.assignment.items() if p == partition))

    def mask(self, dataset: DatasetArtifact, partition: str) -> np.ndarray:
        """Samples of ``dataset`` in ``partition``; refuses an unassigned group."""
        self._check_partition(partition)
        labels = dataset.group_labels
        unassigned = sorted(set(labels) - set(self.assignment))
        if unassigned:
            raise ModelContractError(
                f"groups {unassigned[:5]}{'...' if len(unassigned) > 5 else ''} are in the dataset "
                "but not in the split; re-split rather than guessing their partition"
            )
        return np.asarray([self.assignment[g] == partition for g in labels], dtype=bool)

    def indices(self, dataset: DatasetArtifact, partition: str) -> np.ndarray:
        return np.flatnonzero(self.mask(dataset, partition))

    def _check_partition(self, partition: str) -> None:
        if partition not in self.spec.fractions:
            raise ModelContractError(f"no partition {partition!r}; have {list(self.spec.fractions)}")

    def to_dict(self) -> dict:
        return {
            "spec": self.spec.to_dict(),
            "groups": {p: list(self.groups(p)) for p in self.partitions},
        }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class ModelSpec:
    """A reusable architecture bound to a scientific task.

    ``architecture`` names a network shape the backend knows (``"autoencoder"``,
    ``"mlp"``, ``"ridge"``); ``task`` names what it is *for*
    (``"ire_anomaly_stage1"``).  Keeping them apart is #669 section 5.
    """

    architecture: str
    task: str
    backend: str = "torch"
    hyperparameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "hyperparameters", _frozen_mapping(self.hyperparameters))

    def to_dict(self) -> dict:
        return {
            "architecture": self.architecture,
            "task": self.task,
            "backend": self.backend,
            "hyperparameters": _jsonable(self.hyperparameters),
        }


@dataclass(frozen=True, kw_only=True)
class TrainingConfig:
    """Optimisation settings; every field is recorded in the artifact."""

    seed: int = 0
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.0
    early_stopping_patience: int | None = 10
    normalize_inputs: bool = True
    normalize_targets: bool = True
    #: Sample axes (1-based, as in ``inputs.shape``) pooled into one mean/std;
    #: ``()`` is per element, ``(1,)`` on ``(n, time, channel)`` is per channel.
    normalize_axes: tuple[int, ...] = ()
    target_normalize_axes: tuple[int, ...] = ()
    augmentation: str | None = None
    device: str = "cpu"
    train_partition: str = "train"
    validation_partition: str | None = "validation"

    def __post_init__(self):
        for key in ("normalize_axes", "target_normalize_axes"):
            axes = tuple(int(a) for a in getattr(self, key))
            if any(a < 1 for a in axes):
                raise ModelContractError(f"{key} are sample axes counted from 1; got {axes}")
            object.__setattr__(self, key, axes)

    def to_dict(self) -> dict:
        record = dataclasses.asdict(self)
        record["normalize_axes"] = list(self.normalize_axes)
        record["target_normalize_axes"] = list(self.target_normalize_axes)
        return record


@dataclass(frozen=True, kw_only=True)
class ModelIdentity:
    """Which exact model produced a result (#669 sections 9 and 13).

    ``state_sha256`` hashes the trained weights and is always known.
    ``version`` and ``manifest_sha256`` are known once the artifact has been
    saved or resolved from a registry; ``stage`` records the alias that was
    *requested*, never in place of the version it resolved to.
    """

    name: str
    state_sha256: str
    version: str | None = None
    manifest_sha256: str | None = None
    stage: str | None = None
    status: str | None = None
    resolved_by: str = "in-memory"
    source: str | None = None
    #: For a member of a :class:`ModelBundle`: the bundle's ``name@version`` and manifest hash.
    bundle: str | None = None
    bundle_manifest_sha256: str | None = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass(frozen=True, kw_only=True)
class CalibrationResult:
    """A decision threshold on one model output, and the population behind it."""

    name: str
    output: str
    threshold: float
    method: str
    quantile: float | None
    partition: str
    n_samples: int
    groups: tuple[str, ...] = ()
    #: How the calibration population was restricted within the partition, if at all.
    population: str | None = None

    def to_dict(self) -> dict:
        return {**dataclasses.asdict(self), "groups": list(self.groups)}


@dataclass(frozen=True, kw_only=True, eq=False)
class ModelArtifact:
    """A trained model and everything that identifies it (#669 section 7).

    ``state`` is the backend's serialised trained state.  ``normalization``
    holds the train-partition statistics applied before the network.
    ``training`` is the provenance record: dataset fingerprint and spec, split
    groups per partition, seed, config, software revision, loss history.
    """

    name: str
    model_spec: ModelSpec
    state: bytes
    input_shape: tuple[int, ...]
    target_shape: tuple[int, ...] | None
    kind: str
    outputs: tuple[str, ...]
    normalization: Mapping[str, np.ndarray] = field(default_factory=dict)
    input_names: tuple[str, ...] = ()
    training: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    calibration: Mapping[str, CalibrationResult] = field(default_factory=dict)
    identity: ModelIdentity | None = None

    def __post_init__(self):
        object.__setattr__(self, "input_shape", tuple(int(s) for s in self.input_shape))
        if self.target_shape is not None:
            object.__setattr__(self, "target_shape", tuple(int(s) for s in self.target_shape))
        object.__setattr__(self, "outputs", tuple(self.outputs))
        object.__setattr__(self, "input_names", tuple(self.input_names))
        for key in ("normalization", "training", "metrics", "calibration"):
            object.__setattr__(self, key, _frozen_mapping(getattr(self, key)))

    @property
    def supervised(self) -> bool:
        return self.kind == "supervised"

    @property
    def state_sha256(self) -> str:
        from ._provenance import sha256_bytes

        return sha256_bytes(self.state)

    def resolved_identity(self) -> ModelIdentity:
        """The identity to stamp on a result: the resolved one, else in-memory."""
        if self.identity is not None:
            return self.identity
        return ModelIdentity(name=self.name, state_sha256=self.state_sha256)

    def with_calibration(self, calibration: CalibrationResult) -> "ModelArtifact":
        """A copy carrying ``calibration``; the identity is dropped.

        A calibrated model is a different model: its manifest changes, so the
        identity of the uncalibrated one must not follow it.
        """
        merged = {**self.calibration, calibration.name: calibration}
        return dataclasses.replace(self, calibration=merged, identity=None)

    def replace(self, **changes) -> "ModelArtifact":
        return dataclasses.replace(self, **changes)


@dataclass(frozen=True, kw_only=True)
class TrainingResult:
    """The trained artifact, its per-epoch history and the split it used."""

    artifact: ModelArtifact
    history: Mapping[str, tuple[float, ...]]
    split: Split
    best_epoch: int | None
    #: Dataset indices the model was fitted on (train partition, available, ``train_mask``).
    train_indices: np.ndarray | None = None
    #: For a ``score`` model whose backend can leave a sample out (the k-NN), the
    #: score of each fitted sample computed *without* it -- what training-set
    #: cleaning needs; ``None`` where only in-sample scores exist (the SVM).
    train_scores: np.ndarray | None = None


@dataclass(frozen=True, kw_only=True, eq=False)
class ModelBundle:
    """Several fitted models released as one logical model.

    A two-stage detector publishes its stages together: one version, one
    manifest pinning every member.  ``members`` maps a role (``"stage1"``) to
    its artifact; ``composition`` is a free, JSON-serialisable record of how
    the task combines them (stage order, event-rule version).  Combining the
    members is task code; the bundle only keeps them together and identified.
    """

    name: str
    members: Mapping[str, ModelArtifact]
    composition: Mapping[str, Any] = field(default_factory=dict)
    identity: ModelIdentity | None = None

    def __post_init__(self):
        if not self.members:
            raise ModelContractError("a ModelBundle needs at least one member")
        for role in self.members:
            if not role or not role.replace("_", "").isalnum():
                raise ModelContractError(f"bundle member role {role!r} must be an identifier")
        object.__setattr__(self, "members", MappingProxyType(dict(self.members)))
        object.__setattr__(self, "composition", _frozen_mapping(self.composition))

    def replace(self, **changes) -> "ModelBundle":
        return dataclasses.replace(self, **changes)


@dataclass(frozen=True, kw_only=True)
class EvaluationResult:
    """Metrics of one model on one partition, overall and per group.

    This is training evaluation, not scientific validation: a verdict on
    whether a model may be promoted belongs to ``vaft.validation``
    (#669 section 8).
    """

    partition: str
    metrics: Mapping[str, float]
    per_group: Mapping[str, Mapping[str, float]]
    n_samples: int
    model_identity: ModelIdentity


@dataclass(frozen=True, kw_only=True, eq=False)
class InferenceResult:
    """Model outputs aligned with the input samples.

    ``outputs`` maps an output name to an array whose first axis is the sample
    axis: a scalar ``score`` per window, a ``reconstruction`` image, a
    ``prediction`` profile.  Nothing is collapsed to one scalar (#669 section 3).
    ``model_identity`` is always the immutable resolved identity.
    """

    outputs: Mapping[str, np.ndarray]
    model_identity: ModelIdentity
    groups: np.ndarray | None = None
    sample_meta: Mapping[str, np.ndarray] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "outputs", _frozen_mapping(self.outputs))
        object.__setattr__(self, "sample_meta", _frozen_mapping(self.sample_meta))
        object.__setattr__(self, "provenance", _frozen_mapping(self.provenance))
