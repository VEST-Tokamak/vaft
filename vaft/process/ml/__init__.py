"""Backend-neutral machine-learning processing: data to published model (#669).

Training is a VAFT processing stage like any other.  This package owns the
whole path -- dataset construction, group-aware splitting, windowing,
training, evaluation, threshold calibration, artifact packaging, export,
resolution of published versions and inference -- while trained weights are
distributed through the separate ``VEST-Tokamak/vaft-nn`` registry.

Importing it loads NumPy only.  Frameworks are backends chosen by name in a
:class:`ModelSpec` (``"linear"``, a closed-form NumPy reference; ``"torch"``,
installed with ``pip install 'vaft[ml]'``) and are imported on first use; no
public signature takes or returns a framework object.

Conventions
-----------
A split assigns *groups* -- the independent scientific unit, a shot or a
simulation case -- to partitions, and is made before windowing or
augmentation; it then applies unchanged to every derived dataset.
Normalisation moments come from the train partition only.  Every inference
result carries the immutable identity of the model that produced it: exact
version and manifest SHA-256, never only a stage alias.

Notes
-----
Scientific validation -- whether a model may be promoted -- is a verdict and
belongs in ``vaft.validation``; this package reports metrics only.  Model
outputs are candidates (an anomaly score, an IRE-like interval), not physical
events.  Externally trained surrogates such as TGLF-NN are resolved by their
own adapters (``vaft.code.gacode.tglf.surrogate``), not through this registry.
"""

from ._types import (
    BackendUnavailableError,
    CalibrationResult,
    DatasetArtifact,
    DatasetSpec,
    EvaluationResult,
    FeatureDataset,
    FeatureSpec,
    InferenceResult,
    MLError,
    ModelArtifact,
    ModelContractError,
    ModelIdentity,
    ModelResolutionError,
    ModelSpec,
    Split,
    SplitSpec,
    TrainingConfig,
    TrainingResult,
)
from .artifact import export_model, load_model_artifact, save_model_artifact
from .dataset import build_dataset, dataset_fingerprint, split_groups, window_dataset
from .inference import predict
from .resolver import load_model, resolve_model
from .training import calibrate_threshold, evaluate_model, train_model

__all__ = [
    "BackendUnavailableError",
    "CalibrationResult",
    "DatasetArtifact",
    "DatasetSpec",
    "EvaluationResult",
    "FeatureDataset",
    "FeatureSpec",
    "InferenceResult",
    "MLError",
    "ModelArtifact",
    "ModelContractError",
    "ModelIdentity",
    "ModelResolutionError",
    "ModelSpec",
    "Split",
    "SplitSpec",
    "TrainingConfig",
    "TrainingResult",
    "build_dataset",
    "calibrate_threshold",
    "dataset_fingerprint",
    "evaluate_model",
    "export_model",
    "load_model",
    "load_model_artifact",
    "predict",
    "resolve_model",
    "save_model_artifact",
    "split_groups",
    "train_model",
    "window_dataset",
]
