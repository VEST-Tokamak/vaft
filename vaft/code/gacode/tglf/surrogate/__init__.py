"""TGLF-NN: an accelerated backend for the local TGLF contract.

    TGLFInput ---> features(model.xnames) ---> ONNX ensemble ---> SurrogatePrediction
                          |                                              ^
                          `------ DomainAudit ---------------------------'
                                  (decides whether the prediction is made at all)

The surrogate is an implementation of the *same* physics contract the native backend
runs, not a second scientific input representation (issue #553 section 4).  Its feature
vector is built from :func:`~vaft.code.gacode.tglf.inputs.tglf_parameters` -- the keys
that would be written into ``input.tglf`` -- so the two backends cannot drift apart
without the shared contract changing.

Three properties are deliberate:

* **No weights are vendored.**  VAFT ships no model.  The resolver finds artifacts the
  user already has, and reports every path it searched when it finds none.
* **No network, and no ``onnxruntime`` at import.**  ``import vaft`` stays offline and
  the native adapter stays installable without a machine-learning runtime.
* **Extrapolation is an output, not a footnote.**  Every prediction carries a
  :class:`DomainAudit`, and one outside the training distribution is refused unless the
  caller asks for it by name.  :func:`audit_training_domain` answers "does this model
  apply here" on its own, without the runtime and without a prediction.

Not in this increment: remote download, SHA-256 against a pinned manifest and a
persistent cache (sections 7-9), the ``core_transport`` projection (section 14), and
the native-versus-surrogate comparison (section 15).
"""

from __future__ import annotations

from ._types import (
    DEFAULT_DOMAIN_THRESHOLD,
    DOMAIN_MEASURE,
    DomainAudit,
    ModelContractError,
    ModelIdentity,
    ModelResolutionError,
    SurrogateError,
    SurrogateMetadata,
    SurrogatePrediction,
    TrainingDomainError,
)
from .domain import audit_features, audit_training_domain, build_features
from .inference import ensemble_predict, run_surrogate
from .resolver import (
    KNOWN_DEVICES,
    METADATA_FILES,
    MODELS_COMPATIBILITY_ENVS,
    MODELS_HOME_ENV,
    available_models,
    load_metadata,
    parse_model_name,
    resolve_model,
)

__all__ = [
    "DEFAULT_DOMAIN_THRESHOLD",
    "DOMAIN_MEASURE",
    "KNOWN_DEVICES",
    "METADATA_FILES",
    "MODELS_COMPATIBILITY_ENVS",
    "MODELS_HOME_ENV",
    "DomainAudit",
    "ModelContractError",
    "ModelIdentity",
    "ModelResolutionError",
    "SurrogateError",
    "SurrogateMetadata",
    "SurrogatePrediction",
    "TrainingDomainError",
    "audit_features",
    "audit_training_domain",
    "available_models",
    "build_features",
    "ensemble_predict",
    "load_metadata",
    "parse_model_name",
    "resolve_model",
    "run_surrogate",
]
