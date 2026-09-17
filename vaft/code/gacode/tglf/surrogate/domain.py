"""Project a local TGLF input into a model's feature space, and say where it lands.

This module is deliberately usable without ``onnxruntime``: deciding whether a model
applies to a plasma is a question about the input and the training moments, not about
the network, and it is the question that matters first when a public model meets a
device it was not trained on (issue #553 section 16).

The feature construction follows upstream ``TurbulentTransport.jl``
(``src/tglf_nn.jl``): the model's ``xnames`` are TGLF keys, in the model's own order,
and a name ending ``_log10`` asks for the base-10 logarithm of the key with the suffix
removed.  Every family declares a different subset -- the MAST-U-only networks carry no
``ZEFF``, the UKSTEP ones take the impurity charge ``ZS_3`` as a feature -- so the
vector is assembled from ``xnames`` rather than from any fixed list.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from dataclasses import fields

from .._types import TGLFConfig
from ..inputs import TGLFInput, tglf_parameters
from ._types import (
    DEFAULT_DOMAIN_THRESHOLD,
    DomainAudit,
    ModelContractError,
    ModelIdentity,
    SurrogateMetadata,
)

_LOG10_SUFFIX = "_log10"

#: The names a provenance record can use, which are the local input's own field names.
_INPUT_FIELDS = frozenset(entry.name for entry in fields(TGLFInput))

__all__ = [
    "DEFAULT_DOMAIN_THRESHOLD",
    "audit_features",
    "audit_training_domain",
    "build_features",
]


def _provenance_keys(feature: str) -> tuple[str, ...]:
    """Provenance names a feature could have been recorded under.

    ``TAUS_2`` is the species-2 entry of the ``taus`` array, and ``VEXB_SHEAR`` is the
    scalar ``vexb_shear``; both spellings are tried so a per-species record added later
    is picked up without this needing to change.

    Candidates are reconciled against :class:`TGLFInput`'s own field names, because the
    obvious spelling is not always the real one: the density-fraction array is ``as_``,
    with the trailing underscore that keeps it off the ``as`` keyword, and a record
    under that name would otherwise never be matched.
    """
    base = feature[: -len(_LOG10_SUFFIX)] if feature.endswith(_LOG10_SUFFIX) else feature
    lowered = base.lower()
    head, _, tail = lowered.rpartition("_")
    candidates = [lowered, head] if head and tail.isdigit() else [lowered]
    resolved: list[str] = []
    for name in candidates:
        resolved.append(name)
        if name not in _INPUT_FIELDS and f"{name}_" in _INPUT_FIELDS:
            resolved.append(f"{name}_")
    return tuple(resolved)


def build_features(
    local: TGLFInput,
    metadata: SurrogateMetadata,
    *,
    config: Optional[TGLFConfig] = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """The model's input vector for *local*, and which entries rest on a default.

    The values come from :func:`~vaft.code.gacode.tglf.inputs.tglf_parameters`, so the
    surrogate is fed exactly what the native run would be written from -- the point of
    issue #553 section 4 being that the two share one contract rather than two
    parallel ones.

    Refuses rather than substituting:

    * a two-stage ``*_gknn*`` family, whose features include a base model's fluxes;
    * a channel no TGLF key supplies;
    * a value the log-transform sends to infinity.

    The second element is the tuple of channels whose value came from a TGLF default
    standing in for a quantity the state records as unavailable.  Those are *reported*,
    not refused: ``input.tglf`` carries the same default into the native solver, and a
    surrogate stricter than the code it stands in for is no longer a surrogate of it.
    """
    staged = metadata.requires_base_model_outputs
    if staged:
        raise ModelContractError(
            f"this model takes {', '.join(staged)} as inputs: it is a two-stage "
            f"correction network over another model's predictions, not a drop-in "
            f"surrogate, and has no feature vector derivable from a TGLF input alone"
        )

    parameters = tglf_parameters(local, config)
    unavailable = set(local.missing())

    values = np.empty(len(metadata.xnames), dtype=float)
    assumed: list[str] = []
    absent: list[str] = []
    undefined: list[str] = []

    for index, feature in enumerate(metadata.xnames):
        logarithmic = feature.endswith(_LOG10_SUFFIX)
        key = feature[: -len(_LOG10_SUFFIX)] if logarithmic else feature
        if key not in parameters:
            absent.append(feature)
            values[index] = np.nan
            continue
        raw = float(parameters[key])
        with np.errstate(divide="ignore", invalid="ignore"):
            # A non-positive value here is a reportable contract failure, not a
            # numerical accident, so the refusal below is the handler -- numpy's
            # warning would only obscure it.
            value = np.log10(raw) if logarithmic else raw
        if not np.isfinite(value):
            undefined.append(
                f"{feature} (log10 of {key}={raw:g})" if logarithmic
                else f"{feature} ({key}={raw:g})"
            )
        values[index] = value
        if any(name in unavailable for name in _provenance_keys(feature)):
            assumed.append(feature)

    if absent:
        raise ModelContractError(
            f"this model asks for {', '.join(absent)}, which no TGLF key in the local "
            f"input supplies. The model's input convention differs from the one this "
            f"contract writes; it cannot be run on this state."
        )
    if undefined:
        raise ModelContractError(
            f"the features {', '.join(undefined)} are not finite, so the network "
            f"cannot be fed a normalised coordinate for them. A `log10 of` entry means "
            f"a non-positive value reached a logarithmic channel; the others carry a "
            f"non-finite value from the local input itself."
        )
    return values, tuple(assumed)


def audit_training_domain(
    local: TGLFInput,
    identity: ModelIdentity,
    metadata: SurrogateMetadata,
    *,
    config: Optional[TGLFConfig] = None,
    threshold: float = DEFAULT_DOMAIN_THRESHOLD,
) -> DomainAudit:
    """Where *local* sits relative to the model's training distribution.

    Needs no ONNX runtime and runs no network, so "does this model apply to this
    plasma" is answerable before, and independently of, any prediction.

    The measure is a standard-deviation distance, not containment: the ONNX
    distribution ships the normalisation moments but not the per-input bounds, which
    exist only inside the upstream Julia artifacts.  :attr:`DomainAudit.bounds_available`
    carries that caveat with the result so a caller never reads ``in_domain`` as a
    guarantee it cannot be.
    """
    values, assumed = build_features(local, metadata, config=config)
    return audit_features(values, assumed, identity, metadata, threshold=threshold)


def audit_features(
    values: np.ndarray,
    assumed: Sequence[str],
    identity: ModelIdentity,
    metadata: SurrogateMetadata,
    *,
    threshold: float = DEFAULT_DOMAIN_THRESHOLD,
) -> DomainAudit:
    """:func:`audit_training_domain` for a feature vector that is already built.

    Exists so that a run audits the very vector it then predicts from, rather than
    rebuilding an equal one: the guarantee is structural instead of a property two
    call sites happen to share.
    """
    if threshold <= 0:
        raise ValueError(f"the domain threshold is a positive distance; got {threshold}")
    z = (values - np.asarray(metadata.xm, dtype=float)) / np.asarray(metadata.xsigma, dtype=float)
    scores = {name: float(value) for name, value in zip(metadata.xnames, z)}
    violations = tuple(
        name for name in sorted(scores, key=lambda key: -abs(scores[key]))
        if abs(scores[name]) > threshold
    )
    return DomainAudit(
        model=identity.name,
        z_scores=scores,
        threshold=float(threshold),
        violations=violations,
        max_abs_z=float(np.max(np.abs(z))) if z.size else 0.0,
        assumed=tuple(assumed),
    )
