"""Run a TGLF-NN ensemble on a local TGLF input.

The arithmetic is upstream's (``TurbulentTransport.jl``, ``src/tglf_nn.jl``): take the
log of the channels whose name says so, normalise with ``(x - xm) / xsigma``, run every
ensemble member, denormalise with ``y * ysigma + ym``, and take the mean.  The spread
across members is kept, because it is the one quantity that moves when the input leaves
the training set.

``onnxruntime`` is imported lazily and never at package import: issue #553 section 11
keeps the ML runtime optional, and the native GACODE adapter must stay importable
without it.  Nothing here touches the network -- see
:mod:`~vaft.code.gacode.tglf.surrogate.resolver`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from .._types import TGLFConfig
from ..inputs import TGLFInput
from ._types import (
    DEFAULT_DOMAIN_THRESHOLD,
    DOMAIN_MEASURE,
    ModelContractError,
    ModelIdentity,
    SurrogateError,
    SurrogateMetadata,
    SurrogatePrediction,
    TrainingDomainError,
)
from .domain import audit_features, build_features
from .resolver import load_metadata, resolve_model

__all__ = ["ensemble_predict", "run_surrogate"]


def _onnxruntime():
    """The optional runtime, or a refusal that says how to install it."""
    try:
        import onnxruntime  # noqa: PLC0415 -- optional dependency, imported on demand
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise SurrogateError(
            "the TGLF surrogate backend needs onnxruntime, which VAFT keeps optional "
            "so that the native GACODE adapter stays installable without a machine "
            "learning runtime. Install it with `pip install onnxruntime`."
        ) from exc
    return onnxruntime


@lru_cache(maxsize=64)
def _session(path: str):
    """A cached inference session for one ensemble member.

    Keyed by path: a session is expensive to build and an ensemble of twenty is rebuilt
    on every radius otherwise. Keyed by path *only*, so a model file replaced in place
    within one process keeps serving the old graph -- acceptable for read-only upstream
    artifacts, and the reason resolution records a SHA-256 of what it actually read.
    """
    return _onnxruntime().InferenceSession(path)


def ensemble_predict(
    features: np.ndarray, identity: ModelIdentity, metadata: SurrogateMetadata
) -> tuple[dict[str, float], dict[str, float]]:
    """Mean and standard deviation over the ensemble, per output channel.

    *features* are physical-space values in ``metadata.xnames`` order; normalisation
    and denormalisation happen here so a caller cannot apply one and forget the other.
    """
    values = np.asarray(features, dtype=float)
    if values.shape != (len(metadata.xnames),):
        raise ModelContractError(
            f"{identity.name} takes {len(metadata.xnames)} inputs; got a feature "
            f"vector of shape {values.shape}"
        )
    normalised = (
        (values - np.asarray(metadata.xm, dtype=float))
        / np.asarray(metadata.xsigma, dtype=float)
    ).astype(np.float32).reshape(1, -1)

    directory = Path(identity.directory)
    predictions = []
    for member in identity.members:
        session = _session(str(directory / member))
        raw = session.run(None, {session.get_inputs()[0].name: normalised})[0]
        predictions.append(np.ravel(raw).astype(float))

    stacked = np.vstack(predictions)
    if stacked.shape[1] != len(metadata.ynames):
        raise ModelContractError(
            f"{identity.name} returned {stacked.shape[1]} outputs for "
            f"{len(metadata.ynames)} declared output names; the directory's ynames.txt "
            f"does not describe these graphs"
        )
    denormalised = stacked * np.asarray(metadata.ysigma, dtype=float) + np.asarray(
        metadata.ym, dtype=float
    )
    mean = denormalised.mean(axis=0)
    spread = denormalised.std(axis=0)
    return (
        {name: float(mean[i]) for i, name in enumerate(metadata.ynames)},
        {name: float(spread[i]) for i, name in enumerate(metadata.ynames)},
    )


def run_surrogate(
    local: TGLFInput,
    model: str | Path,
    *,
    model_dir: Optional[str | Path] = None,
    env: Optional[Mapping[str, str]] = None,
    config: Optional[TGLFConfig] = None,
    threshold: float = DEFAULT_DOMAIN_THRESHOLD,
    allow_extrapolation: bool = False,
) -> SurrogatePrediction:
    """Predict the gyro-Bohm fluxes at *local* with the TGLF-NN family *model*.

    The domain audit runs first and, by default, decides whether the prediction is made
    at all: a model asked about a plasma outside its training distribution raises
    :class:`TrainingDomainError` naming the offending inputs.  Pass
    ``allow_extrapolation=True`` to get the numbers anyway -- they are returned with
    ``qualified`` False and the audit attached, because the point of issue #553 section
    12 is that an extrapolated prediction and an in-domain one must not be
    interchangeable downstream.

    That default is not caution for its own sake.  No public TGLF-NN family has VEST
    in domain: every one of them was trained where the ions are about as hot as the
    electrons, and VEST's ohmic plasma runs at ``T_i/T_e`` near 0.1, which lands
    ``TAUS`` four to ten sigma out with nothing in the returned fluxes to show it.
    """
    identity = resolve_model(model, model_dir=model_dir, env=env)
    metadata = load_metadata(identity)
    features, assumed = build_features(local, metadata, config=config)
    audit = audit_features(features, assumed, identity, metadata, threshold=threshold)
    if not audit.in_domain and not allow_extrapolation:
        raise TrainingDomainError(
            f"refusing to predict outside the training distribution -- {audit.summary()}. "
            f"The network will return finite numbers for this input; they are not "
            f"grounded in anything it was trained on. Pass allow_extrapolation=True to "
            f"obtain them anyway, and read `prediction.qualified` before using them.",
            audit,
        )

    outputs, uncertainty = ensemble_predict(features, identity, metadata)
    # Surrogate fidelity (issue #553 sections 12-13): a SAT3-trained network standing
    # in for a SAT1 run is a mismatch, but SAT_RULE is not one of the model's input
    # channels, so the audit cannot see it. It is recorded rather than refused because
    # `TGLFConfig.sat_rule` has a default -- a caller who passed a config for some
    # other reason never made a saturation-rule claim, and refusing would invent one.
    requested = config or TGLFConfig()
    settings = {
        "requested_sat_rule": int(requested.sat_rule),
        "requested_electromagnetic": bool(requested.use_bper or requested.use_bpar),
        "sat_rule_matches_model": (
            None if identity.sat_rule is None
            else int(requested.sat_rule) == int(identity.sat_rule)
        ),
    }
    return SurrogatePrediction(
        outputs=outputs,
        uncertainty=uncertainty,
        model=identity,
        domain=audit,
        features={name: float(features[i]) for i, name in enumerate(metadata.xnames)},
        provenance={
            "backend": "surrogate",
            "format": identity.format,
            "model": identity.name,
            "resolved_by": identity.resolved_by,
            "directory": identity.directory,
            "ensemble_size": identity.ensemble_size,
            "sha256": dict(identity.sha256),
            "upstream_version": identity.upstream_version,
            "sat_rule": identity.sat_rule,
            "electromagnetic": identity.electromagnetic,
            "devices": identity.devices,
            "physics_source": identity.physics_source,
            "domain_measure": DOMAIN_MEASURE,
            "domain_threshold": float(threshold),
            "training_bounds_available": audit.bounds_available,
            "extrapolation_allowed": bool(allow_extrapolation),
            "assumed_features": audit.assumed,
            "rho": float(local.rho),
            **settings,
        },
    )
