"""Bind supplied ML metrics and owner-defined criteria to credibility evidence.

This module computes no predictions, scores, geometry or physical metrics. A
report has no overall verdict and grants no deployment or submission approval.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, asdict
import hashlib
import json
import math

from .credibility import AXES, Evidence
from .model import ValidationStatus

__all__ = ["MetricCriterion", "build_ml_report", "ml_report_evidence"]


def _plain(value):
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _digest(value):
    payload = json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class MetricCriterion:
    """Versioned owner-defined bounds; units/provider/source remain explicit.

    No bound is fitted from the report. Declare these before qualification.
    A metric inside its bounds passes this check only, not scientific acceptance.
    """
    metric: str
    axis: str
    unit: str
    provider: str
    source: str
    revision: str
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self):
        if self.axis not in AXES or any(not isinstance(v, str) or not v.strip()
            for v in (self.metric, self.unit, self.provider, self.source, self.revision)):
            raise ValueError("criteria require a known axis and explicit metric/unit/provider/source/revision")
        bounds = [v for v in (self.lower, self.upper) if v is not None]
        if not bounds or not all(math.isfinite(v) for v in bounds):
            raise ValueError("criteria require finite declared bounds")
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError("criterion lower bound exceeds upper bound")


def _binding(model, evaluation, dataset, split, criteria, protocol):
    from vaft.process import ml
    identity = model.resolved_identity().to_dict()
    if identity != evaluation.model_identity.to_dict():
        raise ValueError("evaluation model identity mismatch")
    if identity.get("state_sha256") and identity["state_sha256"] != model.state_sha256:
        raise ValueError("model state identity mismatch")
    if not dataset.fingerprint:
        raise ValueError("report needs a fingerprinted stable dataset")
    if evaluation.provenance.get("dataset_fingerprint") != dataset.fingerprint:
        raise ValueError("evaluation dataset fingerprint mismatch")
    if evaluation.provenance.get("split_sha256") != _digest(split.to_dict()):
        raise ValueError("evaluation split identity mismatch")
    if any(not protocol.get(k) for k in ("name", "revision", "population")):
        raise ValueError("report protocol needs name/revision/population")
    criteria = tuple(criteria)
    if not criteria or len({c.metric for c in criteria}) != len(criteria):
        raise ValueError("criteria must be nonempty and have unique metric names")
    selected = split.mask(dataset, evaluation.partition) & dataset.available_mask
    if int(selected.sum()) != evaluation.n_samples:
        raise ValueError("evaluation sample population mismatch")
    groups = sorted(set(dataset.group_labels[selected]))
    if set(evaluation.per_group) != set(groups):
        raise ValueError("evaluation group population mismatch")
    usage = ml.record_data_use(dataset, split, "evaluate", groups=groups)
    if usage is not None and evaluation.provenance.get("data_use") != usage:
        raise ValueError("evaluation usage receipt mismatch")
    if evaluation.provenance.get("model_contract_sha256") != model.contract_sha256:
        raise ValueError("evaluation model contract mismatch")
    return {"model_identity":identity, "model_contract_sha256":model.contract_sha256,
        "dataset_fingerprint":dataset.fingerprint, "split_sha256":_digest(split.to_dict()),
        "partition":evaluation.partition, "groups":groups, "n_samples":evaluation.n_samples,
        "data_use":usage, "protocol":_plain(protocol), "protocol_sha256":_digest(protocol),
        "criteria":[asdict(c) for c in criteria], "criteria_sha256":_digest([asdict(c) for c in criteria])}


def build_ml_report(model, evaluation, dataset, split, criteria, *, protocol):
    """Bind numerical evaluation to declared identities and criteria.

    Parameters
    ----------
    model, evaluation, dataset, split
        Existing ML lifecycle objects. Dataset/source contents must remain stable.
    criteria : iterable of MetricCriterion
        Owner-defined, predeclared criteria, never learned from these metrics.
    protocol : mapping
        Name, revision, population and domain-specific scorer/label/scope evidence.

    Returns
    -------
    dict
        JSON-only sidecar with content hash; no overall scientific verdict.

    Notes
    -----
    Retain the hash in a trusted external run/publication record. Adding the
    report hash to the already-bound model manifest would change model identity;
    use a sidecar reference to avoid a circular manifest/report dependency.
    """
    binding = _binding(model, evaluation, dataset, split, tuple(criteria), protocol)
    report = {"schema_version":1, "binding":binding, "metrics":_plain(evaluation.metrics),
        "per_group":_plain(evaluation.per_group), "history_scope":"declared and recorded uses; external history is not retrospectively proved"}
    return {**report, "report_sha256":_digest(report)}


def ml_report_evidence(report, model, evaluation, dataset, split, criteria, *, protocol, expected_report_sha256):
    """Verify a report against trusted inputs and emit existing per-axis Evidence.

    The expected hash comes from the trusted run/publication reference, not an
    untrusted report's own checksum. This is integrity checking, not signature
    authentication or a production promotion decision.
    """
    report = _plain(report)
    digest = _digest({k:v for k,v in report.items() if k != "report_sha256"})
    if digest != expected_report_sha256 or digest != report.get("report_sha256") or report.get("schema_version") != 1:
        raise ValueError("ML report hash/schema mismatch")
    criteria = tuple(criteria)
    if report["binding"] != _binding(model, evaluation, dataset, split, criteria, protocol):
        raise ValueError("ML report identity/protocol/criteria mismatch")
    if report["metrics"] != _plain(evaluation.metrics) or report["per_group"] != _plain(evaluation.per_group):
        raise ValueError("ML report metric source mismatch")
    usage = report["binding"]["data_use"]
    policy = dataset.spec.metadata.get("data_use_policy")
    known_uses = model.training.get("data_uses", ())
    consumed = {g for r in known_uses if r["operation"] in ("train","preprocess","calibrate","select") for g in r["groups"]}
    consumed.update(g for c in model.calibration.values() for g in c.groups)
    result = []
    for criterion in criteria:
        value = report["metrics"].get(criterion.metric)
        axis, role, reasons = criterion.axis, None, []
        status = (ValidationStatus.INDETERMINATE if criterion.metric in report["metrics"] else ValidationStatus.NOT_AVAILABLE) if value is None else ValidationStatus.PASS
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value):
                status = ValidationStatus.INDETERMINATE
            elif (criterion.lower is not None and value < criterion.lower) or (criterion.upper is not None and value > criterion.upper):
                status = ValidationStatus.FAIL
        if not report["binding"]["n_samples"]:
            status = ValidationStatus.NOT_AVAILABLE
            reasons.append("no available evaluation samples")
        if axis == "independent_validation":
            if usage is None or not any(r["operation"] == "train" for r in known_uses):
                status = ValidationStatus.NOT_AVAILABLE
                reasons.append("independent use/history evidence is unavailable")
            else:
                groups = usage["groups"]
                permitted_fit = {"train","preprocess","calibrate","select"}
                fitted = set(groups) & consumed
                reusable = any(permitted_fit & set(policy["allowed_operations"][usage["group_roles"][g]]) for g in groups)
                if fitted or reusable:
                    axis, role = "inference", "used_for_inference"
                    reasons.append("population was consumed or reserved for fitting/calibration/selection; not independent evidence")
                else:
                    role = "independent_validation"
                    reasons.append("independent of recorded uses under owner-declared history and protocol scope")
        result.append(Evidence(axis=axis, key="ml."+criterion.metric, status=status,
            metrics={"value":value, "report_sha256":digest, "model_identity":report["binding"]["model_identity"],
                "unit":criterion.unit, "criterion_revision":criterion.revision, "population":protocol["population"]},
            reasons=tuple(reasons), role=role))
    return tuple(result)
