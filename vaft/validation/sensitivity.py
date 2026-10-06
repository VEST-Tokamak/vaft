"""Sensitivity, linearization and uncertainty evidence, interpreted (issue #1642).

The domain that owns an operation owns its derivatives: EFIT's native response
matrix lives in :mod:`vaft.code.efit.linearization`, a process's Jacobian beside
the process.  The shared algebra -- finite differences, ``J Sigma J^T``,
sampling, the singular-value spectrum -- is :mod:`vaft.formula.sensitivity`.
This module is the third part: what a derivative *is* (its provenance), and
what comparing two of them, or a linear propagation with a sampled one, says
about credibility.

Distinctions kept apart (#1642 "Core distinctions")
---------------------------------------------------
* **Provenance** (:data:`PROVENANCES`): a solver's ``native`` response, a
  ``finite_difference`` estimate around it, an ``autodiff`` one, an
  ``analytic`` one.  They are different evidence; agreement between two is
  verification of both.
* **Kind** (:data:`JACOBIAN_KINDS`): an ``observation`` Jacobian (outputs or
  residuals against parameters: identifiability) is not a ``governing_operator``
  one (``dF/dx`` of ``F(x, lambda) = 0``: uniqueness and branch structure), and
  neither is a ``forward`` one (outputs against inputs: local sensitivity).
* **Perturbation** (:data:`PERTURBATIONS`): a ``physical`` perturbation moves
  the plasma or the measurement; a ``numerical`` one moves a resolution or a
  tolerance.  Physical sensitivity and numerical convergence never share a
  column (#1642 s18).
* **Local versus sampled**: :func:`compare_linear_to_monte_carlo` is the
  escalation test of #1642 s16 -- where they disagree the local Jacobian has
  stopped describing the uncertainty, which is itself evidence.

Every comparison returns metrics and, *only when the caller supplies a
tolerance*, a status: no tolerance is invented here, exactly as
:mod:`vaft.validation.wall_reduction` and :mod:`vaft.validation.neoclassical`
do.  :func:`as_evidence` and :func:`scan_evidence` place results on the
credibility axes of :mod:`vaft.validation.credibility`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .model import ValidationStatus

__all__ = [
    "JACOBIAN_KINDS",
    "PERTURBATIONS",
    "PROVENANCES",
    "Jacobian",
    "as_evidence",
    "compare_jacobians",
    "compare_linear_to_monte_carlo",
    "scan_evidence",
]

PROVENANCES = ("native", "finite_difference", "autodiff", "analytic")
JACOBIAN_KINDS = ("forward", "observation", "governing_operator")
PERTURBATIONS = ("physical", "numerical")


@dataclass(frozen=True)
class Jacobian:
    """A derivative with its provenance: an explicit matrix or a matrix-free product.

    Exactly one of ``matrix`` (``(m, n)``) and ``jvp`` (``v -> J v``) is given.
    ``inputs`` and ``outputs`` name the columns and rows, so two Jacobians are
    compared on the same coordinates rather than on array positions.
    """

    provenance: str
    kind: str
    perturbation: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    matrix: np.ndarray | None = field(default=None, compare=False)
    jvp: Callable[[np.ndarray], np.ndarray] | None = field(default=None, compare=False)
    source: str = ""

    def __post_init__(self) -> None:
        for name, value, allowed in (("provenance", self.provenance, PROVENANCES),
                                     ("kind", self.kind, JACOBIAN_KINDS),
                                     ("perturbation", self.perturbation, PERTURBATIONS)):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
        if (self.matrix is None) == (self.jvp is None):
            raise ValueError("give exactly one of matrix and jvp")
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "outputs", tuple(self.outputs))
        if self.matrix is not None:
            matrix = np.atleast_2d(np.asarray(self.matrix, dtype=float))
            if matrix.shape != (len(self.outputs), len(self.inputs)):
                raise ValueError(f"matrix {matrix.shape} does not match "
                                 f"{len(self.outputs)} outputs x {len(self.inputs)} inputs")
            object.__setattr__(self, "matrix", matrix)

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.outputs), len(self.inputs))

    def apply(self, vector: Any) -> np.ndarray:
        """``J v`` whichever representation this holds."""
        v = np.asarray(vector, dtype=float).ravel()
        if v.size != len(self.inputs):
            raise ValueError(f"vector of {v.size} for {len(self.inputs)} inputs")
        if self.matrix is not None:
            return self.matrix @ v
        return np.asarray(self.jvp(v), dtype=float).ravel()

    def dense(self) -> np.ndarray:
        """The explicit matrix, assembled column by column from ``jvp`` if need be."""
        if self.matrix is not None:
            return self.matrix
        return np.column_stack([self.apply(np.eye(len(self.inputs))[j]) for j in range(len(self.inputs))])


def _graded(value: float, tolerance: tuple[float, float] | None) -> ValidationStatus | None:
    if tolerance is None:
        return None
    if not math.isfinite(value):
        return ValidationStatus.INDETERMINATE
    warn, fail = tolerance
    if value <= warn:
        return ValidationStatus.PASS
    if value <= fail:
        return ValidationStatus.WARN
    return ValidationStatus.FAIL


def compare_jacobians(
    reference: Jacobian,
    candidate: Jacobian,
    *,
    output_scale: Sequence[float] | None = None,
    tolerance: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """One derivative validated against another of different provenance.

    Both are aligned by their named inputs and outputs.  The metric is the
    largest per-output relative error, ``max_i ||(C - R)_i|| / ||R_i||`` (row
    norms), with ``output_scale`` in place of ``||R_i||`` where a row of the
    reference is legitimately near zero.  ``tolerance=(warn, fail)`` grades it;
    without one the result carries no status.
    """
    if set(reference.inputs) != set(candidate.inputs) or set(reference.outputs) != set(candidate.outputs):
        raise ValueError("Jacobians name different inputs or outputs")
    if reference.provenance == candidate.provenance:
        raise ValueError(f"both Jacobians are {reference.provenance}: that is repetition, not validation")
    ref = reference.dense()
    cand = candidate.dense()
    cols = [candidate.inputs.index(name) for name in reference.inputs]
    rows = [candidate.outputs.index(name) for name in reference.outputs]
    cand = cand[np.ix_(rows, cols)]
    ref_norm = np.linalg.norm(ref, axis=1)
    scale = ref_norm if output_scale is None else np.asarray(output_scale, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_output = np.linalg.norm(cand - ref, axis=1) / scale
    worst = float(np.nanmax(per_output)) if np.any(np.isfinite(per_output)) else math.nan
    status = _graded(worst, tolerance)
    result = {
        "reference": reference.provenance,
        "candidate": candidate.provenance,
        "max_relative_error": worst,
        "relative_error_by_output": dict(zip(reference.outputs, per_output.tolist())),
        "tolerance": tolerance,
    }
    if status is not None:
        result["status"] = str(status)
    return result


def compare_linear_to_monte_carlo(
    linear_covariance: np.ndarray,
    sampled: Mapping[str, Any],
    *,
    outputs: Sequence[str] | None = None,
    tolerance: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """The linear ``J Sigma J^T`` against a sampled covariance (#1642 s16).

    ``sampled`` is the dict :func:`vaft.formula.sensitivity.monte_carlo_propagation`
    returns.  The metric is the largest ``|ln(sigma_linear / sigma_sampled)|``
    over outputs -- a log ratio, so over- and under-estimation count alike --
    plus the sampled standard error of a standard deviation,
    ``1/sqrt(2(K-1))``, so the caller can tell a real disagreement from sampling
    noise.  Graded only when ``tolerance=(warn, fail)`` is given.
    """
    lin = np.atleast_2d(np.asarray(linear_covariance, dtype=float))
    mc = np.atleast_2d(np.asarray(sampled["covariance"], dtype=float))
    if lin.shape != mc.shape:
        raise ValueError(f"linear {lin.shape} and sampled {mc.shape} covariances differ in shape")
    names = tuple(outputs) if outputs is not None else tuple(str(i) for i in range(lin.shape[0]))
    sd_lin = np.sqrt(np.clip(np.diag(lin), 0.0, None))
    sd_mc = np.sqrt(np.clip(np.diag(mc), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log(sd_lin / sd_mc)
    finite = np.isfinite(log_ratio)
    worst = float(np.max(np.abs(log_ratio[finite]))) if finite.any() else math.nan
    count = int(sampled.get("samples", 0))
    status = _graded(worst, tolerance)
    result = {
        "max_abs_log_sd_ratio": worst,
        "log_sd_ratio_by_output": dict(zip(names, log_ratio.tolist())),
        "sampling_standard_error": 1.0 / math.sqrt(2.0 * (count - 1)) if count > 1 else math.nan,
        "samples": count,
        "rejected": int(sampled.get("rejected", 0)),
        "tolerance": tolerance,
    }
    if status is not None:
        result["status"] = str(status)
    return result


def as_evidence(result: Mapping[str, Any], *, key: str, axis: str = "numerical", cost: str = "moderate"):
    """A graded comparison as :class:`~vaft.validation.credibility.Evidence`.

    A Jacobian-against-Jacobian check verifies a derivative (``numerical``); a
    linear-against-sampled one says whether local uncertainty is trustworthy
    (``inference``).  An ungraded result (no tolerance given) is
    ``indeterminate``: the metric exists, nobody has said what it must be.
    """
    from .credibility import Evidence

    status = result.get("status", ValidationStatus.INDETERMINATE)
    metrics = {k: v for k, v in result.items() if k != "status"}
    return Evidence(axis, key, status, metrics, cost=cost)


def scan_evidence(report: Mapping[str, Any], *, key: str = "efit_sensitivity.model_form") -> Any:
    """A targeted-scan ensemble report as model-form evidence on the inference axis.

    Reads the dict that ``sensitivity_report.report()`` of the #579 EFIT study
    (#1663) produces -- per-slice ``slices[*].model_form`` (spread over the
    ``good`` members) and ``model_form_admissible`` (over the admissible ones),
    each ``{quantity: {"relative_half_iqr": ...}}``, plus ``marginal`` per-axis
    effects -- without importing ``workflow/``.

    A weight/basis grid is a *targeted parameter scan* (#1642 s15), not a
    Jacobian: it bounds how far the reconstructed quantities move across
    admissible model choices, which is inference evidence (#1639 s4 I: prior
    and regularization dependence).  The result is ungraded
    (``indeterminate``) and ``expensive`` by construction; ``not_available``
    when no slice carries a finite spread.

    The metrics are, per quantity, the median over slices of the relative
    half-IQR among the ``good`` members and among the ``admissible`` members;
    the report's ``marginal`` block is passed through untouched.
    """
    from .credibility import Evidence

    spreads: dict[str, dict[str, list[float]]] = {}
    for entry in report.get("slices", ()):
        for population, block in (("good", "model_form"), ("admissible", "model_form_admissible")):
            for quantity, spread in (entry.get(block) or {}).items():
                value = (spread or {}).get("relative_half_iqr")
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(number):
                    spreads.setdefault(quantity, {}).setdefault(population, []).append(number)
    medians = {
        quantity: {population: float(np.median(values)) for population, values in by_population.items()}
        for quantity, by_population in spreads.items()
    }
    metrics = {"median_relative_half_iqr": medians, "marginal": report.get("marginal", {}),
               "slices": len(report.get("slices", ())), "records": report.get("rows")}
    status = ValidationStatus.INDETERMINATE if spreads else ValidationStatus.NOT_AVAILABLE
    return Evidence("inference", key, status, metrics, cost="expensive")
