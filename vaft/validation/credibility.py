"""The credibility taxonomy: which question a piece of evidence answers (issue #1639).

A scientific result is trusted for six different reasons, and they fail
independently.  A reconstruction can fit every probe and still be unidentified;
a solve can converge and still violate an ordering its model assumed; a datum
can be valid and still be extrapolated.  #1639 names the six questions:

==========================  =====================================================
axis                        question
==========================  =====================================================
``source_evidence`` (E)     does the source datum itself provide credible evidence?
``transformation`` (T)      how faithfully was it turned into what is consumed?
``inference`` (I)           does the evidence determine the requested state?
``applicability`` (A)       do the chosen model's assumptions hold for this state?
``numerical`` (N)           was the calculation performed reliably?
``independent_validation``  does evidence *not* used in the inference support it?
(V)
==========================  =====================================================

This module is the vocabulary and nothing more.  It is deliberately not a
result schema (#1639 §2): a process returns its own result, the assessment
functions in the domain modules return their own plain dicts, and
:class:`Evidence` exists only so that evidence from different domains can be
laid side by side on these axes without anyone inventing a seventh vocabulary.

Rules carried over from #253 and #337
-------------------------------------
* **A metric is not a verdict, and a verdict is not policy.**  ``Evidence``
  carries both the metric and the status an assessment gave it; whether a
  result may be *used* stays with the workflow (#1639 §6, §17).
* **Missing evidence is not failure** (#1639 §18).  An axis with no evidence is
  absent from :func:`compose`; an axis whose evidence was never produced is
  ``not_available``.  Neither is a pass and neither is a fail.
* **The axes are never collapsed.**  :func:`compose` returns one status *per
  axis*; there is no overall credibility score, because "fit = pass,
  independent validation = not available" and "independent validation = fail"
  are different scientific statements (#1639 §18).
* **Precondition is not credibility** (#337, #1639 §5).  "Can this algorithm run
  on this input?" belongs to the process and may stop it; nothing here stops
  anything.
* **Cost is declared, never paid implicitly** (#1639 §8, §14).  Each piece of
  evidence names its cost class so a profile can ask for cheap evidence only.
  Importing this module loads only the status vocabulary; :func:`compose`
  loads :mod:`vaft.validation.equilibrium` on first call, for
  ``aggregate_status``.

Evidence roles
--------------
Agreement with fitted data is not independent validation (#1639 §4 V).  The
:data:`EVIDENCE_ROLES` say which part a datum played, and the adapters below
move a check off the ``independent_validation`` axis when the quantity it
compares against was fitted.  For an equilibrium report that is inferred
from the report itself: when its ``diagnostic_fit.diamagnetic_flux`` entry
records a fitted flux (``fit_role == "fitted"``, i.e. the constraint's fit
weight was non-zero -- VAFT's EFIT fits it by default,
``EFITConfig.use_diamagnetic_flux``), every check that compares against it
-- :data:`DIAMAGNETIC_CHECKS` -- is inference, not validation.  The grade
status is not the test: an ungraded fit (no statistical uncertainty model,
#891) is still a fit.

Pilots
------
:func:`evidence_from_equilibrium_report` (Pilot A, #892) and
:func:`evidence_from_efit_criteria` (Pilot A, the #891/#1331 study criteria v2)
project existing verdicts onto the axes; the virial checks they carry are
Pilot B.  :mod:`vaft.validation.applicability` is Pilot D.  The moment-order
convergence of Pilot C needs no machinery beyond
:func:`vaft.validation.applicability.successive_discrepancy`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .model import CATEGORIES, ValidationStatus

__all__ = [
    "AXES",
    "CATEGORY_AXES",
    "CHECK_AXES",
    "COST_CLASSES",
    "CRITERIA_AXES",
    "DIAMAGNETIC_CHECKS",
    "EVIDENCE_ROLES",
    "Evidence",
    "compose",
    "evidence_from_efit_criteria",
    "evidence_from_equilibrium_report",
]

#: The six credibility axes of #1639 §4, in E, T, I, A, N, V order.
AXES = (
    "source_evidence",
    "transformation",
    "inference",
    "applicability",
    "numerical",
    "independent_validation",
)

#: What a piece of evidence costs to produce (#1639 §14).  ``cheap`` is free
#: from existing results; ``moderate`` needs local computation but no solver
#: rerun; ``expensive`` (ensembles, Monte Carlo, continuation) is opt-in only.
COST_CLASSES = ("cheap", "moderate", "expensive")

#: The part a datum played in the result it bears on (#1639 §4 V).
EVIDENCE_ROLES = (
    "used_for_inference",
    "independent_validation",
    "available_not_used",
    "rejected",
    "assumed",
    "inferred",
)

#: The existing validation categories (:data:`vaft.validation.model.CATEGORIES`)
#: on the axes.  A default only: :data:`CHECK_AXES` refines the checks whose
#: category is broader than the question they answer.
#:
#: ``diagnostic_fit`` is ``inference`` and *not* ``identifiability``: a fit
#: residual says the inference reproduced its constraints, which is necessary
#: for, and never proof of, a determined state (#1639 §4 I).
#: ``physical_validity`` defaults to ``numerical``: a converged solve that
#: violates its own governing identities was not computed reliably.
CATEGORY_AXES: Mapping[str, str] = MappingProxyType({
    "verification": "numerical",
    "source_validity": "source_evidence",
    "diagnostic_fit": "inference",
    "physical_validity": "numerical",
    "independent_validation": "independent_validation",
})

#: Registry checks (:data:`vaft.validation.registry.CHECKS`) whose axis differs
#: from their category's default, with the reason in the comment.
CHECK_AXES: Mapping[str, str] = MappingProxyType({
    # Leave-one-identity-out: each closure is scored on the identity it did not
    # use -- the "infer with A+B, test with C" pattern of #1639 §10.
    "physical_validity.virial_pair_consistency": "independent_validation",
    # Whether a closure can be inverted here is conditioning of the inverse
    # problem, not a property of the equilibrium (registry: "never fail").
    "physical_validity.virial_conditioning": "inference",
    # The measured diamagnetic flux against the reconstruction.  Independent
    # only when the fit did not use it; see DIAMAGNETIC_CHECKS.
    "physical_validity.diamagnetic_flux": "independent_validation",
    # Plausibility of the *inferred* state -- beta_p and li inside physical
    # bounds, a q that keeps its sign, a pressure that is non-negative and
    # falls outward.  A solve can converge onto an implausible state, so this
    # is not "was it computed reliably" (N) but "is the inferred state one the
    # evidence should have produced" (I).  A stated choice, not a default.
    "physical_validity.virial_parameter_plausibility": "inference",
    "physical_validity.q_profile": "inference",
    "physical_validity.pressure_profile": "inference",
})

#: Checks that compare against the measured diamagnetic flux, directly or
#: through the closures it feeds.  None is independent validation when the
#: reconstruction was fitted to that flux.
DIAMAGNETIC_CHECKS = (
    "physical_validity.diamagnetic_flux",
    "independent_validation.diamagnetic_energy",
    "independent_validation.virial_measured_mu_i",
)

#: The #891/#1331 study criteria (``workflow/efit_uncertainty_calibration/
#: criteria.py``, version 2) on the axes -- the complete mapping, since these
#: verdicts have no validation category.  ``admissible`` is the veto that a
#: slice is a reconstruction at all; ``measurement`` is fit quality; ``virial``
#: compares two beta_p of the *same* g-file (the pressure integral and the
#: pair_13 closure), so it is internal consistency of the solution, not
#: independent evidence; ``thomson`` compares against a measurement the
#: magnetic fit never saw, and is the only verdict on V -- as criteria v2 keeps
#: it alone in ``PHYSICAL_CONSISTENCY``.
CRITERIA_AXES: Mapping[str, str] = MappingProxyType({
    "admissible": "numerical",
    "measurement": "inference",
    "virial": "numerical",
    "grad_shafranov": "numerical",
    "thomson": "independent_validation",
})


@dataclass(frozen=True)
class Evidence:
    """One assessment's conclusion, placed on one credibility axis.

    Thin on purpose (#1639 §3): invariant method metadata -- units, tolerance,
    provider, reference -- stays in the registry and the docs, and ``key``
    names where to find it.  ``metrics`` holds the run-specific numbers the
    status was derived from, so the continuous quantity is never lost behind
    the label.
    """

    axis: str
    key: str
    status: ValidationStatus
    metrics: Mapping[str, Any] = field(default_factory=dict, hash=False)
    reasons: tuple[str, ...] = ()
    cost: str = "cheap"
    role: str | None = None

    def __post_init__(self) -> None:
        if self.axis not in AXES:
            raise ValueError(f"unknown credibility axis {self.axis!r}; choose from {AXES}")
        if self.cost not in COST_CLASSES:
            raise ValueError(f"unknown cost class {self.cost!r}; choose from {COST_CLASSES}")
        if self.role is not None and self.role not in EVIDENCE_ROLES:
            raise ValueError(f"unknown evidence role {self.role!r}; choose from {EVIDENCE_ROLES}")
        object.__setattr__(self, "status", ValidationStatus(self.status))
        reasons = (self.reasons,) if isinstance(self.reasons, str) else self.reasons
        object.__setattr__(self, "reasons", tuple(str(reason) for reason in reasons))

    def as_dict(self) -> dict[str, Any]:
        """Plain, JSON-ready fields; a non-finite number becomes ``None``."""
        return {
            "axis": self.axis,
            "key": self.key,
            "status": str(self.status),
            "metrics": _plain(dict(self.metrics)),
            "reasons": list(self.reasons),
            "cost": self.cost,
            "role": self.role,
        }


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _keys(used_for_inference: Iterable[str], known: Iterable[str]) -> set[str]:
    """The caller's fitted keys, refused when they cannot mean anything here.

    Failing open would leave fitted data on the independent-validation axis,
    the one thing this module exists to prevent: a bare string would become a
    set of characters, and a misspelled key would match nothing.
    """
    if isinstance(used_for_inference, str):
        raise TypeError("used_for_inference takes a collection of check keys, not one string")
    keys = set(used_for_inference)
    unknown = sorted(keys - set(known))
    if unknown:
        raise ValueError(f"used_for_inference names unknown checks {unknown}")
    return keys


def compose(evidence: Iterable[Evidence]) -> dict[str, ValidationStatus]:
    """One status per axis that has evidence, never one status overall.

    Within an axis the statuses aggregate with
    :func:`vaft.validation.equilibrium.aggregate_status` semantics: the worst of
    fail > warn > indeterminate wins, ``pass`` needs every input to pass, and a
    pass beside ``not_available`` is ``indeterminate``.  An axis without any
    evidence is *absent*, which is a different statement from
    ``not_available``.  Returned in :data:`AXES` order.
    """
    from .equilibrium import aggregate_status

    by_axis: dict[str, list[ValidationStatus]] = {}
    for item in evidence:
        by_axis.setdefault(item.axis, []).append(item.status)
    return {axis: aggregate_status(by_axis[axis]) for axis in AXES if axis in by_axis}


def _records_a_fitted_constraint(entry: Any) -> bool:
    """Whether a ``diagnostic_fit`` scalar entry says the constraint was fitted.

    The precondition is the fit weight, carried per slice as ``fit_role``
    (``"fitted"`` when ``weight > 0``) and, equivalently, as a finite
    ``sigma_from_weight = 1/weight``; a report graded ``not_available`` for
    want of an uncertainty model still records both.
    """
    if not isinstance(entry, Mapping):
        return False
    records = [entry, *(s for s in (entry.get("slices") or ()) if isinstance(s, Mapping))]
    for record in records:
        if record.get("fit_role") == "fitted":
            return True
        try:
            sigma = float(record.get("sigma_from_weight"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(sigma) and sigma > 0:
            return True
    return False


def evidence_from_equilibrium_report(
    report: Mapping[str, Any],
    *,
    used_for_inference: Iterable[str] = (),
) -> tuple[Evidence, ...]:
    """Pilot A: :func:`vaft.validation.validate_equilibrium` on the credibility axes.

    Parameters
    ----------
    report
        The dict ``validate_equilibrium`` returns.  Each registered check
        present in it becomes one :class:`Evidence`, keyed
        ``"<category>.<check>"`` so :func:`vaft.validation.registry.describe`
        answers what its numbers mean.
    used_for_inference
        Further check keys whose reference measurement the reconstruction was
        fitted to.  Such a check cannot be independent validation; it moves to
        the ``inference`` axis with the role ``used_for_inference``.  The
        :data:`DIAMAGNETIC_CHECKS` move by themselves whenever the report's
        ``diagnostic_fit.diamagnetic_flux`` records a fitted flux (a slice
        with ``fit_role == "fitted"`` or a finite ``sigma_from_weight``, both
        meaning the constraint's fit weight was non-zero), whatever grade the
        entry received.  Unknown keys are refused.

    Returns
    -------
    tuple of Evidence
        In report order.  ``metrics`` keeps the report entry's ``counts`` and
        per-slice ``slices`` as they are -- nothing is recomputed -- and
        ``reasons`` gathers the distinct per-slice reasons.
    """
    from .registry import CHECKS

    fitted = _keys(used_for_inference, CHECKS)
    if _records_a_fitted_constraint((report.get("diagnostic_fit") or {}).get("diamagnetic_flux")):
        fitted.update(DIAMAGNETIC_CHECKS)
    evidence = []
    for category in CATEGORIES:
        checks = report.get(category)
        if not isinstance(checks, Mapping):
            continue
        for name, result in checks.items():
            if not isinstance(result, Mapping) or "status" not in result:
                continue
            key = f"{category}.{name}"
            axis = CHECK_AXES.get(key, CATEGORY_AXES[category])
            if key in fitted or category == "diagnostic_fit":
                role = "used_for_inference"
                if axis == "independent_validation":
                    axis = "inference"
            else:
                role = "independent_validation" if axis == "independent_validation" else None
            metrics = {k: v for k, v in result.items() if k not in ("status", "reason", "reasons")}
            reasons = list(result.get("reasons") or ([result["reason"]] if result.get("reason") else []))
            for entry in result.get("slices") or ():
                if isinstance(entry, Mapping) and entry.get("reason"):
                    reasons.append(str(entry["reason"]))
            evidence.append(Evidence(axis, key, result["status"], metrics, tuple(dict.fromkeys(reasons)),
                                     cost="moderate" if category in ("physical_validity", "independent_validation")
                                     else "cheap", role=role))
    return tuple(evidence)


def evidence_from_efit_criteria(
    evaluation: Mapping[str, Any],
    *,
    used_for_inference: Iterable[str] = (),
) -> tuple[Evidence, ...]:
    """Pilot A: the #891/#1331 study criteria (version 2) on the credibility axes.

    Takes the plain dict that ``criteria.evaluate(record)`` returns, so the
    validation layer depends on its *shape* and never imports ``workflow/``.
    Each verdict becomes one :class:`Evidence` keyed ``"efit_criteria.<name>"``;
    ``good`` and ``physically_consistent`` are *not* re-derived -- they are the
    workflow's policy over these verdicts (#1639 §6).  Criteria v2 keeps
    Thomson out of ``good``; here it is the only verdict on the V axis, so
    ``compose`` reports it apart from fit quality and internal consistency.

    ``used_for_inference`` names verdicts whose reference was fitted, as in
    :func:`evidence_from_equilibrium_report`.
    """
    version = evaluation.get("criteria_version")
    if version is not None and version != 2:
        raise ValueError(f"criteria version {version!r} is not the one these axes describe (2)")
    fitted = _keys(used_for_inference, CRITERIA_AXES)
    evidence = []
    for name, verdict in (evaluation.get("verdicts") or {}).items():
        axis = CRITERIA_AXES.get(name)
        if axis is None or not isinstance(verdict, Mapping):
            continue
        if name in fitted or name == "measurement":
            role = "used_for_inference"
            if axis == "independent_validation":
                axis = "inference"
        else:
            role = "independent_validation" if axis == "independent_validation" else None
        metrics = {k: v for k, v in verdict.items() if k not in ("status", "reasons")}
        evidence.append(Evidence(axis, f"efit_criteria.{name}", verdict.get("status", "not_available"),
                                 metrics, tuple(verdict.get("reasons") or ()), role=role))
    return tuple(evidence)
