"""An optional sidecar that keeps a result's scientific meaning and evidence together (#2165).

A computed value says nothing about what it is, how it got its convention, which
model assumptions it rests on or what supports it.  :class:`ScientificContext`
holds references to those things next to the value, without wrapping it: a
formula still returns its float, a process its array, a code adapter its
record.  A workflow that wants the context builds one at a stage boundary and
passes it on as ``upstream`` of the next stage's context.

What it holds, each by reference to its canonical owner:

* the result's declared :class:`~vaft.data.representation.Representation`;
* the :class:`~vaft.data.representation.TransformationRecord` of each
  transformation executed on the way;
* :class:`~vaft.validation.applicability.ContractEvaluation` outcomes, with
  their margins, limiting assumption and ``UNASSESSED`` inputs intact;
* :class:`~vaft.validation.credibility.Evidence` on the six axes, with their
  fitted or independent roles;
* plain-text unresolved assumptions -- an unconfirmed machine polarity, an
  undeclared convention.

Composition is conservative.  :meth:`ScientificContext.unresolved_assumptions`
gathers every upstream item and never drops one because a later stage
succeeded; :meth:`ScientificContext.credibility` composes each axis separately
with :func:`~vaft.validation.credibility.compose`, so a downstream converged
solve (N) neither hides nor repairs an upstream source question (E), and an
axis without evidence stays absent rather than passing.  Nothing here runs a
check, re-runs a solver, or decides acceptance: the consuming workflow does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

__all__ = ["CONTEXT_SCHEMA", "ScientificContext"]

#: Version of :meth:`ScientificContext.as_dict`.
CONTEXT_SCHEMA = 1


def _unique(items) -> tuple:
    seen: dict = {}
    for item in items:
        seen.setdefault(item, None)
    return tuple(seen)


@dataclass(frozen=True, eq=False)
class ScientificContext:
    """References to what one result means and what supports it; every field optional.

    ``stage`` names the step (``"efit"``, ``"chease"``); ``provider`` the
    dotted path of the function or adapter that produced the result.
    """

    stage: str
    provider: str = ""
    representation: Optional[Any] = None
    transformations: tuple = ()
    applicability: tuple = ()
    evidence: tuple = ()
    unresolved: tuple[str, ...] = ()
    upstream: tuple["ScientificContext", ...] = ()

    def __post_init__(self):
        for name in ("transformations", "applicability", "evidence", "unresolved", "upstream"):
            value = getattr(self, name)
            if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
                value = (value,)
            object.__setattr__(self, name, tuple(value))

    # --- gathered over the chain ------------------------------------------

    def chain(self) -> tuple["ScientificContext", ...]:
        """This context and every upstream one, each once, upstream first."""
        out: list[ScientificContext] = []
        for parent in self.upstream:
            out.extend(parent.chain())
        out.append(self)
        # by identity: a shared upstream stage (one EFIT feeding two codes) counts once
        seen: set[int] = set()
        return tuple(c for c in out if not (id(c) in seen or seen.add(id(c))))

    def unresolved_assumptions(self) -> tuple[str, ...]:
        """Every unresolved item of the chain, from contexts and their transformations; none dropped."""
        items: list[str] = []
        for context in self.chain():
            items.extend(context.unresolved)
            for record in context.transformations:
                items.extend(getattr(record, "unresolved", ()))
        return _unique(items)

    def all_transformations(self) -> tuple:
        """Every transformation record of the chain, upstream first."""
        return tuple(record for context in self.chain() for record in context.transformations)

    def all_evidence(self) -> tuple:
        """Every piece of evidence of the chain, with applicability placed on the A axis.

        An applicability evaluation becomes evidence through
        :func:`vaft.validation.applicability.as_evidence`; ``NOT_APPLICABLE``
        contributes none, because the contract does not concern the state.
        """
        from vaft.validation.applicability import as_evidence

        items: list = []
        for context in self.chain():
            items.extend(context.evidence)
            for evaluation in context.applicability:
                converted = as_evidence(evaluation)
                if converted is not None:
                    items.append(converted)
        return tuple(items)

    def credibility(self, *, include_upstream: bool = True) -> dict:
        """One status per axis that has evidence, composed separately; never one overall status."""
        from vaft.validation.credibility import compose

        evidence = self.all_evidence() if include_upstream else ScientificContext(
            self.stage, evidence=self.evidence, applicability=self.applicability).all_evidence()
        return compose(evidence)

    # --- serialization ----------------------------------------------------------

    def as_dict(self) -> dict:
        """Strict-JSON fields with ``schema_version``; references only, no arrays."""
        rep = self.representation
        return {
            "schema_version": CONTEXT_SCHEMA,
            "stage": self.stage,
            "provider": self.provider,
            "representation": None if rep is None else {
                "quantity": rep.quantity, "unit": rep.unit, "cocos": rep.cocos,
                "fourier": None if rep.fourier is None else rep.fourier.name,
                "evidence": rep.evidence.value,
            },
            "transformations": [record.as_dict() for record in self.transformations],
            "applicability": [_plain(evaluation.as_dict()) for evaluation in self.applicability],
            "evidence": [item.as_dict() for item in self.evidence],
            "unresolved": list(self.unresolved),
            "upstream": [parent.as_dict() for parent in self.upstream],
            "credibility": {axis: str(status) for axis, status in self.credibility().items()},
            "unresolved_assumptions": list(self.unresolved_assumptions()),
        }


def _plain(value: Any) -> Any:
    import math

    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "value") and type(value).__module__ != "builtins":
        return str(value)
    return value
