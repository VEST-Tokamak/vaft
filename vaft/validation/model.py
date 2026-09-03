"""The shared vocabulary of the validation layer (issue #253).

Deliberately small.  This module owns *what an assessment can say*, not how any
particular assessment is computed: the calculations stay with their domain
providers (:mod:`vaft.omas.efit_quality`, :mod:`vaft.omas.vacuum_magnetics`,
:mod:`vaft.process.equilibrium`, :mod:`vaft.formula.statistics`) and only their
interpretation is composed here.

Two rules keep it that way.

**A metric is not a validation result.**  ``residual_rms = 2.3e-4 T`` is a
number; it becomes an assessment only once a criterion is applied to it.  So
nothing here forces a metric to carry a status, and most VAFT metric functions
return plain floats or compact mappings rather than instances of any class in
this module.

**Validation evidence is not production acceptance.**  :class:`ValidationStatus`
spans what the evidence itself can say -- including that it is simply
:attr:`~ValidationStatus.NOT_AVAILABLE`, which is not a pass.  Whether a datum
or a reconstruction may then be *used* is a policy decision belonging to the
consumer (FileDB stage QA, EFIT constraint weighting), not to this layer.

Two shapes, not two vocabularies
--------------------------------
This module holds two things that are easy to mistake for rivals and are not
(issue #337):

:class:`ValidationStatus`
    A **verdict**.  One value summarizing what an assessment concluded.

:class:`ValidationIssue` / :class:`ValidationReport`
    A **findings list**.  Zero or more specific objections, each with a
    severity, a machine-readable code and the field it concerns.  A report with
    no issues is not a richer way of saying ``PASS``; it is the list of things
    that were wrong, which happens to be empty.

They meet at :attr:`ValidationReport.status`, so a caller composing a findings
list into a report of verdicts does not have to invent a third vocabulary --
which is exactly how the codebase ended up with three.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = [
    "CATEGORIES",
    "SEVERITIES",
    "ValidationIssue",
    "ValidationReport",
    "ValidationStatus",
]


class ValidationStatus(str, Enum):
    """What one assessment concluded.

    ``str`` mixin rather than :class:`enum.StrEnum`: the package supports Python
    3.10, where ``StrEnum`` does not exist.  The mixin gives the same practical
    behaviour -- members compare equal to and serialize as their value, so a
    report round-trips through JSON without a custom encoder.

    ``NOT_AVAILABLE`` and ``INDETERMINATE`` are distinct, and neither is a pass:
    the first means the evidence was never produced (no Thomson data for this
    shot), the second that it was produced but does not decide the question (a
    residual whose normalization could not be recovered).
    """

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    INDETERMINATE = "indeterminate"
    NOT_AVAILABLE = "not_available"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


#: Severities a :class:`ValidationIssue` may carry, worst first.  Deliberately
#: shorter than :class:`ValidationStatus`: a finding is something that is wrong,
#: so it has no vocabulary for "not available" or "undecidable" -- those are
#: properties of a *verdict*, and are reached by not raising an issue at all.
SEVERITIES = ("error", "warning")


@dataclass(frozen=True)
class ValidationIssue:
    """One specific objection to a datum or a result.

    ``code`` is the machine-readable identity a caller branches on and a test
    asserts against; ``message`` is for a human and may be reworded freely.
    ``field`` names what the objection is about, so a caller can group findings
    without parsing prose.
    """

    severity: str
    code: str
    field: str
    message: str


@dataclass(frozen=True)
class ValidationReport:
    """The findings from one assessment, in the order they were made.

    Kept deliberately thin (#253 §4): no units, no tolerance, no provenance.
    Those are invariant per check rather than per instance and belong in
    documentation or a registry, not repeated on every result.
    """

    issues: tuple[ValidationIssue, ...] = ()

    @property
    def valid(self) -> bool:
        """Whether nothing rose to ``error``.

        A gate, and the reason this type exists: callers use it to decide
        whether an algorithm may run at all.  It is deliberately coarser than
        :attr:`status` -- a warning does not stop the caller.
        """
        return not any(item.severity == "error" for item in self.issues)

    @property
    def status(self) -> "ValidationStatus":
        """The findings expressed in the shared verdict vocabulary.

        The bridge that keeps one vocabulary in the codebase: an ``error``
        reads as :attr:`~ValidationStatus.FAIL`, a ``warning`` as
        :attr:`~ValidationStatus.WARN`, and no findings as
        :attr:`~ValidationStatus.PASS`.

        ``INDETERMINATE`` and ``NOT_AVAILABLE`` are unreachable from here, and
        that is correct rather than a gap: a findings list can only report what
        it objected to.  "The evidence does not decide" and "the evidence was
        never produced" are statements a *check* makes about itself, so they are
        raised by the check rather than inferred from an empty report.
        """
        if not self.valid:
            return ValidationStatus.FAIL
        if self.issues:
            return ValidationStatus.WARN
        return ValidationStatus.PASS


#: The validation taxonomy.  Verification is a *category* of validation rather
#: than a separate top-level namespace: "was the calculation performed as
#: intended?" and "is the result credible for its intended use?" are different
#: questions about the same object, and VAFT answers both under
#: :mod:`vaft.validation`.
#:
#: This is a classification, not a contract -- no domain is required to
#: implement every category.  Magnetics signal quality (issue #189) is purely
#: ``source_validity``; the vacuum-model benchmark (issue #190) is purely
#: ``diagnostic_fit``; neither has anything to say about the others.
CATEGORIES = (
    "verification",
    "source_validity",
    "diagnostic_fit",
    "physical_validity",
    "independent_validation",
)
