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
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "CATEGORIES",
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
