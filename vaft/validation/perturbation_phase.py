"""The verdict on a toroidal phase audit (D-06).

:func:`vaft.process.perturbation.toroidal_phase_audit` measures how well each
of the two phase hypotheses reproduces a resampled field, and stops there: it
computes, it does not decide.  This module applies the criteria that turn those
two numbers into an answer, and it is where the numbers 3 and 0.10 live.

The two criteria are not redundant.  The **separation** rules out the case that
is by far the most common failure -- a mode the coil currents barely excite --
where both hypotheses come out near one and neither is right.  The **relative
norm** rules out the subtler case where one hypothesis is merely the less bad of
two bad ones.  A file passes only when the winner is close in absolute terms
*and* the loser is far away.

Neither rules out an alias from another harmonic, which produces a clean answer
that is confidently wrong; see
:func:`vaft.process.perturbation.toroidal_phase_audit`.  That is a property of
how the field was sampled, not of the two numbers reaching here, so no
threshold on them can catch it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from vaft.validation.model import ValidationStatus

__all__ = [
    "GPEC_COIL_FIELD",
    "PhaseAuditCriteria",
    "PhaseConventionVerdict",
    "phase_convention_verdict",
]


@dataclass(frozen=True)
class PhaseAuditCriteria:
    """What a toroidal phase audit has to achieve before it settles anything."""

    separation_ratio: float
    """How many times the losing relative norm must exceed the winning one."""
    relative_norm: float
    """How close the winning hypothesis must come, in its own right."""
    name: str = ""
    """The preset this came from, carried into the verdict's reason."""


#: The criteria the hsyun_GPEC ``codex/gpec-flare-cocos-handshake`` branch
#: applied to a GPEC coil field audited against Biot-Savart on its own coils
#: (``library/gpec_phase.py``: ``ratio >= 3.0 and min(...) <= 0.10``), and the
#: criteria decision D-06 adopts.  Measured on that branch's own reference
#: cases, a settled audit clears both by orders of magnitude -- relative norms
#: of 1e-8 to 1e-4 against separations of 1e4 to 1e8 -- so these are a floor
#: below which the measurement stops meaning anything, not a tuned boundary.
GPEC_COIL_FIELD = PhaseAuditCriteria(
    separation_ratio=3.0, relative_norm=0.10, name="GPEC_COIL_FIELD"
)


@dataclass(frozen=True)
class PhaseConventionVerdict:
    """What one phase audit settled, if anything."""

    status: ValidationStatus
    convention: str | None
    """``"stored"`` or ``"conjugate"`` on a pass, ``None`` otherwise."""
    reason: str


def phase_convention_verdict(audit, *, criteria: PhaseAuditCriteria) -> PhaseConventionVerdict:
    """Decide whether a phase audit settled which reconstruction a file belongs to.

    Parameters
    ----------
    audit : ToroidalPhaseAudit
        The measurement, from
        :func:`vaft.process.perturbation.toroidal_phase_audit`.
    criteria : PhaseAuditCriteria
        The two thresholds.  Required, and with no default anywhere: which
        separation counts as settled is a policy about a particular comparison,
        and :data:`GPEC_COIL_FIELD` is the preset for a GPEC coil field.

    Returns
    -------
    PhaseConventionVerdict
        ``PASS`` with the winning convention when both criteria hold,
        ``INDETERMINATE`` with the reason otherwise.  There is no ``FAIL``:
        a measurement that does not separate the hypotheses is evidence about
        the measurement, not about the file.

    Notes
    -----
    **Every criterion is written as a condition to satisfy, not one to fail.**
    A NaN loses every comparison, so ``separation < 3`` and ``winner > 0.10``
    are both False for a NaN and a measurement that produced nothing would
    reach the ``PASS`` at the bottom.  ``not (separation >= 3)`` is True there
    instead, and a non-finite metric is refused outright before either runs.
    """
    winner = min(audit.stored_relative_norm, audit.conjugate_relative_norm)
    label = f" [{criteria.name}]" if criteria.name else ""

    unusable = [
        name
        for name, value in (
            ("stored_relative_norm", audit.stored_relative_norm),
            ("conjugate_relative_norm", audit.conjugate_relative_norm),
            ("separation_ratio", audit.separation_ratio),
        )
        if not math.isfinite(value)
    ]
    if unusable:
        return PhaseConventionVerdict(
            ValidationStatus.INDETERMINATE,
            None,
            f"the audit produced no usable number for {', '.join(unusable)}"
            f"{label}; there is nothing here to compare against a criterion",
        )
    if audit.preferred is None:
        return PhaseConventionVerdict(
            ValidationStatus.INDETERMINATE,
            None,
            f"the two hypotheses are exactly equal at {winner:.3g}{label}; the "
            "stored harmonic is its own conjugate, or the projection returned "
            "nothing, and either way there is no winner to name",
        )
    if not (audit.separation_ratio >= criteria.separation_ratio):
        return PhaseConventionVerdict(
            ValidationStatus.INDETERMINATE,
            None,
            f"the two hypotheses are only {audit.separation_ratio:.3g} apart, under "
            f"{criteria.separation_ratio:g}{label}; the sampled field has little "
            f"n = {audit.n_tor} content",
        )
    if not (winner <= criteria.relative_norm):
        return PhaseConventionVerdict(
            ValidationStatus.INDETERMINATE,
            None,
            f"the {audit.preferred} hypothesis wins by {audit.separation_ratio:.3g} "
            f"but is itself off by {winner:.3g}, over {criteria.relative_norm:g}"
            f"{label}; it is the less bad of two bad answers",
        )
    return PhaseConventionVerdict(
        ValidationStatus.PASS,
        audit.preferred,
        f"the {audit.preferred} hypothesis reproduces the field to {winner:.3g} "
        f"and beats the other by {audit.separation_ratio:.3g}{label}",
    )
