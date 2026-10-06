"""Whether EFIT applies to a shot at all, carried from the constraint step on (#205).

A vacuum shot (or a breakdown failure) has no plasma current to reconstruct:
every constraint instant is below EFIT's ``CUTIP``.  That is a result about
the shot, not a fault of the pipeline, so the constraint step records it and
every EFIT step after it passes it on -- the same way ``efit.run=false`` does
-- instead of failing and being retried.

The verdict travels inside the files the rules already exchange, so no rule
gains an output:

* the constraints ODS carries no ``equilibrium.time`` and an
  ``equilibrium.ids_properties.comment`` beginning with
  :data:`EFIT_NOT_APPLICABLE`;
* the k-file manifest lists no k-file and one comment line beginning with
  :data:`KFILE_MANIFEST_NOT_APPLICABLE`;
* the EFIT status reads ``skipped: not applicable: <reason>``, which
  replication accepts as nothing-to-publish by design.
"""

from __future__ import annotations

from typing import Any

#: Prefix of the constraints ODS comment of a shot EFIT does not apply to.
EFIT_NOT_APPLICABLE = "EFIT not applicable: "
#: Prefix of the k-file manifest line of such a shot (never a k-file path).
KFILE_MANIFEST_NOT_APPLICABLE = "# EFIT not applicable: "


def not_applicable_constraints(reason: str) -> Any:
    """The constraints ODS of a shot EFIT does not apply to: the verdict and nothing else."""
    from omas import ODS

    ods = ODS(consistency_check=False)
    ods["equilibrium.ids_properties.homogeneous_time"] = 1
    ods["equilibrium.ids_properties.comment"] = EFIT_NOT_APPLICABLE + _one_line(reason)
    return ods


def constraints_not_applicable_reason(ods: Any) -> str | None:
    """The recorded reason when ``ods`` is a not-applicable constraints ODS, else ``None``.

    Both marks must agree: a constraints product with time slices is
    reconstructed whatever its comment says.
    """
    try:
        comment = str(ods["equilibrium.ids_properties.comment"])
    except Exception:
        return None
    if not comment.startswith(EFIT_NOT_APPLICABLE):
        return None
    try:
        times = ods["equilibrium.time"]
    except Exception:
        times = ()
    if len(times):
        return None
    return comment[len(EFIT_NOT_APPLICABLE):]


def kfile_manifest_text(reason: str) -> str:
    """The k-file manifest of a shot EFIT does not apply to."""
    return KFILE_MANIFEST_NOT_APPLICABLE + _one_line(reason) + "\n"


def kfile_manifest_not_applicable_reason(text: str) -> str | None:
    """The reason a k-file manifest records, or ``None`` for an ordinary manifest."""
    for line in text.splitlines():
        if line.startswith(KFILE_MANIFEST_NOT_APPLICABLE):
            return line[len(KFILE_MANIFEST_NOT_APPLICABLE):].strip()
    return None


def kfile_manifest_paths(text: str) -> list[str]:
    """The k-file paths a manifest lists; comment lines are not paths."""
    return [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]


def _one_line(reason: str) -> str:
    return " ".join(str(reason).split())


__all__ = [
    "EFIT_NOT_APPLICABLE",
    "KFILE_MANIFEST_NOT_APPLICABLE",
    "constraints_not_applicable_reason",
    "kfile_manifest_not_applicable_reason",
    "kfile_manifest_paths",
    "kfile_manifest_text",
    "not_applicable_constraints",
]
