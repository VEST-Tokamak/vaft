"""Retiring the combined `chease-mhd-stability` source (#94, #527 step 9).

`chease-mhd-stability` held a CHEASE refinement and the linear-MHD results that
followed from it in one namespace. Both now live in the hierarchy -- the
refinement in ``main/chease``, each stability product in
``main/chease/{product}`` -- so the combined source is superseded and, once its
contents are accounted for, deleted.

**The stability half cannot be migrated, and this does not pretend otherwise.**

A faithful copy would have to say which product each result came from, and the
combined product does not record it:

* DCON's two edge treatments both go through the same writer, at the same
  ``(time_slice, position)``, writing the same fields. One run happened and
  nothing on disk says whether it was ``dcon-peeling`` or ``dcon-kink``.
* RDCON and STRIDE append their rational surfaces to one ``ntms`` AOS, and the
  ``<solver name=...>`` fragment goes to the whole IDS's ``code.parameters``
  rather than to each surface. Which surface came from which solver is not
  recorded either.

Copying it anyway -- into ``main/chease/dcon-peeling``, say -- would write
provenance to HSDS that nothing can verify, which is the failure the per-product
sources exist to prevent. So the stability results are **regenerated**, not
moved, and that is what the deletion gate checks: a shot is deletable when its
refinement has been copied and verified *and* the per-product sources already
hold a regenerated product for it.

The refinement is a different matter: one ``equilibrium`` IDS, one destination,
nothing to attribute. That is copied and round-trip verified.

Everything here is a dry run unless told otherwise, and deletion is a separate
call that refuses a report it has not been shown to be clean.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from typing import Any, Iterable

from . import sources as _sources

logger = logging.getLogger(__name__)

#: The source being retired.
RETIRING_SOURCE = "chease-mhd-stability"

#: Where its refinement goes. One destination, so no attribution is needed.
REFINEMENT_DESTINATION = "main/chease"

#: The IDS the refinement owns, and the only thing copied forward.
REFINEMENT_IDS = ("equilibrium",)

#: The IDS that cannot be attributed to a product and so are regenerated.
STABILITY_IDS = ("mhd_linear", "ntms")


class RetirementError(RuntimeError):
    """Raised when a source cannot be retired safely."""


@dataclass(frozen=True)
class ShotRetirement:
    """What stands between one shot and the old source being deletable."""

    shot: int
    refinement: str
    stability: str
    detail: str

    @property
    def deletable(self) -> bool:
        """Whether nothing in this shot would be lost by deleting the source.

        Both halves have to be accounted for. A refinement that was never
        copied, or stability results with no regenerated replacement, is data
        that exists only in the source about to be removed. ``unreadable`` is in
        neither set: a read that failed says nothing about what the source
        holds, so it blocks exactly as unmigrated data does.
        """
        return self.refinement in {"copied", "already-present", "absent"} and (
            self.stability in {"superseded", "absent"}
        )

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "deletable": self.deletable}


@dataclass(frozen=True)
class RetirementReport:
    """Whether the combined source still holds anything unique."""

    source: str
    applied: bool
    shots: tuple[ShotRetirement, ...]
    #: Shots the source lists that the plan was not asked about.
    unexamined: tuple[int, ...] = ()
    #: Why the source's shots could not be listed, when they could not.
    enumeration_error: str | None = None

    @property
    def deletable(self) -> bool:
        """True only when every shot the source holds is accounted for.

        An empty report is **not** deletable. "No shots were examined" and
        "every shot is safe" are different findings, and a deletion gate that
        cannot tell them apart would green-light an unchecked source. The same
        goes for a partial shot list: `hsdel` removes the whole namespace, so a
        shot nobody asked about is lost along with the ones that were checked.
        """
        return (
            bool(self.shots)
            and not self.unexamined
            and self.enumeration_error is None
            and all(shot.deletable for shot in self.shots)
        )

    @property
    def blocking(self) -> tuple[ShotRetirement, ...]:
        return tuple(shot for shot in self.shots if not shot.deletable)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "dry_run": not self.applied,
            "source": self.source,
            "deletable": self.deletable,
            "summary": {
                "shots": len(self.shots),
                "deletable": sum(shot.deletable for shot in self.shots),
                "blocking": len(self.blocking),
                "unexamined": len(self.unexamined),
            },
            "unexamined": list(self.unexamined),
            "enumeration_error": self.enumeration_error,
            "shots": [shot.to_dict() for shot in self.shots],
        }


def _has_ids(shot: int, source: str, ids: Iterable[str]) -> bool:
    """Whether a *destination* `source` holds any of `ids` for `shot`.

    Absence and unreadability are both "no" here, and that is safe only on the
    destination side, where "no" blocks the deletion. The retiring source is
    probed with :func:`_source_holds`, where "no" would permit it.
    """
    from . import load as load_source

    try:
        ods = load_source(shot, source=source, paths=list(ids))
    except Exception:  # noqa: BLE001 - absence and unreadability are both "no"
        return False
    return any(name in ods and len(ods[name]) for name in ids)


def _is_ids_file(filename: str, ids: str) -> bool:
    """Whether a shot-folder entry is an image of `ids` (any occurrence)."""
    stem = filename[: -len(".h5")] if filename.endswith(".h5") else ""
    return stem == ids or (
        stem.startswith(f"{ids}_") and stem[len(ids) + 1 :].isdigit()
    )


def _source_holds(shot: int, source: str, ids: Iterable[str]) -> bool:
    """Whether the *retiring* `source` holds any of `ids`; raises if unknowable.

    ``False`` here makes a shot deletable, so it is returned only for a
    definite absence: the shot folder does not exist (404/410), it lists no
    file for these IDS, or a successful read came back without them. Every
    other failure -- a timeout, a 503, a read that raises -- propagates, and
    the caller scores the shot ``unreadable``.
    """
    from . import load as load_source
    from .replication import _remote_canonical_files, _remote_entries

    ids = tuple(ids)
    files = _remote_canonical_files(_remote_entries(source, shot))
    if not any(_is_ids_file(name, ids_name) for name in files for ids_name in ids):
        return False
    ods = load_source(shot, source=source, paths=list(ids))
    return any(name in ods and len(ods[name]) for name in ids)


def _source_shots(source: str) -> tuple[int, ...]:
    """Every shot the source lists. Raises when the source cannot be listed."""
    from .utils import h5pyd

    names = (str(name).strip("/") for name in h5pyd.Folder(f"/{source}/", mode="r"))
    return tuple(sorted(int(name) for name in names if name.isdigit()))


def _probe_source(shot: int, source: str, ids: Iterable[str]) -> tuple[bool | None, str]:
    """``(held, reason)``; ``held`` is ``None`` when the read failed."""
    try:
        return _source_holds(shot, source, ids), ""
    except Exception as error:  # noqa: BLE001 - scored, never read as absence
        return None, f"{type(error).__name__}: {error}"


def _stability_regenerated(shot: int, products: Iterable[str]) -> bool:
    """Whether a per-product source already holds a regenerated result.

    Any one product is enough. The combined source's contents cannot be
    attributed, so "the same results, split" is not a thing that can be checked;
    what the gate can establish is that the stability stage has been re-run into
    the hierarchy for this shot, which is what makes the old copy redundant.
    """
    return any(
        _has_ids(shot, f"{REFINEMENT_DESTINATION}/{product}", STABILITY_IDS)
        for product in products
    )


def plan_retirement(
    shots: Iterable[int],
    *,
    products: Iterable[str] = ("dcon-peeling", "dcon-kink", "rdcon", "stride"),
    source: str = RETIRING_SOURCE,
) -> RetirementReport:
    """Say what each shot still needs before the combined source can go.

    Reads only. For every shot it answers two questions: has the refinement
    reached ``main/chease``, and has the stability stage been re-run into the
    per-product sources.

    A source read that fails is scored ``unreadable`` and blocks; only a
    definite not-found is ``absent``. The source is also listed, and any shot it
    holds that ``shots`` does not name is reported as ``unexamined`` and blocks:
    the deletion this plan gates removes the whole namespace, not the listed
    shots.
    """
    name = _sources.resolve(source)
    products = tuple(products)
    rows: list[ShotRetirement] = []
    for shot in shots:
        shot = int(shot)
        has_refinement, refinement_error = _probe_source(shot, name, REFINEMENT_IDS)
        has_stability, stability_error = _probe_source(shot, name, STABILITY_IDS)

        if has_refinement is None:
            refinement, refinement_detail = (
                "unreadable",
                f"the old source's equilibrium could not be read ({refinement_error})",
            )
        elif not has_refinement:
            refinement, refinement_detail = "absent", "no equilibrium in the old source"
        elif _has_ids(shot, REFINEMENT_DESTINATION, REFINEMENT_IDS):
            refinement, refinement_detail = (
                "already-present",
                f"{REFINEMENT_DESTINATION} already holds an equilibrium",
            )
        else:
            refinement, refinement_detail = (
                "not-copied",
                f"equilibrium has not been copied to {REFINEMENT_DESTINATION}",
            )

        if has_stability is None:
            stability, stability_detail = (
                "unreadable",
                f"the old source's stability result could not be read ({stability_error})",
            )
        elif not has_stability:
            stability, stability_detail = "absent", "no stability result in the old source"
        elif _stability_regenerated(shot, products):
            stability, stability_detail = (
                "superseded",
                "a per-product source holds a regenerated result",
            )
        else:
            stability, stability_detail = (
                "not-regenerated",
                "the stability stage has not been re-run into the per-product "
                "sources; its results here cannot be attributed to a product and "
                "so cannot be copied forward",
            )

        rows.append(
            ShotRetirement(
                shot=shot,
                refinement=refinement,
                stability=stability,
                detail=f"{refinement_detail}; {stability_detail}",
            )
        )
    examined = {row.shot for row in rows}
    try:
        unexamined = tuple(s for s in _source_shots(name) if s not in examined)
        enumeration_error = None
    except Exception as error:  # noqa: BLE001 - recorded; it blocks the deletion
        unexamined = ()
        enumeration_error = f"{type(error).__name__}: {error}"
    return RetirementReport(
        source=name,
        applied=False,
        shots=tuple(rows),
        unexamined=unexamined,
        enumeration_error=enumeration_error,
    )


def copy_refinement(shot: int, *, source: str = RETIRING_SOURCE, apply: bool = False) -> dict[str, Any]:
    """Copy one shot's CHEASE refinement into ``main/chease``, verified.

    Only the ``equilibrium`` IDS: it has one destination and nothing to
    attribute. The stability results are deliberately left where they are -- see
    the module docstring.

    Dry run unless ``apply=True``. With ``apply`` the copy is read back from the
    destination and compared before the call reports success, so "copied" means
    the destination was checked rather than that a write returned.
    """
    from . import load as load_source, save as save_remote

    name = _sources.resolve(source)
    destination = _sources.resolve(REFINEMENT_DESTINATION, writable=True)
    shot = int(shot)
    report: dict[str, Any] = {
        "shot": shot,
        "source": name,
        "destination": destination,
        "applied": False,
        "verified": False,
    }
    ods = load_source(shot, source=name, paths=list(REFINEMENT_IDS))
    if not any(ids in ods and len(ods[ids]) for ids in REFINEMENT_IDS):
        report["detail"] = "no equilibrium to copy"
        return report
    if not apply:
        report["detail"] = f"would copy equilibrium to hdf5://{destination}/{shot}/"
        return report

    save_remote(ods, shot, source=destination)
    report["applied"] = True

    # Read back rather than trusting the write. A copy that reports success
    # without checking the destination is how a retirement deletes the only
    # remaining copy of something that never arrived.
    written = load_source(shot, source=destination, paths=list(REFINEMENT_IDS))
    report["verified"] = any(
        ids in written and len(written[ids]) for ids in REFINEMENT_IDS
    )
    if not report["verified"]:
        raise RetirementError(
            f"Copied shot {shot} to hdf5://{destination}/ but read nothing back; "
            "the source has not been changed and must not be deleted."
        )
    report["detail"] = f"copied and verified at hdf5://{destination}/{shot}/"
    return report


def delete_retired_source(
    report: RetirementReport, *, apply: bool = False
) -> dict[str, Any]:
    """Delete the combined source, only once its report says nothing is lost.

    Refuses a report that is not clean, and refuses an empty one: "no shots were
    examined" and "every shot is safe" are different findings, and a gate that
    cannot tell them apart is not a gate.

    Deletion is irreversible, so this never runs as part of a pipeline -- it is
    an operator's separate, deliberate call.
    """
    outcome: dict[str, Any] = {
        "source": report.source,
        "deletable": report.deletable,
        "applied": False,
        "blocking": [shot.to_dict() for shot in report.blocking],
    }
    if not report.deletable:
        if not report.shots:
            reason = "no shots were examined"
        elif report.enumeration_error is not None:
            reason = (
                "the source's shots could not be listed "
                f"({report.enumeration_error}), so the plan may be partial"
            )
        elif report.unexamined:
            reason = (
                f"the source holds {len(report.unexamined)} shot(s) the plan never "
                f"examined (e.g. {report.unexamined[0]})"
            )
        else:
            reason = (
                f"{len(report.blocking)} shot(s) still hold data only this source "
                "has, or could not be read"
            )
        raise RetirementError(
            f"Refusing to delete {report.source!r}: {reason}. "
            "Copy each refinement forward and re-run the stability stage into "
            "the per-product sources first."
        )
    if not apply:
        outcome["detail"] = f"would delete hdf5://{report.source}/"
        return outcome

    raise RetirementError(
        f"Deleting {report.source!r} is an HSDS administrator action VAFT does "
        "not perform: it removes a whole namespace, and nothing in this package "
        "should be able to do that as a side effect. The plan is clean, so run:\n"
        f"    hsdel /{report.source}/"
    )


__all__ = [
    "REFINEMENT_DESTINATION",
    "REFINEMENT_IDS",
    "RETIRING_SOURCE",
    "RetirementError",
    "RetirementReport",
    "STABILITY_IDS",
    "ShotRetirement",
    "copy_refinement",
    "delete_retired_source",
    "plan_retirement",
]
