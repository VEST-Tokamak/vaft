"""One-off repairs to already-published HSDS shots (issue #305).

IMPA channels used to be appended to the baseline `magnetics` product, so shots
published before the split carry them inside `main`.  Nothing downstream is
broken by that -- every consumer locates them by identifier -- but `main` is
supposed to mean "routine baseline magnetics", and a lingering optional
diagnostic makes that claim false for exactly the shots nobody thinks to check.

This rewrites published baseline data, so it is a dry run unless told otherwise,
and it refuses any shot where the removal would move a surviving probe index:
k-files and the EFIT constraint builder address probes positionally, so a shift
would silently re-point them.  The array is appended last, so on an untouched
product the removal is a truncation and the refusal never fires.
"""

from __future__ import annotations

import logging
from pathlib import Path
import tempfile
from typing import Any, Iterable

from ..machine_mapping.utils import path_exists
from . import sources as _sources

logger = logging.getLogger(__name__)

#: Both nodes the array lands on, mirroring `vaft.database.composition`.
_PROBE_NODES = ("magnetics.b_field_tor_probe", "magnetics.b_field_pol_probe")

__all__ = [
    "ImpaResidue",
    "ImpaStripError",
    "ImpaStripWriteError",
    "inspect_impa_residue",
    "strip_impa_from_shots",
    "strip_impa_from_source",
]


class ImpaStripError(RuntimeError):
    """Raised when a published shot cannot be stripped safely."""


class ImpaStripWriteError(ImpaStripError):
    """Raised when the rewrite failed after it had started touching the source."""


class ImpaResidue(dict):
    """What one published shot still carries, and whether it can be removed."""

    @property
    def carries_impa(self) -> bool:
        return bool(self["nodes"])

    @property
    def removable(self) -> bool:
        return self.carries_impa and not self["refusals"]


def _probe_count(ods: Any, node: str) -> int:
    try:
        return len(ods[node])
    except (KeyError, IndexError, TypeError, ValueError):
        return 0


def inspect_impa_residue(ods: Any) -> ImpaResidue:
    """Report the IMPA channels in ``ods`` and whether they form a tail block."""
    from ..machine_mapping.impa import impa_probe_indices

    nodes: dict[str, list[int]] = {}
    refusals: list[str] = []
    for node in _PROBE_NODES:
        # Guarded: asking about an absent node would materialize it, and with
        # ``apply`` this ODS is written straight back to the source -- a repair
        # must not add an empty probe array to a published shot.
        if not path_exists(ods, node):
            continue
        indices = impa_probe_indices(ods, node)
        if not indices:
            continue
        nodes[node] = indices
        total = _probe_count(ods, node)
        tail = list(range(total - len(indices), total))
        if sorted(indices) != tail:
            refusals.append(
                f"{node}: IMPA occupies {sorted(indices)}, which is not the tail "
                f"{tail} of {total} probes; removing it would move a surviving index."
            )
    return ImpaResidue({"nodes": nodes, "refusals": refusals})


def _strip(ods: Any, residue: ImpaResidue) -> int:
    removed = 0
    for node, indices in residue["nodes"].items():
        # Highest first, so each deletion is a truncation of the array rather
        # than a shift of the entries after it.
        for index in sorted(indices, reverse=True):
            del ods[f"{node}.{index}"]
            removed += 1
        # A node the array had to itself is dropped rather than left as an empty
        # array: the repaired product should not advertise a probe set it has
        # none of.
        if _probe_count(ods, node) == 0 and node in ods:
            del ods[node]
    return removed


def strip_impa_from_source(
    shot: int,
    *,
    source: str | None = None,
    apply: bool = False,
) -> dict[str, Any]:
    """Remove IMPA channels from one published shot's baseline magnetics.

    Returns a report in both modes.  With ``apply=False`` (the default) nothing
    is written and the report says what would change; with ``apply=True`` the
    magnetics IDS is rewritten into the same source. The pre-write master's
    links are merged into the new master *before* it is uploaded -- the ordering
    replication uses -- so the shot's other IDS are never hidden, even when the
    write is interrupted.

    A shot with no IMPA left is still checked for a master that does not link
    every IDS file in its folder, which is what an interrupted run of an older
    version of this repair left behind. ``master_unlinked`` names those files;
    with ``apply=True`` they are relinked and ``master_repaired`` is set.
    """
    from . import load as load_source, save as save_remote
    from .replication import (
        _fetch_remote_master,
        _master_finalizer,
        merge_remote_master,
        unlinked_remote_files,
    )

    name = _sources.resolve(source, writable=True)
    shot = int(shot)
    ods = load_source(shot, source=name, paths=["magnetics"])
    residue = inspect_impa_residue(ods)
    report: dict[str, Any] = {
        "shot": shot,
        "source": name,
        "carries_impa": residue.carries_impa,
        "channels": {node: sorted(indices) for node, indices in residue["nodes"].items()},
        "refusals": residue["refusals"],
        "applied": False,
        "removed": 0,
        "master_unlinked": [],
        "master_repaired": False,
    }
    if not residue.carries_impa:
        # Not an early return: a previous run that failed after replacing the
        # master leaves exactly this state -- nothing to strip, other stages
        # hidden -- and a re-run is the operator's only handle on it.
        report["master_unlinked"] = list(
            unlinked_remote_files(name, shot, repair=apply)
        )
        report["master_repaired"] = bool(apply and report["master_unlinked"])
        return report
    if residue["refusals"]:
        raise ImpaStripError(
            f"Refusing to strip IMPA from hdf5://{name}/{shot}/: "
            + " ".join(residue["refusals"])
        )
    report["removed"] = len(
        [index for indices in residue["nodes"].values() for index in indices]
    )
    if not apply:
        return report

    _strip(ods, residue)
    with tempfile.TemporaryDirectory(prefix="vaft-strip-impa-") as workdir:
        previous_master = _fetch_remote_master(
            name, shot, Path(workdir) / "master.previous.h5"
        )
        try:
            save_remote(
                ods,
                shot,
                source=name,
                finalize_master=_master_finalizer(name, shot, previous_master),
            )
            # The safety net replication keeps: after a local merge it finds
            # nothing to add and returns without writing.
            merge_remote_master(name, shot, previous_master)
        except Exception as error:
            raise ImpaStripWriteError(
                f"Rewriting hdf5://{name}/{shot}/ failed after the write began "
                f"({type(error).__name__}: {error}). The stored master still "
                "links every IDS it did, but the magnetics payload may or may "
                "not have been replaced; re-run to finish and to check the master."
            ) from error
    report["applied"] = True
    logger.info(
        "shot %s: removed %s IMPA channels from %s", shot, report["removed"], name
    )
    return report


def strip_impa_from_shots(
    shots: Iterable[int],
    *,
    source: str | None = None,
    apply: bool = False,
) -> list[dict[str, Any]]:
    """Run :func:`strip_impa_from_source` over many shots, recording refusals."""
    reports = []
    for shot in shots:
        try:
            reports.append(strip_impa_from_source(shot, source=source, apply=apply))
        except Exception as error:  # noqa: BLE001 - one bad shot must not stop the audit
            reports.append(
                {
                    "shot": int(shot),
                    "source": source,
                    "error": f"{type(error).__name__}: {error}",
                    "applied": False,
                    # "not applied" alone would read as "untouched". A failure
                    # after the write began leaves the remote state unknown.
                    "write_started": isinstance(error, ImpaStripWriteError),
                }
            )
    return reports
