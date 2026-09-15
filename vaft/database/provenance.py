"""Checking that a stability product descends from the equilibrium it claims.

VAFT's pipeline is a chain -- magnetics reconstruct an equilibrium, CHEASE
refines it, a stability solver consumes the refinement -- and until now the
chain recorded only *paths*. A path identifies the file that was read only if
that file never changed afterwards, which is precisely the assumption
:func:`vaft.database.replication.is_reusable` exists so that replication does
not have to make. Provenance was still making it.

Each stage now records the digest of what it consumed and of what it produced::

    EFIT      equilibrium.code.parameters...artifacts.gfile.sha256
                |
    CHEASE    manifest["input"][i]["sha256"]        (what it read)
              manifest["input"][i]["output_sha256"] (what it wrote)
                |
    stability GPECSuiteResult.input_equilibrium_sha256

The links are joinable, so "which EFIT produced this stability result" becomes a
comparison rather than an act of faith. This module does that comparison.

What it deliberately does not do is *re-hash the tree*. A verifier that recomputes
digests answers "do these files agree with each other now", which is a different
and weaker question than "did this stage consume what that stage produced" -- the
file may have been replaced by an identical-looking one, or the chain may span
hosts where the upstream file no longer exists. Comparing recorded digests
answers the question that was asked, and answers it on a manifest alone.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


#: What a stage records when it could not take a digest -- an unreadable file,
#: or a product written before the chain existed. Distinct from a mismatch: it
#: says nothing was claimed, where a mismatch says something false was.
NOT_RECORDED = ""


@dataclass(frozen=True)
class ChainLink:
    """One consumed-produced pair, and whether the two agree."""

    stage: str
    upstream_stage: str
    consumed_sha256: str
    produced_by_upstream_sha256: str
    status: str
    detail: str = ""

    @property
    def verified(self) -> bool:
        return self.status == "verified"


@dataclass(frozen=True)
class ChainReport:
    """Whether a product's recorded ancestry holds together."""

    shot: int | None
    links: tuple[ChainLink, ...]

    @property
    def verified(self) -> bool:
        """True only when every link was checked and agreed.

        An unrecorded digest is not a pass. A chain with gaps is exactly the
        state this module exists to make visible, so it must not report the
        same verdict as one that was checked.
        """
        return bool(self.links) and all(link.verified for link in self.links)

    @property
    def broken(self) -> tuple[ChainLink, ...]:
        """Links whose two digests disagree -- a product from another run."""
        return tuple(link for link in self.links if link.status == "mismatch")

    @property
    def unrecorded(self) -> tuple[ChainLink, ...]:
        return tuple(link for link in self.links if link.status == "not_recorded")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "shot": self.shot,
            "verified": self.verified,
            "summary": {
                "links": len(self.links),
                "verified": sum(link.verified for link in self.links),
                "mismatch": len(self.broken),
                "not_recorded": len(self.unrecorded),
            },
            "links": [asdict(link) for link in self.links],
        }


def _link(stage: str, upstream: str, consumed: str, produced: str) -> ChainLink:
    if consumed == NOT_RECORDED or produced == NOT_RECORDED:
        missing = "both" if consumed == produced else (
            stage if consumed == NOT_RECORDED else upstream
        )
        return ChainLink(
            stage=stage,
            upstream_stage=upstream,
            consumed_sha256=consumed,
            produced_by_upstream_sha256=produced,
            status="not_recorded",
            detail=(
                f"no digest recorded by {missing}; the product predates the "
                "provenance chain, or the file could not be read when it ran"
            ),
        )
    if consumed != produced:
        return ChainLink(
            stage=stage,
            upstream_stage=upstream,
            consumed_sha256=consumed,
            produced_by_upstream_sha256=produced,
            status="mismatch",
            detail=(
                f"{stage} consumed an equilibrium {upstream} did not produce; "
                "the two products come from different runs"
            ),
        )
    return ChainLink(
        stage=stage,
        upstream_stage=upstream,
        consumed_sha256=consumed,
        produced_by_upstream_sha256=produced,
        status="verified",
    )


def verify_chain(
    *,
    chease_manifest: dict[str, Any],
    stability_input_sha256: str,
    efit_gfile_sha256: str | None = None,
    shot: int | None = None,
) -> ChainReport:
    """Check a stability product's recorded ancestry against its upstreams.

    ``chease_manifest`` is the CHEASE stage manifest, whose ``input`` block
    pairs each consumed g-file digest with the digest of the refinement it
    produced. ``stability_input_sha256`` is
    :attr:`~vaft.code.gpec.GPECSuiteResult.input_equilibrium_sha256`.
    ``efit_gfile_sha256``, when given, additionally checks that CHEASE consumed
    the reconstruction that EFIT recorded producing.

    The stability link passes when *any* refinement in the manifest produced the
    equilibrium the solver consumed -- a CHEASE stage refines every time slice
    of a shot, and a stability cell runs on one of them, so requiring a
    particular entry would fail every well-formed chain.
    """
    entries = list(chease_manifest.get("input", []))
    links: list[ChainLink] = []

    if efit_gfile_sha256 is not None:
        consumed = {
            entry.get("sha256", NOT_RECORDED)
            for entry in entries
            if entry.get("sha256", NOT_RECORDED) != NOT_RECORDED
        }
        matched = efit_gfile_sha256 if efit_gfile_sha256 in consumed else (
            next(iter(sorted(consumed))) if consumed else NOT_RECORDED
        )
        links.append(_link("chease", "efit", efit_gfile_sha256, matched))

    produced = {
        entry.get("output_sha256", NOT_RECORDED)
        for entry in entries
        if entry.get("output_sha256", NOT_RECORDED) != NOT_RECORDED
    }
    matched = stability_input_sha256 if stability_input_sha256 in produced else (
        next(iter(sorted(produced))) if produced else NOT_RECORDED
    )
    links.append(_link("stability", "chease", stability_input_sha256, matched))

    return ChainReport(shot=shot, links=tuple(links))


__all__ = [
    "NOT_RECORDED",
    "ChainLink",
    "ChainReport",
    "verify_chain",
]
