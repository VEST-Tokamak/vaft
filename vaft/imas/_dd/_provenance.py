"""Upstream references carried in ``ids_properties.provenance`` (#1128).

Lineage lives in the IDS, like the name does, so a copied Data Entry keeps
it. One whole-IDS provenance node (empty ``path``) lists the upstream IDS
instances this occurrence was derived from, each written as

``"<ids>/<name>"``
    a named upstream instance -- preferred, stable across storage moves;
``"<ids>:<occurrence>"``
    an unnamed upstream occurrence of the same *physical* entry the dependent
    is stored in (store-local; only a name crosses stores).

The node's shape changed in Data Dictionary 3.42, and both are handled here:
older versions store ``node[].sources`` (a string list), 3.42 and DD 4 store
``node[].reference[]`` structures whose ``name`` holds the string. Strings
that match neither form -- a URI written by another tool -- are preserved on
write and ignored for coherence.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import re
from typing import Any, Iterable

_NAME_REF = re.compile(r"^([a-z][a-z0-9_]*)/([^/:\s]+)$")
_OCC_REF = re.compile(r"^([a-z][a-z0-9_]*):(\d+)$")


@dataclass(frozen=True)
class UpstreamRef:
    """One parsed upstream reference: a name or an occurrence, never both."""

    ids: str
    name: str | None = None
    occurrence: int | None = None

    def __post_init__(self) -> None:
        if (self.name is None) == (self.occurrence is None):
            raise ValueError("an upstream reference names exactly one of name/occurrence")

    def __str__(self) -> str:
        return format_reference(self.ids, name=self.name, occurrence=self.occurrence)


def format_reference(ids: str, *, name: str | None = None, occurrence: int | None = None) -> str:
    """Encode an upstream reference: ``"<ids>/<name>"`` or ``"<ids>:<occurrence>"``."""
    if (name is None) == (occurrence is None):
        raise ValueError("give exactly one of name= or occurrence=")
    if name is not None:
        if not name or "/" in name or ":" in name or name != name.strip():
            raise ValueError(f"cannot encode instance name {name!r} in a reference")
        return f"{ids}/{name}"
    if int(occurrence) < 0:
        raise ValueError(f"occurrence must be >= 0, got {occurrence}")
    return f"{ids}:{int(occurrence)}"


def parse_reference(text: str) -> UpstreamRef | None:
    """Parse one reference string; ``None`` if it is not in either VAFT form."""
    text = str(text).strip()
    match = _NAME_REF.match(text)
    if match:
        return UpstreamRef(match.group(1), name=match.group(2))
    match = _OCC_REF.match(text)
    if match:
        return UpstreamRef(match.group(1), occurrence=int(match.group(2)))
    return None


def _node_strings(node: Any) -> list[str]:
    if hasattr(node, "reference"):  # DD >= 3.42
        return [str(ref.name) for ref in node.reference if str(ref.name)]
    return [str(value) for value in node.sources if str(value)]  # DD < 3.42


def read_references(ids: Any) -> tuple[str, ...]:
    """Every reference string on the IDS's whole-IDS provenance nodes."""
    out: list[str] = []
    for node in ids.ids_properties.provenance.node:
        if str(node.path) == "":
            out.extend(_node_strings(node))
    return tuple(out)


def upstream(ids: Any) -> tuple[UpstreamRef, ...]:
    """The parseable upstream references of an IDS, in stored order."""
    return tuple(
        ref for ref in (parse_reference(text) for text in read_references(ids)) if ref
    )


def write_references(ids: Any, references: Iterable[str]) -> None:
    """Set the whole-IDS provenance node to ``references``.

    Replaces only VAFT-form strings on the whole-IDS node; other strings on it,
    and every node that describes a sub-path, are kept.
    """
    references = [str(ref) for ref in references]
    for ref in references:
        if parse_reference(ref) is None:
            raise ValueError(f"not a VAFT upstream reference: {ref!r}")
    nodes = ids.ids_properties.provenance.node
    target = next((node for node in nodes if str(node.path) == ""), None)
    if target is None:
        nodes.resize(len(nodes) + 1, keep=True)
        target = nodes[-1]
        target.path = ""
    if hasattr(target, "reference"):  # DD >= 3.42: reference[] structures
        # Keep foreign references whole (name *and* timestamp, ...), not just
        # their names: snapshot every leaf, rebuild, then append VAFT's.
        kept = [
            {child.metadata.name: copy.deepcopy(child.value) for child in slot}
            for slot in target.reference
            if parse_reference(str(slot.name)) is None
        ]
        target.reference.resize(0)
        target.reference.resize(len(kept) + len(references))
        for slot, fields in zip(target.reference, kept):
            for field, value in fields.items():
                slot[field] = value
        for index, value in enumerate(references, start=len(kept)):
            target.reference[index].name = value
    else:  # DD < 3.42: a plain string list
        kept = [text for text in _node_strings(target) if parse_reference(text) is None]
        target.sources = kept + references


__all__ = [
    "UpstreamRef",
    "format_reference",
    "parse_reference",
    "read_references",
    "upstream",
    "write_references",
]
