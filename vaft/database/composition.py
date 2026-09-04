"""Explicit composition of one shot across named HSDS sources (issue #305).

Reading a source returns that source and nothing else: `main` never silently
carries `impa`, and no union of the two is presented as one shot.  That is what
keeps a missing entry meaningful in each -- absence from `impa` says only that no
IMPA product was published, never that the baseline shot is incomplete.

Analysis that wants both asks for both, here, and the result says where each
channel came from.  The composition is deliberately one-directional: the base
source's probe indices never move, because k-files and the EFIT constraint
builder address probes by index, so the optional channels are appended after
them exactly as an in-product mapping would have.
"""

from __future__ import annotations

import copy
from typing import Any, Iterable, Mapping, Sequence

from . import sources as _sources


#: Both nodes the array can land on: Hall channels are mounted for the toroidal
#: field, the vertical-field sensors are poloidal-plane probes.
_PROBE_NODES = ("magnetics.b_field_tor_probe", "magnetics.b_field_pol_probe")

__all__ = ["compose", "impa_channels"]


def _probe_count(ods: Any, node: str) -> int:
    try:
        return len(ods[node])
    except (KeyError, IndexError, TypeError, ValueError):
        return 0


def impa_channels(ods: Any, node: str) -> list[int]:
    """Return the indices of ``node`` holding IMPA channels, by identifier."""
    from ..machine_mapping.impa import impa_probe_indices

    return impa_probe_indices(ods, node)


def compose(
    shot: int,
    sources: Sequence[str] = ("main", "impa"),
    *,
    paths: Iterable[str] | None = None,
    occurrence: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load ``shot`` from several sources and append the optional channels.

    ``sources[0]`` is the base whose indices are preserved; every further source
    contributes its IMPA-identified probes, appended after whatever the base
    already holds.  Returns the composed ODS and a provenance record naming, for
    each appended index, the source and the index it had there.

    Nothing else in VAFT calls this: composition is the caller's decision, and
    source identity stays visible in the result rather than being flattened
    away.
    """
    from . import load as load_source

    names = [_sources.resolve(name) for name in sources]
    if len(names) < 2:
        raise ValueError(
            "compose needs a base source and at least one source to compose onto "
            f"it; got {names}."
        )

    read: dict[str, Any] = {}
    base_name = names[0]
    base = load_source(
        shot, source=base_name, paths=list(paths) if paths else None, occurrence=occurrence
    )
    provenance: dict[str, Any] = {
        "shot": int(shot),
        "base": base_name,
        "sources": names,
        "appended": [],
        "contributed": {},
    }

    for name in names[1:]:
        extra = load_source(
            shot, source=name, paths=list(paths) if paths else None, occurrence=occurrence
        )
        read[name] = extra
        contributed = 0
        for node in _PROBE_NODES:
            for source_index in impa_channels(extra, node):
                index = _probe_count(base, node)
                base[f"{node}.{index}"] = copy.deepcopy(extra[f"{node}.{source_index}"])
                provenance["appended"].append(
                    {
                        "source": name,
                        "node": node,
                        "source_index": int(source_index),
                        "index": int(index),
                    }
                )
                contributed += 1
        provenance["contributed"][name] = contributed
    return base, provenance
