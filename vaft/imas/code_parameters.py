"""The two halves of keeping `code.parameters` intact across a round trip.

`code.parameters` is a `STR_0D` in the Data Dictionary, and OMAS reaches it
through one object: a `CodeParameters`, which it serializes to XML on the way
to an entry and parses back on the way in.  Anything else under that path --
a plain ODS branch, which is what a JSON or HDF5 stage product restores -- is
not converted, and the Access Layer discards it with no exception and no entry
in the list of paths it wrote (issue #380).

So two things have to happen, and they are mirror images:

`promote_in_place`
    On **load**, turn a flat, leaf-only branch back into a `CodeParameters`,
    so a declaration written by a mapper and carried through a product still
    reaches an entry (issue #478).

`as_entry_payload`
    On **save**, replace a block that cannot be represented as a parameters
    string with the part of it that can, for the duration of the write.

The second exists because the first is all-or-nothing on a block, and a real
equilibrium product mixes the two kinds of content in one field: the declared
COCOS index is a flat leaf, and the EFIT per-slice parser cache is a nested
branch beside it.  Before this, the cache took the declaration down with it
and a replicated equilibrium arrived declaring nothing (issue #642).

What may not be written is measured rather than assumed.  A nested block does
not survive as itself: the XML encoder has no array representation, so every
array in it returns as a whitespace-joined string of `np.float64(...)` reprs
-- present, named, and useless.  Dropping it and saying so is the honest
outcome; ``test/test_code_parameters_contract.py`` holds this module to that
measurement.
"""

from __future__ import annotations

import functools
import logging
from contextlib import contextmanager
from typing import Any, Iterator

_logger = logging.getLogger(__name__)

#: Recorded in the entry in place of content that stayed on the local product.
CACHE_OMITTED_KEY = "parameters_cache_omitted"

_CODE_PARAMETERS = "code.parameters"


def _is_container(value: Any) -> bool:
    """Whether a value cannot be one text node of a parameters document.

    A nested block is the obvious case.  An array is the quiet one: it
    serializes to the repr of its elements and parses back as text, so
    carrying it would corrupt it rather than move it.
    """
    from omas import ODS

    if isinstance(value, (ODS, dict, list, tuple)):
        return True
    return hasattr(value, "shape") and getattr(value, "ndim", 0) > 0


def _split(branch) -> tuple[dict[str, Any], tuple[str, ...]]:
    """``(the leaves an entry can carry, the names of what it cannot)``."""
    leaves: dict[str, Any] = {}
    omitted: list[str] = []
    for key, value in branch.items():
        if _is_container(value):
            omitted.append(str(key))
        else:
            leaves[str(key)] = value
    return leaves, tuple(omitted)


def _code_parameters_paths(ods) -> Iterator[str]:
    """Every ``<ids>.code.parameters`` this ODS actually holds."""
    for ids in list(ods.keys()):
        path = f"{ids}.{_CODE_PARAMETERS}"
        try:
            present = path in ods
        except Exception:  # a dynamic backend may refuse the question
            continue
        if present:
            yield path


def promote_in_place(ods) -> None:
    """Turn every flat, leaf-only ``code.parameters`` branch into a `CodeParameters`.

    A JSON or HDF5 load with consistency checks off keeps the block as a plain
    ODS branch, and OMAS serializes only `CodeParameters` to the XML string the
    leaf holds -- so the block silently vanished on the way to an IMAS entry,
    taking the declared COCOS index with it (issue #478).

    A nested block is left as it was: it is a local parser cache, and
    :func:`as_entry_payload` is what decides what of it an entry sees.
    """
    from omas import ODS
    from omas.omas_core import CodeParameters

    for path in _code_parameters_paths(ods):
        branch = ods[path]
        if not isinstance(branch, ODS):
            continue
        leaves, omitted = _split(branch)
        if omitted or not leaves:
            continue
        promoted = CodeParameters()
        promoted.update(leaves)
        ods[path] = promoted


@contextmanager
def as_entry_payload(ods):
    """Hold ``ods`` in the shape an IMAS entry can accept, then put it back.

    For the duration of the block, every ``code.parameters`` that cannot be
    written as one parameters string is replaced by the leaves of it that can,
    plus a ``parameters_cache_omitted`` note naming what stayed behind -- so a
    reader of the entry learns that the local product holds more, which is
    exactly what nobody was told before.

    A string is left as it is on the way in and put back on the way out --
    OMAS's own ``codeparams_xml_save`` parses any XML-shaped string into a
    `CodeParameters` when it exits, and a caller that handed us a string is
    entitled to still have one.  A block that is already representable is
    promoted if it needs it and otherwise untouched, and an empty one is left
    alone: nothing to carry means nothing written.  The ODS is restored on the
    way out, including when the write raises, because the caller's object is
    not ours to change.

    Content is restored, not object identity: OMAS copies a `CodeParameters`
    on assignment, so no code can put the same object back through
    ``ods[path] = ...``.  A caller holding a reference taken before the write
    is holding a detached copy either way.
    """
    from omas import ODS
    from omas.omas_core import CodeParameters

    replaced: list[tuple[str, Any]] = []
    try:
        for path in _code_parameters_paths(ods):
            branch = ods[path]
            if not isinstance(branch, (ODS, CodeParameters)):
                # A string is already what the leaf holds.  It is recorded
                # anyway, because the decorator below this one parses it back
                # into a tree on its way out and the caller's ODS would keep
                # that.
                replaced.append((path, branch))
                continue
            leaves, omitted = _split(branch)
            if not leaves and not omitted:
                continue  # an empty node, often a materialized read: write nothing
            if not omitted and isinstance(branch, CodeParameters):
                continue  # nothing to do: it will serialize as it is
            payload = CodeParameters()
            payload.update(leaves)
            if omitted:
                payload[CACHE_OMITTED_KEY] = ", ".join(omitted)
                _logger.info(
                    "%s: %s stayed on the local product; an entry carries one "
                    "parameters string, and these are not representable as one",
                    path, ", ".join(omitted),
                )
            replaced.append((path, branch))
            ods[path] = payload
        yield ods
    finally:
        for path, branch in replaced:
            ods[path] = branch


def entry_safe_code_parameters(save):
    """Wrap a save function so it writes through :func:`as_entry_payload`."""

    @functools.wraps(save)
    def wrapper(ods, *args, **kwargs):
        with as_entry_payload(ods):
            return save(ods, *args, **kwargs)

    return wrapper
