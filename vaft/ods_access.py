"""Non-mutating dotted-path access for ODS objects and plain mappings (issue #118).

An OMAS ``ODS`` creates paths on access.  ``ods["missing.path"]`` does not
raise: it materializes an empty placeholder at that path, leaves it there, and
returns it.  Nothing about that is visible where you would look for it --
``ods.flat()`` never reflects it -- and the damage surfaces only much later, on
the next ``consistency_check=True`` load::

    ValueError: magnetics.b_field_pol_probe.0.field.data shape () is
    inconsistent with coordinates: ['magnetics.b_field_pol_probe[:].field.time']

or, for an integer leaf such as a materialized ``validity``::

    ValueError: cannot convert float NaN to integer

So a *probe* corrupts the object it was only supposed to inspect.  That has bitten
this repository repeatedly and independently -- in EFIT constraint generation, in
the magnetics signal-quality validator, and in the plasma-free vacuum benchmark --
each time producing another local workaround.  This module is the one shared
answer, so there is no reason for a tenth.

The mechanism is not new: OMAS already provides non-mutating primitives, and VAFT
simply was not using them.  Measured on omas 0.94.2 / IMAS 3.41.0, across missing
leaves, missing arrays of structures, missing containers, missing IDSs and missing
nested structures:

======================  ==========  ================================================
``ods[path]``           mutates     returns an empty ``ODS`` whose ``str()`` is the path
``path in ods``         safe        correct boolean, and works on a sub-ODS
``ods.get(path, d)``    safe        ``d`` when absent, the value when present
``ods.flat()``          --          **never** reflects a materialized path
======================  ==========  ================================================

Which to use
------------
The distinction is whether absence is a valid state, not whether a read might
fail:

``path_exists`` / ``path_value`` / ``path_count``
    Optional access.  Absence is an expected outcome and is reported, never
    created.  A missing raw signal, an unmapped diagnostic, an IDS this shot
    does not have.

:func:`get_path`
    Required access.  The path is part of the caller's contract, so absence is
    a programming error and raises ``KeyError`` -- loudly, and still without
    creating anything.

:func:`set_path`
    The write primitive.  Vivification is correct here and lives only here.

States kept distinct
--------------------
Absence is one state among several, and collapsing them would trade one silent
bug for another.  ``path_exists`` reports ``False`` for a missing path *and* for
an empty ``STRUCTURE``/``STRUCT_ARRAY`` -- an empty branch carries no datum, and
the Data Dictionary's own convention treats it that way.  It reports ``True`` for
a deliberately empty array (``np.array([])``, which the VEST mappers write for an
unwired channel) and for a present-but-NaN scalar: those are populated nodes
making a statement, not absent ones.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "get_path",
    "path_count",
    "path_exists",
    "path_value",
    "set_path",
]

#: What "this path does not resolve" looks like coming out of OMAS.  The type
#: depends on the accessor, not on the situation: descending through a scalar
#: leaf -- ``equilibrium.code.parameters.<...>`` when ``parameters`` holds a JSON
#: string, say -- raises ``TypeError`` from a subscript but ``AttributeError``
#: from ``in`` and ``.get``, because those try to reach ``omas_data`` on a
#: ``str``.  All of them mean the same thing here: there is nothing at that path.
_ABSENT = (AttributeError, IndexError, KeyError, LookupError, TypeError, ValueError)
def _set_nested_mapping_value(mapping: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    current: Any = mapping

    for index, part in enumerate(parts):
        is_last = index == len(parts) - 1
        next_is_list = not is_last and parts[index + 1].isdigit()

        if isinstance(current, dict):
            if is_last:
                current[part] = value
                return
            next_value = current.get(part)
            if next_is_list:
                if not isinstance(next_value, list):
                    next_value = []
                    current[part] = next_value
            else:
                if not isinstance(next_value, dict):
                    next_value = {}
                    current[part] = next_value
            current = next_value
            continue

        if isinstance(current, list):
            slot = int(part)
            while len(current) <= slot:
                current.append(None)
            if is_last:
                current[slot] = value
                return
            next_value = current[slot]
            if next_is_list:
                if not isinstance(next_value, list):
                    next_value = []
                    current[slot] = next_value
            else:
                if not isinstance(next_value, dict):
                    next_value = {}
                    current[slot] = next_value
            current = next_value
            continue

        raise TypeError(f"Cannot set nested value on non-container at {part!r} in {path!r}")


def _get_nested_mapping_value(mapping: dict[str, Any], path: str) -> Any:
    current: Any = mapping
    for part in path.split("."):
        if isinstance(current, dict):
            current = current[part]
            continue
        if isinstance(current, list):
            current = current[int(part)]
            continue
        raise KeyError(path)
    return current




# ---------------------------------------------------------------------------
# Optional access: absence is a valid answer, never a side effect
# ---------------------------------------------------------------------------

#: Returned by :func:`_resolve` for a path that carries nothing.  A sentinel
#: rather than ``None``, because ``None`` is a value a populated node can hold.
_MISSING = object()


def _resolve(source: Any, path: str) -> Any:
    """The value at ``path``, or :data:`_MISSING`, without creating anything.

    The single resolver behind every accessor here, so "does it exist" and
    "what is it" cannot disagree.

    ``path in source`` is the primary test because it is the one OMAS accessor
    that never vivifies.  It is not universal, though: it reaches for
    ``omas_data`` on whatever the path descends through, so it raises when part
    of the path is not a dynamic ODS node -- inside a ``CodeParameters``
    subtree, or through a leaf holding a JSON string.  It raises for *present*
    paths there too, so treating that as absence would silently drop real data
    (EFIT's ``code.parameters.time_slice.N.aeqdsk.*`` metrics, for one).

    The subscript fallback is therefore only reached where membership is
    unsupported -- and every node kind in that situation is a static container
    that raises instead of vivifying: ``CodeParameters`` raises ``KeyError`` for
    a missing path and leaves nothing behind, a string leaf raises
    ``TypeError``. Both are verified in ``test/test_ods_access.py`` so an OMAS
    change cannot quietly reintroduce the very side effect this module exists to
    prevent.
    """
    if type(source).__module__.partition(".")[0] == "imas":
        # A native IMAS node does not answer dotted paths; it would raise
        # ValueError, which the absence net below would swallow, and every
        # read would come back "absent" while the data sits there.
        raise TypeError(
            f"vaft.ods_access reads OMAS ODS/mapping paths; a native IMAS "
            f"{type(source).__name__} is read by vaft.imas (plot it with vaft.imas.plot_*)"
        )
    if isinstance(source, dict):
        try:
            return _get_nested_mapping_value(source, path)
        except _ABSENT:
            return _MISSING
    try:
        present = path in source
    except _ABSENT:
        try:
            return source[path]
        except _ABSENT:
            return _MISSING
    if not present:
        return _MISSING
    return source[path]


def path_exists(source: Any, path: str) -> bool:
    """Whether a dotted path resolves to actual content, without creating it.

    On an OMAS ODS this rests on ``path in source``, which is non-mutating and
    already treats an empty ``STRUCTURE``/``STRUCT_ARRAY`` branch as absent --
    the semantics the previous implementation had to reconstruct after reading,
    at the cost of materializing whatever it looked at.

    An empty branch counting as non-existent is deliberate and load-bearing:
    dead b-probe channels (the 48xxx campaign's probes 65-68, stored with
    ``field: null``) once sailed through the EFIT constraints filter on a check
    that reported every path as present, and crashed input generation with
    ``float * ODS``.
    """
    return _resolve(source, path) is not _MISSING


def path_value(source: Any, path: str, default: Any = None) -> Any:
    """The value at ``path``, or ``default`` when it carries no content.

    The optional-read counterpart to :func:`get_path`.  Equivalent to::

        if path not in source:
            return default
        return source[path]

    and it inherits :func:`path_exists`'s states exactly, so an empty ODS branch
    reads as ``default`` while a deliberately empty array or a NaN scalar comes
    back as itself.  Callers that must tell "absent" from "present but empty"
    should pass a sentinel rather than rely on ``None``, which a populated node
    can legitimately hold.
    """
    value = _resolve(source, path)
    return default if value is _MISSING else value


def path_count(source: Any, path: str) -> int:
    """Length of the container at ``path``, ``0`` when it is absent or empty.

    The shape that recurred in six near-identical private helpers across this
    repository, four of which materialized the container they were counting.
    A missing array of structures and an empty one both count ``0`` -- neither
    has an entry to iterate -- and so does a node that is not a container of
    entries at all, scalars and strings alike, since a caller asking "how many
    entries" of one is asking about something that has none.
    """
    value = path_value(source, path)
    # A string is sized but is not a container of entries: counting one would
    # return its character count, which reads as a plausible number and is not
    # the question the caller asked.
    if value is None or isinstance(value, (str, bytes)):
        return 0
    try:
        return len(value)
    except TypeError:
        return 0


# ---------------------------------------------------------------------------
# Required access, and the one place vivification belongs
# ---------------------------------------------------------------------------

def get_path(source: Any, path: str) -> Any:
    """Read a dotted path that the caller's contract requires to be present.

    Raises ``KeyError`` when it is not -- and, unlike a bare ``source[path]``,
    without materializing anything on the way out.  A required path that is
    missing is a programming error, and the useful behaviour is to say so at the
    point of the mistake rather than to hand back an empty placeholder that
    fails a consistency check three stages downstream.

    Use :func:`path_value` where absence is a legitimate outcome.
    """
    value = _resolve(source, path)
    if value is _MISSING:
        raise KeyError(path)
    return value


def set_path(source: Any, path: str, value: Any) -> None:
    """Write a dotted path into either a plain dict or an OMAS ODS object.

    The only primitive here that creates structure, which is what a write is
    for.  Anything reaching for a read in order to build a node -- growing an
    array of structures by indexing past its end, for instance -- is doing
    something this module deliberately does not offer, and should say so at the
    call site (see ``vaft.machine_mapping.mhd_linear._ensure_time_slice``).
    """
    if isinstance(source, dict):
        _set_nested_mapping_value(source, path, value)
        return
    source[path] = value
