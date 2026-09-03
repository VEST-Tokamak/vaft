"""Native IMAS access for the plotting recipes (issue #63).

The recipes in :mod:`vaft.plot.backend.recipes` are written in IMAS Data
Dictionary dotted paths (``magnetics.flux_loop.{i}.flux.data``).  imas-python
speaks the same vocabulary but not the same grammar: an ``IDSToplevel`` is
rooted below its IDS name, children are attributes, arrays of structures are
indexed, and an empty leaf holds a sentinel that only ``has_value`` exposes.
:class:`IDSEntry` bundles one shot's toplevels under their IDS names and
:data:`IDS_ACCESSOR` walks the path natively -- no conversion, no copy -- so
the same recipe reads an ODS and an IDS and the two view models can be
compared before anything is drawn.

The one exception is the set of code-backed recipes (``CallableRecipe``)
whose builders call functions written for an ODS: for those, and only those,
:meth:`IDSEntry.as_ods_for` converts the IDS the plot declares, through the
same fast walker :func:`vaft.imas.omas_imas.ods_from_toplevels` uses, and
the equivalence suite covers that path too.

Nothing here sets attributes on an imas-python class.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np

from vaft.plot.backend.access import register_accessor

__all__ = ["IDSEntry", "IDS_ACCESSOR", "is_native_ids", "is_native_entry"]

_MISSING = object()


def _module_root(obj: Any) -> str:
    return type(obj).__module__.partition(".")[0]


def is_native_ids(obj: Any) -> bool:
    """Whether ``obj`` is an imas-python ``IDSToplevel``."""
    return _module_root(obj) == "imas" and type(obj).__name__ == "IDSToplevel"


def is_native_entry(obj: Any) -> bool:
    """Whether ``obj`` is an imas-python ``DBEntry``."""
    return _module_root(obj) == "imas" and type(obj).__name__ == "DBEntry"


def _is_struct_array(node: Any) -> bool:
    return type(node).__name__ == "IDSStructArray"


def _is_structure(node: Any) -> bool:
    # IDSToplevel subclasses IDSStructure; both are containers of children.
    return any(cls.__name__ in ("IDSStructure",) for cls in type(node).__mro__)


def _is_primitive(node: Any) -> bool:
    return any(cls.__name__ == "IDSPrimitive" for cls in type(node).__mro__)


class IDSEntry:
    """One shot's native IDS objects, keyed by IDS name.

    Built from an ``IDSToplevel`` (one IDS), an ``imas.DBEntry``, a
    :class:`vaft.database._local.IMASHandle`, an HSDS lazy handle
    (:class:`vaft.database.lazy_imas.HSDSIMASHandle`), or a mapping of IDS
    name to toplevel.  Toplevels are fetched on first use and cached; an IDS
    the source does not hold reads as absent, never as an error.
    """

    def __init__(self, source: Any, *, occurrence: int = 0) -> None:
        self._occurrence = int(occurrence)
        self._toplevels: dict[str, Any] = {}
        self._absent: set[str] = set()
        self._ods_cache: dict[frozenset, Any] = {}
        self._getter = None
        self._full_getter = None
        self._full: dict[str, Any] = {}
        self._names: tuple[str, ...] | None = None
        if is_native_ids(source):
            name = str(source.metadata.name)
            self._toplevels[name] = source
            self._names = (name,)
        elif is_native_entry(source):
            self._getter = lambda name: source.get(name, self._occurrence, lazy=True)
            # A fully loaded copy for the code-backed plots' conversion: the
            # native walk reads lazily, but a conversion must see every node.
            self._full_getter = lambda name: source.get(name, self._occurrence, lazy=False)
            self._entry = source
        elif isinstance(source, Mapping):
            for name, toplevel in source.items():
                if not is_native_ids(toplevel):
                    raise TypeError(
                        f"IDSEntry mapping values must be IDSToplevel objects; got {type(toplevel).__name__} for {name!r}"
                    )
                self._toplevels[str(name)] = toplevel
            self._names = tuple(str(n) for n in source)
        elif hasattr(source, "get") and hasattr(source, "ids"):
            # vaft's own handles: IMASHandle (local files) and HSDSIMASHandle (lazy remote).
            self._getter = lambda name: source.get(name, self._occurrence)
            self._names = tuple(source.ids) if source.ids else None
            self._handle = source
        else:
            raise TypeError(
                "IDSEntry expects an imas IDSToplevel, a DBEntry, a vaft.imas.IMASHandle, an "
                f"HSDS lazy IMAS handle, or a mapping of IDS name to IDSToplevel; got {type(source).__name__}"
            )
        self.source = source

    # -- toplevels ------------------------------------------------------------
    def toplevel(self, name: str) -> Any:
        """The native toplevel for IDS ``name``, or ``None`` when the source lacks it."""
        if name in self._toplevels:
            return self._toplevels[name]
        if name in self._absent or self._getter is None:
            return None
        if self._names is not None and name not in self._names:
            self._absent.add(name)
            return None
        try:
            toplevel = self._getter(name)
        except Exception:
            self._absent.add(name)
            return None
        if toplevel is None:
            self._absent.add(name)
            return None
        self._toplevels[name] = toplevel
        return toplevel

    @property
    def ids_names(self) -> tuple[str, ...]:
        """IDS names this entry can serve without reading, else those read so far.

        A toplevel, a mapping and vaft's handles say up front; a bare
        ``DBEntry`` does not enumerate cheaply, so only the IDS fetched so far
        are listed for it.
        """
        return self._names if self._names is not None else tuple(self._toplevels)

    @property
    def pulse(self) -> Any:
        """The data entry's pulse number when the source carries one."""
        value = resolve(self, "dataset_description.data_entry.pulse")
        return None if value is _MISSING else value

    # -- the code-backed fallback --------------------------------------------
    def as_ods_for(self, ids_names: Iterable[str]) -> Any:
        """An OMAS ODS holding only ``ids_names``, converted from the native toplevels.

        Used solely for the code-backed recipes whose builders take an ODS.
        Cached per name set; IDS the source lacks are simply left out.
        """
        wanted = frozenset(str(n) for n in ids_names)
        if wanted in self._ods_cache:
            return self._ods_cache[wanted]
        from vaft.imas.omas_imas import ods_from_toplevels

        toplevels = {}
        for name in sorted(wanted):
            top = self._full_toplevel(name)
            if top is not None:
                toplevels[name] = top
        ods = ods_from_toplevels(toplevels)
        self._ods_cache[wanted] = ods
        return ods

    def _full_toplevel(self, name: str) -> Any:
        """A fully loaded toplevel for conversion, or ``None`` when absent."""
        if name in self._full:
            return self._full[name]
        top = self.toplevel(name)
        if top is None:
            return None
        if getattr(top, "_lazy", False):
            if self._full_getter is None:
                raise NotImplementedError(
                    f"plot needs a full copy of the {name!r} IDS to run its builder, and this "
                    "lazily loaded handle cannot supply one; load the shot eagerly "
                    "(vaft.database.load / vaft.imas.load) for the code-backed plots"
                )
            try:
                top = self._full_getter(name)
            except Exception:
                return None
        self._full[name] = top
        return top

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"IDSEntry({sorted(self._toplevels)}, source={type(self.source).__name__})"


# ---------------------------------------------------------------------------
# the native path walk
# ---------------------------------------------------------------------------


def resolve(entry: IDSEntry, path: str) -> Any:
    """The node or value at an IMAS-DD dotted ``path``, or ``_MISSING``.

    The first segment names the IDS and selects the toplevel; digits index an
    array of structures; every other segment is a child.  A primitive yields
    its value only when it holds one; a structure or array is present when
    any child is filled (the same rule OMAS applies to an empty branch).
    """
    segments = path.split(".")
    node = entry.toplevel(segments[0])
    if node is None:
        return _MISSING
    return _walk(node, segments[1:])


def _walk(node: Any, segments: list[str]) -> Any:
    for position, segment in enumerate(segments):
        if segment == ":":
            # OMAS's slice over an array of structures: the remainder of the
            # path gathered from every element, as an array when the leaves
            # are scalars (``pf_active.coil.0.element.:.turns_with_sign``).
            if not _is_struct_array(node):
                return _MISSING
            rest = segments[position + 1 :]
            values = [_walk(node[i], rest) for i in range(len(node))]
            values = [v for v in values if v is not _MISSING]
            if not values:
                return _MISSING
            try:
                return np.asarray(values)
            except (TypeError, ValueError):
                return values
        if segment.isdigit():
            if not _is_struct_array(node):
                return _MISSING
            index = int(segment)
            if index >= len(node):
                return _MISSING
            node = node[index]
            continue
        if not (_is_structure(node) or _is_struct_array(node)):
            return _MISSING
        try:
            node = getattr(node, segment)
        except AttributeError:
            return _MISSING
    return _leaf(node)


def _leaf(node: Any) -> Any:
    if _is_primitive(node):
        return node.value if node.has_value else _MISSING
    if _is_structure(node) or _is_struct_array(node):
        return node if _branch_present(node) else _MISSING
    return node


def _branch_present(node: Any) -> bool:
    """Whether a structure or array of structures holds anything.

    imas-python answers ``has_value`` for fully loaded branches; a lazily
    loaded branch refuses, and there the branch is taken as present and its
    leaves decide (each leaf answers ``has_value`` even when lazy), which is
    how the recipes read anyway.
    """
    try:
        return bool(node.has_value)
    except NotImplementedError:
        return True


class _IDSAccessor:
    """The backend accessor for :class:`IDSEntry` objects."""

    def get(self, obj: IDSEntry, path: str, default: Any = None) -> Any:
        value = resolve(obj, path)
        return default if value is _MISSING else value

    def has(self, obj: IDSEntry, path: str) -> bool:
        return resolve(obj, path) is not _MISSING

    def count(self, obj: IDSEntry, path: str) -> int:
        value = resolve(obj, path)
        if value is _MISSING or isinstance(value, (str, bytes)):
            return 0
        if _is_struct_array(value):
            return len(value)
        if _is_structure(value):
            return 0
        try:
            return len(value)
        except TypeError:
            return 0


IDS_ACCESSOR = _IDSAccessor()
register_accessor(lambda obj: isinstance(obj, IDSEntry), IDS_ACCESSOR)
