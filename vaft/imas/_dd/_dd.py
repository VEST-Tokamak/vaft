"""``DD`` and ``DDView``: one logical IMAS Data Entry and one selected state.

Terminology. In this module **DD is a Data Entry aggregate**, after FUSE's
IMASdd.jl (abstract ``DD``, root object ``dd``). Everywhere else in IMAS and
VAFT "DD" also means the *Data Dictionary* -- ``dd_version``, a "DD path", the
``dd_*`` plotting functions. The class does not redefine that; the IMAS Data
Dictionary stays the schema, and a DD holds native IDS objects of it.

The model (#1127, #1128):

``DD``
    every stored occurrence of every IDS in one logical Data Entry, which may
    span several physical stores during migration. An occurrence's semantic
    name is its own ``ids_properties.name``; unnamed occurrences stay numeric.
``DDView``
    an immutable selection of at most one occurrence per IDS -- each IDS may
    sit at a different occurrence -- sharing the DD's live objects. It is what
    an IMASdd.jl ``dd`` is: one IDS instance per name.

``dd.equilibrium`` resolves the *default* instance. A default comes only from
policy -- ``defaults=`` or a deployment catalog -- never from "occurrence 0
exists", and a missing default is reported, never substituted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from ._bindings import StorageBindings
from ._catalog import Finding, InstanceCatalog
from ._provenance import UpstreamRef, format_reference, upstream
from ._store import EntryStore, MutableEntryStore, StoreSet, stored_names
from ._types import (
    AmbiguousInstanceError,
    CatalogMismatchError,
    CoherenceError,
    InstanceInfo,
    InstanceKey,
    InstanceLookupError,
    InstanceUnavailableError,
    NoDefaultError,
    PhysicalLocator,
    UnknownInstanceError,
    describe_instances,
)

Selector = str | int


def _is_selector(value: Any) -> bool:
    return isinstance(value, str) or (isinstance(value, int) and not isinstance(value, bool))


# ---------------------------------------------------------------------------
# Coherence reporting


@dataclass(frozen=True)
class Dependency:
    """One declared upstream requirement of a selected instance.

    ``status`` is ``"ok"`` (the view selects the required upstream),
    ``"mismatch"`` (it selects a different instance of that IDS),
    ``"missing"`` (the Data Entry does not hold the upstream at all) or
    ``"unselected"`` (it exists but the view selects no instance of that IDS,
    for want of a default).
    """

    instance: InstanceInfo
    requires: str
    source: str
    status: str
    selected: InstanceInfo | None = None

    def __str__(self) -> str:
        chosen = "" if self.selected is None else f"; selected {self.selected}"
        return (
            f"{self.instance} requires {self.requires} ({self.source}): "
            f"{self.status}{chosen}"
        )


@dataclass(frozen=True)
class CoherenceReport:
    """Structured result of :meth:`DDView.validate` (#1132 §14)."""

    selected: Mapping[str, InstanceInfo]
    dependencies: tuple[Dependency, ...]
    scope: tuple[str, ...] | None = None
    #: ``{ids: policy}`` for defaults the policy names but the entry lacks.
    #: The selection leaves those IDS out rather than substitute, and the
    #: state is not coherent: ordinary access to them would fail.
    unavailable_defaults: Mapping[str, Selector] = MappingProxyType({})

    @property
    def mismatches(self) -> tuple[Dependency, ...]:
        return tuple(d for d in self.dependencies if d.status == "mismatch")

    @property
    def missing(self) -> tuple[Dependency, ...]:
        return tuple(d for d in self.dependencies if d.status == "missing")

    @property
    def coherent(self) -> bool:
        return not self.mismatches and not self.missing and not self.unavailable_defaults

    def __bool__(self) -> bool:
        return self.coherent

    def __str__(self) -> str:
        head = "coherent" if self.coherent else "NOT coherent"
        lines = [f"{head}: {len(self.selected)} selected IDS"]
        lines += [
            f"  default {ids} instance {policy!r} is not stored"
            for ids, policy in self.unavailable_defaults.items()
        ]
        lines += [f"  {d}" for d in self.dependencies if d.status != "ok"]
        return "\n".join(lines)


@dataclass(frozen=True)
class ValidationReport:
    """Result of :meth:`DD.validate`: catalog findings plus default coherence."""

    findings: tuple[Finding, ...]
    coherence: CoherenceReport | None

    @property
    def errors(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.level == "error")

    @property
    def ok(self) -> bool:
        return not self.errors and self.coherence is not None and self.coherence.coherent

    def __str__(self) -> str:
        lines = [str(f) for f in self.findings]
        if self.coherence is None:
            lines.append("default-view coherence not evaluated: catalog errors come first")
        else:
            lines.append(str(self.coherence))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# DD


class DD:
    """One logical IMAS Data Entry with every stored IDS occurrence.

    Parameters
    ----------
    stores
        One :class:`EntryStore` (registered as ``"main"``), a mapping of named
        stores, or a :class:`StoreSet`. The DD owns them: :meth:`close`
        closes them.
    catalog
        Optional :class:`InstanceCatalog`: stable allocation, defaults,
        aliases and validation. Never a source of names.
    defaults
        Optional ``{ids: name | occurrence | None}`` default policy; takes
        precedence over the catalog's. ``None`` means "deliberately none".
    bindings
        Optional migration-era :class:`StorageBindings` naming unnamed legacy
        occurrences.
    lazy
        Return imas-python lazy (read-only, read-on-access) toplevels instead
        of fully loaded ones. Lazy objects stop reading once the DD is closed.
    dd_version
        The one Data Dictionary version the DD presents; required when its
        stores disagree. Conversion happens as an IDS is loaded.
    """

    def __init__(
        self,
        stores: StoreSet | EntryStore | Mapping[str, EntryStore],
        *,
        catalog: InstanceCatalog | None = None,
        defaults: Mapping[str, Selector | None] | None = None,
        bindings: StorageBindings | None = None,
        lazy: bool = False,
        dd_version: str | None = None,
    ):
        self._stores = StoreSet.of(stores)
        try:
            self._setup(catalog, defaults, bindings, lazy, dd_version)
        except BaseException:
            # The DD owns its stores from the first line: a refused
            # construction must not leave a DBEntry (or a converted source's
            # scratch tree) open behind it.
            self._stores.close()
            raise

    def _setup(self, catalog, defaults, bindings, lazy, dd_version) -> None:
        import imas

        versions = {store.dd_version for store in self._stores.values()}
        if dd_version is None:
            if len(versions) != 1:
                raise ValueError(
                    f"stores hold different DD versions {sorted(versions)}; pass dd_version="
                )
            dd_version = versions.pop()
        self._dd_version = str(dd_version)
        self._factory = imas.IDSFactory(self._dd_version)
        self._catalog = catalog
        self._bindings = bindings or StorageBindings.single_entry()
        self._lazy = bool(lazy)
        self._defaults = dict(defaults or {})
        for ids, value in self._defaults.items():
            self._require_ids(ids)
            if value is not None and not _is_selector(value):
                raise TypeError(f"default for {ids} must be a name, an occurrence or None")
        self._cache: dict[PhysicalLocator, Any] = {}
        self._index: dict[str, tuple[InstanceInfo, ...]] = {}
        self._checked: set[str] = set()
        self._closed = False

    @classmethod
    def open(
        cls,
        source: str | Path | Sequence[str | Path],
        *,
        catalog: InstanceCatalog | None = None,
        defaults: Mapping[str, Selector | None] | None = None,
        imas_version: str | None = None,
        dd_version: str | None = None,
        lazy: bool = False,
    ) -> "DD":
        """Open one local artifact (anything :func:`vaft.imas.load` reads).

        Works on arbitrary IMAS HDF5/netCDF entries with no VAFT metadata:
        named occurrences are discovered, unnamed ones stay numeric.

        ``dd_version`` is the version the DD presents; the entry itself is
        read at the version it was written with and converted as each IDS
        loads, which is what lets DD-4 data be presented as DD 3 (imas-python
        refuses to open an entry across a major version). ``imas_version`` is
        only for a source whose version cannot be detected.
        """
        from ._imas_store import IMASStore

        store = IMASStore(source, imas_version=imas_version)
        return cls(
            store,
            catalog=catalog,
            defaults=defaults,
            lazy=lazy,
            dd_version=dd_version,
        )

    # -- basic properties ---------------------------------------------------

    @property
    def dd_version(self) -> str:
        """The Data Dictionary version of the IDS objects this DD returns."""
        return self._dd_version

    @property
    def stores(self) -> StoreSet:
        return self._stores

    @property
    def catalog(self) -> InstanceCatalog | None:
        return self._catalog

    @property
    def closed(self) -> bool:
        return self._closed

    def _is_ids(self, name: str) -> bool:
        return self._factory.exists(name)

    def _require_ids(self, name: str) -> None:
        if not isinstance(name, str) or not self._is_ids(name):
            raise ValueError(f"{name!r} is not an IDS of Data Dictionary {self._dd_version}")

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("this DD has been closed; already-loaded IDS objects remain usable")

    # -- discovery ----------------------------------------------------------

    def available_ids(self) -> tuple[str, ...]:
        """IDS names with at least one stored occurrence, in any store."""
        self._require_open()
        names: set[str] = set()
        for store in self._stores.values():
            names.update(store.available_ids())
        return tuple(sorted(name for name in names if self._is_ids(name)))

    def loaded_ids(self) -> tuple[str, ...]:
        """IDS names with at least one occurrence already loaded."""
        return tuple(sorted({locator.ids for locator in self._cache}))

    def instances(self, ids: str) -> tuple[InstanceInfo, ...]:
        """Every stored occurrence of ``ids``, named or not, in store order."""
        self._require_ids(ids)
        cached = self._index.get(ids)
        if cached is not None:
            return cached
        self._require_open()
        infos: list[InstanceInfo] = []
        for store_name, store in self._stores.items():
            occurrences = store.occurrences(ids)
            if not occurrences:
                continue
            names = stored_names(store, ids)
            for occurrence in occurrences:
                locator = PhysicalLocator(store_name, ids, occurrence)
                payload = names.get(occurrence, "").strip() or None
                bound = self._bindings.name_for(locator)
                if payload is not None and bound is not None and payload != bound:
                    raise CatalogMismatchError(
                        f"{locator} stores ids_properties.name={payload!r} but the "
                        f"storage binding names it {bound!r}"
                    )
                name = payload or bound
                source = "payload" if payload else ("binding" if bound else None)
                infos.append(InstanceInfo(ids, locator, name, source))
        result = tuple(infos)
        self._index[ids] = result
        return result

    def _checked_instances(self, ids: str) -> tuple[InstanceInfo, ...]:
        infos = self.instances(ids)
        if self._catalog is not None and ids not in self._checked:
            errors = [f for f in self._catalog.check(infos) if f.level == "error"]
            if errors:
                raise CatalogMismatchError(
                    "stored occurrence metadata disagrees with the catalog: "
                    + "; ".join(f.message for f in errors)
                )
            self._checked.add(ids)
        return infos

    def default(self, ids: str) -> Selector | None:
        """The default-selection policy for ``ids`` (a name, an occurrence, or None).

        This reports policy; whether the default is present in this entry is
        what :meth:`get` / :meth:`has` answer.
        """
        self._require_ids(ids)
        if ids in self._defaults:
            return self._defaults[ids]
        if self._catalog is not None:
            return self._catalog.default(ids)
        return None

    def _canonical(self, ids: str, name: str | None) -> str | None:
        """A name with catalog aliases folded, so a payload still carrying an
        old alias and a selector using the new name meet."""
        if name is None or self._catalog is None:
            return name
        return self._catalog.canonical_name(ids, name)

    def _registered(self, ids: str, name: str) -> bool:
        if self._catalog is not None and self._catalog.entry(ids, name) is not None:
            return True
        return InstanceKey(ids, name) in set(self._bindings.keys(ids))

    def resolve(self, ids: str, instance: Selector | None = None) -> InstanceInfo:
        """Resolve a name, an explicit occurrence, or the default, to one instance."""
        self._require_ids(ids)
        infos = self._checked_instances(ids)
        if instance is None:
            policy = self.default(ids)
            if policy is None:
                raise NoDefaultError(
                    f"{ids} has no default-selection policy here, so bare access cannot "
                    f"choose; stored: {describe_instances(infos)}. Select one with "
                    f"dd.view({ids}=<name or occurrence>) or pass defaults={{{ids!r}: ...}}."
                )
            try:
                return self._match(ids, infos, policy)
            except UnknownInstanceError:
                raise InstanceUnavailableError(
                    f"the default {ids} instance {policy!r} is not stored in this entry "
                    f"(stored: {describe_instances(infos)}); no other instance is substituted"
                ) from None
        if not _is_selector(instance):
            raise TypeError(f"instance must be a name (str) or an occurrence (int), got {instance!r}")
        return self._match(ids, infos, instance)

    def _match(self, ids: str, infos: tuple[InstanceInfo, ...], selector: Selector) -> InstanceInfo:
        if isinstance(selector, str):
            name = self._canonical(ids, selector)
            found = [info for info in infos if self._canonical(ids, info.name) == name]
            label = repr(selector)
        else:
            found = [info for info in infos if info.occurrence == selector]
            label = f"occurrence {selector}"
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            raise AmbiguousInstanceError(
                f"{ids} {label} matches {len(found)} stored occurrences "
                f"({describe_instances(found)}); select by a unique name"
            )
        if isinstance(selector, str) and self._registered(ids, name):
            raise InstanceUnavailableError(
                f"{ids} instance {selector!r} is registered but not stored in this entry; "
                f"stored: {describe_instances(infos)}"
            )
        raise UnknownInstanceError(
            f"no {ids} instance {label}; stored: {describe_instances(infos)}"
        )

    def has(self, ids: str, instance: Selector | None = None) -> bool:
        """Whether ``ids`` is stored (any occurrence), or resolves with ``instance``."""
        if instance is None:
            return bool(self.instances(ids))
        try:
            self.resolve(ids, instance)
        except AmbiguousInstanceError:
            return True
        except InstanceLookupError:
            return False
        return True

    # -- loading ------------------------------------------------------------

    def _load(self, info: InstanceInfo) -> Any:
        locator = info.locator
        cached = self._cache.get(locator)
        if cached is not None:
            return cached
        self._require_open()
        store = self._stores[locator.store]
        value = store.get(locator.ids, locator.occurrence, lazy=self._lazy)
        if store.dd_version != self._dd_version:
            if self._lazy:
                raise ValueError(
                    f"store {locator.store!r} holds DD {store.dd_version}; lazy access "
                    f"cannot convert to {self._dd_version}"
                )
            import imas

            value = imas.convert_ids(value, self._dd_version)
        self._cache[locator] = value
        return value

    def get(self, ids: str, instance: Selector | None = None) -> Any:
        """The native IDS toplevel of one instance (the default when omitted)."""
        return self._load(self.resolve(ids, instance))

    def __getitem__(self, ids: str) -> Any:
        return self.get(ids)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        factory = self.__dict__.get("_factory")
        if factory is not None and factory.exists(name):
            return self.get(name)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __dir__(self) -> list[str]:
        extra = [] if self._closed else list(self.available_ids())
        return sorted(set(super().__dir__()) | set(extra))

    # -- views --------------------------------------------------------------

    def view(self, **selection: Selector) -> "DDView":
        """A DDView selecting the given instances; unlisted IDS use defaults.

        Values are semantic names (``str``) or explicit occurrences (``int``,
        the way to reach an unnamed occurrence). Each is resolved now, so an
        unknown or unavailable selection fails here, not at first access.
        """
        return DDView(self, {ids: self.resolve(ids, value) for ids, value in selection.items()})

    def default_view(self) -> "DDView":
        """The deployment's ordinary scientific state; must be coherent."""
        view = DDView(self, {})
        report = view.validate()
        if not report.coherent:
            raise CoherenceError(f"the default view is not coherent:\n{report}")
        return view

    def validate(self) -> ValidationReport:
        """Catalog agreement for every stored IDS plus default-view coherence."""
        findings: list[Finding] = []
        if self._catalog is not None:
            for ids in self.available_ids():
                findings.extend(self._catalog.check(self.instances(ids)))
        if any(f.level == "error" for f in findings):
            # Resolving defaults over a mismatched IDS raises by design; the
            # findings already say what is wrong.
            return ValidationReport(tuple(findings), None)
        return ValidationReport(tuple(findings), DDView(self, {}).validate())

    # -- persistence --------------------------------------------------------

    def save_instance(
        self,
        ids: str,
        name: str | None = None,
        *,
        data: Any = None,
        occurrence: int | None = None,
        store: str | None = None,
    ) -> InstanceInfo:
        """Persist one instance explicitly; nothing is ever saved implicitly.

        The target is the stored instance ``name`` if there is one, otherwise
        the catalog's allocation for ``name``, otherwise ``occurrence``.
        ``data`` defaults to the live object of an already loaded instance.
        ``ids_properties.name`` is written (or checked) on the payload, so the
        stored occurrence describes itself; sibling occurrences are untouched.
        """
        self._require_open()
        self._require_ids(ids)
        name = self._canonical(ids, name)
        existing: InstanceInfo | None = None
        if name is not None:
            found = [
                info for info in self.instances(ids)
                if self._canonical(ids, info.name) == name
            ]
            if len(found) > 1:
                raise AmbiguousInstanceError(f"{ids} {name!r} is stored more than once")
            existing = found[0] if found else None
        if existing is not None:
            locator = existing.locator
            if occurrence is not None and occurrence != locator.occurrence:
                raise CatalogMismatchError(
                    f"{ids} {name!r} is stored at {locator}, not occurrence {occurrence}"
                )
            if store is not None and store != locator.store:
                raise CatalogMismatchError(f"{ids} {name!r} is stored in {locator.store!r}")
        else:
            entry = None if name is None or self._catalog is None else self._catalog.entry(ids, name)
            if entry is not None:
                if occurrence is not None and occurrence != entry.occurrence:
                    raise CatalogMismatchError(
                        f"the catalog allocates {ids} {name!r} to occurrence "
                        f"{entry.occurrence}, not {occurrence}"
                    )
                occurrence = entry.occurrence
            if occurrence is None:
                raise ValueError(
                    f"{ids} {name!r} is neither stored nor allocated by a catalog; pass occurrence="
                )
            store = store or self._target_store(ids, int(occurrence))
            locator = PhysicalLocator(store, ids, int(occurrence))
            clash = next((i for i in self.instances(ids) if i.locator == locator), None)
            # Naming an unnamed occurrence is the backfill #1128 asks for;
            # overwriting a differently named one is never implied.
            if (
                clash is not None
                and clash.name is not None
                and self._canonical(ids, clash.name) != name
            ):
                raise CatalogMismatchError(
                    f"{locator} already stores {clash.name!r}; refusing to "
                    f"overwrite it with {name!r}"
                )
        # Everything that can refuse is checked before anything is mutated, so
        # a refused save never leaves the shared live object renamed.
        target = self._stores[locator.store]
        if not isinstance(target, MutableEntryStore) or not target.writable:
            raise PermissionError(f"store {locator.store!r} is not writable")
        if data is None:
            data = self._cache.get(locator)
            if data is None:
                raise ValueError(f"nothing loaded for {locator}; pass data=")
        if self._lazy and data is self._cache.get(locator):
            raise ValueError("lazy IDS objects are read-only; open the DD with lazy=False to save")
        if name is not None:
            stored = str(data.ids_properties.name).strip()
            if stored and self._canonical(ids, stored) != name:
                raise CatalogMismatchError(
                    f"the payload's ids_properties.name is {stored!r}, not {name!r}"
                )
            data.ids_properties.name = name
        target.put(data, locator.occurrence)
        if data._dd_version != self._dd_version:
            import imas

            data = imas.convert_ids(data, self._dd_version)
        self._cache[locator] = data
        self._index.pop(ids, None)
        self._checked.discard(ids)
        return next(info for info in self.instances(ids) if info.locator == locator)

    def _target_store(self, ids: str, occurrence: int) -> str:
        """The store to write ``ids`` occurrence ``occurrence`` to.

        The one store already holding that occurrence (naming it in place),
        else the one writable store; anything else needs ``store=``.
        """
        holding = [name for name, s in self._stores.items() if occurrence in s.occurrences(ids)]
        if len(holding) == 1:
            return holding[0]
        writable = [
            name for name, s in self._stores.items()
            if isinstance(s, MutableEntryStore) and s.writable
        ]
        if len(writable) == 1:
            return writable[0]
        if not writable:
            raise PermissionError("this DD has no writable store")
        raise ValueError(f"choose a store= among the writable stores {writable}")

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        """Close the stores this DD owns. Loaded (non-lazy) objects stay usable."""
        if not self._closed:
            self._closed = True
            self._stores.close()

    def __enter__(self) -> "DD":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        state = " closed" if self._closed else ""
        return (
            f"<DD{state} DD-version {self._dd_version}, stores {list(self._stores)}, "
            f"{len(self._cache)} loaded>"
        )


# ---------------------------------------------------------------------------
# DDView


class DDView:
    """An immutable selection of at most one instance per IDS.

    A view holds only its DD and sparse overrides; everything else resolves
    through the DD's defaults, and loaded objects are the DD's shared ones --
    two views selecting the same occurrence return the same object.
    """

    __slots__ = ("_dd", "_overrides")

    def __init__(self, dd: DD, overrides: Mapping[str, InstanceInfo]):
        object.__setattr__(self, "_dd", dd)
        object.__setattr__(self, "_overrides", MappingProxyType(dict(overrides)))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("a DDView is immutable; use view(...) to select differently")

    @property
    def dd(self) -> DD:
        return self._dd

    @property
    def overrides(self) -> Mapping[str, InstanceInfo]:
        """The explicit selections, beyond the DD's defaults."""
        return self._overrides

    def view(self, **selection: Selector) -> "DDView":
        """A new view with further selections; this one is unchanged."""
        merged = dict(self._overrides)
        merged.update({ids: self._dd.resolve(ids, value) for ids, value in selection.items()})
        return DDView(self._dd, merged)

    def instance(self, ids: str) -> InstanceInfo:
        """The instance this view selects for ``ids``."""
        chosen = self._overrides.get(ids)
        return chosen if chosen is not None else self._dd.resolve(ids)

    def get(self, ids: str) -> Any:
        return self._dd._load(self.instance(ids))

    def __getitem__(self, ids: str) -> Any:
        return self.get(ids)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if self._dd._is_ids(name):
            return self.get(name)
        raise AttributeError(f"'DDView' object has no attribute {name!r}")

    def has(self, ids: str) -> bool:
        try:
            self.instance(ids)
        except InstanceLookupError:
            return False
        return True

    def selection(self) -> dict[str, InstanceInfo]:
        """``{ids: instance}`` for every IDS this view resolves.

        Overrides, plus every stored IDS whose default resolves; an IDS with no
        default (or an absent one) is simply not part of the selection.
        """
        return self._select()[0]

    def _select(self) -> tuple[dict[str, InstanceInfo], dict[str, Selector]]:
        chosen = dict(self._overrides)
        unavailable: dict[str, Selector] = {}
        for ids in self._dd.available_ids():
            if ids in chosen:
                continue
            try:
                chosen[ids] = self._dd.resolve(ids)
            except NoDefaultError:
                continue
            except InstanceUnavailableError:
                unavailable[ids] = self._dd.default(ids)
        return dict(sorted(chosen.items())), unavailable

    def locators(self) -> dict[str, PhysicalLocator]:
        return {ids: info.locator for ids, info in self.selection().items()}

    def occurrences(self) -> dict[str, int]:
        """Per-IDS occurrence map of the selection -- never one entry-wide integer."""
        return {ids: info.occurrence for ids, info in self.selection().items()}

    def validate(self, ids: Iterable[str] | None = None) -> CoherenceReport:
        """Check declared lineage of the selected instances (#1132 §14).

        Requirements come from each instance's stored provenance first, then
        from the catalog's ``input``. ``ids`` limits the check to instances in
        that scope (what a projection of those IDS needs). Selection itself is
        permissive: an incoherent view is valid to build and to inspect.
        """
        selection, unavailable = self._select()
        scope = None if ids is None else tuple(ids)
        if scope is not None:
            unavailable = {k: v for k, v in unavailable.items() if k in scope}
        dependencies: list[Dependency] = []
        for dep_ids, info in selection.items():
            if scope is not None and dep_ids not in scope:
                continue
            for ref, source in self._requirements(info):
                dependencies.append(self._judge(info, ref, source, selection))
        return CoherenceReport(
            MappingProxyType(selection),
            tuple(dependencies),
            scope,
            MappingProxyType(unavailable),
        )

    def _requirements(self, info: InstanceInfo) -> list[tuple[UpstreamRef, str]]:
        dd = self._dd
        store = dd._stores[info.store]
        refs: list[tuple[UpstreamRef, str]] = []
        seen: set[str] = set()
        payload = store.get(info.ids, info.occurrence, lazy=True)
        for ref in upstream(payload):
            if str(ref) not in seen:
                seen.add(str(ref))
                refs.append((ref, "provenance"))
        catalog = dd._catalog
        entry = None if catalog is None or info.name is None else catalog.entry(info.ids, info.name)
        for key in entry.inputs if entry is not None else ():
            ref = UpstreamRef(key.ids, name=key.name)
            if str(ref) not in seen:
                seen.add(str(ref))
                refs.append((ref, "catalog"))
        return refs

    def _judge(
        self,
        info: InstanceInfo,
        ref: UpstreamRef,
        source: str,
        selection: Mapping[str, InstanceInfo],
    ) -> Dependency:
        dd = self._dd
        label = format_reference(ref.ids, name=ref.name, occurrence=ref.occurrence)

        def satisfies(candidate: InstanceInfo) -> bool:
            if ref.name is not None:
                return dd._canonical(ref.ids, candidate.name) == dd._canonical(ref.ids, ref.name)
            # "<ids>:<n>" is store-local: occurrence n of the same *physical*
            # entry the dependent is stored in. Across the stores of a
            # migration-era DD every variant is occurrence 0, so a bare number
            # could not name one of them; only a name can.
            return candidate.locator == PhysicalLocator(info.store, ref.ids, ref.occurrence)

        chosen = selection.get(ref.ids)
        if chosen is not None:
            status = "ok" if satisfies(chosen) else "mismatch"
            return Dependency(info, label, source, status, chosen)
        stored = dd.instances(ref.ids) if dd._is_ids(ref.ids) else ()
        status = "unselected" if any(satisfies(c) for c in stored) else "missing"
        return Dependency(info, label, source, status)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, DDView)
            and other._dd is self._dd
            and dict(other._overrides) == dict(self._overrides)
        )

    def __hash__(self) -> int:
        return hash((id(self._dd), frozenset(self._overrides.items())))

    def __repr__(self) -> str:
        chosen = ", ".join(
            f"{ids}={info.name if info.name is not None else info.occurrence!r}"
            for ids, info in sorted(self._overrides.items())
        )
        return f"<DDView {chosen or 'defaults'}>"


__all__ = ["CoherenceReport", "DD", "DDView", "Dependency", "ValidationReport"]
