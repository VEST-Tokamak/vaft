"""Optional deployment catalog of IDS instances (#1129, generic half).

A catalog is *policy*, never the source of stored meaning (#1128): the name
of a stored occurrence is its own ``ids_properties.name``. What a catalog adds
is what an arbitrary Data Entry cannot say about itself --

- the stable occurrence a deployment allocates to each semantic name;
- which name is the default per IDS (or an explicit "no default");
- aliases for terminology migration;
- the lineage a deployment expects, for validation.

A disagreement between catalog and payload is reported, never reconciled by
reinterpreting the payload. VAFT core ships no concrete catalog; the VEST one
belongs to the VEST deployment layer.

Input format (YAML or a mapping), one block per IDS::

    equilibrium:
      default: magnetic-efit        # or null: deliberately no default
      instances:
        - occurrence: 0
          name: magnetic-efit
          description: Canonical magnetic EFIT reconstruction.
        - occurrence: 1
          name: magnetic-efit-chease
          description: CHEASE-refined equilibrium from magnetic EFIT.
          input: {ids: equilibrium, name: magnetic-efit}
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from ._types import CatalogError, InstanceInfo, InstanceKey

_KEBAB = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_ENTRY_FIELDS = {"occurrence", "name", "description", "input", "occurrence_type", "aliases"}
_BLOCK_FIELDS = {"default", "instances"}


@dataclass(frozen=True)
class CatalogEntry:
    """One registered instance: stable occurrence, name, meaning, lineage."""

    ids: str
    occurrence: int
    name: str
    description: str
    inputs: tuple[InstanceKey, ...] = ()
    occurrence_type: str | None = None
    aliases: tuple[str, ...] = ()

    @property
    def key(self) -> InstanceKey:
        return InstanceKey(self.ids, self.name)


@dataclass(frozen=True)
class Finding:
    """One validation result; ``level`` is ``"error"`` or ``"warning"``."""

    level: str
    ids: str
    message: str

    def __str__(self) -> str:
        return f"[{self.level}] {self.ids}: {self.message}"


@dataclass(frozen=True)
class CatalogUpdate:
    """Classification of a catalog change (#1129 §update classification)."""

    compatible: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)


def _as_inputs(ids: str, name: str, raw: Any) -> tuple[InstanceKey, ...]:
    if raw is None:
        return ()
    items = [raw] if isinstance(raw, Mapping) else list(raw)
    out = []
    for item in items:
        if not isinstance(item, Mapping) or set(item) != {"ids", "name"}:
            raise CatalogError(
                f"{ids}/{name}: input must be {{ids, name}} (or a list of them), got {item!r}"
            )
        out.append(InstanceKey(str(item["ids"]), str(item["name"])))
    return tuple(out)


class InstanceCatalog:
    """Validated, immutable deployment policy over IDS instances."""

    def __init__(
        self,
        entries: Iterable[CatalogEntry],
        defaults: Mapping[str, str | None] | None = None,
    ):
        by_ids: dict[str, list[CatalogEntry]] = {}
        for entry in entries:
            by_ids.setdefault(entry.ids, []).append(entry)
        self._entries = {
            ids: tuple(sorted(items, key=lambda e: e.occurrence))
            for ids, items in sorted(by_ids.items())
        }
        self._defaults = dict(defaults or {})
        self._validate()
        self._aliases = {
            (entry.ids, alias): entry.name
            for items in self._entries.values()
            for entry in items
            for alias in entry.aliases
        }

    # -- construction -------------------------------------------------------

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "InstanceCatalog":
        if not isinstance(data, Mapping):
            raise CatalogError(f"a catalog is a mapping of IDS blocks, got {type(data).__name__}")
        entries: list[CatalogEntry] = []
        defaults: dict[str, str | None] = {}
        for ids, block in data.items():
            if not isinstance(block, Mapping):
                raise CatalogError(f"{ids}: block must be a mapping")
            unknown = set(block) - _BLOCK_FIELDS
            if unknown:
                raise CatalogError(f"{ids}: unknown block fields {sorted(unknown)}")
            if "default" in block:
                default = block["default"]
                defaults[ids] = None if default is None else str(default)
            raw = block.get("instances", [])
            if isinstance(raw, Mapping) or not isinstance(raw, (list, tuple)):
                raise CatalogError(
                    f"{ids}: instances must be a list of records (not a name-keyed mapping)"
                )
            for record in raw:
                if not isinstance(record, Mapping):
                    raise CatalogError(f"{ids}: instance record must be a mapping, got {record!r}")
                unknown = set(record) - _ENTRY_FIELDS
                if unknown:
                    raise CatalogError(f"{ids}: unknown instance fields {sorted(unknown)}")
                missing = {"occurrence", "name", "description"} - set(record)
                if missing:
                    raise CatalogError(f"{ids}: instance {record!r} is missing {sorted(missing)}")
                occurrence = record["occurrence"]
                if isinstance(occurrence, bool) or not isinstance(occurrence, int) or occurrence < 0:
                    raise CatalogError(f"{ids}: occurrence must be an integer >= 0, got {occurrence!r}")
                name = str(record["name"])
                description = str(record["description"]).strip()
                if not description:
                    raise CatalogError(f"{ids}/{name}: description must not be empty")
                entries.append(
                    CatalogEntry(
                        ids=str(ids),
                        occurrence=occurrence,
                        name=name,
                        description=description,
                        inputs=_as_inputs(str(ids), name, record.get("input")),
                        occurrence_type=(
                            None if record.get("occurrence_type") is None
                            else str(record["occurrence_type"])
                        ),
                        aliases=tuple(str(a) for a in record.get("aliases", ()) or ()),
                    )
                )
            if ids not in defaults and raw:
                # #1129: a default is recorded explicitly, or its absence is.
                raise CatalogError(
                    f"{ids}: declare `default: <name>` or `default: null` explicitly"
                )
        return cls(entries, defaults)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "InstanceCatalog":
        import yaml

        with open(path, encoding="utf-8") as handle:
            return cls.from_mapping(yaml.safe_load(handle) or {})

    def _validate(self) -> None:
        import imas

        factory = imas.IDSFactory()
        names_by_ids: dict[str, set[str]] = {}
        for ids, items in self._entries.items():
            if not factory.exists(ids):
                raise CatalogError(f"{ids!r} is not an IDS of the Data Dictionary")
            occurrences = [e.occurrence for e in items]
            duplicate = sorted({o for o in occurrences if occurrences.count(o) > 1})
            if duplicate:
                raise CatalogError(f"{ids}: occurrence(s) {duplicate} assigned twice")
            seen: set[str] = set()
            for entry in items:
                for label in (entry.name, *entry.aliases):
                    if not _KEBAB.match(label):
                        raise CatalogError(f"{ids}: {label!r} is not lowercase kebab-case")
                    if label in seen:
                        raise CatalogError(f"{ids}: name or alias {label!r} used twice")
                    seen.add(label)
            names_by_ids[ids] = {e.name for e in items}
        for ids, default in self._defaults.items():
            if not factory.exists(ids):
                raise CatalogError(f"default declared for unknown IDS {ids!r}")
            if default is not None and default not in names_by_ids.get(ids, set()):
                raise CatalogError(
                    f"{ids}: default {default!r} is not a registered name "
                    f"({sorted(names_by_ids.get(ids, ()))})"
                )
        for items in self._entries.values():
            for entry in items:
                for ref in entry.inputs:
                    if ref.name not in names_by_ids.get(ref.ids, set()):
                        raise CatalogError(
                            f"{entry.key}: input {ref} is not a registered instance"
                        )

    # -- queries ------------------------------------------------------------

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(set(self._entries) | set(self._defaults)))

    def entries(self, ids: str) -> tuple[CatalogEntry, ...]:
        return self._entries.get(ids, ())

    def canonical_name(self, ids: str, name: str) -> str:
        """Resolve an alias to its canonical name; other names pass through."""
        return self._aliases.get((ids, name), name)

    def entry(self, ids: str, name: str) -> CatalogEntry | None:
        name = self.canonical_name(ids, name)
        return next((e for e in self.entries(ids) if e.name == name), None)

    def entry_at(self, ids: str, occurrence: int) -> CatalogEntry | None:
        return next((e for e in self.entries(ids) if e.occurrence == occurrence), None)

    def declares_default(self, ids: str) -> bool:
        """Whether the catalog states a default policy (possibly "none")."""
        return ids in self._defaults

    def default(self, ids: str) -> str | None:
        return self._defaults.get(ids)

    @property
    def defaults(self) -> Mapping[str, str | None]:
        return MappingProxyType(self._defaults)

    # -- validation against stored metadata ---------------------------------

    def check(self, infos: Iterable[InstanceInfo]) -> tuple[Finding, ...]:
        """Compare discovered occurrences with the expected allocation.

        Errors: a stored name at the wrong occurrence, or a registered
        occurrence carrying another name. Warnings: a stored name the catalog
        does not register (valid IMAS, just unmanaged). Unnamed occurrences
        are not findings -- the catalog never supplies names.

        Allocation is checked only where a name came from the payload. A
        binding-named occurrence lives in migration-era storage, where every
        variant sits at occurrence 0 of its own entry: the binding *is* its
        allocation until consolidation writes it to the catalog's occurrence.
        """
        findings: list[Finding] = []
        for info in infos:
            if info.name is None or info.ids not in self._entries:
                continue
            if info.name_source == "binding":
                if self.entry(info.ids, info.name) is None:
                    findings.append(Finding(
                        "warning", info.ids,
                        f"storage binding name {info.name!r} ({info.store}) is not in the catalog",
                    ))
                continue
            expected = self.entry(info.ids, info.name)
            at_slot = self.entry_at(info.ids, info.occurrence)
            if expected is not None and expected.occurrence != info.occurrence:
                findings.append(Finding(
                    "error", info.ids,
                    f"stored {info.name!r} is at occurrence {info.occurrence} "
                    f"({info.store}); the catalog allocates occurrence {expected.occurrence}",
                ))
            elif expected is None and at_slot is not None:
                findings.append(Finding(
                    "error", info.ids,
                    f"occurrence {info.occurrence} ({info.store}) stores {info.name!r}; "
                    f"the catalog allocates it to {at_slot.name!r}",
                ))
            elif expected is None:
                findings.append(Finding(
                    "warning", info.ids,
                    f"stored name {info.name!r} (occurrence {info.occurrence}) is not in the catalog",
                ))
            elif expected.name != info.name:
                findings.append(Finding(
                    "warning", info.ids,
                    f"stored name {info.name!r} is an alias of {expected.name!r}; "
                    "the payload should carry the canonical name",
                ))
        return tuple(findings)

    # -- identity -----------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Canonical, JSON-serialisable content plus its sha256."""
        body = {
            ids: {
                "default": self._defaults.get(ids),
                "declares_default": ids in self._defaults,
                "instances": [
                    {
                        "occurrence": e.occurrence,
                        "name": e.name,
                        "description": e.description,
                        "input": [{"ids": k.ids, "name": k.name} for k in e.inputs],
                        "occurrence_type": e.occurrence_type,
                        "aliases": list(e.aliases),
                    }
                    for e in self.entries(ids)
                ],
            }
            for ids in self.ids()
        }
        digest = hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return {"catalog": body, "sha256": digest}

    @staticmethod
    def classify_update(old: "InstanceCatalog", new: "InstanceCatalog") -> CatalogUpdate:
        """Whether ``new`` is a compatible (append-only) successor of ``old``.

        Incompatible: a published name removed or moved to another occurrence,
        or an occurrence reused for another name. Adding instances, aliases or
        descriptions is compatible; changing a default is compatible but
        reported, since ordinary access changes meaning.
        """
        reasons: list[str] = []
        compatible = True
        for ids in old.ids():
            for entry in old.entries(ids):
                moved = new.entry(ids, entry.name)
                if moved is None:
                    compatible = False
                    reasons.append(f"{entry.key} (occurrence {entry.occurrence}) was removed")
                elif moved.occurrence != entry.occurrence:
                    compatible = False
                    reasons.append(
                        f"{entry.key} moved from occurrence {entry.occurrence} to {moved.occurrence}"
                    )
                reused = new.entry_at(ids, entry.occurrence)
                if reused is not None and reused.name != entry.name:
                    compatible = False
                    reasons.append(
                        f"{ids} occurrence {entry.occurrence} reassigned from "
                        f"{entry.name!r} to {reused.name!r}"
                    )
            if old.default(ids) != new.default(ids):
                reasons.append(
                    f"{ids} default changed from {old.default(ids)!r} to {new.default(ids)!r}"
                )
        return CatalogUpdate(compatible, tuple(reasons))

    def __repr__(self) -> str:
        count = sum(len(v) for v in self._entries.values())
        return f"<InstanceCatalog {count} instances over {len(self.ids())} IDS>"


__all__ = ["CatalogEntry", "CatalogUpdate", "Finding", "InstanceCatalog"]
