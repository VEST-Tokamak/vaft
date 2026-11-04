"""VEST machine history that is not one diagnostic's processing configuration.

The ``machine_history`` block of ``vest.yaml`` (issue #2267) holds the shot
boundaries that used to be Python literals:

* the static machine eras -- passive-wall and PF-coil geometry versions, and
  the ``machine_version`` name stage products are keyed by;
* the fast-DAQ trigger offset that places a fast record on the discharge clock;
* the waveform-table generation used when the database cannot be asked;
* the 455xx SXR campaign that may inherit the HXR trigger start.

Every consumer resolves through this module, so a boundary is written once.
The machine eras are read from the packaged ``vest.yaml`` only: they name
product identity, which a campaign configuration must not change.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
import os
from typing import Any

from .utils import (
    VestConfigurationError,
    _cached_document,
    _resolve_info_file_path,
    _shot_number,
    resolve_shot_revisions,
)

MACHINE_HISTORY_KEY = "machine_history"

#: The geometry versions a machine era may name; each has packaged assets.
PF_GEOMETRY_VERSIONS = frozenset({"1906", "2507"})
WALL_GEOMETRY_VERSIONS = frozenset({"1512", "2409"})

_ERA_KEYS = frozenset({"name", "from_shot", "to_shot", "pf_geometry", "wall_geometry", "reference_shot"})


@dataclass(frozen=True)
class VestMachineEra:
    """One static VEST machine era: the geometry a range of shots ran with."""

    name: str
    first_shot: int | None
    last_shot: int | None
    pf_geometry: str
    reference_shot: int
    wall_geometry: str = "1512"

    def contains(self, shot: int) -> bool:
        return (self.first_shot is None or shot >= self.first_shot) and (
            self.last_shot is None or shot <= self.last_shot
        )


def _stat_key(info_file: str | os.PathLike[str] | None) -> tuple[str, int, int]:
    path = _resolve_info_file_path(info_file)
    stat = os.stat(path)
    return path, stat.st_mtime_ns, stat.st_size


def _history(key: tuple[str, int, int]) -> Mapping[str, Any]:
    """The shared, cached ``machine_history`` block. Callers must not mutate it."""
    block = _cached_document(*key).get(MACHINE_HISTORY_KEY)
    if not isinstance(block, Mapping):
        raise VestConfigurationError(
            f"{key[0]}: no {MACHINE_HISTORY_KEY!r} mapping (issue #2267)"
        )
    return block


def _section(block: Mapping[str, Any], *path: str) -> Any:
    node: Any = block
    walked = [MACHINE_HISTORY_KEY]
    for name in path:
        walked.append(name)
        if not isinstance(node, Mapping) or name not in node:
            raise VestConfigurationError(f"vest.yaml has no {'.'.join(walked)}")
        node = node[name]
    return node


def _version(value: Any, allowed: frozenset[str], *, name: str, context: str) -> str:
    text = str(value) if isinstance(value, (str, int)) and not isinstance(value, bool) else None
    if text not in allowed:
        raise VestConfigurationError(
            f"{context}: {name} must be one of {', '.join(sorted(allowed))}, not {value!r}"
        )
    return text


@lru_cache(maxsize=4)
def _machine_eras(key: tuple[str, int, int]) -> tuple[VestMachineEra, ...]:
    context = f"{MACHINE_HISTORY_KEY}.machine_eras"
    entries = _section(_history(key), "machine_eras")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)) or not entries:
        raise VestConfigurationError(f"{context} must be a non-empty list")
    eras: list[VestMachineEra] = []
    for index, entry in enumerate(entries):
        where = f"{context}[{index}]"
        if not isinstance(entry, Mapping):
            raise VestConfigurationError(f"{where} must be a mapping")
        unknown = sorted(str(k) for k in entry if k not in _ERA_KEYS)
        if unknown:
            raise VestConfigurationError(f"{where}: unknown key(s) {', '.join(unknown)}")
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise VestConfigurationError(f"{where}: name must be a non-empty string")
        first = entry.get("from_shot")
        last = entry.get("to_shot")
        first = None if first is None else _shot_number(first, name="from_shot", context=where)
        last = None if last is None else _shot_number(last, name="to_shot", context=where)
        if "reference_shot" not in entry:
            raise VestConfigurationError(f"{where}: reference_shot is required")
        era = VestMachineEra(
            name=name,
            first_shot=first,
            last_shot=last,
            pf_geometry=_version(entry.get("pf_geometry"), PF_GEOMETRY_VERSIONS, name="pf_geometry", context=where),
            reference_shot=_shot_number(entry["reference_shot"], name="reference_shot", context=where),
            wall_geometry=_version(
                entry.get("wall_geometry"), WALL_GEOMETRY_VERSIONS, name="wall_geometry", context=where
            ),
        )
        if not era.contains(era.reference_shot):
            raise VestConfigurationError(f"{where}: reference_shot {era.reference_shot} lies outside the era")
        eras.append(era)

    names = [era.name for era in eras]
    if len(set(names)) != len(names):
        raise VestConfigurationError(f"{context}: era names must be unique")
    # The eras tile every shot, in order: open below, open above, no gap, no overlap.
    if eras[0].first_shot is not None:
        raise VestConfigurationError(f"{context}[0]: the first era must have no from_shot")
    if eras[-1].last_shot is not None:
        raise VestConfigurationError(f"{context}[{len(eras) - 1}]: the last era must have no to_shot")
    for index, (before, after) in enumerate(zip(eras, eras[1:]), 1):
        if before.last_shot is None or after.first_shot is None or after.first_shot != before.last_shot + 1:
            raise VestConfigurationError(
                f"{context}[{index}]: from_shot must be the previous era's to_shot + 1 "
                f"(eras must tile every shot without gap or overlap)"
            )
        if after.first_shot > (after.last_shot if after.last_shot is not None else after.first_shot):
            raise VestConfigurationError(f"{context}[{index}]: from_shot must not exceed to_shot")
    return tuple(eras)


def machine_eras() -> tuple[VestMachineEra, ...]:
    """The static VEST machine eras, in shot order (packaged ``vest.yaml`` only)."""
    return _machine_eras(_stat_key(None))


def machine_era_for_shot(shot: int) -> VestMachineEra:
    """The machine era *shot* belongs to; every non-negative shot has exactly one."""
    number = _shot_number(shot, name="shot", context="machine_era_for_shot")
    for era in machine_eras():
        if era.contains(number):
            return era
    raise ValueError(f"No VEST machine era is defined for shot {shot}")  # pragma: no cover - eras tile


def machine_era(name: str) -> VestMachineEra:
    """Resolve a machine-era name (a static product's ``machine_version``)."""
    eras = machine_eras()
    for era in eras:
        if era.name == name:
            return era
    choices = ", ".join(item.name for item in eras)
    raise ValueError(f"Unknown VEST machine era {name!r}; expected one of: {choices}")


def first_shot_of(attribute: str, version: str) -> int:
    """The first shot of the first era whose *attribute* equals *version*.

    Backs the compatibility constants ``PF_GEOMETRY_2507_FIRST_SHOT`` and
    ``WALL_GEOMETRY_2409_FIRST_SHOT``.
    """
    for era in machine_eras():
        if getattr(era, attribute) == version:
            if era.first_shot is None:
                return 0
            return era.first_shot
    raise ValueError(f"No VEST machine era has {attribute} {version!r}")


@lru_cache(maxsize=4096)
def _resolved_daq(key: tuple[str, int, int], name: str, shot: int) -> Mapping[str, Any]:
    block = _section(_history(key), "daq", name)
    if not isinstance(block, Mapping):
        raise VestConfigurationError(f"{MACHINE_HISTORY_KEY}.daq.{name} must be a mapping")
    base = {k: v for k, v in block.items() if k != "revisions"}
    return resolve_shot_revisions(
        base, block.get("revisions"), shot, context=f"{MACHINE_HISTORY_KEY}.daq.{name}"
    )


def fast_trigger_offset_s(shot: int, *, info_file: str | os.PathLike[str] | None = None) -> float:
    """The fast-DAQ trigger offset [s] added to a fast record's own time axis."""
    number = _shot_number(shot, name="shot", context="fast_trigger_offset_s")
    value = _resolved_daq(_stat_key(info_file), "fast_trigger_offset", number).get("offset_s")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VestConfigurationError(
            f"{MACHINE_HISTORY_KEY}.daq.fast_trigger_offset.offset_s must be a number, not {value!r}"
        )
    return float(value)


def fallback_waveform_generation(shot: int, *, info_file: str | os.PathLike[str] | None = None) -> int:
    """The waveform-table generation assumed for *shot* when the database cannot be asked."""
    number = _shot_number(shot, name="shot", context="fallback_waveform_generation")
    value = _resolved_daq(_stat_key(info_file), "waveform_table_fallback", number).get("generation")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise VestConfigurationError(
            f"{MACHINE_HISTORY_KEY}.daq.waveform_table_fallback.generation must be a positive integer, "
            f"not {value!r}"
        )
    return value


def last_shot_of_waveform_generation(generation: int) -> int:
    """The last shot the packaged fallback assigns to *generation* (``V2_LAST_SHOT`` for 2)."""
    block = _section(_history(_stat_key(None)), "daq", "waveform_table_fallback")
    for revision in block.get("revisions") or ():
        if revision.get("generation") == generation and revision.get("to_shot") is not None:
            return _shot_number(revision["to_shot"], name="to_shot", context="waveform_table_fallback")
    raise ValueError(f"The waveform-table fallback gives generation {generation} no last shot")


@lru_cache(maxsize=4)
def _hxr_fallback_range(key: tuple[str, int, int]) -> range:
    context = f"{MACHINE_HISTORY_KEY}.soft_x_rays.hxr_trigger_fallback"
    block = _section(_history(key), "soft_x_rays", "hxr_trigger_fallback")
    if not isinstance(block, Mapping) or set(block) != {"from_shot", "to_shot"}:
        raise VestConfigurationError(f"{context} must be a mapping of exactly from_shot and to_shot")
    first = _shot_number(block["from_shot"], name="from_shot", context=context)
    last = _shot_number(block["to_shot"], name="to_shot", context=context)
    if first > last:
        raise VestConfigurationError(f"{context}: from_shot must not exceed to_shot")
    return range(first, last + 1)


def sxr_hxr_trigger_fallback_shots(*, info_file: str | os.PathLike[str] | None = None) -> range:
    """The shots whose SXR record may inherit the HXR trigger start (inclusive bounds)."""
    return _hxr_fallback_range(_stat_key(info_file))


__all__ = [
    "MACHINE_HISTORY_KEY",
    "PF_GEOMETRY_VERSIONS",
    "VestMachineEra",
    "WALL_GEOMETRY_VERSIONS",
    "fallback_waveform_generation",
    "fast_trigger_offset_s",
    "first_shot_of",
    "last_shot_of_waveform_generation",
    "machine_era",
    "machine_era_for_shot",
    "machine_eras",
    "sxr_hxr_trigger_fallback_shots",
]
