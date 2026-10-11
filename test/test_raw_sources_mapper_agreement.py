"""raw_sources() reports the fields the mappers actually load (#1742).

Runs each real mapper on a recording stub of ``vaft.database.raw.vest_load``
that returns a synthetic waveform for any field, at one shot per wiring /
calibration era, and requires the set of fields loaded to equal the union of
``raw_sources`` over the registry ids that mapper serves. The comparison is
against what the mappers do, not against a second table of fields.

About ten minutes (the magnetics mapper dominates), so not in the develop
core gate; test_raw_sources.py carries the cheap contract there.
"""

from __future__ import annotations

import importlib
import inspect
from collections import defaultdict

import numpy as np
import pytest
from omas import ODS

import vaft.database.raw as raw_db
from vaft.machine_mapping import magnetics as magnetics_module
from vaft.machine_mapping.raw_daq_sources import raw_sources
from vaft.machine_mapping.registry import load_diagnostic_registry

#: One shot inside each era that changes a field choice: PF9 on field 64
#: (<=19283), the borrowed diamagnetic field 4 (38110-38569), PF2 on field 4
#: (>=38570), the 39437/39438 wiring swap, no diamagnetic sensor (42000),
#: the upper Langmuir probe (>=42675), the fluctuation Mirnovs (>=44156),
#: FL10 compensation off (>=47117), and the toroidal references' end.
ERA_SHOTS = (19000, 30000, 38200, 38600, 39437, 39438, 42000, 42700, 44156, 47200, 48224)

#: `disruption.halo_currents` is `not_implemented` in the registry, with no
#: `mapping`, but its fields are mapped today, by magnetics (magnetics.shunt).
_MAPPER_OVERRIDES = {"disruption.halo_currents": ("vaft.machine_mapping.magnetics", "magnetics")}


def _mapper_groups() -> dict[tuple[str, str], list[str]]:
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for identifier, record in load_diagnostic_registry().items():
        if not record["source"].get("raw_fields"):
            continue
        mapping = _MAPPER_OVERRIDES.get(identifier)
        if mapping is None:
            if not record.get("mapping"):
                continue  # dataset_description-like metadata entries
            mapping = (record["mapping"]["module"], record["mapping"]["entrypoint"])
        groups[mapping].append(identifier)
    return dict(groups)


def _synthetic(shot, field, *args, **kwargs):
    time = np.arange(0.0, 0.5, 4e-5)
    rng = np.random.default_rng(int(field))
    data = 0.01 * rng.standard_normal(time.size) + 0.5 * np.exp(-(((time - 0.31) / 0.01) ** 2))
    return time, data


def _loaded_fields(monkeypatch, module_name: str, entrypoint: str, shot: int) -> set[int]:
    calls: list[tuple[int, int]] = []

    def recording(shot_, field, *args, **kwargs):
        calls.append((int(shot_), int(field)))
        return _synthetic(shot_, field)

    monkeypatch.setattr(raw_db, "vest_load", recording)
    magnetics_module._safe_vest_load_cached.cache_clear()
    function = getattr(importlib.import_module(module_name), entrypoint)
    parameters = inspect.signature(function).parameters
    third = list(parameters)[2]
    window = (0.26, 0.36, 4e-5) if parameters[third].default is inspect.Parameter.empty else ()
    function(ODS(consistency_check=False), shot, *window)
    magnetics_module._safe_vest_load_cached.cache_clear()
    assert {loaded_shot for loaded_shot, _ in calls} <= {shot}, "a mapper read another shot"
    return {field for _, field in calls}


GROUPS = _mapper_groups()


@pytest.mark.parametrize("shot", ERA_SHOTS)
@pytest.mark.parametrize("mapping", sorted(GROUPS), ids=lambda m: m[1])
def test_raw_sources_equal_the_fields_the_mapper_loads(monkeypatch, mapping, shot):
    expected = {
        record.field for identifier in GROUPS[mapping] for record in raw_sources(identifier, shot)
    }
    assert _loaded_fields(monkeypatch, *mapping, shot) == expected


