"""The magnetics status names every field the product lacks, and why (#2164).

``channel_status.magnetics.missing_channels`` used to be built from the raw
archive's field list alone. Since #2135 an equilibrium probe whose record is
too short for the zero-phase low pass is published with an empty ``field``
instead of failing the shot -- so it was archived, absent from the list, and
the component still said ``success``. These tests drive the real diagnostics
stage with synthetic raw dumps (the technique of
``test_efit_constraints_missing_channels.py``) and check the status separates
"never archived" from "archived but unusable" without touching any waveform.
"""

from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from vaft.machine_mapping.magnetics import (
    LIMITER_SHUNT_CHANNELS,
    TOROIDAL_MIRNOV_REFERENCE_CHANNELS,
    vest_equilibrium_magnetics_channel_definitions,
)
from vaft.machine_mapping.magnetics_registry import resolve_equilibrium_magnetics_sensors
from vaft.omas.vest_upstream import (
    MAGNETICS_CHANNEL_NOT_ARCHIVED,
    MAGNETICS_CHANNEL_UNUSABLE,
    _record_unusable_magnetics_channels,
    build_diagnostics_ods,
    build_static_ods,
    machine_era_for_shot,
    write_stage_product,
)

SHOT = 43016
SLOW_DT = 4e-5
N_SAMPLES = 10_000
#: Far below what the zero-phase probe low pass accepts (three filter lengths).
SHORT_RECORD = 12


def _records(kind: str):
    return [record for record in resolve_equilibrium_magnetics_sensors(SHOT) if record.kind == kind]


#: Picked through the registry, by identity: a mid-array probe and flux loop.
SHORT_PROBE = _records("b_field_pol_probe")[33]
EMPTY_PROBE = _records("b_field_pol_probe")[5]
UNARCHIVED_FLUX_LOOP = _records("flux_loop")[3]
EMPTY_SHUNT = LIMITER_SHUNT_CHANNELS[1]


def _fields(n_samples: int = N_SAMPLES) -> dict[str, dict]:
    codes = {int(d["field_code"]) for d in vest_equilibrium_magnetics_channel_definitions()}
    codes |= {int(c["field_code"]) for c in TOROIDAL_MIRNOV_REFERENCE_CHANNELS}
    codes |= {int(c["field_code"]) for c in LIMITER_SHUNT_CHANNELS}
    codes |= {1, 5, 25, 59, 62, 65, 109, 257}
    t = np.arange(n_samples) * SLOW_DT
    fields = {}
    for code in sorted(codes):
        if code == 109:
            data = 6.0e9 * np.clip(np.sin((t - 0.26) / 0.1 * np.pi), 0, None)
        else:
            data = np.sin(t * 10) * 0.01 + 0.02
        fields[str(code)] = {"data": data.tolist(), "type": "slow"}
    return fields


def _write(path: Path, fields: dict[str, dict]) -> Path:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump({"shot": SHOT, "fields": fields}, handle)
    return path


@pytest.fixture(scope="module")
def static_path(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("static")
    static, manifest = build_static_ods(machine_era_for_shot(SHOT).name)
    path = tmp / "static.json"
    write_stage_product(static, manifest, output=path, metadata=tmp / "static-manifest.json")
    return path


def _build(raw: Path, static: Path):
    return build_diagnostics_ods(
        shot=SHOT, raw_source=raw, static_ods=static, tstart=0.26, tend=0.36, dt=4e-5, run=1
    )


@pytest.fixture(scope="module")
def degraded(tmp_path_factory, static_path):
    """One dump with every failure kind at once: short, empty, unarchived.

    A raw-voltage channel (the shunt) has no low pass to be too short for, so
    only an empty record makes it unusable.
    """
    fields = _fields()
    fields[str(SHORT_PROBE.field_code)]["data"] = fields[str(SHORT_PROBE.field_code)]["data"][:SHORT_RECORD]
    fields[str(EMPTY_PROBE.field_code)]["data"] = []
    fields[str(EMPTY_SHUNT["field_code"])]["data"] = []
    del fields[str(UNARCHIVED_FLUX_LOOP.field_code)]
    raw = _write(tmp_path_factory.mktemp("degraded") / "raw.json.gz", fields)
    with pytest.warns(RuntimeWarning):
        return _build(raw, static_path)


def test_a_short_probe_record_is_a_missing_channel_not_a_success(degraded):
    ods, manifest = degraded
    status = manifest["channel_status"]["magnetics"]
    field = SHORT_PROBE.field_code

    # The waveform itself is what #2135 made it: published, empty, in place.
    assert len(ods[f"magnetics.b_field_pol_probe.{SHORT_PROBE.kind_index}.field.data"]) == 0
    assert ods[f"magnetics.b_field_pol_probe.{SHORT_PROBE.kind_index}.identifier"] == SHORT_PROBE.name

    assert status["status"] == "partial"
    assert field in status["missing_channels"]
    assert status["missing_channel_reasons"][str(field)] == MAGNETICS_CHANNEL_UNUSABLE
    assert status["unusable_channels"][str(field)] == (
        f"magnetics.b_field_pol_probe.{SHORT_PROBE.kind_index}.field.data"
    )
    assert f"magnetics:field-{field}" in manifest["quality_summary"]["missing"]


def test_reasons_separate_never_archived_from_archived_but_unusable(degraded):
    _, manifest = degraded
    reasons = manifest["channel_status"]["magnetics"]["missing_channel_reasons"]

    assert reasons[str(UNARCHIVED_FLUX_LOOP.field_code)] == MAGNETICS_CHANNEL_NOT_ARCHIVED
    assert reasons[str(EMPTY_PROBE.field_code)] == MAGNETICS_CHANNEL_UNUSABLE
    assert reasons[str(EMPTY_SHUNT["field_code"])] == MAGNETICS_CHANNEL_UNUSABLE
    # Every listed field has exactly one reason, and nothing has a reason it is not listed for.
    assert sorted(int(field) for field in reasons) == manifest["channel_status"]["magnetics"]["missing_channels"]


def test_well_formed_channels_are_not_reported(degraded):
    ods, manifest = degraded
    status = manifest["channel_status"]["magnetics"]
    reported = set(status["missing_channels"])
    for record in _records("b_field_pol_probe"):
        node = f"magnetics.b_field_pol_probe.{record.kind_index}.field.data"
        assert (len(ods[node]) == 0) == (record.field_code in reported), record.name


def test_a_clean_dump_reports_nothing_unusable(tmp_path, static_path):
    _, manifest = _build(_write(tmp_path / "raw.json.gz", _fields()), static_path)
    status = manifest["channel_status"]["magnetics"]

    assert "unusable_channels" not in status
    assert MAGNETICS_CHANNEL_UNUSABLE not in status["missing_channel_reasons"].values()
    assert set(status["missing_channel_reasons"].values()) <= {MAGNETICS_CHANNEL_NOT_ARCHIVED}


def test_the_status_pass_changes_no_leaf_of_the_product(degraded):
    ods, manifest = degraded
    before = copy.deepcopy(ods)
    status = {"status": "success", "missing_channels": []}
    archived = {int(code) for code in _fields()} - {UNARCHIVED_FLUX_LOOP.field_code}

    _record_unusable_magnetics_channels(status, ods, SHOT, archived)

    assert status["status"] == "partial"
    assert set(ods.flat()) == set(before.flat())
    for path, value in before.flat().items():
        np.testing.assert_array_equal(np.asarray(ods[path]), np.asarray(value), err_msg=path)


def test_an_unavailable_component_keeps_its_archive_reasons():
    from omas import ODS

    status = {"status": "unavailable", "missing_channels": [218, 216]}
    _record_unusable_magnetics_channels(status, ODS(consistency_check=False), SHOT, set())

    assert status["status"] == "unavailable"
    assert status["missing_channels"] == [218, 216]
    assert status["missing_channel_reasons"] == {"216": "not_archived", "218": "not_archived"}
