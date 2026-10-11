"""Contract of vaft.machine_mapping.raw_sources (#1742).

Cheap checks of the record shape, the era boundaries the resolvers follow and
the registry join. That the resolvers agree with what the mappers actually
load is test_raw_sources_mapper_agreement.py.
"""

from __future__ import annotations

import pytest

from vaft.machine_mapping.raw_daq_sources import RawField, raw_field_name, raw_sources
from vaft.machine_mapping.registry import load_diagnostic_registry

#: One shot in each era that changes a field choice (see the agreement test).
ERA_SHOTS = (19000, 38200, 42000, 44156, 48224)


def test_every_implemented_raw_daq_entry_states_a_resolver():
    registry = load_diagnostic_registry()
    missing = [
        identifier
        for identifier, record in registry.items()
        if record["source"]["type"] == "raw_daq" and not record["source"].get("raw_fields")
    ]
    assert missing == []


def test_every_resolver_returns_records_for_every_era():
    registry = load_diagnostic_registry()
    for identifier, record in registry.items():
        if not record["source"].get("raw_fields"):
            continue
        for shot in ERA_SHOTS:
            records = raw_sources(identifier, shot)
            assert all(isinstance(item, RawField) for item in records)
            assert all(item.label and item.raw_unit and item.role for item in records), identifier


def test_era_boundaries_follow_the_mapper_resolvers():
    pf = {record.channel: record.field for record in raw_sources("pf_active", 19283)}
    assert pf["pf_active.coil[8].current"] == 64
    assert {record.channel: record.field for record in raw_sources("pf_active", 19287)}[
        "pf_active.coil[8].current"
    ] == 65
    # The diamagnetic Rogowski borrowed PF2's field 4: no PF circuit reads it then (#993).
    assert 4 not in {record.field for record in raw_sources("pf_active", 38200)}
    assert [record.field for record in raw_sources("magnetics.rogowski_coil", 38200)] == [109, 4]
    # FL10 compensation is off from 47117, the carbon inboard limiter.
    assert [record.role for record in raw_sources("magnetics.ip", 47116)] == ["signal", "fl10_reference"]
    assert [record.role for record in raw_sources("magnetics.ip", 47117)] == ["signal"]
    # No usable diamagnetic sensor: nothing is read for the flux.
    assert raw_sources("magnetics.diamagnetic_flux", 42000) == ()
    langmuir = {record.field for record in raw_sources("langmuir_probes.embedded", 42674)}
    assert langmuir == {99, 100}
    assert {259, 260} <= {record.field for record in raw_sources("langmuir_probes.embedded", 42675)}


def test_recorded_faults_stay_sources_and_carry_their_reason():
    probes = raw_sources("magnetics.b_field_pol_probe", 48224)
    c4_04 = probes[45]
    assert c4_04.channel == "magnetics.b_field_pol_probe[45]"
    assert c4_04.known_fault and "C4-04" in c4_04.known_fault
    assert probes[44].known_fault is None


def test_ip_record_carries_the_calibration_the_mapper_applies():
    from vaft.machine_mapping.utils import resolve_vest_diagnostic

    (signal,) = raw_sources("magnetics.ip", 48224)
    config = resolve_vest_diagnostic(48224, "plasma_current")
    assert signal.field == config["source"]["field"]
    assert dict(signal.calibration) == dict(config["calibration"])
    assert signal.raw_unit == "V"


def test_unknown_and_non_raw_ids_are_refused():
    with pytest.raises(KeyError):
        raw_sources("no.such.diagnostic", 48224)
    with pytest.raises(ValueError, match="raw_fields"):
        raw_sources("thomson_scattering", 48224)


def test_offline_names():
    assert raw_field_name(109) == "Rogowski Coil-Inner #3"
    assert raw_field_name(25) == "Flux Loop - #10 / Arc Current"
    assert raw_field_name(999999) == ""


def test_registry_rejects_a_malformed_resolver_name():
    from vaft.machine_mapping.registry import DiagnosticRegistryError, validate_diagnostic_registry

    registry = load_diagnostic_registry()
    broken = {key: dict(value) for key, value in registry.items()}
    broken["tf"]["source"] = {**broken["tf"]["source"], "raw_fields": "not a spec"}
    with pytest.raises(DiagnosticRegistryError, match="raw_fields"):
        validate_diagnostic_registry(broken)
