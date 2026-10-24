"""vest.yaml owns the equilibrium magnetic sensors (#2158).

Until #2158 the 75 sensors were three independently maintained YAMLs joined by
list position (`test_magnetics_geometry_join.py` guarded the join). They are now
one registry, ``0.diagnostics.equilibrium_magnetics.sensors``, read by
``vaft.machine_mapping.magnetics_registry``. This module checks:

* the registry schema, uniqueness, completeness and ordering;
* field-by-field equality with the three retired YAMLs, kept byte-identical as
  frozen, non-runtime fixtures in ``test/data/legacy_magnetics``;
* that no runtime copy or reader of them survives;
* the shot-era wiring (39437/39438) and the DAQ-mode boundary (46403/46404);
* that every public output -- static ODS magnetics leaves, channel
  definitions, wiring, known faults, raw-source records, constraint groups,
  EFUND projection -- is identical to what the pre-#2158 code produced
  (``pre_2158_outputs.json``, generated from a git-archive export of develop
  75b154e31, not from this code);
* the validator, on synthetic configurations.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from pathlib import Path

import pytest
import yaml
from omas import ODS

from vaft.machine_mapping import magnetics_registry as registry
from vaft.machine_mapping.magnetics import (
    TOROIDAL_MIRNOV_REFERENCE_CHANNELS,
    _pinned_channel_index,
    _pinned_probe_index,
    equilibrium_magnetics_processing_config,
    known_magnetics_faults,
    magnetics_wiring_for_shot,
    vest_equilibrium_magnetics_channel_definitions,
    vest_md_channel_definitions,
    vfit_magnetics_static,
)
from vaft.machine_mapping.magnetics_registry import (
    FLUX_LOOP_KIND,
    PROBE_KIND,
    REGISTRY_CONFIG_PATH,
    resolve_equilibrium_magnetics_registry,
    resolve_equilibrium_magnetics_sensors,
)
from vaft.machine_mapping.utils import VestConfigurationError, package_data_path

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "test" / "data" / "legacy_magnetics"
VEST_YAML = Path(package_data_path("vest.yaml"))
RETIRED = ("MD.yaml", "VEST_MagneticsGeometry_Full_ver_2302.yaml", "table.yaml")
POSITION_IDS = {
    35: ("bpol_C_z+0.060", "C2-04"),
    46: ("bpol_C_z-0.380", "C4-05"),
    47: ("bpol_C_z-0.420", "C4-06"),
    61: ("bpol_L_z-1.0328", "L-06"),
    62: ("bpol_L_z-1.0728", "L-07"),
}


@pytest.fixture(scope="module")
def inventory():
    return resolve_equilibrium_magnetics_sensors()


@pytest.fixture(scope="module")
def before():
    return json.loads((LEGACY / "pre_2158_outputs.json").read_text(encoding="utf-8"))


def _legacy(name):
    return yaml.safe_load((LEGACY / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_the_registry_holds_64_probes_then_11_flux_loops(inventory):
    assert len(inventory) == 75
    assert [record.kind for record in inventory] == [PROBE_KIND] * 64 + [FLUX_LOOP_KIND] * 11
    assert [record.index for record in inventory] == list(range(75))
    assert [record.kind_index for record in inventory] == list(range(64)) + list(range(11))


@pytest.mark.parametrize("attribute", ["id", "index", "field_code", "name"])
def test_identities_are_unique(inventory, attribute):
    values = [getattr(record, attribute) for record in inventory]
    assert len(set(values)) == len(values)


def test_geometry_and_calibration_are_well_formed(inventory):
    for record in inventory:
        assert math.isfinite(record.r) and math.isfinite(record.z), record.id
        assert math.isfinite(record.calibration) and record.calibration != 0.0, record.id
        assert record.rewired is False


def test_ids_are_label_shaped_except_the_five_ladder_disputed_positions(inventory):
    """#2148 Q4 is not resolved by #2158: the five ids stay position-shaped."""
    position = {record.index: (record.id, record.ladder_candidate) for record in inventory if record.id_basis == "position"}
    assert position == POSITION_IDS
    for record in inventory:
        if record.id_basis == "table":
            assert record.id == registry.label_shaped_id(record.kind, record.name)
            assert record.ladder_candidate is None
    # No position-shaped id is a label-shaped id another position carries.
    label_ids = {registry.label_shaped_id(r.kind, label) for r in inventory for label in (r.name, *r.aliases)}
    assert not {sensor_id for sensor_id, _ in POSITION_IDS.values()} & label_ids


def test_aliases_are_exactly_the_other_era_labels(inventory):
    aliased = {record.index: record.aliases for record in inventory if record.aliases}
    assert aliased == {35: ("MagneticFieldProbe_C2-04_Bz",), 47: ("MagneticFieldProbe_C4-05",)}


def test_the_unsuffixed_c4_labels_are_kept_verbatim(inventory):
    """#2148 Q2: ODS identifiers depend on these strings; no `_Bz` is invented."""
    names = {record.field_code: record.name for record in inventory}
    assert [names[code] for code in (223, 224, 225, 226)] == [
        "MagneticFieldProbe_C4-03",
        "MagneticFieldProbe_C4-04",
        "MagneticFieldProbe_C4-05",
        "MagneticFieldProbe_C4-06",
    ]


def test_the_field_171_sign_conflict_is_recorded_not_resolved(inventory):
    """#2148 Q1: both values stay until a follow-up issue decides the sign."""
    assert inventory[36].field_code == 171 and inventory[36].calibration == -0.004528986
    reference = [c for c in TOROIDAL_MIRNOV_REFERENCE_CHANNELS if int(c["field_code"]) == 171]
    assert [float(c["gain"]) for c in reference] == [0.004529]


def test_the_positional_late_flux_baseline_rule_hits_loops_13_to_15(inventory):
    """#2148 Q7: `flux_loop_number in (9, 10, 11)` counts positions (1-based).

    The rule in vaft.process.magnetics is preserved as it is; with the registry
    order those positions are physical loops #13-#15, not #9-#11.
    """
    loops = [record for record in inventory if record.kind == FLUX_LOOP_KIND]
    assert [loops[number - 1].id for number in (9, 10, 11)] == ["fl_13", "fl_14", "fl_15"]


def test_the_block_records_its_migration_provenance():
    source = registry.registry_source()
    assert source["wiring_layout"] == "2409"
    assert source["calibration_operation"] == "divide"
    for key, asset in (("calibration", "MD.yaml"), ("geometry", "VEST_MagneticsGeometry_Full_ver_2302.yaml"), ("names", "table.yaml")):
        assert source[key]["asset"] == asset
        assert source[key]["sha256"] == hashlib.sha256((LEGACY / asset).read_bytes()).hexdigest()
    # The 2302 file declares ver_2409.mat; that is recorded, not reinterpreted.
    assert source["geometry"]["declared_source"] == _legacy(RETIRED[1])["source"]
    assert source["calibration"]["declared_source"] == _legacy(RETIRED[0])["source"]
    assert source["names"]["declared_source"] == _legacy(RETIRED[2])["source"]


def test_the_records_are_frozen(inventory):
    with pytest.raises(dataclasses.FrozenInstanceError):
        inventory[0].field_code = 1
    definitions = vest_equilibrium_magnetics_channel_definitions()
    definitions[0]["field_code"] = -1
    assert vest_equilibrium_magnetics_channel_definitions()[0]["field_code"] == 179


# ---------------------------------------------------------------------------
# Frozen fixtures: field-by-field equality, no runtime copy, no reader
# ---------------------------------------------------------------------------


def test_the_registry_equals_the_frozen_legacy_yamls_field_by_field(inventory):
    md = _legacy("MD.yaml")["channels"]
    geometry = _legacy("VEST_MagneticsGeometry_Full_ver_2302.yaml")["channels"]
    names = {entry["field_code"]: entry["name"] for entry in _legacy("table.yaml")["entries"]}
    assert len(md) == len(geometry) == len(names) == len(inventory)
    for record, cal, geo in zip(inventory, md, geometry):
        assert (record.field_code, record.kind, record.calibration) == (cal["field_code"], cal["kind"], cal["calibration"])
        assert (record.field_code, record.kind, record.r, record.z) == (geo["field_code"], geo["kind"], geo["r"], geo["z"])
        assert record.name == names[record.field_code]
    assert set(names) == {record.field_code for record in inventory}


def test_the_retired_yamls_have_no_runtime_copy():
    geometry = ROOT / "vaft" / "data" / "geometry"
    assert not list(geometry.glob("*.yaml"))
    for name in RETIRED:
        assert not list((ROOT / "vaft").rglob(name)), name


def test_no_runtime_code_names_the_retired_yamls():
    pattern = re.compile(r"(?<![\w.-])(MD|table)\.ya?ml\b|VEST_MagneticsGeometry_Full_ver_", re.IGNORECASE)
    offenders = [
        str(path.relative_to(ROOT))
        for top in ("vaft", "workflow")
        for path in (ROOT / top).rglob("*.py")
        if pattern.search(path.read_text(encoding="utf-8", errors="replace"))
    ]
    assert offenders == []


def test_no_second_sensor_inventory_in_vest_yaml():
    """The #843 duplicate and the dead `0.magnetics.flux_loop` placeholder stay gone."""
    content = yaml.safe_load(VEST_YAML.read_text(encoding="utf-8"))
    assert "magnetics" not in content[0]
    paths = []

    def walk(node, path):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{path}.{key}")
        elif isinstance(node, list) and node and all(isinstance(item, dict) for item in node):
            if sum(1 for item in node if {"r", "z", "field_code"} <= set(item)) >= 11:
                paths.append(path)
            for index, item in enumerate(node):
                walk(item, f"{path}[{index}]")

    walk(content, "")
    assert paths == [f".0.{REGISTRY_CONFIG_PATH.removeprefix('0.')}.records"]


# ---------------------------------------------------------------------------
# Shot eras
# ---------------------------------------------------------------------------


def test_39437_and_39438_swap_the_two_outboard_fields_not_the_sensors(inventory):
    early = resolve_equilibrium_magnetics_sensors(39437)
    late = resolve_equilibrium_magnetics_sensors(39438)
    assert [r.id for r in early] == [r.id for r in late] == [r.id for r in inventory]
    assert [(r.r, r.z) for r in early] == [(r.r, r.z) for r in inventory]
    assert late == inventory
    changed = {r.index for r, s in zip(early, late) if r != s}
    assert changed == {35, 47}
    plus, minus = early[35], early[47]
    assert (plus.id, plus.field_code, plus.calibration, plus.name, plus.rewired) == (
        "bpol_C_z+0.060", 170, 0.004539265, "MagneticFieldProbe_C2-04_Bz", True,
    )
    assert (minus.id, minus.field_code, minus.calibration, minus.name, minus.rewired) == (
        "bpol_C_z-0.420", 225, -0.00452284, "MagneticFieldProbe_C4-05", True,
    )
    assert resolve_equilibrium_magnetics_registry(39437).layout == "pre-39438"
    assert resolve_equilibrium_magnetics_registry(39438).layout == "2409"


def test_the_46403_46404_daq_transition_does_not_touch_the_sensors(inventory):
    assert resolve_equilibrium_magnetics_sensors(46403) == resolve_equilibrium_magnetics_sensors(46404) == inventory
    assert equilibrium_magnetics_processing_config(46403).daq_mode == "legacy"
    assert equilibrium_magnetics_processing_config(46404).daq_mode == "native_daq"


def test_shot_zero_applies_the_wiring_whose_bounds_contain_it():
    """`None` is the inventory; 0 is a shot and falls in the `to_shot: 39437` era."""
    assert resolve_equilibrium_magnetics_registry(0).layout == "pre-39438"
    assert resolve_equilibrium_magnetics_registry(None).layout == "2409"


def test_the_fingerprint_identifies_the_resolved_records(inventory):
    fingerprint = registry.registry_fingerprint()
    assert re.fullmatch(r"[0-9a-f]{64}", fingerprint)
    payload = json.dumps([r.as_dict() for r in inventory], sort_keys=True, separators=(",", ":"))
    assert fingerprint == hashlib.sha256(payload.encode()).hexdigest()
    assert registry.registry_fingerprint(39438) == fingerprint
    assert registry.registry_fingerprint(39437) != fingerprint


def test_the_fingerprint_ignores_the_rest_of_vest_yaml(tmp_path):
    copy = tmp_path / "vest.yaml"
    copy.write_text(VEST_YAML.read_text(encoding="utf-8") + "\n# an unrelated edit\n", encoding="utf-8")
    assert registry.registry_fingerprint(info_file=str(copy)) == registry.registry_fingerprint()


def test_the_legacy_index_adapter_lands_on_the_same_sensor():
    assert _pinned_probe_index({"index": 35, "r": 0.796, "z": 0.06}, context="t") == 35
    assert _pinned_probe_index({"id": "bpol_C_z+0.060"}, context="t") == 35
    assert _pinned_channel_index({"index": 3, "r": 0.592, "z": -0.685}, FLUX_LOOP_KIND, context="t") == 3
    assert _pinned_channel_index({"id": "fl_06"}, FLUX_LOOP_KIND, context="t") == 3
    with pytest.raises(VestConfigurationError, match="not at r="):
        _pinned_probe_index({"id": "bpol_C_z+0.060", "r": 0.796, "z": -0.42}, context="t")
    with pytest.raises(VestConfigurationError, match="is a flux_loop"):
        _pinned_probe_index({"id": "fl_06"}, context="t")
    with pytest.raises(VestConfigurationError, match="exactly one"):
        _pinned_probe_index({"id": "bpol_C_z+0.060", "index": 35, "r": 0.796, "z": 0.06}, context="t")
    with pytest.raises(VestConfigurationError, match="no sensor with id"):
        _pinned_probe_index({"id": "bpol_C2-04"}, context="t")


# ---------------------------------------------------------------------------
# Public outputs identical to the pre-#2158 code
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shot", ["0", "39437", "39438", "39915", "46404"])
def test_static_magnetics_leaves_are_identical_to_the_pre_2158_code(before, shot):
    ods = ODS(consistency_check=False)
    vfit_magnetics_static(ods, int(shot))
    leaves = {key: value for key, value in ods.flat().items()}
    expected = before["static_magnetics"][shot]
    assert list(leaves) == list(expected)
    for key, value in expected.items():
        assert type(leaves[key]) is type(value) and leaves[key] == value, key


def test_channel_definitions_are_identical_to_the_pre_2158_code(before):
    definitions = before["channel_definitions"]
    assert [dict(c) for c in vest_equilibrium_magnetics_channel_definitions()] == definitions["inventory"]
    assert vest_md_channel_definitions() == vest_equilibrium_magnetics_channel_definitions()
    for shot, expected in definitions.items():
        if shot == "inventory":
            continue
        actual = vest_equilibrium_magnetics_channel_definitions(int(shot))
        assert [list(c) for c in actual] == [list(c) for c in expected]
        assert [dict(c) for c in actual] == expected


@pytest.mark.parametrize("shot", ["0", "39437", "39438", "39915", "46403", "46404"])
def test_wiring_is_identical_to_the_pre_2158_code(before, shot):
    wiring = magnetics_wiring_for_shot(int(shot))
    expected = before["wiring"][shot]
    assert wiring.layout == expected["layout"]
    assert [dict(o) for o in wiring.overrides] == expected["overrides"]
    assert wiring.provenance == expected["provenance"]
    assert json.loads(json.dumps(wiring.provenance)) == expected["provenance"]


def test_known_faults_are_identical_to_the_pre_2158_code(before):
    for shot, expected in before["known_faults"].items():
        actual = [[kind, index, reason] for (kind, index), reason in sorted(known_magnetics_faults(int(shot)).items())]
        assert actual == expected, shot


@pytest.mark.parametrize("shot", ["0", "39437", "39438", "46404"])
def test_raw_source_records_are_identical_to_the_pre_2158_code(before, shot):
    from vaft.machine_mapping.raw_daq_sources import b_field_pol_probe_raw_fields, flux_loop_raw_fields

    loops = [[r.field, r.label, r.channel] for r in flux_loop_raw_fields(int(shot))]
    probes = [[r.field, r.label, r.channel] for r in b_field_pol_probe_raw_fields(int(shot))][:64]
    assert loops == before["raw_fields"][shot]["flux_loop"]
    assert probes == before["raw_fields"][shot]["b_field_pol_probe"]


def test_constraint_groups_are_identical_to_the_pre_2158_code(before):
    from vaft.machine_mapping.utils import _load_magnetics_channel_groups

    assert _load_magnetics_channel_groups() == before["constraint_groups"]


def test_the_efund_projection_is_identical_to_the_pre_2158_code(before):
    """`rsi`/`zsi`/`lpname` and `xmp2`/`ymp2`/`amp2`/`smp2`/`mpnam2` come from these."""
    from vaft.machine_mapping.efund_geometry import _flux_loops, _probes

    ods = ODS(consistency_check=False)
    vfit_magnetics_static(ods)
    projected = {**_flux_loops(ods), **_probes(ods)}
    expected = before["efund_projection"]
    for key in ("loop_r", "loop_z", "probe_r", "probe_z", "probe_angle_deg", "probe_length"):
        assert projected[key].tolist() == expected[key], key
    assert list(projected["loop_names"]) == expected["loop_names"]
    assert list(projected["probe_names"]) == expected["probe_names"]


def test_the_static_manifest_records_the_registry_fingerprint():
    from vaft.machine_mapping.provenance import vest_processing_provenance

    record = vest_processing_provenance(39437)["equilibrium_magnetics"]["sensor_registry"]
    assert record == {"source": REGISTRY_CONFIG_PATH, "fingerprint": registry.registry_fingerprint(39437)}


# ---------------------------------------------------------------------------
# The validator, on synthetic configurations
# ---------------------------------------------------------------------------


def _variant(tmp_path, mutate):
    content = yaml.safe_load(VEST_YAML.read_text(encoding="utf-8"))
    mutate(content[0]["diagnostics"]["equilibrium_magnetics"])
    path = tmp_path / "vest.yaml"
    path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
    return str(path)


def _records(block):
    return block["sensors"]["records"]


def _overrides(block):
    return block["processing"]["wiring"]["revisions"][0]["probe_overrides"]


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda b: _records(b).pop(), "expected"),
        (lambda b: _records(b)[3].update(index=4), "not its list position"),
        (lambda b: _records(b)[3].update(id=_records(b)[2]["id"], id_basis="position"), "position-shaped"),
        (lambda b: _records(b)[3].update(calibration=0.0), "non-zero"),
        (lambda b: _records(b)[3].update(r=float("nan")), "finite"),
        (lambda b: _records(b)[3].update(field_code=_records(b)[4]["field_code"]), "duplicate field_code"),
        (lambda b: _records(b)[3].update(id="bpol_H9-99"), "implies"),
        (lambda b: _records(b)[3].update(colour="red"), "unknown keys"),
        (lambda b: _records(b)[35].update(id="bpol_C_z+0.100"), "position-shaped"),
        (lambda b: _records(b)[35].update(aliases=["NotALabel"]), "not the label"),
        (lambda b: _records(b).insert(0, _records(b).pop(70)), "not its list position"),
    ],
)
def test_malformed_records_are_rejected(tmp_path, mutate, match):
    path = _variant(tmp_path, mutate)
    with pytest.raises(VestConfigurationError, match=match):
        resolve_equilibrium_magnetics_sensors(info_file=path)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda b: _overrides(b)[0].update(id="bpol_nope"), "no sensor with id"),
        (lambda b: _overrides(b)[1].update(field_code=171), "two positions|neither its name"),
        (lambda b: _overrides(b)[1].update(id="bpol_C_z+0.060", r=0.796, z=0.06), "overridden twice"),
        (lambda b: _overrides(b)[0].update(z=-0.42), "not at r="),
        (lambda b: _overrides(b)[0].update(calibration=0), "non-zero"),
        (lambda b: _overrides(b)[0].update(gain=1.0), "unknown override keys"),
        # Moving a label to a sensor that does not list it as an alias.
        (lambda b: _records(b)[35].update(aliases=[]), "neither its name"),
    ],
)
def test_malformed_wiring_is_rejected(tmp_path, mutate, match):
    path = _variant(tmp_path, mutate)
    with pytest.raises(VestConfigurationError, match=match):
        resolve_equilibrium_magnetics_sensors(39437, info_file=path)


def test_a_legacy_index_override_resolves_like_the_id_form(tmp_path):
    def legacy(block):
        _overrides(block)[0] = {"index": 35, "r": 0.796, "z": 0.06, "field_code": 170, "calibration": 0.004539265}
        _overrides(block)[1] = {"index": 47, "r": 0.796, "z": -0.42, "field_code": 225, "calibration": -0.00452284}

    path = _variant(tmp_path, legacy)
    assert resolve_equilibrium_magnetics_sensors(39437, info_file=path) == resolve_equilibrium_magnetics_sensors(39437)


def test_the_raw_db_name_table_agrees_with_every_registry_label(inventory):
    """An independent reference (#2148 Q6): `sql_table.txt` names every DAQ field.

    It stays the raw-database label table -- a different contract, covering
    fields the registry does not -- and is not a second sensor inventory: it
    carries no position or calibration. Field 25 has two legacy names there.
    """
    from vaft.database.raw import SQL_TABLE_PATH

    table = json.loads(Path(SQL_TABLE_PATH).read_text(encoding="utf-8"))
    names: dict[int, set[str]] = {}
    for name, field in table.items():
        names.setdefault(int(field), set()).add(name)
    for record in inventory:
        assert record.name in names.get(record.field_code, set()), record.id
    assert names[25] == {"Flux Loop - #10", "Arc Current"}
