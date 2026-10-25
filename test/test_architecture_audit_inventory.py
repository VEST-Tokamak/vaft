"""The Phase A audit inventories are reproducible and match their sources (#2148).

``workflow/architecture_audit/build_inventory.py`` derives
``docs/architecture/consolidation/inventory.json`` and
``magnetics_inventory.json`` from the source tree. These tests regenerate both
and require byte equality with the committed files, so the committed audit
cannot drift silently from the code it describes. The magnetics inventory is
also checked field by field against the three external YAMLs and the
``vest.yaml`` wiring, re-read here independently of the generator.

When this fails after an intentional code change, regenerate::

    python workflow/architecture_audit/build_inventory.py
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "workflow" / "architecture_audit" / "build_inventory.py"
OUTPUT = ROOT / "docs" / "architecture" / "consolidation"
#: The three retired magnetics YAMLs, kept as a fixture since #2158.
GEOMETRY = ROOT / "test" / "data" / "legacy_magnetics"


def _generator():
    spec = importlib.util.spec_from_file_location("_architecture_audit_build_inventory", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def products() -> dict[str, str]:
    return _generator().render(ROOT)


def test_generation_is_deterministic(products):
    assert _generator().render(ROOT) == products


@pytest.mark.parametrize("name", ["inventory.json", "magnetics_inventory.json"])
def test_committed_inventory_equals_a_fresh_generation(products, name):
    relative = f"docs/architecture/consolidation/{name}"
    committed = (ROOT / relative).read_text(encoding="utf-8")
    assert committed == products[relative], (
        f"{relative} is stale; run python workflow/architecture_audit/build_inventory.py"
    )


def test_every_row_has_the_required_fields(products):
    rows = json.loads(products["docs/architecture/consolidation/inventory.json"])["rows"]
    required = {
        "implementation", "source_file", "owning_layer", "operation", "callers",
        "production_reachability", "configuration_source", "algorithm_semantics",
        "classification", "scientific_risk", "recommended_action", "related_issue",
    }
    allowed = {"canonical", "duplicate", "legacy", "intentional_specialization", "unverified"}
    assert rows
    for row in rows:
        assert required <= set(row), row["implementation"]
        assert row["classification"] in allowed, row["implementation"]
        assert (ROOT / row["source_file"]).is_file(), row["source_file"]
    keys = [(row["source_file"], row["implementation"]) for row in rows]
    assert keys == sorted(keys) and len(keys) == len(set(keys))


def _load(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def sensors(products):
    return json.loads(products["docs/architecture/consolidation/magnetics_inventory.json"])["sensors"]


def test_magnetics_inventory_is_64_probes_and_11_loops_with_unique_identities(sensors):
    kinds = [sensor["kind"] for sensor in sensors]
    assert kinds.count("b_field_pol_probe") == 64
    assert kinds.count("flux_loop") == 11
    assert kinds == ["b_field_pol_probe"] * 64 + ["flux_loop"] * 11
    assert [sensor["canonical_index"] for sensor in sensors] == list(range(75))
    for key in ("proposed_id", "field_code"):
        values = [sensor[key] for sensor in sensors]
        assert len(set(values)) == len(values), key


def test_magnetics_inventory_matches_the_three_yamls_field_by_field(sensors):
    md = _load(GEOMETRY / "MD.yaml")["channels"]
    geometry = _load(GEOMETRY / "VEST_MagneticsGeometry_Full_ver_2302.yaml")["channels"]
    names = {int(e["field_code"]): e["name"] for e in _load(GEOMETRY / "table.yaml")["entries"]}
    assert len(md) == len(geometry) == len(names) == len(sensors) == 75
    for sensor, cal, geo in zip(sensors, md, geometry):
        index = sensor["canonical_index"]
        assert sensor["field_code"] == cal["field_code"] == geo["field_code"], index
        assert sensor["md_field_code"] == cal["field_code"], index
        assert sensor["kind"] == cal["kind"] == geo["kind"] == sensor["md_kind"], index
        assert sensor["calibration"] == cal["calibration"], index
        assert (sensor["r"], sensor["z"]) == (geo["r"], geo["z"]), index
        assert sensor["name"] == names[cal["field_code"]], index
    assert set(names) == {sensor["field_code"] for sensor in sensors}


def _named_index(entry, records, *, kind="b_field_pol_probe", within_kind=False):
    """The canonical index an override or fault entry names, in either form the runtime takes.

    ``{id: ...}`` since #2158, or the legacy ``{index, r, z}`` pinned by
    position (``index`` within the kind for flux-loop faults). Restated here,
    not imported from the generator.
    """
    if "id" in entry:
        return next(int(r["index"]) for r in records if r["id"] == entry["id"])
    rows = [r for r in records if r["kind"] == kind] if within_kind else records
    record = rows[int(entry["index"])]
    assert record["kind"] == kind and (record["r"], record["z"]) == (entry["r"], entry["z"]), entry
    return int(record["index"])


def test_magnetics_inventory_carries_the_vest_yaml_wiring_revisions(sensors):
    vest = _load(ROOT / "vaft" / "machine_mapping" / "vest.yaml")
    magnetics = vest[0]["diagnostics"]["equilibrium_magnetics"]
    wiring = magnetics["processing"]["wiring"]
    records = magnetics["sensors"]["records"]
    expected = {}
    for revision in wiring["revisions"]:
        for entry in revision["probe_overrides"]:
            index = _named_index(entry, records)
            # An id entry's r/z are optional; absent, the position is the record's.
            expected.setdefault(index, []).append(
                (revision.get("from_shot"), revision.get("to_shot"), int(entry["field_code"]), float(entry["calibration"]),
                 float(entry.get("r", records[index]["r"])), float(entry.get("z", records[index]["z"])))
            )
    actual = {
        sensor["canonical_index"]: [
            (r["from_shot"], r["to_shot"], r["field_code"], r["calibration"], sensor["r"], sensor["z"])
            for r in sensor["wiring_revisions"]
        ]
        for sensor in sensors
        if sensor["wiring_revisions"]
    }
    assert actual == expected
    # The pre-39438 layout swaps the two disputed outboard positions (#956).
    assert sorted(expected) == [35, 47]


def test_no_proposed_id_reuses_another_positions_label(sensors):
    """A label-shaped id must name the position whose DAQ label it is (#2148 Q4)."""
    stems = {}
    for sensor in sensors:
        label = sensor["name"].removeprefix("MagneticFieldProbe_").removesuffix("_Bz")
        stems[label] = sensor["canonical_index"]
    for sensor in sensors:
        stem = sensor["proposed_id"].removeprefix("bpol_")
        if stem in stems:
            assert stems[stem] == sensor["canonical_index"], sensor["proposed_id"]
    assert sorted(s["canonical_index"] for s in sensors if s["id_basis"] == "position") == [35, 46, 47, 61, 62]


def test_magnetics_inventory_carries_the_vest_yaml_known_faults(sensors):
    vest = _load(ROOT / "vaft" / "machine_mapping" / "vest.yaml")
    magnetics = vest[0]["diagnostics"]["equilibrium_magnetics"]
    records = magnetics["sensors"]["records"]
    faults = magnetics["processing"]["known_faults"]
    expected = {}
    for block in [faults, *(faults.get("revisions") or ())]:
        for kind, key in (("b_field_pol_probe", "probes"), ("flux_loop", "flux_loops")):
            for entry in block.get(key) or ():
                index = _named_index(entry, records, kind=kind, within_kind=kind == "flux_loop")
                expected.setdefault(index, []).append((block.get("from_shot"), block.get("to_shot"), entry["reason"]))
    actual = {
        sensor["canonical_index"]: [(f["from_shot"], f["to_shot"], f["reason"]) for f in sensor["known_faults"]]
        for sensor in sensors
        if sensor["known_faults"]
    }
    assert actual == expected


@pytest.mark.parametrize(
    ("entry", "kind", "scope", "expected"),
    [
        ({"id": "bpol_C4-04"}, "b_field_pol_probe", "canonical", None),
        ({"id": "fl_12"}, "flux_loop", "kind", None),
        # The legacy forms the runtime adapter takes (vaft.machine_mapping.magnetics_registry).
        ({"index": 35, "r": None, "z": None}, "b_field_pol_probe", "canonical", 35),
        ({"index": 7, "r": None, "z": None}, "flux_loop", "kind", 71),
    ],
)
def test_the_generator_resolves_both_entry_forms_the_runtime_accepts(entry, kind, scope, expected):
    vest = _load(ROOT / "vaft" / "machine_mapping" / "vest.yaml")
    records = vest[0]["diagnostics"]["equilibrium_magnetics"]["sensors"]["records"]
    if "index" in entry:
        target = records[expected]
        entry = {**entry, "r": target["r"], "z": target["z"]}
    else:
        expected = next(int(r["index"]) for r in records if r["id"] == entry["id"])
    assert _generator()._entry_canonical_index(entry, kind, records, index_scope=scope) == expected
    from vaft.machine_mapping.magnetics_registry import sensor_for_entry

    assert sensor_for_entry(entry, kind=kind, context="test", index_scope=scope).index == expected


@pytest.mark.parametrize(
    "entry",
    [
        {"index": 35, "r": 0.0, "z": 0.0},  # pinned to a place the probe is not
        {"index": 35},  # a legacy entry without its pin
        {"index": 70, "r": 0.138, "z": 0.805},  # a flux loop's canonical index, as a probe
        {"id": "fl_12"},  # a flux loop's id, as a probe
        {"id": "bpol_C4-04", "index": 40},  # both forms at once
        {"id": "bpol_C4-04", "r": 0.796},  # half a pin
    ],
)
def test_the_generator_refuses_an_entry_the_runtime_refuses(entry):
    vest = _load(ROOT / "vaft" / "machine_mapping" / "vest.yaml")
    records = vest[0]["diagnostics"]["equilibrium_magnetics"]["sensors"]["records"]
    from vaft.machine_mapping.magnetics_registry import sensor_for_entry
    from vaft.machine_mapping.utils import VestConfigurationError

    with pytest.raises(ValueError):
        _generator()._entry_canonical_index(entry, "b_field_pol_probe", records, index_scope="canonical")
    # The runtime refuses it too, so the audit does not accept what production would not.
    with pytest.raises(VestConfigurationError):
        sensor_for_entry(entry, kind="b_field_pol_probe", context="test", index_scope="canonical")


def test_the_migration_matrix_is_complete_and_points_at_real_files():
    """#2164 §4: every row carries the fields, a known status, and an existing source file."""
    import re

    matrix = json.loads((OUTPUT / "migration_matrix.json").read_text(encoding="utf-8"))
    required = {
        "workflow", "rule_or_entrypoint", "source_file", "current_api", "replacement_api",
        "effective_configuration_source", "input_product", "output_product", "scientific_semantics",
        "compatibility_requirement", "required_regression_test", "migration_status",
    }
    statuses = set(matrix["migration_status_values"])
    covered = {row["workflow"] for row in matrix["rows"]}
    assert {"pipeline_1", "pipeline_2", "pipeline_3", "running_linear_stability", "vaft.cli"} <= covered
    for row in matrix["rows"]:
        name = f"{row['workflow']}:{row['rule_or_entrypoint']}"
        assert required <= set(row), name
        assert row["migration_status"] in statuses, name
        assert row["classification"] in matrix["classification_values"], name
        if row["migration_status"] in {"blocked", "provenance_pending"}:
            assert row["blocked_on"], name
        if row["migration_status"] in {"done", "in_review"}:
            assert row["evidence"], name
        if row["migration_status"] == "retired":
            continue
        for path in re.split(r"[;,]\s*", row["source_file"]):
            assert list(ROOT.glob(path.strip())), f"{name}: {path}"
