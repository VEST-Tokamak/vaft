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
GEOMETRY = ROOT / "vaft" / "data" / "geometry"


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


def test_magnetics_inventory_carries_the_vest_yaml_wiring_revisions(sensors):
    vest = _load(ROOT / "vaft" / "machine_mapping" / "vest.yaml")
    wiring = vest[0]["diagnostics"]["equilibrium_magnetics"]["processing"]["wiring"]
    expected = {}
    for revision in wiring["revisions"]:
        for entry in revision["probe_overrides"]:
            expected.setdefault(int(entry["index"]), []).append(
                (revision.get("from_shot"), revision.get("to_shot"), int(entry["field_code"]), float(entry["calibration"]),
                 float(entry["r"]), float(entry["z"]))
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
