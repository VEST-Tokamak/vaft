"""The optional deployment catalog (#1129, generic half).

A catalog validates allocation and states defaults; it never supplies the name
of a stored occurrence, which is the payload's own ``ids_properties.name``
(#1128). No VEST allocation appears here: these are test-only names.
"""

from __future__ import annotations

import pytest

pytest.importorskip("imas")

from vaft.imas import InstanceCatalog, InstanceInfo, PhysicalLocator  # noqa: E402
from vaft.imas._dd import CatalogError  # noqa: E402


def _block(default="instance-a", **extra):
    block = {
        "default": default,
        "instances": [
            {"occurrence": 0, "name": "instance-a", "description": "First."},
            {
                "occurrence": 3,
                "name": "instance-b",
                "description": "Derived from instance-a.",
                "input": {"ids": "equilibrium", "name": "instance-a"},
                "aliases": ["instance-b-old"],
            },
        ],
    }
    block.update(extra)
    return block


def _catalog(**extra):
    return InstanceCatalog.from_mapping({"equilibrium": _block(**extra)})


def _info(occurrence, name, store="main"):
    return InstanceInfo("equilibrium", PhysicalLocator(store, "equilibrium", occurrence), name, "payload")


def test_parses_records_defaults_inputs_and_aliases():
    catalog = _catalog()
    assert catalog.default("equilibrium") == "instance-a"
    assert [e.occurrence for e in catalog.entries("equilibrium")] == [0, 3]
    assert catalog.entry("equilibrium", "instance-b").inputs[0].name == "instance-a"
    assert catalog.canonical_name("equilibrium", "instance-b-old") == "instance-b"
    assert catalog.entry_at("equilibrium", 3).name == "instance-b"


def test_an_explicit_null_default_is_recorded_as_a_policy():
    catalog = _catalog(default=None)
    assert catalog.declares_default("equilibrium")
    assert catalog.default("equilibrium") is None


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda b: b["instances"].append({"occurrence": 0, "name": "instance-c", "description": "x"}), "assigned twice"),
        (lambda b: b["instances"].append({"occurrence": 5, "name": "instance-a", "description": "x"}), "used twice"),
        (lambda b: b.update(default="nope"), "not a registered name"),
        (lambda b: b["instances"][1].update(input={"ids": "equilibrium", "name": "nope"}), "not a registered instance"),
        (lambda b: b["instances"][0].update(name="Instance_A"), "kebab-case"),
        (lambda b: b["instances"][0].pop("description"), "missing"),
        (lambda b: b["instances"][0].update(occurrence=-1), "integer >= 0"),
        (lambda b: b.pop("default"), "explicitly"),
        (lambda b: b.update(instances={"instance-a": {}}), "list of records"),
    ],
)
def test_rejects_an_inconsistent_catalog(mutate, message):
    block = _block()
    mutate(block)
    with pytest.raises(CatalogError, match=message):
        InstanceCatalog.from_mapping({"equilibrium": block})


def test_rejects_an_unknown_ids():
    with pytest.raises(CatalogError, match="not an IDS"):
        InstanceCatalog.from_mapping({"not_an_ids": _block()})


def test_check_accepts_agreement_and_sparse_sets():
    assert _catalog().check([_info(0, "instance-a")]) == ()


def test_check_never_supplies_names_for_unnamed_occurrences():
    assert _catalog().check([_info(0, None), _info(3, None)]) == ()


def test_check_rejects_a_name_at_the_wrong_occurrence():
    (finding,) = _catalog().check([_info(4, "instance-b")])
    assert finding.level == "error" and "allocates occurrence 3" in finding.message


def test_check_rejects_another_name_at_an_allocated_occurrence():
    (finding,) = _catalog().check([_info(3, "something-else")])
    assert finding.level == "error" and "allocates it to 'instance-b'" in finding.message


def test_check_only_warns_on_an_unmanaged_name():
    (finding,) = _catalog().check([_info(9, "third-party")])
    assert finding.level == "warning"


def test_snapshot_is_canonical_and_hashed():
    one, two = _catalog().snapshot(), _catalog().snapshot()
    assert one == two and len(one["sha256"]) == 64
    assert one["catalog"]["equilibrium"]["instances"][1]["input"] == [
        {"ids": "equilibrium", "name": "instance-a"}
    ]


def test_appending_an_instance_is_compatible():
    old = _catalog()
    block = _block()
    block["instances"].append({"occurrence": 9, "name": "instance-c", "description": "New."})
    update = InstanceCatalog.classify_update(old, InstanceCatalog.from_mapping({"equilibrium": block}))
    assert update.compatible and not update.reasons


def test_moving_or_reusing_an_occurrence_is_incompatible():
    old = _catalog()
    block = _block()
    block["instances"][1]["occurrence"] = 5
    block["instances"].append({"occurrence": 3, "name": "instance-c", "description": "Reuse."})
    update = InstanceCatalog.classify_update(old, InstanceCatalog.from_mapping({"equilibrium": block}))
    assert not update.compatible
    assert any("moved from occurrence 3 to 5" in r for r in update.reasons)
    assert any("reassigned" in r for r in update.reasons)


def test_a_default_change_is_reported():
    update = InstanceCatalog.classify_update(_catalog(), _catalog(default="instance-b"))
    assert update.compatible and any("default changed" in r for r in update.reasons)


def test_from_yaml(tmp_path):
    path = tmp_path / "catalog.yaml"
    path.write_text(
        "equilibrium:\n  default: null\n  instances:\n"
        "    - {occurrence: 0, name: instance-a, description: First.}\n"
    )
    assert InstanceCatalog.from_yaml(path).entry("equilibrium", "instance-a").occurrence == 0
