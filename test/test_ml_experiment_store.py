"""JSON-only hierarchy qualification; no downstream research project dependency."""
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from vaft.process.ml.experiments import (
    ExperimentStore,
    experiment_digest,
    hierarchy_record,
)


def test_content_identity_and_immutable_roundtrip(tmp_path):
    store = ExperimentStore(tmp_path)
    value = hierarchy_record("campaign", {"a": 1, "b": 2})
    assert value == hierarchy_record("campaign", {"b": 2, "a": 1})
    assert store.put(value) == store.put(value)
    assert store.get(value["id"]) == value
    assert len(list(store.all())) == 1
    with pytest.raises(ValueError):
        experiment_digest({"nan": float("nan")})
    with pytest.raises(ValueError):
        store.get("../outside")


def test_complete_publication_under_concurrent_ingestion(tmp_path):
    store = ExperimentStore(tmp_path)
    value = hierarchy_record("campaign", {"data": list(range(100))})
    with ThreadPoolExecutor(max_workers=4) as pool:
        ids = list(pool.map(store.put, [value] * 16))
    assert len(set(ids)) == 1
    assert list(store.all()) == [value]
    assert not list(tmp_path.glob(".record-*"))


def test_parent_types_and_shared_evidence(tmp_path):
    store = ExperimentStore(tmp_path)
    family = hierarchy_record("family", {"name": "ridge"})
    store.put(family)
    trial = hierarchy_record("trial", {"config": {"alpha": 1}}, links={"family": family["id"]})
    store.put(trial)
    run = hierarchy_record("run", {"seed": 0}, links={"trial": trial["id"]})
    store.put(run)
    evaluation = hierarchy_record("evaluation", {"metrics": {"mse": .2}}, links={"run": run["id"]})
    store.put(evaluation)
    assert {r["id"] for r in store.ancestry(evaluation["id"])} == {
        family["id"], trial["id"], run["id"], evaluation["id"]}
    with pytest.raises(ValueError):
        store.put(hierarchy_record("run", {}, links={"trial": family["id"]}))


def test_tampered_content_and_partial_controller_files(tmp_path):
    store = ExperimentStore(tmp_path)
    value = hierarchy_record("campaign", {"name": "one"})
    store.put(value)
    (tmp_path / ".controller.json").write_text('{}')
    assert list(store.all()) == [value]
    value["payload"]["name"] = "tampered"
    (tmp_path / f"{value['id']}.json").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="identity"):
        store.get(value["id"])


def test_content_address_cannot_be_rebound(tmp_path):
    store = ExperimentStore(tmp_path)
    a = hierarchy_record("campaign", {"name": "a"})
    b = hierarchy_record("campaign", {"name": "b"})
    store.put(a)
    store.put(b)
    (tmp_path / f"{a['id']}.json").write_text(json.dumps(b))
    with pytest.raises(ValueError, match="filename identity"):
        store.get(a["id"])


def test_unpublished_parent_and_missing_record(tmp_path):
    store = ExperimentStore(tmp_path)
    family = hierarchy_record("family", {"name": "ridge"})
    trial = hierarchy_record("trial", {}, links={"family": family["id"]})
    with pytest.raises(ValueError, match="unpublished family"):
        store.put(trial)
    with pytest.raises(KeyError):
        store.get(family["id"])
    assert not list(tmp_path.glob("*.json"))
