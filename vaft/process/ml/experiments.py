"""Local JSON experiment records; optional, explicitly selected MLflow index.

ModelArtifact and vaft-nn remain canonical. Tracking never uploads weights or
raw scientific observations. Paths/references are recorded only by the caller.
"""
from __future__ import annotations

__all__ = ['experiment_record', 'save_experiment', 'load_experiment', 'index_mlflow']

import json
from pathlib import Path
import uuid

from ._types import _jsonable, ModelContractError


def experiment_record(result, *, references=None, run_id=None):
    """A JSON-only run index retaining immutable model identity and provenance."""
    model = result.artifact
    return _jsonable({"run_schema": 1, "run_id": run_id or str(uuid.uuid4()),
        "model_identity": model.resolved_identity().to_dict(), "model_spec": model.model_spec.to_dict(),
        "training": dict(model.training), "history": dict(result.history), "metrics": dict(model.metrics),
        "references": dict(references or {}), "scientific_status": "unqualified"})


def save_experiment(record, path):
    """Write a new local JSON record without overwriting an existing run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(_jsonable(record), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return path


def load_experiment(path):
    """Inspect a VAFT run or legacy research JSON report without dropping fields."""
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ModelContractError("experiment report must be a JSON object")
    return record


def index_mlflow(record, directory, *, experiment="vaft-local"):
    """Explicit local, server-free MLflow PoC; JSON and model identity stay canonical.

    Only a local directory is accepted. No remote tracking URI, implicit global
    client configuration, weights or dataset artifacts are used.
    """
    try:
        from mlflow.tracking import MlflowClient
    except ImportError as exc:
        raise ImportError("local MLflow tracking needs optional mlflow; pip install 'vaft[tracking]'") from exc
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    # SQLite works with current MLflow defaults and requires no tracking server.
    client = MlflowClient(tracking_uri="sqlite:///" + str(root / "tracking.db"))
    existing = client.get_experiment_by_name(experiment)
    eid = existing.experiment_id if existing else client.create_experiment(experiment, artifact_location=root.as_uri())
    run = client.create_run(eid, tags={"vaft.run_id": str(record.get("run_id", "legacy")),
        "vaft.model_identity": json.dumps(record.get("model_identity", {}), sort_keys=True)})
    rid = run.info.run_id
    try:
        client.log_param(rid, "configuration", json.dumps(record.get("model_spec", {}), sort_keys=True))
        for key, values in record.get("history", {}).items():
            for step, value in enumerate(values):
                client.log_metric(rid, key.replace("/", "_"), float(value), step=step)
        client.set_tag(rid, "vaft.provenance", json.dumps(record.get("training", {}), sort_keys=True))
        client.set_terminated(rid, status="FINISHED")
    except Exception:
        client.set_terminated(rid, status="FAILED")
        raise
    return {"tracking_run_id": rid, "run_id": record.get("run_id"), "uri": "sqlite:///" + str(root / "tracking.db")}
