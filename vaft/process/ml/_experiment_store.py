"""Immutable hierarchical experiment records independent of tracking engines."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

KINDS = {"campaign", "track", "study", "family", "trial", "run", "evaluation",
         "decision", "manifest", "event", "legacy", "authorization", "search"}
REQUIRED_LINKS = {
    "search": {"study": "study", "family": "family"},
    "track": {"campaign": "campaign"},
    "study": {"track": "track"},
    "trial": {"family": "family"},
    "run": {"trial": "trial"},
    "evaluation": {"run": "run"},
    "decision": {"study": "study", "run": "run"},
    "manifest": {"decision": "decision", "run": "run", "study": "study"},
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def experiment_digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def hierarchy_record(kind, payload, *, links=None):
    if kind not in KINDS:
        raise ValueError(f"unknown record kind: {kind}")
    body = {"schema": 1, "kind": kind, "payload": payload, "links": links or {}}
    # Detach nested caller-owned dictionaries and reject NaN/non-JSON values.
    body = json.loads(canonical(body))
    return {**body, "id": experiment_digest(body)}


def verify(value):
    body = {k: v for k, v in value.items() if k != "id"}
    if (body.get("schema") != 1 or body.get("kind") not in KINDS
            or value.get("id") != experiment_digest(body)):
        raise ValueError("record schema/content identity mismatch")
    if not isinstance(body.get("payload"), dict) or not isinstance(body.get("links"), dict):
        raise TypeError("record payload/links must be mappings")
    return value


def atomic_json(path, value):
    """Publish complete content exclusively, including under concurrent ingestion."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".record-")
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(canonical(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if json.loads(path.read_text()) != value:
                raise ValueError(f"immutable content conflict: {path}") from None
    finally:
        os.unlink(temporary)
    return path


class ExperimentStore:
    """One canonical store. Studies reference shared evidence rather than copying it."""

    def __init__(self, directory):
        self.directory = Path(directory)

    def get(self, identity):
        if not isinstance(identity, str) or len(identity) != 64 or any(
                c not in "0123456789abcdef" for c in identity):
            raise ValueError("expected full content SHA256")
        value = verify(json.loads((self.directory / f"{identity}.json").read_text()))
        if value["id"] != identity:
            raise ValueError("record filename identity mismatch")
        return value

    def put(self, value):
        verify(value)
        for name, kind in REQUIRED_LINKS.get(value["kind"], {}).items():
            identity = value["links"].get(name)
            if not identity or self.get(identity)["kind"] != kind:
                raise ValueError(f"missing/invalid {name} reference")
        for name, identities in value["links"].items():
            for identity in identities if isinstance(identities, list) else [identities]:
                self.get(identity)
        atomic_json(self.directory / f"{value['id']}.json", value)
        return value["id"]

    def all(self, kind=None):
        for path in sorted(self.directory.glob("*.json")):
            if path.name.startswith("."):
                continue  # Controller claims are not canonical scientific records.
            value = self.get(path.stem)
            if kind is None or value["kind"] == kind:
                yield value

    def ancestry(self, identity):
        seen, pending = set(), [identity]
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            value = self.get(current)
            yield value
            for refs in value["links"].values():
                pending.extend(refs if isinstance(refs, list) else [refs])

