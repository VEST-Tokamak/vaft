"""On-disk model artifacts: manifest, hashes, export (#669 sections 7, 11, 12).

An artifact directory holds the backend's weights, the normalisation moments
and ``manifest.json``.  The manifest pins the SHA-256 and size of every other
file, so the manifest's own SHA-256 identifies the whole artifact -- the hash
``vaft-nn`` records for a published version.
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

import numpy as np

from ._backends import get_backend
from ._provenance import sha256_bytes, sha256_file
from ._types import (
    CalibrationResult,
    ModelArtifact,
    ModelContractError,
    ModelIdentity,
    ModelResolutionError,
    ModelSpec,
    _jsonable,
)

__all__ = ["export_model", "load_model_artifact", "save_model_artifact"]

#: Manifest layout version; provisional until the first published model (#669 section 11).
MANIFEST_SCHEMA_VERSION = 0
MANIFEST = "manifest.json"
NORMALIZATION = "normalization.npz"

_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+([-+][0-9A-Za-z.-]+)?$")


def _npz(arrays) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **{k: np.asarray(v) for k, v in arrays.items()})
    return buffer.getvalue()


def _manifest(model: ModelArtifact, version: str, files: dict) -> dict:
    backend = get_backend(model.model_spec.backend)
    return _jsonable({
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "model": model.name,
        "version": version,
        "task": model.model_spec.task,
        "backend": model.model_spec.backend,
        "inference_format": backend.inference_format,
        "model_spec": model.model_spec.to_dict(),
        "input_shape": list(model.input_shape),
        "target_shape": None if model.target_shape is None else list(model.target_shape),
        "supervised": model.supervised,
        "outputs": list(model.outputs),
        "input_names": list(model.input_names),
        "files": files,
        "training": dict(model.training),
        "normalization": {
            "file": NORMALIZATION,
            "convention": "per-element z-score (x - mean) / std applied before the network",
            "arrays": sorted(model.normalization),
        },
        "calibration": {k: v.to_dict() for k, v in model.calibration.items()} or None,
        "metrics": dict(model.metrics),
        "compatibility": {"vaft": f">={model.training.get('vaft_version', '0')}"},
    })


def save_model_artifact(model, directory, *, version):
    """Write a model as an immutable, hash-pinned artifact directory.

    The directory is what gets attached to a ``vaft-nn`` release; its
    ``manifest.json`` is what gets committed to the registry.

    Parameters
    ----------
    model : ModelArtifact
        Trained model; ``model.name`` must be lower_snake_case [-].
    directory : str or path-like
        Target directory; must not exist or be empty [-].
    version : str
        Semantic version of this artifact, e.g. ``"0.1.0"`` [-].

    Returns
    -------
    ModelArtifact
        The same model carrying its saved identity: version, manifest SHA-256
        and weight SHA-256 [-].

    Raises
    ------
    ModelContractError
        On an invalid name or version, or a non-empty target directory.

    Processing steps
    ----------------
    1. Write the backend weights and the normalisation moments.
    2. Hash each file and write ``manifest.json`` (sorted keys, two-space
       indent) pinning every hash and size.
    3. Hash the manifest; that hash is the artifact's identity.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [vaft669] VEST-Tokamak/vaft#669, sections 10-11: the ``vaft-nn`` release
       layout and manifest contract.
    """
    if not _NAME.match(model.name):
        raise ModelContractError(f"model name {model.name!r} must match {_NAME.pattern}")
    if not _VERSION.match(str(version)):
        raise ModelContractError(f"version {version!r} is not a semantic version")
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ModelContractError(f"{directory} is not empty; a published version is never overwritten")
    directory.mkdir(parents=True, exist_ok=True)
    backend = get_backend(model.model_spec.backend)
    payloads = {
        f"weights{backend.weights_suffix}": model.state,
        NORMALIZATION: _npz(model.normalization),
    }
    files = {}
    for filename, payload in payloads.items():
        (directory / filename).write_bytes(payload)
        files[filename] = {"sha256": sha256_bytes(payload), "size": len(payload)}
    text = json.dumps(_manifest(model, str(version), files), indent=2, sort_keys=True) + "\n"
    (directory / MANIFEST).write_text(text, encoding="utf-8", newline="\n")
    identity = ModelIdentity(
        name=model.name,
        state_sha256=model.state_sha256,
        version=str(version),
        manifest_sha256=sha256_bytes(text.encode("utf-8")),
        resolved_by="saved",
        source=str(directory),
    )
    return model.replace(identity=identity)


def _verify_files(manifest: dict, directory: Path) -> None:
    problems = []
    for filename, record in manifest["files"].items():
        path = directory / filename
        if not path.is_file():
            problems.append(f"{filename}: missing")
        elif path.stat().st_size != record["size"]:
            problems.append(f"{filename}: size {path.stat().st_size} != {record['size']}")
        elif sha256_file(path) != record["sha256"]:
            problems.append(f"{filename}: sha256 differs from the manifest")
    if problems:
        raise ModelResolutionError(f"artifact in {directory} fails verification: " + "; ".join(problems))


def load_model_artifact(directory, *, manifest=None, verify=True):
    """Load an artifact directory, verifying every file against its manifest.

    Parameters
    ----------
    directory : str or path-like
        Directory holding the weights and normalisation files [-].
    manifest : str or path-like or None, optional
        Manifest to trust; ``directory/manifest.json`` when ``None``.  The
        resolver passes the registry's reviewed copy here, so a tampered cache
        cannot vouch for itself [-].
    verify : bool, optional
        Check the SHA-256 and size of every file before loading [-].

    Returns
    -------
    ModelArtifact
        The model with an identity naming the directory, version and manifest
        hash [-].

    Raises
    ------
    ModelResolutionError
        When the manifest is missing or a file is missing or differs.
    ModelContractError
        When the manifest's schema version is not one this VAFT reads.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [vaft669] VEST-Tokamak/vaft#669, section 13: resolution verifies identity,
       version and artifact hash.
    """
    directory = Path(directory)
    manifest_path = Path(manifest) if manifest is not None else directory / MANIFEST
    if not manifest_path.is_file():
        raise ModelResolutionError(f"no manifest at {manifest_path}")
    raw = manifest_path.read_bytes()
    record = json.loads(raw)
    if record.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ModelContractError(
            f"manifest schema_version {record.get('schema_version')!r}; this VAFT reads {MANIFEST_SCHEMA_VERSION}"
        )
    if verify:
        _verify_files(record, directory)
    spec_record = record["model_spec"]
    spec = ModelSpec(
        architecture=spec_record["architecture"],
        task=spec_record.get("task", record["task"]),
        backend=spec_record.get("backend", record["backend"]),
        hyperparameters=spec_record.get("hyperparameters", {}),
    )
    backend = get_backend(spec.backend)
    state = (directory / f"weights{backend.weights_suffix}").read_bytes()
    with np.load(directory / NORMALIZATION, allow_pickle=False) as payload:
        normalization = {key: payload[key] for key in payload.files}
    calibration = {
        key: CalibrationResult(**{**value, "groups": tuple(value.get("groups", ()))})
        for key, value in (record.get("calibration") or {}).items()
    }
    model = ModelArtifact(
        name=record["model"],
        model_spec=spec,
        state=state,
        input_shape=tuple(record["input_shape"]),
        target_shape=None if record["target_shape"] is None else tuple(record["target_shape"]),
        supervised=bool(record["supervised"]),
        outputs=tuple(record["outputs"]),
        normalization=normalization,
        input_names=tuple(record.get("input_names", ())),
        training=record.get("training", {}),
        metrics=record.get("metrics", {}),
        calibration=calibration,
    )
    identity = ModelIdentity(
        name=model.name,
        state_sha256=model.state_sha256,
        version=record["version"],
        manifest_sha256=sha256_bytes(raw),
        resolved_by="path",
        source=str(directory),
    )
    return model.replace(identity=identity)


def export_model(model, directory, *, format="onnx"):
    """Export the network to an inference format and prove it agrees.

    The training artifact and the deployment artifact need not be the same
    thing: a model trained in PyTorch can be served by ONNX Runtime.  The
    export is accepted only after the exported graph reproduces the source
    model on a probe batch.

    Parameters
    ----------
    model : ModelArtifact
        Trained or loaded model [-].
    directory : str or path-like
        Output directory; must not exist or be empty [-].
    format : str, optional
        Target format; ``"onnx"`` is the only one implemented [-].

    Returns
    -------
    dict
        Export record written to ``export.json``: format, opset, parity
        difference and tolerance, runtimes, and the source model identity [-].

    Raises
    ------
    ModelContractError
        When the backend cannot export to ``format`` or the parity check fails.
    BackendUnavailableError
        When the exporter or runtime is not installed.

    Output semantics
    ----------------
    The exported graph maps *normalised* inputs to *normalised* outputs;
    ``normalization.npz`` is written beside it and must be applied around it,
    exactly as :func:`predict` does.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [vaft669] VEST-Tokamak/vaft#669, section 12: training and inference
       artifacts are separate, with numerical validation between them.
    """
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ModelContractError(f"{directory} is not empty")
    directory.mkdir(parents=True, exist_ok=True)
    backend = get_backend(model.model_spec.backend)
    output_shape = model.target_shape if model.supervised else model.input_shape
    record = backend.export(
        model.model_spec, model.state, model.input_shape, output_shape, directory / f"model.{format}", format
    )
    (directory / NORMALIZATION).write_bytes(_npz(model.normalization))
    record = {**record, "source": model.resolved_identity().to_dict()}
    (directory / "export.json").write_text(json.dumps(_jsonable(record), indent=2, sort_keys=True) + "\n")
    return record
