"""Several fitted models released as one version (#669, #973 section 24).

A bundle directory is flat -- every file is a GitHub Release asset -- so each
member's files carry its role as a prefix: ``stage1.manifest.json``,
``stage1.weights.pt``, ``stage1.normalization.npz``.  The bundle's
``manifest.json`` pins every file and each member manifest's SHA-256, so its
own SHA-256 identifies the whole release, exactly as for a single model.
"""

from __future__ import annotations

import json
from pathlib import Path

from ._provenance import sha256_bytes, sha256_file
from ._types import ModelBundle, ModelContractError, ModelIdentity, ModelResolutionError, _jsonable
from .artifact import (
    MANIFEST,
    MANIFEST_SCHEMA_VERSION,
    _check_name_version,
    _empty_directory,
    _load_artifact,
    _verify_files,
    _write_artifact,
)

__all__ = ["load_model_bundle", "save_model_bundle"]


def save_model_bundle(bundle, directory, *, version, model_card=None, applicability=None):
    """Write a multi-model bundle as one immutable, hash-pinned release.

    Parameters
    ----------
    bundle : ModelBundle
        Members by role and the composition record [-].
    directory : str or path-like
        Target directory; must not exist or be empty [-].
    version : str
        Semantic version shared by the bundle and every member [-].
    model_card : mapping or None, optional
        Intended use, population, validation evidence and limits [-].
    applicability : mapping or None, optional
        Input domain the bundle was trained and validated on [-].

    Returns
    -------
    ModelBundle
        The bundle carrying its saved identity; every member carries its own
        identity and the bundle's [-].

    Raises
    ------
    ModelContractError
        On an invalid bundle or member name or version, or a non-empty
        directory.

    Processing steps
    ----------------
    1. Write each member as an ordinary artifact whose file names are prefixed
       by its role.
    2. Write ``manifest.json`` with ``kind: "bundle"``, every file's SHA-256
       and size, each member manifest's SHA-256 and the composition record.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [vaft973] VEST-Tokamak/vaft#973, section 24: one logical model release
       may contain two fitted stage artifacts.
    """
    if not isinstance(bundle, ModelBundle):
        raise ModelContractError("bundle must be a ModelBundle")
    _check_name_version(bundle.name, version)
    for member in bundle.members.values():
        _check_name_version(member.name, version)
    directory = _empty_directory(directory)
    members = {}
    member_raw = {}
    for role, member in bundle.members.items():
        raw = _write_artifact(member, directory, str(version), prefix=f"{role}.")
        member_raw[role] = raw
        members[role] = {"manifest": f"{role}.{MANIFEST}", "manifest_sha256": sha256_bytes(raw)}
    files = {
        path.name: {"sha256": sha256_file(path), "size": path.stat().st_size}
        for path in sorted(directory.iterdir())
        if path.name != MANIFEST
    }
    record = _jsonable({
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": "bundle",
        "model": bundle.name,
        "version": str(version),
        "members": members,
        "composition": dict(bundle.composition),
        "files": files,
        "model_card": dict(model_card) if model_card is not None else None,
        "applicability": dict(applicability) if applicability is not None else None,
        "compatibility": {
            "vaft": ">=" + str(next(iter(bundle.members.values())).training.get("vaft_version", "0"))
        },
    })
    raw = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")
    (directory / MANIFEST).write_bytes(raw)
    identity = ModelIdentity(
        name=bundle.name,
        state_sha256=sha256_bytes(raw),
        version=str(version),
        manifest_sha256=sha256_bytes(raw),
        resolved_by="saved",
        source=str(directory),
    )
    saved_members = {
        role: member.replace(identity=_member_identity(member, role, sha256_bytes(member_raw[role]), identity))
        for role, member in bundle.members.items()
    }
    return bundle.replace(members=saved_members, identity=identity)


def _member_identity(member, role, manifest_sha256, bundle_identity):
    return ModelIdentity(
        name=member.name,
        state_sha256=member.state_sha256,
        version=bundle_identity.version,
        manifest_sha256=manifest_sha256,
        stage=bundle_identity.stage,
        status=bundle_identity.status,
        resolved_by=bundle_identity.resolved_by,
        source=bundle_identity.source,
        bundle=f"{bundle_identity.name}@{bundle_identity.version}#{role}",
        bundle_manifest_sha256=bundle_identity.manifest_sha256,
    )


def load_model_bundle(directory, *, manifest=None, verify=True):
    """Load a bundle directory, verifying every file and member manifest.

    Parameters
    ----------
    directory : str or path-like
        Directory holding the bundle's files [-].
    manifest : str or path-like or None, optional
        Bundle manifest to trust; ``directory/manifest.json`` when ``None``.
        The resolver passes the registry's reviewed copy [-].
    verify : bool, optional
        Check every file's SHA-256 and size first [-].

    Returns
    -------
    ModelBundle
        The members, each with an identity naming the bundle, and the
        composition record [-].

    Raises
    ------
    ModelResolutionError
        When a file is missing or differs, or a member manifest differs from
        the hash the bundle pins.
    ModelContractError
        When the manifest is not a bundle manifest this VAFT reads.

    Applicability
    -------------
    Machine-independent.
    """
    directory = Path(directory)
    manifest_path = Path(manifest) if manifest is not None else directory / MANIFEST
    if not manifest_path.is_file():
        raise ModelResolutionError(f"no manifest at {manifest_path}")
    raw = manifest_path.read_bytes()
    record = json.loads(raw)
    if record.get("schema_version") != MANIFEST_SCHEMA_VERSION or record.get("kind") != "bundle":
        raise ModelContractError(f"{manifest_path} is not a schema-{MANIFEST_SCHEMA_VERSION} bundle manifest")
    if verify:
        _verify_files(record, directory)
    identity = ModelIdentity(
        name=record["model"],
        state_sha256=sha256_bytes(raw),
        version=record["version"],
        manifest_sha256=sha256_bytes(raw),
        resolved_by="path",
        source=str(directory),
    )
    members = {}
    for role, entry in record["members"].items():
        member_raw = (directory / entry["manifest"]).read_bytes()
        if sha256_bytes(member_raw) != entry["manifest_sha256"]:
            raise ModelResolutionError(f"member {role!r} manifest differs from the hash the bundle pins")
        member = _load_artifact(directory, json.loads(member_raw), verify=verify, prefix=f"{role}.")
        members[role] = member.replace(identity=_member_identity(member, role, entry["manifest_sha256"], identity))
    return ModelBundle(
        name=record["model"], members=members, composition=record.get("composition") or {}, identity=identity
    )


def _with_identity(bundle: ModelBundle, identity: ModelIdentity) -> ModelBundle:
    """Re-stamp a loaded bundle and its members with a resolver identity."""
    members = {
        role: member.replace(identity=_member_identity(member, role, member.identity.manifest_sha256, identity))
        for role, member in bundle.members.items()
    }
    return bundle.replace(members=members, identity=identity)
