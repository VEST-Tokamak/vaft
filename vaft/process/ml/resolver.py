"""Resolve published ``vaft-nn`` models to verified local artifacts (#669 section 13).

The registry is a checkout of ``VEST-Tokamak/vaft-nn``: for each model,
``models/<name>/releases.yaml`` lists the versions, their lifecycle status and
the SHA-256 of each version's manifest, and maps stage aliases to versions;
``models/<name>/versions/<version>/manifest.json`` is the reviewed manifest.
The artifact files themselves live in a local cache,
``<cache>/<name>/<version>/``, filled from the version's GitHub Release.

Fetching release assets into the cache is not implemented yet; download them
with ``gh release download <name>-v<version> -R VEST-Tokamak/vaft-nn -D <dir>``.
"""

from __future__ import annotations

import os
from pathlib import Path

from ._provenance import sha256_file
from ._types import ModelContractError, ModelIdentity, ModelResolutionError

__all__ = ["load_model", "resolve_model"]

REGISTRY_ENV = "VAFT_NN_HOME"
CACHE_ENV = "VAFT_NN_CACHE"
_DEFAULT_CACHE = Path("~/.cache/vaft-nn")


def _registry_root(registry, env) -> Path:
    if registry is not None:
        return Path(registry).expanduser()
    value = env.get(REGISTRY_ENV)
    if not value:
        raise ModelResolutionError(
            f"no model registry: pass registry= or set ${REGISTRY_ENV} to a checkout of "
            "VEST-Tokamak/vaft-nn"
        )
    return Path(value).expanduser()


def _cache_root(cache, env) -> Path:
    if cache is not None:
        return Path(cache).expanduser()
    return Path(env.get(CACHE_ENV) or _DEFAULT_CACHE).expanduser()


def resolve_model(name, *, version=None, stage=None, registry=None, cache=None, env=None):
    """Resolve a published model to one exact, hash-verified local version.

    Parameters
    ----------
    name : str
        Registry model name, e.g. ``"ire_detector"`` [-].
    version : str or None, optional
        Exact version; give this or ``stage`` [-].
    stage : str or None, optional
        Lifecycle alias such as ``"production"``, resolved to an exact version
        before anything is loaded [-].
    registry : str or path-like or None, optional
        ``vaft-nn`` checkout; ``$VAFT_NN_HOME`` when ``None`` [-].
    cache : str or path-like or None, optional
        Artifact cache root; ``$VAFT_NN_CACHE``, else ``~/.cache/vaft-nn`` [-].
    env : mapping or None, optional
        Environment to read the two variables from; ``os.environ`` when ``None`` [-].

    Returns
    -------
    ModelIdentity
        Name, exact version, manifest SHA-256, requested stage, lifecycle status
        and the verified artifact directory (``source``) [-].

    Raises
    ------
    ModelResolutionError
        When the registry, model, version, stage, manifest or a cached file is
        missing, or a hash differs from the one the registry pins.
    ModelContractError
        When both or neither of ``version`` and ``stage`` are given.

    Processing steps
    ----------------
    1. Read ``models/<name>/releases.yaml`` from the registry.
    2. Map ``stage`` to its version, or take ``version``; refuse a stage that
       points at a deprecated version.
    3. Check the registry manifest's SHA-256 against the pinned one.
    4. Check every file the manifest lists in ``<cache>/<name>/<version>/``
       for presence, size and SHA-256.

    Convention
    ----------
    A stage alias is mutable and a version is not.  The returned identity
    always carries the exact version and manifest hash; ``stage`` only records
    what was asked for.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    The cache is not filled automatically; fetching release assets from the
    private ``vaft-nn`` repository is tracked in #669 (phase 6).

    Provenance
    ----------
    .. [vaft669] VEST-Tokamak/vaft#669, sections 9 and 13: lifecycle aliases
       resolve to immutable artifacts, and resolution verifies the hash.
    """
    import json

    import yaml

    if (version is None) == (stage is None):
        raise ModelContractError("give exactly one of version= or stage=")
    env = os.environ if env is None else env
    root = _registry_root(registry, env)
    model_dir = root / "models" / name
    index_path = model_dir / "releases.yaml"
    if not index_path.is_file():
        known = sorted(p.name for p in (root / "models").iterdir() if p.is_dir()) if (root / "models").is_dir() else []
        raise ModelResolutionError(f"model {name!r} is not in the registry at {root}; known: {known}")
    index = yaml.safe_load(index_path.read_text(encoding="utf-8")) or {}
    entries = {str(e["version"]): e for e in index.get("versions", [])}
    if stage is not None:
        stages = index.get("stages") or {}
        if stage not in stages:
            raise ModelResolutionError(f"model {name!r} has no stage {stage!r}; stages: {sorted(stages)}")
        version = str(stages[stage])
    version = str(version)
    if version not in entries:
        raise ModelResolutionError(f"model {name!r} has no version {version}; versions: {sorted(entries)}")
    entry = entries[version]
    if stage is not None and entry.get("status") == "deprecated":
        raise ModelResolutionError(f"stage {stage!r} of {name!r} points at deprecated version {version}")
    manifest_path = model_dir / "versions" / version / "manifest.json"
    if not manifest_path.is_file():
        raise ModelResolutionError(f"registry has no manifest at {manifest_path}")
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != entry.get("manifest_sha256"):
        raise ModelResolutionError(
            f"{manifest_path} has sha256 {manifest_sha}, but releases.yaml pins {entry.get('manifest_sha256')}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model") != name or str(manifest.get("version")) != version:
        raise ModelResolutionError(
            f"{manifest_path} describes {manifest.get('model')} {manifest.get('version')}, not {name} {version}"
        )
    directory = _cache_root(cache, env) / name / version
    problems = []
    for filename, record in manifest["files"].items():
        path = directory / filename
        if not path.is_file():
            problems.append(f"{filename}: missing")
        elif path.stat().st_size != record["size"] or sha256_file(path) != record["sha256"]:
            problems.append(f"{filename}: differs from the registry manifest")
    if problems:
        tag = entry.get("release") or f"{name}-v{version}"
        raise ModelResolutionError(
            f"cached artifact {directory} is not {name} {version}: " + "; ".join(problems)
            + f". Fetch it with: gh release download {tag} -R VEST-Tokamak/vaft-nn -D {directory}"
        )
    weights = next(k for k in manifest["files"] if k.startswith("weights"))
    return ModelIdentity(
        name=name,
        state_sha256=manifest["files"][weights]["sha256"],
        version=version,
        manifest_sha256=manifest_sha,
        stage=stage,
        status=entry.get("status"),
        resolved_by="registry",
        source=str(directory),
    )


def load_model(name, *, version=None, stage=None, registry=None, cache=None, env=None):
    """Resolve a published model and load it, ready for :func:`predict`.

    Parameters
    ----------
    name : str
        Registry model name [-].
    version : str or None, optional
        Exact version; give this or ``stage`` [-].
    stage : str or None, optional
        Lifecycle alias such as ``"production"`` [-].
    registry : str or path-like or None, optional
        ``vaft-nn`` checkout; ``$VAFT_NN_HOME`` when ``None`` [-].
    cache : str or path-like or None, optional
        Artifact cache root; ``$VAFT_NN_CACHE``, else ``~/.cache/vaft-nn`` [-].
    env : mapping or None, optional
        Environment to read the two variables from [-].

    Returns
    -------
    ModelArtifact
        The model, carrying the identity :func:`resolve_model` returned [-].

    Applicability
    -------------
    Machine-independent.
    """
    from .artifact import load_model_artifact

    identity = resolve_model(name, version=version, stage=stage, registry=registry, cache=cache, env=env)
    root = _registry_root(registry, os.environ if env is None else env)
    manifest = root / "models" / name / "versions" / identity.version / "manifest.json"
    model = load_model_artifact(identity.source, manifest=manifest, verify=False)
    if model.state_sha256 != identity.state_sha256:
        raise ModelResolutionError(f"weights of {name} {identity.version} changed after verification")
    return model.replace(identity=identity)
