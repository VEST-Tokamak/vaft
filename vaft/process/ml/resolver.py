"""Resolve published ``vaft-nn`` models to verified local artifacts (#669 section 13).

``vaft-nn`` is an external dependency of VAFT's ML layer, configured like the
external codes: ``$VAFT_NN_HOME`` names a checkout of
``VEST-Tokamak/vaft-nn``.  For each model, ``models/<name>/releases.yaml``
lists the versions, their lifecycle status and the SHA-256 of each version's
manifest, and maps stage aliases to versions;
``models/<name>/versions/<version>/manifest.json`` is the reviewed manifest.

The artifact files are GitHub Release assets, copied into a local cache
(``$VAFT_NN_CACHE``, else the platform cache directory) by :func:`fetch_model`.
The repository is private, so fetching goes through the GitHub CLI and the
login ``gh auth login`` already holds; VAFT never stores or passes a token.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from ._provenance import sha256_bytes, sha256_file
from ._types import ModelContractError, ModelIdentity, ModelResolutionError

__all__ = ["fetch_model", "load_model", "resolve_model"]

REGISTRY_ENV = "VAFT_NN_HOME"
CACHE_ENV = "VAFT_NN_CACHE"
DEFAULT_REPOSITORY = "VEST-Tokamak/vaft-nn"
_GITHUB_REMOTE = re.compile(r"github\.com[:/](?P<repo>[^/\s]+/[^/\s]+?)(?:\.git)?/?$")


def _registry_root(registry, env) -> Path:
    if registry is not None:
        return Path(registry).expanduser()
    value = env.get(REGISTRY_ENV)
    if not value:
        raise ModelResolutionError(
            f"vaft-nn model registry is not configured: set ${REGISTRY_ENV} to a checkout of "
            f"{DEFAULT_REPOSITORY} containing models/ (git clone git@github.com:{DEFAULT_REPOSITORY}.git), "
            "or pass registry=."
        )
    return Path(value).expanduser()


def default_cache_root(env=None) -> Path:
    """``$VAFT_NN_CACHE``, else the platform's user cache directory + ``vaft-nn``."""
    env = os.environ if env is None else env
    if env.get(CACHE_ENV):
        return Path(env[CACHE_ENV]).expanduser()
    if os.name == "nt":
        return Path(env.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "vaft-nn"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "vaft-nn"
    return Path(env.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "vaft-nn"


def _cache_root(cache, env) -> Path:
    return Path(cache).expanduser() if cache is not None else default_cache_root(env)


def _registry_entry(name, version, stage, registry, env):
    """Registry-side resolution: exact version, pinned manifest, its hash."""
    import yaml

    if (version is None) == (stage is None):
        raise ModelContractError("give exactly one of version= or stage=")
    root = _registry_root(registry, env)
    model_dir = root / "models" / name
    index_path = model_dir / "releases.yaml"
    if not index_path.is_file():
        models = root / "models"
        known = sorted(p.name for p in models.iterdir() if p.is_dir()) if models.is_dir() else []
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
    raw = manifest_path.read_bytes()
    manifest_sha = sha256_bytes(raw)
    if manifest_sha != entry.get("manifest_sha256"):
        raise ModelResolutionError(
            f"{manifest_path} has sha256 {manifest_sha}, but releases.yaml pins {entry.get('manifest_sha256')}"
        )
    manifest = json.loads(raw)
    if manifest.get("model") != name or str(manifest.get("version")) != version:
        raise ModelResolutionError(
            f"{manifest_path} describes {manifest.get('model')} {manifest.get('version')}, not {name} {version}"
        )
    return root, version, entry, manifest_path, manifest, manifest_sha


def _cache_problems(directory: Path, manifest: dict) -> list[str]:
    problems = []
    for filename, record in manifest["files"].items():
        path = directory / filename
        if not path.is_file():
            problems.append(f"{filename}: missing")
        elif path.stat().st_size != record["size"] or sha256_file(path) != record["sha256"]:
            problems.append(f"{filename}: differs from the registry manifest")
    return problems


def _identity(name, version, stage, entry, manifest, manifest_sha, directory) -> ModelIdentity:
    if manifest.get("kind", "model") == "bundle":
        state = manifest_sha
    else:
        weights = next(k for k in manifest["files"] if k.startswith("weights"))
        state = manifest["files"][weights]["sha256"]
    return ModelIdentity(
        name=name,
        state_sha256=state,
        version=version,
        manifest_sha256=manifest_sha,
        stage=stage,
        status=entry.get("status"),
        resolved_by="registry",
        source=str(directory),
    )


def resolve_model(name, *, version=None, stage=None, registry=None, cache=None, env=None):
    """Resolve a published model to one exact, hash-verified local version.

    Parameters
    ----------
    name : str
        Registry model name, e.g. ``"ire_ae_vaft_v1"`` [-].
    version : str or None, optional
        Exact version; give this or ``stage`` [-].
    stage : str or None, optional
        Lifecycle alias such as ``"production"``, resolved to an exact version
        before anything is loaded [-].
    registry : str or path-like or None, optional
        ``vaft-nn`` checkout; ``$VAFT_NN_HOME`` when ``None`` [-].
    cache : str or path-like or None, optional
        Artifact cache root; ``$VAFT_NN_CACHE``, else the platform cache
        directory [-].
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

    Provenance
    ----------
    .. [vaft669] VEST-Tokamak/vaft#669, sections 9 and 13: lifecycle aliases
       resolve to immutable artifacts, and resolution verifies the hash.
    """
    env = os.environ if env is None else env
    _, version, entry, _, manifest, manifest_sha = _registry_entry(name, version, stage, registry, env)
    directory = _cache_root(cache, env) / name / version
    problems = _cache_problems(directory, manifest)
    if problems:
        raise ModelResolutionError(
            f"cached artifact {directory} is not {name} {version}: " + "; ".join(problems)
            + f". Fetch it with vaft.process.ml.fetch_model({name!r}, version={version!r}), or: "
            + f"gh release download {entry.get('release') or f'{name}-v{version}'} -R {DEFAULT_REPOSITORY} -D {directory}"
        )
    return _identity(name, version, stage, entry, manifest, manifest_sha, directory)


def _repository(root: Path) -> str:
    """``owner/repo`` of the registry checkout's GitHub origin, else the default."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return DEFAULT_REPOSITORY
    match = _GITHUB_REMOTE.search(completed.stdout.strip()) if completed.returncode == 0 else None
    return match.group("repo") if match else DEFAULT_REPOSITORY


def _download_release(repository: str, tag: str, destination: Path) -> None:
    """Download every asset of one release with the GitHub CLI's own login."""
    gh = shutil.which("gh")
    if gh is None:
        raise ModelResolutionError(
            "fetching from the private vaft-nn repository needs the GitHub CLI: install gh "
            "(https://cli.github.com) and run `gh auth login` once"
        )
    completed = subprocess.run(
        [gh, "release", "download", tag, "-R", repository, "-D", str(destination)],
        capture_output=True, text=True, timeout=1800, check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()[-1:] or ["no output"]
        raise ModelResolutionError(
            f"gh release download {tag} -R {repository} failed: {detail[0]}. "
            "Check `gh auth status` and that your account can read the repository."
        )


def _swap_into_place(staging: Path, target: Path) -> None:
    """Replace ``target`` with ``staging``, restoring the old copy if that fails."""
    retired = None
    try:
        if target.exists():
            retired = Path(tempfile.mkdtemp(prefix=f".{target.name}-old-", dir=target.parent))
            retired.rmdir()
            os.replace(target, retired)
        os.replace(staging, target)
    except OSError as error:
        if retired is not None and retired.exists():
            if not target.exists():
                os.replace(retired, target)
            else:
                shutil.rmtree(retired, ignore_errors=True)
        raise ModelResolutionError(f"could not move the verified download into {target}: {error}") from error
    if retired is not None:
        shutil.rmtree(retired, ignore_errors=True)


def fetch_model(name, *, version=None, stage=None, registry=None, cache=None, env=None):
    """Download a published model's release assets into the cache, verified.

    The registry checkout says *which* files a version is (their hashes); the
    GitHub Release holds the bytes.  This copies the bytes into the cache and
    accepts them only if they are exactly the files the registry pins.

    Parameters
    ----------
    name : str
        Registry model name [-].
    version : str or None, optional
        Exact version; give this or ``stage`` [-].
    stage : str or None, optional
        Lifecycle alias, resolved to an exact version first [-].
    registry : str or path-like or None, optional
        ``vaft-nn`` checkout; ``$VAFT_NN_HOME`` when ``None`` [-].
    cache : str or path-like or None, optional
        Artifact cache root; ``$VAFT_NN_CACHE``, else the platform cache
        directory [-].
    env : mapping or None, optional
        Environment to read the variables from; ``os.environ`` when ``None`` [-].

    Returns
    -------
    ModelIdentity
        The identity :func:`resolve_model` returns for the fetched version [-].

    Raises
    ------
    ModelResolutionError
        When ``gh`` is missing or cannot read the release, or a downloaded
        file is missing or differs from the registry manifest; the cache is
        then left as it was.

    Processing steps
    ----------------
    1. Resolve the exact version and its manifest from the registry.
    2. Download the release ``<name>-v<version>`` (or the tag the registry
       records) with ``gh release download`` into a temporary directory inside
       the cache root; the repository is the registry checkout's GitHub origin.
    3. Verify every file the registry manifest lists, then replace
       ``<cache>/<name>/<version>`` with the verified directory.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Needs the GitHub CLI logged in to an account that can read the private
    repository; VAFT reads no token itself.

    Provenance
    ----------
    .. [vaft669] VEST-Tokamak/vaft#669, sections 10 and 13: weights are release
       artifacts outside git history, verified against the registry on resolution.
    """
    env = os.environ if env is None else env
    root, version, entry, _, manifest, _ = _registry_entry(name, version, stage, registry, env)
    cache_root = _cache_root(cache, env)
    target = cache_root / name / version
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{name}-{version}-", dir=target.parent))
    try:
        _download_release(_repository(root), entry.get("release") or f"{name}-v{version}", staging)
        problems = _cache_problems(staging, manifest)
        if problems:
            raise ModelResolutionError(
                f"release assets of {name} {version} do not match the registry manifest: " + "; ".join(problems)
            )
        # Only files the registry pins enter the cache.
        for extra in [p for p in staging.iterdir() if p.name not in manifest["files"]]:
            if extra.is_dir():
                shutil.rmtree(extra)
            else:
                extra.unlink()
        try:
            _swap_into_place(staging, target)
        except ModelResolutionError:
            # A concurrent fetch of the same version may have won the swap;
            # its copy is as good as ours if it verifies.
            if _cache_problems(target, manifest):
                raise
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return resolve_model(name, version=version, stage=None, registry=root, cache=cache_root, env=env)


def load_model(name, *, version=None, stage=None, registry=None, cache=None, env=None, fetch=False):
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
        Artifact cache root; ``$VAFT_NN_CACHE``, else the platform cache
        directory [-].
    env : mapping or None, optional
        Environment to read the variables from [-].
    fetch : bool, optional
        On a cache miss or mismatch, run :func:`fetch_model` and retry [-].

    Returns
    -------
    ModelArtifact or ModelBundle
        The model (or bundle of models), carrying the identity
        :func:`resolve_model` returned [-].

    Applicability
    -------------
    Machine-independent.
    """
    from .artifact import load_model_artifact
    from .bundle import _with_identity, load_model_bundle

    env = os.environ if env is None else env
    try:
        identity = resolve_model(name, version=version, stage=stage, registry=registry, cache=cache, env=env)
    except ModelResolutionError:
        if not fetch:
            raise
        fetched = fetch_model(name, version=version, stage=stage, registry=registry, cache=cache, env=env)
        identity = ModelIdentity(**{**fetched.to_dict(), "stage": stage})
    root = _registry_root(registry, env)
    manifest = root / "models" / name / "versions" / identity.version / "manifest.json"
    kind = json.loads(manifest.read_bytes()).get("kind", "model")
    # verify=False skips only the size/presence pass resolve_model just made;
    # every byte loaded is still hashed, and the manifest re-read here must be
    # the one whose hash the registry pins.
    if kind == "bundle":
        loaded = load_model_bundle(identity.source, manifest=manifest, verify=False)
    else:
        loaded = load_model_artifact(identity.source, manifest=manifest, verify=False)
    if loaded.identity.manifest_sha256 != identity.manifest_sha256:
        raise ModelResolutionError(f"the registry manifest of {name} {identity.version} changed after verification")
    if kind == "bundle":
        return _with_identity(loaded, identity)
    return loaded.replace(identity=identity)
