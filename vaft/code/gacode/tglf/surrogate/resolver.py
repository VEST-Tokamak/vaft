"""Find a TGLF-NN model artifact without vendoring one, and without the network.

Issue #553 section 5 forbids committing model weights to VAFT, and section 10 requires
``import vaft`` to stay offline-safe.  Both are satisfied the same way: VAFT stores no
weights and resolves them from wherever the user already has them, reporting every
place it looked when it finds nothing.

Resolution order, most explicit first::

    1. a path the caller passed
    2. a configured local model directory
    3. $TURBULENTTRANSPORTHOME/models/<name>
    4. a TurbulentTransport.jl checkout in the user's Julia depot

The remote steps of section 6 -- download, SHA-256 against a pinned manifest, a
persistent cache -- are a later increment.  The hashes computed here are of the bytes
actually resolved, which is integrity metadata for a run, not verification against an
upstream manifest; :attr:`ModelIdentity.sha256` is what a future manifest would be
checked against.

Environment variable naming follows VAFT's existing external-code convention
(``GACODEHOME``, ``EFITHOME``, ``CHEASEHOME``) rather than the ``*_ROOT`` spelling the
issue sketched; the ``_ROOT`` form is accepted as a compatibility alias exactly as
``vaft.code.gacode`` accepts ``GACODE_ROOT``.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from ._types import (
    ModelContractError,
    ModelIdentity,
    ModelResolutionError,
    SurrogateMetadata,
)

#: Root of a TurbulentTransport.jl checkout, in VAFT's ``<CODE>HOME`` convention.
MODELS_HOME_ENV = "TURBULENTTRANSPORTHOME"

#: Accepted for the spelling issue #553 section 6 sketched.
MODELS_COMPATIBILITY_ENVS = ("TURBULENTTRANSPORT_ROOT",)

#: Files that must sit beside an ONNX ensemble for it to be usable.
METADATA_FILES = ("xnames.txt", "ynames.txt", "xm.txt", "xsigma.txt", "ym.txt", "ysigma.txt")

#: Device tokens the upstream naming convention uses.
KNOWN_DEVICES = frozenset(
    {
        "d3d", "d3dedge", "d3dnearedge", "mastu", "nstx", "ukstep",
        "iter", "fpp", "stfpp",
    }
)

__all__ = [
    "KNOWN_DEVICES",
    "METADATA_FILES",
    "MODELS_COMPATIBILITY_ENVS",
    "MODELS_HOME_ENV",
    "available_models",
    "load_metadata",
    "parse_model_name",
    "resolve_model",
]


def parse_model_name(name: str) -> dict:
    """What the upstream family name states about the model's physics.

    ``sat3_em_d3d+mastu+nstx_azf-1`` is the saturation rule, the electromagnetic
    identity, the training devices and a tag, in that order.  Everything this cannot
    place is returned under ``tags`` verbatim rather than guessed at.

    Returns a mapping with ``sat_rule``, ``electromagnetic``, ``devices`` and ``tags``;
    the first two are ``None`` when the name does not spell them.
    """
    parts = [part for part in str(name).split("_") if part]
    sat_rule: Optional[int] = None
    electromagnetic: Optional[bool] = None
    devices: list[str] = []
    tags: list[str] = []

    index = 0
    if parts and parts[0].startswith("sat") and parts[0][3:4].isdigit():
        sat_rule = int(parts[0][3])
        remainder = parts[0][4:]
        if remainder:
            tags.append(remainder)
        index = 1
    if index < len(parts) and parts[index] in ("em", "es"):
        electromagnetic = parts[index] == "em"
        index += 1
    while index < len(parts):
        pieces = parts[index].split("+")
        if not all(piece.lower() in KNOWN_DEVICES for piece in pieces):
            break
        devices.extend(pieces)
        index += 1
    tags.extend(parts[index:])
    return {
        "sat_rule": sat_rule,
        "electromagnetic": electromagnetic,
        "devices": tuple(devices),
        "tags": tuple(tags),
    }


def _julia_model_roots(env: Optional[Mapping[str, str]] = None) -> list[Path]:
    """``models`` directories of every TurbulentTransport.jl copy in the Julia depot.

    Read, never installed: finding an existing checkout is not the automatic clone
    issue #553 section 10 forbids.  More than one is returned so the caller can refuse
    the ambiguity rather than silently pick a version.

    Takes *env* rather than reading the process environment directly, so a caller that
    passes one is genuinely isolated from the machine -- a depot found behind its back
    would make ``env=`` a half-truth.
    """
    environment = os.environ if env is None else env
    depots = environment.get("JULIA_DEPOT_PATH")
    if depots:
        candidates = [Path(part).expanduser() for part in depots.split(os.pathsep) if part]
    else:
        # Julia's default depot is ``$HOME/.julia``, so the home directory is read from
        # the same environment as everything else. A supplied environment with neither
        # variable therefore has no depot -- Julia would not find one either, and
        # reaching past it to the real home is what made ``env=`` a half-truth.
        home = environment.get("HOME") or (str(Path.home()) if env is None else None)
        candidates = [Path(home).expanduser() / ".julia"] if home else []
    roots: list[Path] = []
    for depot in candidates:
        package = depot / "packages" / "TurbulentTransport"
        if not package.is_dir():
            continue
        roots.extend(sorted(child / "models" for child in package.iterdir() if (child / "models").is_dir()))
    return roots


def _package_version(models_dir: Path) -> Optional[str]:
    """The ``version = "x.y.z"`` line of the checkout owning *models_dir*."""
    project = models_dir.parent / "Project.toml"
    if not project.is_file():
        return None
    for line in project.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("version"):
            _, _, raw = stripped.partition("=")
            return raw.strip().strip('"') or None
    return None


def _is_model_directory(path: Path) -> bool:
    return path.is_dir() and all((path / name).is_file() for name in METADATA_FILES)


def _search_roots(model_dir: Optional[str | Path], env: Optional[Mapping[str, str]]) -> list[tuple[Path, str]]:
    """Every directory a named model may live under, with what put it there."""
    environment = os.environ if env is None else env
    roots: list[tuple[Path, str]] = []
    if model_dir is not None:
        roots.append((Path(model_dir).expanduser(), "model directory"))
    for variable in (MODELS_HOME_ENV, *MODELS_COMPATIBILITY_ENVS):
        value = environment.get(variable)
        if value:
            roots.append((Path(value).expanduser() / "models", variable))
    for root in _julia_model_roots(env):
        roots.append((root, "julia depot"))
    return roots


def available_models(
    *, model_dir: Optional[str | Path] = None, env: Optional[Mapping[str, str]] = None
) -> dict[str, Path]:
    """Model families found right now, name to the directory that wins the search.

    Search precedence matches :func:`resolve_model`, so the directory shown is the one
    that would be used -- with one difference worth knowing: a name held by two roots
    with *differing* bytes is listed here but refused there, because resolving it would
    mean choosing a version. Listing is not a promise that resolution succeeds.
    """
    found: dict[str, Path] = {}
    for root, _ in _search_roots(model_dir, env):
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if child.name not in found and _is_model_directory(child):
                found[child.name] = child
    return found


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_digests(directory: Path) -> dict[str, str]:
    """SHA-256 of everything that defines the model's behaviour.

    The graphs are only half of it: ``xnames`` decides which TGLF key feeds which
    channel and ``xm``/``xsigma`` decide what the network is actually shown, so a copy
    with the same weights and different moments is a different model. Upstream's
    ``models/SEMVER`` calls a patch bump "training data / space ... updated", which is
    exactly a change to these files and to nothing else.
    """
    members = {path.name: _digest(path) for path in sorted(directory.glob("*.onnx"))}
    members.update(
        {name: _digest(directory / name) for name in METADATA_FILES}
    )
    return members


def resolve_model(
    model: str | Path,
    *,
    model_dir: Optional[str | Path] = None,
    env: Optional[Mapping[str, str]] = None,
) -> ModelIdentity:
    """Locate *model* and record what was found.

    *model* is either a path to an ensemble directory or an upstream family name.  A
    name that matches nothing raises :class:`ModelResolutionError` naming every
    directory searched, because an offline failure whose message does not say where to
    put the model is not actionable.
    """
    candidate = Path(model).expanduser()
    searched: list[str] = []
    # A bare family name is a name, even when the working directory happens to hold a
    # folder called that; only a spelled-out path, or a directory that really is an
    # ensemble, takes the path branch.
    spelled_out = os.sep in str(model) or str(model).startswith("~")
    if spelled_out or _is_model_directory(candidate):
        if not _is_model_directory(candidate):
            missing = [name for name in METADATA_FILES if not (candidate / name).is_file()]
            raise ModelResolutionError(
                f"{candidate} is not a usable ONNX model directory: it is missing "
                f"{', '.join(missing) if missing else 'its normalisation files'}. "
                f"An ensemble directory holds *.onnx beside {', '.join(METADATA_FILES)}."
            )
        return _identify(candidate, "explicit path", env=env)

    name = str(model)
    matches: list[tuple[Path, str]] = []
    for root, origin in _search_roots(model_dir, env):
        searched.append(str(root))
        target = root / name
        if _is_model_directory(target):
            matches.append((target, origin))
    if matches:
        return _disambiguate(name, matches, env=env)

    hint = (
        f"Set ${MODELS_HOME_ENV} to a TurbulentTransport.jl checkout, pass "
        f"model_dir=, or pass an explicit directory path."
    )
    known = sorted(available_models(model_dir=model_dir, env=env))
    close = [other for other in known if name.lower() in other.lower()]
    suggestion = f" Resolvable names include: {', '.join(close or known[:6])}." if known else ""
    raise ModelResolutionError(
        f"no ONNX model named {name!r} was found. Searched: "
        f"{', '.join(searched) if searched else '(no search root configured)'}. "
        f"{hint}{suggestion}"
    )


def _describe(directory: Path) -> str:
    version = _package_version(directory.parent)
    return f"{directory}" + (f" (TurbulentTransport {version})" if version else "")


def _disambiguate(
    name: str, matches: Sequence[tuple[Path, str]], *, env: Optional[Mapping[str, str]]
) -> ModelIdentity:
    """Resolve *name* when more than one search root holds it.

    A Julia depot routinely carries several installed versions of the same package, and
    upstream's own ``models/SEMVER`` says a version bump can mean different training
    data or different TGLF settings behind an unchanged family name. Picking the first
    match would therefore silently choose a model.

    Copies whose bytes agree are the same model and the extras are recorded; copies
    whose bytes differ are different models sharing a name, and that is refused with
    both locations named so the caller can choose one.
    """
    primary, origin = matches[0]
    identity = _identify(primary, origin, env=env)
    others = []
    for directory, _ in matches[1:]:
        if _artifact_digests(directory) != dict(identity.sha256):
            raise ModelResolutionError(
                f"{name!r} resolves to more than one model and the copies are not the "
                f"same bytes: {_describe(primary)} and {_describe(directory)}. Upstream "
                f"versions the networks behind a stable family name, so choosing for "
                f"you would pick a model silently. Pass model_dir= or an explicit "
                f"directory path to say which one."
            )
        others.append(str(directory))
    return identity if not others else replace(identity, alternatives=tuple(others))


def _identify(directory: Path, resolved_by: str, *, env: Optional[Mapping[str, str]]) -> ModelIdentity:
    members = sorted(path for path in directory.glob("*.onnx"))
    if not members:
        raise ModelResolutionError(
            f"{directory} carries the normalisation files but no *.onnx ensemble "
            f"member. Upstream ships some families as Julia .bson only; those cannot "
            f"be read from Python."
        )
    physics = parse_model_name(directory.name)
    # Read wherever a checkout's Project.toml sits, not only in a depot: the route the
    # README recommends is $TURBULENTTRANSPORTHOME, and it must not be the one that
    # drops the upstream version from a run's provenance.
    version = _package_version(directory.parent)
    return ModelIdentity(
        name=directory.name,
        directory=str(directory),
        members=tuple(path.name for path in members),
        sha256=_artifact_digests(directory),
        resolved_by=resolved_by,
        sat_rule=physics["sat_rule"],
        electromagnetic=physics["electromagnetic"],
        devices=physics["devices"],
        tags=physics["tags"],
        upstream_version=version,
    )


def load_metadata(identity: ModelIdentity) -> SurrogateMetadata:
    """Read the normalisation contract shipped beside the ensemble."""
    directory = Path(identity.directory)

    def words(name: str) -> list[str]:
        return (directory / name).read_text(encoding="utf-8").split()

    def numbers(name: str) -> np.ndarray:
        return np.asarray(words(name), dtype=float)

    staged = [word for word in words("xnames.txt") if word.startswith("OUT_")]
    if staged:
        raise ModelContractError(
            f"{identity.name} takes {', '.join(staged)} as inputs: it is a two-stage "
            f"correction network over another model's predictions, not a drop-in "
            f"surrogate. It cannot be evaluated from a TGLF input alone, and its "
            f"normalisation files describe the correction rather than the fluxes."
        )

    try:
        return SurrogateMetadata(
            xnames=tuple(words("xnames.txt")),
            ynames=tuple(words("ynames.txt")),
            xm=numbers("xm.txt"),
            xsigma=numbers("xsigma.txt"),
            ym=numbers("ym.txt"),
            ysigma=numbers("ysigma.txt"),
        )
    except ValueError as exc:
        raise ModelContractError(
            f"{identity.name}: a normalisation file in {directory} does not parse as "
            f"numbers ({exc})"
        ) from exc
