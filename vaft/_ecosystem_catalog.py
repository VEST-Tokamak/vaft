"""The dependency and external-code catalog, for the documentation site (#1648).

``python -m vaft._ecosystem_catalog --output docs/_data/ecosystem.yml`` writes
what ``/reference/software-dependencies/`` and ``/reference/external-codes/``
render.  Every fact comes from its owner:

* package names, version constraints and extras from ``pyproject.toml``;
* what each dependency and extra is *for*, and every external-code fact
  that has no other home, from :mod:`vaft._ecosystem`;
* each code's ``{CODE}HOME`` variable from the adapter's own constant (the
  one :func:`vaft.code._executables.executable_from_home` is called with);
* platform support from the installers that exist, never extrapolated from
  VAFT's own platforms.

Nothing is fetched: upstream links and DOIs are navigation targets, recorded
in the committed registry.  The snapshot records a checksum of every file it
describes so ``docs/build.py`` can prove which tree it came from.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vaft import _ecosystem as ecosystem

SCHEMA_VERSION = 1
_GENERATOR = "python -m vaft._ecosystem_catalog --output docs/_data/ecosystem.yml"
_PACKAGE = Path(__file__).resolve().parent
_ROOT = _PACKAGE.parent
PYPROJECT = _ROOT / "pyproject.toml"
_NAME = re.compile(r"^\s*([A-Za-z0-9_.\-]+)")


class EcosystemCatalogError(ValueError):
    """The registry and the repository disagree."""


def _relative(path: str | Path) -> str:
    return Path(path).resolve().relative_to(_ROOT).as_posix()


def requirement_name(requirement: str) -> str:
    """The distribution name of a PEP 508 requirement string."""
    match = _NAME.match(requirement)
    if not match:
        raise EcosystemCatalogError(f"cannot read a package name from {requirement!r}")
    return match.group(1)


def load_project(path: Path = PYPROJECT) -> Mapping[str, Any]:
    try:
        import tomllib
    except ImportError:  # Python 3.10
        import tomli as tomllib  # type: ignore[no-redef]
    return tomllib.loads(path.read_text(encoding="utf-8"))["project"]


def home_variable(spec: str) -> str:
    """``module:CONSTANT`` -> the adapter's variable name; a bare name is a literal."""
    if ":" not in spec:
        return spec
    module, _, constant = spec.partition(":")
    return str(getattr(importlib.import_module(module), constant))


def _module_source(module: str) -> str:
    imported = importlib.import_module(module)
    return _relative(imported.__file__)


def _platforms(code: ecosystem.ExternalCode) -> list[str]:
    """What the installers cover; nothing is claimed for a code VAFT does not install."""
    platforms = []
    names = [Path(path).name for path in code.installers]
    posix = [n for n in names if n.endswith(".sh") and not n.startswith("windows")]
    if any(n == "linux.sh" or (n.startswith("install_") and n.endswith(".sh")) for n in posix):
        platforms.append("Linux")
    if any(n == "macos.sh" or (n.startswith("install_") and n.endswith(".sh")) for n in posix):
        platforms.append("macOS")
    if any(n.endswith(".ps1") or n == "windows.sh" for n in names):
        platforms.append("Windows")
    if code.installation == "python_package":
        platforms.append("wherever the Python package installs")
    return platforms


def _execution(code: ecosystem.ExternalCode) -> list[str]:
    """How VAFT runs it: through vaft.code.execution's backends, in process, or not at all."""
    if code.scheduler_backed:
        return ["local subprocess", "Slurm", "Slurm over ssh"]
    return {"in_process_python": ["in process"], "native_reader": ["reads finished results"]}[code.mode]


def _standardized(entry: ecosystem.Standardized) -> dict:
    kind, _, target = entry.via.partition(":")
    row = {"ids": list(entry.ids), "via": kind, "target": target, "url": ""}
    if kind == "stage":
        from vaft.database.sources import STAGE_REPLICATION

        stage = STAGE_REPLICATION.get(target)
        if stage is None:
            raise EcosystemCatalogError(f"{entry.via} names no STAGE_REPLICATION stage")
        if tuple(stage.ids) != tuple(entry.ids):
            raise EcosystemCatalogError(f"{entry.via} owns {stage.ids}, not {entry.ids}")
        row["url"] = f"/reference/pipeline-graph/#view=publication&focus=stage:{target}"
    elif kind == "mapper":
        module, _, function = target.rpartition(".")
        if not callable(getattr(importlib.import_module(module), function, None)):
            raise EcosystemCatalogError(f"{entry.via} is not a callable")
    else:
        raise EcosystemCatalogError(f"unknown standardized mapping {entry.via!r}")
    return row


def _code(code: ecosystem.ExternalCode) -> dict:
    for path in (*code.installers, *([code.checker] if code.checker else [])):
        if not (_ROOT / path).is_file():
            raise EcosystemCatalogError(f"{code.id}: {path} does not exist")
    return {
        "id": code.id,
        "name": code.name,
        "roles": list(code.roles),
        "adapter": code.adapter,
        "adapter_source": _module_source(code.adapter),
        "mode": code.mode,
        "mode_label": ecosystem.MODE_LABELS[code.mode],
        "scheduler_backed": code.scheduler_backed,
        "home_variable": home_variable(code.home) if code.home else "",
        "installation": code.installation,
        "installation_label": ecosystem.INSTALLATION_LABELS[code.installation],
        "access": code.access,
        "installers": list(code.installers),
        "checker": code.checker,
        "extra": code.extra,
        "maturity": code.maturity,
        "platforms": _platforms(code),
        "execution": _execution(code),
        "native": code.native,
        "standardized": [_standardized(entry) for entry in code.standardized],
        "workflow": [
            {"rule": node, "url": f"/reference/pipeline-graph/#focus={node}"} for node in code.workflow
        ],
        "install_section": code.install_section,
        "links": [
            {"role": link.role, "title": link.title, "url": link.url, "doi": link.doi} for link in code.links
        ],
        "note": code.note,
    }


def ecosystem_snapshot(provenance: Mapping[str, str] | None = None) -> dict:
    """Dependencies and external codes as one deterministic, documentation-ready mapping."""
    project = load_project()
    runtime = [requirement_name(r) for r in project["dependencies"]]
    extras = project.get("optional-dependencies", {})
    missing = sorted(set(runtime) ^ set(ecosystem.RUNTIME_ROLES))
    if missing:
        raise EcosystemCatalogError(f"runtime dependencies and RUNTIME_ROLES differ: {missing}")
    if set(extras) != set(ecosystem.EXTRA_ROLES):
        raise EcosystemCatalogError(
            f"extras and EXTRA_ROLES differ: {sorted(set(extras) ^ set(ecosystem.EXTRA_ROLES))}")
    capabilities = {c.id: c for c in ecosystem.CAPABILITIES}

    dependencies = []
    for requirement in project["dependencies"]:
        name = requirement_name(requirement)
        capability, purpose = ecosystem.RUNTIME_ROLES[name]
        dependencies.append({"name": name, "requirement": requirement, "scope": "runtime",
                             "capability": capability, "purpose": purpose})
    extra_rows = []
    for extra, requirements in extras.items():
        capability, purpose = ecosystem.EXTRA_ROLES[extra]
        extra_rows.append({
            "name": extra,
            "install": f'pip install "vaft[{extra}]"' if extra != "dev" else 'pip install -e ".[dev]"',
            "scope": capabilities[capability].scope,
            "capability": capability,
            "purpose": purpose,
            "requirements": list(requirements),
        })

    sources = [PYPROJECT, Path(ecosystem.__file__).resolve(), Path(__file__).resolve()]
    sources += [_ROOT / _module_source(code.adapter) for code in ecosystem.EXTERNAL_CODES]
    snapshot: dict = {
        "schema_version": SCHEMA_VERSION,
        "generator": _GENERATOR,
        "source": [
            {"path": _relative(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in sorted(set(sources), key=_relative)
        ],
        "capabilities": [
            {"id": c.id, "title": c.title, "scope": c.scope, "summary": c.summary} for c in ecosystem.CAPABILITIES
        ],
        "dependencies": dependencies,
        "extras": extra_rows,
        "lifecycle": [{"id": key, "text": text} for key, text in ecosystem.INTEGRATION_LIFECYCLE],
        "codes": [_code(code) for code in ecosystem.EXTERNAL_CODES],
    }
    if provenance:
        snapshot["provenance"] = {key: provenance[key] for key in sorted(provenance)}
    return snapshot


def _dump(snapshot: Mapping[str, Any]) -> str:
    import yaml

    return yaml.safe_dump(dict(snapshot), allow_unicode=True, sort_keys=False, default_flow_style=False, width=100)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export VAFT's dependency and external-code catalog for the docs.")
    parser.add_argument("--output", required=True, help="YAML destination for the snapshot")
    parser.add_argument("--provenance-commit", help="Commit the source tree was taken from, recorded in the snapshot")
    parser.add_argument("--provenance-ref", help="Ref that commit was resolved from, recorded in the snapshot")
    parser.add_argument("--check", action="store_true",
                        help="Do not write; exit 1 unless --output already holds what this tree derives")
    arguments = parser.parse_args(argv)
    try:
        if arguments.check:
            import yaml

            recorded = yaml.safe_load(Path(arguments.output).read_text(encoding="utf-8"))
            if ecosystem_snapshot(recorded.get("provenance")) != recorded:
                raise SystemExit(f"{arguments.output} is stale: regenerate it with {_GENERATOR}")
            return
        provenance = {
            key: value
            for key, value in (("commit", arguments.provenance_commit), ("ref", arguments.provenance_ref))
            if value
        }
        destination = Path(arguments.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(_dump(ecosystem_snapshot(provenance or None)), encoding="utf-8")
    except EcosystemCatalogError as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":  # pragma: no cover - exercised through the module CLI
    main()
