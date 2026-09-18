"""Verify the vaft-nn model registry VAFT resolves published models from.

vaft-nn is not a build: ``$VAFT_NN_HOME`` is a checkout of the private
``VEST-Tokamak/vaft-nn`` repository holding each model's release index and
reviewed manifests, and the weights are GitHub Release assets fetched into a
local cache. So the layers here are the checkout, its revision, the index of
every model, the cache, and access to the releases through the GitHub CLI.

    python install/check_vaft_nn.py
    python install/check_vaft_nn.py --registry ~/git/vaft-nn --json
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _external_code_common import (  # noqa: E402
    FAIL,
    PASS,
    SKIP,
    WARN,
    CheckResult,
    check_source_revision,
    emit,
)

TITLE = "vaft-nn model registry check"
RERUN = "python install/check_vaft_nn.py"
PROJECT = "vaft-nn"
CLONE = (
    "Clone the registry and point VAFT at it:\n"
    "         git clone git@github.com:VEST-Tokamak/vaft-nn.git ~/git/vaft-nn\n"
    "         export VAFT_NN_HOME=~/git/vaft-nn"
)


def _root(registry: Optional[str]) -> Optional[Path]:
    value = registry or os.environ.get("VAFT_NN_HOME")
    return Path(value).expanduser() if value else None


def check_registry(registry: Optional[str]) -> CheckResult:
    """``$VAFT_NN_HOME`` names a vaft-nn checkout with ``models/`` and ``schemas/``."""
    label = f"{PROJECT} registry"
    root = _root(registry)
    if root is None:
        return CheckResult(label, FAIL, "$VAFT_NN_HOME is not set and no --registry was given", CLONE)
    missing = [name for name in ("models", "schemas", "scripts/check_registry.py") if not (root / name).exists()]
    if missing:
        return CheckResult(label, FAIL, f"{root} is missing {', '.join(missing)}", CLONE)
    return CheckResult(label, PASS, str(root))


def _models(root: Path) -> list[str]:
    models = root / "models"
    return sorted(p.name for p in models.iterdir() if p.is_dir()) if models.is_dir() else []


def check_models(registry: Optional[str]) -> CheckResult:
    """Every model's releases.yaml parses and every listed version's manifest matches its pin."""
    label = f"{PROJECT} models"
    root = _root(registry)
    if root is None or not (root / "models").is_dir():
        return CheckResult(label, SKIP, "no registry")
    try:
        from vaft.process.ml import resolver
    except Exception as error:  # noqa: BLE001 - reported, not raised
        return CheckResult(label, FAIL, f"vaft.process.ml cannot be imported: {error}",
                           "Run install/check_vaft_environment.py first.")
    import yaml

    names = _models(root)
    if not names:
        return CheckResult(label, WARN, "the registry publishes no model yet",
                           "Expected until the first model is published (#669).")
    problems, versions = [], 0
    for name in names:
        index_path = root / "models" / name / "releases.yaml"
        try:
            index = yaml.safe_load(index_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as error:
            problems.append(f"{name}: releases.yaml unreadable ({type(error).__name__})")
            continue
        for entry in index.get("versions", []):
            versions += 1
            try:
                resolver._registry_entry(name, str(entry["version"]), None, root, os.environ)
            except Exception as error:  # noqa: BLE001
                problems.append(f"{name} {entry.get('version')}: {error}")
    if problems:
        return CheckResult(label, FAIL, "; ".join(problems[:5]),
                           "Pull the registry (git -C $VAFT_NN_HOME pull) or fix the release index.")
    return CheckResult(label, PASS, f"{len(names)} model(s), {versions} version(s): {', '.join(names)}")


def check_cache(registry: Optional[str], cache: Optional[str]) -> CheckResult:
    """Which published versions are present and verified in the local cache."""
    label = f"{PROJECT} cache"
    root = _root(registry)
    if root is None or not (root / "models").is_dir():
        return CheckResult(label, SKIP, "no registry")
    try:
        from vaft.process.ml import resolver
    except Exception:  # noqa: BLE001 - check_models already reports it
        return CheckResult(label, SKIP, "vaft.process.ml cannot be imported")
    import yaml

    cache_root = Path(cache).expanduser() if cache else resolver.default_cache_root()
    ready, absent, broken = [], [], []
    for name in _models(root):
        try:
            index = yaml.safe_load((root / "models" / name / "releases.yaml").read_text(encoding="utf-8")) or {}
            versions = [str(entry["version"]) for entry in index.get("versions", [])]
        except Exception:  # noqa: BLE001 - check_models reports an unreadable index
            continue
        for version in versions:
            try:
                resolver.resolve_model(name, version=version, registry=root, cache=cache_root)
                ready.append(f"{name} {version}")
            except Exception as error:  # noqa: BLE001
                # Nothing cached yet is normal; a cached copy that differs is not.
                (broken if "differs" in str(error) else absent).append(f"{name} {version}")
    if broken:
        return CheckResult(
            label, FAIL, f"{cache_root}: cached files differ from the registry: {', '.join(broken[:5])}",
            "Re-fetch them: vaft.process.ml.fetch_model(name, version=...) replaces a bad copy.",
        )
    if not ready and not absent:
        return CheckResult(label, SKIP, f"nothing published to cache ({cache_root})")
    if absent:
        return CheckResult(
            label, WARN, f"{cache_root}: {len(ready)} verified, not fetched: {', '.join(absent[:5])}",
            "Fetch on demand: vaft.process.ml.fetch_model(name, version=...) or load_model(..., fetch=True).",
        )
    return CheckResult(label, PASS, f"{cache_root}: {', '.join(ready)}")


def check_github_cli() -> CheckResult:
    """The GitHub CLI is installed and logged in -- reported as yes/no, never a token."""
    label = "GitHub CLI access"
    gh = shutil.which("gh")
    if gh is None:
        return CheckResult(label, WARN, "gh is not installed, so release assets cannot be fetched",
                           "Install the GitHub CLI (https://cli.github.com) and run `gh auth login`.")
    try:
        completed = subprocess.run([gh, "auth", "status"], capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return CheckResult(label, WARN, "`gh auth status` did not answer", "Run `gh auth status` yourself.")
    if completed.returncode != 0:
        return CheckResult(label, WARN, "gh is installed but not logged in", "Run `gh auth login` once.")
    return CheckResult(label, PASS, "gh is installed and logged in")


def run_checks(
    *,
    registry: Optional[str] = None,
    cache: Optional[str] = None,
    skip_network: bool = False,
) -> list[CheckResult]:
    """Run every layer in the order resolution depends on them."""
    root = _root(registry)
    results = [
        check_registry(registry),
        check_source_revision(str(root) if root else None, project=PROJECT),
        check_models(registry),
        check_cache(registry, cache),
    ]
    if not skip_network:
        results.append(check_github_cli())
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check_vaft_nn",
        description="Verify the vaft-nn model registry, its cache and release access.",
    )
    parser.add_argument("--registry", help="vaft-nn checkout (default: $VAFT_NN_HOME)")
    parser.add_argument("--cache", help="artifact cache (default: $VAFT_NN_CACHE, else the platform cache)")
    parser.add_argument("--skip-network", action="store_true", help="do not ask gh whether it is logged in")
    parser.add_argument("--json", action="store_true", dest="as_json", help="emit JSON")
    arguments = parser.parse_args(argv)
    results = run_checks(registry=arguments.registry, cache=arguments.cache, skip_network=arguments.skip_network)
    return emit(results, title=TITLE, rerun=RERUN, as_json=arguments.as_json)


if __name__ == "__main__":
    raise SystemExit(main())
