"""The ``optimize`` extra (#1877): Optuna is opt-in and never reached by an import.

The contract this file pins is the packaging one from #1875/#1877, not an
optimizer: the extra exists with a bounded version range, Optuna is neither a
core dependency nor an implicit ``[dev]`` requirement, and no VAFT module
imports it at module level -- the suite must stay runnable, and meaningful,
without Optuna installed.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "vaft"


def _project():
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        tomllib = pytest.importorskip("tomli")
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def test_the_optimize_extra_is_declared_with_a_bounded_range():
    extras = _project()["optional-dependencies"]
    assert extras["optimize"] == ["optuna>=5,<6"], extras.get("optimize")


def test_optuna_is_neither_a_core_dependency_nor_in_dev():
    project = _project()
    names = [r.split("[")[0].split(">")[0].split("<")[0].split("=")[0].strip().lower() for r in project["dependencies"]]
    assert "optuna" not in names
    dev = [r.lower() for r in project["optional-dependencies"]["dev"]]
    assert not any(r.startswith("optuna") for r in dev), dev


def _module_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in tree.body:  # module level only: a function-local import is the contract
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
        elif isinstance(node, (ast.If, ast.Try)):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Import):
                    found.update(alias.name.split(".")[0] for alias in sub.names)
                elif isinstance(sub, ast.ImportFrom) and sub.module:
                    found.add(sub.module.split(".")[0])
    return found


def test_no_vaft_module_imports_optuna_at_module_level():
    offenders = sorted(
        str(path.relative_to(ROOT))
        for path in PACKAGE.rglob("*.py")
        if "optuna" in _module_level_imports(path)
    )
    assert offenders == [], offenders


def test_importing_the_package_and_its_namespaces_never_loads_optuna():
    """Holds whether or not Optuna is installed: the import must not be reached."""
    import importlib

    sys.modules.pop("optuna", None)
    for name in ("vaft", "vaft.process", "vaft.code", "vaft.formula", "vaft.omas"):
        importlib.import_module(name)
    assert "optuna" not in sys.modules
