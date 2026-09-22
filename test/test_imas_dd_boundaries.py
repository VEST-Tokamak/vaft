"""The DD core runs on native IMAS alone (#1127, #1132).

``vaft/imas/_dd/`` must not import OMAS, ``vaft.omas``, the vendored OMAS bridge
``vaft.imas.omas_imas`` or the VEST database layer: ODS projection (#1131) is a
separate adapter, and VEST allocation policy belongs to the deployment. The one
permitted reach into ``vaft.database`` is ``_local``, the shared local-file
detector that ``IMASHandle`` lives in.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "vaft" / "imas" / "_dd"

FORBIDDEN = ("omas", "vaft.omas", "vaft.imas.omas_imas", "vaft.database")
ALLOWED = ("vaft.database._local",)


def _absolute(module: str | None, level: int, path: Path) -> str:
    if level == 0:
        return module or ""
    parts = ["vaft", "imas", "_dd"][: 3 - (level - 1)]
    return ".".join(parts + ([module] if module else []))


def _imports(path: Path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            yield _absolute(node.module, node.level, path)


def _forbidden(name: str) -> bool:
    if any(name == ok or name.startswith(ok + ".") for ok in ALLOWED):
        return False
    return any(name == bad or name.startswith(bad + ".") for bad in FORBIDDEN)


def test_the_dd_core_imports_no_omas_and_no_deployment_layer():
    modules = sorted(PACKAGE.glob("*.py"))
    assert modules, "vaft/imas/_dd has moved; update this test"
    offenders = {
        path.name: sorted(name for name in _imports(path) if _forbidden(name))
        for path in modules
    }
    assert {k: v for k, v in offenders.items() if v} == {}


def test_the_relative_import_resolver_is_right():
    path = PACKAGE / "_dd.py"
    assert _absolute("_types", 1, path) == "vaft.imas._dd._types"
    assert _absolute("database._local", 3, path) == "vaft.database._local"
    assert _absolute("omas_imas", 2, path) == "vaft.imas.omas_imas"
