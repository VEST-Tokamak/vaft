"""The per-issue studies under ``vaft.validation.studies`` run as modules (#1756).

Each script used to be a ``PYTHONPATH=. python validation/<folder>/<script>.py``
path run, which imports whichever ``vaft`` is first on the path.  Now every
folder is a regular package and a script is reached as
``python -m vaft.validation.studies.<folder>.<script>``.  These tests import
each module (or ask it for ``--help``) so a moved script that still carries a
path hack, a stale ``validation.`` import or module-level work fails here
rather than on a server.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDIES = ROOT / "vaft" / "validation" / "studies"
PACKAGE = "vaft.validation.studies"

#: module -> the optional third-party package it needs at import time
OPTIONAL = {
    f"{PACKAGE}.fixed_free_1608.vacuum_response": "OpenFUSIONToolkit",
    f"{PACKAGE}.ec_launcher_cad_266.extract_launcher_geometry": "OCP",
}
#: argparse scripts: ``--help`` exits 0 without touching any input
WITH_HELP = (
    f"{PACKAGE}.fixed_free_1608.closure",
    f"{PACKAGE}.fixed_free_1608.matrix",
    f"{PACKAGE}.fixed_free_1608.vfixed_fit",
    f"{PACKAGE}.fixed_free_1608.vacuum_response",
    f"{PACKAGE}.nice_issue_666.synthetic_equilibria.plot_overviews",
)


def _scripts() -> list[str]:
    found = []
    for path in sorted(STUDIES.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        found.append(".".join(path.relative_to(ROOT).with_suffix("").parts))
    return found


SCRIPTS = _scripts()


def test_every_study_folder_is_a_package_and_every_script_has_a_main_guard():
    assert SCRIPTS, "no study scripts found"
    for module in SCRIPTS:
        path = ROOT / Path(*module.split(".")).with_suffix(".py")
        assert (path.parent / "__init__.py").is_file(), module
        tree = ast.parse(path.read_text(encoding="utf-8"))
        guarded = any(
            isinstance(node, ast.If) and "__main__" in ast.dump(node.test) for node in tree.body
        )
        assert guarded, f"{module} runs work at import time; wrap it in main() behind __name__ == '__main__'"
        # no path hack and no import of the old top-level validation/ tree
        text = path.read_text(encoding="utf-8")
        assert "sys.path" not in text, module
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("validation."), f"{module} imports the old validation/ tree"


@pytest.mark.parametrize("module", SCRIPTS)
def test_each_study_script_imports_as_a_module(module):
    needs = OPTIONAL.get(module)
    if needs and importlib.util.find_spec(needs) is None:
        pytest.skip(f"{module} needs {needs}")
    imported = importlib.import_module(module)
    assert callable(getattr(imported, "main", None)) or module.endswith("extract_launcher_geometry"), module


@pytest.mark.parametrize("module", WITH_HELP)
def test_argparse_scripts_answer_help_when_run_with_python_m(module):
    needs = OPTIONAL.get(module)
    if needs and importlib.util.find_spec(needs) is None:
        pytest.skip(f"{module} needs {needs}")
    result = subprocess.run([sys.executable, "-m", module, "--help"], capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "usage:" in result.stdout
