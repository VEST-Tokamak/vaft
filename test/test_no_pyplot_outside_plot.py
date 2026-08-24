"""Rendering must stay inside ``vaft.plot`` (issue #63 acceptance criterion).

Only ``vaft.plot`` may import ``matplotlib.pyplot``.  Other namespaces build the
typed view models and delegate.  ``matplotlib.path.Path`` is geometry, not
rendering, and is deliberately not flagged.
"""

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "vaft"
RENDERING_HOME = PACKAGE_ROOT / "plot"

#: Modules still awaiting their adapter slice.  This list must only shrink.
#: ``vaft.code`` and ``vaft.data`` adapters land with the rest of issue #63.
ALLOWLIST = {
    "code/chease.py",
    "data/vfit.py",
}


def _python_files():
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if RENDERING_HOME in path.parents or path == RENDERING_HOME:
            continue
        yield path


def _pyplot_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "matplotlib.pyplot" or alias.name.startswith(
                    "matplotlib.pyplot."
                ):
                    found.append(f"line {node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "matplotlib.pyplot":
                found.append(f"line {node.lineno}: from {module} import ...")
            elif module == "matplotlib":
                for alias in node.names:
                    if alias.name == "pyplot":
                        found.append(f"line {node.lineno}: from matplotlib import pyplot")
    return found


@pytest.mark.parametrize(
    "path", list(_python_files()), ids=lambda p: str(p.relative_to(PACKAGE_ROOT))
)
def test_module_does_not_import_pyplot(path):
    relative = str(path.relative_to(PACKAGE_ROOT))
    offenders = _pyplot_imports(path)
    if relative in ALLOWLIST:
        pytest.skip(f"{relative} is an allowlisted in-flight shim")
    assert not offenders, f"{relative} imports pyplot: {offenders}"


def test_allowlist_entries_still_exist_and_still_need_the_exemption():
    for relative in sorted(ALLOWLIST):
        path = PACKAGE_ROOT / relative
        assert path.exists(), f"{relative} is allowlisted but missing"
        assert _pyplot_imports(path), (
            f"{relative} no longer imports pyplot; remove it from ALLOWLIST"
        )


def test_geometry_use_of_matplotlib_path_is_not_flagged():
    # These modules use matplotlib.path.Path for point-in-polygon tests.
    for relative in ("process/equilibrium.py", "omas/process_wrapper.py"):
        path = PACKAGE_ROOT / relative
        assert "matplotlib.path" in path.read_text(encoding="utf-8")
        assert not _pyplot_imports(path)
