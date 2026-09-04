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
#: ``vaft.code.chease`` was migrated to ``vaft.plot`` view models in issue #139.
ALLOWLIST = {
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
    "path", list(_python_files()), ids=lambda p: p.relative_to(PACKAGE_ROOT).as_posix()
)
def test_module_does_not_import_pyplot(path):
    # ALLOWLIST is keyed by POSIX path, so compare in that grammar --
    # str() yields "data\vfit.py" on Windows and never matches.
    relative = path.relative_to(PACKAGE_ROOT).as_posix()
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
    # This module uses matplotlib.path.Path for point-in-polygon tests.
    path = PACKAGE_ROOT / "process/equilibrium.py"
    assert "matplotlib.path" in path.read_text(encoding="utf-8")
    assert not _pyplot_imports(path)


def test_process_wrapper_does_not_shadow_pathlib_with_matplotlib_path():
    # process_wrapper.py takes filesystem paths (camera pose/intrinsics
    # overrides) and does no point-in-polygon work, so a module-level
    # `from matplotlib.path import Path` there shadows pathlib.Path and
    # breaks those overrides. Geometry users import it locally instead.
    path = PACKAGE_ROOT / "omas/process_wrapper.py"
    source = path.read_text(encoding="utf-8")
    assert "from matplotlib.path import Path" not in source
    assert not _pyplot_imports(path)
