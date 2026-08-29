"""Structural and portability contracts for user-facing notebooks."""

from __future__ import annotations

import ast
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
MACHINE_PATH_PREFIXES = ("/home/", "/Users/", "/srv/")
DEPRECATED_CALLS = {
    "vaft.database.exist_ts_file",
    "vaft.database.ods.load",
}


def _notebook_paths():
    """Return real notebooks, excluding macOS AppleDouble sidecars on SSDs."""
    return (
        path
        for path in sorted(NOTEBOOKS.glob("*.ipynb"))
        if not path.name.startswith("._")
    )


def _attribute_name(node: ast.AST) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def test_all_notebooks_are_valid_and_python_cells_compile():
    failures = []
    for path in _notebook_paths():
        try:
            book = nbformat.read(path, as_version=4)
            nbformat.validate(book)
        except Exception as error:  # report every invalid notebook together
            failures.append(f"{path.name}: {type(error).__name__}: {error}")
            continue

        for index, cell in enumerate(book.cells):
            if cell.cell_type != "code":
                continue
            try:
                compile(cell.source, f"{path.name}:cell-{index}", "exec")
            except SyntaxError as error:
                failures.append(f"{path.name}:cell-{index}: {error}")

    assert failures == []


def test_notebooks_avoid_deprecated_database_calls_and_machine_paths():
    failures = []
    for path in _notebook_paths():
        book = nbformat.read(path, as_version=4)
        for index, cell in enumerate(book.cells):
            if cell.cell_type != "code":
                continue
            tree = ast.parse(cell.source, filename=f"{path.name}:cell-{index}")
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    name = _attribute_name(node.func)
                    if name in DEPRECATED_CALLS:
                        failures.append(f"{path.name}:cell-{index}: {name}")
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    if node.value.startswith(MACHINE_PATH_PREFIXES):
                        failures.append(
                            f"{path.name}:cell-{index}: user-specific path {node.value!r}"
                        )

    assert failures == []


def test_load_omas_json_accepts_pathlike_input():
    import vaft

    fixture = ROOT / "test" / "data" / "contracts" / "thomson_scattering.json"
    ods = vaft.omas.load_omas_json(fixture, consistency_check=False)

    assert len(ods) > 0


def test_fluctuation_notebook_configured_ods_branch(monkeypatch, tmp_path):
    notebook_path = NOTEBOOKS / "fluctuation_diagnostics_analysis.ipynb"
    book = nbformat.read(notebook_path, as_version=4)
    import vaft

    sample = vaft.data.sample(39915, representation="omas")
    monkeypatch.setenv("VAFT_DIAGNOSTICS_ODS", str(sample))
    monkeypatch.setenv("VAFT_DOCS_OUTPUT_DIR", str(tmp_path))

    namespace = {}
    for index in (1, 3):
        exec(compile(book.cells[index].source, f"{notebook_path.name}:cell-{index}", "exec"), namespace)

    assert namespace["source"] == sample
    assert len(namespace["ods"]) > 0
