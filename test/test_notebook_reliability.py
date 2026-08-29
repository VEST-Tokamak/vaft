"""Structural and portability contracts for user-facing notebooks."""

from __future__ import annotations

import ast
from pathlib import Path

import nbformat
import pytest


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

    # Located by content, not by index: the notebook is restructured from time to
    # time and pinned indices make every edit look like a regression.
    def cell_containing(marker):
        for index, cell in enumerate(book.cells):
            if cell.cell_type == "code" and marker in cell.source:
                return index, cell.source
        raise AssertionError(f"no code cell contains {marker!r}")

    namespace = {}
    for marker in ("output_dir =", "VAFT_DIAGNOSTICS_ODS"):
        index, source = cell_containing(marker)
        exec(compile(source, f"{notebook_path.name}:cell-{index}", "exec"), namespace)

    assert namespace["source"] == sample
    assert len(namespace["ods"]) > 0


def _code_cells(path):
    book = nbformat.read(path, as_version=4)
    for index, cell in enumerate(book.cells):
        if cell.cell_type == "code":
            yield index, cell.source


#: Every spelling that selects a Matplotlib backend. A detector that matched
#: only the two spellings the notebooks happened to use would let an aliased
#: ``mpl.use`` or a bare ``os.environ[...] = ...`` reintroduce the bug.
_BACKEND_SETTER_ATTRS = ("use", "switch_backend")


def _is_backend_pin(node: ast.AST, use_aliases: frozenset[str]) -> bool:
    """True for any call or assignment that selects a Matplotlib backend."""
    if isinstance(node, ast.Assign):
        # os.environ["MPLBACKEND"] = ... / os.environ.update({"MPLBACKEND": ...})
        for target in node.targets:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == "MPLBACKEND"
            ):
                return True
        return False

    if not isinstance(node, ast.Call):
        return False

    # `use(...)` / `switch_backend(...)` imported by name.
    if isinstance(node.func, ast.Name) and node.func.id in use_aliases:
        return True

    name = _attribute_name(node.func)
    if not name:
        return False
    # matplotlib.use, mpl.use, plt.switch_backend, matplotlib.pyplot.use, ...
    if name.rsplit(".", 1)[-1] in _BACKEND_SETTER_ATTRS:
        return True
    # get_ipython().run_line_magic("matplotlib", ...) -- the programmatic form
    # of the %matplotlib magic.
    if name.rsplit(".", 1)[-1] == "run_line_magic":
        first = node.args[0] if node.args else None
        return isinstance(first, ast.Constant) and first.value == "matplotlib"
    if name.endswith("environ.setdefault") or name.endswith("environ.update"):
        first = node.args[0] if node.args else None
        if isinstance(first, ast.Constant) and first.value == "MPLBACKEND":
            return True
        if isinstance(first, ast.Dict):
            return any(
                isinstance(key, ast.Constant) and key.value == "MPLBACKEND"
                for key in first.keys
            )
    return False


def _backend_use_aliases(tree: ast.AST) -> frozenset[str]:
    """Names bound by ``from matplotlib import use`` and friends."""
    aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "matplotlib"
        ):
            for alias in node.names:
                if alias.name in _BACKEND_SETTER_ATTRS:
                    aliases.add(alias.asname or alias.name)
    return frozenset(aliases)


def test_notebooks_do_not_pin_a_noninteractive_backend_unconditionally():
    """A pin outside an interactivity guard silences figures in Jupyter and VS Code.

    Notebooks may still fall back to Agg when no kernel is present -- that is
    what the ``"ipykernel" not in sys.modules`` guard expresses -- but a pin at
    cell top level applies to interactive frontends too (issues #175/#179/#182).
    """
    failures = []
    for path in sorted(NOTEBOOKS.glob("*.ipynb")):
        for index, source in _code_cells(path):
            tree = ast.parse(source, filename=f"{path.name}:cell-{index}")
            aliases = _backend_use_aliases(tree)
            guarded = {
                node
                for statement in ast.walk(tree)
                if isinstance(statement, ast.If)
                for node in ast.walk(statement)
            }
            for node in ast.walk(tree):
                if _is_backend_pin(node, aliases) and node not in guarded:
                    failures.append(f"{path.name}:cell-{index}: unguarded backend pin")

    assert failures == []


#: Modules whose use is unambiguous: a bare ``itertools.x`` in a notebook means
#: the stdlib module, so a missing import is a NameError waiting to happen.
STDLIB_MODULES_USED_BY_NAME = frozenset(
    {
        "collections",
        "functools",
        "glob",
        "itertools",
        "json",
        "logging",
        "math",
        "os",
        "re",
        "shutil",
        "subprocess",
        "sys",
        "tempfile",
        "time",
        "warnings",
    }
)


def test_notebooks_import_every_stdlib_module_they_reference():
    """Guards the ``itertools`` class of failure from issue #180."""
    failures = []
    for path in sorted(NOTEBOOKS.glob("*.ipynb")):
        bound = set()
        used = {}
        for index, source in _code_cells(path):
            tree = ast.parse(source, filename=f"{path.name}:cell-{index}")
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        bound.add((alias.asname or alias.name).split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        bound.add(alias.asname or alias.name)
                elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    for target in ast.walk(node):
                        if isinstance(target, ast.Name) and isinstance(target.ctx, ast.Store):
                            bound.add(target.id)
                elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                    module = node.value.id
                    if module in STDLIB_MODULES_USED_BY_NAME:
                        used.setdefault(module, f"{path.name}:cell-{index}")

        for module, where in sorted(used.items()):
            if module not in bound:
                failures.append(f"{where}: uses {module}.* but never imports {module}")

    assert failures == []


def test_notebook_references_to_packaged_data_resolve():
    """Guards the wrong-subdirectory class of failure from issue #177."""
    import re

    pattern = re.compile(r"vaft/data/([\w./-]+)")
    failures = []
    for path in sorted(NOTEBOOKS.glob("*.ipynb")):
        for index, source in _code_cells(path):
            tree = ast.parse(source, filename=f"{path.name}:cell-{index}")
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                    continue
                match = pattern.search(node.value)
                if match and not (ROOT / "vaft" / "data" / match.group(1)).exists():
                    failures.append(
                        f"{path.name}:cell-{index}: missing packaged data "
                        f"vaft/data/{match.group(1)}"
                    )

    assert failures == []


def test_verification_notebook_loads_the_summary_sheets(monkeypatch):
    """The cells that broke in issues #151/#181 must execute against the real sheets.

    Compiling a cell cannot catch ``from … import EXPECTED_COLUMNS`` against a
    module that no longer defines it, nor a ``KeyError`` from a renamed sheet
    column, so run the offline cells: 1 (bootstrap), 3 (volume-averaged sheet),
    4 (scatter plot over its columns), 6 (equilibrium history sheet).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    notebook_path = NOTEBOOKS / "verification_and_validation.ipynb"
    book = nbformat.read(notebook_path, as_version=4)
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    # Cell 2 asks the remote database which shots have core profiles; the
    # offline cells only need the resulting list, and an empty one keeps the
    # regeneration branch from reaching the network.
    namespace = {"core_profile_shots": []}
    for index in (1, 3, 4, 6):
        source = book.cells[index].source
        exec(compile(source, f"{notebook_path.name}:cell-{index}", "exec"), namespace)

    preset = namespace["volume_preset"]
    assert set(preset.columns) <= set(namespace["volume_df"].columns)
    assert len(namespace["plot_df"]) > 0
    assert not namespace["eq_df"].empty
    plt.close("all")


def test_verification_notebook_refuses_to_regenerate_without_target_shots():
    """An empty shot list must not become a full-namespace scan.

    ``vaft.database.summary(None, ...)`` means *every* shot in the namespace, so
    passing ``None`` when no core-profile shot was found would open the whole
    remote database instead of doing nothing.
    """
    notebook_path = NOTEBOOKS / "verification_and_validation.ipynb"
    book = nbformat.read(notebook_path, as_version=4)
    source = book.cells[3].source

    namespace = {
        "SKIP_VOLUME_AVERAGED_REGEN": False,
        "core_profile_shots": [],
        "output_path": NOTEBOOKS / "does-not-exist.xlsx",
        "generate_volume_averaged_parameter_sheet": lambda *a, **k: pytest.fail(
            "regeneration was attempted with no target shots"
        ),
    }
    with pytest.raises(RuntimeError, match="core_profile"):
        exec(compile(source, f"{notebook_path.name}:cell-3", "exec"), namespace)


def test_the_backend_guard_selects_inline_inside_an_agg_pinned_kernel():
    """The guard must recover a kernel whose environment pins Agg.

    This is the reported failure from issues #175/#179/#182: a kernel running
    with `MPLBACKEND=Agg` renders nothing and warns "FigureCanvasAgg is
    non-interactive". Relying on ipykernel to preset MPLBACKEND is not enough --
    older releases do not, which is why the guard asks for inline explicitly.
    """
    nbclient = pytest.importorskip("nbclient")
    import os

    guard = next(
        source
        for _, source in _code_cells(NOTEBOOKS / "confinement_time_scaling.ipynb")
        if "_ipython" in source
    ).split("import json")[0]

    book = nbformat.v4.new_notebook()
    book.cells = [
        nbformat.v4.new_code_cell(
            guard
            + "\nimport matplotlib\nprint(matplotlib.get_backend())\n"
            "import matplotlib.pyplot as plt\n"
            "fig, ax = plt.subplots()\nax.plot([0, 1], [0, 1])\nplt.show()\n"
        )
    ]

    previous = os.environ.get("MPLBACKEND")
    os.environ["MPLBACKEND"] = "Agg"
    try:
        nbclient.NotebookClient(
            book, timeout=180, kernel_name="python3", allow_errors=True
        ).execute()
    finally:
        if previous is None:
            os.environ.pop("MPLBACKEND", None)
        else:
            os.environ["MPLBACKEND"] = previous

    outputs = book.cells[0].outputs
    rendered = [
        output
        for output in outputs
        if output.output_type in ("display_data", "execute_result")
        and "image/png" in output.get("data", {})
    ]
    streams = "".join(o.get("text", "") for o in outputs if o.output_type == "stream")
    assert "inline" in streams, f"guard left the backend as: {streams.strip()}"
    assert len(rendered) == 1, "the figure did not reach the frontend"
