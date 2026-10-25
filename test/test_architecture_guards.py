"""Architecture guards for the VEST configuration consolidation (#2147, Phase E #2168).

Each active guard pins a specific known architectural defect rather than a
library or a filename pattern at large, and each comes with:

* an **allowlist of the current state**, entry by entry with its justification,
  so the guard passes today and blocks only *new* occurrences;
* a **stale-entry check**: an allowlist entry that no longer matches reality
  fails the guard, so the allowlist can only shrink as the migration lands
  (B1 #2158 rewires the loaders, B3 #2162 the processing duplicates);
* a **negative control** that feeds the same detector a synthetic violation in
  ``tmp_path`` and asserts it is reported for the right reason.

Active guards
-------------
(a) No new runtime readers of the retired split-file magnetics inventory
    (``MD.yaml``, ``VEST_MagneticsGeometry_Full_ver_2302.yaml``,
    ``table.yaml``): an AST scan of ``vaft/`` and ``workflow/`` string literals
    (docstrings and bare string statements excluded, they do not execute), plus
    a scan for every reference to the private loader functions -- in their
    owner module too, so a new public wrapper around a loader is a finding.
(b) Layer contract over the import graph (grimp, the ``dev`` extra both
    develop gates install): ``vaft.process`` does not import
    ``vaft.machine_mapping`` or ``workflow``; ``vaft.formula`` imports nothing
    from ``vaft.process``, ``vaft.machine_mapping``, ``vaft.omas`` or ``omas``.
(c) No second sensor inventory: no YAML under ``vaft/`` or ``workflow/`` holds
    sensor records (``r``/``z`` plus ``field_code``, ``index`` or ``name``, as a
    list or a mapping) outside the allowlisted locations.

(d) Canonical registry schema and uniqueness, through B1's own validator (#2158).
(e) An explicit ``info_file`` that does not exist raises: every public
    ``vaft.machine_mapping`` function with an ``info_file`` parameter is
    discovered and called with a missing path (audit gap G1, #2159).
(f) Overrides go through the validated semantics: machine facts, unknown keys
    and calibrations without ``calibration_override_reason`` are refused by
    both public override entry points, and no code outside the validated
    owners merges or applies an ``override`` mapping (G5/G7, #2159).
(g) No module under ``vaft/`` or ``workflow/`` uses a deprecated alias
    (register rows L5, L6, P9, P10). Names are resolved through the imports,
    so ``vaft.code.efit.legacy``'s own, different ``vfit_signal_startend`` is
    not the alias.
(h) ``vaft.process`` names no ``vest.yaml`` path, references no
    ``vaft.machine_mapping`` module dynamically, and compares no shot number
    with an era literal. ``np.interp``, ``scipy.signal`` and integration stay
    legal: the guard targets configuration reads, not numerics.

The register these guards serve is
``docs/architecture/consolidation/retirement_register.md``.
"""

from __future__ import annotations

import ast
import importlib
import io
import re
import sys
import textwrap
import tokenize
from pathlib import Path
from typing import Iterable, Mapping, NamedTuple

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = ("vaft", "workflow")
REGISTER = "docs/architecture/consolidation/retirement_register.md"


class Finding(NamedTuple):
    """One located occurrence. ``key`` is what an allowlist entry names."""

    key: tuple
    location: str  # "path:line"
    detail: str


def check_against_allowlist(
    findings: Iterable[Finding],
    allowlist: Mapping[tuple, str],
    *,
    rule: str,
    expected: str,
) -> None:
    """Fail on any finding outside ``allowlist`` and on any allowlist entry that matched nothing."""
    findings = list(findings)
    new = [f for f in findings if f.key not in allowlist]
    matched = {f.key for f in findings}
    stale = sorted(set(allowlist) - matched, key=repr)
    messages = []
    if new:
        lines = "\n".join(f"  {f.location}: {f.detail}" for f in sorted(new, key=lambda f: f.location))
        messages.append(
            f"{rule}\n{lines}\nExpected instead: {expected}\n"
            f"If the new occurrence is justified, add it to the allowlist with a reason and "
            f"record it in {REGISTER}."
        )
    if stale:
        lines = "\n".join(f"  {entry!r}" for entry in stale)
        messages.append(
            f"stale allowlist entries for rule '{rule.splitlines()[0]}' -- they no longer match the "
            f"source. Remove them (the migration has retired them) and update {REGISTER}:\n{lines}"
        )
    assert not messages, "\n\n".join(messages)


def _tracked_like(path: Path, directory: Path) -> bool:
    """Skip caches and dot-directories (``.snakemake/`` script copies, ``__pycache__``)."""
    parts = path.relative_to(directory).parts
    return not any(part.startswith(".") or part == "__pycache__" for part in parts)


def _python_sources(base: Path, roots: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        directory = base / root
        if directory.is_dir():
            files.extend(p for p in directory.rglob("*.py") if _tracked_like(p, directory))
    return sorted(files)


def _snakemake_sources(base: Path, roots: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        directory = base / root
        if directory.is_dir():
            for pattern in ("Snakefile", "*.smk"):
                files.extend(p for p in directory.rglob(pattern) if _tracked_like(p, directory))
    return sorted(files)


def _relative(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


# ---------------------------------------------------------------------------
# (a) retired magnetics YAML readers
# ---------------------------------------------------------------------------

#: The three files the canonical registry (#2158) replaces, and how a string
#: literal names each. Case-insensitive, because macOS and Windows filesystems
#: are: ``"md.yaml"`` opens ``MD.yaml`` there. The geometry pattern is the
#: family prefix, so ``f"VEST_MagneticsGeometry_Full_ver_{v}.yaml"`` matches on
#: its literal part. Boundaries keep ``efit_table.yaml`` and ``cmd.yaml`` out.
#: Not caught (by design, documented in the register): names assembled from
#: fragments, ``with_suffix``, and globs.
RETIRED_MAGNETICS_YAML: dict[str, re.Pattern[str]] = {
    "MD.yaml": re.compile(r"(?<![\w.-])MD\.ya?ml\b", re.IGNORECASE),
    "VEST_MagneticsGeometry_Full_ver_2302.yaml": re.compile(r"VEST_MagneticsGeometry_Full_ver_", re.IGNORECASE),
    "table.yaml": re.compile(r"(?<![\w.-])table\.ya?ml\b", re.IGNORECASE),
}

RETIRED_YAML_RULE = (
    "retired split-file magnetics YAML named by runtime code (#2168 Scope D2: no new readers of "
    "MD.yaml / VEST_MagneticsGeometry_Full_ver_2302.yaml / table.yaml)"
)
RETIRED_YAML_EXPECTED = (
    "resolve sensors through vaft.machine_mapping.magnetics public resolvers "
    "(vest_equilibrium_magnetics_channel_definitions, magnetics_wiring_for_shot), and after #2158 "
    "through the canonical vest.yaml registry resolver"
)

#: (path, enclosing qualname, retired file) -> why it is still allowed.
#: B1 (#2158) retired every runtime reader; only the audit generator names them.
RETIRED_YAML_ALLOWLIST: dict[tuple[str, str, str], str] = {
    ("workflow/architecture_audit/build_inventory.py", "<module>", name):
        "Phase A audit generator (#2148): joins the frozen fixture copies in "
        "test/data/legacy_magnetics as reconciliation evidence, not runtime configuration (#2158)"
    for name in ("MD.yaml", "VEST_MagneticsGeometry_Full_ver_2302.yaml", "table.yaml")
}

#: Private loaders of the retired files, defined in vaft/machine_mapping/magnetics.py.
RETIRED_LOADERS = ("_load_equilibrium_magnetics_channels", "_load_static_channels", "_load_names_by_code")
RETIRED_LOADER_RULE = (
    "private loader of the retired magnetics YAML referenced by a new caller "
    "(#2168 Scope D2: runtime-reader dependency)"
)

#: (path, enclosing qualname, loader) -> why the reference is still allowed.
#: Empty since B1 (#2158) deleted the loaders and rewired every caller onto the
#: registry; the rule stays so a reintroduced loader or caller is caught.
RETIRED_LOADER_ALLOWLIST: dict[tuple[str, str, str], str] = {}

#: Raw-text prefilter (plain substrings of the lower-cased text, ~20x faster than
#: a case-insensitive regex): only files that could produce a finding are parsed.
_PREFILTER_NEEDLES = ("md.y", "table.y", "vest_magneticsgeometry_full_ver_", *(n.lower() for n in RETIRED_LOADERS))


def _might_match(text: str) -> bool:
    lowered = text.lower()
    return any(needle in lowered for needle in _PREFILTER_NEEDLES)


class _Collector(ast.NodeVisitor):
    """Executable string constants and name references, each with its enclosing qualname.

    Docstrings and other bare string statements are skipped: they have no
    runtime effect and documentation may name historical files.
    """

    def __init__(self) -> None:
        self.stack: list[str] = []
        self.strings: list[tuple[str, int, str]] = []
        self.refs: list[tuple[str, int, str]] = []

    @property
    def qualname(self) -> str:
        return ".".join(self.stack) or "<module>"

    def _scoped(self, node: ast.AST, name: str) -> None:
        self.stack.append(name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._scoped(node, node.name)

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scoped(node, node.name)

    def visit_Expr(self, node: ast.Expr) -> None:
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            self.strings.append((self.qualname, node.lineno, node.value))
            # getattr(module, "_load_static_channels") is a reference too.
            self.refs.append((self.qualname, node.lineno, node.value))

    def visit_Name(self, node: ast.Name) -> None:
        self.refs.append((self.qualname, node.lineno, node.id))

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.refs.append((self.qualname, node.lineno, node.attr))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self.refs.append((self.qualname, node.lineno, alias.name))


def _collected(base: Path, roots: Iterable[str]) -> Iterable[tuple[str, _Collector]]:
    for path in _python_sources(base, roots):
        text = path.read_text(encoding="utf-8")
        if not _might_match(text):
            continue
        collector = _Collector()
        collector.visit(ast.parse(text, filename=str(path)))
        yield _relative(path, base), collector


def _string_token_types() -> set[int]:
    # 3.12+ tokenizes f-strings (and 3.14 t-strings) into parts; STRING otherwise.
    names = ("STRING", "FSTRING_MIDDLE", "TSTRING_MIDDLE")
    return {getattr(tokenize, name) for name in names if hasattr(tokenize, name)}


def retired_yaml_findings(base: Path, roots: Iterable[str] = SCAN_ROOTS) -> list[Finding]:
    """Every executable string literal under ``roots`` naming a retired magnetics YAML."""
    findings: list[Finding] = []
    for rel, collector in _collected(base, roots):
        for qualname, line, value in collector.strings:
            for name, pattern in RETIRED_MAGNETICS_YAML.items():
                if pattern.search(value):
                    findings.append(Finding((rel, qualname, name), f"{rel}:{line}", f"in {qualname}: names {name}"))
    for path in _snakemake_sources(base, roots):
        # Snakemake files are lexically Python but not parseable as Python; their
        # string tokens are what can name a file.
        rel = _relative(path, base)
        text = path.read_text(encoding="utf-8")
        if not _might_match(text):
            continue
        types = _string_token_types()
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type in types:
                for name, pattern in RETIRED_MAGNETICS_YAML.items():
                    if pattern.search(token.string):
                        line = token.start[0]
                        findings.append(Finding((rel, "<snakefile>", name), f"{rel}:{line}", f"names {name}"))
    return findings


def retired_loader_findings(base: Path, roots: Iterable[str] = SCAN_ROOTS) -> list[Finding]:
    """Every reference (import, call, attribute, getattr string) to a retired loader."""
    findings: list[Finding] = []
    for rel, collector in _collected(base, roots):
        for qualname, line, name in collector.refs:
            if name in RETIRED_LOADERS:
                findings.append(Finding((rel, qualname, name), f"{rel}:{line}", f"in {qualname}: uses {name}"))
    return findings


def test_no_new_runtime_reader_of_the_retired_magnetics_yaml():
    check_against_allowlist(
        retired_yaml_findings(REPO_ROOT),
        RETIRED_YAML_ALLOWLIST,
        rule=RETIRED_YAML_RULE,
        expected=RETIRED_YAML_EXPECTED,
    )
    # The one allowlisted reader must join the frozen fixture, never a packaged copy.
    generator = (REPO_ROOT / "workflow" / "architecture_audit" / "build_inventory.py").read_text(encoding="utf-8")
    assert 'GEOMETRY_DIR = Path("test/data/legacy_magnetics")' in generator


def test_no_new_consumer_of_the_retired_magnetics_loaders():
    check_against_allowlist(
        retired_loader_findings(REPO_ROOT),
        RETIRED_LOADER_ALLOWLIST,
        rule=RETIRED_LOADER_RULE,
        expected=RETIRED_YAML_EXPECTED,
    )


def test_retired_yaml_scanner_reports_a_synthetic_reader_and_ignores_docstrings(tmp_path):
    package = tmp_path / "vaft" / "machine_mapping"
    package.mkdir(parents=True)
    (package / "new_reader.py").write_text(
        textwrap.dedent(
            '''
            """Module docstring mentioning MD.yaml is documentation, not a reader."""
            from pathlib import Path

            def load(root: Path):
                """Reads table.yaml -- also only a docstring."""
                "a bare string statement naming VEST_MagneticsGeometry_Full_ver_2302.yaml"
                return (root / "MD.yaml").read_text(), f"{root}/geometry/table.yml"

            def load_lowercase(root: Path, version: str):
                return root / "md.yaml", root / f"VEST_MagneticsGeometry_Full_ver_{version}.yaml"

            NOT_RETIRED = ("efit_table.yaml", "CMD.yaml", "cmd.yaml", "VEST_DiscretizedCoilGeometry_Full_ver_2507.mat")
            '''
        ),
        encoding="utf-8",
    )
    findings = retired_yaml_findings(tmp_path)
    assert sorted({f.key for f in findings}) == [
        ("vaft/machine_mapping/new_reader.py", "load", "MD.yaml"),
        ("vaft/machine_mapping/new_reader.py", "load", "table.yaml"),
        ("vaft/machine_mapping/new_reader.py", "load_lowercase", "MD.yaml"),
        ("vaft/machine_mapping/new_reader.py", "load_lowercase", "VEST_MagneticsGeometry_Full_ver_2302.yaml"),
    ]
    with pytest.raises(AssertionError, match=r"new_reader\.py:8: in load: names MD\.yaml(.|\n)*Expected instead"):
        check_against_allowlist(findings, {}, rule=RETIRED_YAML_RULE, expected=RETIRED_YAML_EXPECTED)


def test_retired_yaml_scanner_reads_snakemake_strings(tmp_path):
    rules = tmp_path / "workflow" / "pipeline"
    rules.mkdir(parents=True)
    (rules / "Snakefile").write_text(
        'rule geometry:\n    input: "vaft/data/geometry/VEST_MagneticsGeometry_Full_ver_2302.yaml"\n'
        "    shell: 'cat {input}'\n"
        "rule calibration:\n    input: f\"{GEOMETRY}/MD.yaml\"\n",
        encoding="utf-8",
    )
    findings = retired_yaml_findings(tmp_path)
    assert [f.key for f in findings] == [
        ("workflow/pipeline/Snakefile", "<snakefile>", "VEST_MagneticsGeometry_Full_ver_2302.yaml"),
        ("workflow/pipeline/Snakefile", "<snakefile>", "MD.yaml"),
    ]
    with pytest.raises(AssertionError, match=r"workflow/pipeline/Snakefile:5: names MD\.yaml"):
        check_against_allowlist(findings, {}, rule=RETIRED_YAML_RULE, expected=RETIRED_YAML_EXPECTED)


def test_retired_loader_scanner_reports_a_synthetic_consumer(tmp_path):
    (tmp_path / "workflow").mkdir()
    (tmp_path / "workflow" / "script.py").write_text(
        "from vaft.machine_mapping.magnetics import _load_static_channels\n"
        "import vaft.machine_mapping.magnetics as m\n"
        "def names():\n"
        "    return m._load_names_by_code(), getattr(m, '_load_equilibrium_magnetics_channels')\n",
        encoding="utf-8",
    )
    owner = tmp_path / "vaft" / "machine_mapping"
    owner.mkdir(parents=True)
    # Inside the owner module a new public wrapper is a new consumer as well;
    # the loader's own definition is not.
    (owner / "magnetics.py").write_text(
        "def _load_static_channels():\n    return []\n\n"
        "def geometry_rows():\n    return _load_static_channels()\n",
        encoding="utf-8",
    )
    findings = retired_loader_findings(tmp_path)
    assert sorted(f.key for f in findings) == [
        ("vaft/machine_mapping/magnetics.py", "geometry_rows", "_load_static_channels"),
        ("workflow/script.py", "<module>", "_load_static_channels"),
        ("workflow/script.py", "names", "_load_equilibrium_magnetics_channels"),
        ("workflow/script.py", "names", "_load_names_by_code"),
    ]
    with pytest.raises(AssertionError, match=r"magnetics\.py:5: in geometry_rows: uses _load_static_channels"):
        check_against_allowlist(findings, {}, rule=RETIRED_LOADER_RULE, expected=RETIRED_YAML_EXPECTED)


def test_a_stale_allowlist_entry_fails():
    live = [Finding(("a.py", "f", "MD.yaml"), "a.py:1", "names MD.yaml")]
    allowlist = {("a.py", "f", "MD.yaml"): "live", ("gone.py", "g", "MD.yaml"): "retired already"}
    with pytest.raises(AssertionError, match=r"stale allowlist entries(.|\n)*gone\.py") as caught:
        check_against_allowlist(live, allowlist, rule="rule", expected="x")
    assert "a.py" not in str(caught.value).split("stale", 1)[1].replace("gone.py", "")


# ---------------------------------------------------------------------------
# (b) layer contract over the import graph
# ---------------------------------------------------------------------------

LAYER_RULE = "layer contract violated (#2168 Scope E1)"
LAYER_EXPECTED = (
    "vaft.process takes machine facts as arguments (resolved by vaft.machine_mapping and passed "
    "in by the caller); vaft.formula depends on nothing machine- or representation-specific"
)


def layer_rules(root: str) -> dict[str, tuple[str, ...]]:
    """Source layer -> layers it must not import, for a top-level package ``root``."""
    return {
        f"{root}.process": (f"{root}.machine_mapping", "workflow"),
        f"{root}.formula": (f"{root}.process", f"{root}.machine_mapping", f"{root}.omas", "omas"),
    }


#: (importer module, imported module) -> justification. Every entry is a lazy,
#: function-level import that exists today; each is a known debt, not a design.
LAYER_ALLOWLIST: dict[tuple[str, str], str] = {
    ("vaft.process.impurity", "vaft.machine_mapping.core_profiles"):
        "machine preset 'vest' resolves vest_impurity_model; should be resolved by the caller (debt, #2168)",
    ("vaft.process.kinetic_closure", "vaft.machine_mapping.core_profiles"):
        "T_i record grammar (classify_ti_record, ti_record_fields) lives in machine_mapping; "
        "machine-independent, should move to vaft.process or vaft.omas (debt, #2168)",
    ("vaft.process.transport_state", "vaft.machine_mapping.core_profiles"):
        "same T_i record grammar plus policy_for_ods (VEST policy lookup); debt, #2168",
    ("vaft.process.local_mhd_criteria", "vaft.machine_mapping.utils"):
        "generic ODS helper path_exists housed in machine_mapping.utils; should move to vaft.omas (debt, #2168)",
    ("vaft.process.q_profile_context", "vaft.machine_mapping.utils"):
        "generic ODS helper path_exists housed in machine_mapping.utils; should move to vaft.omas (debt, #2168)",
}


def layer_findings(graph, root: str) -> list[Finding]:
    """Direct imports that cross a forbidden layer edge, with file-line detail."""
    findings: list[Finding] = []
    rules = layer_rules(root)
    for module in sorted(graph.modules):
        for source, forbidden in rules.items():
            if module != source and not module.startswith(source + "."):
                continue
            for imported in sorted(graph.find_modules_directly_imported_by(module)):
                if not any(imported == f or imported.startswith(f + ".") for f in forbidden):
                    continue
                for detail in graph.get_import_details(importer=module, imported=imported) or [{}]:
                    findings.append(
                        Finding(
                            (module, imported),
                            f"{module}:{detail.get('line_number', '?')}",
                            f"{source} must not import {imported} ({detail.get('line_contents', '').strip()})",
                        )
                    )
    return findings


def _build_graph(root: str):
    # grimp is in the `dev` extra that both develop gates install. A missing
    # grimp there is a broken gate, so it fails on CI rather than skipping.
    import os

    if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS") or os.environ.get("SLURM_JOB_ID"):
        import grimp
    else:
        grimp = pytest.importorskip("grimp", reason="install the dev or architecture extra")
    return grimp.build_graph(root, include_external_packages=True, cache_dir=None)


def test_layer_contract_holds_over_the_import_graph():
    import vaft

    assert Path(vaft.__file__).resolve().parent == REPO_ROOT / "vaft", (
        f"vaft resolves to {vaft.__file__}, not this checkout; the import graph would describe another "
        "tree. Run with PYTHONPATH=<this checkout> (see the worktree import-path trap)."
    )
    check_against_allowlist(
        layer_findings(_build_graph("vaft"), "vaft"),
        LAYER_ALLOWLIST,
        rule=LAYER_RULE,
        expected=LAYER_EXPECTED,
    )


def test_layer_contract_reports_a_synthetic_violation(tmp_path, monkeypatch):
    root = "archguard_synthetic_pkg"
    files = {
        "__init__.py": "",
        "process/__init__.py": "",
        "process/bad.py": f"def f():\n    from {root}.machine_mapping.vest import POLICY\n    return POLICY\n",
        "process/orchestrated.py": "import workflow.stage\n",
        "process/fine.py": f"from {root}.formula import kernel\n",
        "machine_mapping/__init__.py": "",
        "machine_mapping/vest.py": "POLICY = 1\n",
        "formula/__init__.py": "",
        "formula/kernel.py": "import omas\n",
        "omas/__init__.py": "",
    }
    for relative, text in files.items():
        path = tmp_path / root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    findings = layer_findings(_build_graph(root), root)
    # grimp squashes an external import to its top-level package: workflow.stage -> workflow.
    assert sorted(f.key for f in findings) == [
        (f"{root}.formula.kernel", "omas"),
        (f"{root}.process.bad", f"{root}.machine_mapping.vest"),
        (f"{root}.process.orchestrated", "workflow"),
    ]
    with pytest.raises(AssertionError, match=rf"{root}\.process\.bad:2: {root}\.process must not import"):
        check_against_allowlist(findings, {}, rule=LAYER_RULE, expected=LAYER_EXPECTED)


# ---------------------------------------------------------------------------
# (c) no second sensor inventory in YAML
# ---------------------------------------------------------------------------

INVENTORY_RULE = (
    "YAML sensor records (r/z plus field_code, index or name) outside the allowlisted inventory "
    "(#2168 Scope B: no second independently maintained registry)"
)
INVENTORY_EXPECTED = (
    "add sensors to, or reference them from, the canonical inventory (vest.yaml registry after #2158), "
    "not a new table"
)

#: Keys that make an r/z record a *sensor* record rather than a bare position.
SENSOR_IDENTITY_KEYS = frozenset({"field_code", "index", "name"})

_WIRING = "0.diagnostics.equilibrium_magnetics.processing"

#: (YAML path, record-collection path; list items as [], mapping values as {}) -> justification.
INVENTORY_ALLOWLIST: dict[tuple[str, str], str] = {
    ("vaft/machine_mapping/vest.yaml", "0.diagnostics.equilibrium_magnetics.sensors.records"):
        "THE canonical equilibrium magnetics registry (#2158), validated by "
        "vaft.machine_mapping.magnetics_registry; the single allowed sensor authority",
    ("vaft/machine_mapping/vest.yaml", f"{_WIRING}.wiring.revisions[].probe_overrides"):
        "per-era wiring overrides naming a registry id; r/z are pins the resolver checks "
        "against that record, not an inventory (#2158)",
}


def _is_sensor_record(item) -> bool:
    if not isinstance(item, dict):
        return False
    keys = {str(k).lower() for k in item}
    return {"r", "z"} <= keys and bool(keys & SENSOR_IDENTITY_KEYS)


def _record_lists(node, path: str, out: set[str]) -> None:
    if isinstance(node, dict):
        # A mapping of name -> {r, z, ...} is an inventory too.
        if sum(_is_sensor_record(value) for value in node.values()) >= 2:
            out.add(f"{path}{{}}")
        for key, value in node.items():
            _record_lists(value, f"{path}.{key}" if path else str(key), out)
    elif isinstance(node, list):
        if any(_is_sensor_record(item) for item in node):
            out.add(path)
        for item in node:
            _record_lists(item, f"{path}[]", out)


def sensor_inventory_findings(base: Path, roots: Iterable[str] = SCAN_ROOTS) -> list[Finding]:
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    findings: list[Finding] = []
    for root in roots:
        directory = base / root
        if not directory.is_dir():
            continue
        paths = [*directory.rglob("*.yaml"), *directory.rglob("*.yml")]
        for path in sorted(p for p in paths if _tracked_like(p, directory)):
            rel = _relative(path, base)
            text = path.read_text(encoding="utf-8")
            if not (re.search(r"\br\s*:", text, re.I) and re.search(r"\bz\s*:", text, re.I)):
                continue
            document = yaml.load(text, Loader=loader)  # noqa: S506 - a safe loader
            located: set[str] = set()
            _record_lists(document, "", located)
            for record_path in sorted(located):
                findings.append(
                    Finding((rel, record_path), rel, f"'{record_path}' holds sensor records (r/z + identity key)")
                )
    return findings


def test_no_second_sensor_inventory_in_yaml():
    check_against_allowlist(
        sensor_inventory_findings(REPO_ROOT),
        INVENTORY_ALLOWLIST,
        rule=INVENTORY_RULE,
        expected=INVENTORY_EXPECTED,
    )


def test_sensor_inventory_scanner_reports_a_synthetic_table(tmp_path):
    data = tmp_path / "vaft" / "data"
    data.mkdir(parents=True)
    (data / "new_probes.yaml").write_text(
        "probes:\n  sets:\n    - channels:\n        - {field_code: 101, R: 0.1, z: 0.2}\n"
        "        - {field_code: 102, r: 0.1, z: 0.3}\n"
        "loops:\n  FL1: {r: 0.45, z: 0.1, name: FL1}\n  FL2: {r: 0.45, z: -0.1, name: FL2}\n",
        encoding="utf-8",
    )
    # Names without positions, and positions without an identity, are not inventories.
    (data / "names.yaml").write_text("channels:\n  - {field_code: 101, name: B01}\n", encoding="utf-8")
    (data / "outline.yaml").write_text("limiter:\n  - {r: 0.1, z: 0.0}\n  - {r: 0.2, z: 0.1}\n", encoding="utf-8")
    findings = sensor_inventory_findings(tmp_path)
    assert [f.key for f in findings] == [
        ("vaft/data/new_probes.yaml", "loops{}"),
        ("vaft/data/new_probes.yaml", "probes.sets[].channels"),
    ]
    with pytest.raises(AssertionError, match=r"new_probes\.yaml: 'probes\.sets\[\]\.channels' holds sensor records"):
        check_against_allowlist(findings, {}, rule=INVENTORY_RULE, expected=INVENTORY_EXPECTED)


# ---------------------------------------------------------------------------
# (d) canonical registry schema, through B1's validator
# ---------------------------------------------------------------------------


def test_canonical_registry_schema_and_uniqueness(tmp_path):
    """75 records, unique id/index/effective field code, finite r/z, nonzero calibration, 64/11 kinds.

    Calls B1's own validator (vaft.machine_mapping.magnetics_registry), not a
    CI-only reinterpretation of the schema; negative control: duplicate id.
    The per-rule negative controls live in test/test_magnetics_registry.py.
    """
    import math

    from vaft.machine_mapping.utils import VestConfigurationError
    from vaft.machine_mapping.magnetics_registry import resolve_equilibrium_magnetics_sensors

    for shot in (None, 39437, 39438):
        records = resolve_equilibrium_magnetics_sensors(shot)
        kinds = [r.kind for r in records]
        assert kinds == ["b_field_pol_probe"] * 64 + ["flux_loop"] * 11
        assert [r.index for r in records] == list(range(75))
        for attribute in ("id", "field_code"):
            values = [getattr(r, attribute) for r in records]
            assert len(set(values)) == len(values), (shot, attribute)
        assert all(math.isfinite(r.r) and math.isfinite(r.z) and r.calibration != 0 for r in records)

    vest = REPO_ROOT / "vaft" / "machine_mapping" / "vest.yaml"
    content = yaml.safe_load(vest.read_text(encoding="utf-8"))
    records = content[0]["diagnostics"]["equilibrium_magnetics"]["sensors"]["records"]
    records[47] = dict(records[46], index=47)
    variant = tmp_path / "vest.yaml"
    variant.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
    with pytest.raises(VestConfigurationError, match="duplicate id"):
        resolve_equilibrium_magnetics_sensors(info_file=str(variant))


# ---------------------------------------------------------------------------
# (e) configuration loading: an explicit, missing path never falls back
# ---------------------------------------------------------------------------

MISSING_PATH_RULE = (
    "an explicit info_file that does not exist did not raise VestConfigurationError "
    "(#2168 Scope D3 / audit gap G1: silent fallback to the packaged vest.yaml)"
)
MISSING_PATH_EXPECTED = (
    "read the configuration through machine_mapping.utils._resolve_info_file_path "
    "(or _policy_document), which refuses a missing explicit path"
)

#: Arguments for the required parameters of a public resolver, by name, so
#: every ``info_file``-taking function is called without a hand-kept list. A
#: new resolver with an unlisted required parameter fails the guard, asking
#: for an entry here rather than being silently skipped.
_RESOLVER_ARGUMENTS = {
    "shot": lambda: 39915,
    "ods": lambda: importlib.import_module("omas").ODS(),
    "diagnostic": lambda: "tf",
    "diagnostics": lambda: ("tf",),
    "assembly_key": lambda: "langmuir_probes",
    "beam": lambda: 0,
    "entry": lambda: {"id": "fl_01"},
    "kind": lambda: "flux_loop",
    "context": lambda: "architecture guard",
}


def public_info_file_resolvers(package: str = "vaft.machine_mapping") -> list[tuple[str, object]]:
    """``(qualname, callable)`` for every public function in ``package`` with an ``info_file`` parameter.

    Only submodules whose source names ``info_file`` are imported. Decorated
    functions (``lru_cache``, ``functools.wraps``) are unwrapped to read their
    signature, but the decorated object is what gets called.
    """
    import importlib.util
    import inspect
    import pkgutil

    root = importlib.import_module(package)
    resolvers: list[tuple[str, object]] = []
    for info in sorted(pkgutil.walk_packages(root.__path__, prefix=f"{package}."), key=lambda i: i.name):
        spec = importlib.util.find_spec(info.name)
        origin = getattr(spec, "origin", None)
        if not origin or not origin.endswith(".py") or "info_file" not in Path(origin).read_text(encoding="utf-8"):
            continue
        module = importlib.import_module(info.name)
        for name, value in sorted(vars(module).items()):
            target = inspect.unwrap(value) if callable(value) else None
            if (
                name.startswith("_")
                or not inspect.isfunction(target)
                or target.__module__ != module.__name__
                or "info_file" not in inspect.signature(target).parameters
            ):
                continue
            resolvers.append((f"{module.__name__}.{name}", value))
    return resolvers


def missing_path_findings(resolvers, missing: str) -> list[Finding]:
    """Resolvers that accept a missing explicit ``info_file`` without :class:`VestConfigurationError`."""
    import inspect

    from vaft.machine_mapping.utils import VestConfigurationError

    findings: list[Finding] = []
    for qualname, resolver in resolvers:
        target = inspect.unwrap(resolver)
        arguments = {"info_file": missing}
        for name, parameter in inspect.signature(target).parameters.items():
            if name == "info_file" or parameter.default is not inspect.Parameter.empty:
                continue
            if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
                continue
            if name not in _RESOLVER_ARGUMENTS:
                raise AssertionError(
                    f"{qualname}: required parameter {name!r} has no entry in _RESOLVER_ARGUMENTS; "
                    "add a representative value so the missing-path guard can call it"
                )
            arguments[name] = _RESOLVER_ARGUMENTS[name]()
        location = f"{inspect.getsourcefile(target)}:{inspect.getsourcelines(target)[1]}"
        try:
            resolver(**arguments)
        except VestConfigurationError as error:
            if "does not exist" in str(error):
                continue
            detail = f"{qualname} raised VestConfigurationError for another reason: {error}"
        except Exception as error:  # noqa: BLE001 - the finding is the wrong exception
            detail = f"{qualname} raised {type(error).__name__} instead: {error}"
        else:
            detail = f"{qualname} returned normally"
        findings.append(Finding((qualname,), location, detail))
    return findings


def test_explicit_missing_config_path_raises(tmp_path):
    """Every public machine_mapping resolver taking ``info_file`` refuses a missing explicit path (G1, #2177)."""
    from vaft.machine_mapping.effective_config import resolve_vest_processing_config
    from vaft.machine_mapping.utils import _resolve_info_file_path

    resolvers = public_info_file_resolvers()
    # The discovery must see the resolvers B2 hardened; an empty or shrunken
    # list would make the guard pass vacuously.
    names = {qualname.rsplit(".", 1)[1] for qualname, _ in resolvers}
    assert {
        "resolve_vest_processing_config",
        "resolve_equilibrium_magnetics_sensors",
        "resolve_impa_config",
        "resolve_vest_diagnostic",
        "vest_processing_provenance",
        "vest_efit_coilset_policy",
    } <= names, sorted(names)
    missing = str(tmp_path / "campaign" / "vest.yaml")
    check_against_allowlist(
        missing_path_findings(resolvers, missing), {}, rule=MISSING_PATH_RULE, expected=MISSING_PATH_EXPECTED
    )
    # Control: the same explicit path, once it exists, is used as given -- the
    # refusal above is about the missing file, not about explicit paths.
    real = tmp_path / "campaign" / "vest.yaml"
    real.parent.mkdir()
    real.write_bytes((REPO_ROOT / "vaft" / "machine_mapping" / "vest.yaml").read_bytes())
    assert _resolve_info_file_path(str(real)) == str(real)
    resolved = resolve_vest_processing_config(39915, "tf", info_file=str(real))
    assert resolved.provenance["source"]["path"] == str(real)


def test_missing_path_guard_reports_a_silent_fallback(tmp_path):
    """Negative control: a resolver that falls back to the packaged file is reported, with its location."""
    import os

    from vaft.machine_mapping.utils import VestConfigurationError, package_data_path

    def falls_back(shot, info_file=None):
        path = info_file if info_file and os.path.exists(info_file) else package_data_path("vest.yaml")
        return path

    def wrong_error(shot, info_file=None):
        raise FileNotFoundError(info_file)

    def refuses(shot, info_file=None):
        raise VestConfigurationError(f"VEST configuration file {info_file!r} does not exist")

    findings = missing_path_findings(
        [("synthetic.falls_back", falls_back), ("synthetic.wrong_error", wrong_error), ("synthetic.refuses", refuses)],
        str(tmp_path / "nope.yaml"),
    )
    assert [f.key for f in findings] == [("synthetic.falls_back",), ("synthetic.wrong_error",)]
    assert "returned normally" in findings[0].detail and "FileNotFoundError" in findings[1].detail
    with pytest.raises(AssertionError, match=r"test_architecture_guards\.py:\d+: synthetic\.falls_back returned"):
        check_against_allowlist(findings, {}, rule=MISSING_PATH_RULE, expected=MISSING_PATH_EXPECTED)

    def needs_unknown(campaign, info_file=None):
        return None

    with pytest.raises(AssertionError, match=r"required parameter 'campaign' has no entry"):
        missing_path_findings([("synthetic.needs_unknown", needs_unknown)], str(tmp_path / "nope.yaml"))


# ---------------------------------------------------------------------------
# (f) overrides go through the validated semantics
# ---------------------------------------------------------------------------

#: (override, the diagnostic and shot it targets, the refusal it must produce).
#: One unsafe override of each class through each public override entry point:
#: the generic resolver (every diagnostic) and the equilibrium-magnetics path
#: the routine workflow's ``vest.magnetics.processing`` block takes.
UNSAFE_OVERRIDES = [
    pytest.param(
        "generic", 46404, "equilibrium_magnetics", {"processing": {"window": {"daq_mode": "legacy"}}}, "machine fact",
        id="generic-machine-fact-daq-mode",
    ),
    pytest.param(
        "generic", 39915, "plasma_current", {"source": {"field": 1}}, "machine fact",
        id="generic-machine-fact-source-field",
    ),
    pytest.param(
        "generic", 39915, "equilibrium_magnetics",
        {"sensors": {"records": []}}, "machine fact", id="generic-machine-fact-sensor-registry",
    ),
    pytest.param(
        "generic", 39915, "plasma_current", {"processing": {"baseline": {"lookbak": 1}}}, "names no configured key",
        id="generic-unknown-key",
    ),
    pytest.param(
        "generic", 39915, "plasma_current", {"calibration": {"factor": 1.0e-5}}, "calibration_override_reason",
        id="generic-calibration-without-reason",
    ),
    pytest.param(
        "generic", 39915, "pf_active", {"processing": {"coil_gains": {0: 1.0}}}, "calibration_override_reason",
        id="generic-coil-gain-without-reason",
    ),
    pytest.param("magnetics", 46404, None, {"daq_mode": "legacy"}, "cannot be overridden", id="magnetics-machine-fact"),
    pytest.param("magnetics", 39915, None, {"lowpass_cutof": 1.0}, "unknown", id="magnetics-unknown-key"),
    pytest.param(
        "magnetics", 39915, None, {"flux_output_per_radian": False}, "calibration_override_reason",
        id="magnetics-calibration-without-reason",
    ),
]


def _apply_override(entry: str, shot: int, diagnostic: str | None, override, **options):
    if entry == "generic":
        from vaft.machine_mapping.effective_config import resolve_vest_processing_config

        return resolve_vest_processing_config(shot, diagnostic, overrides=override, **options)
    from vaft.omas.vest_upstream import resolve_magnetics_processing

    return resolve_magnetics_processing(shot, override)


@pytest.mark.parametrize(("entry", "shot", "diagnostic", "override", "refusal"), UNSAFE_OVERRIDES)
def test_unsafe_override_is_rejected(entry, shot, diagnostic, override, refusal):
    """A machine fact, an unknown key or an unflagged calibration is refused (G5/G7, #2177/#2184)."""
    from vaft.machine_mapping.utils import VestConfigurationError

    with pytest.raises(VestConfigurationError, match=refusal):
        _apply_override(entry, shot, diagnostic, override)


def test_safe_overrides_are_accepted_and_recorded():
    """Positive controls: the refusals above are about the class of key, not about overriding at all."""
    from vaft.machine_mapping.effective_config import resolve_vest_processing_config
    from vaft.omas.vest_upstream import resolve_magnetics_processing

    processing = resolve_vest_processing_config(39915, "plasma_current", overrides={"processing": {"baseline": {"lookback": 10}}})
    assert processing.provenance["overrides"]["accepted"][0]["path"] == "processing.baseline.lookback"
    calibration = resolve_vest_processing_config(
        39915, "plasma_current", overrides={"calibration": {"factor": 1.0e-5}}, calibration_override_reason="bench"
    )
    assert calibration.provenance["overrides"]["calibration_override_reason"] == "bench"
    restated = resolve_vest_processing_config(
        46404, "equilibrium_magnetics", overrides={"processing": {"window": {"daq_mode": "native_daq"}}}
    )
    assert restated.provenance["overrides"]["accepted"] == ()
    assert resolve_magnetics_processing(46404, {"daq_mode": "native_daq"}).config.daq_mode == "native_daq"
    flagged = resolve_magnetics_processing(
        39915, {"flux_output_per_radian": False, "calibration_override_reason": "bench"}
    )
    assert flagged.overridden and flagged.config.flux_output_per_radian is False


OVERRIDE_SITE_RULE = (
    "an override mapping is merged or applied outside the validated owners "
    "(#2168 Scope D3: bypassing validated override semantics)"
)
OVERRIDE_SITE_EXPECTED = (
    "pass overrides to machine_mapping.effective_config.resolve_vest_processing_config, or for the "
    "equilibrium magnetics processing to vaft.omas.vest_upstream.resolve_magnetics_processing"
)

#: Functions that apply override mappings; ``_deep_merge``/``replace`` are the
#: two primitives, and an ``override``-named argument is what marks the call.
_OVERRIDE_APPLIERS = frozenset({"_deep_merge", "replace"})

#: (path, enclosing function) -> why this override application is validated.
OVERRIDE_SITE_ALLOWLIST: dict[tuple[str, str], str] = {
    ("vaft/machine_mapping/utils.py", "resolve_shot_revisions_with_provenance"):
        "applies a vest.yaml revision after the shared revision validator refused keys absent from the base (G5, #2177)",
    ("vaft/machine_mapping/utils.py", "resolve_diagnostics_time_policies"):
        "time-policy overrides are analysis windows by design (they may add windows or re-point components); "
        "every resolved window is validated (tend > tstart, dt > 0) before use",
    ("vaft/omas/vest_upstream.py", "resolve_magnetics_processing"):
        "the equilibrium-magnetics override path: unknown keys, machine facts and unflagged calibrations are "
        "refused before replace(), the result passes VestMagneticsProcessingConfig.__post_init__ (#2184)",
}


def override_site_findings(base: Path, roots: Iterable[str] = SCAN_ROOTS) -> list[Finding]:
    findings: list[Finding] = []
    for path in _python_sources(base, roots):
        text = path.read_text(encoding="utf-8")
        if "override" not in text or not any(f"{name}(" in text for name in _OVERRIDE_APPLIERS):
            continue
        rel = _relative(path, base)
        for qualname, node in _calls_with_scope(ast.parse(text, filename=str(path))):
            func = node.func
            called = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
            if called not in _OVERRIDE_APPLIERS:
                continue
            arguments = [*node.args, *(keyword.value for keyword in node.keywords)]
            names = {n.id for argument in arguments for n in ast.walk(argument) if isinstance(n, ast.Name)}
            if any("override" in name.lower() for name in names):
                findings.append(
                    Finding((rel, qualname), f"{rel}:{node.lineno}", f"in {qualname}: {ast.unparse(node)[:80]}")
                )
    return findings


def _calls_with_scope(tree: ast.AST) -> Iterable[tuple[str, ast.Call]]:
    """Every call with the qualname of its enclosing function or class."""

    def walk(node: ast.AST, stack: tuple[str, ...]):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                yield from walk(child, (*stack, child.name))
                continue
            if isinstance(child, ast.Call):
                yield ".".join(stack) or "<module>", child
            yield from walk(child, stack)

    yield from walk(tree, ())


def test_overrides_are_applied_only_by_validated_owners():
    check_against_allowlist(
        override_site_findings(REPO_ROOT),
        OVERRIDE_SITE_ALLOWLIST,
        rule=OVERRIDE_SITE_RULE,
        expected=OVERRIDE_SITE_EXPECTED,
    )


def test_override_site_scanner_reports_a_synthetic_bypass(tmp_path):
    (tmp_path / "workflow").mkdir()
    (tmp_path / "workflow" / "stage.py").write_text(
        "from dataclasses import replace\n"
        "from vaft.machine_mapping.utils import _deep_merge\n"
        "def run(config, block, overrides, window):\n"
        "    merged = _deep_merge(block, overrides['vest'])\n"
        "    tuned = replace(config, **overrides)\n"
        "    other = replace(config, window=window)  # no override: not an override site\n"
        "    return merged, tuned, 'x'.replace('a', 'b')\n",
        encoding="utf-8",
    )
    findings = override_site_findings(tmp_path)
    assert [(f.key, f.location) for f in findings] == [
        (("workflow/stage.py", "run"), "workflow/stage.py:4"),
        (("workflow/stage.py", "run"), "workflow/stage.py:5"),
    ]
    with pytest.raises(AssertionError, match=r"workflow/stage\.py:4: in run: _deep_merge\(block, overrides"):
        check_against_allowlist(findings, {}, rule=OVERRIDE_SITE_RULE, expected=OVERRIDE_SITE_EXPECTED)


# ---------------------------------------------------------------------------
# (g) no internal use of deprecated aliases
# ---------------------------------------------------------------------------

DEPRECATED_RULE = (
    "production code calls a deprecated compatibility alias (#2168 Scope E: deprecated-API usage; "
    "retirement register rows L5, L6, P9, P10)"
)

#: Fully qualified deprecated name -> its canonical replacement. A name counts
#: only when it *resolves* to one of these owners: ``vaft.code.efit.legacy``
#: defines its own, different ``vfit_signal_startend`` (threshold 0.06, the
#: H-alpha window), which is not the deprecated alias and is not reported.
DEPRECATED_APIS: dict[str, str] = {
    **{
        f"{owner}.{name}": replacement
        for owner in ("vaft.process.signal_processing", "vaft.process")
        for name, replacement in (
            ("VEST_CoilCurrentNoiseReduction", "vaft.process.signal_processing.vest_coil_current_noise_reduction"),
            ("vfit_signal_startend", "vaft.process.signal_processing.detect_active_window"),
            ("vfit_signal_start_end", "vaft.process.signal_processing.detect_active_window"),
            ("signal_onoffset", "vaft.process.signal_processing.signal_on_offset"),
        )
    },
    "vaft.machine_mapping.utils.process_signal": "vaft.process.process_signal",
    **{
        f"{owner}.{name}": replacement
        for owner in ("vaft.machine_mapping.magnetics", "vaft.machine_mapping")
        for name, replacement in (
            ("vest_md_channel_definitions", "vaft.machine_mapping.magnetics.vest_equilibrium_magnetics_channel_definitions"),
            ("vfit_md", "vaft.machine_mapping.magnetics.vfit_equilibrium_magnetics"),
        )
    },
}

#: (path, qualname, deprecated name) -> why it is still allowed. Empty: no
#: production module uses an alias, so any use is a regression.
DEPRECATED_ALLOWLIST: dict[tuple[str, str, str], str] = {}


def _module_name(rel: str) -> str:
    parts = list(Path(rel).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


class _DeprecatedUseCollector(ast.NodeVisitor):
    """Resolve imports, then report names and attribute chains that land on a deprecated alias.

    Bindings are collected per module (an import anywhere in the file binds
    its name for the attribute and name lookups); a function or class
    defining a deprecated *name* locally shadows nothing here, because only
    references that resolve through an import, or bare references inside the
    owner module itself, are reported.
    """

    def __init__(self, module: str, is_package: bool) -> None:
        self.module = module
        self.package = module if is_package else module.rpartition(".")[0]
        self.bindings: dict[str, str] = {}
        self.stack: list[str] = []
        self.uses: list[tuple[str, int, str]] = []

    @property
    def qualname(self) -> str:
        return ".".join(self.stack) or "<module>"

    def _absolute(self, node: ast.ImportFrom) -> str:
        if not node.level:
            return node.module or ""
        base = self.package.split(".")
        base = base[: len(base) - (node.level - 1)] if node.level > 1 else base
        return ".".join([*base, *([node.module] if node.module else [])])

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.asname:
                self.bindings[alias.asname] = alias.name
            else:
                root = alias.name.split(".")[0]
                self.bindings.setdefault(root, root)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        source = self._absolute(node)
        for alias in node.names:
            target = f"{source}.{alias.name}"
            self.bindings[alias.asname or alias.name] = target
            if target in DEPRECATED_APIS:
                self.uses.append((self.qualname, node.lineno, target))

    def _scoped(self, node: ast.AST, name: str) -> None:
        self.stack.append(name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._scoped(node, node.name)

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scoped(node, node.name)

    def _dotted(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return self.bindings.get(node.id)
        if isinstance(node, ast.Attribute):
            base = self._dotted(node.value)
            return f"{base}.{node.attr}" if base else None
        return None

    def visit_Attribute(self, node: ast.Attribute) -> None:
        dotted = self._dotted(node)
        if dotted in DEPRECATED_APIS:
            self.uses.append((self.qualname, node.lineno, dotted))
            return
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if not isinstance(node.ctx, ast.Load):
            return
        # A bare call of an alias inside its owner module.
        own = f"{self.module}.{node.id}"
        if own in DEPRECATED_APIS and node.id not in self.bindings:
            self.uses.append((self.qualname, node.lineno, own))

    def visit_Call(self, node: ast.Call) -> None:
        # getattr(module, "alias")
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            base = self._dotted(node.args[0])
            if base and f"{base}.{node.args[1].value}" in DEPRECATED_APIS:
                self.uses.append((self.qualname, node.lineno, f"{base}.{node.args[1].value}"))
        self.generic_visit(node)


_DEPRECATED_NEEDLES = tuple(sorted({name.rsplit(".", 1)[1] for name in DEPRECATED_APIS}))


def deprecated_use_findings(base: Path, roots: Iterable[str] = SCAN_ROOTS) -> list[Finding]:
    findings: list[Finding] = []
    for path in _python_sources(base, roots):
        text = path.read_text(encoding="utf-8")
        if not any(needle in text for needle in _DEPRECATED_NEEDLES):
            continue
        rel = _relative(path, base)
        collector = _DeprecatedUseCollector(_module_name(rel), path.name == "__init__.py")
        collector.visit(ast.parse(text, filename=str(path)))
        for qualname, line, name in collector.uses:
            short = name.rsplit(".", 1)[1]
            findings.append(
                Finding(
                    (rel, qualname, short),
                    f"{rel}:{line}",
                    f"in {qualname}: uses deprecated {name}; use {DEPRECATED_APIS[name]}",
                )
            )
    return findings


def test_no_internal_use_of_deprecated_apis():
    """No module under vaft/ or workflow/ uses a deprecated alias (register rows L5, L6, P9, P10)."""
    check_against_allowlist(
        deprecated_use_findings(REPO_ROOT),
        DEPRECATED_ALLOWLIST,
        rule=DEPRECATED_RULE,
        expected="the canonical replacement named in the finding",
    )


def test_deprecated_api_scanner_reports_synthetic_uses_and_ignores_the_efit_specialization(tmp_path):
    files = {
        "workflow/stage.py": (
            "from vaft.process.signal_processing import signal_onoffset\n"
            "import vaft.machine_mapping as mm\n"
            "import vaft.process.signal_processing as sp\n"
            "def run(t, y):\n"
            "    a = mm.vfit_md(39915)\n"
            "    b = sp.VEST_CoilCurrentNoiseReduction(y)\n"
            "    c = getattr(mm, 'vest_md_channel_definitions')()\n"
            "    d = sp.signal_on_offset(t, y)  # the canonical name is fine\n"
            "    return a, b, c, d\n"
        ),
        "vaft/machine_mapping/stage.py": (
            "from .utils import process_signal\n"
            "from . import magnetics\n"
            "def f():\n"
            "    return magnetics.vest_md_channel_definitions()\n"
        ),
        # The EFIT H-alpha window keeps its own, different function of the alias's name.
        "vaft/code/efit/legacy.py": (
            "def vfit_signal_startend(time, data):\n    return time[0], time[-1]\n"
            "def window(t, y):\n    return vfit_signal_startend(t, y)\n"
        ),
        # Inside the owner module a bare call of an alias is a use; its definition is not.
        "vaft/process/signal_processing.py": (
            "def signal_on_offset(t, y):\n    return t\n"
            "def signal_onoffset(t, y):\n    return signal_on_offset(t, y)\n"
            "def helper(t, y):\n    return signal_onoffset(t, y)\n"
        ),
        "vaft/process/fine.py": "from .signal_processing import detect_active_window, process_signal\n",
    }
    for relative, text in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    findings = deprecated_use_findings(tmp_path)
    assert sorted((f.key, f.location) for f in findings) == [
        (("vaft/machine_mapping/stage.py", "<module>", "process_signal"), "vaft/machine_mapping/stage.py:1"),
        (("vaft/machine_mapping/stage.py", "f", "vest_md_channel_definitions"), "vaft/machine_mapping/stage.py:4"),
        (("vaft/process/signal_processing.py", "helper", "signal_onoffset"), "vaft/process/signal_processing.py:6"),
        (("workflow/stage.py", "<module>", "signal_onoffset"), "workflow/stage.py:1"),
        (("workflow/stage.py", "run", "VEST_CoilCurrentNoiseReduction"), "workflow/stage.py:6"),
        (("workflow/stage.py", "run", "vest_md_channel_definitions"), "workflow/stage.py:7"),
        (("workflow/stage.py", "run", "vfit_md"), "workflow/stage.py:5"),
    ]
    with pytest.raises(
        AssertionError,
        match=r"workflow/stage\.py:5: in run: uses deprecated vaft\.machine_mapping\.vfit_md; "
        r"use vaft\.machine_mapping\.magnetics\.vfit_equilibrium_magnetics",
    ):
        check_against_allowlist(findings, {}, rule=DEPRECATED_RULE, expected="x")


# ---------------------------------------------------------------------------
# (h) vaft.process reads no VEST configuration and branches on no shot number
# ---------------------------------------------------------------------------

PROCESS_CONFIG_RULE = (
    "vaft.process reads VEST configuration or encodes a machine era (#2168 Scope E1: "
    "vaft.process is machine-independent; CONTRIBUTING 'Machine policy does not live in vaft/process')"
)
PROCESS_CONFIG_EXPECTED = (
    "resolve the value in vaft.machine_mapping (vest.yaml, shot-era revisions) and pass it in as an argument"
)

#: An executable string that *is* a path to vest.yaml (not prose mentioning
#: it): the whole literal ends in the file name.
_VEST_YAML_PATH = re.compile(r"(?:^|[/\\])vest\.ya?ml$", re.IGNORECASE)
#: A literal that is exactly a machine_mapping module or the packaged
#: geometry data -- what importlib.import_module / importlib.resources.files
#: would be handed. The static imports are the layer guard's (b) business.
_MACHINE_MODULE = re.compile(r"^(?:vaft\.machine_mapping|vaft\.data\.geometry)(?:\.\w+)*$")
#: Names that hold a shot number.
_SHOT_NAME = re.compile(r"^(?:shot|shot_?number|shotnumber|shot_no|nshot|pulse(?:_number)?)$", re.IGNORECASE)
#: The smallest literal that can be a VEST shot boundary (a ``shot < 0`` or
#: ``shot is None`` validation is not an era).
_SHOT_ERA_MINIMUM = 1000

#: (path, qualname, kind) -> justification. Empty today.
PROCESS_CONFIG_ALLOWLIST: dict[tuple[str, str, str], str] = {}


def _name_of(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def process_config_findings(base: Path, roots: Iterable[str] = ("vaft/process",)) -> list[Finding]:
    findings: list[Finding] = []
    for path in _python_sources(base, roots):
        rel = _relative(path, base)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        docstrings = {
            id(node.value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        }
        for qualname, node in _nodes_with_scope(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                value = node.value.strip()
                if _VEST_YAML_PATH.search(value):
                    findings.append(Finding((rel, qualname, "vest.yaml path"), f"{rel}:{node.lineno}", f"in {qualname}: names the file {value!r}"))
                elif _MACHINE_MODULE.match(value):
                    findings.append(Finding((rel, qualname, "machine module"), f"{rel}:{node.lineno}", f"in {qualname}: dynamic reference to {value!r}"))
            elif isinstance(node, ast.Compare):
                operands = [node.left, *node.comparators]
                if not any(_SHOT_NAME.match(_name_of(o) or "") for o in operands):
                    continue
                eras = [
                    o.value for o in operands
                    if isinstance(o, ast.Constant) and isinstance(o.value, (int, float))
                    and not isinstance(o.value, bool) and abs(o.value) >= _SHOT_ERA_MINIMUM
                ]
                if eras:
                    findings.append(
                        Finding((rel, qualname, "shot era"), f"{rel}:{node.lineno}", f"in {qualname}: {ast.unparse(node)}")
                    )
    return findings


def _nodes_with_scope(tree: ast.AST) -> Iterable[tuple[str, ast.AST]]:
    def walk(node: ast.AST, stack: tuple[str, ...]):
        for child in ast.iter_child_nodes(node):
            scope = (*stack, child.name) if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else stack
            yield ".".join(scope) or "<module>", child
            yield from walk(child, scope)

    yield from walk(tree, ())


def test_process_reads_no_vest_configuration_and_no_shot_era():
    check_against_allowlist(
        process_config_findings(REPO_ROOT),
        PROCESS_CONFIG_ALLOWLIST,
        rule=PROCESS_CONFIG_RULE,
        expected=PROCESS_CONFIG_EXPECTED,
    )


def test_process_config_scanner_reports_synthetic_violations_and_keeps_numerics_legal(tmp_path):
    package = tmp_path / "vaft" / "process"
    package.mkdir(parents=True)
    (package / "bad.py").write_text(
        textwrap.dedent(
            '''
            """Docstring citing vaft/machine_mapping/vest.yaml is documentation."""
            import importlib
            from importlib import resources
            import numpy as np
            from scipy import integrate, signal

            def policy(shot):
                root = resources.files("vaft.machine_mapping") / "vest.yaml"
                module = importlib.import_module("vaft.machine_mapping.magnetics")
                if shot >= 39438:
                    return root, module
                return 46404 < shot

            def fine(t, y, shot, fs):
                taps = signal.firwin(251, 0.1)
                grid = np.interp(t, t, y)
                total = integrate.trapezoid(y, t)
                if shot < 0 or shot is None:
                    raise ValueError("vest.yaml configures no policy for this shot")
                return taps, grid, total, fs >= 25000
            '''
        ),
        encoding="utf-8",
    )
    findings = process_config_findings(tmp_path)
    assert sorted((f.key, f.location) for f in findings) == [
        (("vaft/process/bad.py", "policy", "machine module"), "vaft/process/bad.py:10"),
        (("vaft/process/bad.py", "policy", "machine module"), "vaft/process/bad.py:9"),
        (("vaft/process/bad.py", "policy", "shot era"), "vaft/process/bad.py:11"),
        (("vaft/process/bad.py", "policy", "shot era"), "vaft/process/bad.py:13"),
        (("vaft/process/bad.py", "policy", "vest.yaml path"), "vaft/process/bad.py:9"),
    ]
    with pytest.raises(AssertionError, match=r"vaft/process/bad\.py:11: in policy: shot >= 39438(.|\n)*pass it in"):
        check_against_allowlist(findings, {}, rule=PROCESS_CONFIG_RULE, expected=PROCESS_CONFIG_EXPECTED)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-q", "-n", "0"]))
