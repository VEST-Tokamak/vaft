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

Placeholders (skipped until their owners land): registry schema/uniqueness
(#2158), explicit-missing-config-path raises and unsafe-override rejection
(#2159), deprecated-API usage.

The register these guards serve is
``docs/architecture/consolidation/retirement_register.md``.
"""

from __future__ import annotations

import ast
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
# (d) placeholders, enabled when their owners land
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


@pytest.mark.skip(reason="enabled by #2158/#2159")
def test_explicit_missing_config_path_raises():
    """_resolve_info_file_path with an explicit, missing info_file raises VestConfigurationError.

    Negative control: today it silently falls back to the packaged vest.yaml.
    """


@pytest.mark.skip(reason="enabled by #2158/#2159")
def test_unsafe_override_is_rejected():
    """Overriding a machine fact (sensors, wiring, daq_mode) or an unknown key raises; calibration needs the flag."""


@pytest.mark.skip(reason="enabled by #2158/#2159")
def test_no_internal_use_of_deprecated_apis():
    """No module under vaft/ or workflow/ calls a deprecated alias (vfit_md, vest_md_channel_definitions,
    vfit_signal_startend, signal_onoffset, VEST_CoilCurrentNoiseReduction, machine_mapping.utils.process_signal).

    Enabled once B3 (#2162) migrates vaft/code/efit/legacy.py off vfit_signal_startend.
    """


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-q", "-n", "0"]))
