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
(e) An explicit ``info_file`` that does not exist raises, naming the path:
    every public function with an ``info_file`` parameter in
    ``vaft.machine_mapping``, ``vaft.omas`` and ``vaft.process`` is discovered
    and called with a missing path (audit gap G1, #2159).
(f) Overrides go through the validated semantics: machine facts, unknown keys
    and calibrations without ``calibration_override_reason`` are refused by
    both public override entry points, and no code outside the validated
    owners merges or applies an ``override`` mapping (G5/G7, #2159).
(g) No module under ``vaft/`` or ``workflow/`` uses a deprecated alias
    (register rows L5, L6, P9, P10). Names are resolved through the imports,
    so ``vaft.code.efit.legacy``'s own, different ``vfit_signal_startend`` is
    not the alias.
(h) ``vaft.process`` names no ``vest.yaml`` path, references no
    ``vaft.machine_mapping`` module dynamically, compares no shot number
    with an era literal and binds no shot-boundary name to one. ``np.interp``, ``scipy.signal`` and integration stay
    legal: the guard targets configuration reads, not numerics.
(i) No digital filter is designed (``scipy.signal.firwin``, ``butter``, ...)
    outside ``vaft.process``, apart from two allowlisted donor designs;
    applying a filter stays legal everywhere.

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

#: Extra keyword arguments for one resolver, when the generic values above do
#: not reach the configuration read (a resolver that needs a mode selected).
_RESOLVER_OPTIONS = {
    "vaft.process.impurity.resolve_impurity_composition": {"machine_preset": "vest", "shot": 39915},
}

#: Packages whose public ``info_file`` functions are discovered. The config
#: file is read only in ``vaft.machine_mapping``; the other two pass
#: ``info_file`` through to it, and are called so a pass-through that drops
#: the path (and lets the resolver fall back) is caught too.
_RESOLVER_PACKAGES = ("vaft.machine_mapping", "vaft.omas", "vaft.process")


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
    """Resolvers that accept a missing explicit ``info_file`` without the refusal naming that path.

    The refusal must be a :class:`VestConfigurationError` whose message names
    ``missing``: a resolver that ignores ``info_file`` and fails on something
    else ("diagnostic 'tf' does not exist") is a finding, not a pass.
    """
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
        arguments.update(_RESOLVER_OPTIONS.get(qualname, {}))
        try:
            location = f"{inspect.getsourcefile(target)}:{inspect.getsourcelines(target)[1]}"
        except (OSError, TypeError):
            location = qualname
        try:
            resolver(**arguments)
        except VestConfigurationError as error:
            # The message quotes the path with !r, which doubles backslashes on Windows.
            if (missing in str(error) or repr(missing) in str(error)) and "does not exist" in str(error):
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
    from vaft.machine_mapping.utils import VestConfigurationError, _resolve_info_file_path

    resolvers = [resolver for package in _RESOLVER_PACKAGES for resolver in public_info_file_resolvers(package)]
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
        "edge_q_estimate",
        "resolve_impurity_composition",
    } <= names, sorted(names)
    missing = str(tmp_path / "campaign" / "vest.yaml")
    check_against_allowlist(
        missing_path_findings(resolvers, missing), {}, rule=MISSING_PATH_RULE, expected=MISSING_PATH_EXPECTED
    )
    # The public getters that spell the path differently (`file`, or
    # `options["info_file"]`) are outside the discovery; they are called here.
    from vaft.machine_mapping.utils import get_diagnostic_info, get_static_info, raw_database_info

    spelled_differently = [
        ("raw_database_info", lambda: raw_database_info(missing, 39915, "pf_active")),
        ("get_diagnostic_info", lambda: get_diagnostic_info("vest", "tf", {"info_file": missing})),
        ("get_static_info", lambda: get_static_info("vest", "tf", {"info_file": missing})),
    ]
    for name, call in spelled_differently:
        with pytest.raises(VestConfigurationError, match="does not exist") as refused:
            call()
        assert missing in str(refused.value) or repr(missing) in str(refused.value), name
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

    def ignores_the_path(shot, info_file=None):
        raise VestConfigurationError("diagnostic 'tf' does not exist")

    findings = missing_path_findings(
        [
            ("synthetic.falls_back", falls_back),
            ("synthetic.wrong_error", wrong_error),
            ("synthetic.refuses", refuses),
            ("synthetic.ignores_the_path", ignores_the_path),
        ],
        str(tmp_path / "nope.yaml"),
    )
    assert [f.key for f in findings] == [
        ("synthetic.falls_back",),
        ("synthetic.wrong_error",),
        ("synthetic.ignores_the_path",),
    ]
    assert "returned normally" in findings[0].detail and "FileNotFoundError" in findings[1].detail
    assert "for another reason: diagnostic 'tf' does not exist" in findings[2].detail
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
        "generic", 39915, "plasma_current", {"calibration": {"factor": 1.0e-5}}, "is a calibration; overriding it needs",
        id="generic-calibration-without-reason",
    ),
    pytest.param(
        "generic", 39915, "pf_active", {"processing": {"coil_gains": {0: 1.0}}}, "is a calibration; overriding it needs",
        id="generic-coil-gain-without-reason",
    ),
    pytest.param("magnetics", 46404, None, {"daq_mode": "legacy"}, "cannot be overridden", id="magnetics-machine-fact"),
    pytest.param(
        "magnetics", 39915, None, {"lowpass_cutof": 1.0}, "unknown magnetics processing key",
        id="magnetics-unknown-key",
    ),
    pytest.param(
        "magnetics", 39915, None, {"flux_output_per_radian": False}, "change the calibration of every channel",
        id="magnetics-calibration-without-reason",
    ),
]


def _apply_override(entry: str, shot: int, diagnostic: str | None, override):
    if entry == "generic":
        from vaft.machine_mapping.effective_config import resolve_vest_processing_config

        return resolve_vest_processing_config(shot, diagnostic, overrides=override)
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

#: The two primitives that apply an override mapping, resolved through the
#: module's imports (so ``from ...utils import _deep_merge as merge`` is still
#: ``_deep_merge``; ``str.replace`` is not ``dataclasses.replace``).
_OVERRIDE_APPLIERS = ("_deep_merge", "dataclasses.replace", "copy.replace")

#: (path, enclosing function, the override argument) -> why this application
#: is validated. The validated merge in ``effective_config`` applies the
#: already-validated ``changes``, not the raw override, and is not a site.
OVERRIDE_SITE_ALLOWLIST: dict[tuple[str, str, str], str] = {
    ("vaft/machine_mapping/utils.py", "resolve_shot_revisions_with_provenance", "override"):
        "applies a vest.yaml revision after the shared revision validator refused keys absent from the base (G5, #2177)",
    ("vaft/machine_mapping/utils.py", "resolve_diagnostics_time_policies", "overrides"):
        "time-policy overrides are analysis windows by design (they may add windows or re-point components); "
        "every resolved window is validated (tend > tstart, dt > 0) before use",
    ("vaft/omas/vest_upstream.py", "resolve_magnetics_processing", "overrides"):
        "the equilibrium-magnetics override path: unknown keys, machine facts and unflagged calibrations are "
        "refused before replace(), the result passes VestMagneticsProcessingConfig.__post_init__ (#2184)",
}


def _import_bindings(tree: ast.AST) -> dict[str, str]:
    """Local name -> fully qualified name, for every import in the module (relative ones kept relative)."""
    bindings: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bindings[alias.asname or alias.name.split(".")[0]] = alias.name if alias.asname else alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            source = "." * node.level + (node.module or "")
            for alias in node.names:
                bindings[alias.asname or alias.name] = f"{source}.{alias.name}" if source else alias.name
    return bindings


def _qualified(node: ast.AST, bindings: Mapping[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        return bindings.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        base = _qualified(node.value, bindings)
        return f"{base}.{node.attr}" if base else None
    return None


def _is_override_applier(qualified: str | None) -> bool:
    if not qualified:
        return False
    return qualified == "_deep_merge" or qualified.endswith("._deep_merge") or qualified in _OVERRIDE_APPLIERS


def _override_markers(argument: ast.AST) -> list[str]:
    """The override-named parts of an argument: names, attributes and string subscript keys."""
    markers = []
    for node in ast.walk(argument):
        label = (
            node.id if isinstance(node, ast.Name)
            else node.attr if isinstance(node, ast.Attribute)
            else node.slice.value if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
            else None
        )
        if label and "override" in label.lower():
            markers.append(label)
    return markers


def override_site_findings(base: Path, roots: Iterable[str] = SCAN_ROOTS) -> list[Finding]:
    """Calls of ``_deep_merge``/``dataclasses.replace`` with an argument whose name contains ``override``."""
    findings: list[Finding] = []
    for path in _python_sources(base, roots):
        text = path.read_text(encoding="utf-8")
        if "override" not in text.lower() or ("_deep_merge" not in text and "replace" not in text):
            continue
        rel = _relative(path, base)
        tree = ast.parse(text, filename=str(path))
        bindings = _import_bindings(tree)
        for qualname, node in _calls_with_scope(tree):
            if not _is_override_applier(_qualified(node.func, bindings)):
                continue
            for argument in [*node.args, *(keyword.value for keyword in node.keywords)]:
                for marker in _override_markers(argument):
                    findings.append(
                        Finding(
                            (rel, qualname, marker),
                            f"{rel}:{node.lineno}",
                            f"in {qualname}: {ast.unparse(node)[:80]}",
                        )
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
        "import dataclasses\n"
        "from dataclasses import replace\n"
        "from vaft.machine_mapping.utils import _deep_merge as merge\n"
        "def run(config, block, overrides, window, options, label):\n"
        "    merged = merge(block, overrides['vest'])\n"
        "    tuned = replace(config, **options['override'])\n"
        "    again = dataclasses.replace(config, window=self.window_override)\n"
        "    other = replace(config, window=window)  # no override: not an override site\n"
        "    return merged, tuned, again, 'x'.replace('a', label + str(overrides))\n",
        encoding="utf-8",
    )
    findings = override_site_findings(tmp_path)
    assert [(f.key, f.location) for f in findings] == [
        (("workflow/stage.py", "run", "overrides"), "workflow/stage.py:5"),
        (("workflow/stage.py", "run", "override"), "workflow/stage.py:6"),
        (("workflow/stage.py", "run", "window_override"), "workflow/stage.py:7"),
    ]
    with pytest.raises(AssertionError, match=r"workflow/stage\.py:5: in run: merge\(block, overrides"):
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

    Bindings are module-wide and accumulate in source order: an import binds
    its name for every later lookup in the file, a function-local import
    included (a slight over-approximation; a use above its import is missed).
    A function or class defining a deprecated *name* locally shadows nothing
    here, because only references that resolve through an import, bare
    references inside the owner module itself, and star imports from an owner
    module are reported.
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
            if alias.name == "*":
                if source in _DEPRECATED_OWNERS:
                    # Every deprecated name of the owner becomes a bare name here.
                    self.uses.append((self.qualname, node.lineno, f"{source}.*"))
                continue
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
_DEPRECATED_OWNERS = frozenset(name.rsplit(".", 1)[0] for name in DEPRECATED_APIS)


def _deprecated_detail(name: str) -> tuple[str, str]:
    if name.endswith(".*"):
        owner = name[:-2]
        return "*", f"star-imports {owner}, which exports deprecated aliases; import the canonical names explicitly"
    return name.rsplit(".", 1)[1], f"uses deprecated {name}; use {DEPRECATED_APIS[name]}"


def deprecated_use_findings(base: Path, roots: Iterable[str] = SCAN_ROOTS) -> list[Finding]:
    findings: list[Finding] = []
    for path in _python_sources(base, roots):
        text = path.read_text(encoding="utf-8")
        if "import *" not in text and not any(needle in text for needle in _DEPRECATED_NEEDLES):
            continue
        rel = _relative(path, base)
        collector = _DeprecatedUseCollector(_module_name(rel), path.name == "__init__.py")
        collector.visit(ast.parse(text, filename=str(path)))
        for qualname, line, name in collector.uses:
            short, detail = _deprecated_detail(name)
            findings.append(Finding((rel, qualname, short), f"{rel}:{line}", f"in {qualname}: {detail}"))
    for path in _snakemake_sources(base, roots):
        findings.extend(_snakefile_deprecated_uses(path, base))
    return findings


def _snakefile_deprecated_uses(path: Path, base: Path) -> list[Finding]:
    """Deprecated aliases a Snakefile imports or reads through its imports.

    Snakefiles cannot be parsed as Python, but their import lines can: those
    are parsed for the bindings, and NAME tokens (``name`` or ``bound.name``)
    are resolved through them, so ``from vaft.process import process_signal``
    -- the canonical function -- is not the deprecated
    ``machine_mapping.utils.process_signal``.
    """
    text = path.read_text(encoding="utf-8")
    if not any(needle in text for needle in _DEPRECATED_NEEDLES):
        return []
    rel = _relative(path, base)
    imports = [line.strip() for line in text.splitlines() if re.match(r"\s*(?:from\s+\S+\s+)?import\s", line)]
    collector = _DeprecatedUseCollector("<snakefile>", False)
    for statement in imports:
        try:
            collector.visit(ast.parse(statement))
        except SyntaxError:  # a continued import line; its names are not resolved
            continue
    findings: list[Finding] = []
    tokens = [t for t in tokenize.generate_tokens(io.StringIO(text).readline) if t.type in (tokenize.NAME, tokenize.OP)]
    for index, token in enumerate(tokens):
        if token.type != tokenize.NAME:
            continue
        if index >= 2 and tokens[index - 1].string == "." and tokens[index - 2].type == tokenize.NAME:
            base_name = collector.bindings.get(tokens[index - 2].string)
            dotted = f"{base_name}.{token.string}" if base_name else None
        else:
            dotted = collector.bindings.get(token.string)
        if dotted in DEPRECATED_APIS:
            findings.append(
                Finding(
                    (rel, "<snakefile>", token.string),
                    f"{rel}:{token.start[0]}",
                    f"uses deprecated {dotted}; use {DEPRECATED_APIS[dotted]}",
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
        "workflow/star.py": "from vaft.process.signal_processing import *\nfrom vaft.formula import *\n",
        "workflow/pipeline/Snakefile": (
            "from vaft.machine_mapping import vfit_md\n"
            "from vaft.process import process_signal\n"
            "import vaft.machine_mapping.utils as mmu\n"
            "rule probes:\n    run:\n        vfit_md(39915)\n"
            "        process_signal(t, y, {})  # the canonical function\n"
            "        mmu.process_signal(t, y)\n"
            "rule names:\n    shell: 'echo vest_md_channel_definitions'\n"
        ),
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
        (("workflow/pipeline/Snakefile", "<snakefile>", "process_signal"), "workflow/pipeline/Snakefile:8"),
        (("workflow/pipeline/Snakefile", "<snakefile>", "vfit_md"), "workflow/pipeline/Snakefile:1"),
        (("workflow/pipeline/Snakefile", "<snakefile>", "vfit_md"), "workflow/pipeline/Snakefile:6"),
        (("workflow/stage.py", "<module>", "signal_onoffset"), "workflow/stage.py:1"),
        (("workflow/stage.py", "run", "VEST_CoilCurrentNoiseReduction"), "workflow/stage.py:6"),
        (("workflow/stage.py", "run", "vest_md_channel_definitions"), "workflow/stage.py:7"),
        (("workflow/stage.py", "run", "vfit_md"), "workflow/stage.py:5"),
        (("workflow/star.py", "<module>", "*"), "workflow/star.py:1"),
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

#: An executable string that *is* a path to vest.yaml (no whitespace, so not
#: prose mentioning it): the whole literal ends in the file name.
_VEST_YAML_PATH = re.compile(r"^\S*(?:^|[/\\])vest\.ya?ml$", re.IGNORECASE)
#: A literal that is exactly a machine_mapping module or the packaged
#: geometry data -- what importlib.import_module / importlib.resources.files
#: would be handed. The static imports are the layer guard's (b) business.
_MACHINE_MODULE = re.compile(r"^(?:vaft\.machine_mapping|vaft\.data\.geometry)(?:\.\w+)*$")
#: Names that hold a shot number, in a comparison.
_SHOT_NAME = re.compile(r"^(?:shots?|shot_?number|shotnumber|shot_no|shot_id|nshot|pulse(?:_number)?)$", re.IGNORECASE)
#: Names that hold a shot *boundary* when bound to a literal (``late_shot_min``,
#: ``first_shot``): a constant or default carrying a machine era.
_SHOT_BOUNDARY_NAME = re.compile(
    r"^(?:shot|pulse(?:_number)?)$"
    r"|(?:^|_)(?:first|last|min|max|start|end|from|to|late|early)_shot(?:$|_)"
    r"|(?:^|_)shot_(?:min|max|start|end|first|last|from|to|boundary|threshold|limit|cutoff)$",
    re.IGNORECASE,
)
#: The smallest literal that can be a VEST shot boundary (a ``shot < 0`` or
#: ``shot is None`` validation is not an era).
_SHOT_ERA_MINIMUM = 1000

#: (path, qualname, kind) -> justification.
PROCESS_CONFIG_ALLOWLIST: dict[tuple[str, str, str], str] = {
    **{
        ("vaft/process/magnetics.py", "VestMagneticsProcessingConfig", f"shot era constant {name}"): (
            "the legacy hardcoded window ladder (late_shot_min 41660, transient_shot_min/max 41446/41451) "
            "that window_for_shot falls back to only when no window_override is set; every vest.yaml era sets "
            "one (checked over shots 30000-50000), so the ladder is a legacy default, not the policy in force "
            "(debt, #2206: move or delete with the ladder)"
        )
        for name in ("late_shot_min", "transient_shot_min", "transient_shot_max")
    },
}


def _name_of(node: ast.AST) -> str | None:
    """The name a comparison operand reads: a name, an attribute, a string subscript key, or int()/float() of one."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
        return node.slice.value.rsplit(".", 1)[-1]
    if isinstance(node, ast.Call) and _name_of(node.func) in {"int", "float", "int64", "asarray"} and node.args:
        return _name_of(node.args[0])
    return None


def _era_literals(node: ast.AST) -> list[float]:
    """Numeric literals of shot-era magnitude in ``node`` (a constant, or a tuple/list/set of them)."""
    values = node.elts if isinstance(node, (ast.Tuple, ast.List, ast.Set)) else [node]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        values = [node.operand]
    return [
        v.value for v in values
        if isinstance(v, ast.Constant) and isinstance(v.value, (int, float))
        and not isinstance(v.value, bool) and abs(v.value) >= _SHOT_ERA_MINIMUM
    ]


def process_config_findings(base: Path, roots: Iterable[str] = ("vaft/process",)) -> list[Finding]:
    findings: list[Finding] = []

    def add(rel, qualname, kind, node, detail):
        findings.append(Finding((rel, qualname, kind), f"{rel}:{node.lineno}", f"in {qualname}: {detail}"))

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
                    add(rel, qualname, "vest.yaml path", node, f"names the file {value!r}")
                elif _MACHINE_MODULE.match(value):
                    add(rel, qualname, "machine module", node, f"dynamic reference to {value!r}")
            elif isinstance(node, ast.Compare):
                operands = [node.left, *node.comparators]
                if any(_SHOT_NAME.match(_name_of(o) or "") for o in operands) and any(
                    _era_literals(o) for o in operands
                ):
                    add(rel, qualname, "shot era", node, ast.unparse(node))
            elif isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names = [_name_of(t) or "" for t in targets]
                bound = [n for n in names if _SHOT_BOUNDARY_NAME.search(n)]
                if bound and _era_literals(node.value):
                    add(rel, qualname, f"shot era constant {bound[0]}", node, ast.unparse(node))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                positional = [*node.args.posonlyargs, *node.args.args]
                pairs = list(zip(positional[len(positional) - len(node.args.defaults):], node.args.defaults))
                pairs += [(a, d) for a, d in zip(node.args.kwonlyargs, node.args.kw_defaults) if d is not None]
                for argument, default in pairs:
                    if _SHOT_BOUNDARY_NAME.search(argument.arg) and _era_literals(default):
                        add(
                            rel, qualname, f"shot era constant {argument.arg}", default,
                            f"default {argument.arg}={ast.unparse(default)}",
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
            '''\
            """Docstring citing vaft/machine_mapping/vest.yaml is documentation."""
            import importlib
            from dataclasses import dataclass
            from importlib import resources
            import numpy as np
            from scipy import integrate, signal

            def policy(shot, ods, first_shot=39438):
                root = resources.files("vaft.machine_mapping") / "vest.yaml"
                module = importlib.import_module("vaft.machine_mapping.magnetics")
                if shot >= 39438 or int(shot) in (46403, 46404):
                    return root, module
                return ods["dataset_description.data_entry.pulse"] > 40000

            @dataclass
            class Ladder:
                late_shot_min: int = 41660
                sample_count: int = 25000
                MAX_SHOTS = 5000
                shot_window_ms: float = 1000.0

            def fine(t, y, shot, fs, first_shot):
                taps = signal.firwin(251, 0.1)
                grid = np.interp(t, t, y)
                total = integrate.trapezoid(y, t)
                if shot < 0 or shot is None or shot >= first_shot:
                    raise ValueError("see vaft/machine_mapping/vest.yaml: no policy for this shot")
                return taps, grid, total, fs >= 25000
            '''
        ),
        encoding="utf-8",
    )
    findings = process_config_findings(tmp_path)
    assert sorted((f.key[1:], f.location) for f in findings) == [
        (("Ladder", "shot era constant late_shot_min"), "vaft/process/bad.py:17"),
        (("policy", "machine module"), "vaft/process/bad.py:10"),
        (("policy", "machine module"), "vaft/process/bad.py:9"),
        (("policy", "shot era"), "vaft/process/bad.py:11"),  # shot >= 39438
        (("policy", "shot era"), "vaft/process/bad.py:11"),  # int(shot) in (46403, 46404)
        (("policy", "shot era"), "vaft/process/bad.py:13"),  # ods[...pulse] > 40000
        (("policy", "shot era constant first_shot"), "vaft/process/bad.py:8"),
        (("policy", "vest.yaml path"), "vaft/process/bad.py:9"),
    ]
    with pytest.raises(AssertionError, match=r"vaft/process/bad\.py:11: in policy: shot >= 39438(.|\n)*pass it in"):
        check_against_allowlist(findings, {}, rule=PROCESS_CONFIG_RULE, expected=PROCESS_CONFIG_EXPECTED)


# ---------------------------------------------------------------------------
# (i) filters are designed in vaft.process
# ---------------------------------------------------------------------------

FILTER_DESIGN_RULE = (
    "a digital filter is designed outside vaft.process (#2168 / migration_plan.md E: "
    "machine_mapping must not design filters outside an allowlist)"
)
FILTER_DESIGN_EXPECTED = (
    "design the filter with vaft.process.signal_processing (fir_filter_coefficients, pf_current_filter, "
    "butterworth_highpass, ...) and pass the machine values from vest.yaml"
)

#: scipy.signal design functions; applying a filter (lfilter, filtfilt,
#: sosfilt) is legal anywhere, and so is everything in vaft/process.
FILTER_DESIGNERS = frozenset(
    f"scipy.signal.{name}"
    for name in (
        "firwin", "firwin2", "firls", "remez", "kaiserord", "butter", "cheby1", "cheby2", "ellip",
        "bessel", "iirfilter", "iirdesign", "iirnotch", "iirpeak", "savgol_coeffs",
    )
)

#: (path, qualname, designer) -> why it is still allowed.
FILTER_DESIGN_ALLOWLIST: dict[tuple[str, str, str], str] = {
    ("vaft/machine_mapping/tf.py", "vfit_tf_current_detailed", "firwin"):
        "TF current low pass: the donor's Hamming firwin(taps, cutoff, fs) with the vest.yaml tf.processing "
        "values, applied through apply_fir_filter (#2135). Debt: route through fir_filter_coefficients("
        "window='hamming') once a coefficient-equality test shows it unchanged (#2206)",
    ("vaft/machine_mapping/magnetics.py", "vfit_plasma_current", "firwin"):
        "Ip reference comparison low pass: the donor's Hamming firwin with the vest.yaml comparison values, "
        "applied through apply_fir_filter (#2135). Same debt (#2206)",
}


def filter_design_findings(base: Path, roots: Iterable[str] = SCAN_ROOTS) -> list[Finding]:
    findings: list[Finding] = []
    for path in _python_sources(base, roots):
        rel = _relative(path, base)
        if rel.startswith("vaft/process/"):
            continue
        text = path.read_text(encoding="utf-8")
        if "signal" not in text:
            continue
        tree = ast.parse(text, filename=str(path))
        bindings = _import_bindings(tree)
        for qualname, node in _calls_with_scope(tree):
            called = _qualified(node.func, bindings)
            if called in FILTER_DESIGNERS:
                short = called.rsplit(".", 1)[1]
                findings.append(Finding((rel, qualname, short), f"{rel}:{node.lineno}", f"in {qualname}: {called}()"))
    return findings


def test_filters_are_designed_in_process():
    check_against_allowlist(
        filter_design_findings(REPO_ROOT),
        FILTER_DESIGN_ALLOWLIST,
        rule=FILTER_DESIGN_RULE,
        expected=FILTER_DESIGN_EXPECTED,
    )


def test_filter_design_scanner_reports_synthetic_designs(tmp_path):
    files = {
        "vaft/machine_mapping/new.py": (
            "import scipy.signal\n"
            "from scipy import signal\n"
            "from scipy.signal import butter as design, filtfilt\n"
            "import numpy as np\n"
            "def f(x, fs):\n"
            "    a = signal.firwin(251, 2500.0, fs=fs)\n"
            "    b, c = design(4, 0.1)\n"
            "    d = scipy.signal.cheby1(4, 1, 0.2)\n"
            "    return filtfilt(b, c, x), np.interp(x, x, x), signal.lfilter(a, 1.0, x)\n"
        ),
        "workflow/stage.py": "from scipy.signal import iirnotch\ndef g():\n    return iirnotch(50.0, 30.0, fs=1000.0)\n",
        # vaft/process owns filter design.
        "vaft/process/signal_processing.py": "from scipy import signal\ndef h():\n    return signal.firwin(11, 0.1)\n",
    }
    for relative, text in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    findings = filter_design_findings(tmp_path)
    assert sorted((f.key, f.location) for f in findings) == [
        (("vaft/machine_mapping/new.py", "f", "butter"), "vaft/machine_mapping/new.py:7"),
        (("vaft/machine_mapping/new.py", "f", "cheby1"), "vaft/machine_mapping/new.py:8"),
        (("vaft/machine_mapping/new.py", "f", "firwin"), "vaft/machine_mapping/new.py:6"),
        (("workflow/stage.py", "g", "iirnotch"), "workflow/stage.py:3"),
    ]
    with pytest.raises(AssertionError, match=r"new\.py:6: in f: scipy\.signal\.firwin\(\)(.|\n)*fir_filter_coefficients"):
        check_against_allowlist(findings, {}, rule=FILTER_DESIGN_RULE, expected=FILTER_DESIGN_EXPECTED)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-q", "-n", "0"]))
