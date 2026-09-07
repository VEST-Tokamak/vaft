"""Nothing may write a nested `code.parameters` path without saying why (#561).

`test_code_parameters_contract.py` measures what the field supports.  This
module holds the source to it: a *flat* leaf under `code.parameters` survives
replication, because `vaft.omas.load` promotes a leaf-only block back to an
`omas.CodeParameters` on its way out of the stage product (#478), and a
*nested* one does not -- the promotion declines it, the Access Layer discards
the plain branch that is left, and nothing raises.  Worse, the promotion is
all-or-nothing on the block, so a nested sub-path takes every flat leaf beside
it down as well.

So a nested path is not forbidden -- EFIT's per-slice parser cache is genuinely
useful and genuinely local -- but it must be a decision somebody wrote down,
not something that arrives in a diff.  Every file that mentions one is listed
in ``ALLOWED`` with the role it plays and the reason it is acceptable for that
data to stop at the FileDB.  A new one fails this test until it is listed or
serialized into the single string.

The scan reads string literals through the AST rather than matching the file's
text, so the many prose mentions of these paths -- in module docstrings, in the
comments that explain the rule -- are not findings.  ``test/`` is scanned too:
the shape this issue was opened about (``mhd_linear.code.parameters.cases.N``)
was invented by a test, not by a producer.

What it cannot see: a path assembled from fragments that never appear together
in one literal, such as a sub-path handed to a helper that prefixes the field.
The guard is a net for the ordinary way these paths are written -- a literal
naming the field -- not a proof that none exists.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCANNED = ("vaft", "workflow", "test")

#: A sub-path of `code.parameters`, in any IDS and at any depth -- the leading
#: `(?:^|\.)` is what also catches `core_sources.source.0.code.parameters.x`
#: and the `f"{ids}.code.parameters..."` the mappers build.
SUBPATH = re.compile(r"(?:^|\.)code\.parameters\.(?P<rest>\S+)")

#: Files that mention a nested path, each with why that data may stop at the
#: local product.  ``role`` is what the file does with it; the guard scans
#: mentions, since an f-string assigned to a variable does not say on its own
#: whether it will be read or written.
ALLOWED: dict[str, tuple[tuple[str, str, str], ...]] = {
    # -- EFIT's per-slice parser cache: the a/k/m-file namelists as they were
    # read, kept so a reconstruction can be re-run and audited locally.  It is
    # not representable as one XML parameters string (numeric-looking keys,
    # per-slice lists), which is exactly why #478's promotion declines it.
    "vaft/data/aeqdsk.py": (
        ("equilibrium.code.parameters.time_slice.", "write", "a-file parser cache (#380)"),
    ),
    "vaft/data/keqdsk.py": (
        ("equilibrium.code.parameters.time_slice.", "write", "k-file namelist cache (#380)"),
    ),
    "vaft/data/meqdsk.py": (
        ("equilibrium.code.parameters.time_slice.", "write", "m-file variable cache (#380)"),
    ),
    "vaft/code/efit/magnetic.py": (
        ("equilibrium.code.parameters.time_slice.", "write",
         "constraints input, mapping diagnostics and artifact hashes per slice; the "
         "replicated copy of this provenance is the JSON string generate_efit_ods.py "
         "writes (#380)"),
    ),
    "vaft/database/_summary.py": (
        ("equilibrium.code.parameters.time_slice.", "read",
         "shot overview reads the cache from the local product it summarizes"),
    ),
    "vaft/omas/efit_quality.py": (
        ("equilibrium.code.parameters.time_slice.", "read", "fit quality reads the cache"),
    ),
    "vaft/plot/backend/recipes.py": (
        ("equilibrium.code.parameters.time_slice.", "read", "convergence plot reads the cache"),
    ),
    # -- Tests that build or read that cache as a fixture.
    "test/test_aeqdsk.py": (
        ("equilibrium.code.parameters.time_slice.", "read", "a-file cache fixture"),
    ),
    "test/test_database_summary.py": (
        ("equilibrium.code.parameters.time_slice.", "write", "shot overview fixture"),
    ),
    "test/test_diamagnetic_flux_sign.py": (
        ("equilibrium.code.parameters.time_slice.", "write", "k-file namelist fixture"),
    ),
    "test/test_efit_config.py": (
        ("equilibrium.code.parameters.time_slice.", "write", "k-file namelist fixture"),
    ),
    "test/test_efit_fit_quality.py": (
        ("equilibrium.code.parameters.time_slice.", "write", "a/m-file cache fixture"),
    ),
    "test/test_efit_km_mapping.py": (
        ("equilibrium.code.parameters.time_slice.", "write", "k/m-file cache fixture"),
    ),
    "test/test_eqdsk_derived_quantities.py": (
        ("equilibrium.code.parameters.time_slice.", "read", "a-file cache fixture"),
    ),
    "test/test_ods_access.py": (
        ("equilibrium.code.parameters.time_slice.", "read",
         "the accessor's own tests for descending into the cache"),
    ),
    "test/test_code_parameters_entry_payload.py": (
        ("equilibrium.code.parameters.time_slice.", "write",
         "fixtures for the save-side split: the shape is written in order to show "
         "that it stays local and is named rather than carried (#642)"),
        ("core_profiles.code.parameters.fits.", "write",
         "second-IDS fixture for the same"),
    ),
    "test/test_equilibrium_psi_to_weber.py": (
        ("equilibrium.code.parameters.efit_collection.", "write",
         "pins that promotion declines a nested cache, and that the COCOS index is "
         "unreadable beside one -- the reading half of the loss #561 measures"),
    ),
}

_DOCSTRING_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _docstring_ids(tree: ast.AST) -> set[int]:
    """Every string node that is a docstring, so prose is not a finding."""
    found: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, _DOCSTRING_OWNERS):
            continue
        body = getattr(node, "body", [])
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            found.add(id(body[0].value))
    return found


def _literals(tree: ast.AST) -> list[tuple[int, str]]:
    """Every string literal, with an f-string rendered as its template.

    ``f"...time_slice.{index}.aeqdsk"`` becomes ``...time_slice.{}.aeqdsk``, so
    a path built from a loop variable is one literal rather than two fragments.
    """
    skip = _docstring_ids(tree)
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            rendered = "".join(
                part.value if isinstance(part, ast.Constant) and isinstance(part.value, str) else "{}"
                for part in node.values
            )
            out.append((node.lineno, rendered))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip:
            out.append((node.lineno, node.value))
    return out


def _nested_paths(path: Path) -> list[tuple[int, str]]:
    """The nested `code.parameters` sub-paths a file names.

    A single segment -- ``code.parameters.cocos`` -- is the flat shape the
    loader promotes, and is not reported.  Anything with a further separator
    under it is.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:  # pragma: no cover - the repo does not hold one
        return []
    findings: list[tuple[int, str]] = []
    for lineno, text in _literals(tree):
        match = SUBPATH.search(text)
        if match is None or "." not in match.group("rest").rstrip("."):
            continue
        findings.append((lineno, text))
    # An f-string yields both its template and its own constant fragments; the
    # template covers them.  Compare only within a line, so a literal that
    # merely happens to be a prefix of a template elsewhere in the file is
    # still reported -- a guard may not lose a finding to a coincidence.
    by_line: dict[int, set[str]] = {}
    for lineno, text in findings:
        if "{}" in text:
            by_line.setdefault(lineno, set()).add(text)
    return [
        (lineno, text)
        for lineno, text in findings
        if not any(
            text != template and template.startswith(text)
            for template in by_line.get(lineno, ())
        )
    ]


def _sources() -> list[Path]:
    return [
        path
        for directory in SCANNED
        for path in sorted((ROOT / directory).rglob("*.py"))
        if (ROOT / directory).exists()
    ]


SOURCES = _sources()
IDS = [str(path.relative_to(ROOT)) for path in SOURCES]


@pytest.mark.parametrize("path", SOURCES, ids=IDS)
def test_a_nested_parameter_path_is_declared_or_absent(path):
    relative = str(path.relative_to(ROOT).as_posix())
    findings = _nested_paths(path)
    if not findings:
        return
    declared = ALLOWED.get(relative, ())
    undeclared = [
        (lineno, text)
        for lineno, text in findings
        if not any(prefix in text for prefix, _, _ in declared)
    ]
    assert not undeclared, (
        f"{relative} names a nested sub-path of code.parameters:\n"
        + "\n".join(f"  line {lineno}: {text}" for lineno, text in undeclared)
        + "\n\n"
        "`code.parameters` is a STR_0D in the Data Dictionary. A nested sub-path\n"
        "survives the local product and is then discarded by the Access Layer with\n"
        "no exception, no returned path and no warning a pipeline reads -- and it\n"
        "takes every flat leaf beside it down with it, which is how an EFIT product\n"
        "loses its declared COCOS index (#380, #478, measured in\n"
        "test_code_parameters_contract.py).\n\n"
        "Serialize the payload into the single string instead:\n"
        '    ods["equilibrium.code.parameters"] = json.dumps(payload, sort_keys=True)\n'
        "or append an XML fragment inside the <parameters> envelope, as\n"
        "vaft.machine_mapping.mhd_linear does.\n\n"
        "If this data is genuinely local-product-only, add the file to ALLOWED in\n"
        "this module with its role and the reason it may stop at the FileDB.\n"
        "See docs/_guide/Data_structures.md, 'What survives on code.parameters'."
    )


def test_no_workflow_script_writes_a_nested_parameter_path():
    """The generalisation of #380's single-file check.

    That check read one stage script and forbade one literal. Every pipeline
    stage writes a product that is replicated, so the rule is the same for all
    of them: nothing under `workflow/` may put provenance in a shape that stops
    at the FileDB.
    """
    scanned = {
        str(path.relative_to(ROOT).as_posix()): _nested_paths(path)
        for path in SOURCES
        if path.is_relative_to(ROOT / "workflow")
    }
    offenders = {name: found for name, found in scanned.items() if found}

    assert offenders == {}


def test_the_allow_list_names_files_that_exist_and_still_match():
    """An entry that no longer applies is a comment pretending to be a guard."""
    stale = {}
    for relative, entries in ALLOWED.items():
        path = ROOT / relative
        if not path.exists():
            stale[relative] = "file is gone"
            continue
        findings = [text for _, text in _nested_paths(path)]
        for prefix, role, reason in entries:
            assert role in ("read", "write"), (relative, role)
            assert reason, relative
            if not any(prefix in text for text in findings):
                stale[relative] = f"no literal contains {prefix!r} any more"

    assert stale == {}
