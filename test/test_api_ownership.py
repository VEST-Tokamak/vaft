"""The canonical API ownership matrix is true of the code (#2162 deliverable 12A).

``docs/architecture/consolidation/api_ownership.json`` is curated by hand; this
test is what keeps it honest.  For every row it checks that:

* the canonical API resolves (unless the row is still only ``proposed``, in
  which case it must *not* exist yet -- a merged proposal has to update its
  row), and that it is one of the row's ``resolves`` targets;
* every previous implementation named ``removed`` is gone and every other one
  still exists; a module that does not import is an error, not "gone";
  a descriptive entry (call sites rather than a name) carries a ``check``
  that no function in a module still references the replaced primitive;
* ``affected_callers`` is exactly the set of functions in ``vaft/`` and
  ``workflow/`` that *reference* one of the ``resolves`` objects -- resolved
  through the imports in scope (module level, then function level) and
  compared by object identity, so a re-export or a ``functools.partial``
  counts and a same-named function elsewhere does not;
* ``retained_direct_callers`` (primitives used directly, on purpose) equals
  the resolved references too, so a new direct use is a visible change;
* the regression tests it names exist;
* for FIR application, each caller's ``short_record`` is the recorded one
  (``"raise"`` when absent), and the TF and Ip-reference minimum record
  lengths follow from vest.yaml's tap counts.
"""

from __future__ import annotations

import ast
import importlib
import json
import textwrap
from functools import lru_cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "docs/architecture/consolidation/api_ownership.json"
SCAN_ROOTS = ("vaft", "workflow")
SKIP = ("workflow/architecture_audit/",)
_MISSING = object()


def _matrix():
    return json.loads(MATRIX.read_text(encoding="utf-8"))


ROWS = _matrix()["rows"]


# ---------------------------------------------------------------------------
# Name resolution
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _resolve(qualified: str):
    """The object ``qualified`` names, or ``_MISSING`` when its module imports
    but the final attribute is absent.  A module or container that is not
    there raises: a typo or a broken import must not read as "removed"."""
    parts = qualified.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:cut])
        try:
            obj = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name:
                continue  # this prefix is not a module: try a shorter one
            raise
        rest = parts[cut:]
        for index, attr in enumerate(rest):
            obj = getattr(obj, attr, _MISSING)
            if obj is _MISSING:
                if index < len(rest) - 1:
                    raise AssertionError(f"{qualified}: {attr!r} is not in {'.'.join(parts[:cut + index])}")
                return _MISSING
        return obj
    raise AssertionError(f"{qualified}: no importable module prefix")


def _exists(qualified: str) -> bool:
    return _resolve(qualified) is not _MISSING


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _absolute(module: str, level: int, name: str | None, is_package: bool) -> str:
    if level == 0:
        return name or ""
    base = module.split(".") if is_package else module.split(".")[:-1]
    if level > 1:
        base = base[: len(base) - (level - 1)]
    return ".".join(base + ([name] if name else []))


def _import_bindings(statements, module: str, is_package: bool) -> dict[str, str]:
    """Bindings made by the import statements directly in ``statements``
    (descending into if/try/with blocks, not into nested defs)."""
    bindings: dict[str, str] = {}

    def visit(nodes):
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            if isinstance(node, ast.ImportFrom):
                source = _absolute(module, node.level, node.module, is_package)
                for alias in node.names:
                    bindings[alias.asname or alias.name] = f"{source}.{alias.name}"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        bindings[alias.asname] = alias.name
                    else:
                        head = alias.name.split(".")[0]
                        bindings[head] = head
            for field in ("body", "orelse", "finalbody", "handlers"):
                visit(getattr(node, field, []) or [])

    visit(statements)
    return bindings


def scan_source(source: str, module: str, *, is_package: bool = False):
    """``[(caller, dotted target, node, is_call)]`` for every Name/Attribute
    reference that an import or a top-level def in scope binds."""
    tree = ast.parse(source)
    found = []
    module_scope = _import_bindings(tree.body, module, is_package)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            module_scope.setdefault(node.name, f"{module}.{node.name}")

    def dotted(expr, scope):
        chain = []
        while isinstance(expr, ast.Attribute):
            chain.append(expr.attr)
            expr = expr.value
        if isinstance(expr, ast.Name) and expr.id in scope:
            return ".".join([scope[expr.id]] + chain[::-1])
        return None

    def walk(node, stack, scope, call_funcs):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                inner = dict(scope)
                if not isinstance(child, ast.ClassDef):
                    inner.update(_import_bindings(child.body, module, is_package))
                walk(child, stack + [child.name], inner, call_funcs)
                continue
            if isinstance(child, (ast.Name, ast.Attribute)) and isinstance(child.ctx, ast.Load):
                target = dotted(child, scope)
                if target is not None:
                    caller = ".".join([module] + stack) if stack else f"{module}.<module>"
                    found.append((caller, target, child, id(child) in call_funcs))
                    continue  # the longest chain only
            if isinstance(child, ast.Call):
                call_funcs.add(id(child.func))
            walk(child, stack, scope, call_funcs)

    walk(tree, [], module_scope, set())
    return found


@lru_cache(maxsize=None)
def _references():
    refs = []
    for root in SCAN_ROOTS:
        for path in sorted((ROOT / root).rglob("*.py")):
            rel = path.relative_to(ROOT).as_posix()
            if rel.startswith(SKIP):
                continue
            source = path.read_text(encoding="utf-8")
            refs.extend(scan_source(source, _module_name(path), is_package=path.name == "__init__.py"))
    return refs


@lru_cache(maxsize=None)
def _object_of(dotted: str):
    try:
        return _resolve(dotted)
    except (AssertionError, ImportError, Exception):  # a dynamic or local name: not an API reference
        return _MISSING


def _referencing(targets):
    wanted = [_resolve(t) for t in targets]
    wanted = [w for w in wanted if w is not _MISSING]
    callers = set()
    for caller, target, _node, _is_call in _references():
        obj = _object_of(target)
        if obj is not _MISSING and any(obj is w for w in wanted):
            callers.add(caller)
    return callers


def _calls_to(target):
    wanted = _resolve(target)
    out = {}
    for caller, dotted, node, is_call in _references():
        if _object_of(dotted) is wanted:
            out.setdefault(caller, []).append((node, is_call))
    return out


# ---------------------------------------------------------------------------
# The matrix
# ---------------------------------------------------------------------------

_IDS = [row["operation"] for row in ROWS]


@pytest.mark.parametrize("row", ROWS, ids=_IDS)
def test_the_canonical_api_exists_unless_only_proposed(row):
    exists = _exists(row["canonical_api"])
    if row["migration_status"] == "proposed":
        assert not exists, (
            f"{row['canonical_api']} now exists: update this row from 'proposed' "
            "(status, previous implementations, callers, tests)"
        )
        for previous in row["previous_implementations"]:
            if previous["status"] == "current" and _exists(previous["name"]):
                # the current implementations must not already delegate somewhere new
                body = _source_of(previous["name"])
                assert row["canonical_api"].rsplit(".", 1)[-1] not in body
    else:
        assert exists, row["canonical_api"]
        assert row["canonical_api"].startswith(row["canonical_module"] + ".")
        assert row["canonical_api"] in row["resolves"]


def _source_of(qualified):
    import inspect

    return inspect.getsource(_resolve(qualified))


@pytest.mark.parametrize("row", ROWS, ids=_IDS)
def test_previous_implementations_are_where_the_matrix_says(row):
    statuses = set(_matrix()["previous_status_values"])
    for previous in row["previous_implementations"]:
        assert previous["status"] in statuses
        check = previous.get("check")
        if check is not None:
            assert previous["status"] == "removed"
            left = {c for c in _referencing([check["primitive"]]) if c.startswith(check["no_reference_in"] + ".")}
            assert not left, f"{check['primitive']} is still used directly in {sorted(left)}"
            continue
        present = _exists(previous["name"])
        if previous["status"] == "removed":
            assert not present, f"{previous['name']} is listed as removed but still exists"
        else:
            assert present, f"{previous['name']} is listed as {previous['status']} but does not exist"


@pytest.mark.parametrize("row", ROWS, ids=_IDS)
def test_affected_callers_are_exactly_the_resolved_references(row):
    actual = _referencing(row["resolves"]) - set(row["resolves"])
    recorded = set(row["affected_callers"])
    assert recorded == actual, {
        "missing from the matrix": sorted(actual - recorded),
        "no longer referencing": sorted(recorded - actual),
    }


@pytest.mark.parametrize("row", ROWS, ids=_IDS)
def test_retained_direct_uses_are_exactly_the_listed_ones(row):
    for primitive, recorded in row.get("retained_direct_callers", {}).items():
        actual = _referencing([primitive])
        assert set(recorded) == actual, {
            "primitive": primitive,
            "missing from the matrix": sorted(actual - set(recorded)),
            "no longer using it": sorted(set(recorded) - actual),
        }


@pytest.mark.parametrize("row", ROWS, ids=_IDS)
def test_rows_are_well_formed(row):
    doc = _matrix()
    assert row["migration_status"] in doc["migration_status_values"]
    for key in (
        "operation", "canonical_api", "canonical_module", "previous_implementations",
        "affected_callers", "numerical_equivalence", "preserved_specializations",
        "configuration_dependencies", "regression_tests", "migration_status",
    ):
        assert key in row, key
    for test in row["regression_tests"]:
        assert (ROOT / test).is_file(), test
    if row["migration_status"] != "proposed":
        assert row["regression_tests"], "every row that describes merged code names its tests"


# ---------------------------------------------------------------------------
# FIR application: the short-record policy
# ---------------------------------------------------------------------------


def _fir_row():
    return next(row for row in ROWS if row["canonical_api"].endswith(".apply_fir_filter"))


def test_short_record_policy_per_caller_is_the_documented_one():
    row = _fir_row()
    sites = _calls_to("vaft.process.signal_processing.apply_fir_filter")
    assert set(row["short_record_by_caller"]) == set(sites) - {"vaft.process.signal_processing.apply_fir_filter"}
    for caller, uses in sites.items():
        if caller == "vaft.process.signal_processing.apply_fir_filter":
            continue
        for node, is_call in uses:
            assert is_call, f"{caller} passes apply_fir_filter on without calling it: record its policy by hand"
            call = _parent_call(caller, node)
            assert not any(k.arg is None for k in call.keywords), f"{caller}: **kwargs hide short_record"
            value = next((k.value for k in call.keywords if k.arg == "short_record"), None)
            policy = "raise" if value is None else ast.literal_eval(value)
            assert policy == row["short_record_by_caller"][caller], caller


def _parent_call(caller, func_node):
    module = caller
    while module and not _module_file(module):
        module = module.rsplit(".", 1)[0]
    tree = ast.parse(_module_file(module).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and node.func.lineno == func_node.lineno and node.func.col_offset == func_node.col_offset:
            return node
    raise AssertionError(f"call for {caller} not found")


def _module_file(module):
    base = ROOT.joinpath(*module.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


@pytest.mark.parametrize(
    "diagnostic, path, caller",
    [
        ("tf", ("processing", "taps"), "vaft.machine_mapping.tf.vfit_tf_current_detailed"),
        ("plasma_current", ("processing", "reference_comparison", "taps"), "vaft.machine_mapping.magnetics.vfit_plasma_current"),
    ],
)
def test_tf_and_ip_reference_refuse_records_below_their_filtfilt_minimum(diagnostic, path, caller):
    """The documented refusal, with the tap count read from vest.yaml."""
    import numpy as np

    from vaft.machine_mapping.utils import resolve_vest_diagnostic
    from vaft.process.signal_processing import apply_fir_filter

    assert _fir_row()["short_record_by_caller"][caller] == "raise"
    config = resolve_vest_diagnostic(41672, diagnostic)
    for key in path:
        config = config[key]
    n = int(config)
    minimum = 3 * n + 1
    assert f"{minimum} samples" in " ".join(_fir_row()["preserved_specializations"]) or diagnostic != "tf"
    taps = np.ones(n) / n
    with pytest.raises(ValueError):
        apply_fir_filter(taps, np.ones(minimum - 1))
    assert apply_fir_filter(taps, np.ones(minimum)).shape == (minimum,)
    assert apply_fir_filter(taps, np.ones(minimum - 1), short_record="causal").shape == (minimum - 1,)


# ---------------------------------------------------------------------------
# The resolver itself, on synthetic sources, and negative controls
# ---------------------------------------------------------------------------


def _targets(source, module="pkg.sub.mod", is_package=False):
    return {(c, t, is_call) for c, t, _n, is_call in scan_source(textwrap.dedent(source), module, is_package=is_package)}


SOURCE_TWO_FIRWIN = """
import scipy.signal as ss
from scipy import signal

def f():
    ss.firwin(3, 0.1)
    signal.firwin(3, 0.1)
"""


def test_resolver_relative_aliased_attribute_and_lazy_imports():
    found = _targets(
        """
        import scipy.signal as ss
        from scipy import signal
        from ..other import helper
        from . import sibling

        def f():
            ss.firwin(3, 0.1)
            signal.firwin(3, 0.1)
            helper()
            sibling.thing()

        def g():
            from vaft.process.signal_processing import smooth
            return smooth

        def h():
            smooth(1)  # not bound here: g's lazy import does not leak
        """
    )
    assert ("pkg.sub.mod.f", "scipy.signal.firwin", True) in found
    assert ("pkg.sub.mod.f", "pkg.other.helper", True) in found
    assert ("pkg.sub.mod.f", "pkg.sub.sibling.thing", True) in found
    assert ("pkg.sub.mod.g", "vaft.process.signal_processing.smooth", False) in found
    assert not any(c == "pkg.sub.mod.h" for c, _t, _k in found)
    raw = scan_source(textwrap.dedent(SOURCE_TWO_FIRWIN), "pkg.sub.mod")
    assert sum(1 for c, t, _n, _k in raw if t == "scipy.signal.firwin") == 2


def test_resolver_sees_a_non_call_reference():
    found = _targets(
        """
        from functools import partial
        from vaft.process.signal_processing import apply_fir_filter

        def f():
            return partial(apply_fir_filter, short_record="causal")
        """
    )
    assert ("pkg.sub.mod.f", "vaft.process.signal_processing.apply_fir_filter", False) in found


def test_a_reexport_resolves_to_the_same_object():
    assert _resolve("vaft.process.smooth") is _resolve("vaft.process.signal_processing.smooth")


def test_callers_are_resolved_through_imports_not_by_bare_name():
    canonical = _referencing(["vaft.process.signal_processing.smooth"])
    legacy = _referencing(["vaft.code.efit.legacy.smooth"])
    assert "vaft.code.efit.legacy.vest_Halpha_tstart_tend" in legacy
    assert "vaft.code.efit.legacy.vest_Halpha_tstart_tend" not in canonical


def test_a_typo_in_a_module_is_an_error_not_removed():
    with pytest.raises(AssertionError):
        _resolve("vaft.process.magnetixs._firwin_order")
    assert _resolve("vaft.process.magnetics._firwin_order") is _MISSING


def test_a_dropped_or_invented_caller_is_caught():
    row = dict(ROWS[0])
    row["affected_callers"] = row["affected_callers"][1:] + ["vaft.nowhere.invented"]
    with pytest.raises(AssertionError):
        test_affected_callers_are_exactly_the_resolved_references(row)


def test_a_removed_name_that_still_exists_is_caught():
    row = {"previous_implementations": [
        {"name": "vaft.process.signal_processing.smooth", "status": "removed", "retirement": "-"}
    ]}
    with pytest.raises(AssertionError, match="listed as removed"):
        test_previous_implementations_are_where_the_matrix_says(row)


def test_a_wrong_short_record_policy_is_caught(monkeypatch):
    row = _fir_row()
    flipped = dict(row["short_record_by_caller"])
    flipped["vaft.machine_mapping.tf.vfit_tf_current_detailed"] = "causal"
    monkeypatch.setitem(row, "short_record_by_caller", flipped)
    with pytest.raises(AssertionError, match="vfit_tf_current_detailed"):
        test_short_record_policy_per_caller_is_the_documented_one()


def test_a_new_direct_primitive_use_is_caught():
    row = {"retained_direct_callers": {"scipy.signal.firwin": ["vaft.process.impa.impa_lowpass"]}}
    with pytest.raises(AssertionError):
        test_retained_direct_uses_are_exactly_the_listed_ones(row)
