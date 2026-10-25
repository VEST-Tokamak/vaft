"""The canonical API ownership matrix is true of the code (#2162 deliverable 12A).

``docs/architecture/consolidation/api_ownership.json`` is curated by hand; this
test is what keeps it honest.  For every row it checks that:

* the canonical API resolves (unless the row is still only ``proposed``, in
  which case it must *not* exist yet -- a merged proposal has to update its row);
* every previous implementation named ``removed`` is gone, and every one
  named ``delegating_alias`` / ``retained_specialization`` / ``current`` still
  exists;
* ``affected_callers`` is exactly the set of functions in ``vaft/`` and
  ``workflow/`` whose calls *resolve* (through their module's imports, not by
  bare name) to one of the row's ``resolves`` targets;
* the regression tests it names exist;
* for FIR application, each caller's ``short_record`` is the one recorded
  (``"raise"`` when the keyword is absent), so the TF and Ip-reference
  refusal of short records is pinned where it is documented.
"""

from __future__ import annotations

import ast
import importlib
import json
from functools import lru_cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "docs/architecture/consolidation/api_ownership.json"
SCAN_ROOTS = ("vaft", "workflow")
SKIP = ("workflow/architecture_audit/",)


def _matrix():
    return json.loads(MATRIX.read_text(encoding="utf-8"))


ROWS = _matrix()["rows"]


def _resolve(qualified: str):
    parts = qualified.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:cut]))
        except ImportError:
            continue
        try:
            for attr in parts[cut:]:
                obj = getattr(obj, attr)
        except AttributeError:
            return None
        return obj
    return None


def _module_name(path: Path) -> str:
    rel = path.relative_to(ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _absolute(module: str, level: int, name: str | None, is_package: bool) -> str:
    if level == 0:
        return name or ""
    base = module.split(".")
    base = base if is_package else base[:-1]
    base = base[: len(base) - (level - 1)] if level > 1 else base
    return ".".join(base + ([name] if name else []))


@lru_cache(maxsize=None)
def _call_sites():
    """``{resolved callee: {caller qualname: [call nodes]}}`` over the scan roots."""
    sites: dict[str, dict[str, list[ast.Call]]] = {}
    for root in SCAN_ROOTS:
        for path in sorted((ROOT / root).rglob("*.py")):
            rel = path.relative_to(ROOT).as_posix()
            if rel.startswith(SKIP):
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            module = _module_name(path)
            is_package = path.name == "__init__.py"
            bindings: dict[str, str] = {}
            top_defs = {
                node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            }
            for name in top_defs:
                bindings[name] = f"{module}.{name}"
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    source = _absolute(module, node.level, node.module, is_package)
                    for alias in node.names:
                        bindings[alias.asname or alias.name] = f"{source}.{alias.name}"
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        bindings[alias.asname or alias.name.split(".")[0]] = (
                            alias.name if alias.asname else alias.name.split(".")[0]
                        )

            def visit(node, stack):
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        visit(child, stack + [child.name])
                        continue
                    if isinstance(child, ast.Call):
                        target = _target(child.func)
                        if target:
                            caller = ".".join([module] + stack) if stack else f"{module}.<module>"
                            sites.setdefault(target, {}).setdefault(caller, []).append(child)
                    visit(child, stack)

            def _target(func):
                if isinstance(func, ast.Name):
                    return bindings.get(func.id)
                chain = []
                while isinstance(func, ast.Attribute):
                    chain.append(func.attr)
                    func = func.value
                if isinstance(func, ast.Name) and func.id in bindings:
                    return ".".join([bindings[func.id]] + chain[::-1])
                return None

            visit(tree, [])
    return sites


def _callers_of(targets) -> set[str]:
    sites = _call_sites()
    return {caller for target in targets for caller in sites.get(target, {})}


@pytest.mark.parametrize("row", ROWS, ids=[row["operation"] for row in ROWS])
def test_the_canonical_api_exists_unless_only_proposed(row):
    exists = _resolve(row["canonical_api"]) is not None
    if row["migration_status"] == "proposed":
        assert not exists, (
            f"{row['canonical_api']} now exists: update this row from 'proposed' "
            "(status, previous implementations, callers, tests)"
        )
    else:
        assert exists, row["canonical_api"]
        assert row["canonical_api"].startswith(row["canonical_module"] + ".")


@pytest.mark.parametrize("row", ROWS, ids=[row["operation"] for row in ROWS])
def test_previous_implementations_are_where_the_matrix_says(row):
    statuses = set(_matrix()["previous_status_values"])
    for previous in row["previous_implementations"]:
        assert previous["status"] in statuses
        name = previous["name"]
        if " " in name:  # a description of call sites, not a dotted name
            continue
        present = _resolve(name) is not None
        if previous["status"] == "removed":
            assert not present, f"{name} is listed as removed but still exists"
        else:
            assert present, f"{name} is listed as {previous['status']} but does not exist"


@pytest.mark.parametrize("row", ROWS, ids=[row["operation"] for row in ROWS])
def test_affected_callers_are_exactly_the_resolved_call_sites(row):
    actual = _callers_of(row["resolves"])
    recorded = set(row["affected_callers"])
    assert recorded == actual, {
        "missing from the matrix": sorted(actual - recorded),
        "no longer calling": sorted(recorded - actual),
    }


@pytest.mark.parametrize("row", ROWS, ids=[row["operation"] for row in ROWS])
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
    if row["migration_status"] in ("migrated", "retained"):
        assert row["regression_tests"], "a migrated or retained row names its regression tests"


def _fir_row():
    return next(row for row in ROWS if row["canonical_api"].endswith(".apply_fir_filter"))


def test_short_record_policy_per_caller_is_the_documented_one():
    row = _fir_row()
    sites = _call_sites()["vaft.process.signal_processing.apply_fir_filter"]
    assert set(row["short_record_by_caller"]) == set(sites)
    for caller, calls in sites.items():
        for call in calls:
            value = next((k.value for k in call.keywords if k.arg == "short_record"), None)
            policy = "raise" if value is None else ast.literal_eval(value)
            assert policy == row["short_record_by_caller"][caller], caller


def test_tf_and_ip_reference_refuse_short_records():
    """The behaviour the matrix documents for TF and the Ip reference, run."""
    import numpy as np

    from vaft.process.signal_processing import apply_fir_filter

    taps = np.ones(26) / 26
    with pytest.raises(ValueError):
        apply_fir_filter(taps, np.ones(78))  # 3 * 26 + 1 = 79 is the minimum
    assert apply_fir_filter(taps, np.ones(79)).shape == (79,)
    assert apply_fir_filter(taps, np.ones(78), short_record="causal").shape == (78,)


def test_direct_firwin_designs_are_the_listed_ones():
    row = next(row for row in ROWS if row["canonical_api"].endswith(".fir_filter_coefficients"))
    assert _callers_of(["scipy.signal.firwin"]) == set(row["direct_firwin_callers"])


# --- negative controls: the checks can fail --------------------------------


def test_callers_are_resolved_through_imports_not_by_bare_name():
    # vest_Halpha_tstart_tend calls code.efit.legacy.smooth, not the canonical one.
    canonical = _callers_of(["vaft.process.signal_processing.smooth"])
    legacy = _callers_of(["vaft.code.efit.legacy.smooth"])
    assert "vaft.code.efit.legacy.vest_Halpha_tstart_tend" in legacy
    assert "vaft.code.efit.legacy.vest_Halpha_tstart_tend" not in canonical


def test_a_dropped_or_invented_caller_is_caught():
    row = dict(ROWS[0])
    row["affected_callers"] = row["affected_callers"][1:] + ["vaft.nowhere.invented"]
    with pytest.raises(AssertionError):
        test_affected_callers_are_exactly_the_resolved_call_sites(row)


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
