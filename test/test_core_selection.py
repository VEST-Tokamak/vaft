"""The develop gate's contract with itself.

`core` is what gates every pull request into `develop` (#515), and it is
applied indirectly: test/core_selection.py declares a list, test/conftest.py
turns that list into markers. Two things can quietly go wrong with an
arrangement like that. A module can be renamed or deleted, leaving a dead entry
that silently removes coverage from the gate. And the marker can stop being
applied in time to matter -- `-m` filtering is itself a
``pytest_collection_modifyitems`` implementation, so if the hook ordering ever
changed, ``-m core`` would select nothing at all and the gate would go green
without running anything.

Neither failure announces itself. These tests do.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

from core_selection import CORE_MODULES, TEST_ROOT, core_paths

REPO_ROOT = TEST_ROOT.parent


def test_every_declared_module_exists():
    """A renamed or deleted module must not silently leave the gate."""
    missing = [str(path.relative_to(REPO_ROOT)) for path in core_paths() if not path.is_file()]
    assert not missing, (
        "test/core_selection.py names modules that do not exist: "
        f"{missing}. Update the list rather than leaving the gate pointing at nothing."
    )


def test_no_duplicate_entries():
    duplicates = sorted({name for name in CORE_MODULES if CORE_MODULES.count(name) > 1})
    assert not duplicates, f"declared twice in CORE_MODULES: {duplicates}"


def test_every_entry_is_a_test_module():
    wrong = [name for name in CORE_MODULES if not Path(name).name.startswith("test_")]
    assert not wrong, f"CORE_MODULES may only name test modules: {wrong}"


def _declared_groups() -> list[list[str]]:
    """The CORE_MODULES entries, split into the comment-headed groups of the source.

    Read from the source text because the grouping *is* comments: the point of
    the list is that a reader can see what the gate covers, and that only works
    if each group says what it protects and stays in order.
    """
    source = (TEST_ROOT / "core_selection.py").read_text().splitlines()
    start = next(i for i, line in enumerate(source) if line.startswith("CORE_MODULES"))
    groups: list[list[str]] = []
    for line in source[start + 1 :]:
        stripped = line.strip()
        if stripped == ")":
            break
        if stripped.startswith("#"):
            if groups and groups[-1]:
                groups.append([])
            continue
        entry = re.fullmatch(r'"([^"]+)",', stripped)
        if entry:
            if not groups:
                groups.append([])
            groups[-1].append(entry.group(1))
    return [group for group in groups if group]


def test_each_group_is_commented_and_sorted():
    groups = _declared_groups()
    assert groups, "no groups parsed out of CORE_MODULES"
    assert sum(len(group) for group in groups) == len(CORE_MODULES)
    unsorted = [group for group in groups if group != sorted(group)]
    assert not unsorted, (
        "each commented group in CORE_MODULES is kept sorted, so a diff to the "
        f"develop gate reads as one line: {unsorted}"
    )


def _markers_applied_in(path: Path) -> set[str]:
    """Marker names a module actually applies, read from its syntax tree.

    Parsed rather than grepped: a test module that talks *about* a marker in a
    docstring or an assertion message -- this one does -- is not applying it,
    and a substring scan cannot tell the difference.
    """
    applied = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "mark"
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "pytest"
        ):
            applied.add(node.attr)
    return applied


def test_core_excludes_slow_modules():
    """`slow` is release confidence and belongs to the main gate.

    `slow` is applied module-wide -- notebook and tutorial execution -- so
    there is nothing to salvage from such a module at this altitude: it is kept
    out of the list entirely rather than deselected in the gate expression.
    `perf` is different; see the gate-expression test below.
    """
    offenders = [
        name
        for name, path in zip(CORE_MODULES, core_paths())
        if path.is_file() and "slow" in _markers_applied_in(path)
    ]
    assert not offenders, (
        f"{offenders} carry pytest.mark.slow; that is release-confidence "
        "coverage and runs on the main gate, not on every develop PR."
    )


WORKFLOW = REPO_ROOT / ".github" / "workflows" / "package-ci.yml"

#: What CI must run for the develop gate. `perf` tests assert wall-clock
#: ratios, and a job whose entire purpose is to finish in minutes is the last
#: place such a budget should be believed -- the Windows leg drops them for the
#: same reason. They are deselected here rather than excluding their modules,
#: because a module like test_formula_catalog.py is twenty API-contract tests
#: and one timing budget, and the twenty are exactly what develop wants.
GATE_EXPRESSION = 'core and not perf'


@pytest.mark.skipif(not WORKFLOW.is_file(), reason="this branch has no Package CI workflow")
def test_the_develop_gate_runs_the_declared_expression():
    """CI must select what this module says it selects."""
    workflow = WORKFLOW.read_text()
    assert f'-m "{GATE_EXPRESSION}"' in workflow, (
        "the core-test job in .github/workflows/package-ci.yml no longer runs "
        f'-m "{GATE_EXPRESSION}". The develop gate and this contract have to '
        "move together, or one of them is a lie."
    )


def test_core_marker_is_registered():
    """`pytest -m core` must be a documented selector, not an unknown mark."""
    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    assert '"core: ' in pyproject, "register the `core` marker in pyproject.toml"


def test_dash_m_core_selects_the_declared_modules_and_nothing_else():
    """The gate is only real if the marker is applied before `-m` filters.

    Collected over one core module and one non-core module rather than the
    whole suite: importing 240 test modules a second time would cost the gate
    more than this check is worth, and the ordering question the test exists to
    answer is answered just as well by two files.
    """
    inside = "test_import.py"
    outside = "test_aeqdsk.py"
    assert inside in CORE_MODULES, f"{inside} is the core control; pick another"
    assert outside not in CORE_MODULES, f"{outside} is the non-core control; pick another"
    assert (TEST_ROOT / outside).is_file()

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-m",
            GATE_EXPRESSION,
            f"test/{inside}",
            f"test/{outside}",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    collected = [
        line for line in completed.stdout.splitlines() if "::" in line and line.startswith("test/")
    ]
    assert collected, (
        "`-m core` selected nothing from a core module. The marker is being "
        "applied after `-m` filtering, so the develop gate would run an empty "
        f"suite:\n{completed.stdout}"
    )
    strays = sorted({line.split("::")[0] for line in collected} - {f"test/{inside}"})
    assert not strays, f"`-m core` selected modules that are not declared core: {strays}"
