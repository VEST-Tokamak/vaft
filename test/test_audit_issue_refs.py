""".github/scripts/audit_issue_refs.py: what counts as an issue/PR reference, and what does not."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / ".github" / "scripts" / "audit_issue_refs.py"


@pytest.fixture(scope="module")
def audit():
    spec = importlib.util.spec_from_file_location("audit_issue_refs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up there
    spec.loader.exec_module(module)
    return module


def refs(audit, line, path="vaft/example.py"):
    return [(r.number, r.kind, r.status) for r in audit.scan_line(path, 1, "", line)]


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("Collision processes: a classification diagram (#1215).", [(1215, "bare", "")]),
        ("see issue #307 and PR #1592", [(307, "issue", ""), (1592, "pr", "")]),
        ("Use the shared vocabulary (issues #330, #529):", [(330, "issue", ""), (529, "issue", "")]),
        ("submodules not yet converted (#418-#421)", [(418, "bare", ""), (421, "bare", "")]),
        ("GH-77 tracks it", [(77, "gh", "")]),
        ("https://github.com/VEST-Tokamak/vaft/pull/2140", [(2140, "url-pr", "")]),
        ("https://github.com/VEST-Tokamak/vaft/issues/66#issuecomment-1", [(66, "url-issue", "")]),
    ],
)
def test_references_are_found(audit, line, expected):
    assert refs(audit, line) == expected


@pytest.mark.parametrize(
    "line",
    [
        "x = '&#1215;'",                      # HTML entity
        "see https://example.org/page#123",   # URL fragment
        "fill='#000' stroke='#333'",          # colours
        "color: #123456;",
        "Flux Loop - #10 and loops #9-#11",   # VEST numbered sensors
        "shot #39915 at 0.32 s",              # shot number
        "\\raisebox{#2}{#1}",                 # TeX macro arguments
        "the reduced-wall contract (vfit #8)",
        "value = data[3]  #",
    ],
)
def test_non_references_are_excluded(audit, line):
    assert all(status == "excluded" for _, _, status in refs(audit, line))


def test_a_ticket_a_test_matches_on_is_not_a_colour(audit):
    """`match="#358"` is a reference the deprecation test pins, not `#358` the colour."""
    assert refs(audit, 'with pytest.warns(DeprecationWarning, match="#358"):') == [(358, "bare", "")]


def test_another_repository_is_excluded(audit):
    [ref] = audit.scan_line("x.md", 1, "", "https://github.com/VEST-Tokamak/vfit/issues/3")
    assert ref.status == "excluded" and "vfit" in ref.reason


def test_generated_and_data_files_are_excluded(audit):
    assert audit.exclusion_for("docs/assets/diagrams/x.svg")
    assert audit.exclusion_for("notebooks/data/table.csv")
    assert audit.exclusion_for("test/data/fixture.txt")
    assert not audit.exclusion_for("vaft/plot/display.py")
    assert not audit.exclusion_for("notebooks/analysis.ipynb")


def test_baseline_counts_per_path_and_number(audit):
    """A third copy of a number the baseline counts twice is the one reported."""
    found = [audit.Reference("a.py", n, 1215, "bare", "ctx") for n in (1, 2, 3)]
    audit.apply_baseline(found, {"a.py": {"1215": 2}})
    assert [r.status for r in found] == ["legacy", "legacy", "new"]


def test_retained_entries_carry_their_reason(audit):
    [ref] = [audit.Reference("a.py", 1, 358, "bare", "ctx")]
    audit.apply_baseline([ref], {"a.py": {"358": {"count": 1, "status": "retained",
                                                  "category": "compat", "reason": "pinned warning"}}})
    assert (ref.status, ref.category, ref.reason, ref.action) == (
        "retained", "compat", "pinned warning", "retain")


def test_notebooks_are_read_by_cell_source(audit, tmp_path):
    notebook = {"cells": [{"cell_type": "markdown", "source": ["Intro\n", "See #1714.\n"]},
                          {"cell_type": "code", "source": "x = 1", "outputs": [{"text": "#999"}]}]}
    (tmp_path / "n.ipynb").write_text(json.dumps(notebook), encoding="utf-8")
    lines = list(audit.text_lines(tmp_path, "n.ipynb"))
    assert (2, "markdown[0]", "See #1714.") in lines
    assert not any("#999" in text for _, _, text in lines)


def test_the_baseline_is_well_formed():
    baseline = json.loads((REPO_ROOT / ".github" / "issue-refs-baseline.json").read_text(encoding="utf-8"))
    assert baseline["schema"] == 1
    for path, numbers in baseline["references"].items():
        for number, entry in numbers.items():
            assert number.isdigit(), (path, number)
            if isinstance(entry, int):  # legacy: a count only
                assert entry >= 1, (path, number)
                continue
            assert entry["status"] == "retained" and entry["count"] >= 1, (path, number)
            assert entry.get("reason"), f"{path} #{number} is retained without a reason"
