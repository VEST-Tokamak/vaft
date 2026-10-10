"""Pipeline 3 must not depend on the legacy ``processed_shots.h5`` registry.

The registry was written by the quarantined polling scripts and is not
maintained by the Snakemake pipeline 2. A summary generator that listed shots
through ``exist_shot(data_filter=...)`` would silently come out empty on a source
filled by pipeline 2 alone (issue #1842).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FILTERS = {"ts", "cx", "cp", "thomson_scattering", "charge_exchange", "core_profiles"}

SOURCES = sorted((ROOT / "workflow" / "automatic_pipeline_3_data_summary").glob("*.py")) + [
    ROOT / "vaft" / "database" / "_summary.py"
]


def _registry_uses(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "attr", getattr(node.func, "id", ""))
            if name in {"exist_shot", "_read_processed_registry", "read_legacy_processed_registry"}:
                keywords = {k.arg: k.value for k in node.keywords}
                data_filter = keywords.get("data_filter")
                if name != "exist_shot" or data_filter is not None or len(node.args) >= 3:
                    found.append(f"{path.name}:{node.lineno} {name}")
        elif isinstance(node, ast.Constant) and node.value == "processed_shots.h5":
            found.append(f"{path.name}:{node.lineno} processed_shots.h5")
    return found


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_summary_code_does_not_read_the_legacy_registry(path):
    if not path.exists():
        pytest.skip("workflow scripts are not part of the distribution")
    assert _registry_uses(path) == []


def test_the_summary_still_lists_shots_without_a_filter():
    """The listing it does use is the unfiltered, strict one."""
    text = (ROOT / "vaft" / "database" / "_summary.py").read_text(encoding="utf-8")
    assert "exist_shot(source=source, sort=1, strict=True)" in text
