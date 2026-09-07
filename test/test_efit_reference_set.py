"""The EFIT-quality reference set: every declared claim is checked, not asserted.

The set exists so #171, #196, #468, #459 and #579 all run on the same shots
for stated reasons. What is pinned here is that each declared file exists,
that the arms a shot claims are the arms its data supports, and the two
properties the set was built to guarantee: one shot covers both arms with its
kinetic data inside the reconstructed window, and the magnetics arm varies.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

TABLE = Path(__file__).resolve().parent / "data" / "efit_reference_set.json"
SCRIPT = Path(__file__).resolve().parents[1] / "workflow" / "efit_reference_set" / "build_reference_set.py"


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location("build_reference_set", SCRIPT)
    loaded = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = loaded
    try:
        spec.loader.exec_module(loaded)
    except Exception:
        del sys.modules[spec.name]
        raise
    yield loaded
    sys.modules.pop(spec.name, None)


@pytest.fixture(scope="module")
def table():
    return json.loads(TABLE.read_text(encoding="utf-8"))


def test_the_declared_set_and_the_built_table_are_the_same_shots(table, module):
    assert table["schema_version"] == module.SCHEMA
    assert [item["shot"] for item in table["shots"]] == [entry.shot for entry in module.REFERENCE_SET]


def test_every_declared_file_is_in_the_checkout(table):
    """A set that has drifted from the repository must fail, not report less."""
    assert table["summary"]["missing_files"] == []
    for item in table["shots"]:
        files = item["files"]
        for record in (files["product"], files["thomson_mat"], files["kinetic_ods"]):
            if record is not None:
                assert record["exists"], record["path"]
                assert record["bytes"] > 0
        for record in files["equilibrium_reference"]:
            assert record["exists"], record["path"]


def test_a_shot_claims_only_the_arms_its_data_supports(table):
    for item in table["shots"]:
        arms = set(item["arms"])
        assert arms <= {"magnetics", "kinetic"}
        assert ("magnetics" in arms) == bool(item["magnetics"].get("available"))
        kinetic = item["thomson"].get("available") or item["kinetic_ods_contents"].get("available")
        assert ("kinetic" in arms) == bool(kinetic)
        if not item["magnetics"].get("available"):
            assert item["magnetics"]["reason"]


def test_39915_is_the_dual_arm_anchor_with_kinetic_data_inside_its_window(table):
    """The property the whole set turns on.

    Thomson measured outside the reconstructed window cannot check that
    reconstruction, so the overlap is asserted rather than the file's presence.
    """
    row = next(item for item in table["shots"] if item["shot"] == 39915)
    assert row["arms"] == ["kinetic", "magnetics"]
    thomson = row["thomson"]
    assert thomson["available"] and thomson["channels"] == 5
    assert thomson["overlaps_window"] is True
    assert thomson["samples_in_window"] == thomson["samples"] == 10
    window = row["magnetics"]["window"]
    assert window[0] <= thomson["times"][0] and thomson["times"][1] <= window[1]
    assert table["summary"]["both_arms"] == [39915]


def test_the_magnetics_arm_actually_varies(table):
    """Three shots with the same verdict and the same defects would not be a set."""
    magnetics = [item for item in table["shots"] if item["magnetics"].get("available")]
    assert len(magnetics) == 3
    condemned = [len(item["magnetics"]["condemned"]) for item in magnetics]
    assert condemned == sorted(condemned) and len(set(condemned)) == 3
    windows = [tuple(item["magnetics"]["window"]) for item in magnetics]
    assert len(set(windows)) == 3
    fractions = [item["magnetics"]["min_usable_fraction"] for item in magnetics]
    assert max(fractions) - min(fractions) > 0.01


def test_48224_is_the_kinetic_case_and_says_what_it_cannot_do(table):
    row = next(item for item in table["shots"] if item["shot"] == 48224)
    assert row["arms"] == ["kinetic"]
    assert row["magnetics"]["available"] is False
    assert "no pre-EFIT product" in row["magnetics"]["reason"]
    contents = row["kinetic_ods_contents"]
    assert contents["available"]
    assert {"thomson_scattering", "charge_exchange", "core_profiles", "equilibrium"} <= set(contents["ids"])
    assert contents["charge_exchange_channels"] == 40
    assert contents["core_profiles"]["measured_points"] == 7
    assert contents["equilibrium_times"] == [0.3]
    # The near-axis defect must stay visible in the set, not only in issue #317.
    assert any("psi_N < 0.05" in note for note in row["notes"])


def test_the_report_renders_from_the_committed_table(module, table):
    text = module.markdown(table)
    assert text.startswith("# The EFIT-quality reference set")
    for item in table["shots"]:
        assert f"## {item['shot']}" in text
    assert "Missing files" not in text
