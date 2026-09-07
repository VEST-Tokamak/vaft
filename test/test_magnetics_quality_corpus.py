"""The magnetics-quality corpus: the policy, the packaged rows, self-consistency.

What is pinned here is the *policy* and the *shape*, plus the handful of
findings the studies now rest on. Counts that are properties of a detector
threshold are deliberately not pinned: those belong to
`test_magnetics_signal_quality.py`, and duplicating them here would make one
threshold change fail in two places for the same reason.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

TABLE = Path(__file__).resolve().parent / "data" / "magnetics_quality.json"
SCRIPT = Path(__file__).resolve().parents[1] / "workflow" / "magnetics_quality" / "scan_magnetics_quality.py"
PACKAGED = (39915, 41524, 41672)


@pytest.fixture(scope="module")
def module():
    """Load the script by path, registered.

    ``sys.modules`` registration is not optional here: a dataclass resolves
    its annotations through ``sys.modules[cls.__module__].__dict__``, so an
    unregistered by-path load makes ``FitnessPolicy`` unconstructable.
    """
    spec = importlib.util.spec_from_file_location("scan_magnetics_quality", SCRIPT)
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


def test_the_table_records_the_policy_it_was_scanned_with(table, module):
    """A verdict without its policy is unreadable a year later."""
    assert table["schema_version"] == module.SCHEMA
    recorded = table["summary"]["policy"]
    # Through JSON: the policy's tuples come back as lists, and what is being
    # pinned is the recorded value, not Python's representation of it.
    assert recorded == json.loads(json.dumps(module.FitnessPolicy().as_dict()))
    assert set(recorded) == {
        "min_family_witnesses",
        "small_families",
        "min_usable_fraction",
        "min_window_coverage",
        "min_window_coverage_unfit",
    }


def test_the_summary_agrees_with_the_rows_it_summarizes(table):
    rows = table["rows"]
    summary = table["summary"]
    assert summary["shots"] == len(rows)
    assert summary["assessed"] == sum(1 for row in rows if row["status"] == "assessed")
    assert summary["absent"] == sum(1 for row in rows if row["status"] == "absent")
    assert summary["errors"] == [row["shot"] for row in rows if row["status"] == "error"]
    verdicts: dict[str, int] = {}
    for row in rows:
        if row["status"] == "assessed":
            verdicts[row["verdict"]] = verdicts.get(row["verdict"], 0) + 1
    assert summary["verdicts"] == verdicts
    counted: dict[str, int] = {}
    for row in rows:
        for name in row.get("condemned", []):
            counted[name] = counted.get(name, 0) + 1
    assert summary["condemned_by_shot_count"] == counted


def test_every_packaged_shot_was_assessed(table):
    assert [row["shot"] for row in table["rows"]] == list(PACKAGED)
    assert all(row["status"] == "assessed" for row in table["rows"])


@pytest.mark.parametrize("shot", PACKAGED)
def test_a_row_carries_the_evidence_behind_its_verdict(table, shot):
    row = next(item for item in table["rows"] if item["shot"] == shot)
    assert row["verdict"] in {"fit", "degraded", "unfit"}
    assert row["reasons"] if row["verdict"] != "fit" else True
    window = row["window"]
    assert window["end"] > window["start"] and window["slices"] >= 1
    decisions = row["decisions"]
    assert decisions["channels"] == row["nbprobe"] + 11
    assert decisions["slices"] == window["slices"]
    assert sum(decisions["state_slice_counts"].values()) == decisions["channels"] * decisions["slices"]
    assert 0.0 <= decisions["min_usable_fraction"] <= 1.0


@pytest.mark.parametrize("shot", PACKAGED)
def test_the_routine_window_lies_inside_the_measured_record(table, shot):
    """39915 holds its last value from 0.34 s; its window ends before that.

    The hold is real (issue #244's class of defect) and a study that widened
    the window past the measured end would be fitting held values, so the
    coverage is asserted rather than assumed.
    """
    row = next(item for item in table["rows"] if item["shot"] == shot)
    span = row["measured_span"]
    assert span["covers_window"] is True
    assert span["coverage_fraction"] == pytest.approx(1.0)
    assert span["last"] >= row["window"]["end"]
    # Per channel, not only in the median: a median over 87 channels cannot
    # show one that stopped early, and one that did would be fitted on held
    # values for the rest of the window.
    assert span["channels_covering_window"] == span["channels_with_a_span"] > 0
    assert span["earliest_last"] >= row["window"]["end"]


def test_h3_08_is_condemned_on_every_packaged_shot(table):
    """The one channel every packaged shot agrees about."""
    condemned = table["summary"]["condemned_by_shot_count"]
    h3_08 = [name for name in condemned if "H3-08" in name]
    assert len(h3_08) == 1
    assert condemned[h3_08[0]] == len(PACKAGED)


def test_the_products_carry_no_projected_validity_for_efit_channels(table):
    """"All valid" and "never looked at" are the same bytes in a product.

    This is why the sweep assesses and gates into a copy instead of reading
    what it is handed; if a future product does carry a projection, this test
    is the place that notices.
    """
    for row in table["rows"]:
        projection = row["decisions"]["projection_in_product"]
        assert projection["efit_facing"] == row["decisions"]["channels"]
        assert projection["efit_facing_with_projected_validity"] == 0
        assert projection["with_projected_validity"] > 0  # the IMPA channels do carry one


def test_the_flux_loops_agree_with_the_vacuum_model_everywhere(table):
    """Nothing in the evidence rejects a flux loop on any packaged shot (#295)."""
    for row in table["rows"]:
        model = row["model"]
        assert model["consulted"] and model["available"], row["shot"]
        assert set(model["loop_states"]) == {"usable"}
        assert model["loop_states"]["usable"] == 11
        assert model["normalized_residual"]["n"] == 11
        assert 0.0 < model["normalized_residual"]["median"] < 0.1


def test_the_verdict_rules_are_the_ones_the_readme_states(module):
    """The policy is executable, not prose: exercise it on synthetic summaries."""
    policy = module.FitnessPolicy()
    healthy = {
        "families": {"inboard": {"min_witnesses": 20, "exempt": False}},
        "min_usable_fraction": 1.0,
        "rejected_channels": [],
        "missing_channels": [],
    }
    covered = {"coverage_fraction": 1.0, "last": 0.4}
    assert module._verdict(healthy, covered, policy)["verdict"] == "fit"

    starved = dict(healthy, families={"inboard": {"min_witnesses": 2, "exempt": False}})
    assert module._verdict(starved, covered, policy)["verdict"] == "unfit"

    exempt = dict(healthy, families={"inboard_flux_loop": {"min_witnesses": 2, "exempt": True}})
    assert module._verdict(exempt, covered, policy)["verdict"] == "fit"

    thin = dict(healthy, min_usable_fraction=0.5)
    assert module._verdict(thin, covered, policy)["verdict"] == "unfit"

    rejected = dict(healthy, rejected_channels=["b_field_pol_probe[25]"])
    assert module._verdict(rejected, covered, policy)["verdict"] == "degraded"

    short = module._verdict(healthy, {"coverage_fraction": 0.95, "last": 0.33}, policy)
    assert short["verdict"] == "degraded" and "held values" in short["reasons"][0]

    missing_record = module._verdict(healthy, {"coverage_fraction": 0.5, "last": 0.31}, policy)
    assert missing_record["verdict"] == "unfit"


def test_shot_specifications_parse_the_way_the_corpus_scan_does(module):
    assert module.parse_shots("39915") == [39915]
    assert module.parse_shots("39915,41524") == [39915, 41524]
    assert module.parse_shots("41000-41003") == [41000, 41001, 41002, 41003]
    assert module.parse_shots("39915,39915") == [39915]
    with pytest.raises(ValueError, match="reversed"):
        module.parse_shots("41003-41000")


def test_the_report_renders_from_the_committed_table(module, table):
    text = module.markdown(table)
    assert text.startswith("# Magnetics quality")
    for shot in PACKAGED:
        assert f"## {shot}" in text
    assert "Policy: " in text
