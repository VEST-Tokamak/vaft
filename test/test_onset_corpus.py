"""The plasma-onset corpus of issue #409: the raw-database study behind the rules, as data.

``test/data/onset_corpus.json`` is written by ``workflow/plasma_onset/scan_corpus.py``
from the raw database; it records, for every shot in the scanned range, the
raw-side plasma-window verdict (``detect_plasma_window``) and both detectors'
windows.  These tests hold the table to the policy it was scanned with, to
the packaged products, and to the numbers the policy comments quote -- so a
retuned rule or a drifted detector shows up here rather than in prose.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from vaft.machine_mapping.magnetics import PLASMA_WINDOW_ANALYSIS_RANGE, PLASMA_WINDOW_H_ALPHA_RAW
from vaft.machine_mapping.utils import resolve_plasma_timing_policy
from vaft.omas.plasma_timing import SOURCE_H_PRIMARY, plasma_timing

from _plasma_timing_fixtures import pipeline_ods

TABLE = Path(__file__).parent / "data" / "onset_corpus.json"
SCRIPT = Path(__file__).resolve().parents[1] / "workflow" / "plasma_onset" / "scan_corpus.py"

#: The corpus numbers the policy comments quote (vest.yaml `plasma_timing`,
#: vaft/process/onset.py): pinned here so the prose and the data cannot drift.
PINNED = {
    "shots_requested": "39900-41700",
    "judged": 1718,
    "absent": 83,
    "windows": 1247,
    "by_source": {"h_alpha_raw": 1213, "ip": 34, "analysis_range": 471},
    "light_windows": 1213,
    "current_windows": 884,
    "both": 850,
    "no_plasma": 471,
}


@pytest.fixture(scope="module")
def table() -> dict:
    return json.loads(TABLE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def scan_module():
    spec = importlib.util.spec_from_file_location("scan_corpus", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_table_was_scanned_with_the_current_policy(table):
    """A retuned rule invalidates the corpus: the scan must be re-run, not the prose edited."""
    policy = resolve_plasma_timing_policy().as_dict()
    assert table["schema_version"] == 1
    assert table["shots_requested"] == PINNED["shots_requested"]
    for key in ("window", "baseline_lead_s", "h_alpha", "ip", "usability"):
        assert table["policy"][key] == policy[key], key


def test_the_summary_is_what_the_rows_say(table, scan_module):
    policy = resolve_plasma_timing_policy()
    assert table["summary"] == scan_module.summarize(table["rows"], policy)
    assert table["summary"]["inside_range"] is True
    assert table["summary"]["shots"] == len(table["rows"]) == 1801


def test_every_window_is_inside_the_range_and_a_fallback_has_none(table):
    policy = resolve_plasma_timing_policy()
    for row in table["rows"]:
        if row["status"] != "judged":
            assert row["reason"]
            continue
        if row["fallback"]:
            assert row["source"] == PLASMA_WINDOW_ANALYSIS_RANGE
            assert not (row["h_alpha"] and row["h_alpha"]["found"])
            assert not (row["ip"] and row["ip"]["found"])
        else:
            # 40414: a one-sample window at 0.2861 s -- the detector's answer, kept as data
            assert policy.window.tstart <= row["start"] <= row["end"] <= policy.window.tend, row["shot"]


@pytest.mark.parametrize("shot", [39915, 41524, 41672])
def test_the_packaged_products_agree_with_their_corpus_rows(table, shot):
    row = next(r for r in table["rows"] if r["shot"] == shot)
    timing = plasma_timing(pipeline_ods(shot))

    assert row["source"] == PLASMA_WINDOW_H_ALPHA_RAW and timing.source == SOURCE_H_PRIMARY
    assert row["start"] == pytest.approx(timing.onset, abs=1e-3)
    assert row["end"] == pytest.approx(timing.offset, abs=1e-3)
    assert row["ip"]["found"] and row["ip"]["start"] == pytest.approx(timing.ip.start, abs=1e-3)


def test_the_quoted_corpus_numbers_are_the_measured_ones(table):
    """The numbers the policy comments cite come from this table."""
    summary = table["summary"]
    for key, value in PINNED.items():
        if key == "shots_requested":
            continue
        assert summary[key] == value, key
