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

SHOTS_REQUESTED = "39900-41700"


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
    assert table["shots_requested"] == SHOTS_REQUESTED
    for key in ("window", "baseline_lead_s", "h_alpha", "ip", "usability", "agreement"):
        assert table["policy"][key] == policy[key], key


def test_the_summary_is_what_the_rows_say(table, scan_module):
    policy = resolve_plasma_timing_policy()
    summary = table["summary"]
    assert summary == scan_module.summarize(table["rows"], policy)
    assert summary["inside_range"] is True
    assert summary["errors"] == []                       # a detector fault is never filed as missing data
    assert summary["shots"] == len(table["rows"]) == 1801
    assert summary["judged"] + summary["absent"] == summary["shots"]
    assert sum(summary["by_source"].values()) == summary["judged"]
    assert summary["windows"] + summary["no_plasma"] == summary["judged"]
    assert sum(summary["agreement"].values()) == summary["both"]


def test_every_window_is_inside_the_range_and_a_fallback_has_none(table):
    policy = resolve_plasma_timing_policy()
    for row in table["rows"]:
        if row["status"] != "judged":
            assert row["reason"]
            continue
        if row["fallback"]:
            assert row["source"] == PLASMA_WINDOW_ANALYSIS_RANGE
            assert not (row["h_alpha"] and row["h_alpha"]["start"] is not None)
            assert not (row["ip"] and row["ip"]["start"] is not None)
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
    assert row["ip"]["start"] == pytest.approx(timing.ip.start, abs=1e-3)
    assert row["agreement"] == timing.agreement == "consistent"


def test_the_raw_and_ods_readers_share_one_agreement_vocabulary(table):
    """A corpus row's agreement is the ODS composer's word for the same pair of windows."""
    from vaft.machine_mapping.utils import (
        AGREEMENT_CONSISTENT,
        AGREEMENT_HALPHA_LEADS_IP_LARGE,
        AGREEMENT_IP_BEFORE_HALPHA,
    )

    words = {AGREEMENT_CONSISTENT, AGREEMENT_IP_BEFORE_HALPHA, AGREEMENT_HALPHA_LEADS_IP_LARGE}
    assert set(table["summary"]["agreement"]) <= words
    for row in table["rows"]:
        if row.get("agreement") is not None:
            assert row["agreement"] in words
            if row["agreement"] != AGREEMENT_CONSISTENT:
                assert row["agreement"] in row["flags"]
