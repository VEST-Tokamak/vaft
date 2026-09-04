"""Which shots the sparse IMPA source is allowed to contain (issue #305).

Absence from `impa` means "no published IMPA product". That stays true only if
the pipeline attempts the stage where the array was recording, rather than for
every eligible shot -- which would fill the source with empty products and turn
absence back into an ambiguous signal.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import sys

import pytest


WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / "workflow"
    / "automatic_pipeline_1_routine_data_processing"
)

pytestmark = pytest.mark.skipif(
    not WORKFLOW.exists(), reason="workflow scripts are not part of the distribution"
)


@pytest.fixture(scope="module")
def select():
    sys.path.insert(0, str(WORKFLOW))
    try:
        from select_impa_shots import select_impa_shots
    finally:
        sys.path.remove(str(WORKFLOW))
    return select_impa_shots


def _dump(path: Path, fields) -> Path:
    payload = {"shot": 0, "fields": {str(field): {"data": [0.0, 1.0]} for field in fields}}
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return path


def test_a_shot_archiving_the_array_is_selected(tmp_path, select):
    dump = _dump(tmp_path / "vest_39915_daq_raw.json.gz", range(114, 122))
    report, = select([39915], {39915: dump})
    assert report["selected"]
    assert report["archived_fields"] == [114, 115, 116, 117, 118, 119, 120, 121]


def test_a_shot_without_the_array_is_left_out_rather_than_published_empty(tmp_path, select):
    dump = _dump(tmp_path / "vest_39915_daq_raw.json.gz", [1, 12, 25])
    report, = select([39915], {39915: dump})
    assert not report["selected"]
    assert report["reason"] == "no wired IMPA channel is archived"


def test_selection_uses_the_shot_era_channel_list_not_the_default(tmp_path, select):
    """The 2022-04-23 block wires seven channels; 121 is not missing there."""
    dump = _dump(tmp_path / "vest_35375_daq_raw.json.gz", range(114, 121))
    report, = select([35375], {35375: dump})
    assert report["selected"]
    assert 121 not in report["expected_fields"]


def test_a_shot_with_no_raw_dump_is_reported_rather_than_raising(select):
    report, = select([39915], {})
    assert not report["selected"] and report["reason"] == "no raw dump"
