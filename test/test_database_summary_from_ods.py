"""The offline preset summary equals the database one for the same data (#1232).

``vaft.database.summary(odss=...)`` / ``summary_from_ods`` run the preset's
extractor on in-memory ODS. The database path is exercised with ``open``
replaced by a function serving a copy of the same packaged sample, so the two
tables must be identical.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy

import pandas as pd
import pytest

import vaft
from vaft import database

SHOT = 39915


@pytest.fixture(scope="module")
def sample():
    return vaft.omas.sample_ods(SHOT)


def _serve(sample, opened):
    @contextmanager
    def fake_open(shot, *, source=None, paths=None, **kwargs):
        opened.append((shot, source, tuple(paths)))
        if shot != SHOT:
            raise FileNotFoundError(f"shot {shot} not stored")
        yield copy.deepcopy(sample)

    return fake_open


@pytest.mark.parametrize("preset", ["equilibrium_global", "shot_overview"])
def test_the_offline_table_equals_the_database_table(sample, monkeypatch, preset):
    opened = []
    monkeypatch.setattr(database, "open", _serve(sample, opened))
    from_database = database.summary((SHOT, SHOT), preset=preset, source="public")
    offline = database.summary(odss={SHOT: sample}, preset=preset, source="public")

    assert opened and len(from_database) > 0
    pd.testing.assert_frame_equal(offline, from_database)
    assert offline.attrs["skipped"] == from_database.attrs["skipped"] == []
    assert tuple(offline.columns) == database.get_summary_preset(preset).columns


def test_summary_from_ods_is_the_same_call(sample):
    pd.testing.assert_frame_equal(
        database.summary_from_ods({SHOT: sample}, preset="shot_overview"),
        database.summary(odss={SHOT: sample}, preset="shot_overview"),
    )


def test_the_shot_can_come_from_the_ods(sample):
    by_key = database.summary_from_ods({SHOT: sample}, preset="shot_overview")
    by_record = database.summary_from_ods([sample], preset="shot_overview")
    pd.testing.assert_frame_equal(by_key, by_record)


def test_the_callers_ods_is_not_modified(sample):
    before = set(sample.flat().keys())
    database.summary_from_ods({SHOT: sample}, preset="equilibrium_global")
    assert set(sample.flat().keys()) == before


def _labelled_preset(fail_for: int):
    """A preset with both source columns whose extractor fails for one shot."""
    import vaft.database._summary as summary_module

    def extract(ods, shot):
        if shot == fail_for:
            raise KeyError("equilibrium.time")
        return [{"shot": shot, "eq_index": 0, "time_s": 0.1}]

    return summary_module.SummaryPreset(
        columns=("shot", "pulse_time_begin", "eq_index", "time_s", "source", "equilibrium_source"),
        paths=("equilibrium",),
        key_columns=("shot", "eq_index", "time_s"),
        replace_groups=("shot",),
        sort_columns=("shot", "time_s", "eq_index"),
        extractor=extract,
    )


def test_a_failing_shot_is_skipped_and_recorded(sample, monkeypatch, caplog):
    import vaft.database._summary as summary_module

    monkeypatch.setitem(summary_module.PRESETS, "labelled", _labelled_preset(fail_for=1))
    table = database.summary_from_ods({SHOT: sample, 1: sample}, preset="labelled")
    assert table["shot"].tolist() == [SHOT]
    assert table.attrs["skipped"] == [{"shot": 1, "reason": "KeyError: 'equilibrium.time'"}]
    assert "Shot 1" in caplog.text


def test_the_database_path_records_skipped_shots_too(sample, monkeypatch):
    monkeypatch.setattr(database, "open", _serve(sample, []))
    table = database.summary((SHOT, SHOT + 1), preset="shot_overview", source="public")
    assert [entry["shot"] for entry in table.attrs["skipped"]] == [SHOT + 1]


def test_odss_excludes_a_shot_range(sample):
    with pytest.raises(TypeError, match="omit shot_range"):
        database.summary((SHOT, SHOT), odss={SHOT: sample})


def test_an_ods_without_its_shot_needs_a_mapping():
    from omas import ODS

    with pytest.raises(ValueError, match="mapping"):
        database.summary_from_ods([ODS(consistency_check=False)])


def test_the_source_label_defaults_to_local(sample, monkeypatch):
    import vaft.database._summary as summary_module

    monkeypatch.setitem(summary_module.PRESETS, "labelled", _labelled_preset(fail_for=-1))
    table = database.summary_from_ods({SHOT: sample}, preset="labelled")
    assert table[["source", "equilibrium_source"]].iloc[0].tolist() == ["local", "local"]
    named = database.summary(odss={SHOT: sample}, preset="labelled", source="campaign-a")
    assert named["source"].iloc[0] == "campaign-a"
