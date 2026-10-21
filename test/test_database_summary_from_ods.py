"""The offline preset summary equals the database one for the same data (#1232).

``vaft.database.summary(odss=...)`` / ``summary_from_ods`` run the preset's
extractor on in-memory ODS. The database path is exercised with ``open``
replaced by a function that serves what the lazy database view serves: only
the root IDS named in ``paths`` (``FileNotFoundError`` for one the shot does
not store), copied from the same packaged sample. The two tables must be
identical, including where an IDS outside the preset would change the result.
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


def _serve(stored, opened):
    """``open`` as the lazy database view behaves: only the requested root IDS."""
    from omas import ODS

    @contextmanager
    def fake_open(shot, *, source=None, paths=None, **kwargs):
        opened.append((shot, source, tuple(paths)))
        if shot not in stored:
            raise FileNotFoundError(f"shot {shot} not stored")
        ods = stored[shot]
        view = ODS(imas_version=ods.imas_version, consistency_check=False)
        for name in dict.fromkeys(path.split(".", 1)[0] for path in paths):
            if name not in ods:
                raise FileNotFoundError(f"IDS not stored for shot {shot}: {name}")
            view[name] = copy.deepcopy(ods[name])
        yield view

    return fake_open


@pytest.mark.parametrize(
    ("preset", "shot"),
    [("equilibrium_global", 39915), ("equilibrium_global", 41672),
     ("shot_overview", 39915), ("volume_averaged", 48224)],
)
def test_the_offline_table_equals_the_database_table(monkeypatch, preset, shot):
    stored = {shot: vaft.omas.sample_ods(shot)}
    opened = []
    monkeypatch.setattr(database, "open", _serve(stored, opened))
    from_database = database.summary((shot, shot), preset=preset, source="public")
    offline = database.summary(odss=stored, preset=preset, source="public")

    assert opened and len(from_database) > 0
    pd.testing.assert_frame_equal(offline, from_database)
    assert offline.attrs["skipped"] == from_database.attrs["skipped"] == []
    assert tuple(offline.columns) == database.get_summary_preset(preset).columns


def test_an_ids_outside_the_preset_does_not_change_the_table(sample):
    """The diamagnetic update reads magnetics, which equilibrium_global does not request."""
    from omas import ODS

    assert "magnetics" in sample
    without = ODS(imas_version=sample.imas_version, consistency_check=False)
    for name in sample.keys():
        if name != "magnetics":
            without[name] = copy.deepcopy(sample[name])
    pd.testing.assert_frame_equal(
        database.summary_from_ods({SHOT: sample}),
        database.summary_from_ods({SHOT: without}),
    )


def test_a_missing_preset_ids_is_skipped_by_the_database_but_not_offline(monkeypatch):
    """The documented divergence: open() refuses a requested IDS the shot lacks.

    The packaged 48224 sample stores core_profiles but no magnetics, which the
    core_profiles preset also requests. Offline, the extractor runs on what is
    there; the database path skips the shot.
    """
    stored = {48224: vaft.omas.sample_ods(48224)}
    assert "magnetics" not in stored[48224]
    monkeypatch.setattr(database, "open", _serve(stored, []))
    from_database = database.summary((48224, 48224), preset="core_profiles", source="public")
    offline = database.summary(odss=stored, preset="core_profiles", source="public")
    assert from_database.empty and "magnetics" in from_database.attrs["skipped"][0]["reason"]
    assert offline["shot"].tolist() == [48224] and offline.attrs["skipped"] == []


def _snapshot(ods) -> dict:
    import numpy as np

    return {key: np.asarray(value).tolist() if hasattr(value, "shape") else value
            for key, value in ods.flat().items()}


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
    keys, values = list(sample.keys()), _snapshot(sample)
    database.summary_from_ods({SHOT: sample}, preset="equilibrium_global")
    database.summary_from_ods([sample], preset="shot_overview")
    assert list(sample.keys()) == keys
    assert _snapshot(sample) == values


def test_an_ods_without_its_shot_is_not_modified_by_the_refusal():
    from omas import ODS

    ods = ODS(consistency_check=False)
    ods["equilibrium.time"] = [0.1]
    with pytest.raises(ValueError, match="mapping"):
        database.summary_from_ods([ods])
    assert list(ods.keys()) == ["equilibrium"]


def test_a_shot_given_twice_is_refused(sample):
    with pytest.raises(ValueError, match="more than once"):
        database.summary_from_ods([sample, sample], preset="shot_overview")


def test_source_none_means_local(sample, monkeypatch):
    import vaft.database._summary as summary_module

    monkeypatch.setitem(summary_module.PRESETS, "labelled", _labelled_preset(fail_for=-1))
    table = database.summary_from_ods({SHOT: sample}, preset="labelled", source=None)
    assert table["source"].iloc[0] == "local"


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
    monkeypatch.setattr(database, "open", _serve({SHOT: sample}, []))
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
