"""Regression coverage for compatibility access used by shipped workflows."""

import pytest

import vaft.database as database
from vaft.database import ods as database_ods


def test_compatibility_attributes_resolve():
    assert callable(database.exist_ts_file)
    assert callable(database_ods.exist_ts_file)
    assert callable(database_ods.load)
    assert callable(database.load)


def test_ods_load_warns_and_forwards(monkeypatch):
    called = {}

    def fake_load_ods(shot, directory="public", **kwargs):
        called.update(shot=shot, directory=directory, kwargs=kwargs)
        return "ODS"

    monkeypatch.setattr(database_ods, "load_ods", fake_load_ods)
    with pytest.warns(DeprecationWarning):
        result = database_ods.load(
            39915, directory="public", paths=["magnetics"]
        )
    assert result == "ODS"
    assert called == {
        "shot": 39915,
        "directory": "public",
        "kwargs": {"paths": ["magnetics"]},
    }


def test_exist_ts_file_warns_and_forwards(monkeypatch):
    called = {}

    def fake_exist_shot(*args, **kwargs):
        called.update(kwargs)
        return "TS"

    monkeypatch.setattr(database_ods, "exist_shot", fake_exist_shot)
    with pytest.warns(DeprecationWarning):
        assert database_ods.exist_ts_file() == "TS"
    assert called == {"data_filter": "ts"}
