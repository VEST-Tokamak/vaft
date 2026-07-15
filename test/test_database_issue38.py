"""Regression tests for issue #38.

`vaft.database.exist_ts_file()` and `vaft.database.ods.load()` were called by six
shipped notebooks/pipeline scripts but never existed, raising AttributeError at
runtime. These tests pin the compatibility shims: the attributes must resolve and
be callable via every access pattern used in the codebase, forward to the real
implementations, and emit a DeprecationWarning (not AttributeError).
"""
import warnings

import pytest

import vaft.database as db
from vaft.database import ods as db_ods


def test_attributes_resolve_and_are_callable():
    # the exact access patterns used by the six call sites
    assert callable(db.exist_ts_file)          # notebooks: vaft.database.exist_ts_file()
    assert callable(db_ods.exist_ts_file)      # pipeline:  db_ods.exist_ts_file()
    assert callable(db_ods.load)               # notebook/pipeline: db_ods.load(shot, directory=...)
    assert callable(db.load)                    # recommended replacement


def test_ods_load_forwards_to_load_ods_with_deprecation(monkeypatch):
    called = {}

    def fake_load_ods(shot, directory="public", **kwargs):
        called.update(shot=shot, directory=directory, kwargs=kwargs)
        return "ODS"

    monkeypatch.setattr(db_ods, "load_ods", fake_load_ods)
    with pytest.warns(DeprecationWarning):
        out = db_ods.load(39915, directory="public_omas", paths=["magnetics"])
    assert out == "ODS"
    assert called == {"shot": 39915, "directory": "public_omas", "kwargs": {"paths": ["magnetics"]}}


def test_exist_ts_file_forwards_to_exist_shot_ts(monkeypatch):
    called = {}

    def fake_exist_shot(*args, **kwargs):
        called.update(kwargs)
        return "TS_DATAFRAME"

    # the shim imports exist_shot into the ods module namespace
    monkeypatch.setattr(db_ods, "exist_shot", fake_exist_shot)
    with pytest.warns(DeprecationWarning):
        out = db_ods.exist_ts_file()
    assert out == "TS_DATAFRAME"
    assert called.get("data_filter") == "ts"


def test_calling_does_not_raise_attributeerror():
    # even offline the shims must fail (if at all) on connection/backend, never AttributeError
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for fn in (db.exist_ts_file, db_ods.exist_ts_file, db_ods.load):
            try:
                fn(39915) if fn is db_ods.load else fn()
            except AttributeError:
                pytest.fail(f"{fn.__name__} raised AttributeError -- shim missing")
            except Exception:
                pass  # connection / h5pyd / backend errors are acceptable here
