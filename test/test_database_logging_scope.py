"""The database helpers quieten the HTTP stack, never the root logger (#1936).

``is_connect``, ``exist_shot``, the processed-registry reader and the ODS/IDS
loaders used to call
``logging.getLogger().setLevel(logging.WARNING)``: one *Test connection* click
in the hosted GUI silenced INFO logging for every session in the process.
"""

from __future__ import annotations

import logging

import pytest

from vaft.database import utils


@pytest.fixture
def logger_levels():
    """Restore the root, h5pyd and urllib3 levels whatever the test did."""
    names = ("", *utils._NOISY_HTTP_LOGGERS)
    saved = {name: logging.getLogger(name).level for name in names}
    yield
    for name, level in saved.items():
        logging.getLogger(name).setLevel(level)


def _server_info(state):
    return lambda *a, **k: {"state": state}


def test_is_connect_leaves_the_root_logger_alone(logger_levels, monkeypatch):
    monkeypatch.setattr(utils.h5pyd, "getServerInfo", _server_info("READY"))
    logging.getLogger().setLevel(logging.INFO)
    for name in utils._NOISY_HTTP_LOGGERS:
        logging.getLogger(name).setLevel(logging.NOTSET)

    assert utils.is_connect() is True

    assert logging.getLogger().level == logging.INFO
    for name in utils._NOISY_HTTP_LOGGERS:
        assert logging.getLogger(name).level == logging.WARNING


def test_an_explicitly_configured_http_logger_is_kept(logger_levels, monkeypatch):
    monkeypatch.setattr(utils.h5pyd, "getServerInfo", _server_info("READY"))
    logging.getLogger("urllib3").setLevel(logging.DEBUG)
    logging.getLogger("h5pyd").setLevel(logging.NOTSET)

    utils.is_connect()

    assert logging.getLogger("urllib3").level == logging.DEBUG
    assert logging.getLogger("h5pyd").level == logging.WARNING


def test_exist_shot_leaves_the_root_logger_alone(logger_levels, monkeypatch):
    monkeypatch.setattr(utils, "resolve_source", lambda source, directory=None: source)
    monkeypatch.setattr(utils, "_exist_shot_ts", lambda sort=True, source=None: None)
    logging.getLogger().setLevel(logging.DEBUG)

    utils.exist_shot("public", data_filter="ts")

    assert logging.getLogger().level == logging.DEBUG


def test_the_registry_reader_leaves_the_root_logger_alone(logger_levels, monkeypatch):
    def unreachable(*a, **k):
        raise OSError(404, "Not Found")

    monkeypatch.setattr(utils, "processed_registry_uri", lambda source=None: "/public/x.h5")
    monkeypatch.setattr(utils.h5pyd, "File", unreachable)
    logging.getLogger().setLevel(logging.INFO)

    utils._exist_shot_ts(source="public")

    assert logging.getLogger().level == logging.INFO


def test_no_database_module_sets_the_root_level():
    """The loaders (ods.load, ids.load/save) carried the same root-level call."""
    import re
    from pathlib import Path

    root_level = re.compile(r"getLogger\((\"\"|''|None)?\)\.setLevel|logging\.root\.setLevel")
    package = Path(utils.__file__).parent
    offenders = [
        path.relative_to(package).as_posix()
        for path in sorted(package.rglob("*.py"))
        if root_level.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []
