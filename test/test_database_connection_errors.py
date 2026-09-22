"""An unreachable HSDS is a connection error; a 403 or 404 is not.

h5pyd raises a bare ``OSError("Connection Error")`` from inside
``except requests.ConnectionError`` for a transport failure, and
``OSError(status, reason)`` for an HTTP error. The folder helpers used to catch
``urllib3.exceptions.MaxRetryError``, which h5pyd never raises, so an
unreachable server escaped as an OSError while the "Connection error" branch
never ran.
"""

from __future__ import annotations

import pytest
import requests

from vaft.database import utils


def _connection_failure():
    """Raise exactly what h5pyd's httpconn does when the endpoint is unreachable."""
    try:
        raise requests.exceptions.ConnectionError("Max retries exceeded")
    except requests.exceptions.ConnectionError:
        raise OSError("Connection Error")


def _forbidden():
    raise OSError(403, "Forbidden")


def _not_found():
    raise OSError(404, "Not Found")


def test_the_connection_failure_form_is_recognised():
    with pytest.raises(OSError) as info:
        _connection_failure()
    assert utils._is_connection_failure(info.value)


@pytest.mark.parametrize("raiser", [_forbidden, _not_found])
def test_an_http_status_is_not_a_connection_failure(raiser):
    with pytest.raises(OSError) as info:
        raiser()
    assert not utils._is_connection_failure(info.value)


def test_a_plain_oserror_without_the_requests_cause_is_not_one():
    assert not utils._is_connection_failure(OSError("Connection Error"))


@pytest.fixture
def folder(monkeypatch):
    def install(raiser):
        monkeypatch.setattr(utils.h5pyd, "Folder", lambda *a, **k: raiser())
    return install


def test_namespace_listing_reports_an_unreachable_server(folder, capsys):
    folder(_connection_failure)
    assert utils._get_namespace_folders("public") == []
    assert "Connection error" in capsys.readouterr().out


def test_exist_shot_reports_an_unreachable_server(folder, capsys):
    folder(_connection_failure)
    assert utils.exist_shot("public", 39915) is False
    assert "Connection error" in capsys.readouterr().out


@pytest.mark.parametrize("raiser", [_forbidden, _not_found])
def test_namespace_listing_does_not_hide_an_http_error(folder, raiser):
    folder(raiser)
    with pytest.raises(OSError) as info:
        utils._get_namespace_folders("public")
    assert info.value.args[0] in (403, 404)


def test_exist_shot_does_not_hide_a_forbidden_folder(folder):
    folder(_forbidden)
    with pytest.raises(OSError) as info:
        utils.exist_shot("public", 39915)
    assert info.value.args[0] == 403
