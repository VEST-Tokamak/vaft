"""Signing in to the GUI with an HSDS account (``vaft gui --auth hsds``)."""

from __future__ import annotations

import base64
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import ClassVar

import pytest

from vaft.gui import auth as gui_auth

ACCOUNTS = {"alice": "s3cret"}


class _FakeHSDS(BaseHTTPRequestHandler):
    """``GET /about``: 200 for a known account, 401 otherwise, as HSDS answers."""

    status_override: int | None = None
    requests: ClassVar[list[str]] = []

    def do_GET(self):
        type(self).requests.append(self.path)
        if type(self).status_override is not None:
            status = type(self).status_override
        else:
            header = self.headers.get("Authorization", "")
            user, _, password = base64.b64decode(header.removeprefix("Basic ")).decode().partition(":")
            status = 200 if self.path == "/about" and ACCOUNTS.get(user) == password else 401
        self.send_response(status)
        self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture
def hsds():
    _FakeHSDS.status_override = None
    _FakeHSDS.requests = []
    server = HTTPServer(("127.0.0.1", 0), _FakeHSDS)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_hsds_decides_the_account(hsds):
    assert gui_auth.check_hsds_login("alice", "s3cret", hsds) is True
    assert gui_auth.check_hsds_login("alice", "wrong", hsds) is False
    assert gui_auth.check_hsds_login("mallory", "s3cret", hsds) is False
    assert _FakeHSDS.requests == ["/about"] * 3


def test_an_empty_field_is_refused_without_asking(hsds):
    assert gui_auth.check_hsds_login("alice", "", hsds) is False
    assert gui_auth.check_hsds_login("", "s3cret", hsds) is False
    assert not _FakeHSDS.requests


def test_no_verdict_is_neither_a_sign_in_nor_a_wrong_password(hsds):
    _FakeHSDS.status_override = 503
    with pytest.raises(gui_auth.HSDSUnavailable, match="503"):
        gui_auth.check_hsds_login("alice", "s3cret", hsds)
    with pytest.raises(gui_auth.HSDSUnavailable, match="unreachable"):
        gui_auth.check_hsds_login("alice", "s3cret", "http://127.0.0.1:9", timeout=1)


def test_the_endpoint_is_the_one_h5pyd_reads(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HS_ENDPOINT", raising=False)
    with pytest.raises(ValueError, match="HS_ENDPOINT"):
        gui_auth.hsds_endpoint()
    (tmp_path / ".hscfg").write_text("hs_endpoint = http://hsds.example:5101/\n")
    assert gui_auth.hsds_endpoint() == "http://hsds.example:5101"
    monkeypatch.setenv("HS_ENDPOINT", "http://127.0.0.1:5101")
    assert gui_auth.hsds_endpoint() == "http://127.0.0.1:5101"


def test_the_login_form_signs_in_through_hsds(hsds):
    pytest.importorskip("panel")
    from tornado.testing import AsyncHTTPTestCase
    from tornado.web import Application

    provider = gui_auth.hsds_auth_provider(hsds)
    outcomes = {}

    class Login(AsyncHTTPTestCase):
        def get_app(self):
            return Application([(r"/login", provider.login_handler)], cookie_secret="test-secret")

        def runTest(self):
            for name, body in {
                "right": "username=alice&password=s3cret",
                "wrong": "username=alice&password=nope",
            }.items():
                response = self.fetch("/login", method="POST", body=body, follow_redirects=False, raise_error=False)
                outcomes[name] = (response.code, response.headers.get("Location", ""),
                                  "user=" in response.headers.get("Set-Cookie", ""))
            _FakeHSDS.status_override = 503
            response = self.fetch("/login", method="POST", body="username=alice&password=s3cret",
                                  follow_redirects=False, raise_error=False)
            outcomes["down"] = (response.code, response.headers.get("Location", ""),
                                "user=" in response.headers.get("Set-Cookie", ""))

    result = Login().run()
    assert result is None or result.wasSuccessful(), getattr(result, "errors", None)
    assert outcomes["right"][0] == 302 and outcomes["right"][2], "a valid account gets the session cookie"
    assert outcomes["wrong"][0] == 302 and "Invalid" in outcomes["wrong"][1] and not outcomes["wrong"][2]
    assert outcomes["down"][0] == 302 and "HSDS" in outcomes["down"][1] and not outcomes["down"][2]
