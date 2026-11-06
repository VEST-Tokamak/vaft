"""``python -m vaft.deploy.gui.smoke`` against a local signed-in Panel server."""

from __future__ import annotations

import secrets
import socket
import time
import urllib.request

import pytest

pn = pytest.importorskip("panel")

from vaft.deploy.gui import smoke  # noqa: E402


@pytest.fixture(scope="module")
def signed_in_server():
    """A threaded Panel server behind the password sign-in ``vaft gui --auth password`` uses."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    password = secrets.token_urlsafe(12)
    thread = pn.serve(
        {"/": lambda: pn.pane.Markdown("signed in")}, port=port, address="127.0.0.1", show=False,
        threaded=True, basic_auth=password, cookie_secret=secrets.token_urlsafe(32),
        websocket_origin=[f"127.0.0.1:{port}"],
    )
    url = f"http://127.0.0.1:{port}/"
    deadline = time.time() + 30
    while True:
        try:
            urllib.request.urlopen(url + "login", timeout=5).read()
            break
        except OSError:
            if time.time() > deadline:
                thread.stop()
                pytest.fail("the signed-in test server did not start")
            time.sleep(0.2)
    yield url, password
    thread.stop()


def _by_name(results):
    return {name: (passed, detail) for name, passed, detail in results}


def test_a_correct_sign_in_passes_every_check(signed_in_server):
    url, password = signed_in_server
    results = _by_name(smoke.run_checks(url, username="reader", password=password))
    assert set(results) == {"login-required", "login-page", "bad-login-refused", "login", "websocket"}
    assert all(passed for passed, _ in results.values()), results
    assert all(password not in detail for _, detail in results.values()), "the password is never reported"


def test_a_wrong_password_fails_the_login_check(signed_in_server):
    url, _ = signed_in_server
    results = _by_name(smoke.run_checks(url, username="reader", password="not-it"))
    assert results["login-required"][0] and results["bad-login-refused"][0]
    assert not results["login"][0] and "websocket" not in results


def test_without_credentials_only_the_gate_is_checked(signed_in_server, capsys, monkeypatch):
    url, _ = signed_in_server
    monkeypatch.delenv("VAFT_SMOKE_USERNAME", raising=False)
    monkeypatch.delenv("VAFT_SMOKE_PASSWORD", raising=False)
    assert smoke.main([url]) == 0
    out = capsys.readouterr().out
    assert "PASS  login-required" in out and "PASS  login-page" in out and "PASS  bad-login-refused" in out
    assert "set VAFT_SMOKE_USERNAME" in out


def test_an_unprotected_page_fails_the_gate():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    thread = pn.serve({"/": lambda: pn.pane.Markdown("open")}, port=port, address="127.0.0.1", show=False,
                      threaded=True, websocket_origin=[f"127.0.0.1:{port}"])
    try:
        time.sleep(1.5)
        results = _by_name(smoke.run_checks(f"http://127.0.0.1:{port}/"))
        assert not results["login-required"][0], "an open page must not pass as signed-in-only"
    finally:
        thread.stop()


def test_an_unreachable_server_is_one_clear_failure():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    (name, passed, detail), = smoke.run_checks(f"http://127.0.0.1:{port}/")
    assert name == "reachable" and not passed and str(port) in detail


def test_a_deny_or_a_foreign_redirect_is_not_a_sign_in_gate():
    """An nginx 403 or a redirect to another place says nothing about the GUI's sign-in."""
    import http.server
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path.startswith("/deny/"):
                self.send_response(403)
            else:  # an http->https style redirect elsewhere
                self.send_response(301)
                self.send_header("Location", "https://elsewhere.example/gui-test/")
            self.end_headers()

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        port = server.server_address[1]
        (name, passed, detail), = smoke.run_checks(f"http://127.0.0.1:{port}/deny/")
        assert name == "login-required" and not passed and "403" in detail
        results = _by_name(smoke.run_checks(f"http://127.0.0.1:{port}/gui-test/"))
        assert not results["login-required"][0], "a redirect elsewhere is not a login redirect"
        assert not results["login-page"][0]
    finally:
        server.shutdown()
