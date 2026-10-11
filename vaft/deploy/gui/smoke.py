"""Smoke checks for a hosted ``vaft gui`` (#1862): ``python -m vaft.deploy.gui.smoke URL``.

Run against a deployed instance -- through its nginx route, or an SSH tunnel to
its port -- after an install, an upgrade or a rollback.  Each check prints one
line, PASS or FAIL with the reason; the exit status is the number of failures.

* ``login-required`` -- the page is not served without signing in;
* ``bad-login-refused`` -- a wrong password does not sign in;
* ``login`` -- the account in ``VAFT_SMOKE_USERNAME`` / ``VAFT_SMOKE_PASSWORD``
  signs in and the page is served;
* ``websocket`` -- the page's websocket upgrades (HTTP 101) with that sign-in,
  through any proxy in between.

The password is read from the environment and never printed or logged, nor
put on the command line.  Without credentials only the first two checks run.
``--insecure`` accepts a self-signed certificate (the test instance's).
"""

from __future__ import annotations

import argparse
import base64
import http.client
import http.cookiejar
import os
import secrets
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable

__all__ = ["main", "run_checks"]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # report redirects instead of following them
        return None


def _opener(context: ssl.SSLContext | None, jar: http.cookiejar.CookieJar, *, follow: bool) -> urllib.request.OpenerDirector:
    handlers: list = [urllib.request.HTTPCookieProcessor(jar), urllib.request.HTTPSHandler(context=context)]
    if not follow:
        handlers.append(_NoRedirect())
    return urllib.request.build_opener(*handlers)


def _status(opener: urllib.request.OpenerDirector, url: str, data: bytes | None = None) -> tuple[int, str]:
    """``(status, Location header)`` of one request, redirects not followed."""
    try:
        with opener.open(urllib.request.Request(url, data=data), timeout=30) as response:
            return response.status, response.headers.get("Location", "")
    except urllib.error.HTTPError as error:
        return error.code, error.headers.get("Location", "")


def _login(opener, base: str, username: str, password: str) -> tuple[int, str]:
    form = urllib.parse.urlencode({"username": username, "password": password}).encode()
    return _status(opener, urllib.parse.urljoin(base, "login"), data=form)


def _websocket_upgrade(base: str, jar: http.cookiejar.CookieJar, context: ssl.SSLContext | None) -> int:
    """The HTTP status of a websocket upgrade to the page's ``ws`` endpoint."""
    parts = urllib.parse.urlsplit(urllib.parse.urljoin(base, "ws"))
    secure = parts.scheme == "https"
    port = parts.port or (443 if secure else 80)
    connection = (
        http.client.HTTPSConnection(parts.hostname, port, context=context, timeout=30)
        if secure else http.client.HTTPConnection(parts.hostname, port, timeout=30)
    )
    cookies = "; ".join(f"{cookie.name}={cookie.value}" for cookie in jar)
    origin = f"{parts.scheme}://{parts.netloc}"
    connection.request("GET", parts.path, headers={
        "Connection": "Upgrade", "Upgrade": "websocket", "Sec-WebSocket-Version": "13",
        "Sec-WebSocket-Key": base64.b64encode(secrets.token_bytes(16)).decode(),
        "Origin": origin, "Cookie": cookies,
    })
    status = connection.getresponse().status
    connection.close()
    return status


def run_checks(
    url: str, *, username: str | None = None, password: str | None = None, insecure: bool = False,
) -> list[tuple[str, bool, str]]:
    """Run the checks against ``url`` (the page, e.g. ``https://host/gui-test/``).

    Returns ``(name, passed, detail)`` per check; ``detail`` never holds the password.
    """
    base = url if url.endswith("/") else url + "/"
    context = ssl._create_unverified_context() if insecure else ssl.create_default_context()  # noqa: S323
    results: list[tuple[str, bool, str]] = []

    anonymous = _opener(context, http.cookiejar.CookieJar(), follow=False)
    try:
        status, location = _status(anonymous, base)
    except (OSError, urllib.error.URLError) as error:
        return [("reachable", False, f"{base}: {error}")]
    gated = status in (301, 302, 303, 307, 401, 403) and (status != 302 or "login" in location)
    results.append(("login-required", gated, f"GET without sign-in -> {status} {location}".strip()))

    jar = http.cookiejar.CookieJar()
    wrong = _opener(context, jar, follow=False)
    status, location = _login(wrong, base, username or "vaft-smoke", secrets.token_urlsafe(18))
    page_status, _ = _status(wrong, base)
    refused = page_status != 200
    results.append(("bad-login-refused", refused, f"wrong password -> login {status}, page {page_status}"))

    if username and password:
        jar = http.cookiejar.CookieJar()
        signed = _opener(context, jar, follow=False)
        status, location = _login(signed, base, username, password)
        page_status, _ = _status(signed, base)
        results.append(("login", page_status == 200, f"sign-in as {username} -> login {status}, page {page_status}"))
        if page_status == 200:
            try:
                upgrade = _websocket_upgrade(base, jar, context)
            except OSError as error:
                results.append(("websocket", False, f"upgrade failed: {error}"))
            else:
                results.append(("websocket", upgrade == 101, f"upgrade -> {upgrade}"))
    return results


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m vaft.deploy.gui.smoke", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url", help="the page, e.g. https://147.46.36.244/gui-test/")
    parser.add_argument("--insecure", action="store_true", help="accept a self-signed certificate")
    args = parser.parse_args(list(argv) if argv is not None else None)
    results = run_checks(
        args.url, username=os.environ.get("VAFT_SMOKE_USERNAME"),
        password=os.environ.get("VAFT_SMOKE_PASSWORD"), insecure=args.insecure,
    )
    for name, passed, detail in results:
        print(f"{'PASS' if passed else 'FAIL'}  {name:18s} {detail}")
    if not os.environ.get("VAFT_SMOKE_PASSWORD"):
        print("note: set VAFT_SMOKE_USERNAME and VAFT_SMOKE_PASSWORD to check sign-in and the websocket")
    return sum(not passed for _, passed, _ in results)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
