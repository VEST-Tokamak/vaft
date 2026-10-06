"""Sign in to the GUI with an HSDS account (``vaft gui --auth hsds``).

The login form's user name and password are put to the HSDS the GUI reads
from: ``GET /about`` answers 200 for a valid account and 401 for anything
else, whatever the account may read.  The password is used for that one
request and kept nowhere; the data is still read with the service's own
credentials (``HS_USERNAME`` / ``.hscfg``), so every reader sees the same
shots.
"""

from __future__ import annotations

import base64
import logging
import os
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

#: How long one sign-in waits for HSDS; the login request blocks the server.
TIMEOUT_S = 5.0


class HSDSUnavailable(RuntimeError):
    """HSDS gave no verdict on an account: unreachable, or an unexpected answer."""


def hsds_endpoint() -> str:
    """The endpoint h5pyd reads from: ``$HS_ENDPOINT``, else the active ``.hscfg``."""
    endpoint = os.environ.get("HS_ENDPOINT")
    if not endpoint:
        from ..database.hscfg import active_path, read_values

        path = active_path()
        endpoint = read_values(path).get("hs_endpoint") if path.is_file() else None
    if not endpoint:
        raise ValueError("--auth hsds needs the HSDS endpoint: set HS_ENDPOINT or hs_endpoint in .hscfg")
    return endpoint.rstrip("/")


def check_hsds_login(username: str, password: str, endpoint: str, *, timeout: float = TIMEOUT_S) -> bool:
    """Whether HSDS at ``endpoint`` accepts ``username`` / ``password``.

    Raises :class:`HSDSUnavailable` when HSDS cannot say: a sign-in must not
    succeed, nor be reported as a wrong password, because HSDS is down.
    """
    if not username or not password:
        return False
    token = base64.b64encode(f"{username}:{password}".encode()).decode("ascii")
    request = urllib.request.Request(f"{endpoint}/about", headers={"Authorization": f"Basic {token}"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
    except urllib.error.HTTPError as error:
        status = error.code
    except (urllib.error.URLError, OSError) as error:
        raise HSDSUnavailable(f"HSDS at {endpoint} is unreachable: {error}") from None
    if status == 200:
        return True
    if status in (401, 403):
        return False
    raise HSDSUnavailable(f"HSDS at {endpoint} answered {status} to a sign-in")


def hsds_auth_provider(endpoint: str) -> Any:
    """A Panel auth provider whose login form signs in with an HSDS account."""
    import tornado.escape
    from panel.auth import BasicAuthProvider, BasicLoginHandler

    class HSDSLoginHandler(BasicLoginHandler):
        def _validate(self, username: str, password: str) -> bool:
            return check_hsds_login(username, password, endpoint)

        def post(self) -> None:
            try:
                super().post()
            except HSDSUnavailable as error:
                logger.warning("vaft gui sign-in: %s", error)
                message = "HSDS cannot check accounts right now; try again shortly."
                self.redirect(self.request.uri + "?error=" + tornado.escape.url_escape(message))

    class HSDSAuthProvider(BasicAuthProvider):
        @property
        def login_handler(self) -> type:
            HSDSLoginHandler._login_endpoint = self._login_endpoint
            HSDSLoginHandler._login_template = self._login_template
            return HSDSLoginHandler

    return HSDSAuthProvider()


__all__ = ["TIMEOUT_S", "HSDSUnavailable", "check_hsds_login", "hsds_auth_provider", "hsds_endpoint"]
