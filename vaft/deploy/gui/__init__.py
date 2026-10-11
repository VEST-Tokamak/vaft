"""The hosted-GUI deployment files (``vaft gui --hosted``, #1086, #1406).

Package data, listed in ``pyproject.toml`` and ``MANIFEST.in`` and required by
``test/verify_dist.py``:

``vaft-gui.service``
    the systemd unit, for ``/etc/systemd/system/``
``vaft-gui.env.example``
    the environment file it reads, for ``/etc/vaft-gui.env`` (mode 0600)
``nginx-vaft-gui.conf``
    the nginx site serving ``/gui/`` over HTTPS next to HSDS
``vaft-gui-proxy.conf``
    the websocket proxy snippet the site includes, for ``/etc/nginx/snippets/``

An isolated **test instance** next to a production one (#1862) -- own user,
venv, port 5016 and ``/gui-test/`` prefix, loopback only:

``vaft-gui-test.service``, ``vaft-gui-test.env.example``
    its systemd unit and environment file (a dedicated read-only HSDS account)
``nginx-vaft-gui-test-locations.conf``, ``vaft-gui-test-proxy-headers.conf``
    locations to include in the existing 443 server, internal addresses only

and :mod:`vaft.deploy.gui.smoke`, the smoke checks to run after an install,
upgrade or rollback (``python -m vaft.deploy.gui.smoke URL``).

The GUI guide ("Host it for a team") says where each one goes.
"""
