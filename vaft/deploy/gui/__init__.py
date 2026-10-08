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

The GUI guide ("Host it for a team") says where each one goes.
"""
