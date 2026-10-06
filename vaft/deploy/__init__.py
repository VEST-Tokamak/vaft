"""Deployment templates that ship with the package (#1755).

Nothing here is imported by the library.  The subpackages hold the files a
server needs to run VAFT for a team -- ``vaft.deploy.gui`` the nginx site,
systemd unit and environment template behind ``vaft gui --hosted`` -- so a
PyPI install carries the same templates a repository checkout does.  Read them
through :func:`importlib.resources.files`, never through a path relative to a
checkout::

    from importlib.resources import files
    files("vaft.deploy.gui").joinpath("vaft-gui.service").read_text(encoding="utf-8")
"""
