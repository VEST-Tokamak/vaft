"""Hosted-session isolation and cleanup (#1863): one server, several browser sessions.

A real Bokeh server serves :func:`vaft.gui.app.session_page` -- the page
``vaft gui`` serves -- and plain HTTP requests open sessions on it, each
building its own :class:`~vaft.gui.shell.Shell`.  The tests drive the shells
directly (no browser), so they check what the server holds, not what a page
shows.
"""

from __future__ import annotations

import socket
import time
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest

from vaft.gui.state import Source

pn = pytest.importorskip("panel")

from vaft.gui import app as gui_app  # noqa: E402
from vaft.gui.shell import WORKSPACES  # noqa: E402


@pytest.fixture(scope="module")
def ods():
    from vaft.omas import sample_ods

    return sample_ods(39915)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def server(ods, monkeypatch, request):
    """A threaded server of the production page; yields (open_session, shells).

    ``@pytest.mark.parametrize("server", [ms], indirect=True)`` sets how long a
    session no browser connects to is kept (default: a minute, longer than a test).
    """
    lifetime = getattr(request, "param", 60_000)
    monkeypatch.setattr("vaft.gui.state.load_source", lambda source: ods)
    shells = []
    real_build = gui_app.build_shell

    def recording_build(**options):
        shell = real_build(**options)
        shells.append(shell)
        return shell

    monkeypatch.setattr(gui_app, "build_shell", recording_build)
    port = _free_port()
    thread = pn.serve(
        {"/": gui_app.session_page(sample=39915, plot="plasma_current_time")},
        port=port, address="127.0.0.1", show=False, threaded=True,
        websocket_origin=[f"127.0.0.1:{port}"],
        # A session without a browser connection is discarded after this,
        # as a closed tab is.
        unused_session_lifetime=lifetime, check_unused_sessions=100,
    )
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5).read()
            break
        except OSError:
            time.sleep(0.2)
    shells.clear()  # the readiness probe's session is not one of ours

    def open_session():
        before = len(shells)
        urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=30).read()
        assert len(shells) == before + 1, "each page request builds its own shell"
        return shells[-1]

    try:
        yield open_session, shells
    finally:
        thread.stop()


def _wait(condition, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(0.1)
    return condition()


def test_two_sessions_share_nothing_a_reader_changes(server):
    open_session, _shells = server
    first, second = open_session(), open_session()
    assert first is not second and first.selection is not second.selection
    one, two = first.workspaces["plots"].app, second.workspaces["plots"].app
    assert one is not two and one.session is not two.session

    # The first reader opens a shot, moves the slice, switches workspace.
    assert one.load(Source("sample", 39915))
    one.plot.value = "equilibrium_field_psi"
    state = one.session.state
    other = next(v for v in state.spec("time_slice").options if v != state["time_slice"])
    state.set("time_slice", other)
    first.show("database")

    # The second reader sees none of it (its own sample loads only once a
    # browser connects, which none does here).
    assert first.selection.value.sources == (Source("sample", 39915),) and first.selection.value.time
    assert second.selection.value.sources == () and second.selection.value.time is None
    assert second.active == "plots" and "database" not in second.workspaces
    assert two.session.plot is None and two.session.figure is None and two.session.ods is None


def test_session_activity_leaves_the_process_wide_registry_alone(server):
    open_session, _shells = server
    before = [(spec.name, spec.factory) for spec in WORKSPACES.specs()]
    shell = open_session()
    shell.show("database")
    shell.show("diagnostics")
    assert [(spec.name, spec.factory) for spec in WORKSPACES.specs()] == before


@pytest.mark.parametrize("server", [3000], indirect=True)
def test_a_discarded_session_releases_its_data_figures_and_uploads(server):
    import matplotlib.pyplot as plt

    open_session, _shells = server
    baseline = set(plt.get_fignums())
    shell = open_session()
    explorer = shell.workspaces["plots"].app
    assert explorer.load(Source("sample", 39915))
    explorer.renderer.value = "matplotlib"  # a pyplot figure, which must be closed
    uploaded = explorer._store_upload(["note.json"], [b"{}"])
    folder = Path(explorer._uploads.name)
    assert uploaded and folder.exists() and explorer.session.ods is not None

    # No browser ever connected: the server discards the session like a closed tab.
    assert _wait(lambda: not shell.workspaces), "the shell was closed when its session ended"
    assert explorer.session.ods is None and explorer.session.figure is None
    assert not folder.exists(), "uploads are deleted with the session"
    assert set(plt.get_fignums()) <= baseline, "no figure outlives its session"
