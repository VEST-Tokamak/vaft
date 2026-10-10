"""The application shell (#1174): workspaces, the shared selection, status; never served."""

from __future__ import annotations

import os
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import pytest

from vaft.gui.selection import Selection, SelectionState
from vaft.gui.shell import WorkspaceRegistry, WorkspaceSpec
from vaft.gui.state import BrowserSession, Source

pn = pytest.importorskip("panel")

from vaft.gui import app as gui_app  # noqa: E402
from vaft.gui import workspaces as gui_workspaces  # noqa: E402
from vaft.gui.shell import Shell  # noqa: E402


# -- the selection, without Panel -----------------------------------------------------
def test_a_selection_notifies_once_per_change_and_names_who_changed_it():
    state = SelectionState()
    seen = []
    state.subscribe(lambda selection, origin: seen.append((selection, origin)))
    assert state.update(origin="db", namespace="main")
    assert not state.update(origin="db", namespace="main"), "no change, no notification"
    assert state.update(origin="plots", sources=[Source("shot", 39915, "main")], time="0.31 s")
    assert [origin for _, origin in seen] == ["db", "plots"]
    assert state.value.sources == (Source("shot", 39915, "main"),) and "39915" in state.value.label
    assert Selection().label == "nothing open"
    with pytest.raises(TypeError, match="no field"):
        state.update(shot=1)
    with pytest.raises(TypeError, match="Source"):
        Selection(sources=(39915,))


def test_the_registry_orders_workspaces_and_refuses_a_second_of_one_name():
    registry = WorkspaceRegistry()
    registry.register(WorkspaceSpec("b", "B", object, order=20))
    registry.register(WorkspaceSpec("a", "A", object, order=10))
    registry.register(WorkspaceSpec("c", "C", object, order=20))
    assert registry.names() == ["a", "b", "c"], "by order, then registration"
    with pytest.raises(ValueError, match="already registered"):
        registry.register(WorkspaceSpec("a", "A again", object))
    with pytest.raises(ValueError, match="plain key"):
        registry.register(WorkspaceSpec("a b", "spaced", object))
    with pytest.raises(KeyError, match="workspaces: a, b, c"):
        registry.get("nope")


def test_vaft_gui_lists_its_builtin_workspaces():
    import vaft.gui

    assert vaft.gui.WORKSPACES.names()[:4] == ["plots", "diagnostics", "equilibrium", "database"]


# -- the shell ------------------------------------------------------------------------
class _Toy:
    built = 0

    def __init__(self, shell):
        type(self).built += 1
        self.shell = shell
        self.shown = 0
        self.closed = False
        self.marker = pn.pane.Markdown("toy main")

    def sidebar(self):
        return [pn.pane.Markdown("toy side")]

    def main(self):
        return [self.marker]

    def activate(self):
        self.shown += 1

    def close(self):
        self.closed = True


class _Broken:
    def __init__(self, shell):
        raise RuntimeError("cannot build\nsecond line")


@pytest.fixture
def toys():
    registry = WorkspaceRegistry()
    _Toy.built = 0
    registry.register(WorkspaceSpec("one", "One", _Toy, "first", order=1))
    registry.register(WorkspaceSpec("two", "Two", _Toy, "second", order=2))
    registry.register(WorkspaceSpec("broken", "Broken", _Broken, order=3))
    return registry


def test_workspaces_are_built_when_first_shown_and_swapped_in(toys):
    shell = Shell(toys)
    assert shell.active == "one" and _Toy.built == 1, "only the shown workspace is built"
    assert shell.main_box.objects[0] is shell.workspaces["one"].marker
    shell.nav.value = "two"
    assert shell.active == "two" and _Toy.built == 2 and shell.hint.object == "second"
    shell.show("one")
    assert shell.nav.value == "one" and _Toy.built == 2, "built once, shown again"
    assert shell.workspaces["one"].shown == 2
    shell.close()
    assert not shell.workspaces


def test_a_workspace_that_cannot_be_built_leaves_the_previous_one(toys):
    shell = Shell(toys)
    shell.nav.value = "broken"
    assert shell.active == "one" and shell.nav.value == "one"
    assert shell.alert.visible and shell.alert.object == "Broken: cannot build"


def test_an_unknown_first_workspace_is_refused_with_the_choices(toys):
    with pytest.raises(KeyError, match="one, two, broken"):
        Shell(toys, initial="nope")


def test_the_status_line_follows_the_selection_and_the_connection(toys, monkeypatch):
    shell = Shell(toys)
    assert "nothing open" in shell.status.object and "not checked" in shell.status.object
    shell.selection.update(sources=(Source("sample", 39915),), time="t = 0.310 s")
    assert "sample 39915" in shell.status.object and "t = 0.310 s" in shell.status.object
    monkeypatch.setattr("vaft.database.utils.is_connect", lambda: True)
    assert shell.check_connection() == "ready" and "ready" in shell.status.object

    def unreachable():
        raise OSError("connection refused")

    monkeypatch.setattr("vaft.database.utils.is_connect", unreachable)
    assert shell.check_connection() == "unreachable (connection refused)"


# -- the built-in workspaces ----------------------------------------------------------
_PLOTS = {
    "equilibrium_field_psi": ("matplotlib", "plotly"),
    "plasma_current_time": ("matplotlib",),
}


class _Session(BrowserSession):
    def _discover(self, data):
        return [SimpleNamespace(name=n, subject=n.split("_")[0], backends=b) for n, b in _PLOTS.items()]


@pytest.fixture(scope="module")
def ods():
    from vaft.omas import sample_ods

    return sample_ods(39915)


@pytest.fixture
def shell(ods, monkeypatch, tmp_path):
    loaded = []

    def load(source):
        loaded.append(source)
        return ods

    monkeypatch.setattr("vaft.gui.state.load_source", load)
    monkeypatch.setattr("vaft.database.hscfg.active_path", lambda cwd=None: tmp_path / ".hscfg")
    for key in ("HS_ENDPOINT", "HS_USERNAME", "HS_PASSWORD", "HS_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    app = gui_app.BrowserApp(_Session(), plot="equilibrium_field_psi")
    built = Shell(factories={"plots": lambda shell: gui_workspaces.PlotWorkspace(shell, app)})
    assert app.load(Source("sample", 39915))
    built.loaded = loaded
    yield built
    built.close()


def test_the_explorer_publishes_what_is_open_and_the_time_it_shows(shell):
    selection = shell.selection.value
    assert selection.sources == (Source("sample", 39915),)
    explorer = shell.workspaces["plots"].app
    state = explorer.session.state
    assert selection.time is not None and selection.time == explorer.selected_time()
    other = next(v for v in state.spec("time_slice").options if v != state["time_slice"])
    state.set("time_slice", other)
    assert shell.selection.value.time == explorer.selected_time() != selection.time, "moving the slice moves the time"
    explorer.plot.value = "plasma_current_time"
    assert shell.selection.value.time is None, "a time trace selects no time"


def test_the_database_workspace_opens_shots_in_the_explorer(shell):
    shell.show("database")
    database = shell.workspaces["database"]
    names = [source.name for source in database._sources]
    assert database.namespace.value in names and "main" in names
    assert "| `main` |" in database.table.object
    explorer = shell.workspaces["plots"].app
    opened = []
    explorer.load = lambda sources: opened.append(tuple(sources)) or True  # no HSDS here
    database.shots.value = "41524, 41672"
    assert database.open_shots()
    assert shell.active == "plots"
    wanted = (Source("shot", 41524, database.namespace.value), Source("shot", 41672, database.namespace.value))
    assert shell.selection.value.sources == wanted
    assert opened == [wanted], "the explorer opened them, once"
    assert explorer.kind.value == "shot" and explorer.shots.value == "41524, 41672"


def test_a_bad_shot_list_is_reported_and_opens_nothing(shell):
    shell.show("database")
    database = shell.workspaces["database"]
    before = shell.selection.value
    database.shots.value = "39915, x"
    assert not database.open_shots()
    assert shell.alert.visible and shell.alert.object.startswith("Database:")
    database.shots.value = ""
    assert not database.open_shots() and shell.selection.value == before and shell.active == "database"


def test_a_shot_opened_in_the_explorer_moves_the_database_namespace(shell):
    shell.show("database")
    database = shell.workspaces["database"]
    names = list(database.namespace.options)
    other = next(name for name in names if name != database.namespace.value)
    explorer = shell.workspaces["plots"].app
    explorer.session.sources = (Source("shot", 41524, other),)  # as a database load leaves it
    explorer._changed()
    assert shell.selection.value.namespace == other and database.namespace.value == other


def test_credentials_are_summarised_without_their_secrets(shell, tmp_path, monkeypatch):
    path = tmp_path / ".hscfg"
    path.write_text("hs_endpoint = https://hsds.example\nhs_username = alice\nhs_password = hunter2-secret\n",
                    encoding="utf-8")
    path.chmod(0o644)
    shell.show("database")
    database = shell.workspaces["database"]
    summary = database.refresh_credentials()
    text = database.credentials.object
    assert "hunter2-secret" not in text and "hunter2-secret" not in repr(summary)
    assert "https://hsds.example" in text and "alice" in text and "Password: configured" in text
    assert "API key: not set" in text
    if os.name == "nt":
        # hscfg.insecure_permissions is False on Windows by design (no POSIX mode bits),
        # so the GUI must not hand out advice the platform cannot act on.
        assert "chmod 600" not in text and "warning" not in summary
    else:
        assert "chmod 600" in text
    monkeypatch.setenv("HS_API_KEY", "key-from-env")
    summary = gui_workspaces.credential_summary(path)
    assert summary["hs_api_key"] == "configured (environment)" and "key-from-env" not in repr(summary)


def test_the_connection_check_reaches_the_status_line(shell, monkeypatch):
    shell.show("database")
    monkeypatch.setattr("vaft.database.utils.is_connect", lambda: False)
    assert shell.workspaces["database"].check() == "not ready"
    assert "not ready" in shell.workspaces["database"].connection.object and "not ready" in shell.status.object


def test_build_shell_serves_the_shell_with_the_first_source_drawn(monkeypatch, ods):
    monkeypatch.setattr("vaft.gui.state.load_source", lambda source: ods)
    onload = []
    monkeypatch.setattr(pn.state, "onload", lambda callback: onload.append(callback))
    shell = gui_app.build_shell(sample=39915, plot="plasma_current_time", workspace="database")
    try:
        assert shell.active == "database" and set(shell.workspaces) == {"plots", "database"}
        explorer = shell.workspaces["plots"].app
        assert explorer.sample.value == [39915]
        for callback in onload:
            callback()
        assert shell.selection.value.sources == (Source("sample", 39915),)
    finally:
        shell.close()


def test_serve_builds_a_shell_per_session(monkeypatch):
    calls = []
    monkeypatch.setattr(pn, "serve", lambda panels, **kwargs: calls.append(panels))
    built = []
    monkeypatch.setattr(gui_app, "build_shell", lambda **options: built.append(options) or SimpleNamespace(
        view=lambda: "page", close=lambda: None,
    ))
    monkeypatch.setattr(pn.state, "on_session_destroyed", lambda callback: None)
    gui_app.serve(port=5123, sample=[39915], workspace="database")
    assert calls[-1]["/"]() == "page"
    assert built[-1] == {"sample": [39915], "workspace": "database", "hosted": False}


def test_an_explorer_handed_over_already_loaded_is_published(monkeypatch, ods):
    monkeypatch.setattr("vaft.gui.state.load_source", lambda source: ods)
    app = gui_app.BrowserApp(_Session(), plot="plasma_current_time")
    assert app.load(Source("sample", 39915))
    shell = Shell(factories={"plots": lambda shell: gui_workspaces.PlotWorkspace(shell, app)})
    try:
        assert shell.selection.value.sources == (Source("sample", 39915),)
    finally:
        shell.close()


# -- cold-review cases ----------------------------------------------------------------
def test_a_failed_open_leaves_the_selection_on_what_is_open_and_can_be_retried(shell):
    shell.show("database")
    database = shell.workspaces["database"]
    explorer = shell.workspaces["plots"].app
    before = explorer.session.sources
    attempts = []

    def fail(sources):
        attempts.append(tuple(sources))
        explorer.alert.object = "403 Forbidden"
        return False

    explorer.load = fail
    database.shots.value = "41524"
    assert database.open_shots()
    assert shell.selection.value.sources == before, "the failed shots are not reported as open"
    assert shell.alert.visible and "403" in shell.alert.object
    shell.show("database")
    assert database.open_shots() and len(attempts) == 2, "the same shots are tried again"


def test_a_selection_returning_to_what_is_open_drops_the_pending_request(shell):
    shell.show("database")
    plots = shell.workspaces["plots"]
    open_now = plots.app.session.sources
    shell.selection.update(origin="elsewhere", sources=(Source("shot", 41524, "main"),))
    shell.selection.update(origin="elsewhere", sources=open_now)
    loads = []
    plots.app.load = lambda sources: loads.append(sources) or True
    shell.show("plots")
    assert loads == []


def test_build_shell_refuses_an_unknown_workspace_before_loading(monkeypatch):
    built = []
    monkeypatch.setattr(gui_app, "build_app", lambda **options: built.append(options))
    with pytest.raises(KeyError, match="no workspace named 'nope'"):
        gui_app.build_shell(sample=39915, workspace="nope")
    assert built == []


def test_a_silent_server_is_given_up_on(toys, monkeypatch):
    import threading

    release = threading.Event()
    monkeypatch.setattr("vaft.database.utils.is_connect", lambda: release.wait(5))
    shell = Shell(toys)
    try:
        assert shell.check_connection(timeout=0.05) == "no answer within 0.05 s"
    finally:
        release.set()


def test_a_workspace_whose_layout_raises_is_reported(toys):
    class _NoLayout(_Toy):
        def main(self):
            raise RuntimeError("no main")

    toys.register(WorkspaceSpec("nolayout", "No layout", _NoLayout, order=4))
    shell = Shell(toys)
    shell.nav.value = "nolayout"
    assert shell.active == "one" and shell.alert.object == "No layout: no main"


def test_a_hosted_shell_keeps_the_server_s_credentials_to_itself(ods, monkeypatch, tmp_path):
    path = tmp_path / ".hscfg"
    path.write_text("hs_endpoint = http://127.0.0.1:5101\nhs_username = service-reader\nhs_password = s3cret\n")
    monkeypatch.setattr("vaft.database.hscfg.active_path", lambda cwd=None: path)
    monkeypatch.setattr("vaft.gui.state.load_source", lambda source: ods)
    app = gui_app.BrowserApp(_Session(), hosted=True)
    hosted = Shell(factories={"plots": lambda shell: gui_workspaces.PlotWorkspace(shell, app)}, hosted=True)
    try:
        hosted.show("database")
        text = hosted.workspaces["database"].credentials.object
        assert "service-reader" not in text and "127.0.0.1" not in text and str(path) not in text
        assert "read-only account" in text
        # files stay refused through the shell's shared selection, too
        hosted.selection.update(sources=(Source("file", str(path)),))
        hosted.show("plots")
        assert not any(source.kind == "file" for source in app.session.sources)
    finally:
        hosted.close()
