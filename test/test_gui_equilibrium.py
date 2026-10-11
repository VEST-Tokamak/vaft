"""The Equilibrium workspace (#1352): modes, one shared slice, validation verdicts; never served."""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import pytest

from vaft.gui import equilibrium
from vaft.gui.state import Source

pn = pytest.importorskip("panel")

from vaft.gui.shell import Shell  # noqa: E402


# -- presentation rules, without Panel -----------------------------------------------------
def _record(view, quantity="", domain="equilibrium", subject="equilibrium"):
    return SimpleNamespace(name=f"{view}_{quantity}", view=view, quantity=quantity, domain=domain, subject=subject)


def test_every_equilibrium_plot_lands_in_one_mode():
    assert equilibrium.mode_of(_record("field", "psi")) == "inspect"
    assert equilibrium.mode_of(_record("profile", "q")) == "inspect"
    assert equilibrium.mode_of(_record("overview", "residuals")) == "constraints"
    assert equilibrium.mode_of(_record("overview", "convergence")) == "constraints"
    assert equilibrium.mode_of(_record("time", "q95")) == "time"
    assert equilibrium.mode_of(_record("overview", "histories")) == "time"
    assert equilibrium.mode_of(_record("table", "fit_quality")) == "quality"
    assert equilibrium.mode_of(_record("something_new", "unheard_of")) == "inspect", "never lost"
    assert not equilibrium.is_equilibrium(_record("time", "ip", domain="magnetics", subject="plasma_current"))


def test_the_registry_fills_every_mode():
    from vaft.plot import available_plots

    modes = {equilibrium.mode_of(r) for r in available_plots(status=None) if equilibrium.is_equilibrium(r)}
    assert modes == set(equilibrium.MODES)


def test_a_report_becomes_rows_and_markdown_with_its_verdicts_unchanged():
    report = {
        "status": "fail", "time": [0.32],
        "summary": {"verification": "pass", "physical_validity": "fail"},
        "verification": {"structure": {"status": "pass", "slices": [{"issues": []}]},
                         "continuity": {"status": "not_available", "reason": "needs two slices"}},
        "physical_validity": {"q_profile": {"status": "fail", "slices": [{"reason": "q < 1 | on axis"}]}},
    }
    rows = equilibrium.quality_rows(report)
    assert [(r["check"], r["status"]) for r in rows] == [
        ("structure", "pass"), ("continuity", "not_available"), ("q_profile", "fail"),
    ]
    text = equilibrium.quality_markdown("sample 39915", report)
    assert "overall **fail** at 320.0 ms" in text and "physical_validity: **fail**" in text
    assert "needs two slices" in text and "q < 1 \\| on axis" in text, "a pipe in a reason keeps the table"


# -- the workspace -------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ods():
    from vaft.omas import sample_ods

    return sample_ods(39915)


@pytest.fixture
def workspace(ods, monkeypatch):
    monkeypatch.setattr("vaft.gui.state.load_source", lambda source: ods)
    shell = Shell(initial="equilibrium")
    shell.selection.update(origin="test", sources=(Source("sample", 39915),))
    built = shell.workspaces["equilibrium"]
    yield built
    shell.close()


def _offered(app):
    return {value for options in app.plot.groups.values() for value in options.values()}


def test_each_mode_offers_only_its_equilibrium_plots(workspace):
    app = workspace.app
    for mode in equilibrium.MODES:
        workspace.mode.value = mode
        offered = _offered(app)
        assert offered, mode
        records = [app.session.capability(name) for name in offered]
        assert all(equilibrium.is_equilibrium(r) and equilibrium.mode_of(r) == mode for r in records), mode
    assert workspace.quality_card.visible and workspace.mode.value == "quality"
    workspace.mode.value = "inspect"
    assert not workspace.quality_card.visible


def test_one_time_slice_follows_the_reader_across_plots(workspace):
    app = workspace.app
    app.plot.value = "equilibrium_field_psi"
    state = app.session.state
    other = next(v for v in state.spec("time_slice").options if v != state["time_slice"])
    state.set("time_slice", other)
    assert workspace.time_slice == other
    app.plot.value = "equilibrium_profile_q"
    assert app.session.state["time_slice"] == other, "the new plot opens on the chosen slice"
    assert app.selected_time() is not None


def test_checking_the_slice_shows_the_validation_verdicts(workspace):
    workspace.mode.value = "quality"
    reports = workspace.run_checks()
    assert len(reports) == 1 and reports[0]["time_slices"] == [workspace.time_slice]
    text = workspace.quality.object
    assert "sample 39915" in text and "| verification |" in text and "overall **" in text


def test_checks_without_a_shot_are_reported(monkeypatch):
    shell = Shell(initial="equilibrium")
    try:
        assert shell.workspaces["equilibrium"].run_checks() == []
        assert shell.alert.visible and "open a shot" in shell.alert.object
    finally:
        shell.close()


# -- cold-review cases ----------------------------------------------------------------
def _move_slice(app):
    state = app.session.state
    other = next(v for v in state.spec("time_slice").options if v != state["time_slice"])
    state.set("time_slice", other)
    return other


def test_coming_back_to_a_plot_keeps_the_chosen_slice(workspace):
    app = workspace.app
    app.plot.value = "equilibrium_field_psi"
    chosen = _move_slice(app)
    app.plot.value = "equilibrium_geometry_boundary"  # no slice control
    app.plot.value = "equilibrium_field_psi"
    assert workspace.time_slice == chosen and app.session.state["time_slice"] == chosen


def test_a_shot_the_slice_does_not_fit_is_reported_without_hiding_the_others(workspace, ods):
    import copy

    short = copy.deepcopy(ods)
    while len(short["equilibrium.time_slice"]) > 1:
        del short["equilibrium.time_slice"][len(short["equilibrium.time_slice"]) - 1]
    session = workspace.app.session
    session.sources = (Source("sample", 39915), Source("sample", 41524))
    session.ods = [ods, short]
    workspace.time_slice = 4
    session._release()  # nothing on screen: the remembered slice is checked
    reports = workspace.run_checks()
    assert len(reports) == 1 and reports[0]["time_slices"] == [4]
    text = workspace.quality.object
    assert "sample 39915" in text and "sample 41524** -- not checked: slice 4 is not one of its 1" in text
    assert not workspace.check_slice.disabled


def test_the_checked_slice_is_the_one_on_screen(workspace):
    app = workspace.app
    app.plot.value = "equilibrium_field_psi"
    on_screen = _move_slice(app)
    workspace.time_slice = None if on_screen != 0 else 1  # a stale memory must not win
    (report,) = workspace.run_checks()
    assert report["time_slices"] == [on_screen]


def test_odd_report_shapes_still_render():
    import numpy as np

    text = equilibrium.quality_markdown("x", {"status": "pass", "summary": None, "time": np.array([0.3, np.nan])})
    assert "300.0 ms" in text and "nan" not in text
