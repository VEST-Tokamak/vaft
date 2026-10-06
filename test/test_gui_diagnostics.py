"""The Routine Diagnostics workspace (#1348): registry-driven, processed-first; never served."""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import pytest

from vaft.gui import diagnostics
from vaft.gui.state import Source

pn = pytest.importorskip("panel")

from vaft.gui.shell import Shell  # noqa: E402


def _plot(name, *paths):
    return SimpleNamespace(name=name, required_paths=paths)


# -- the registry join, without Panel ----------------------------------------------------
def test_a_diagnostic_owns_the_plots_whose_required_data_lies_under_its_path():
    plots = [
        _plot("flux_loop_time_flux", "magnetics.flux_loop[:].flux.data"),
        _plot("ip_time", "magnetics.ip[:].data"),
        _plot("ip_like", "magnetics.ipx.data"),  # a sibling, not a child
        _plot("thomson_profile", "thomson_scattering.channel[:].t_e.data"),
    ]
    assert diagnostics.diagnostic_plots({"ids_path": "magnetics.flux_loop"}, plots) == ["flux_loop_time_flux"]
    assert diagnostics.diagnostic_plots({"ids_path": "magnetics.ip"}, plots) == ["ip_time"]
    assert diagnostics.diagnostic_plots({"ids_path": "thomson_scattering"}, plots) == ["thomson_profile"]
    assert diagnostics.diagnostic_plots({"ids_path": ""}, plots) == []


def test_the_real_registry_reaches_the_main_diagnostics():
    from vaft.machine_mapping.registry import load_diagnostic_registry
    from vaft.plot import available_plots

    registry, plots = load_diagnostic_registry(), list(available_plots(status=None))
    groups = diagnostics.diagnostics_by_category(registry, plots)
    keys = {key for options in groups.values() for key in options.values()}
    assert {"magnetics.ip", "magnetics.flux_loop", "thomson_scattering", "interferometer"} <= keys
    assert "dataset_description" not in keys, "a registry entry without plots is not offered"
    assert "plasma_current_time" in diagnostics.diagnostic_plots(registry["magnetics.ip"], plots)


def test_the_card_states_the_registry_record_and_the_missing_raw_api():
    text = diagnostics.describe_diagnostic(
        "magnetics.ip", {"name": "Plasma current", "ids_path": "magnetics.ip", "family": "magnetics",
                         "category": "Magnetics", "availability": "Routine", "mapping_status": "implemented",
                         "source": {"type": "raw_daq"}}, ["plasma_current_time"],
    )
    assert "Plasma current" in text and "`magnetics.ip`" in text and "`plasma_current_time`" in text
    assert "raw_daq" in text and "not exposed by an API" in text


# -- the workspace -------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ods():
    from vaft.omas import sample_ods

    return sample_ods(39915)


@pytest.fixture
def shell(ods, monkeypatch):
    monkeypatch.setattr("vaft.gui.state.load_source", lambda source: ods)
    built = Shell(initial="diagnostics")
    yield built
    built.close()


def _offered(app):
    return {value for options in app.plot.groups.values() for value in options.values()}


def test_choosing_a_diagnostic_narrows_the_explorer_to_its_plots(shell):
    workspace = shell.workspaces["diagnostics"]
    shell.selection.update(origin="test", sources=(Source("sample", 39915),))
    workspace.diagnostic.value = "magnetics.flux_loop"
    app = workspace.app
    assert app.session.sources == (Source("sample", 39915),), "the shared selection was opened"
    assert _offered(app) and _offered(app) <= set(workspace.plots_of("magnetics.flux_loop"))
    assert app.session.plot in workspace.plots_of("magnetics.flux_loop")
    assert "Flux" in workspace.card.object or "flux" in workspace.card.object
    workspace.diagnostic.value = "magnetics.ip"
    assert app.session.plot in workspace.plots_of("magnetics.ip"), "a plot of the new diagnostic is drawn"
    assert shell.selection.value.sources == (Source("sample", 39915),)


def test_a_diagnostic_the_shot_cannot_show_draws_nothing_and_says_so(shell, monkeypatch):
    workspace = shell.workspaces["diagnostics"]
    shell.selection.update(origin="test", sources=(Source("sample", 39915),))
    app = workspace.app
    drawable = set(app.session.catalog().names())
    empty = next(
        key for options in workspace.diagnostic.groups.values() for key in options.values()
        if not set(workspace.plots_of(key)) & drawable
    )
    workspace.diagnostic.value = empty
    assert app.session.plot is None and app.download.disabled
    app.scope.value = "all"
    assert _offered(app) == set(workspace.plots_of(empty)), "All supported shows why"
