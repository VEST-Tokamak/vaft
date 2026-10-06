"""The discovery-driven plot explorer (#1172): availability, labels, search; never served."""

from __future__ import annotations

from dataclasses import replace

import matplotlib

matplotlib.use("Agg")

import pytest

from vaft.gui import catalog_view
from vaft.gui.state import BrowserSession, Source

pn = pytest.importorskip("panel")

from vaft.gui import app as gui_app  # noqa: E402


@pytest.fixture(scope="module")
def ods():
    from vaft.omas import sample_ods

    return sample_ods(39915)


@pytest.fixture(scope="module")
def catalog(ods):
    from vaft.omas import available_plots

    return available_plots(ods, available_only=False)


# -- presentation, without Panel ------------------------------------------------------
def test_groups_are_by_subject_with_the_canonical_name_as_the_value(catalog):
    groups = catalog_view.group_options(catalog)
    values = [name for options in groups.values() for name in options.values()]
    assert values == [record.name for record in catalog if record.available], "available only, catalog order"
    record = catalog.find("plasma_current_time")
    options = groups[record.heading]
    assert options[catalog_view.plot_label(record)] == "plasma_current_time"
    assert catalog_view.plot_label(record) == "time", "view (and quantity), the subject is the group"
    assert catalog_view.plot_label(catalog.find("equilibrium_field_psi")) == "field / psi"


def test_all_supported_lists_unavailable_plots_marked(catalog):
    groups = catalog_view.group_options(catalog, unavailable=True)
    entries = [(label, name) for options in groups.values() for label, name in options.items()]
    assert sorted(name for _, name in entries) == sorted(record.name for record in catalog)
    missing = next(record for record in catalog if record.available is False)
    assert any(name == missing.name and label.endswith(catalog_view.UNAVAILABLE) for label, name in entries)


def test_a_search_uses_the_registry_query_and_the_visible_text(catalog):
    assert catalog_view.matching_names("") is None
    assert "plasma_current_time" in catalog_view.matching_names("ip", catalog), "an alias the registry knows"
    found = catalog_view.matching_names("psi", catalog)
    assert "equilibrium_field_psi" in found and "plasma_current_time" not in found


def test_the_card_states_why_a_plot_cannot_be_drawn(catalog):
    missing = next(record for record in catalog if record.available is False)
    text = catalog_view.describe(missing)
    assert "Unavailable here" in text and missing.reason in text and f"`{missing.name}`" in text
    shown = catalog_view.describe(catalog.find("equilibrium_field_psi"))
    assert "Unavailable" not in shown and "Controls:" in shown and "`equilibrium`" in shown
    assert catalog_view.describe(None) == ""


# -- the session ------------------------------------------------------------------------
def test_the_session_keeps_every_supported_plot_and_its_availability(ods, monkeypatch):
    monkeypatch.setattr("vaft.gui.state.load_source", lambda source: ods)
    session = BrowserSession()
    session.open(Source("sample", 39915))
    everything, available = session.full_catalog(), session.catalog()
    assert len(everything) > len(available) > 0
    assert all(record.available for record in available)
    missing = next(record for record in everything if record.available is False)
    assert session.capability(missing.name).reason == missing.reason
    assert session.capability("no_such_plot") is None
    session.close()
    assert session.capability("plasma_current_time") is None


def test_shots_are_combined_naming_the_shot_that_cannot_draw(monkeypatch, catalog):
    import vaft.database

    blocked = "plasma_current_time"

    def fake(shot, source=None, **_):
        if shot == 41524:
            return catalog.with_records([
                replace(record, available=False, reason="no magnetics") if record.name == blocked else record
                for record in catalog
            ])
        return catalog

    monkeypatch.setattr(vaft.database, "available_plots", fake)
    monkeypatch.setattr(vaft.database, "stored_ids", lambda shot, source=None: ["magnetics"])
    session = BrowserSession()
    session.open([Source("shot", 39915, "main"), Source("shot", 41524, "main")])
    record = session.capability(blocked)
    assert record.available is False
    assert record.reason == f"{Source('shot', 41524, 'main').label}: no magnetics"
    assert blocked not in session.catalog().names()
    session.close()


# -- the explorer ------------------------------------------------------------------------
@pytest.fixture
def app(ods, monkeypatch):
    monkeypatch.setattr("vaft.gui.state.load_source", lambda source: ods)
    built = gui_app.BrowserApp(plot="plasma_current_time")
    assert built.load(Source("sample", 39915))
    yield built
    built.close()


def _listed(app):
    return {value for options in app.plot.groups.values() for value in options.values()}


def test_the_selector_lists_labels_and_switching_scope_adds_unavailable_plots(app):
    available = _listed(app)
    assert available == set(app.session.catalog().names())
    app.scope.value = "all"
    assert _listed(app) == set(app.session.full_catalog().names()) > available
    assert app.plot.value == "plasma_current_time", "the plot on screen stays selected"


def test_choosing_an_unavailable_plot_explains_and_draws_nothing(app):
    app.scope.value = "all"
    missing = next(record for record in app.session.full_catalog() if record.available is False)
    app.plot.value = missing.name
    assert app.session.plot is None and app.download.disabled
    assert app.alert.visible and missing.reason in app.alert.object
    assert "Unavailable here" in app.about.object and not app.about_card.collapsed
    app.plot.value = "plasma_current_time"
    assert app.session.plot == "plasma_current_time" and not app.alert.visible
    assert "Unavailable" not in app.about.object


def test_a_search_narrows_the_list_and_keeps_what_is_drawn(app):
    app.search.value = "psi"
    listed = _listed(app)
    assert "equilibrium_field_psi" in listed and "plasma_current_time" not in listed
    assert app.session.plot == "plasma_current_time", "a search draws nothing new"
    app.search.value = "ip"
    assert app.plot.value == "plasma_current_time"
    app.search.value = "zzzz-no-such-plot"
    assert _listed(app) == set()
    app.search.value = ""
    assert _listed(app) == set(app.session.catalog().names())


def test_loading_new_sources_clears_the_search(app):
    app.search.value = "psi"
    assert app.load(Source("sample", 39915))
    assert app.search.value == "" and _listed(app) == set(app.session.catalog().names())
