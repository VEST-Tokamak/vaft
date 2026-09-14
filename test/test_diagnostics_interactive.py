"""Issue #482: the diagnostics overview with its panels and processing chosen live.

A composite has no facts of its own, so its controls are folded from its
members; ``members=`` picks panels by name and is an ordinary option, so the
static call and ``interactive=True`` draw the same figure for the same
values.  ``plot_diagnostics_time_interactive`` is a thin entry point over
that in all three namespaces.
"""

from __future__ import annotations

import copy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import vaft
from vaft.omas.entries import normalize_entries
from vaft.plot.backend.options import validate_options
from vaft.plot.backend.recipes import RECIPES, build_model, member_options_for
from vaft.plot.controls import controls_for
from vaft.plot.navigation import ControlState

NAME = "diagnostics_overview"


@pytest.fixture(scope="module")
def sample():
    return vaft.omas.load(vaft.data.sample(39915, representation="omas"))


@pytest.fixture(scope="module")
def record(sample):
    return next(r for r in vaft.omas.available_plots(sample, detail=True) if r.name == NAME)


def _series_of(model):
    return [(m.title, [(s.label, s.x.size) for s in getattr(m, "series", ())]) for m in model.models]


# ---------------------------------------------------------------------------
# the composite record
# ---------------------------------------------------------------------------

def test_the_composite_states_its_members_and_which_the_input_can_draw(record):
    declared = RECIPES[NAME].members
    assert record.members["options"] == declared
    assert record.members["available"] == record.members["default"]
    assert set(record.members["available"]) < set(declared)
    assert len(record.members["available"]) == 5, record.members["available"]
    assert record.members["labels"]["flux_loop_time_flux"] == "flux_loop / time / flux"


def test_a_member_the_input_loses_leaves_the_available_set(sample):
    reduced = copy.deepcopy(sample)
    del reduced["barometry"]
    record = next(r for r in vaft.omas.available_plots(reduced, detail=True) if r.name == NAME)
    assert "barometry_time_pressure" not in record.members["available"]
    assert "barometry_time_pressure" in record.members["options"]
    assert "barometry_time_pressure" not in [c for c in controls_for(record) if c.name == "members"][0].options


def test_the_controls_are_folded_from_the_members_not_listed_by_hand(record):
    names = [c.name for c in controls_for(record)]
    assert names == ["members", "selection", "x", "orientation", "validity"]
    by_name = {c.name: c for c in controls_for(record)}
    # Presets only: individual channel indices mean nothing across panels.
    assert "channels" not in names
    assert by_name["selection"].options == ("active", "valid", "all")
    assert by_name["x"].options == ("time", "index")
    # Different quantities per panel, so no shared unit control.
    assert "yunit" not in names and "units" not in names
    # The overview masks flagged channels by default, and the control starts there.
    assert by_name["validity"].default == "mask"
    assert by_name["members"].kind == "multi" and by_name["members"].group == "layout"
    assert by_name["members"].labels[0] == "flux_loop / time / flux"


def test_discovery_names_the_interaction_and_its_entry_point(record, sample):
    assert record.interaction == ("static", "controls")
    assert record.interaction_entry_points["controls"] == "plot_diagnostics_time_interactive()"
    printed = str(vaft.omas.available_plots(sample, query="diagnostics"))
    assert "members: 5 of 10 available" in printed
    detailed = str(vaft.omas.available_plots(sample, query="diagnostics", detail=True))
    assert "impa_time_field (unavailable)" in detailed


# ---------------------------------------------------------------------------
# members= as an option
# ---------------------------------------------------------------------------

def test_members_picks_panels_in_declared_order(sample):
    entries = normalize_entries(sample)
    model = build_model(NAME, entries, members=["mirnov_time_voltage", "flux_loop_time_flux"])
    assert [m.title for m in model.models] == [
        build_model("flux_loop_time_flux", entries, _panel_member=True).title,
        build_model("mirnov_time_voltage", entries, _panel_member=True).title,
    ]
    assert model.ncols == 2
    assert member_options_for(NAME) == RECIPES[NAME].members
    assert member_options_for("plasma_current_time") is None


def test_an_unknown_member_is_refused_by_name(sample):
    with pytest.raises(ValueError, match="not panels of this overview; its members are"):
        build_model(NAME, normalize_entries(sample), members=["plasma_current_time"])
    with pytest.raises(ValueError, match="is not an overview and takes no members="):
        validate_options("plasma_current_time", {"members": ["x"]})
    validate_options(NAME, {"members": ["flux_loop_time_flux"]})


def test_an_unavailable_member_is_left_out_not_resurrected(sample):
    entries = normalize_entries(sample)
    model = build_model(NAME, entries, members=["impa_time_field", "barometry_time_pressure"])
    assert len(model.models) == 1
    with pytest.raises(ValueError, match="none of the panels impa_time_field have data"):
        build_model(NAME, entries, members=["impa_time_field"])


def test_an_empty_choice_means_the_default(sample):
    entries = normalize_entries(sample)
    assert _series_of(build_model(NAME, entries, members=())) == _series_of(build_model(NAME, entries))


# ---------------------------------------------------------------------------
# static and interactive agree
# ---------------------------------------------------------------------------

def test_the_static_figure_for_the_controls_values_is_the_interactive_one(sample, record):
    entries = normalize_entries(sample)
    state = ControlState(controls_for(record))
    # The controls' starting values reproduce the static default exactly,
    # renderer modes included (validity="mask" is the overview's own default).
    assert state.as_style() == {"validity": "mask"}
    live = build_model(NAME, entries, **state.as_options())
    assert _series_of(live) == _series_of(build_model(NAME, entries))
    state.update(members=("flux_loop_time_flux", "spectrometer_uv_time_intensity"), selection="all")
    live = build_model(NAME, entries, **state.as_options())
    static = build_model(NAME, entries, members=["flux_loop_time_flux", "spectrometer_uv_time_intensity"], selection="all")
    assert _series_of(live) == _series_of(static)


def test_toggling_a_panel_off_redraws_with_fewer_axes(sample):
    result = vaft.omas.plot_diagnostics_overview(sample, interactive=True, interaction_backend="none")
    assert [c.name for c in result.controls][0] == "members"
    assert result.axes.shape == (5,)
    result.state.set("members", ("flux_loop_time_flux", "mirnov_time_voltage"))
    assert result.axes.shape == (2,)
    result.state.set("members", ())
    assert result.axes.shape == (5,)
    plt.close(result.figure)


def test_a_preset_reaches_every_panel(sample):
    entries = normalize_entries(sample)
    active = build_model(NAME, entries, members=["flux_loop_time_flux"], selection="active")
    everything = build_model(NAME, entries, members=["flux_loop_time_flux"], selection="all")
    assert len(everything.models[0].series) >= len(active.models[0].series)
    build_model(NAME, entries, selection="all", orientation="intuitive", x="index")


# ---------------------------------------------------------------------------
# entry points, three namespaces
# ---------------------------------------------------------------------------

def test_the_omas_entry_point_is_the_interactive_overview(sample):
    result = vaft.omas.plot_diagnostics_time_interactive(sample, backend="none")
    assert [c.name for c in result.controls] == ["members", "selection", "x", "orientation", "validity"]
    assert result.state["validity"] == "mask"
    with pytest.raises(ValueError, match="backend must be one of"):
        vaft.omas.plot_diagnostics_time_interactive(sample, backend="tk")
    plt.close(result.figure)


def test_the_imas_twins_offer_what_omas_offers(sample):
    imas = pytest.importorskip("imas")
    entry = imas.DBEntry(str(vaft.data.data_path("samples/39915/imas.nc")), "r", dd_version="3.41.0")
    result = vaft.imas.plot_diagnostics_time_interactive(entry, backend="none")
    assert [c.name for c in result.controls] == ["members", "selection", "x", "orientation", "validity"]
    assert result.axes.shape == (5,)
    plt.close(result.figure)
    explorer = vaft.imas.plot_equilibrium_interactive(entry, backend="none")
    twin = vaft.omas.plot_equilibrium_interactive(sample, backend="none")
    assert explorer.navigator.usable == twin.navigator.usable
    assert explorer.axes.shape == twin.axes.shape
    plt.close(explorer.figure)
    plt.close(twin.figure)


def test_omas_and_imas_agree_with_members(sample):
    imas = pytest.importorskip("imas")
    from vaft.imas.entries import normalize_entries as imas_entries

    entry = imas.DBEntry(str(vaft.data.data_path("samples/39915/imas.nc")), "r", dd_version="3.41.0")
    options = dict(members=["flux_loop_time_flux", "b_field_probe_time_field"], selection="all")
    left = build_model(NAME, normalize_entries(sample), **options)
    right = build_model(NAME, imas_entries(entry), **options)
    assert _series_of(left) == _series_of(right)
    for a, b in zip(left.models, right.models):
        for sa, sb in zip(a.series, b.series):
            np.testing.assert_allclose(sa.y, sb.y, equal_nan=True)


def test_the_database_twins_exist_and_explore_one_shot(monkeypatch):
    from vaft.database import plotting as db

    assert callable(vaft.database.plot_diagnostics_time_interactive)
    assert callable(vaft.database.plot_equilibrium_interactive)
    with pytest.raises(ValueError, match="one shot at a time"):
        db._load_for_interaction([1, 2], None, NAME)
    seen = {}

    def fake_load(shot, source=None, paths=None, occurrence=None):
        seen.update(shot=shot, source=source, paths=paths)
        return vaft.omas.load(vaft.data.sample(39915, representation="omas"))

    monkeypatch.setattr("vaft.database.load", fake_load)
    monkeypatch.setattr(db, "_resolve_source", lambda source: source or "resolved")
    result = db.plot_diagnostics_time_interactive(39915, backend="none")
    assert seen["shot"] == 39915 and seen["source"] == "resolved"
    assert set(seen["paths"]) >= {"dataset_description", "magnetics"}
    assert [c.name for c in result.controls][0] == "members"
    plt.close(result.figure)


def test_plotly_redraws_the_composite_too(sample):
    pytest.importorskip("plotly")
    result = vaft.omas.plot_diagnostics_overview(
        sample, interactive=True, backend="plotly", interaction_backend="none",
    )
    before = len(result.figure.layout.annotations)
    result.state.set("members", ("flux_loop_time_flux",))
    assert len(result.figure.layout.annotations) < before
