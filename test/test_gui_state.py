"""The GUI's toolkit-free session: sources, catalog and the plot on screen (#1086)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from vaft.gui.state import BrowserSession, Source


@pytest.fixture(scope="module")
def session():
    opened = BrowserSession()
    opened.open(Source("sample", 39915))
    yield opened
    opened.close()


def test_source_normalises_and_refuses_unknown_kinds():
    assert Source("sample", "39915").value == 39915
    assert Source("shot", 41524, "main").label == "shot 41524 (main)"
    assert Source("file", "/data/39915/eq.json").label == "39915/eq.json"
    assert Source("file", "eq.json").label == "eq.json"
    with pytest.raises(ValueError, match="kind must be one of"):
        Source("url", "http://x")


def test_the_catalog_is_the_discovery_api_s(session):
    from vaft.omas import available_plots

    catalog = session.catalog()
    assert session.catalog() is catalog, "computed once per source"
    assert catalog.names() == available_plots(session.ods).names()
    groups = session.grouped_plots()
    assert "equilibrium_field_psi" in groups["equilibrium"]
    assert [name for names in groups.values() for name in names] == list(catalog.names())


def test_select_draws_with_the_record_s_controls(session):
    drawn = session.select("equilibrium_field_psi")
    names = [control.name for control in drawn.state.controls]
    assert "time_slice" in names
    assert session.figure is drawn.figure and session.state is drawn.state

    redraws = []
    drawn.state.subscribe(redraws.append)
    slice_control = drawn.state.spec("time_slice")
    other = next(v for v in slice_control.options if v != drawn.state["time_slice"])
    drawn.state.set("time_slice", other)
    assert redraws and drawn.state["time_slice"] == other


def test_a_refused_value_leaves_the_state_as_it_was(session):
    drawn = session.select("equilibrium_field_psi")
    before = drawn.state.values
    with pytest.raises(ValueError):
        drawn.state.set("time_slice", -12345)
    assert drawn.state.values == before


def test_plotly_where_the_plot_declares_it(session):
    import plotly.graph_objects as go

    drawn = session.select("equilibrium_field_psi")
    assert session.renderer == "plotly" and isinstance(drawn.figure, go.Figure)
    before = drawn.figure
    slice_control = drawn.state.spec("time_slice")
    drawn.state.set("time_slice", next(v for v in slice_control.options if v != drawn.state["time_slice"]))
    assert session.figure is not before, "a control change rebuilds the Plotly figure"

    assert not session.supports_plotly("machine_geometry_poloidal")
    session.select("machine_geometry_poloidal")
    assert session.renderer == "matplotlib"
    session.select("equilibrium_field_psi", renderer="matplotlib")
    assert session.renderer == "matplotlib" and hasattr(session.figure, "savefig")
    with pytest.raises(ValueError, match="renderer must be one of"):
        session.select("equilibrium_field_psi", renderer="bokeh")


def test_selecting_again_releases_the_previous_figure(session):
    first = session.select("plasma_current_time", renderer="matplotlib").figure
    session.select("equilibrium_profile_q")
    assert not plt.fignum_exists(first.number)


def test_an_unknown_plot_raises_and_keeps_the_current_one(session):
    drawn = session.select("plasma_current_time", renderer="matplotlib")
    with pytest.raises(Exception):
        session.select("no_such_plot")
    assert session.interactive is drawn and plt.fignum_exists(drawn.figure.number)


def test_nothing_open_is_an_error():
    with pytest.raises(RuntimeError, match="no source"):
        BrowserSession().catalog()


def test_the_preview_size_is_validated():
    from vaft.gui.figure import DisplaySize

    with pytest.raises(ValueError, match="at least 50 px"):
        DisplaySize(width=10)
    assert DisplaySize(640, None).width == 640


def test_several_sources_open_together_and_export(session):
    both = BrowserSession()
    both.open([Source("sample", 39915), Source("sample", 41524)])
    try:
        assert both.label == "samples 39915, 41524" and len(both.ods) == 2
        with pytest.raises(RuntimeError, match="no plot"):
            both.export()
        both.select("plasma_current_time")
        assert both.export("pdf")[:4] == b"%PDF"
        with pytest.raises(ValueError, match="format must be one of"):
            both.export("gif")
    finally:
        both.close()
    with pytest.raises(ValueError, match="once"):
        BrowserSession().open([Source("sample", 39915), Source("sample", 39915)])


@pytest.fixture
def fake_database(monkeypatch):
    """vaft.database answering from the packaged sample, recording every read."""
    import omas

    import vaft.database
    from vaft.omas import available_plots, sample_ods

    sample = sample_ods(39915)
    catalog = available_plots(sample)
    reads = []

    def fake_available(shot, source=None, **_):
        return catalog

    def fake_stored(shot, source=None):
        # wall is declared by the psi map's overlay but not stored, as on 39915
        return tuple(name for name in sample.keys() if name != "wall")

    def fake_load(shot, source=None, *, paths=None, **_):
        stored = fake_stored(shot)
        absent = [name for name in paths if name not in stored]
        if absent:
            raise FileNotFoundError(f"IDS not stored for shot {shot}: {', '.join(absent)}")
        reads.append((shot, source, tuple(paths)))
        part = omas.ODS()
        for name in paths:
            part[name] = sample[name]
        return part

    monkeypatch.setattr(vaft.database, "available_plots", fake_available)
    monkeypatch.setattr(vaft.database, "stored_ids", fake_stored)
    monkeypatch.setattr(vaft.database, "load", fake_load)
    return reads


def test_database_shots_load_only_the_ids_a_plot_reads(fake_database):
    from vaft.plot.backend.recipes import required_ids

    reads = fake_database
    session = BrowserSession()
    session.open(Source("shot", 39915, "main"))
    try:
        assert session.database and reads == [], "opening lists IDS, it reads no data"
        assert session.held_ids() == [] and "magnetics" in session.plotted_ids()
        session.select("plasma_current_time")
        wanted = {"dataset_description", *required_ids("plasma_current_time")}
        assert len(reads) == 1 and set(reads[0][2]) == wanted and reads[0][:2] == (39915, "main")
        assert set(session.held_ids()) == wanted
        session.select("plasma_current_time", renderer="matplotlib")
        assert len(reads) == 1, "an IDS in memory is not read again"
        assert session.load_ids(["pf_active", "magnetics"]) == sorted({"pf_active", "magnetics"} - wanted)
        assert {"pf_active", "magnetics"} <= set(session.held_ids())
        assert session.export("png")[:4] == b"\x89PNG"
        # the psi map declares wall for an overlay; the shot does not store it
        session.select("equilibrium_field_psi")
        assert "wall" not in session.held_ids() and "wall" not in session.plotted_ids()
    finally:
        session.close()
    assert session.held_ids() == []


def test_database_shots_are_compared_and_fail_without_changing_anything(fake_database, monkeypatch):
    import vaft.database

    session = BrowserSession()
    session.open([Source("shot", 39915), Source("shot", 41524)])
    session.select("plasma_current_time")
    assert isinstance(session.ods, list) and len(session.ods) == 2
    held = session.held_ids()

    def refuse(*_args, **_kwargs):
        raise OSError("403 Forbidden")

    monkeypatch.setattr(vaft.database, "load", refuse)
    with pytest.raises(OSError, match="403"):
        session.select("pf_coil_time_current")
    assert session.plot == "plasma_current_time" and session.held_ids() == held
    with pytest.raises(ValueError, match="database shots are compared with database shots only"):
        BrowserSession().open([Source("shot", 39915), Source("sample", 39915)])
    with pytest.raises(RuntimeError, match="database shots only"):
        BrowserSession().load_ids(["magnetics"])
    session.close()
