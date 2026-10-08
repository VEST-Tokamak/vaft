"""vaft.database.plot_* adapters (issue #63, H·2), without a live HSDS server.

An adapter opens only the IDS its plot declares, in the source the caller
named, lazily by default, and delegates rendering to the OMAS adapter; its
discovery answers from the shot's IDS domains without downloading.
"""

from types import ModuleType
from unittest.mock import Mock, patch

import matplotlib

matplotlib.use("Agg")

import contourpy  # noqa: F401 -- imported before any patch.dict("sys.modules"): a pybind11
# module the patch drops on exit cannot be imported again ("FillType is already registered").
import matplotlib.pyplot as plt
import numpy as np
import pytest

import vaft
import vaft.database as database
import vaft.omas
from vaft.plot.backend.recipes import required_ids
from vaft.plot.registry import canonical_names


def _fake_module(name, **attrs):
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


@pytest.fixture(scope="module")
def sample_ods():
    import contextlib, io, warnings
    with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return vaft.omas.load(str(vaft.data.data_path("samples/39915/omas.json.gz")))


@pytest.fixture(autouse=True)
def _every_ids_stored(monkeypatch):
    """A shot that stores every declared IDS, unless a test installs a fake h5pyd.

    The adapters list the shot's domains before opening (optional IDS such as
    an overlay's wall are opened only where stored); the forwarding tests mock
    the store, so the listing must not reach a server.  A test that patches
    ``lazy_ods.h5pyd`` gets the real listing over its fake.
    """
    from vaft.database import lazy_ods, plotting

    listed = plotting.stored_ids
    everything = ("dataset_description", *{root for name in canonical_names() for root in required_ids(name)})

    def stored_ids(shot, source=None):
        fake = getattr(lazy_ods, "h5pyd", None)
        return listed(shot, source) if hasattr(fake, "folder_calls") else everything

    monkeypatch.setattr(plotting, "stored_ids", stored_ids)


# ---------------------------------------------------------------------------
# forwarding: what gets opened, where
# ---------------------------------------------------------------------------

def test_a_plot_opens_only_the_ids_it_declares(sample_ods):
    open_ods = Mock(return_value=sample_ods)
    with patch.dict("sys.modules", {"vaft.database.lazy_ods": _fake_module("vaft.database.lazy_ods", open_ods=open_ods, h5pyd=None)}):
        figure, axes = database.plot_plasma_current_time(39915)
    plt.close(figure)
    args, kwargs = open_ods.call_args
    assert args[0] == 39915 and kwargs["source"] == "main"
    assert kwargs["ids"] == ["dataset_description", "magnetics"]
    assert [line.get_label() for line in axes.lines] == ["39915"]


def test_a_composite_opens_the_union_of_its_members(sample_ods):
    open_ods = Mock(return_value=sample_ods)
    with patch.dict("sys.modules", {"vaft.database.lazy_ods": _fake_module("vaft.database.lazy_ods", open_ods=open_ods, h5pyd=None)}):
        figure, _ = database.plot_magnetics_overview(39915, source="main")
    plt.close(figure)
    assert open_ods.call_args.kwargs["ids"] == ["dataset_description", *required_ids("magnetics_overview")]


def test_eager_loading_stages_the_declared_ids_and_honours_occurrence(sample_ods):
    load_ods = Mock(return_value=sample_ods)
    with patch.dict("sys.modules", {"vaft.database.ods": _fake_module("vaft.database.ods", load_ods=load_ods)}):
        figure, _ = database.plot_plasma_current_time(39915, lazy=False, occurrence=2)
    plt.close(figure)
    kwargs = load_ods.call_args.kwargs
    assert kwargs["paths"] == ["dataset_description", "magnetics"]
    # database.load maps a whole-shot occurrence onto each requested IDS.
    assert kwargs["occurrence"] == {"dataset_description": 2, "magnetics": 2} and kwargs["source"] == "main"


def test_occurrence_needs_the_eager_path_and_unknown_sources_fail_before_io():
    open_ods = Mock()
    with patch.dict("sys.modules", {"vaft.database.lazy_ods": _fake_module("vaft.database.lazy_ods", open_ods=open_ods, h5pyd=None)}):
        with pytest.raises(ValueError, match="occurrence is available with lazy=False only"):
            database.plot_plasma_current_time(39915, occurrence=1)
        with pytest.raises(Exception, match="typoo"):
            database.plot_plasma_current_time(39915, source="typoo")
    assert not open_ods.called


def test_a_shot_list_opens_each_and_labels_by_shot(sample_ods):
    open_ods = Mock(return_value=sample_ods)
    with patch.dict("sys.modules", {"vaft.database.lazy_ods": _fake_module("vaft.database.lazy_ods", open_ods=open_ods, h5pyd=None)}):
        figure, axes = database.plot_plasma_current_time([39915, 41524])
    plt.close(figure)
    assert [call.args[0] for call in open_ods.call_args_list] == [39915, 41524]
    assert [line.get_label() for line in axes.lines] == ["39915", "41524"]


# ---------------------------------------------------------------------------
# end to end over a fake h5pyd: lazy store opened, read, closed
# ---------------------------------------------------------------------------

def _fake_store_module():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("_lazy_ods_fixtures", Path(__file__).with_name("test_lazy_ods.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _shot_files(fx, source="main", shot=39915):
    datasets = {
        "ids_properties&homogeneous_time": fx.FakeDataset(1),
        "time": fx.FakeDataset([0.1, 0.2, 0.3]),
        "ip[]&AOS_SHAPE": fx.FakeDataset([1]),
        "ip[]&data": fx.FakeDataset([[1.0e5, 2.0e5, 3.0e5]]),
        "ip[]&data_SHAPE": fx.FakeDataset([[3]]),
        "ip[]&time": fx.FakeDataset([[0.1, 0.2, 0.3]]),
        "ip[]&time_SHAPE": fx.FakeDataset([[3]]),
    }
    description = {
        "data_entry&pulse": fx.FakeDataset(shot),
    }
    return {
        f"hdf5://{source}/{shot}/magnetics.h5": fx.FakeFile("magnetics", fx.FakeGroup(datasets)),
        f"hdf5://{source}/{shot}/dataset_description.h5": fx.FakeFile("dataset_description", fx.FakeGroup(description)),
        f"hdf5://{source}/{shot}/equilibrium.h5": fx.FakeFile("equilibrium", fx.FakeGroup({})),
    }


def test_end_to_end_over_a_fake_hsds_store(monkeypatch):
    fx = _fake_store_module()
    module = fx.FakeH5pyd(_shot_files(fx))
    from vaft.database import lazy_ods
    monkeypatch.setattr(lazy_ods, "h5pyd", module)
    figure, axes = database.plot_plasma_current_time(39915)
    assert [line.get_label() for line in axes.lines] == ["39915"]
    # One listing picks the declared IDS the shot stores; the store opened
    # with those ids lists nothing again.
    assert module.folder_calls == ["/main/39915/"]
    opened = {uri.rsplit("/", 1)[-1] for uri in module.opened}
    assert opened <= {"magnetics.h5", "dataset_description.h5"}
    assert all(file.closed for uri, file in module.files.items() if uri in module.opened)
    figure.savefig  # the figure outlives the store
    plt.close(figure)


def test_available_plots_answers_from_the_domain_list_without_opening(monkeypatch):
    fx = _fake_store_module()
    module = fx.FakeH5pyd(_shot_files(fx))
    from vaft.database import lazy_ods
    monkeypatch.setattr(lazy_ods, "h5pyd", module)
    catalog = database.available_plots(39915, available_only=False)
    assert module.opened == []
    assert module.folder_calls == ["/main/39915/"]
    assert catalog.find("plasma_current_time").available is True
    psi = catalog.find("thomson_scattering_time_electron_density")
    assert psi.available is False and "requires IDS thomson_scattering" in psi.reason
    assert str(catalog).startswith("Available plots — #39915 (main)")
    assert "channels:" not in str(catalog)  # leaf-level facts need a loaded ODS


def test_stored_ids_lists_the_domains_without_opening(monkeypatch):
    fx = _fake_store_module()
    module = fx.FakeH5pyd(_shot_files(fx))
    from vaft.database import lazy_ods
    monkeypatch.setattr(lazy_ods, "h5pyd", module)
    stored = database.stored_ids(39915)
    assert module.opened == [] and module.folder_calls == ["/main/39915/"]
    assert "magnetics" in stored and "master" not in stored
    assert "stored_ids" in dir(database)


def test_available_plots_takes_a_loaded_object_or_nothing(sample_ods):
    assert database.available_plots(sample_ods).names() == vaft.omas.available_plots(sample_ods).names()
    assert database.available_plots().names() == vaft.omas.available_plots().names()


# ---------------------------------------------------------------------------
# contract
# ---------------------------------------------------------------------------

def test_every_canonical_plot_has_a_database_adapter():
    for name in canonical_names():
        function = getattr(database, f"plot_{name}")
        assert name in function.__doc__ and function.__name__ == f"plot_{name}"
    assert "plotting" in dir(database) and "plot_plasma_current_time" in dir(database)


def test_ax_and_show_follow_the_contract(sample_ods):
    open_ods = Mock(return_value=sample_ods)
    figure, axes = plt.subplots()
    with patch.dict("sys.modules", {"vaft.database.lazy_ods": _fake_module("vaft.database.lazy_ods", open_ods=open_ods, h5pyd=None)}):
        returned_figure, returned_axes = database.plot_plasma_current_time(39915, ax=axes)
    assert returned_axes is axes and returned_figure is figure
    plt.close(figure)


def test_the_module_keeps_the_layering():
    import subprocess, sys
    code = "import sys, vaft.database.plotting; print('matplotlib.pyplot' in sys.modules, 'vaft.machine_mapping' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["False", "False"], out.stdout


# ---------------------------------------------------------------------------
# Independent review of the database adapters
# ---------------------------------------------------------------------------

def test_ids_level_availability_agrees_with_the_leaf_rule_where_ids_are_all_it_has(sample_ods):
    """A code-backed plot declaring IDS but no paths needs any one of them,
    at IDS level exactly as at leaf level (machine geometry overviews)."""
    from vaft.plot.backend.discovery import describe_by_ids, missing_required_ids
    from vaft.plot.backend.recipes import entry_supports
    present = set(sample_ods.keys())
    catalog = describe_by_ids(sorted(present), source="#39915 (main)", available_only=False)
    for name in ("machine_geometry_poloidal", "machine_geometry_topview"):
        assert catalog.find(name).available == entry_supports(sample_ods, name) is True, name
    assert missing_required_ids({"pf_active"}, "machine_geometry_poloidal") is None
    assert "thomson_scattering" in missing_required_ids(set(), "machine_geometry_poloidal")


def test_an_all_zero_occurrence_is_the_lazy_default(sample_ods):
    open_ods = Mock(return_value=sample_ods)
    with patch.dict("sys.modules", {"vaft.database.lazy_ods": _fake_module("vaft.database.lazy_ods", open_ods=open_ods, h5pyd=None)}):
        figure, _ = database.plot_plasma_current_time(39915, occurrence={"magnetics": 0})
        plt.close(figure)
        figure, _ = database.plot_plasma_current_time(39915, occurrence={})
        plt.close(figure)
        with pytest.raises(ValueError, match="lazy=False only"):
            database.plot_plasma_current_time(39915, occurrence={"magnetics": 1})
    assert open_ods.call_count == 2


def test_an_unknown_adapter_names_the_package():
    with pytest.raises(AttributeError, match="module 'vaft.database' has no attribute 'plot_nonexistent'"):
        database.plot_nonexistent


# ---------------------------------------------------------------------------
# dd_* / extract_* (umbrella #434) and interactive over the lazy path
# ---------------------------------------------------------------------------

def test_the_three_verbs_cover_the_same_plots_as_vaft_omas():
    canonical = set(canonical_names())
    for prefix in ("plot_", "extract_", "dd_"):
        assert {n[len(prefix):] for n in dir(database) if n.startswith(prefix)} >= canonical
    function = database.extract_plasma_current_time
    assert function.__name__ == "extract_plasma_current_time"
    assert "plot_plasma_current_time" in function.__doc__ and "rendering keyword" in function.__doc__


def test_dd_touches_nothing_and_agrees_with_vaft_omas():
    # Imported before the sys.modules patch: patch.dict drops modules first
    # imported inside it, which would give the two sides different DDPath classes.
    expected = vaft.omas.dd_flux_loop_time_flux()
    open_ods = Mock()
    with patch.dict("sys.modules", {"vaft.database.lazy_ods": _fake_module("vaft.database.lazy_ods", open_ods=open_ods, h5pyd=None)}):
        paths = database.dd_flux_loop_time_flux()
    assert not open_ods.called
    assert paths == expected
    assert paths and paths[0].canonical == "magnetics/flux_loop(:)/flux/data"


def test_extract_opens_the_declared_ids_and_returns_the_undrawn_model(sample_ods):
    from vaft.omas.entries import normalize_entries
    from vaft.plot.backend.recipes import build_model
    from vaft.plot.models import LineSeries

    open_ods = Mock(return_value=sample_ods)
    with patch.dict("sys.modules", {"vaft.database.lazy_ods": _fake_module("vaft.database.lazy_ods", open_ods=open_ods, h5pyd=None)}):
        model = database.extract_flux_loop_time_flux(39915, selection="inboard")
    args, kwargs = open_ods.call_args
    assert args[0] == 39915 and kwargs["source"] == "main"
    assert kwargs["ids"] == ["dataset_description", "magnetics"]
    assert isinstance(model, LineSeries)
    expected = build_model("flux_loop_time_flux", normalize_entries(sample_ods, label=["39915"]), selection="inboard")
    assert [s.label for s in model.series] == [s.label for s in expected.series]
    assert [s.entry for s in model.series] == ["39915"] * len(model.series)
    np.testing.assert_array_equal(model.series[0].y, expected.series[0].y)
    assert model.to_xarray().sizes["series"] == len(expected.series)


def test_extract_refuses_a_rendering_keyword_and_the_lazy_occurrence(sample_ods):
    open_ods = Mock(return_value=sample_ods)
    with patch.dict("sys.modules", {"vaft.database.lazy_ods": _fake_module("vaft.database.lazy_ods", open_ods=open_ods, h5pyd=None)}):
        with pytest.raises(TypeError, match="draws nothing; ax=.*plot_plasma_current_time"):
            database.extract_plasma_current_time(39915, ax=None)
        with pytest.raises(ValueError, match="occurrence is available with lazy=False only"):
            database.extract_plasma_current_time(39915, occurrence=1)
    assert not open_ods.called


def test_extract_end_to_end_closes_the_lazy_store(monkeypatch):
    fx = _fake_store_module()
    module = fx.FakeH5pyd(_shot_files(fx))
    from vaft.database import lazy_ods
    monkeypatch.setattr(lazy_ods, "h5pyd", module)
    model = database.extract_plasma_current_time(39915)
    assert [s.entry for s in model.series] == ["39915"]
    np.testing.assert_allclose(model.series[0].y, [100.0, 200.0, 300.0])  # kA
    assert all(file.closed for uri, file in module.files.items() if uri in module.opened)
    assert model.to_xarray().attrs["display_unit"] == "kA"


def test_interactive_loads_eagerly_over_the_lazy_path(sample_ods):
    """The controls redraw after the call returns, when a lazy store would be closed."""
    open_ods = Mock(return_value=sample_ods)
    load_ods = Mock(return_value=sample_ods)
    modules = {
        "vaft.database.lazy_ods": _fake_module("vaft.database.lazy_ods", open_ods=open_ods, h5pyd=None),
        "vaft.database.ods": _fake_module("vaft.database.ods", load_ods=load_ods),
    }
    with patch.dict("sys.modules", modules):
        result = database.plot_plasma_current_time(
            39915, interactive=True, interaction_backend="none", occurrence=1,
        )
    plt.close(result.figure)
    assert not open_ods.called
    assert load_ods.call_args.kwargs["paths"] == ["dataset_description", "magnetics"]
    # interactive=True takes the eager path's whole contract: the occurrence is honoured.
    assert load_ods.call_args.kwargs["occurrence"] == {"dataset_description": 1, "magnetics": 1}
    assert "dd" not in dir(database) and "render" not in dir(database) and "dd_plasma_current_time" in dir(database)
    result.state.set("yunit", "MA")  # rebuilds from the loaded ODS, long after the call returned
    assert result.axes.get_ylabel().endswith("[MA]")


# ---------------------------------------------------------------------------
# optional IDS the shot does not store (an overlay's wall)
# ---------------------------------------------------------------------------

def _stores(monkeypatch, *ids):
    from vaft.database import plotting
    monkeypatch.setattr(plotting, "stored_ids", lambda shot, source=None: ids)


def test_an_optional_ids_the_shot_lacks_is_not_opened(monkeypatch, sample_ods):
    """equilibrium_field_psi declares wall for an overlay; a shot without it still draws."""
    assert "wall" in required_ids("equilibrium_field_psi")
    _stores(monkeypatch, "dataset_description", "equilibrium", "magnetics", "pf_active", "pf_passive")
    open_ods = Mock(return_value=sample_ods)
    load_ods = Mock(return_value=sample_ods)
    modules = {
        "vaft.database.lazy_ods": _fake_module("vaft.database.lazy_ods", open_ods=open_ods, h5pyd=None),
        "vaft.database.ods": _fake_module("vaft.database.ods", load_ods=load_ods),
    }
    expected = ["dataset_description", "equilibrium", "pf_active", "pf_passive"]
    with patch.dict("sys.modules", modules):
        figure, _ = database.plot_equilibrium_field_psi(39915)
        plt.close(figure)
        assert open_ods.call_args.kwargs["ids"] == expected
        figure, _ = database.plot_equilibrium_field_psi(39915, lazy=False)
        plt.close(figure)
        assert load_ods.call_args.kwargs["paths"] == expected
        database.extract_equilibrium_field_psi(39915)
        assert open_ods.call_args.kwargs["ids"] == expected
    # The catalog agrees: the plot is available on this shot.
    assert database.available_plots(39915).find("equilibrium_field_psi").available is True


def test_a_required_ids_the_shot_lacks_is_refused_by_name_before_io(monkeypatch):
    _stores(monkeypatch, "dataset_description", "wall", "pf_active")
    open_ods, load_ods = Mock(), Mock()
    modules = {
        "vaft.database.lazy_ods": _fake_module("vaft.database.lazy_ods", open_ods=open_ods, h5pyd=None),
        "vaft.database.ods": _fake_module("vaft.database.ods", load_ods=load_ods),
    }
    message = r"plot 'equilibrium_field_psi' needs IDS equilibrium, which shot 39915 does not store in 'main'.*Stored IDS: dataset_description, pf_active, wall"
    with patch.dict("sys.modules", modules):
        for call in (
            lambda: database.plot_equilibrium_field_psi(39915),
            lambda: database.plot_equilibrium_field_psi(39915, lazy=False),
            lambda: database.extract_equilibrium_field_psi(39915),
        ):
            with pytest.raises(FileNotFoundError, match=message):
                call()
    assert not open_ods.called and not load_ods.called


def test_the_explorers_load_only_what_the_shot_stores(monkeypatch, sample_ods):
    from vaft.database import plotting
    from vaft.omas.interactive import equilibrium_explorer_ids

    assert "wall" in equilibrium_explorer_ids()
    _stores(monkeypatch, "dataset_description", "equilibrium", "magnetics")
    load = Mock(return_value=sample_ods)
    monkeypatch.setattr(database, "load", load)
    ods = plotting._load_for_interaction(39915, None, equilibrium_explorer_ids(), "equilibrium_overview")
    assert ods is sample_ods
    assert load.call_args.kwargs["paths"] == [i for i in equilibrium_explorer_ids() if i in ("dataset_description", "equilibrium", "magnetics")]
    _stores(monkeypatch, "dataset_description", "magnetics")
    with pytest.raises(FileNotFoundError, match="needs IDS equilibrium"):
        plotting._load_for_interaction(39915, None, equilibrium_explorer_ids(), "equilibrium_overview")


def _psi_shot_files(fx, shot=39915):
    psi = np.add.outer(np.linspace(0.0, 1.0, 4), np.linspace(0.0, 0.5, 3))  # (dim1, dim2)
    equilibrium = {
        "ids_properties&homogeneous_time": fx.FakeDataset(1),
        "time": fx.FakeDataset([0.1]),
        "time_slice[]&AOS_SHAPE": fx.FakeDataset([1]),
        "time_slice[]&time": fx.FakeDataset([0.1]),
        "time_slice[]&profiles_2d[]&AOS_SHAPE": fx.FakeDataset([[1]]),
        "time_slice[]&profiles_2d[]&grid&dim1": fx.FakeDataset([[[0.2, 0.4, 0.6, 0.8]]]),
        "time_slice[]&profiles_2d[]&grid&dim1_SHAPE": fx.FakeDataset([[[4]]]),
        "time_slice[]&profiles_2d[]&grid&dim2": fx.FakeDataset([[[-0.5, 0.0, 0.5]]]),
        "time_slice[]&profiles_2d[]&grid&dim2_SHAPE": fx.FakeDataset([[[3]]]),
        # IMAS HDF5 stores the two trailing leaf dimensions in reverse order.
        "time_slice[]&profiles_2d[]&psi": fx.FakeDataset(psi.T[None, None]),
        "time_slice[]&profiles_2d[]&psi_SHAPE": fx.FakeDataset([[[3, 4]]]),
    }
    description = {"data_entry&pulse": fx.FakeDataset(shot)}
    return {
        f"hdf5://main/{shot}/equilibrium.h5": fx.FakeFile("equilibrium", fx.FakeGroup(equilibrium)),
        f"hdf5://main/{shot}/dataset_description.h5": fx.FakeFile("dataset_description", fx.FakeGroup(description)),
    }


def test_a_shot_without_wall_draws_end_to_end_over_a_fake_hsds_store(monkeypatch):
    """The lazy path used to 404 on the declared-but-absent wall domain."""
    fx = _fake_store_module()
    module = fx.FakeH5pyd(_psi_shot_files(fx))
    from vaft.database import lazy_ods
    monkeypatch.setattr(lazy_ods, "h5pyd", module)
    model = database.extract_equilibrium_field_psi(39915)
    assert module.folder_calls == ["/main/39915/"]
    assert {uri.rsplit("/", 1)[-1] for uri in module.opened} <= {"equilibrium.h5", "dataset_description.h5"}
    assert all(file.closed for uri, file in module.files.items() if uri in module.opened)
    assert model is not None
