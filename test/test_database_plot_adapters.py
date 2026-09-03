"""vaft.database.plot_* adapters (issue #63, H·2), without a live HSDS server.

An adapter opens only the IDS its plot declares, in the source the caller
named, lazily by default, and delegates rendering to the OMAS adapter; its
discovery answers from the shot's IDS domains without downloading.
"""

from types import ModuleType
from unittest.mock import Mock, patch

import matplotlib

matplotlib.use("Agg")

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
    assert module.folder_calls == []  # ids were given: no folder listing
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
