"""vaft.imas.plot_* adapters over native IMAS objects (issue #63, H·1).

One adapter per canonical plot, the renderer contract on every path, native
classes never patched, and only native reads for the path-driven plots.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import vaft
import vaft.imas
import vaft.omas
from vaft.imas.access import IDSEntry
from vaft.plot.registry import canonical_names, get_spec

imas = pytest.importorskip("imas")

SAMPLE = "samples/39915/imas.nc"


@pytest.fixture(scope="module")
def entry():
    return imas.DBEntry(str(vaft.data.data_path(SAMPLE)), "r", dd_version="3.41.0")


@pytest.fixture(scope="module")
def magnetics(entry):
    return entry.get("magnetics")


def test_every_canonical_plot_has_an_imas_adapter():
    for name in canonical_names():
        function = getattr(vaft.imas, f"plot_{name}")
        assert name in (function.__doc__ or ""), name
    assert {n for n in dir(vaft.imas) if n.startswith("plot_")} == {f"plot_{n}" for n in canonical_names()}


def test_native_classes_are_never_patched(entry):
    from imas.ids_toplevel import IDSToplevel
    before = set(dir(IDSToplevel)) | {f"DBEntry.{n}" for n in dir(imas.DBEntry)}
    vaft.imas.plot_plasma_current_time(entry)
    plt.close("all")
    after = set(dir(IDSToplevel)) | {f"DBEntry.{n}" for n in dir(imas.DBEntry)}
    assert before == after
    assert not any(n.startswith("plot_") for n in dir(IDSToplevel))


def test_every_offered_plot_renders_with_the_contract(entry):
    for row in vaft.imas.available_plots(entry):
        figure, axes = getattr(vaft.imas, f"plot_{row.name}")(entry)
        assert figure is not None and axes is not None
        plt.close(figure)


def test_a_bare_toplevel_is_labelled_by_position_and_a_dbentry_by_pulse(entry, magnetics):
    figure, axes = vaft.imas.plot_flux_loop_time_flux([magnetics, magnetics], selection=[0])
    assert [line.get_label() for line in axes.lines] == ["0", "1"]
    plt.close(figure)
    figure, axes = vaft.imas.plot_flux_loop_time_flux([magnetics, magnetics], selection=[0], label=["a", "b"])
    assert [line.get_label() for line in axes.lines] == ["a", "b"]
    plt.close(figure)
    with pytest.raises(ValueError, match="received 3 labels for 2 entries"):
        vaft.imas.plot_flux_loop_time_flux([magnetics, magnetics], label=["a", "b", "c"])
    assert [l for l, _ in vaft.imas.normalize_entries(entry)] == ["39915"]


def test_ax_and_show_follow_the_renderer_contract(magnetics):
    figure, axes = plt.subplots()
    returned_figure, returned_axes = vaft.imas.plot_plasma_current_time(magnetics, ax=axes)
    assert returned_axes is axes and returned_figure is figure
    plt.close(figure)


def test_the_wrong_ids_is_refused_by_name(magnetics):
    with pytest.raises(ValueError, match="not available in this input.*vaft.imas.available_plots"):
        vaft.imas.plot_equilibrium_time_q95(magnetics)


def test_an_ods_is_pointed_at_the_omas_namespace():
    import omas
    with pytest.raises(TypeError, match="vaft.omas.plot_"):
        vaft.imas.plot_plasma_current_time(omas.ODS())


def test_a_native_object_handed_to_a_model_is_rejected(magnetics):
    from vaft.plot.models import Series
    with pytest.raises(TypeError, match="vaft.imas.plot_"):
        Series(x=magnetics.flux_loop[0].flux.time, y=magnetics.flux_loop[0].flux.data)
    with pytest.raises(TypeError, match="vaft.imas.plot_"):
        Series(x=IDSEntry(magnetics), y=[0.0])


def test_code_backed_plots_convert_only_the_ids_they_declare(entry):
    bundle = IDSEntry(entry)
    ods = bundle.as_ods_for(("equilibrium", "wall", "dataset_description"))
    assert set(ods.keys()) <= {"equilibrium", "wall", "dataset_description"}
    assert bundle.as_ods_for(("wall", "equilibrium", "dataset_description")) is ods  # cached per name set
    figure, axes = vaft.imas.plot_equilibrium_overview(entry)
    assert axes.shape == (2, 2)
    plt.close(figure)
    detail = str(vaft.imas.available_plots(entry, query="equilibrium", view="overview", detail=True))
    assert "converted per IDS" in detail


def test_a_lazy_hsds_handle_reads_natively():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("_lazy_imas_fixtures", Path(__file__).with_name("test_lazy_imas.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from vaft.database.lazy_imas import HSDSIMASHandle
    from vaft.plot.backend import access
    handle = HSDSIMASHandle(1, ids=["equilibrium", "magnetics"], imas_version="3.41.0", h5pyd_module=module._fake_hsds())
    bundle = IDSEntry(handle)
    # Native, lazy reads through the accessor: only the leaves touched are fetched.
    assert np.allclose(access.get(bundle, "equilibrium.time"), [0.1, 0.2])
    assert np.allclose(access.get(bundle, "magnetics.ip.0.data"), [1.0, 2.0, 3.0])
    assert access.count(bundle, "equilibrium.time_slice") == 2
    assert not access.has(bundle, "magnetics.flux_loop.0.flux.data")
    assert handle.metrics["payload_selection_count"] >= 2
    # Discovery evaluates the same reads and never converts; a code-backed plot
    # cannot convert a lazy remote handle and says so.
    assert isinstance(vaft.imas.available_plots(handle).names(), tuple)
    with pytest.raises(NotImplementedError, match="lazily loaded handle"):
        bundle.as_ods_for(("equilibrium",))
