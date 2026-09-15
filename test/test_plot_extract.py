"""``dd_*`` / ``extract_*`` beside every ``plot_*``, and ``to_xarray()`` (umbrella #434).

The three verbs cover one set of plots on both packages; ``extract_*`` is
exactly what ``plot_*`` builds before drawing; a rendering keyword is refused
by name; and every view model round-trips through ``to_xarray()`` losslessly
and writes to netCDF.
"""

from __future__ import annotations

import os
import subprocess
import sys
import warnings

import numpy as np
import pytest

import vaft
import vaft.imas
import vaft.omas
from vaft.plot import models
from vaft.plot.backend.recipes import build_model
from vaft.plot.registry import canonical_names, get_spec

from test_imas_omas_plot_equivalence import assert_models_equal

INTERACTIVE_ENTRY_POINTS = {"diagnostics_time_interactive", "equilibrium_interactive"}


@pytest.fixture(scope="module")
def ods():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return vaft.omas.load(vaft.data.sample(39915, representation="omas"))


@pytest.fixture(scope="module")
def entry():
    imas = pytest.importorskip("imas")
    return imas.DBEntry(str(vaft.data.data_path("samples/39915/imas.nc")), "r", dd_version="3.41.0")


def _stems(module, prefix):
    return {name[len(prefix):] for name in dir(module) if name.startswith(prefix)}


# ---------------------------------------------------------------------------
# the facades
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("package", [vaft.omas, vaft.imas])
def test_the_three_verbs_cover_the_same_plots(package):
    canonical = set(canonical_names())
    assert _stems(package, "dd_") == canonical
    # vaft.omas already had extract_labels_from_odc and extract_flux_surface_contours.
    assert canonical <= _stems(package, "extract_")
    # plot_* has the same canonical set plus the interactive entry points
    # (and, on vaft.omas, the deprecated renamed aliases).
    assert canonical | INTERACTIVE_ENTRY_POINTS <= _stems(package, "plot_")
    exported = set(package.plotting.__all__)
    for name in canonical:
        assert f"dd_{name}" in exported and f"extract_{name}" in exported
    assert len(package.plotting.__all__) == len(exported), "duplicate names in __all__"


@pytest.mark.parametrize("package", [vaft.omas, vaft.imas])
def test_the_facades_are_real_named_functions(package):
    function = package.extract_plasma_current_time
    assert function.__name__ == "extract_plasma_current_time"
    assert function.__module__ == f"{package.__name__}.plotting"
    assert "plot_plasma_current_time" in function.__doc__
    assert "LineSeries" in function.__doc__
    assert package.dd_plasma_current_time.__name__ == "dd_plasma_current_time"


def test_reaching_a_facade_does_not_import_matplotlib_at_package_import():
    code = "import vaft.omas, vaft.imas, sys; print('matplotlib.pyplot' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False", out.stdout


def test_dd_takes_no_data_and_agrees_between_packages():
    for name in ("plasma_current_time", "equilibrium_field_psi", "diagnostics_overview"):
        paths = vaft.omas.dd_plasma_current_time() if name == "plasma_current_time" else getattr(vaft.omas, f"dd_{name}")()
        assert paths == getattr(vaft.imas, f"dd_{name}")()
        assert paths == vaft.plot.dd(name)
        assert all(type(p).__name__ == "DDPath" for p in paths)
    with pytest.raises(TypeError):
        vaft.omas.dd_plasma_current_time("39915")


def _available(ods):
    return [record.name for record in vaft.omas.available_plots(ods)]


def test_extract_is_exactly_what_plot_builds(ods):
    from vaft.omas.entries import normalize_entries

    checked = 0
    for name in _available(ods):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            extracted = getattr(vaft.omas, f"extract_{name}")(ods)
            built = build_model(name, normalize_entries(ods))
        assert type(extracted) is type(built), name
        assert_models_equal(extracted, built, where=name)
        checked += 1
    assert checked > 50


@pytest.mark.parametrize("name", ["plasma_current_time", "equilibrium_profile_q", "flux_loop_time_flux"])
def test_extract_agrees_between_the_packages(name, ods, entry):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from_ods = getattr(vaft.omas, f"extract_{name}")(ods)
        from_entry = getattr(vaft.imas, f"extract_{name}")(entry)
    assert_models_equal(from_ods, from_entry, where=name)
    assert from_entry.series[0].entry == "39915"


def test_extract_takes_the_extraction_options(ods):
    model = vaft.omas.extract_plasma_current_time(ods, yunit="MA")
    assert model.y_unit == "MA"
    model = vaft.omas.extract_flux_loop_time_flux(ods, selection="inboard", layout="subplots")
    assert isinstance(model, models.Panels)
    two = vaft.omas.extract_plasma_current_time([ods, ods], label=("a", "b"))
    assert [s.entry for s in two.series] == ["a", "b"]


@pytest.mark.parametrize("keyword", ["ax", "backend", "interactive", "format", "theme", "show"])
def test_a_rendering_keyword_is_refused_by_name(ods, keyword):
    with pytest.raises(TypeError, match=f"draws nothing; {keyword}=.*plot_plasma_current_time"):
        vaft.omas.extract_plasma_current_time(ods, **{keyword: "x"})


def test_an_unknown_option_is_refused_as_plot_refuses_it(ods):
    with pytest.raises(ValueError, match="does not take an option named 'selecton'"):
        vaft.omas.extract_plasma_current_time(ods, selecton="all")
    with pytest.raises(ValueError, match="does not take an option named 'color'"):
        vaft.omas.extract_plasma_current_time(ods, color="k")


def test_missing_data_is_refused_with_the_plot_message(ods):
    name = "thomson_scattering_time_electron_temperature"
    with pytest.raises(ValueError) as extracted:
        getattr(vaft.omas, f"extract_{name}")(ods)
    with pytest.raises(ValueError) as plotted:
        getattr(vaft.omas, f"plot_{name}")(ods)
    assert str(extracted.value) == str(plotted.value)
    assert "vaft.omas.available_plots(ods)" in str(extracted.value)


def test_the_generic_entry_points_dispatch_on_the_source(ods, entry):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from_ods = vaft.plot.extract("plasma_current_time", ods)
        from_entry = vaft.plot.extract("plasma_current_time", entry)
        from_list = vaft.plot.extract("plasma_current_time", [ods])
    assert_models_equal(from_ods, vaft.omas.extract_plasma_current_time(ods))
    assert_models_equal(from_entry, vaft.imas.extract_plasma_current_time(entry))
    assert_models_equal(from_list, from_ods)
    with pytest.raises(TypeError, match="vaft.omas.extract_\\*.*vaft.imas.extract_\\*"):
        vaft.plot.extract("plasma_current_time", "39915")


# ---------------------------------------------------------------------------
# to_xarray
# ---------------------------------------------------------------------------


def _restored(ds, name, k):
    return ds[name].values[k, ..., : int(ds.length.values[k])]


def _check_series_dataset(ds, model):
    assert ds.attrs["model"] == type(model).__name__
    assert ds.sizes["series"] == len(model.series)
    for k, trace in enumerate(model.series):
        assert int(ds.length[k]) == trace.y.size
        np.testing.assert_array_equal(_restored(ds, "y", k), np.asarray(trace.y, dtype=float))
        np.testing.assert_array_equal(_restored(ds, "x", k), np.asarray(trace.x, dtype=float))
        assert str(ds.label.values[k]) == trace.label
        assert str(ds.entry.values[k]) == trace.entry
        assert str(ds.channel.values[k]) == trace.channel
        assert str(ds.role.values[k]) == trace.role
        assert int(ds.index.values[k]) == (-1 if trace.index is None else trace.index)
        if trace.validity is None:
            assert np.isnan(ds.validity.values[k])
        else:
            assert int(ds.validity.values[k]) == trace.validity
        if trace.position is not None:
            assert (float(ds.position_r[k]), float(ds.position_z[k])) == trace.position
        if trace.yerr is not None:
            yerr = np.asarray(trace.yerr, dtype=float)
            expected = np.vstack([yerr, yerr]) if yerr.ndim == 1 else yerr
            np.testing.assert_array_equal(_restored(ds, "yerr", k), expected)
            assert bool(ds.has_yerr.values[k])
        if trace.valid_mask is not None:
            np.testing.assert_array_equal(_restored(ds, "valid_mask", k), trace.valid_mask)
            assert bool(ds.has_valid_mask.values[k])
    assert ds.attrs["y_unit"] == model.y_unit
    assert ds.attrs["title"] == model.title
    if model.display is not None:
        assert ds.attrs["display_unit"] == model.display.unit
        assert ds.attrs["display_scale"] == model.display.scale


def test_a_ragged_line_series_round_trips_losslessly(tmp_path):
    model = models.LineSeries(
        (
            models.Series(
                np.arange(3.0), np.array([1.0, np.nan, 3.0]), label="short", channel="ch3",
                index=3, validity=-1, valid_mask=[True, False, True], yerr=[0.1, 0.2, 0.3],
                position=(0.5, -0.1), role="measured", style={"linestyle": "--"},
            ),
            models.Series(np.arange(5.0), np.arange(5.0) * 2, label="long", entry="39915",
                          yerr=np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]], dtype=float)),
            models.Series(np.arange(4.0), np.arange(4.0), label="plain"),
        ),
        x_label="Time", y_label="Current", x_unit="s", y_unit="kA", title="ragged",
        x_limits=(0.0, 5.0), log_y=False,
    )
    ds = model.to_xarray(dd_paths=("magnetics/ip(:)/data",), plot_name="plasma_current_time")
    _check_series_dataset(ds, model)
    assert ds.sizes["sample"] == 5
    assert ds.attrs["dd_paths"] == '["magnetics/ip(:)/data"]'
    assert ds.attrs["plot_name"] == "plasma_current_time"
    assert ds.attrs["x_limits"] == "[0.0, 5.0]"
    assert '"linestyle": "--"' in str(ds.series_style.values[0])
    assert not bool(ds.has_yerr.values[2]) and not bool(ds.has_valid_mask.values[1])
    ds.to_netcdf(tmp_path / "ragged.nc")


def test_a_single_series_converts_as_one_row():
    trace = models.Series(np.arange(3.0), np.arange(3.0), label="one")
    ds = trace.to_xarray()
    assert ds.sizes["series"] == 1 and str(ds.label.values[0]) == "one"


SAMPLE_MODELS = {
    "plasma_current_time": models.LineSeries,
    "flux_loop_time_flux": models.LineSeries,
    "equilibrium_profile_q": models.Profile1D,
    "equilibrium_field_psi": models.Field2D,
    "wall_geometry_poloidal": models.GeometryLayers,
    "mirnov_spectrogram": models.Spectrogram,
    "mirnov_spectrum": models.PowerSpectrum,
    "equilibrium_overview": models.Panels,
}


@pytest.mark.parametrize("name, kind", SAMPLE_MODELS.items())
def test_every_sample_model_converts_and_writes(name, kind, ods, tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = getattr(vaft.omas, f"extract_{name}")(ods)
    assert isinstance(model, kind)
    converted = model.to_xarray()
    if kind is models.Panels:
        assert type(converted).__name__ == "DataTree"
        assert len(converted.children) == len(model.models)
        for (child_name, child), member in zip(converted.children.items(), model.models):
            assert child_name.endswith(type(member).__name__)
            assert child.to_dataset().attrs["model"] == type(member).__name__
        assert converted.attrs["ncols"] == model.ncols
        assert any(isinstance(m, models.TextPanel) for m in model.models)
    else:
        assert converted.attrs["model"] == kind.__name__
        if kind in (models.LineSeries, models.Profile1D):
            _check_series_dataset(converted, model)
        elif kind is models.Field2D:
            np.testing.assert_array_equal(converted["values"].values, model.values)
            np.testing.assert_array_equal(converted["r"].values, model.r)
            np.testing.assert_array_equal(converted["z"].values, model.z)
            assert converted.sizes["overlay_layer"] == len(model.overlays)
        elif kind is models.GeometryLayers:
            assert converted.sizes["layer"] == len(model.layers)
            for k, layer in enumerate(model.layers):
                np.testing.assert_array_equal(_restored(converted, "r", k), layer.r)
                assert str(converted.kind.values[k]) == layer.kind
        elif kind is models.Spectrogram:
            np.testing.assert_array_equal(converted["magnitude"].values, model.magnitude)
            np.testing.assert_array_equal(converted["frequency"].values, model.frequency)
        elif kind is models.PowerSpectrum:
            np.testing.assert_array_equal(converted["psd"].values, model.psd)
    converted.to_netcdf(tmp_path / f"{name}.nc")


def test_profile_reference_lines_and_coordinate_label_survive():
    model = models.Profile1D(
        (models.Series(np.linspace(0, 1, 4), np.arange(4.0), label="q"),),
        coordinate_label="rho", y_label="q", title="safety factor",
        reference_lines=(models.ReferenceLine(0.95, "q95", {"color": "k"}),),
    )
    ds = model.to_xarray()
    assert ds.attrs["coordinate_label"] == "rho"
    assert '"x": 0.95' in ds.attrs["reference_lines"] and '"label": "q95"' in ds.attrs["reference_lines"]


def test_hand_built_models_convert(tmp_path):
    overlay = models.GeometryLayer(np.array([1.0, 2.0]), np.array([3.0, 4.0]), kind="points", label="pts")
    image = models.Image2D(np.zeros((3, 4)), value_label="counts", overlays=(overlay,), vmin=0.0, vmax=1.0)
    ds = image.to_xarray()
    assert ds["values"].shape == (3, 4) and ds.sizes == {"row": 3, "column": 4, "overlay_layer": 1, "overlay_point": 2}
    np.testing.assert_array_equal(ds["overlay_r"].values[0], overlay.r)
    assert ds.attrs["vmin"] == 0.0 and ds.attrs["cmap"] == "gray"

    sequence = models.ImageSequence((np.zeros((2, 2)), np.ones((2, 2))), time=[0.1, 0.2])
    ds = sequence.to_xarray()
    assert ds["frames"].shape == (2, 2, 2) and list(ds.time.values) == [0.1, 0.2]

    layers = models.Geometry3DLayers(
        (models.Geometry3DLayer(np.arange(3.0), np.arange(3.0), np.arange(3.0), label="coil"),
         models.Geometry3DLayer(np.arange(2.0), np.arange(2.0), np.arange(2.0), label="short")),
    )
    ds = layers.to_xarray()
    assert ds.sizes["layer"] == 2 and list(ds.length.values) == [3, 2]
    np.testing.assert_array_equal(_restored(ds, "z", 1), np.arange(2.0))
    assert layers.layers[0].to_xarray().sizes["layer"] == 1

    text = models.TextPanel(("Ip = 100 kA", "q95 = 4"), title="global")
    ds = text.to_xarray()
    assert list(ds.line.values) == ["Ip = 100 kA", "q95 = 4"] and ds.attrs["title"] == "global"

    spectrum = models.PowerSpectrum(
        np.array([1.0, 10.0, 100.0]), np.array([1.0, 0.1, 0.01]),
        fits=(models.Series(np.array([1.0, 10.0]), np.array([1.0, 0.1]), label="fit"),),
        reference_slopes=(models.ReferenceSlope(-5 / 3, label="-5/3"),),
        marker_frequencies=((10.0, "f0"),),
    )
    ds = spectrum.to_xarray()
    assert ds.sizes["fit"] == 1 and str(ds.fit_label.values[0]) == "fit"
    assert '"-5/3"' in ds.attrs["reference_slopes"] and '"f0"' in ds.attrs["marker_frequencies"]
    for k, converted in enumerate((image.to_xarray(), sequence.to_xarray(), layers.to_xarray(), text.to_xarray(), ds)):
        converted.to_netcdf(tmp_path / f"hand_{k}.nc")


def test_attributes_stay_netcdf_plain():
    model = models.LineSeries((models.Series(np.arange(2.0), np.arange(2.0)),), y_limits=None, log_y=True)
    ds = model.to_xarray(extra=("a", "b"), mapping={"k": 1}, nothing=None, flag=True)
    for value in ds.attrs.values():
        assert isinstance(value, (str, int, float)), value
    assert ds.attrs["extra"] == '["a", "b"]' and ds.attrs["mapping"] == '{"k": 1}'
    assert ds.attrs["nothing"] == "" and ds.attrs["flag"] == 1 and ds.attrs["log_y"] == 1


def test_the_registry_model_matches_what_extract_returns(ods):
    for name in _available(ods)[:20]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = getattr(vaft.omas, f"extract_{name}")(ods)
        assert isinstance(model, models.ViewModel), name
        assert hasattr(model, "to_xarray"), name
        spec = get_spec(name)
        assert isinstance(model, (spec.model, models.Panels)), name
