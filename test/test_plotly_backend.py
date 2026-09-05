"""Matplotlib and Plotly draw the same view models (issue #491).

Semantics are compared, never pixels: the same traces, names, arrays, axis
titles, validity marks, panel count and title reach both libraries.
"""

from __future__ import annotations

import ast
import contextlib
import io
import pathlib
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import omas
import pytest

import vaft
import vaft.omas
import vaft.plot
from vaft.omas.entries import normalize_entries
from vaft.plot import registry
from vaft.plot.backends import RENDER_BACKENDS, render_backends_for, renderer_for, resolve_render_backend
from vaft.plot.backend.recipes import build_model
from vaft.plot.models import (
    Field2D, Geometry3DLayer, Geometry3DLayers, GeometryLayer, GeometryLayers, Image2D, ImageSequence,
    LineSeries, Panels, PowerSpectrum, Profile1D, Series, Spectrogram, TextPanel,
)
from vaft.plot.plotly import PLOTLY_MODELS

go = pytest.importorskip("plotly.graph_objects")


@pytest.fixture(scope="module")
def sample():
    with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return vaft.omas.load(str(vaft.data.data_path("samples/39915/omas.json.gz")))


def _minimal(model_type):
    x = np.linspace(0.0, 1.0, 8)
    if model_type is LineSeries:
        return LineSeries(series=(Series(x=x, y=x**2, label="a"),), y_label="y", y_unit="m^-3")
    if model_type is Profile1D:
        return Profile1D(series=(Series(x=x, y=1.0 - x, label="a"),), y_label="y")
    if model_type is Field2D:
        r, z = np.linspace(0.1, 1.0, 5), np.linspace(-1.0, 1.0, 4)
        return Field2D(r=r, z=z, values=np.outer(z, r), value_label="psi")
    if model_type is GeometryLayers:
        return GeometryLayers(layers=(GeometryLayer(r=x, z=x, kind="polygon", label="wall"),))
    if model_type is Geometry3DLayers:
        return Geometry3DLayers(layers=(Geometry3DLayer(x=x, y=x, z=x, label="coil"),))
    if model_type is Image2D:
        return Image2D(values=np.outer(x, x), value_label="counts")
    if model_type is ImageSequence:
        return ImageSequence(frames=tuple(np.full((4, 4), i) for i in range(3)), time=np.array([0.0, 0.1, 0.2]))
    if model_type is PowerSpectrum:
        frequency = np.linspace(1.0, 100.0, 16)
        return PowerSpectrum(frequency=frequency, psd=frequency**-2.0, label="a")
    if model_type is Spectrogram:
        time, frequency = np.linspace(0.0, 1.0, 6), np.linspace(0.0, 5.0, 4)
        return Spectrogram(time=time, frequency=frequency, magnitude=np.ones((frequency.size, time.size)))
    if model_type is Panels:
        return Panels(models=(LineSeries(series=(Series(x=x, y=x, label="a"),), y_label="y"),
                              LineSeries(series=(Series(x=x, y=-x, label="b"),), y_label="y")))
    raise AssertionError(model_type)


_SPECS = list(registry.specs())


def test_the_public_backends_and_their_resolution():
    assert RENDER_BACKENDS == ("matplotlib", "plotly")
    assert resolve_render_backend(None) == "matplotlib"
    with pytest.raises(ValueError, match="backend must be one of"):
        resolve_render_backend("bokeh")
    assert set(PLOTLY_MODELS) == {LineSeries, Profile1D, Field2D, Spectrogram, TextPanel, Panels}


@pytest.mark.parametrize("spec", _SPECS, ids=[s.name for s in _SPECS])
def test_every_spec_is_drawn_by_plotly_or_refused_by_name(spec):
    model = _minimal(spec.model)
    if spec.model in PLOTLY_MODELS:
        figure = renderer_for(spec, model, "plotly")(model)
        assert isinstance(figure, go.Figure) and figure.to_dict()["data"]
        assert render_backends_for(spec) == ("matplotlib", "plotly")
    else:
        with pytest.raises(NotImplementedError, match=f"plot_{spec.stem} does not currently support backend='plotly'"):
            renderer_for(spec, model, "plotly")
        assert render_backends_for(spec) == ("matplotlib",)


def test_the_adapter_returns_a_native_plotly_figure(sample, monkeypatch):
    figure = vaft.omas.plot_plasma_current_time(sample, backend="plotly")
    assert isinstance(figure, go.Figure)
    mpl_figure, axes = vaft.omas.plot_plasma_current_time(sample)
    assert isinstance(mpl_figure, matplotlib.figure.Figure)
    plt.close(mpl_figure)
    with pytest.raises(TypeError, match="ax= is a Matplotlib axes"):
        vaft.omas.plot_plasma_current_time(sample, backend="plotly", ax=axes)
    shown = []
    monkeypatch.setattr(go.Figure, "show", lambda self, *a, **k: shown.append(self))
    vaft.omas.plot_plasma_current_time(sample, backend="plotly", show=True)
    assert len(shown) == 1
    with pytest.raises(NotImplementedError, match="plot_machine_geometry_poloidal does not currently support"):
        vaft.omas.plot_machine_geometry_poloidal(sample, backend="plotly")


def _traces(figure):
    return [t for t in figure.data if isinstance(t.meta, dict) and t.meta.get("vaft") == "trace"]


def _subplot_count(figure):
    return sum(1 for key in figure.layout if key.startswith("xaxis"))


@pytest.mark.parametrize("name, options", [
    ("plasma_current_time", {}),
    ("diamagnetic_flux_time", {}),
    ("flux_loop_time_flux", {}),
    ("flux_loop_time_flux", {"selection": "all", "validity": "show"}),
    ("equilibrium_profile_q", {"time_slice": 4}),
])
def test_traces_agree_between_the_libraries(sample, name, options):
    adapter = getattr(vaft.omas, f"plot_{name}")
    mpl_figure, axes = adapter(sample, **options)
    figure = adapter(sample, backend="plotly", **options)
    lines = [line for line in axes.lines]
    traces = _traces(figure)
    assert len(traces) == len(lines)
    for line, trace in zip(lines, traces):
        assert np.allclose(np.asarray(line.get_xdata(), dtype=float), np.asarray(trace.x, dtype=float))
        assert np.allclose(np.asarray(line.get_ydata(), dtype=float), np.asarray(trace.y, dtype=float), equal_nan=True)
        if line.get_linestyle() == "--":
            assert trace.line.dash == "dash" and trace.name.endswith("(invalid)")
    assert figure.layout.title.text == axes.get_title()
    assert figure.layout.yaxis.title.text.replace("<sup>", "^").replace("</sup>", "") == \
        axes.get_ylabel().replace("$^{", "^").replace("}$", "")
    plt.close(mpl_figure)


def test_layouts_and_overviews_keep_their_panels(sample):
    mpl_figure, axes = vaft.omas.plot_flux_loop_time_flux(sample, layout="subplots")
    figure = vaft.omas.plot_flux_loop_time_flux(sample, backend="plotly", layout="subplots")
    visible = [a for a in axes.ravel() if a.get_visible()]
    assert _subplot_count(figure) == len(visible) and len(_traces(figure)) == sum(len(a.lines) for a in visible)
    # A shared time base labels time on the lowest panel of each column only.
    titled = [key for key in figure.layout if key.startswith("xaxis") and figure.layout[key].title.text]
    assert 1 <= len(titled) <= axes.shape[1]
    plt.close(mpl_figure)
    mpl_figure, axes = vaft.omas.plot_diagnostics_overview(sample)
    figure = vaft.omas.plot_diagnostics_overview(sample, backend="plotly")
    drawn = [a for a in axes.ravel() if a.get_visible()]
    assert _subplot_count(figure) >= len(drawn) and figure.layout.title.text == mpl_figure._suptitle.get_text()
    assert len(_traces(figure)) == sum(len(a.lines) for a in drawn)
    plt.close(mpl_figure)
    mpl_figure, axes = vaft.omas.plot_equilibrium_overview(sample, time_slice=4)
    figure = vaft.omas.plot_equilibrium_overview(sample, backend="plotly", time_slice=4)
    assert any(isinstance(t, go.Contour) for t in figure.data)
    assert len(figure.layout.annotations) >= 1  # the global-quantities text panel
    plt.close(mpl_figure)


def test_the_flux_map_and_the_spectrogram_carry_over(sample):
    mpl_figure, axes = vaft.omas.plot_equilibrium_field_psi(sample, time_slice=4)
    figure = vaft.omas.plot_equilibrium_field_psi(sample, backend="plotly", time_slice=4)
    contours = [t for t in figure.data if isinstance(t, go.Contour)]
    assert len(contours) == 2  # the surfaces and the grey secondary levels
    field = [t for t in contours if t.meta["vaft"] == "field"][0]
    assert field.contours.coloring == "lines" and np.isnan(np.asarray(field.z, dtype=float)).any()
    overlays = [t for t in figure.data if isinstance(t.meta, dict) and t.meta.get("vaft") == "overlay"]
    model = build_model("equilibrium_field_psi", normalize_entries(sample), time_slice=4)
    assert len(overlays) == sum(1 for layer in model.overlays if layer.kind != "text")
    assert figure.layout.yaxis.scaleanchor == "x"
    plt.close(mpl_figure)
    figure = vaft.omas.plot_mirnov_spectrogram(sample, backend="plotly")
    heat = [t for t in figure.data if isinstance(t, go.Heatmap)]
    assert len(heat) == 1 and heat[0].reversescale
    mpl_figure, axes = vaft.omas.plot_mirnov_spectrogram(sample)
    top = figure.layout.yaxis.range[1] if figure.layout.yaxis.range else float(np.max(heat[0].y))
    assert top == pytest.approx(axes.get_ylim()[1], rel=0.05)
    plt.close(mpl_figure)


def test_validity_and_uncertainty_semantics_are_the_same():
    ods = omas.ODS(consistency_check=False)
    time = np.linspace(0.0, 1.0, 10)
    ods["magnetics.time"] = time
    # Channel 0 is valid with a flagged interval; channel 1 is flagged as a
    # whole (no per-sample verdict), so the Series rule condemns it.
    for index, (values, validity) in enumerate(((np.sin(time), 0), (np.cos(time), -1))):
        ods[f"magnetics.b_field_pol_probe.{index}.field.data"] = values
        ods[f"magnetics.b_field_pol_probe.{index}.field.time"] = time
        ods[f"magnetics.b_field_pol_probe.{index}.field.validity"] = validity
        ods[f"magnetics.b_field_pol_probe.{index}.field.data_error_upper"] = np.full(10, 0.1)
    ods["magnetics.b_field_pol_probe.0.field.validity_timed"] = np.where(time > 0.7, -1, 0)
    figure = vaft.omas.plot_b_field_probe_time_field(ods, backend="plotly", selection="all")
    traces = _traces(figure)
    assert [t.name.endswith("(invalid)") for t in traces] == [False, True]
    assert traces[1].line.dash == "dash"
    assert len(figure.layout.shapes) == 1  # the flagged interval of channel 0
    assert sum(1 for t in figure.data if t.meta.get("vaft") == "band") == 4
    masked = vaft.omas.plot_b_field_probe_time_field(ods, backend="plotly", selection="all", validity="mask")
    assert len(_traces(masked)) == 1 and not masked.layout.shapes


def test_the_legend_policy_gives_way_to_a_count(sample):
    figure = vaft.omas.plot_b_field_probe_time_field(sample, backend="plotly")
    assert figure.layout.showlegend is False
    assert any(a.text.endswith("traces") for a in figure.layout.annotations)
    figure = vaft.omas.plot_flux_loop_time_flux(sample, backend="plotly", selection=[0, 1])
    assert figure.layout.showlegend is True


def test_discovery_advertises_the_backends(sample):
    catalog = vaft.omas.available_plots(sample, detail=True)
    assert catalog.find("plasma_current_time").backends == ("matplotlib", "plotly")
    assert catalog.find("machine_geometry_poloidal").backends == ("matplotlib",)
    assert catalog.find("diagnostics_overview").backends == ("matplotlib", "plotly")
    assert "backends: matplotlib | plotly" in str(vaft.omas.available_plots(sample, query="plasma_current", detail=True))


def test_the_plotly_modules_never_import_matplotlib():
    root = pathlib.Path(vaft.plot.__file__).parent
    for path in [root / "backends.py", *sorted((root / "plotly").glob("*.py"))]:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            assert not any(name.startswith("matplotlib") for name in names), path


def test_the_cli_writes_a_plotly_figure_as_html(tmp_path, sample):
    from types import ModuleType
    from unittest.mock import Mock, patch

    from vaft.cli import plot as plot_cli

    module = ModuleType("vaft.database.lazy_ods")
    module.open_ods = Mock(return_value=sample)
    module.h5pyd = None
    target = tmp_path / "ip.html"
    with patch.dict("sys.modules", {"vaft.database.lazy_ods": module}):
        assert plot_cli.main(["plasma_current_time", "--shot", "39915", "--out", str(target),
                              "--option", "backend=plotly"]) == 0
        assert target.exists() and b"plotly" in target.read_bytes()
        assert plot_cli.main(["plasma_current_time", "--shot", "39915", "--out", str(tmp_path / "ip.png"),
                              "--option", "backend=plotly"]) == 1


def test_mathtext_labels_reach_plotly_as_text(sample):
    from vaft.plot.plotly import PLOTLY_MODELS
    from vaft.plot.plotly._style import plain_text

    assert plain_text(r"Normalized Toroidal Flux $\rho_N$") == "Normalized Toroidal Flux ρ<sub>N</sub>"
    assert plain_text(r"perturbed energy $\delta W$") == "perturbed energy δW"
    assert plain_text(r"$\psi_N$") == "ψ<sub>N</sub>" and plain_text("Pressure") == "Pressure"
    figure = vaft.omas.plot_equilibrium_field_psi(sample, backend="plotly", time_slice=4, style="normalized")
    field = [t for t in figure.data if t.meta["vaft"] == "field"][0]
    assert "$" not in field.colorbar.title.text and "ψ<sub>N</sub>" in field.colorbar.title.text
    figure = vaft.omas.plot_equilibrium_profile_q(sample, backend="plotly", time_slice=4)
    assert "$" not in figure.layout.xaxis.title.text
    # The lazy table answers every dict access, not only the overridden ones.
    assert PLOTLY_MODELS.get(LineSeries) is not None and len(list(PLOTLY_MODELS.items())) == 6
