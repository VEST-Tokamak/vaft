"""Every canonical renderer must honor the issue #62 axes/show/return contract."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from vaft.plot import registry
from vaft.plot.models import (
    Field2D,
    Geometry3DLayer,
    Geometry3DLayers,
    GeometryLayer,
    GeometryLayers,
    Image2D,
    ImageSequence,
    LineSeries,
    Panels,
    PowerSpectrum,
    Profile1D,
    Series,
    Spectrogram,
)

_SPECS = registry.specs()
_SPEC_IDS = [spec.name for spec in _SPECS]


def _minimal_model(model_type):
    x = np.linspace(0.0, 1.0, 8)
    if model_type is LineSeries:
        return LineSeries(series=(Series(x=x, y=x**2, label="a"),), y_label="y")
    if model_type is Profile1D:
        return Profile1D(series=(Series(x=x, y=1.0 - x, label="a"),), y_label="y")
    if model_type is Field2D:
        r, z = np.linspace(0.1, 1.0, 5), np.linspace(-1.0, 1.0, 4)
        return Field2D(r=r, z=z, values=np.outer(z, r), value_label="psi")
    if model_type is GeometryLayers:
        return GeometryLayers(
            layers=(GeometryLayer(r=x, z=x, kind="polygon", label="wall"),)
        )
    if model_type is Geometry3DLayers:
        return Geometry3DLayers(
            layers=(Geometry3DLayer(x=x, y=x, z=x, label="coil"),)
        )
    if model_type is Image2D:
        return Image2D(values=np.outer(x, x), value_label="counts")
    if model_type is ImageSequence:
        frames = tuple(np.full((4, 4), i) for i in range(3))
        return ImageSequence(frames=frames, time=np.array([0.0, 0.1, 0.2]))
    if model_type is PowerSpectrum:
        frequency = np.linspace(1.0, 100.0, 16)
        return PowerSpectrum(frequency=frequency, psd=frequency**-2.0, label="a")
    if model_type is Spectrogram:
        time, frequency = np.linspace(0.0, 1.0, 6), np.linspace(0.0, 5.0, 4)
        return Spectrogram(
            time=time, frequency=frequency,
            magnitude=np.ones((frequency.size, time.size)),
        )
    if model_type is Panels:
        return Panels(
            models=(
                LineSeries(series=(Series(x=x, y=x, label="a"),), y_label="y"),
                LineSeries(series=(Series(x=x, y=-x, label="b"),), y_label="y"),
            )
        )
    raise AssertionError(f"no minimal model for {model_type!r}")


def _panel_count(model):
    return len(model.models) if isinstance(model, Panels) else 1


def _call_renderer(spec, model, **kwargs):
    """Call a renderer and return its ``(figure, axes)``.

    Every renderer returns ``(Figure, Axes)`` except
    ``<domain>_animation_<quantity>`` renderers (``spec.view == "animation"``),
    which return a third element, a ``FuncAnimation`` -- the one documented
    exception to the contract (see ``vaft/plot/__init__.py``'s docstring).
    """
    result = spec.renderer(model, **kwargs)
    if spec.view == "animation":
        figure, axes, _anim = result
        return figure, axes
    return result


@pytest.mark.parametrize("spec", _SPECS, ids=_SPEC_IDS)
def test_renderer_returns_figure_and_axes(spec):
    model = _minimal_model(spec.model)
    figure, axes = _call_renderer(spec, model)
    assert isinstance(figure, Figure)
    if isinstance(axes, np.ndarray):
        assert axes.size >= _panel_count(model)
        assert all(isinstance(item, Axes) for item in axes.ravel())
    else:
        assert isinstance(axes, Axes)
        assert axes.figure is figure
    plt.close(figure)


@pytest.mark.parametrize("spec", _SPECS, ids=_SPEC_IDS)
def test_renderer_draws_into_supplied_axes_without_new_figure(spec):
    model = _minimal_model(spec.model)
    panels = _panel_count(model)
    if panels == 1:
        figure, target = plt.subplots()
    else:
        figure, target = plt.subplots(panels, 1, squeeze=False)
    before = set(plt.get_fignums())

    returned_figure, returned_axes = _call_renderer(spec, model, ax=target)

    assert returned_figure is figure
    assert set(plt.get_fignums()) == before
    first = returned_axes.ravel()[0] if isinstance(returned_axes, np.ndarray) else returned_axes
    assert first.figure is figure
    plt.close(figure)


@pytest.mark.parametrize("spec", _SPECS, ids=_SPEC_IDS)
def test_renderer_does_not_show_by_default(spec, monkeypatch):
    def explode():
        raise AssertionError(f"{spec.name} called plt.show() with show=False")

    monkeypatch.setattr(plt, "show", explode)
    figure, _ = _call_renderer(spec, _minimal_model(spec.model))
    plt.close(figure)


@pytest.mark.parametrize("spec", _SPECS, ids=_SPEC_IDS)
def test_renderer_shows_when_asked(spec, monkeypatch):
    calls = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: calls.append(1))
    figure, _ = _call_renderer(spec, _minimal_model(spec.model), show=True)
    assert calls == [1]
    plt.close(figure)


@pytest.mark.parametrize("spec", _SPECS, ids=_SPEC_IDS)
def test_renderer_rejects_the_wrong_model(spec):
    wrong = Series(x=[0.0, 1.0], y=[0.0, 1.0])
    with pytest.raises(TypeError, match="vaft.plot.models"):
        spec.renderer(wrong)


def test_renderers_reject_ods_with_an_actionable_message():
    from omas import ODS

    ods = ODS(consistency_check=False)
    ods["magnetics.ip.0.time"] = np.array([0.0, 1.0])
    ods["magnetics.ip.0.data"] = np.array([0.0, 1000.0])

    from vaft.plot import plasma_current_time

    with pytest.raises(TypeError, match="vaft.plot.models.LineSeries"):
        plasma_current_time(ods)


def test_models_reject_ods_inside_a_series():
    from omas import ODS

    with pytest.raises(TypeError, match="vaft.omas.plot_"):
        Series(x=ODS(consistency_check=False), y=[0.0, 1.0])


def test_supplying_the_wrong_number_of_axes_is_reported():
    from vaft.plot import summary_time_energy

    model = _minimal_model(Panels)
    figure, single = plt.subplots()
    with pytest.raises(ValueError, match="panels"):
        summary_time_energy(model, ax=single)
    plt.close(figure)
