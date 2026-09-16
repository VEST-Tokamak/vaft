"""Two defaults that made a correct plot unreadable.

Both were found by plotting real published data rather than a fixture, and
both are about the *unspecified* case: an explicit argument always decided
correctly, so nothing in the suite noticed.

* A geometry view went straight to `axes.legend`, bypassing the shared policy
  that replaces a legend with a count note past `LEGEND_MAX_ENTRIES`. Forty
  soft X-ray sight lines drew forty legend entries over the drawing (#764).
* A spectrogram is analysed to Nyquist. For a diagnostic sampled far above its
  physics band that put the content in the bottom few percent of the axis, so
  the default render read as an empty map (#765).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from vaft.plot.models import (
    Geometry3DLayer,
    Geometry3DLayers,
    GeometryLayer,
    GeometryLayers,
    Spectrogram,
)
from vaft.plot.renderers.geometry import (
    render_geometry_3d_layers,
    render_geometry_layers,
)
from vaft.plot.style import LEGEND_MAX_ENTRIES, apply_legend


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _layers(count: int) -> GeometryLayers:
    return GeometryLayers(
        layers=[
            GeometryLayer(
                r=np.array([0.0, 1.0]), z=np.array([float(i), float(i) + 1.0]),
                label=f"channel {i}",
            )
            for i in range(count)
        ],
        title="geometry",
    )


def _count_note(axes) -> str | None:
    for text in axes.texts:
        if "traces" in text.get_text():
            return text.get_text()
    return None


class TestGeometryLegendFollowsThePolicy:
    def test_a_many_layer_view_summarises_instead_of_covering_itself(self):
        count = LEGEND_MAX_ENTRIES + 4
        _, axes = render_geometry_layers(_layers(count))
        assert axes.get_legend() is None
        assert _count_note(axes) == f"{count} traces"

    def test_a_few_layer_view_still_gets_its_legend(self):
        _, axes = render_geometry_layers(_layers(3))
        assert axes.get_legend() is not None
        assert _count_note(axes) is None

    def test_a_single_labelled_layer_still_gets_its_legend(self):
        """A lone labelled layer among unlabelled ones still needs naming.

        This is where geometry departs from the line policy, which drops the
        legend for a lone trace because the title already names it.
        """
        _, axes = render_geometry_layers(_layers(1))
        assert axes.get_legend() is not None

    def test_legend_true_still_forces_one_past_the_threshold(self):
        count = LEGEND_MAX_ENTRIES + 4
        _, axes = render_geometry_layers(_layers(count), legend=True)
        legend = axes.get_legend()
        assert legend is not None
        assert len(legend.get_texts()) == count

    def test_legend_false_still_suppresses(self):
        _, axes = render_geometry_layers(_layers(3), legend=False)
        assert axes.get_legend() is None
        assert _count_note(axes) is None


class TestTheSamePolicyOnAThreeDimensionalView:
    """The 3-D views share the policy, and a 3-D axes places text differently.

    ``Axes3D.text`` takes ``(x, y, z, s)``, so the note the policy writes in
    axes coordinates has to go through ``text2D`` there.
    """

    @staticmethod
    def _layers(count: int) -> Geometry3DLayers:
        return Geometry3DLayers(
            layers=tuple(
                Geometry3DLayer(
                    x=np.array([0.0, 1.0]),
                    y=np.array([0.0, 1.0]),
                    z=np.array([float(i), float(i) + 1.0]),
                    label=f"coil {i}",
                )
                for i in range(count)
            )
        )

    def test_a_many_layer_view_summarises_rather_than_raising(self):
        count = LEGEND_MAX_ENTRIES + 4
        _, axes = render_geometry_3d_layers(self._layers(count))
        assert axes.get_legend() is None
        assert _count_note(axes) == f"{count} traces"

    def test_a_few_layer_view_still_gets_its_legend(self):
        _, axes = render_geometry_3d_layers(self._layers(3))
        assert axes.get_legend() is not None


class TestTheLoneEntryDistinction:
    def test_a_lone_trace_gets_no_legend_by_default(self):
        _, axes = plt.subplots()
        axes.plot([0, 1], [0, 1], label="only")
        apply_legend(axes, legend=None)
        assert axes.get_legend() is None

    def test_lone_entry_keeps_it(self):
        _, axes = plt.subplots()
        axes.plot([0, 1], [0, 1], label="only")
        apply_legend(axes, legend=None, lone_entry=True)
        assert axes.get_legend() is not None


class TestSpectrogramDefaultBand:
    """The unspecified case: a map analysed to Nyquist that is nearly all background."""

    @staticmethod
    def _stft(signal: np.ndarray, sample_rate: float = 500_000.0):
        from vaft.plot.backend.recipes import _spectrogram_result

        time = np.arange(signal.size) / sample_rate
        return _spectrogram_result(
            time, signal, method="stft", sample_rate=sample_rate, options={}
        )

    @staticmethod
    def _tone(sample_rate: float = 500_000.0, frequency: float = 2_000.0, seconds: float = 0.05):
        time = np.arange(0.0, seconds, 1.0 / sample_rate)
        return np.sin(2 * np.pi * frequency * time)

    def test_a_narrowband_channel_is_zoomed_to_its_content(self):
        from vaft.plot.backend.recipes import _default_display_band

        result = self._stft(self._tone())
        ceiling = _default_display_band(result)
        assert ceiling is not None
        assert 2_000.0 < ceiling < 20_000.0, (
            "a 2 kHz tone sampled at 500 kHz must not be drawn on a 250 kHz axis"
        )

    def test_a_realistic_noise_floor_does_not_defeat_the_zoom(self):
        """The rule reads brightness, not accumulated magnitude.

        A 1% white-noise floor spreads a large *share* of the total magnitude
        across the whole band while being invisible in the render. A rule based
        on cumulative share gives up here and shows the full axis; measuring
        each row against the map's peak does not.
        """
        from vaft.plot.backend.recipes import _default_display_band

        tone = self._tone()
        noise = np.random.default_rng(0).standard_normal(tone.size)
        result = self._stft(tone + 0.01 * noise)
        ceiling = _default_display_band(result)
        assert ceiling is not None and ceiling < 20_000.0

    def test_a_broadband_channel_keeps_its_whole_axis(self):
        """The zoom is data-driven, so genuinely broadband content is not cropped."""
        from vaft.plot.backend.recipes import _default_display_band

        noise = np.random.default_rng(1).standard_normal(self._tone().size)
        assert _default_display_band(self._stft(noise)) is None

    def test_an_empty_map_asks_for_no_zoom(self):
        from vaft.plot.backend.recipes import _default_display_band

        frequency = np.linspace(0.0, 1000.0, 10)
        time = np.linspace(0.0, 1.0, 5)
        empty = Spectrogram(
            time=time, frequency=frequency, magnitude=np.zeros((10, time.size))
        )
        assert _default_display_band(empty) is None

    def test_a_map_of_nothing_but_nans_asks_for_no_zoom(self):
        from vaft.plot.backend.recipes import _default_display_band

        frequency = np.linspace(0.0, 1000.0, 10)
        time = np.linspace(0.0, 1.0, 5)
        nans = Spectrogram(
            time=time, frequency=frequency, magnitude=np.full((10, time.size), np.nan)
        )
        assert _default_display_band(nans) is None


class TestSpectrogramDefaultReachesTheModel:
    """End to end through the recipe, which is where the default was missing."""

    @staticmethod
    def _narrowband_ods():
        from omas import ODS

        sample_rate = 500_000.0
        time = np.arange(0.0, 0.05, 1.0 / sample_rate)
        ods = ODS()
        ods["magnetics.b_field_pol_probe.0.voltage.data"] = np.sin(2 * np.pi * 2_000.0 * time)
        ods["magnetics.b_field_pol_probe.0.voltage.time"] = time
        ods["magnetics.b_field_pol_probe.0.name"] = "synthetic"
        return ods

    def _model(self, **options):
        from vaft.omas.entries import normalize_entries
        from vaft.plot.backend.recipes import build_model

        return build_model(
            "mirnov_spectrogram", normalize_entries(self._narrowband_ods()), **options
        )

    def test_the_built_model_carries_the_default_ceiling(self):
        model = self._model()
        assert model.max_frequency is not None
        assert model.max_frequency < model.frequency[-1] / 10

    def test_an_explicit_ceiling_still_decides(self):
        model = self._model(max_frequency=123_456.0)
        assert model.max_frequency == 123_456.0

    def test_an_explicit_none_still_means_the_whole_axis(self):
        """Passing it explicitly is a decision, even when the decision is None."""
        assert self._model(max_frequency=None).max_frequency is None

    def test_a_named_analysis_band_is_not_cropped_inside(self):
        """`frequency_range` is already a chosen band.

        Zooming within it would hide part of what was asked for, and -- since
        the ceiling sets `ylim` from zero -- would put axis below the band's
        lower edge where nothing was analysed.
        """
        model = self._model(frequency_range=(1e3, 5e4))
        assert model.max_frequency is None
        assert model.frequency[-1] > 4.9e4
