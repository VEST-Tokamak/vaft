"""Plots for the fluctuation-diagnostics tutorial (issue #1005).

The coherence plot is checked numerically against a lag built into two
synthetic channels stored on different time bases; the ridge overlay against a
chirp; the coverage figure for what it must say on its face.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from omas import ODS

import vaft.omas
import vaft.plot
from vaft.omas.entries import normalize_entries
from vaft.plot.backend.recipes import build_model, missing_required_path
from vaft.plot.models import Panels, Spectrogram
from vaft.process.fluctuation import cross_spectrum

F0 = 8_000.0
LAG = 20e-6


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _two_diagnostics() -> ODS:
    """A Mirnov probe at 250 kHz and a soft X-ray channel at 1 MHz, y lagging x by LAG."""
    rng = np.random.default_rng(4)
    ods = ODS(consistency_check=False)
    t_mirnov = 0.30 + np.arange(10_000) / 250e3
    t_sxr = 0.301 + np.arange(36_000) / 1e6
    ods["magnetics.b_field_pol_probe.0.name"] = "probe A"
    ods["magnetics.b_field_pol_probe.0.voltage.time"] = t_mirnov
    ods["magnetics.b_field_pol_probe.0.voltage.data"] = (
        np.sin(2 * np.pi * F0 * t_mirnov) + 0.3 * rng.standard_normal(t_mirnov.size)
    )
    ods["soft_x_rays.channel.0.name"] = "sxr 0"
    ods["soft_x_rays.channel.0.brightness.time"] = t_sxr
    ods["soft_x_rays.channel.0.brightness.data"] = (
        0.5 * np.sin(2 * np.pi * F0 * (t_sxr - LAG)) + 0.3 * rng.standard_normal(t_sxr.size)
    )[None, :]
    return ods


class TestCrossSpectrumPlot:
    def test_the_pure_plot_draws_coherence_its_line_and_phase(self):
        time = np.arange(20_000) / 200e3
        rng = np.random.default_rng(0)
        x = np.sin(2 * np.pi * F0 * time) + rng.standard_normal(time.size)
        y = np.sin(2 * np.pi * F0 * (time - LAG)) + rng.standard_normal(time.size)
        result = cross_spectrum(time, x, time, y, nperseg=400)
        figure, axes = vaft.plot.plot_cross_spectrum(result, x_label="A", y_label="B")
        coherence_axes, phase_axes = np.asarray(axes).ravel()
        levels = [line.get_ydata() for line in coherence_axes.lines if len(line.get_ydata()) == 2]
        assert levels and np.allclose(levels[0], result.significance_95)
        assert "Phase" in phase_axes.get_ylabel()
        assert "B relative to A" in coherence_axes.get_title()

    def test_the_model_marks_phase_only_where_coherent(self):
        time = np.arange(20_000) / 200e3
        rng = np.random.default_rng(1)
        x = np.sin(2 * np.pi * F0 * time) + rng.standard_normal(time.size)
        y = np.sin(2 * np.pi * F0 * (time - LAG)) + rng.standard_normal(time.size)
        result = cross_spectrum(time, x, time, y, nperseg=400)
        model = vaft.plot.cross_spectrum_model(result)
        assert isinstance(model, Panels) and len(model.models) == 2
        marked = model.models[1].series[1]
        coherent = result.frequency[result.coherence > result.significance_95] / 1e3
        np.testing.assert_allclose(marked.x, coherent)


class TestDiagnosticsSpectrumCoherence:
    def test_different_rates_meet_and_the_phase_is_the_lag(self):
        ods = _two_diagnostics()
        model = build_model(
            "diagnostics_spectrum_coherence", normalize_entries(ods),
            x_signal="mirnov:0", y_signal="sxr:0", nperseg=500,
        )
        coherence, phase = model.models
        assert "250 kHz grid" in coherence.title
        assert "resampled y" in coherence.title
        frequency = coherence.series[0].x * 1e3
        peak = int(np.argmin(np.abs(frequency - F0)))
        assert coherence.series[0].y[peak] > 0.9
        marked = phase.series[1]
        at_tone = marked.y[np.argmin(np.abs(marked.x * 1e3 - F0))]
        assert at_tone == pytest.approx(np.degrees(-2 * np.pi * F0 * LAG), abs=3.0)

    def test_channels_can_be_named_by_path_and_by_name(self):
        ods = _two_diagnostics()
        by_path = build_model(
            "diagnostics_spectrum_coherence", normalize_entries(ods),
            x_signal="magnetics.b_field_pol_probe.0", y_signal="soft_x_rays.channel.0", nperseg=500,
        )
        by_name = build_model(
            "diagnostics_spectrum_coherence", normalize_entries(ods),
            x_signal="probe A", y_signal=("soft_x_rays", "sxr 0"), nperseg=500,
        )
        np.testing.assert_allclose(by_path.models[0].series[0].y, by_name.models[0].series[0].y)

    def test_the_default_compares_the_first_two_channels(self):
        ods = _two_diagnostics()
        model = build_model("diagnostics_spectrum_coherence", normalize_entries(ods), nperseg=500)
        assert "sxr 0 relative to x = probe A" in model.models[0].title

    def test_an_unknown_channel_is_refused_by_name(self):
        ods = _two_diagnostics()
        with pytest.raises(ValueError, match="no fluctuation channel is named"):
            build_model("diagnostics_spectrum_coherence", normalize_entries(ods), x_signal="nope")
        with pytest.raises(ValueError, match="unknown diagnostic"):
            build_model("diagnostics_spectrum_coherence", normalize_entries(ods), x_signal="ece:0")

    def test_one_channel_is_not_enough(self):
        ods = ODS(consistency_check=False)
        ods["magnetics.b_field_pol_probe.0.voltage.time"] = np.arange(100) / 1e5
        ods["magnetics.b_field_pol_probe.0.voltage.data"] = np.zeros(100)
        assert "two fluctuation channels" in missing_required_path(ods, "diagnostics_spectrum_coherence")

    def test_the_packaged_sample_renders_through_the_adapter(self):
        ods = vaft.omas.sample_ods(39915)
        figure, axes = vaft.omas.plot_diagnostics_spectrum_coherence(
            ods, time_range=(0.305, 0.33), nperseg=500, max_frequency=60e3
        )
        assert len(np.asarray(axes).ravel()) == 2


class TestSpectrogramRidge:
    def _chirp_ods(self):
        fs = 250e3
        time = 0.3 + np.arange(10_000) / fs
        rate = (7e3 - 10e3) / (time[-1] - time[0])
        phase = 2 * np.pi * (10e3 * (time - time[0]) + 0.5 * rate * (time - time[0]) ** 2)
        ods = ODS(consistency_check=False)
        ods["magnetics.b_field_pol_probe.0.name"] = "probe"
        ods["magnetics.b_field_pol_probe.0.voltage.time"] = time
        ods["magnetics.b_field_pol_probe.0.voltage.data"] = np.sin(phase)
        return ods, time, rate

    def test_track_draws_the_chirp_within_a_bin(self):
        ods, time, rate = self._chirp_ods()
        model = build_model(
            "mirnov_spectrogram", normalize_entries(ods), nperseg=500, track=(3e3, 20e3), max_jump=1.5e3
        )
        assert isinstance(model, Spectrogram) and model.ridge_time is not None
        truth = 10e3 + rate * (model.ridge_time - time[0])
        assert np.nanmax(np.abs(model.ridge_frequency - truth)) <= 250e3 / 500
        figure, axes = vaft.plot.mirnov_spectrogram(model)
        assert any(line.get_label().startswith("ridge 3-20 kHz") for line in axes.lines)

    def test_no_track_draws_no_ridge(self):
        ods, _, _ = self._chirp_ods()
        model = build_model("mirnov_spectrogram", normalize_entries(ods), nperseg=500)
        assert model.ridge_time is None
        figure, axes = vaft.plot.mirnov_spectrogram(model)
        assert not axes.lines

    def test_track_true_searches_the_drawn_band(self):
        ods, _, _ = self._chirp_ods()
        model = build_model(
            "mirnov_spectrogram", normalize_entries(ods), nperseg=500, track=True,
            frequency_range=(5e3, 15e3),
        )
        finite = model.ridge_frequency[np.isfinite(model.ridge_frequency)]
        assert finite.size and finite.min() >= 5e3 and finite.max() <= 15e3


class TestFrequencyCoverage:
    def test_it_draws_every_row_and_says_what_the_bands_are_not(self):
        diagnostics = {"Mirnov": (250e3, 125e3), "SXR": 488e3}
        figure, axes = vaft.plot.plot_fluctuation_frequency_coverage(diagnostics)
        labels = [tick.get_text() for tick in axes.get_yticklabels()]
        assert "Mirnov" in labels and "SXR" in labels
        assert "tearing mode / NTM" in labels
        assert "not an identification rule" in axes.get_xlabel().replace("\n", " ")
        texts = [text.get_text() for text in axes.texts]
        assert any("125 kHz" in text for text in texts) and any("488 kHz" in text for text in texts)

    def test_the_bandwidths_helper_feeds_it(self):
        ods = _two_diagnostics()
        bandwidths = vaft.omas.fluctuation_bandwidths(ods)
        assert bandwidths["Mirnov / magnetic probes"] == pytest.approx((250e3, 125e3))
        assert bandwidths["Soft X-ray"] == pytest.approx((1e6, 5e5))
        figure, axes = vaft.plot.plot_fluctuation_frequency_coverage(bandwidths, phenomena={})
        assert len(axes.get_yticklabels()) == 2

    def test_nothing_to_draw_is_refused(self):
        with pytest.raises(ValueError, match="nothing to draw"):
            vaft.plot.plot_fluctuation_frequency_coverage({}, phenomena={})
