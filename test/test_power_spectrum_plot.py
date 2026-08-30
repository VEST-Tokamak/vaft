"""The PSD view model and renderer, including the no-built-in-slopes rule."""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import vaft.plot
from vaft.plot import registry
from vaft.plot.models import PowerSpectrum, ReferenceSlope, Series
from vaft.plot.renderers.spectra import render_power_spectrum


@pytest.fixture
def spectrum():
    frequency = np.logspace(2.0, 5.0, 256)
    return PowerSpectrum(frequency=frequency, psd=frequency**-2.0, label="channel 0")


class TestReferenceSlopeModel:
    def test_bare_numbers_are_coerced(self):
        model = PowerSpectrum(frequency=[1.0, 2.0], psd=[1.0, 0.5],
                              reference_slopes=[-1.5, -2.0])

        assert [item.slope for item in model.reference_slopes] == [-1.5, -2.0]
        assert all(isinstance(item, ReferenceSlope) for item in model.reference_slopes)

    def test_labels_and_anchors_are_preserved(self):
        reference = ReferenceSlope(slope=-2.0, label="steep branch", anchor=(1.0e4, 3.2e-9))

        assert reference.label == "steep branch"
        assert reference.anchor == (1.0e4, 3.2e-9)

    def test_anchor_must_be_positive_for_log_axes(self):
        with pytest.raises(ValueError, match="positive in both coordinates"):
            ReferenceSlope(slope=-2.0, anchor=(0.0, 1.0))

    def test_arbitrary_slopes_are_accepted(self):
        # No value is privileged: positive, zero and fractional slopes all work.
        for slope in (-3.7, -1.0, 0.0, 0.5, 2.0):
            assert ReferenceSlope(slope=slope).slope == pytest.approx(slope)

    def test_nothing_is_drawn_by_default(self):
        model = PowerSpectrum(frequency=[1.0, 2.0], psd=[1.0, 0.5])

        assert model.reference_slopes == ()
        assert model.marker_frequencies == ()
        assert model.fits == ()

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="equal length"):
            PowerSpectrum(frequency=[1.0, 2.0, 3.0], psd=[1.0, 0.5])

    def test_from_result_takes_units_from_the_process_result(self):
        from vaft.process.fluctuation import compute_psd

        time = np.arange(4096) / 1e5
        result = compute_psd(time, np.random.default_rng(0).standard_normal(4096),
                             units="T**2/Hz")

        model = PowerSpectrum.from_result(result)

        assert model.y_label == "PSD [T**2/Hz]"
        assert model.reference_slopes == ()  # never synthesized from a result


class TestRenderer:
    def test_draws_one_line_per_requested_guide(self, spectrum):
        model = PowerSpectrum(
            frequency=spectrum.frequency, psd=spectrum.psd,
            reference_slopes=[-1.5, ReferenceSlope(-2.5, label="mine")],
        )

        figure, axes = render_power_spectrum(model)

        assert len(axes.get_lines()) == 3  # PSD plus two guides
        labels = [text.get_text() for text in axes.get_legend().get_texts()]
        assert "mine" in labels
        assert "f^-1.5" in labels  # bare exponent when the caller gave no label
        plt.close(figure)

    def test_guide_follows_the_requested_slope(self, spectrum):
        model = PowerSpectrum(
            frequency=spectrum.frequency, psd=spectrum.psd, reference_slopes=[-3.0]
        )

        figure, axes = render_power_spectrum(model)

        guide = axes.get_lines()[1]
        x, y = (guide.get_xdata(), guide.get_ydata())
        measured = np.log(y[1] / y[0]) / np.log(x[1] / x[0])
        assert measured == pytest.approx(-3.0, abs=1e-9)
        plt.close(figure)

    def test_explicit_anchor_places_the_guide(self, spectrum):
        anchor = (1.0e3, 5.0e-3)
        model = PowerSpectrum(
            frequency=spectrum.frequency, psd=spectrum.psd,
            reference_slopes=[ReferenceSlope(-2.0, anchor=anchor)],
        )

        figure, axes = render_power_spectrum(model)

        guide = axes.get_lines()[1]
        x, y = (guide.get_xdata(), guide.get_ydata())
        expected = anchor[1] * (x / anchor[0]) ** -2.0
        np.testing.assert_allclose(y, expected)
        plt.close(figure)

    def test_unanchored_guide_sits_on_the_data(self, spectrum):
        model = PowerSpectrum(
            frequency=spectrum.frequency, psd=spectrum.psd, reference_slopes=[-2.0]
        )

        figure, axes = render_power_spectrum(model)

        guide = axes.get_lines()[1]
        # The default anchor is the measured PSD at the geometric-mean frequency,
        # so an f^-2 guide on an f^-2 spectrum overlays the data exactly.
        np.testing.assert_allclose(
            guide.get_ydata(), guide.get_xdata() ** -2.0, rtol=1e-6
        )
        plt.close(figure)

    def test_marker_frequencies_use_the_callers_label(self, spectrum):
        model = PowerSpectrum(
            frequency=spectrum.frequency, psd=spectrum.psd,
            marker_frequencies=[(1.0e4, "f_ci (mine)")],
        )

        figure, axes = render_power_spectrum(model)

        labels = [text.get_text() for text in axes.get_legend().get_texts()]
        assert "f_ci (mine)" in labels
        plt.close(figure)

    def test_fit_segments_are_drawn_as_given(self, spectrum):
        fit = Series(x=np.array([1e3, 1e4]), y=np.array([1e-6, 1e-8]), label="fit -2.00")
        model = PowerSpectrum(frequency=spectrum.frequency, psd=spectrum.psd, fits=(fit,))

        figure, axes = render_power_spectrum(model)

        assert len(axes.get_lines()) == 2
        plt.close(figure)

    def test_axes_are_logarithmic_by_default(self, spectrum):
        figure, axes = render_power_spectrum(spectrum)

        assert axes.get_xscale() == "log"
        assert axes.get_yscale() == "log"
        plt.close(figure)

    def test_rejects_a_data_object(self):
        with pytest.raises(TypeError, match="PowerSpectrum"):
            render_power_spectrum({"frequency": [1.0], "psd": [1.0]})


class TestRegistration:
    def test_the_spectrum_view_is_registered(self):
        names = {
            spec.name for spec in registry.specs() if spec.view == "spectrum"
        }
        assert names == {
            "magnetics_spectrum_mirnov",
            "soft_x_rays_spectrum",
            "interferometer_spectrum",
        }

    def test_every_spectrum_renderer_is_exported(self):
        for name in ("magnetics_spectrum_mirnov", "soft_x_rays_spectrum",
                     "interferometer_spectrum"):
            assert hasattr(vaft.plot, name)
            assert name in vaft.plot.__all__

    def test_domain_wrappers_share_the_generic_body(self, spectrum):
        # Diagnostic independence: the same model renders identically whichever
        # domain wrapper draws it.
        results = []
        for name in ("magnetics_spectrum_mirnov", "soft_x_rays_spectrum",
                     "interferometer_spectrum"):
            figure, axes = getattr(vaft.plot, name)(spectrum)
            results.append(axes.get_lines()[0].get_ydata())
            plt.close(figure)

        for other in results[1:]:
            np.testing.assert_array_equal(results[0], other)


class TestNoBuiltInSlopes:
    """VAFT ships no reference-slope constants, in the library or the notebooks."""

    SOURCES = (
        "vaft/plot/renderers/spectra.py",
        "vaft/plot/models.py",
        "vaft/process/fluctuation.py",
        "vaft/omas/_plot_recipes.py",
    )

    def test_no_slope_constants_in_the_spectral_sources(self):
        root = Path(__file__).resolve().parents[1]
        forbidden = re.compile(r"-\s*(5\s*/\s*3|8\s*/\s*3|1\.6{2,}7?|2\.6{2,}7?)\b")

        for relative in self.SOURCES:
            text = (root / relative).read_text(encoding="utf-8")
            assert not forbidden.search(text), relative
            assert "kolmogorov" not in text.lower(), relative

    def test_the_renderer_labels_a_slope_with_its_bare_value(self, spectrum):
        # Whatever the number, the fallback label is the exponent and nothing
        # more -- no regime name is attached to any particular value.
        model = PowerSpectrum(
            frequency=spectrum.frequency, psd=spectrum.psd,
            reference_slopes=[-5 / 3, -8 / 3],
        )

        figure, axes = render_power_spectrum(model)

        labels = [text.get_text() for text in axes.get_legend().get_texts()]
        # This model has no PSD label, so the legend holds only the guides.
        assert labels == ["f^-1.66667", "f^-2.66667"]
        plt.close(figure)
