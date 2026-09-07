"""Issue #484: how a spectrogram's time-frequency map is computed.

``method=`` names the transform -- scipy's STFT (the default since this
change), VAFT's hand-rolled Hann FFT, or an optional continuous wavelet --
each with its own parameters, all advertised per plot name and offered as a
control.  ``frequency_range=`` is the analysed band for every method.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import vaft
from vaft.omas.entries import normalize_entries
from vaft.plot.backend.recipes import (
    SPECTROGRAM_METHODS,
    SPECTROGRAM_PARAMETERS,
    build_model,
)
from vaft.plot.models import Spectrogram

NAME = "mirnov_spectrogram"


@pytest.fixture(scope="module")
def sample():
    return vaft.omas.load(vaft.data.sample(39915, representation="omas"))


@pytest.fixture(scope="module")
def entries(sample):
    return normalize_entries(sample)


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def test_the_vocabulary_and_its_parameters():
    assert SPECTROGRAM_METHODS == ("stft", "hann_fft", "cwt")
    assert set(SPECTROGRAM_PARAMETERS) == set(SPECTROGRAM_METHODS)
    assert SPECTROGRAM_PARAMETERS["stft"] == ("nperseg", "noverlap", "window", "detrend")
    assert SPECTROGRAM_PARAMETERS["hann_fft"] == ("window_size", "time_resolution")
    assert SPECTROGRAM_PARAMETERS["cwt"] == ("frequency_range", "n_frequencies")


def test_the_default_is_the_scipy_stft(entries):
    default = build_model(NAME, entries)
    stft = build_model(NAME, entries, method="stft")
    assert isinstance(default, Spectrogram)
    np.testing.assert_allclose(default.magnitude, stft.magnitude)
    np.testing.assert_allclose(default.frequency, stft.frequency)
    # The hand-rolled transform is still reachable and is a different map.
    hann = build_model(NAME, entries, method="hann_fft")
    assert hann.magnitude.shape != stft.magnitude.shape
    np.testing.assert_allclose(hann.frequency, stft.frequency)  # same window, same bins


def test_each_method_reads_its_own_parameters(entries):
    coarse = build_model(NAME, entries, method="stft", nperseg=256)
    fine = build_model(NAME, entries, method="stft", nperseg=1024)
    assert coarse.frequency.size == 129 and fine.frequency.size == 513
    overlapped = build_model(NAME, entries, method="stft", nperseg=256, noverlap=224)
    assert overlapped.time.size > coarse.time.size
    boxcar = build_model(NAME, entries, method="stft", nperseg=256, window="boxcar")
    assert not np.allclose(boxcar.magnitude, coarse.magnitude)
    stepped = build_model(NAME, entries, method="hann_fft", window_size=256, time_resolution=8)
    dense = build_model(NAME, entries, method="hann_fft", window_size=256, time_resolution=1)
    assert stepped.time.size * 7 < dense.time.size


def test_the_frequency_range_is_the_band_the_map_holds(entries):
    banded = build_model(NAME, entries, frequency_range=(1e3, 5e4))
    whole = build_model(NAME, entries)
    assert banded.frequency.min() >= 1e3 and banded.frequency.max() <= 5e4
    assert banded.frequency.size < whole.frequency.size
    assert banded.magnitude.shape == (banded.frequency.size, banded.time.size)
    for method in ("stft", "hann_fft"):
        model = build_model(NAME, entries, method=method, frequency_range=(1e3, 5e4))
        assert model.frequency.min() >= 1e3 and model.frequency.max() <= 5e4


def test_the_wavelet_needs_a_band_and_says_how_to_install_itself(entries):
    with pytest.raises(ValueError, match=r"method='cwt' analyses a named band"):
        build_model(NAME, entries, method="cwt")
    with pytest.raises(ValueError, match="0 < f0 < f1"):
        build_model(NAME, entries, method="cwt", frequency_range=(5e4, 1e3))
    pytest.importorskip
    try:
        import fcwt  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="pip install fcwt"):
            build_model(NAME, entries, method="cwt", frequency_range=(1e3, 5e4))
    else:  # pragma: no cover - only where the optional package is installed
        model = build_model(NAME, entries, method="cwt", frequency_range=(1e3, 5e4))
        assert model.frequency.min() >= 1e3 and model.frequency.max() <= 5e4


def test_an_unknown_method_is_refused_naming_the_options(sample, entries):
    with pytest.raises(ValueError, match="method= one of stft, hann_fft, cwt; got 'wavelet'"):
        build_model(NAME, entries, method="wavelet")
    with pytest.raises(ValueError, match="method must be one of stft, hann_fft, cwt"):
        vaft.omas.plot_mirnov_spectrogram(sample, method="wavelet")


def test_every_spectrogram_takes_the_method_including_the_interferometer(sample, entries):
    from vaft.plot.backend.discovery import ANALYSIS_METHODS

    for name in ("mirnov_spectrogram", "soft_x_rays_spectrogram", "interferometer_spectrogram"):
        assert ANALYSIS_METHODS[name] == SPECTROGRAM_METHODS
    assert ANALYSIS_METHODS["mirnov_spectrum"] == ("Welch PSD",)
    # The interferometer's own builder honours it too (no channel in this shot,
    # so the refusal must be about the data, never about the method).
    with pytest.raises((ValueError, KeyError)):
        build_model("interferometer_spectrogram", entries, method="hann_fft")


def test_discovery_advertises_the_methods_and_their_parameters(sample):
    record = next(r for r in vaft.omas.available_plots(sample) if r.name == NAME)
    assert record.analysis_methods == SPECTROGRAM_METHODS
    assert record.analysis["default"] == "stft"
    assert record.analysis["methods"]["hann_fft"] == ("window_size", "time_resolution")
    compact = str(vaft.omas.available_plots(sample, query="mirnov"))
    assert "methods: stft (default) | hann_fft | cwt" in compact
    detail = str(vaft.omas.available_plots(sample, query="mirnov", detail=True))
    assert "stft: nperseg, noverlap, window, detrend" in detail


def test_the_control_layer_offers_the_method(sample):
    from vaft.plot.controls import controls_for

    record = next(r for r in vaft.omas.available_plots(sample) if r.name == NAME)
    control = next(c for c in controls_for(record) if c.name == "method")
    assert control.options == SPECTROGRAM_METHODS and control.default == "stft"
    result = vaft.omas.plot_mirnov_spectrogram(sample, interactive=True, interaction_backend="none")
    result.state.set("method", "hann_fft")
    assert result.figure is not None


def test_omas_and_imas_agree_for_every_available_method(sample):
    from test_imas_omas_plot_equivalence import assert_models_equal
    from vaft.imas.access import IDSEntry

    with vaft.imas.load(vaft.data.sample(39915, representation="imas"), imas_version="3.41.0") as handle:
        entry = IDSEntry(handle)
        for options in ({}, {"method": "hann_fft", "window_size": 256}, {"frequency_range": (1e3, 5e4)}):
            expected = build_model(NAME, [("39915", sample)], **options)
            actual = build_model(NAME, [("39915", entry)], **options)
            assert_models_equal(actual, expected)


def test_both_renderers_draw_every_method(sample):
    for method in ("stft", "hann_fft"):
        figure, axes = vaft.omas.plot_mirnov_spectrogram(sample, method=method, window_size=256, nperseg=256)
        assert axes.images or axes.collections
        plt.close(figure)
    plotly = vaft.omas.plot_mirnov_spectrogram(sample, backend="plotly", frequency_range=(1e3, 5e4))
    assert len(plotly.data) == 1
