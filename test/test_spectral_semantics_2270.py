"""Issue #2270 P3 (B5): spectral parameter semantics in the plot layer.

No spectral algorithm moved -- every transform already had its owner in
``vaft.process`` -- so this pins the pass-through defects the audit found:

* the colour bar names the method's scale, not one "Magnitude" for three
  differently-scaled maps;
* an untimed signal needs ``sample_rate=`` instead of being drawn on a sample
  index labelled seconds with frequencies in cycles per sample labelled Hz;
* an explicit ``noverlap`` the window cannot hold is refused, not clamped;
* a Welch PSD refuses the spectrogram-only options it used to drop;
* the interferometer says when its trend high-pass is not applied.

What is unchanged is pinned too: the default map is exactly
``vaft.process.fluctuation.compute_spectrogram``'s.
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from omas import ODS

from vaft.omas.entries import normalize_entries
from vaft.plot.backend.recipes import SPECTROGRAM_METHODS, SPECTROGRAM_VALUE_SCALES, build_model

FS = 250e3
F0 = 12e3


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _probe(timed: bool = True, n: int = 10_000) -> ODS:
    rng = np.random.default_rng(3)
    t = 0.3 + np.arange(n) / FS
    ods = ODS(consistency_check=False)
    ods["magnetics.b_field_pol_probe.0.name"] = "probe A"
    if timed:
        ods["magnetics.b_field_pol_probe.0.voltage.time"] = t
    ods["magnetics.b_field_pol_probe.0.voltage.data"] = (
        np.sin(2 * np.pi * F0 * t) + 0.2 * rng.standard_normal(n)
    )
    return ods


def _interferometer(n: int = 20_000, fs: float = 1e6) -> ODS:
    t = 0.3 + np.arange(n) / fs
    ods = ODS(consistency_check=False)
    ods["interferometer.channel.0.name"] = "FIR"
    ods["interferometer.channel.0.n_e_line.time"] = t
    ods["interferometer.channel.0.n_e_line.data"] = 1e19 * (1 + 0.01 * np.sin(2 * np.pi * 20e3 * t))
    return ods


def _rate(t):
    """The rate the plot derives: one over the median positive step."""
    steps = np.diff(t)
    return 1.0 / float(np.median(steps[steps > 0]))


def _model(name, ods, **options):
    return build_model(name, normalize_entries(ods), **options)


# --- unchanged ---------------------------------------------------------------


def test_the_default_map_is_still_the_process_layers_stft():
    from vaft.process.fluctuation import compute_spectrogram

    ods = _probe()
    t = np.asarray(ods["magnetics.b_field_pol_probe.0.voltage.time"])
    y = np.asarray(ods["magnetics.b_field_pol_probe.0.voltage.data"])
    model = _model("mirnov_spectrogram", ods)
    expected = compute_spectrogram(t, y, sample_rate=_rate(t), nperseg=500, overlap=0.5,
                                   window="hann", detrend="constant")
    np.testing.assert_array_equal(model.magnitude, expected.magnitude)
    np.testing.assert_array_equal(model.frequency, expected.frequency)


def test_both_window_options_are_still_accepted_together():
    """Deliberate (#484): switching method keeps the resolution the caller chose."""
    for method in ("stft", "hann_fft"):
        _model("mirnov_spectrogram", _probe(), method=method, nperseg=256, window_size=256)


# --- the colour bar names the scale ----------------------------------------


@pytest.mark.parametrize("method", ["stft", "hann_fft"])
def test_the_value_label_names_the_methods_scale(method):
    model = _model("mirnov_spectrogram", _probe(), method=method)
    assert model.value_label == f"Magnitude ({SPECTROGRAM_VALUE_SCALES[method]})"
    interferometer = _model("interferometer_spectrogram", _interferometer(), method=method)
    assert interferometer.value_label.startswith("Fluctuation Magnitude (")


def test_every_method_has_a_scale():
    assert set(SPECTROGRAM_VALUE_SCALES) == set(SPECTROGRAM_METHODS)


# --- untimed signals ---------------------------------------------------------


def test_an_untimed_signal_needs_a_sample_rate():
    with pytest.raises(ValueError, match="no matching time base; pass sample_rate="):
        _model("mirnov_spectrogram", _probe(timed=False))


def test_with_a_sample_rate_the_axes_are_seconds_and_hertz():
    model = _model("mirnov_spectrogram", _probe(timed=False), sample_rate=FS)
    assert float(np.max(model.frequency)) == pytest.approx(FS / 2)
    assert float(np.max(model.time)) < 10_000 / FS  # seconds from the first sample
    peak = model.frequency[int(np.argmax(np.mean(model.magnitude, axis=1)))]
    assert peak == pytest.approx(F0, abs=FS / 500)


# --- explicit overlap --------------------------------------------------------


@pytest.mark.parametrize("noverlap", [256, 300, -1])
def test_an_overlap_the_window_cannot_hold_is_refused(noverlap):
    with pytest.raises(ValueError, match="0 <= noverlap < nperseg"):
        _model("mirnov_spectrogram", _probe(), nperseg=256, noverlap=noverlap)


def test_a_valid_overlap_is_honoured():
    from vaft.process.fluctuation import compute_spectrogram

    ods = _probe()
    t = np.asarray(ods["magnetics.b_field_pol_probe.0.voltage.time"])
    y = np.asarray(ods["magnetics.b_field_pol_probe.0.voltage.data"])
    model = _model("mirnov_spectrogram", ods, nperseg=256, noverlap=192)
    expected = compute_spectrogram(t, y, sample_rate=_rate(t), nperseg=256, overlap=0.75)
    np.testing.assert_array_equal(model.magnitude, expected.magnitude)


# --- the PSD refuses what it cannot use --------------------------------------


@pytest.mark.parametrize(
    "option",
    [{"max_frequency": 5e4}, {"frequency_range": (1e3, 5e4)}, {"method": "hann_fft"},
     {"track": True}, {"window_size": 256}, {"overlap": 0.5}],
    ids=lambda option: next(iter(option)),
)
def test_a_psd_refuses_spectrogram_only_options(option):
    key = next(iter(option))
    with pytest.raises(ValueError, match=f"takes no {key}"):
        _model("mirnov_spectrum", _probe(), **option)


def test_a_psd_still_takes_its_own_options():
    model = _model("mirnov_spectrum", _probe(), nperseg=512, noverlap=256, window="hann")
    assert np.asarray(model.frequency).size == 257


# --- the interferometer's trend filter ---------------------------------------


def test_the_interferometer_says_when_its_high_pass_is_skipped():
    with pytest.warns(UserWarning, match="high-pass is not applied"):
        _model("interferometer_spectrogram", _interferometer(), highpass_cutoff=1e9)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _model("interferometer_spectrogram", _interferometer())


# --- cold review -------------------------------------------------------------


def test_the_stft_label_names_its_density_scaling():
    """scipy's magnitude mode is density-scaled: a unit sine is not |X| ~ N/4."""
    assert "density" in SPECTROGRAM_VALUE_SCALES["stft"]
    assert SPECTROGRAM_VALUE_SCALES["hann_fft"] == "amplitude 2|X|/N"
    assert SPECTROGRAM_VALUE_SCALES["cwt"] == "CWT |W|"


def test_a_valid_overlap_above_ninety_nine_percent_is_not_capped():
    ods = _probe(n=4_000)
    capped_before = _model("mirnov_spectrogram", ods, nperseg=256, noverlap=253)
    exact = _model("mirnov_spectrogram", ods, nperseg=256, noverlap=255)
    assert exact.time.size > capped_before.time.size  # 255 is honoured, not cut to 253


def test_a_window_shortened_to_the_record_keeps_the_callers_overlap_fraction():
    from vaft.process.fluctuation import compute_spectrogram

    ods = _probe()
    model = _model("mirnov_spectrogram", ods, nperseg=4096, noverlap=3072, time_range=(0.3, 0.31))
    t = np.asarray(ods["magnetics.b_field_pol_probe.0.voltage.time"])
    y = np.asarray(ods["magnetics.b_field_pol_probe.0.voltage.data"])
    keep = (t >= 0.3) & (t <= 0.31)
    expected = compute_spectrogram(t[keep], y[keep], sample_rate=_rate(t[keep]),
                                   nperseg=int(keep.sum()), overlap=0.75)
    np.testing.assert_array_equal(model.magnitude, expected.magnitude)


def test_a_disabled_high_pass_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _model("interferometer_spectrogram", _interferometer(), highpass_cutoff=0.0)


def test_track_false_is_not_refused_by_a_psd():
    _model("mirnov_spectrum", _probe(), track=False)
