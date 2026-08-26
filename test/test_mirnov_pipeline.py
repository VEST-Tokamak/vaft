import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vaft.machine_mapping.magnetics import vfit_magnetics_for_shot
from vaft.machine_mapping.utils import get_path, path_exists, set_path
from vaft.plot.mirnov import mirnov_signal, mirnov_spectrogram, toroidal_mode_spectrum, toroidal_phase_mode_fit
from vaft.process.magnetics import mirnov_spectrogram as compute_mirnov_spectrogram
from vaft.process.magnetics import toroidal_mode_analysis, toroidal_phase_fit_at_time


TSTART = 0.26
TEND = 0.34
NATIVE_MIRNOV_DT = 4e-6  # 250 kHz raw Mirnov digitiser
# Raw voltages are cropped to the analysis window but never resampled, so they
# keep the native spacing rather than the 40 us processed grid.
NATIVE_SAMPLES = round((TEND - TSTART) / NATIVE_MIRNOV_DT)


def test_magnetics_mapping_preserves_raw_mirnov_voltage():
    payload = {}
    vfit_magnetics_for_shot(
        payload,
        shot=44740,
        tstart=TSTART,
        tend=TEND,
        dt=4e-5,
        raw_source="vaft/data/legacy/shot_{shot}.json.gz",
    )

    mapped_time = np.asarray(get_path(payload, "magnetics.time"))
    field_data = np.asarray(get_path(payload, "magnetics.b_field_pol_probe.0.field.data"))
    voltage_time = np.asarray(get_path(payload, "magnetics.b_field_pol_probe.0.voltage.time"))
    voltage_data = np.asarray(get_path(payload, "magnetics.b_field_pol_probe.0.voltage.data"))

    assert len(get_path(payload, "magnetics.b_field_pol_probe")) == 68
    assert field_data.size == mapped_time.size
    assert voltage_time.size == NATIVE_SAMPLES
    assert voltage_data.size == voltage_time.size
    assert voltage_time.size != mapped_time.size

    assert get_path(payload, "magnetics.b_field_pol_probe.67.type.index") == 2
    assert np.isclose(get_path(payload, "magnetics.b_field_pol_probe.67.toroidal_angle"), 4 * np.pi / 3)
    assert np.asarray(get_path(payload, "magnetics.b_field_pol_probe.67.voltage.data")).size == NATIVE_SAMPLES
    assert get_path(payload, "magnetics.b_field_pol_probe.67.voltage.validity") == 0
    assert np.asarray(get_path(payload, "magnetics.b_field_pol_probe.64.voltage.data")).size == 0
    assert get_path(payload, "magnetics.b_field_pol_probe.64.voltage.validity") == -2
    # Phase-reference channels are diagnostics only and never become EFIT
    # constraints, so their processed field is present but explicitly empty
    # rather than absent -- an empty array keeps strict IMAS validation valid.
    assert path_exists(payload, "magnetics.b_field_pol_probe.67.field.data")
    assert np.asarray(get_path(payload, "magnetics.b_field_pol_probe.67.field.data")).size == 0
    assert np.asarray(get_path(payload, "magnetics.b_field_pol_probe.67.field.time")).size == 0


def test_mirnov_spectrogram_recovers_peak_frequency():
    sample_rate = 100_000.0
    frequency = 12_000.0
    time = np.arange(4096, dtype=float) / sample_rate
    data = np.sin(2.0 * np.pi * frequency * time)

    result = compute_mirnov_spectrogram(
        time,
        data,
        sample_rate=sample_rate,
        window_size=500,
        time_resolution=32,
    )

    peak_frequency = result.frequency[np.argmax(np.max(result.magnitude, axis=1))]
    assert abs(peak_frequency - frequency) <= sample_rate / 500


def test_toroidal_mode_analysis_recovers_phase_mode():
    sample_rate = 10_000.0
    time = np.arange(4096, dtype=float) / sample_rate
    phase_geometry = np.pi / 6
    expected_n = 2
    signal_a = np.sin(2.0 * np.pi * 1_000.0 * time)
    signal_b = np.sin(2.0 * np.pi * 1_000.0 * time + expected_n * phase_geometry)

    result = toroidal_mode_analysis(
        signal_a,
        signal_b,
        sample_rate=sample_rate,
        phase_geometry=phase_geometry,
        peak_threshold=0.05,
        nperseg=1024,
    )

    assert expected_n in set(result.n.astype(int))


def test_toroidal_phase_fit_at_time_recovers_mode_line():
    sample_rate = 100_000.0
    time = np.arange(4096, dtype=float) / sample_rate
    angles = np.deg2rad([0.0, 120.0, 180.0, 240.0])
    frequency = 10_000.0
    expected_n = 2
    phases = 0.4 - expected_n * angles
    signals = np.vstack([np.sin(2.0 * np.pi * frequency * time + phase) for phase in phases])

    result = toroidal_phase_fit_at_time(
        time,
        signals,
        angles,
        center_time=0.020,
        sample_rate=sample_rate,
        window_size=512,
        frequencies=[frequency],
        candidate_n=range(0, 5),
    )

    assert len(result.modes) == 1
    assert result.modes[0].n == expected_n
    assert result.modes[0].rms_error < 0.05


def _tiny_mirnov_ods():
    sample_rate = 100_000.0
    time = np.arange(2048, dtype=float) / sample_rate
    signal_a = np.sin(2.0 * np.pi * 8_000.0 * time)
    signal_b = np.sin(2.0 * np.pi * 8_000.0 * time + np.pi / 3)
    ods = {}
    for index, data in enumerate((signal_a, signal_b)):
        set_path(ods, f"magnetics.b_field_pol_probe.{index}.name", f"BP{index}")
        set_path(ods, f"magnetics.b_field_pol_probe.{index}.voltage.time", time)
        set_path(ods, f"magnetics.b_field_pol_probe.{index}.voltage.data", data)
    return ods


def _tiny_phase_ods():
    sample_rate = 100_000.0
    time = np.arange(4096, dtype=float) / sample_rate
    angles = np.deg2rad([0.0, 120.0, 180.0, 240.0])
    frequency = 8_000.0
    phases = 0.2 - 2 * angles
    ods = {}
    for index, (angle, phase) in enumerate(zip(angles, phases)):
        data = np.sin(2.0 * np.pi * frequency * time + phase)
        set_path(ods, f"magnetics.b_field_pol_probe.{index}.name", f"TOR{index}")
        set_path(ods, f"magnetics.b_field_pol_probe.{index}.toroidal_angle", angle)
        set_path(ods, f"magnetics.b_field_pol_probe.{index}.voltage.time", time)
        set_path(ods, f"magnetics.b_field_pol_probe.{index}.voltage.data", data)
    return ods


def test_mirnov_plot_helpers_return_figures(monkeypatch, tmp_path):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path))
    ods = _tiny_mirnov_ods()

    fig, ax = mirnov_signal(ods, channels=[0, 1], show=False)
    assert len(ax.lines) == 2
    plt.close(fig)

    fig, ax = mirnov_spectrogram(
        ods,
        channel=0,
        preprocess=False,
        window_size=256,
        time_resolution=8,
        show=False,
    )
    assert ax.collections
    plt.close(fig)

    fig, axes = toroidal_mode_spectrum(
        ods,
        channel_pair=(0, 1),
        preprocess=False,
        phase_geometry=np.pi / 6,
        peak_threshold=0.05,
        show=False,
    )
    assert len(axes) == 3
    assert axes[0].lines
    plt.close(fig)

    fig, ax, result = toroidal_phase_mode_fit(
        _tiny_phase_ods(),
        center_time=0.020,
        channels=[0, 1, 2, 3],
        frequencies=[8_000.0],
        candidate_n=range(0, 5),
        window_size=512,
        preprocess=False,
        show=False,
        return_result=True,
    )
    assert result.modes[0].n == 2
    assert ax.lines
    assert ax.collections
    plt.close(fig)
