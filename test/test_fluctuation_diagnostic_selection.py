"""Explicit representative diagnostic selection and canonical map plots."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from vaft.omas.fluctuation import DiagnosticSelection, select_fluctuation_records
from vaft.plot.fluctuation import (
    plot_cross_diagnostic_coherence_spectrogram,
    plot_multi_diagnostic_coherent_fraction,
    plot_multi_diagnostic_coherent_spectrogram,
    plot_multi_diagnostic_participation,
    plot_multi_diagnostic_phase,
)
from vaft.process.fluctuation import (
    coherent_components, cross_spectral_matrix, cross_spectrogram,
)


def _ods():
    time = np.arange(8) / 1_000
    return {
        "magnetics": {"b_field_pol_probe": [
            {"name": "M1", "field": {"time": time, "data": np.arange(8.)}},
            {"name": "M2", "voltage": {"time": time, "data": np.arange(8.)}},
        ]},
        "soft_x_rays": {"channel": [
            {"name": "S1", "brightness": {"time": time, "data": np.arange(8.) * 2}},
        ]},
        "interferometer": {"channel": [
            {"name": "I1", "n_e_line": {"time": time, "data": np.arange(8.) * 3}},
        ]},
        "spectrometer_uv": {"channel": [
            {"processed_line": [
                {"label": "H-alpha_6563", "intensity": {"time": time, "data": np.arange(8.)}},
                {"label": "CIII_4650", "intensity": {"time": time, "data": np.arange(8.) * 4}},
            ]},
        ]},
        "camera_visible": {"channel": [
            {"detector": [{"frame": [
                {"time": float(t), "image_raw": np.full((3, 4), i, dtype=float)}
                for i, t in enumerate(time)
            ]}]},
        ]},
    }


def test_one_explicit_record_per_diagnostic_with_source_provenance():
    selections = [
        DiagnosticSelection("mirnov", channel="M1"),
        DiagnosticSelection("soft_x_rays", channel=0, quantity="brightness", units="V"),
        DiagnosticSelection("interferometer", channel="I1"),
        DiagnosticSelection("camera_visible", channel=0, region=(0, 2, 1, 3),
                            background_frames=3),
        DiagnosticSelection("spectrometer_uv", emission="H-alpha_6563", units="count/s"),
    ]
    result = select_fluctuation_records(_ods(), selections)
    assert tuple(record.name for record in result.records) == tuple(
        item.diagnostic for item in selections
    )
    assert result.records[0].units == "T"
    assert result.records[2].units == "m^-2"
    assert "ROI(0, 2, 1, 3)" in result.sources[3]
    assert "H-alpha_6563" in result.sources[4]
    assert tuple(record.source for record in result.records) == result.sources
    assert result.records[3].data.shape == (8,)


def test_raw_mirnov_and_ambiguous_selection_are_rejected():
    ods = _ods()
    with pytest.raises(ValueError, match="integrated/calibrated field"):
        select_fluctuation_records(ods, [DiagnosticSelection("mirnov", channel="M2")])
    with pytest.raises(ValueError, match="exactly one representative"):
        select_fluctuation_records(ods, [DiagnosticSelection("mirnov", channel=0),
                                        DiagnosticSelection("mirnov", channel=1)])
    with pytest.raises(ValueError, match="explicit ROI"):
        select_fluctuation_records(ods, [DiagnosticSelection("camera_visible", channel=0)])
    with pytest.raises(ValueError, match="nonnegative integer"):
        select_fluctuation_records(ods, [DiagnosticSelection(
            "camera_visible", channel=0, detector=0.9, region=(0, 2, 1, 3)
        )])
    with pytest.raises(ValueError, match="positive integer"):
        select_fluctuation_records(ods, [DiagnosticSelection(
            "camera_visible", channel=0, background_frames=2.9, region=(0, 2, 1, 3)
        )])
    with pytest.raises(ValueError, match="emission identity"):
        select_fluctuation_records(ods, [DiagnosticSelection("spectrometer_uv")])


def test_camera_temporal_component_and_uv_emission_identity():
    ods = _ods()
    time = np.arange(8) / 1_000
    component = select_fluctuation_records(ods, [DiagnosticSelection(
        "camera_visible", component_time=time, component_data=np.arange(8),
        component_label="first SVD score", units="a.u."
    )])
    np.testing.assert_array_equal(component.records[0].data, np.arange(8))
    assert "first SVD score" in component.sources[0]
    line = select_fluctuation_records(ods, [DiagnosticSelection(
        "spectrometer_uv", emission="CIII", channel=0, units="count/s"
    )])
    assert "CIII_4650" in line.sources[0]
    with pytest.raises(ValueError, match="provenance label"):
        select_fluctuation_records(ods, [DiagnosticSelection(
            "camera_visible", component_time=time, component_data=np.arange(8)
        )])
    with pytest.raises(ValueError, match="exceeds frame shape"):
        select_fluctuation_records(ods, [DiagnosticSelection(
            "camera_visible", channel=0, region=(0, 2, 1, 9)
        )])


def test_sxr_multiband_and_quantity_selection_keep_declared_units():
    ods = _ods()
    channel = ods["soft_x_rays"]["channel"][0]
    channel["brightness"]["data"] = np.stack([np.arange(8.), 2 * np.arange(8.)])
    channel["power"] = {"time": channel["brightness"]["time"],
                        "data": 3 * np.arange(8.)}
    with pytest.raises(ValueError, match="select energy_band"):
        select_fluctuation_records(ods, [DiagnosticSelection(
            "soft_x_rays", channel=0, quantity="brightness", units="W.m^-2.sr^-1"
        )])
    selected = select_fluctuation_records(ods, [DiagnosticSelection(
        "soft_x_rays", channel=0, quantity="brightness", energy_band=1,
        units="W.m^-2.sr^-1"
    )])
    np.testing.assert_array_equal(selected.records[0].data, 2 * np.arange(8.))
    assert "energy_band1" in selected.sources[0]
    assert selected.records[0].units == "W.m^-2.sr^-1"
    power = select_fluctuation_records(ods, [DiagnosticSelection(
        "soft_x_rays", channel=0, quantity="power", units="W"
    )])
    np.testing.assert_array_equal(power.records[0].data, 3 * np.arange(8.))
    with pytest.raises(ValueError, match="specify units explicitly"):
        select_fluctuation_records(ods, [DiagnosticSelection(
            "soft_x_rays", channel=0, quantity="power"
        )])
    with pytest.raises(ValueError, match="SXR quantity"):
        select_fluctuation_records(ods, [DiagnosticSelection(
            "soft_x_rays", channel=0, units="W"
        )])
    channel["brightness"]["data"] = np.arange(64.).reshape(8, 8)
    with pytest.raises(ValueError, match="energy_axis"):
        select_fluctuation_records(ods, [DiagnosticSelection(
            "soft_x_rays", channel=0, quantity="brightness", energy_band=2, units="V"
        )])
    square = select_fluctuation_records(ods, [DiagnosticSelection(
        "soft_x_rays", channel=0, quantity="brightness", energy_band=2,
        energy_axis=0, units="V"
    )])
    np.testing.assert_array_equal(square.records[0].data, np.arange(64.).reshape(8, 8)[2])
    with pytest.raises(ValueError, match="specify units explicitly"):
        select_fluctuation_records(ods, [DiagnosticSelection(
            "spectrometer_uv", emission="H-alpha_6563"
        )])


def test_canonical_maps_keep_time_frequency_mask_and_units():
    fs = 40_000
    time = np.arange(8_000) / fs
    x = np.sin(2 * np.pi * 2_000 * time)
    y = np.sin(2 * np.pi * 2_000 * time + 0.4)
    pair = cross_spectrogram(time, x, time, y, nperseg=200)
    figure, axes = plot_cross_diagnostic_coherence_spectrogram(pair)
    assert axes.get_ylabel() == "Frequency [kHz]"
    assert "coherence" in figure.axes[1].get_ylabel().lower()
    plt.close(figure)

    selected = select_fluctuation_records(_ods(), [
        DiagnosticSelection("mirnov", channel=0),
        DiagnosticSelection("soft_x_rays", channel=0, quantity="brightness", units="V"),
    ])
    assert len(selected.records) == 2
    from vaft.process.fluctuation import FluctuationRecord
    matrix = cross_spectral_matrix([
        FluctuationRecord("a", time, x, source="first selected channel"),
        FluctuationRecord("b", time, y, source="chosen ROI"),
    ], nperseg=200)
    components = coherent_components(matrix, reference="a")
    assert components.sources == matrix.sources == ("first selected channel", "chosen ROI")
    for plotter, args in (
        (plot_multi_diagnostic_coherent_spectrogram, (components,)),
        (plot_multi_diagnostic_coherent_fraction, (components,)),
        (plot_multi_diagnostic_participation, (components, "b")),
        (plot_multi_diagnostic_phase, (components, "b")),
    ):
        figure, axes = plotter(*args)
        assert axes.get_xlabel() == "Time [s]"
        assert axes.collections
        plt.close(figure)
