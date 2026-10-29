"""CGYRO temporal domains, per-run scales and synchronized native/SI views."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from vaft.code.gacode.cgyro import (
    CgyroOutputs,
    CgyroTimeNormalization,
    collect_cgyro_outputs,
    gyrokinetic_elapsed_time_s,
    gyrokinetic_normalized_time,
    gyrokinetic_time_scale_s,
    read_field_history,
)
from vaft.code.gacode.cgyro.time import time_coordinate_metadata
from vaft.code.gacode.result_identity import collection_record, encode, parameters
from vaft.formula.constants import QE
from vaft.machine_mapping.gyrokinetics import _factors
from vaft.plot import gyrokinetics as plots
from vaft.process.gyrokinetics import time_window_average

DATA = Path(__file__).parent / "data/gacode"
RUN = DATA / "cgyro_nonlinear_39915_em"
RETAINED = json.loads((DATA / "cgyro_39915_time_normalization.json").read_text())


@pytest.fixture
def norm():
    return CgyroTimeNormalization.from_metadata(RETAINED["normalization"])


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_native_unit_conversion_and_gkdb_consistency(norm):
    t = np.array([0, 0.2, 100, 1037.0])
    seconds = gyrokinetic_elapsed_time_s(t, norm)
    np.testing.assert_allclose(gyrokinetic_normalized_time(seconds, norm), t)
    assert gyrokinetic_time_scale_s(norm) == pytest.approx(3.975704271409646e-6)
    assert seconds[-1] == pytest.approx(0.004122805329451803)
    # GKDB uses R0 / v_th, v_th=sqrt(2)*c_s, not seconds or a/c_s.
    rmaj = RETAINED["verified_native_local_inputs"]["RMAJ"]
    gkdb_seconds = norm.reference_length_m * rmaj / (np.sqrt(2) * norm.sound_speed_m_s)
    np.testing.assert_allclose(seconds, t * _factors(rmaj, 1)["time"] * gkdb_seconds)
    assert RETAINED["discharge_time_s"] == 0.317
    assert norm.metadata()["experimental_trajectory_equivalence"] is False


def test_native_equilibrium_kev_si_units_and_semantic_accessors(norm):
    eq = {
        "a_meters": norm.reference_length_m,
        "temp_norm": norm.electron_temperature_j / (1000 * QE),
        "mass_norm": norm.reference_mass_kg / 1e-27,
        "vth_norm": norm.sound_speed_m_s,
    }
    native = CgyroOutputs("native-test", equilibrium=eq, time=np.array([0, 3, 6.0]))
    assert native.gyrokinetic_normalized_time is native.time
    np.testing.assert_allclose(
        native.gyrokinetic_elapsed_time_s, native.time * gyrokinetic_time_scale_s(eq)
    )
    assert native.time_normalization.electron_temperature_j == pytest.approx(
        norm.electron_temperature_j
    )
    # An explicitly absent speed can be reconstructed from the verified SI quantities.
    del eq["vth_norm"]
    assert CgyroTimeNormalization.from_equilibrium(eq).sound_speed_m_s == pytest.approx(
        np.sqrt(norm.electron_temperature_j / norm.reference_mass_kg)
    )


@pytest.mark.parametrize("key", ["a_meters", "temp_norm", "mass_norm", "vth_norm"])
@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf])
def test_invalid_reference_quantities_refuse_conversion(key, bad):
    eq = {
        "a_meters": 0.3,
        "temp_norm": 0.1,
        "mass_norm": 3.34358,
        "vth_norm": np.sqrt(100 * QE / 3.34358e-27),
    }
    eq[key] = bad
    with pytest.raises(ValueError):
        gyrokinetic_time_scale_s(eq)


def test_missing_contradictory_mass_and_speed(norm):
    for eq in ({}, {"a_meters": 0.3}, {"a_meters": None}):
        with pytest.raises(ValueError):
            gyrokinetic_time_scale_s(eq)
    with pytest.raises(ValueError, match="deuterium"):
        replace(norm, reference_mass_kg=norm.reference_mass_kg / 2)
    with pytest.raises(ValueError, match="sound speed"):
        replace(norm, sound_speed_m_s=norm.sound_speed_m_s * np.sqrt(2))
    with pytest.raises(ValueError, match="saved time scale"):
        CgyroTimeNormalization.from_metadata(
            {**norm.metadata(), "gyrokinetic_time_scale_s": 1}
        )
    with pytest.raises((ValueError, TypeError)):
        plots.plot_flux_trace([0, 1], {"Qi": [1, 2]}, time_unit="ms")


def test_real_native_zero_scales_and_retained_normalization_provenance(norm):
    native = collect_cgyro_outputs(RUN)
    with pytest.raises(ValueError):
        _ = native.gyrokinetic_elapsed_time_s
    assert time_coordinate_metadata(native)["status"] == "unavailable"
    assert (
        read_field_history(RUN).metadata["time_normalization"]["status"]
        == "unavailable"
    )
    assert (
        hashlib.sha256(
            (RUN / RETAINED["source"]["native_input_file"]).read_bytes()
        ).hexdigest()
        == RETAINED["source"]["native_input_sha256"]
    )
    local = SimpleNamespace(
        minor_radius=norm.reference_length_m,
        electron_temperature=norm.electron_temperature_j,
        sound_speed=norm.sound_speed_m_s,
    )
    metadata = time_coordinate_metadata(native, local)
    assert metadata["status"] == "available"
    # The companion normalization must never replace positive contradictory output.
    native.equilibrium = {
        "a_meters": norm.reference_length_m * 2,
        "temp_norm": norm.electron_temperature_j / (1000 * QE),
        "mass_norm": 3.34358,
        "vth_norm": norm.sound_speed_m_s,
    }
    assert time_coordinate_metadata(native, local)["status"] == "unavailable"


def test_xml_roundtrip_preserves_normalization_discharge_and_native_window(norm):
    from test_cgyro_adapter import tglf_local

    from vaft.code.gacode.cgyro import cgyro_input_from_tglf

    local = tglf_local()
    local = replace(
        local,
        normalisation=replace(
            local.normalisation,
            minor_radius=norm.reference_length_m,
            electron_temperature=norm.electron_temperature_j,
            sound_speed=norm.sound_speed_m_s,
        ),
    )
    local = cgyro_input_from_tglf(local)
    record = collection_record(
        "CGYRO",
        [(local, collect_cgyro_outputs(RUN))],
        state_identity=RETAINED["state_identity"],
        windows=[(100, 104)],
    )
    record["experimental_time_s"] = 0.317
    restored = parameters(encode(record))
    assert restored["experimental_time_s"] == 0.317
    run = restored["local_runs"][0]
    assert run["averaging_window"] == [100, 104]
    np.testing.assert_allclose(
        run["averaging_window_elapsed_s"],
        np.array([100, 104]) * gyrokinetic_time_scale_s(norm),
    )
    assert CgyroTimeNormalization.from_metadata(
        run["time_normalization"]
    ).gyrokinetic_time_scale_s == gyrokinetic_time_scale_s(norm)


def test_restart_alignment_window_and_display_remain_equivalent(norm):
    native = collect_cgyro_outputs(RUN)
    trace = native.flux_time_trace()[:, 1].sum(axis=0)
    t = native.time[: len(trace)]
    si = gyrokinetic_elapsed_time_s(t, norm)
    np.testing.assert_allclose(
        time_window_average(t, trace, window=(100, 104)),
        time_window_average(
            si, trace, window=gyrokinetic_elapsed_time_s([100, 104], norm)
        ),
    )
    _, a = plots.plot_flux_trace(t, {"Q": trace}, window=(100, 104))
    _, b = plots.plot_flux_trace(
        t, {"Q": trace}, window=(100, 104), time_unit="ms", time_normalization=norm
    )
    np.testing.assert_array_equal(a.lines[0].get_xdata(), t)
    np.testing.assert_allclose(b.lines[0].get_xdata(), si * 1000)
    np.testing.assert_allclose(b.patches[0].get_x(), si[0] * 1000)
    assert a.get_xlabel() == "Gyrokinetic elapsed time [a/c_s]"
    assert b.get_xlabel() == "Gyrokinetic elapsed time [ms]"


def test_predator_prey_markers_fit_and_colorbar_transform_together(norm):
    # No refitting in a different time domain: native fit/peak coordinates are displayed together.
    t = np.array([0.0, 1, 2, 3])
    values = np.array([1.0, 2, 1, 2])
    fit = SimpleNamespace(
        time_window=(1.0, 3.0),
        turbulence_fit=values[1:],
        zonal_fit=values[1:],
        r2_turbulence=1,
        r2_zonal=1,
    )
    metrics = SimpleNamespace(turbulence_peak_times=[1.0], zonal_peak_times=[2.0])
    _, axes = plots.plot_predator_prey_trace(
        t,
        values,
        values,
        transport=values,
        fit=fit,
        metrics=metrics,
        time_unit="ms",
        time_normalization=norm,
    )
    scale = gyrokinetic_time_scale_s(norm) * 1000
    np.testing.assert_allclose(axes[0].lines[1].get_xdata(), t[1:] * scale)
    np.testing.assert_allclose(axes[1].lines[2].get_xdata(), [2 * scale] * 2)
    for axis in axes:
        np.testing.assert_allclose(axis.lines[0].get_xdata(), t * scale)
    fig, axis = plots.plot_predator_prey_phase_space(
        values, values, time=t, time_unit="ms", time_normalization=norm
    )
    np.testing.assert_allclose(axis.collections[0].get_array(), t[:-1] * scale)
    assert fig.axes[-1].get_ylabel() == "Gyrokinetic elapsed time [ms]"
    _, axis = plots.plot_predator_prey_lag(
        t, values, peak_lag=1.0, time_unit="ms", time_normalization=norm
    )
    np.testing.assert_allclose(axis.lines[-1].get_xdata(), [scale] * 2)
    assert "[ms]" in axis.get_xlabel()


def test_fields_animation_and_causal_diagnostics_use_same_coordinate(norm):
    h = read_field_history(RUN, stride=5)
    diagnostic = {"Q": {"time": h.time, "values": np.arange(len(h.time)), "unit": "GB"}}
    with pytest.raises(ValueError, match="no verified"):
        plots.plot_gyrokinetics_field_potential(h, time_unit="ms")
    movie = plots.plot_gyrokinetics_field_potential(
        h,
        animation=True,
        diagnostics=diagnostic,
        time_unit="ms",
        time_normalization=norm,
    )
    np.testing.assert_allclose(
        movie.driver.values, h.time * gyrokinetic_time_scale_s(norm) * 1000
    )
    # Static frame shares the very same model/diagnostic conversion, with no future data.
    _, axes = plots.plot_gyrokinetics_field_potential(
        h,
        frame_index=2,
        diagnostics=diagnostic,
        time_unit="ms",
        time_normalization=norm,
    )
    assert "Gyrokinetic elapsed time=" in axes[0].get_title()
    np.testing.assert_allclose(
        axes[1].lines[0].get_xdata(), h.time[:3] * gyrokinetic_time_scale_s(norm) * 1000
    )
    assert axes[1].get_xlabel() == "Gyrokinetic elapsed time [ms]"
    restored = replace(
        h, metadata={**h.metadata, "time_normalization": norm.metadata()}
    )
    _, axes = plots.plot_gyrokinetics_field_potential(restored, time_unit="ms")
    assert "ms" in axes[0].get_title()


@pytest.mark.parametrize("radius", ["0.7", "0.8"])
def test_real_profile_model_two_rounded_normalization(radius):
    from vaft.code.gacode.cgyro.outputs import parse_equilibrium

    # Real PROFILE_MODEL=2 output writes only five significant figures, including
    # mass_norm=3.3436 instead of the internal constant 3.34358. Do not reject it.
    path = DATA / f"cgyro_oracle_48224/r{radius}/out.cgyro.equilibrium"
    eq = parse_equilibrium(path, 3)
    norm = CgyroTimeNormalization.from_equilibrium(eq, source=str(path))
    assert norm.reference_mass_kg == pytest.approx(3.3436e-27)
    assert norm.electron_temperature_j == eq["temp_norm"] * 1e3 * QE
    assert norm.sound_speed_m_s == eq["vth_norm"]
    assert gyrokinetic_elapsed_time_s(1.0, norm) == eq["a_meters"] / eq["vth_norm"]
