"""Shot-era equilibrium-magnetics acquisition policy (issue #195).

The pre-43685 expectations below duplicate the legacy hardcoded thresholds
on `VestMagneticsProcessingConfig` on purpose: they prove the migration to
`vest.yaml` is behavior-preserving for every era that already worked.
"""

import numpy as np
import pytest

from vaft.machine_mapping.magnetics import (
    UnsupportedMagneticsGeometryError,
    equilibrium_magnetics_processing_config,
    require_supported_magnetics_geometry,
    vfit_equilibrium_magnetics,
)
from vaft.process.magnetics import (
    UnsupportedMagneticsDaqModeError,
    VestMagneticsProcessingConfig,
    vest_equilibrium_magnetics_signals,
    vest_flux_loop_legacy,
)


@pytest.mark.parametrize(
    ("shot", "expected_window"),
    [
        (41445, (6000, 8500, 8500)),
        (41446, (6500, 9000, 5000)),
        (41451, (6500, 9000, 5000)),
        (41452, (6000, 8500, 8500)),
        (41659, (6000, 8500, 8500)),
        (41660, (6500, 9000, 5000)),
        (43684, (6500, 9000, 5000)),
    ],
)
def test_yaml_config_reproduces_legacy_windows_for_pre_43685_eras(shot, expected_window):
    config = equilibrium_magnetics_processing_config(shot)
    assert config.window_for_shot(shot) == expected_window
    assert config.window_for_shot(shot) == VestMagneticsProcessingConfig().window_for_shot(shot)
    assert config.flux_baseline_window is None
    assert config.daq_mode == "legacy"


@pytest.mark.parametrize(
    ("shot", "expected_probe_baseline"),
    [
        (43684, 5000),
        (43685, 1750),
    ],
)
def test_bpol_baseline_boundary_43685(shot, expected_probe_baseline):
    config = equilibrium_magnetics_processing_config(shot)
    _, _, probe_baseline_end = config.window_for_shot(shot)
    assert probe_baseline_end == expected_probe_baseline


@pytest.mark.parametrize(
    ("shot", "expected_flux_window", "expected_flux_samples"),
    [
        (43684, None, None),
        # Slow-DAQ era: MATLAB index_FL_start/end = 6001:6500 = 0.24--0.26 s.
        (43685, (0.24, 0.26), None),
        (46403, (0.24, 0.26), None),
        # Native-DAQ era: flux loops moved onto the fast acquisition and take
        # the probes' leading-sample baseline instead.
        (46404, None, 1750),
    ],
)
def test_flux_baseline_rule_switches_with_the_acquisition_era(
    shot, expected_flux_window, expected_flux_samples
):
    config = equilibrium_magnetics_processing_config(shot)
    assert config.flux_baseline_window == expected_flux_window
    assert config.flux_baseline_samples == expected_flux_samples


@pytest.mark.parametrize("shot", [43685, 46403, 46404])
def test_late_eras_output_the_documented_026_to_036_second_window(shot):
    """Indices 6500..9000 on the 4e-5 s grid are exactly 0.26--0.36 s."""
    config = equilibrium_magnetics_processing_config(shot)
    index_start, index_end, _ = config.window_for_shot(shot)
    timebase = config.timebase()
    assert timebase[index_start] == pytest.approx(0.26)
    assert timebase[index_end] == pytest.approx(0.36)


@pytest.mark.parametrize(
    ("shot", "expected_daq_mode"),
    [
        (46402, "legacy"),
        (46403, "legacy"),
        (46404, "native_daq"),
    ],
)
def test_daq_mode_boundary_46404(shot, expected_daq_mode):
    assert equilibrium_magnetics_processing_config(shot).daq_mode == expected_daq_mode


def test_native_daq_era_processes_flux_loops_on_the_fast_acquisition():
    """The native-DAQ path is implemented, not stubbed: it runs and differs
    from the slow-DAQ era only in how the flux-loop baseline is chosen."""
    config = equilibrium_magnetics_processing_config(46404)
    time = np.arange(0.26, 0.36, 4e-6)
    waveform = np.sin(np.linspace(0.0, 10.0, time.size))
    channels = [{"field_code": 1, "kind": "flux_loop", "calibration": 1.0}]

    _target_time, flux, _probes = vest_equilibrium_magnetics_signals(
        46404, channels, lambda _shot, _field: (time, waveform), config=config
    )
    assert len(flux) == 1
    assert np.all(np.isfinite(flux[0]))


def test_unknown_daq_mode_is_rejected():
    config = VestMagneticsProcessingConfig(daq_mode="something_else")
    with pytest.raises(UnsupportedMagneticsDaqModeError, match="daq_mode"):
        vest_equilibrium_magnetics_signals(
            46404, [], lambda _shot, _field: None, config=config
        )


def test_physical_time_flux_baseline_is_sample_rate_independent():
    """A 0.24--0.26 s baseline window must select the same physical interval
    whether the loop is acquired on the slow or the fast grid."""
    slow_time = np.arange(0.20, 0.40, 4e-5)
    fast_time = np.arange(0.20, 0.40, 4e-6)
    config = VestMagneticsProcessingConfig(flux_baseline_window=(0.24, 0.26))

    def _ramp(t):
        return np.linspace(0.0, 1.0, t.size)

    slow_result = vest_flux_loop_legacy(
        slow_time, _ramp(slow_time), 1.0, flux_loop_number=1, config=config
    )
    fast_result = vest_flux_loop_legacy(
        fast_time, _ramp(fast_time), 1.0, flux_loop_number=1, config=config
    )

    # Both baselines are removed over the same physical window, so the
    # processed waveform crosses zero at the same physical times.
    slow_at_025 = np.interp(0.25, slow_time, slow_result)
    fast_at_025 = np.interp(0.25, fast_time, fast_result)
    assert slow_at_025 == pytest.approx(fast_at_025, abs=1e-6)


# --------------------------------------------------------------------------
# Shot 39204 geometry provenance (issue #195 section 5)
# --------------------------------------------------------------------------


def test_shot_39204_is_not_silently_given_the_shipped_geometry():
    """The legacy VFIT source overrides the magnetics geometry for shot
    39204 specifically (ver_2310). This repository ships ver_2302 only, so
    the shot must fail clearly rather than be assigned wrong sensor
    positions -- and must not reach for an external MATLAB file."""
    with pytest.raises(UnsupportedMagneticsGeometryError, match="2310"):
        require_supported_magnetics_geometry(39204)


@pytest.mark.parametrize("shot", [39203, 39205, 43685, 46402, None])
def test_other_shots_are_unaffected_by_the_geometry_guard(shot):
    require_supported_magnetics_geometry(shot)


def test_geometry_guard_fires_before_any_raw_data_is_touched():
    """vfit_equilibrium_magnetics must refuse shot 39204 up front rather
    than partway through processing. `raw_source` points at a path that does
    not exist, so reaching the loader at all would surface a different
    error than the geometry guard."""
    with pytest.raises(UnsupportedMagneticsGeometryError):
        vfit_equilibrium_magnetics(39204, raw_source="/nonexistent/raw.json.gz")


def test_native_flux_baseline_matches_the_matlab_leading_sample_fit():
    """Pin the donor semantics from VEST_MagneticSignalProcessing2.m:

        coeff   = polyfit(timeFastFL(1:1750), inttemp(1:1750), 1);
        inttemp = inttemp - polyval(coeff, timeFastFL);

    i.e. a linear fit over the first 1750 samples of the *fast* flux
    acquisition, not the 0.24--0.26 s slow-DAQ window of the era before it.
    """
    from vaft.compat import cumtrapz_compat

    time = np.arange(0.26, 0.36, 4e-6)
    raw = np.sin(np.linspace(0.0, 12.0, time.size)) + 0.3
    calibration = 2.0
    config = VestMagneticsProcessingConfig(flux_baseline_samples=1750)

    result = vest_flux_loop_legacy(
        time, raw, calibration, flux_loop_number=1, config=config
    )

    integrated = -cumtrapz_compat(raw / calibration, x=time, initial=0) / (2 * np.pi)
    fit = np.polyfit(time[:1750], integrated[:1750], 1)
    expected = integrated - np.polyval(fit, time)

    np.testing.assert_allclose(result, expected)


def test_native_and_slow_daq_flux_baselines_actually_differ():
    """Guard against the two era rules silently collapsing into one."""
    time = np.arange(0.20, 0.40, 4e-6)
    raw = np.sin(np.linspace(0.0, 12.0, time.size)) + 0.3

    native = vest_flux_loop_legacy(
        time, raw, 1.0, flux_loop_number=1,
        config=VestMagneticsProcessingConfig(flux_baseline_samples=1750),
    )
    slow = vest_flux_loop_legacy(
        time, raw, 1.0, flux_loop_number=1,
        config=VestMagneticsProcessingConfig(flux_baseline_window=(0.24, 0.26)),
    )
    assert not np.allclose(native, slow)


def test_daq_mode_must_agree_with_the_flux_baseline_rule():
    """`daq_mode` must not be decorative. A config claiming the native era
    while carrying no native flux rule would silently process flux loops the
    legacy way, so the mismatch is rejected in both directions."""
    with pytest.raises(UnsupportedMagneticsDaqModeError, match="flux_baseline_samples"):
        vest_equilibrium_magnetics_signals(
            46404, [], lambda _s, _f: None,
            config=VestMagneticsProcessingConfig(daq_mode="native_daq"),
        )

    with pytest.raises(UnsupportedMagneticsDaqModeError, match="native-DAQ rule"):
        vest_equilibrium_magnetics_signals(
            46403, [], lambda _s, _f: None,
            config=VestMagneticsProcessingConfig(
                daq_mode="legacy", flux_baseline_samples=1750
            ),
        )


@pytest.mark.parametrize("shot", [41445, 43685, 46403, 46404])
def test_configs_built_from_vest_yaml_are_always_self_consistent(shot):
    """Every era the YAML can produce must pass the consistency check."""
    config = equilibrium_magnetics_processing_config(shot)
    vest_equilibrium_magnetics_signals(shot, [], lambda _s, _f: None, config=config)
