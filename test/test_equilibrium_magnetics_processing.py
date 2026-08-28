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
    ("shot", "expected_flux_window"),
    [
        (43684, None),
        (43685, (0.24, 0.26)),
    ],
)
def test_flux_baseline_window_boundary_43685(shot, expected_flux_window):
    config = equilibrium_magnetics_processing_config(shot)
    assert config.flux_baseline_window == expected_flux_window


@pytest.mark.parametrize("shot", [43685, 46402, 46403])
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
        (46403, "native_daq"),
    ],
)
def test_daq_mode_boundary_46403(shot, expected_daq_mode):
    assert equilibrium_magnetics_processing_config(shot).daq_mode == expected_daq_mode


def test_native_daq_era_fails_loudly_instead_of_reusing_the_legacy_path():
    """Shots >= 46403 must not be silently processed through the legacy
    algorithm: the ported native-DAQ semantics do not exist yet, and wrong
    equilibrium inputs are worse than an explicit failure (#195)."""
    config = equilibrium_magnetics_processing_config(46403)
    time = np.arange(2000, dtype=float) * 4e-6
    waveform = np.sin(np.linspace(0.0, 10.0, time.size))
    channels = [{"field_code": 1, "kind": "b_field_pol_probe", "calibration": 1.0}]

    with pytest.raises(UnsupportedMagneticsDaqModeError, match="46403"):
        vest_equilibrium_magnetics_signals(
            46403, channels, lambda _shot, _field: (time, waveform), config=config
        )


def test_native_daq_era_can_still_opt_into_the_legacy_path_explicitly():
    """The explicit processing_config override remains an escape hatch."""
    time = np.arange(2000, dtype=float) * 4e-6
    waveform = np.sin(np.linspace(0.0, 10.0, time.size))
    channels = [{"field_code": 1, "kind": "b_field_pol_probe", "calibration": 1.0}]

    _target_time, _flux, probes = vest_equilibrium_magnetics_signals(
        46403,
        channels,
        lambda _shot, _field: (time, waveform),
        config=VestMagneticsProcessingConfig(),
    )
    assert len(probes) == 1


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
    than partway through processing."""
    def _loader_that_must_not_run(_shot, _field):  # pragma: no cover
        raise AssertionError("raw data must not be loaded for an unsupported shot")

    with pytest.raises(UnsupportedMagneticsGeometryError):
        vfit_equilibrium_magnetics(39204, raw_source=None)
