"""Shot-era equilibrium-magnetics acquisition policy (issue #195).

The pre-43685 expectations below duplicate the legacy hardcoded thresholds
on `VestMagneticsProcessingConfig` on purpose: they prove the migration to
`vest.yaml` is behavior-preserving for every era that already worked.
"""

import numpy as np
import pytest

from vaft.machine_mapping.magnetics import (
    UnsupportedMagneticsGeometryError,
    _pinned_probe_index,
    equilibrium_magnetics_processing_config,
    known_magnetics_faults,
    magnetics_wiring_for_shot,
    require_supported_magnetics_geometry,
    vfit_equilibrium_magnetics,
)
from vaft.machine_mapping.utils import VestConfigurationError
from vaft.process.magnetics import (
    DegenerateBaselineWindowError,
    UnsupportedMagneticsDaqModeError,
    VestMagneticsProcessingConfig,
    vest_b_field_pol_probe_legacy,
    vest_equilibrium_magnetics_detailed,
    vest_equilibrium_magnetics_signals,
    vest_flux_loop_flux_from_voltage,
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
# Probe wiring history (issues #195 section 5, #956)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("shot", "layout", "plus_006", "minus_042"),
    [
        (29350, "pre-39438", (170, 0.004539265), (225, -0.00452284)),
        (39204, "pre-39438", (170, 0.004539265), (225, -0.00452284)),
        (39437, "pre-39438", (170, 0.004539265), (225, -0.00452284)),
        (39438, "2409", (225, 0.004543389), (170, -0.00452284)),
        (48000, "2409", (225, 0.004543389), (170, -0.00452284)),
    ],
)
def test_the_outboard_wiring_switches_overnight_to_39438(shot, layout, plus_006, minus_042):
    """Z=+0.06 and Z=-0.42 trade fields 170 and 225 (with their signs)
    between 2023-05-29 (39437) and 2023-05-30 (39438); see vest.yaml."""
    wiring = magnetics_wiring_for_shot(shot)
    channels = wiring.channels

    assert wiring.layout == layout
    assert (channels[35]["field_code"], channels[35]["calibration"]) == plus_006
    assert (channels[47]["field_code"], channels[47]["calibration"]) == minus_042


def test_only_the_two_outboard_positions_are_rewired():
    """The other rows where VFIT's 2302 file differs (field 197 sign, fields
    179/180 calibration) are not supported by the raw signals and stay."""
    before = magnetics_wiring_for_shot(39437).channels
    after = magnetics_wiring_for_shot(39438).channels

    changed = {index for index, (a, b) in enumerate(zip(before, after)) if a != b}
    assert changed == {35, 47}
    assert [c["kind"] for c in before] == [c["kind"] for c in after]
    assert sorted(c["field_code"] for c in before) == sorted(c["field_code"] for c in after)


def test_an_override_names_the_position_it_rewires():
    """Pinned by r and z, so a reordered asset cannot rewire the wrong probe."""
    with pytest.raises(VestConfigurationError, match="not the probe at"):
        _pinned_probe_index({"index": 35, "r": 0.796, "z": -0.42}, context="test")


@pytest.mark.parametrize(
    ("shot", "faulted"),
    [
        (29350, True),
        (36480, True),
        (36481, False),
        (36821, False),
        (36822, True),
        (36905, True),
        (36906, False),
        (39438, False),
    ],
)
def test_the_z_plus_006_probe_is_recorded_broken_where_the_scan_found_it(shot, faulted):
    faults = known_magnetics_faults(shot)
    assert (("b_field_pol_probe", 35) in faults) is faulted
    assert set(faults) <= {("b_field_pol_probe", 35)}


def test_39204_is_processed_with_the_layout_of_its_neighbours():
    """VFIT's `shot == 39204 -> ver_2310` override does not survive the raw
    signals (#956); 39204 is no longer refused."""
    require_supported_magnetics_geometry(39204)
    assert magnetics_wiring_for_shot(39204).channels == magnetics_wiring_for_shot(39203).channels


def test_the_processing_reads_the_shots_wiring(monkeypatch):
    import vaft.machine_mapping.magnetics as magnetics

    seen = {}

    def record(shot, channels, *args, **kwargs):
        seen[shot] = [int(channel["field_code"]) for channel in channels]
        raise RuntimeError("stop")

    monkeypatch.setattr(magnetics, "vest_equilibrium_magnetics_detailed", record)
    for shot in (39437, 39438):
        with pytest.raises(RuntimeError, match="stop"):
            magnetics.vfit_equilibrium_magnetics_detailed(shot, raw_source="/nonexistent/raw.json.gz")

    assert (seen[39437][35], seen[39437][47]) == (170, 225)
    assert (seen[39438][35], seen[39438][47]) == (225, 170)


def test_the_guard_still_refuses_a_shot_whose_layout_is_unknown(monkeypatch):
    """No shot needs it today; the mechanism stays, and fires before any raw
    data is touched (`raw_source` does not exist)."""
    import vaft.machine_mapping.magnetics as magnetics

    monkeypatch.setitem(magnetics.UNSUPPORTED_MAGNETICS_GEOMETRY_SHOTS, 12345, "9999")
    with pytest.raises(UnsupportedMagneticsGeometryError, match="9999"):
        vfit_equilibrium_magnetics(12345, raw_source="/nonexistent/raw.json.gz")


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


# ---------------------------------------------------------------------------
# Degenerate baseline window error handling (issue #639)
# ---------------------------------------------------------------------------

def test_b_field_pol_probe_degenerate_baseline_raises():
    time = np.linspace(0.0, 0.01, 1000)
    raw = np.sin(np.linspace(0.0, 10.0, 1000))
    cfg = VestMagneticsProcessingConfig(window_override=(0, 500, 1))

    with pytest.raises(DegenerateBaselineWindowError) as exc_info:
        vest_b_field_pol_probe_legacy(
            time, raw, 1.0, shot=41445, config=cfg, channel="probe_test"
        )

    msg = str(exc_info.value)
    assert "shot 41445" in msg
    assert "probe_test" in msg
    assert "1 valid sample(s)" in msg
    assert "requires >= 2" in msg
    assert "time span: [0, 0.01] s" in msg


def test_b_field_pol_probe_allow_zero_fallback():
    time = np.linspace(0.0, 0.01, 1000)
    raw = np.sin(np.linspace(0.0, 10.0, 1000))
    cfg = VestMagneticsProcessingConfig(window_override=(0, 500, 1))

    with pytest.warns(UserWarning, match="Degenerate baseline window.*allow_zero_fallback=True"):
        result = vest_b_field_pol_probe_legacy(
            time, raw, 1.0, shot=41445, config=cfg, allow_zero_fallback=True
        )

    assert result.shape == raw.shape


def test_flux_loop_degenerate_window_raises():
    time = np.linspace(0.0, 0.1, 1000)
    raw = np.cos(np.linspace(0.0, 5.0, 1000))
    cfg = VestMagneticsProcessingConfig(flux_baseline_window=(0.5, 0.6))

    with pytest.raises(DegenerateBaselineWindowError) as exc_info:
        vest_flux_loop_legacy(
            time, raw, 1.0, flux_loop_number=3, config=cfg, shot=43685
        )

    msg = str(exc_info.value)
    assert "shot 43685" in msg
    assert "flux_loop_3" in msg
    assert "0 valid sample(s)" in msg
    assert "requires >= 2" in msg
    assert "time window [0.5, 0.6] s" in msg


def test_flux_loop_degenerate_window_allow_zero_fallback():
    time = np.linspace(0.0, 0.1, 1000)
    raw = np.cos(np.linspace(0.0, 5.0, 1000))
    cfg = VestMagneticsProcessingConfig(
        flux_baseline_window=(0.5, 0.6), allow_zero_fallback=True
    )

    with pytest.warns(UserWarning, match="Degenerate baseline window.*allow_zero_fallback=True"):
        result = vest_flux_loop_legacy(
            time, raw, 1.0, flux_loop_number=3, config=cfg, shot=43685
        )

    assert result.shape == raw.shape


def test_flux_loop_flux_from_voltage_degenerate_samples_raises_and_fallback():
    time = np.linspace(0.0, 0.1, 1000)
    voltage = np.ones(1000)
    cfg = VestMagneticsProcessingConfig(flux_baseline_samples=1)

    with pytest.raises(DegenerateBaselineWindowError) as exc_info:
        vest_flux_loop_flux_from_voltage(
            time, voltage, flux_loop_number=2, config=cfg, shot=46404
        )
    assert "1 valid sample(s)" in str(exc_info.value)

    with pytest.warns(UserWarning, match="Falling back to zero baseline"):
        result = vest_flux_loop_flux_from_voltage(
            time, voltage, flux_loop_number=2, config=cfg, allow_zero_fallback=True
        )
    assert result.shape == voltage.shape


def test_flux_loop_two_segment_out_of_bounds_raises_and_fallback():
    # Signal with fewer samples than flux_baseline_first_start (3499)
    time = np.linspace(0.0, 0.01, 500)
    raw = np.sin(np.linspace(0.0, 5.0, 500))
    cfg = VestMagneticsProcessingConfig()

    with pytest.raises(DegenerateBaselineWindowError) as exc_info:
        vest_flux_loop_legacy(time, raw, 1.0, flux_loop_number=1, config=cfg)
    assert "0 valid sample(s)" in str(exc_info.value)

    with pytest.warns(UserWarning, match="allow_zero_fallback=True"):
        result = vest_flux_loop_legacy(
            time, raw, 1.0, flux_loop_number=1, config=cfg, allow_zero_fallback=True
        )
    assert result.shape == raw.shape


def test_vest_equilibrium_magnetics_detailed_channel_raises_and_fallback():
    shot = 41445
    # Signal too short for default probe baseline (8500 samples)
    time = np.linspace(0.0, 0.01, 500)
    data = np.ones(500)
    channels = [
        {"field_code": 1, "calibration": 1.0, "kind": "b_field_pol_probe", "name": "Bpol_1"},
    ]
    loader = lambda s, f: (time, data)
    cfg = VestMagneticsProcessingConfig(window_override=(0, 200, 1))

    with pytest.raises(DegenerateBaselineWindowError) as exc_info:
        vest_equilibrium_magnetics_detailed(
            shot, channels, loader, config=cfg
        )
    assert "Bpol_1" in str(exc_info.value)
    assert "shot 41445" in str(exc_info.value)

    # With allow_zero_fallback=True, it succeeds with warning
    with pytest.warns(UserWarning, match="Falling back to zero baseline"):
        res = vest_equilibrium_magnetics_detailed(
            shot, channels, loader, config=cfg, allow_zero_fallback=True
        )
    assert len(res.probes) == 1
    assert res.probes[0].size > 0


def test_b_field_pol_probe_integer_channel_zero():
    time = np.linspace(0.0, 0.01, 1000)
    raw = np.sin(np.linspace(0.0, 10.0, 1000))
    cfg = VestMagneticsProcessingConfig(window_override=(0, 500, 1))

    with pytest.raises(DegenerateBaselineWindowError) as exc_info:
        vest_b_field_pol_probe_legacy(
            time, raw, 1.0, shot=41445, config=cfg, channel=0
        )
    assert "channel 0" in str(exc_info.value)


def test_nan_contaminated_baseline_window_raises():
    time = np.linspace(0.0, 0.01, 1000)
    raw = np.full(1000, np.nan)
    # Even if 500 samples are in-bounds, they are all NaN
    cfg = VestMagneticsProcessingConfig(window_override=(0, 500, 500))

    with pytest.raises(DegenerateBaselineWindowError) as exc_info:
        vest_b_field_pol_probe_legacy(time, raw, 1.0, shot=41445, config=cfg)
    assert "1 valid sample(s)" in str(exc_info.value)
