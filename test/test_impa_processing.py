"""Unit tests for the generic IMPA algorithms.

Every case here is synthetic so the physics of each stage is pinned without
touching the VEST database.
"""

import numpy as np
import pytest

from vaft.process.impa import (
    ImpaProcessingConfig,
    fit_impa_crosstalk,
    remove_bz_crosstalk,
    TfWindowCriteria,
    find_tf_calibration_window,
    fit_impa_geometry,
    fit_impa_tf_coupling,
    impa_calibrate_signals,
    legacy_impa_compensation,
    legacy_impa_position,
    process_impa,
    remove_tf_pickup,
    toroidal_field,
    validate_impa,
)

MU0 = 4.0e-7 * np.pi
SAMPLE_RATE = 25_000.0


def _time(n: int = 25_000) -> np.ndarray:
    return np.arange(n, dtype=float) / SAMPLE_RATE


def _tf_ramp(time: np.ndarray, peak: float = 12_000.0) -> np.ndarray:
    """A trapezoid: quiet start, ramp up, flat top, ramp down.

    The TF stays off for the first 0.11 s as it does on VEST, so the legacy
    2500-sample baseline window really is field-free.
    """
    return np.interp(
        time,
        [0.0, 0.11, 0.16, 0.21, 0.60, 0.90],
        [0.0, 0.0, peak * 0.4, peak, peak * 0.9, 0.0],
    )


def _synthetic_array(time, i_tf, radii, alpha, offset=0.0, bz=0.0, noise=1e-6, seed=0):
    """Build probe waveforms, with a little noise so plateaus look analogue."""
    bt = toroidal_field(np.asarray(radii)[:, None], i_tf[None, :])
    alpha = np.asarray(alpha, dtype=float)[:, None]
    values = alpha * bt + np.sqrt(1.0 - alpha**2) * bz + offset
    if noise:
        values = values + np.random.default_rng(seed).normal(0.0, noise, values.shape)
    return values


# ---------------------------------------------------------------------------
# window selection
# ---------------------------------------------------------------------------
def test_window_prefers_the_best_conditioned_interval_not_the_longest():
    time = _time()
    # A long, perfectly flat TF stretch cannot separate a probe's coupling from
    # its offset; a shorter ramp can, so the finder must prefer the ramp.
    i_tf = np.zeros_like(time)
    i_tf[time >= 0.12] = 12_000.0                     # instantaneous turn-on
    ramp = (time >= 0.60) & (time < 0.70)
    i_tf[ramp] = np.interp(time[ramp], [0.60, 0.70], [12_000.0, 6_000.0])
    i_tf[time >= 0.70] = 6_000.0
    ip = np.zeros_like(time)
    ip[(time >= 0.59) & (time < 0.60)] = 50_000.0     # splits the two candidates

    window, _ = find_tf_calibration_window(time, i_tf, ip)

    assert window is not None
    flat_duration = 0.59 - 0.12
    assert window.duration < flat_duration  # the ramp is the shorter interval
    assert window.start_time >= 0.60
    assert window.metrics["tf_dynamic_range"] > 0.4
    assert window.metrics["candidate_intervals"] == 2


def test_window_is_rejected_when_the_tf_never_reaches_the_minimum():
    time = _time(2_000)
    window, reasons = find_tf_calibration_window(time, np.full_like(time, 10.0))

    assert window is None
    assert any("I_TF" in reason for reason in reasons)


def test_window_is_rejected_while_a_plasma_is_present():
    time = _time()
    i_tf = _tf_ramp(time)
    window, reasons = find_tf_calibration_window(time, i_tf, np.full_like(time, 50_000.0))

    assert window is None
    assert any("Ip" in reason for reason in reasons)


def test_window_is_rejected_while_the_pf_coils_are_driven():
    time = _time()
    i_tf = _tf_ramp(time)
    pf = np.full((3, time.size), 20_000.0)
    window, reasons = find_tf_calibration_window(time, i_tf, np.zeros_like(time), pf)

    assert window is None
    assert any("PF" in reason for reason in reasons)


def test_isolated_noise_spikes_do_not_fragment_a_clean_interval():
    time = _time()
    i_tf = _tf_ramp(time)
    ip = np.zeros_like(time)
    ip[::500] = 50_000.0  # single-sample spikes on an otherwise quiet trace

    window, _ = find_tf_calibration_window(time, i_tf, ip)

    assert window is not None
    assert window.duration > 0.1


def test_window_is_rejected_when_every_clean_interval_is_too_short():
    time = _time()
    i_tf = _tf_ramp(time)
    ip = np.full_like(time, 50_000.0)
    # 2.4 ms of quiet: long enough to survive the 1 ms median filter, still
    # below the 5 ms minimum duration.
    ip[6000:6060] = 0.0

    window, reasons = find_tf_calibration_window(time, i_tf, ip)

    assert window is None
    assert any("shorter than" in reason for reason in reasons)


# ---------------------------------------------------------------------------
# fits
# ---------------------------------------------------------------------------
def test_coupling_regression_recovers_a_known_alpha_and_offset():
    time = _time()
    i_tf = _tf_ramp(time)
    radii = 0.4 + np.arange(4) * 0.05
    alpha = np.array([0.05, -0.10, 0.15, 0.02])
    offset = 1.0e-3
    measured = _synthetic_array(time, i_tf, radii, alpha, offset=offset, noise=0.0)

    window, _ = find_tf_calibration_window(time, i_tf)
    fit = fit_impa_tf_coupling(measured, i_tf, radii, window)

    np.testing.assert_allclose(fit.alpha, alpha, atol=1e-6)
    np.testing.assert_allclose(fit.beta, offset, atol=1e-9)
    np.testing.assert_allclose(fit.tilt_deg, np.degrees(np.arcsin(alpha)), atol=1e-4)
    np.testing.assert_allclose(fit.coupling_ratio, alpha / radii, atol=1e-6)
    assert np.all(fit.r_squared > 0.999)
    assert not np.any(fit.bound_hit)


def test_coupling_regression_flags_a_probe_outside_the_tilt_bounds():
    time = _time()
    i_tf = _tf_ramp(time)
    radii = np.array([0.4, 0.45])
    measured = _synthetic_array(time, i_tf, radii, [0.02, 0.9])

    window, _ = find_tf_calibration_window(time, i_tf)
    fit = fit_impa_tf_coupling(measured, i_tf, radii, window, tilt_bounds_deg=(-10.0, 10.0))

    assert not fit.bound_hit[0]
    assert fit.bound_hit[1]


def test_geometry_fit_recovers_a_known_r0():
    time = _time()
    i_tf = _tf_ramp(time)
    radii = 0.37 + np.arange(8) * 0.05
    measured = _synthetic_array(time, i_tf, radii, np.ones(8), noise=0.0)

    window, _ = find_tf_calibration_window(time, i_tf)
    fit = fit_impa_geometry(measured, i_tf, window, pitch=0.05)

    assert fit.r0 == pytest.approx(0.37, abs=1e-4)
    np.testing.assert_allclose(fit.r, radii, atol=1e-4)
    assert fit.nrmse < 1e-6
    assert fit.monotonic and fit.within_bounds and not fit.bound_hit


def test_geometry_fit_reports_a_large_residual_for_a_non_uniform_array():
    time = _time()
    i_tf = _tf_ramp(time)
    # Real spacing far from the assumed uniform pitch.
    radii = np.array([0.37, 0.38, 0.41, 0.47, 0.55, 0.60, 0.64, 0.74])
    measured = _synthetic_array(time, i_tf, radii, np.ones(8))

    window, _ = find_tf_calibration_window(time, i_tf)
    fit = fit_impa_geometry(measured, i_tf, window, pitch=0.05)

    assert fit.nrmse > 0.1


# ---------------------------------------------------------------------------
# compensation
# ---------------------------------------------------------------------------
def test_tf_pickup_removal_round_trips_an_injected_vertical_field():
    time = _time()
    i_tf = _tf_ramp(time)
    radii = np.array([0.4, 0.5])
    alpha = np.array([0.08, -0.05])
    bz = 2.0e-3 * np.sin(2 * np.pi * 5 * time)
    measured = _synthetic_array(time, i_tf, radii, alpha, bz=bz, noise=0.0)

    pickup = alpha[:, None] * toroidal_field(radii[:, None], i_tf[None, :])
    recovered = remove_tf_pickup(measured, pickup, alpha)

    np.testing.assert_allclose(recovered, np.vstack([bz, bz]), atol=1e-12)


def test_toroidally_aligned_probe_yields_nan_rather_than_a_fabricated_value():
    measured = np.ones((1, 4))
    pickup = np.zeros((1, 4))

    recovered = remove_tf_pickup(measured, pickup, np.array([1.0]))

    assert np.all(np.isnan(recovered))


def test_calibration_applies_gain_and_baseline_in_the_legacy_order():
    time = _time(4_000)
    raw = np.vstack([np.full(time.size, 1.5), 1.5 + np.linspace(0.0, 1.0, time.size)])

    calibrated = impa_calibrate_signals(
        raw, gain=2.0 / 15.0, cutoff_hz=250.0, sample_rate=SAMPLE_RATE, baseline="first_sample"
    )

    # A constant input is entirely baseline, so it must vanish.
    assert np.allclose(calibrated[0], 0.0, atol=1e-9)
    assert calibrated[1, -1] == pytest.approx(2.0 / 15.0, rel=5e-2)


def test_unknown_baseline_is_rejected():
    with pytest.raises(ValueError, match="baseline"):
        ImpaProcessingConfig(baseline="whatever")


# ---------------------------------------------------------------------------
# validation and orchestration
# ---------------------------------------------------------------------------
def test_process_impa_reports_valid_for_a_well_behaved_synthetic_array():
    time = _time()
    i_tf = _tf_ramp(time)
    radii = 0.4 + np.arange(8) * 0.05
    alpha = np.full(8, 0.05)
    # Small compared with the TF pickup, so the linear TF model still
    # describes the measurement to within the residual tolerance.
    bz = 3.0e-5 * np.sin(2 * np.pi * 3 * time)
    # process_impa filters and gain-calibrates, so hand it volts.
    measured = _synthetic_array(time, i_tf, radii, alpha, bz=bz)
    raw = measured / (2.0 / 15.0)

    result = process_impa(
        time,
        raw,
        i_tf,
        r=radii,
        config=ImpaProcessingConfig(orientation="poloidal"),
    )

    assert result.quality.status == "valid"
    assert result.window is not None
    assert result.geometry.method == "configured"
    np.testing.assert_allclose(result.coupling.alpha, alpha, atol=1e-3)
    assert np.all(np.isfinite(result.b_z))
    assert result.provenance["reference_shot_used"] is False


def test_process_impa_reports_invalid_without_a_calibration_window():
    time = _time(2_000)
    raw = np.random.default_rng(0).normal(size=(8, time.size))

    result = process_impa(time, raw, np.zeros_like(time))

    assert result.quality.status == "invalid"
    assert result.quality.checks["calibration_window"] == "invalid"
    assert result.coupling is None
    assert result.geometry.method == "nominal_uncalibrated"


def test_a_toroidal_array_is_rejected_by_the_poloidal_model():
    """The legacy near-vertical model cannot describe a toroidal mounting."""
    time = _time()
    i_tf = _tf_ramp(time)
    radii = 0.4 + np.arange(8) * 0.05
    raw = _synthetic_array(time, i_tf, radii, np.ones(8)) / (2.0 / 15.0)

    result = process_impa(
        time, raw, i_tf, r=radii, config=ImpaProcessingConfig(orientation="poloidal")
    )

    assert result.quality.status == "invalid"
    assert result.quality.checks["tf_coupling"] == "invalid"
    assert any("tilt bounds" in reason for reason in result.quality.reasons)


def test_a_toroidal_array_is_accepted_by_the_toroidal_model():
    """The same array is exactly what a toroidal mounting should look like."""
    time = _time()
    i_tf = _tf_ramp(time)
    radii = 0.4 + np.arange(8) * 0.05
    raw = _synthetic_array(time, i_tf, radii, np.ones(8)) / (2.0 / 15.0)

    result = process_impa(time, raw, i_tf, r=radii)  # toroidal is the default

    assert result.quality.status == "valid"
    assert result.quality.checks["tf_coupling"] == "valid"
    np.testing.assert_allclose(result.coupling.alpha, 1.0, atol=1e-3)


def test_a_poloidal_array_is_rejected_by_the_toroidal_model():
    time = _time()
    i_tf = _tf_ramp(time)
    radii = 0.4 + np.arange(8) * 0.05
    raw = _synthetic_array(time, i_tf, radii, np.full(8, 0.05)) / (2.0 / 15.0)

    result = process_impa(time, raw, i_tf, r=radii)

    assert result.quality.checks["tf_coupling"] == "invalid"
    assert any("toroidally aligned" in reason for reason in result.quality.reasons)


def test_a_reference_shot_calibration_can_be_reused():
    """A reference taken with the array in place calibrates another shot."""
    time = _time()
    i_tf = _tf_ramp(time)
    radii = 0.42 + np.arange(8) * 0.05
    raw = _synthetic_array(time, i_tf, radii, np.ones(8)) / (2.0 / 15.0)
    reference = process_impa(time, raw, i_tf)

    # A shot with no clean TF interval of its own still calibrates.
    quiet = process_impa(time, raw, i_tf, ip=np.full_like(time, 50_000.0), reference=reference)

    assert quiet.window is None
    assert quiet.provenance["reference_shot_used"] is True
    assert quiet.geometry.method.startswith("reference_shot")
    np.testing.assert_allclose(quiet.geometry.r, reference.geometry.r)
    assert quiet.quality.checks["calibration_window"] == "warning"
    assert quiet.quality.status in ("valid", "warning")


def test_missing_channels_are_reported_and_never_silently_zero_filled():
    time = _time()
    i_tf = _tf_ramp(time)
    radii = 0.4 + np.arange(8) * 0.05
    raw = _synthetic_array(time, i_tf, radii, np.full(8, 0.05)) / (2.0 / 15.0)
    valid = np.ones(8, dtype=bool)
    valid[3] = False

    result = process_impa(time, raw, i_tf, r=radii, channel_valid=valid)

    assert result.quality.checks["channels_present"] == "warning"
    assert np.all(np.isnan(result.b_measured[3]))
    assert any("unavailable" in reason for reason in result.quality.reasons)


def test_validate_impa_escalates_a_dead_channel_to_invalid():
    time = _time(1_000)
    raw = np.ones((2, time.size))
    quality = validate_impa(
        time,
        raw,
        raw,
        raw,
        np.ones(2, dtype=bool),
        None,
        _nominal_geometry(),
        None,
        expected_channels=2,
    )

    assert quality.status == "invalid"
    assert quality.checks["channel_signals"] == "invalid"


def _nominal_geometry():
    from vaft.process.impa import ImpaGeometryFit

    return ImpaGeometryFit(
        r=np.array([0.4, 0.45]), z=np.zeros(2), method="configured", pitch=0.05
    )


# ---------------------------------------------------------------------------
# legacy ports
# ---------------------------------------------------------------------------
def test_legacy_position_recovers_r0_from_a_single_sample():
    time = _time()
    tf_raw = _tf_ramp(time) / -3.0e4
    radii = 0.42 + np.arange(8) * 0.05
    i_tf = _tf_ramp(time)
    measured = _synthetic_array(time, i_tf, radii, np.ones(8))
    raw = measured / (2.0 / 15.0)

    result = legacy_impa_position(time, raw, tf_raw, target_time=0.30)

    assert result["r0"] == pytest.approx(0.42, abs=2e-3)
    assert not result["bound_hit"]


def test_legacy_compensation_saturates_when_a_probe_is_toroidally_aligned():
    time = _time()
    i_tf = _tf_ramp(time)
    tf_raw = i_tf / 3.0e4
    radii = 0.4 + np.arange(2) * 0.05
    raw = _synthetic_array(time, i_tf, radii, np.ones(2)) / (-2.0 / 15.0)

    result = legacy_impa_compensation(time, raw, tf_raw, radii, target_time=0.29)

    # The legacy model can only explain the signal with a tilt far beyond its
    # own +/-10 degree bound, so every channel saturates.
    assert np.all(result["bound_hit"])


# ---------------------------------------------------------------------------
# vertical-field sensors and their toroidal crosstalk
# ---------------------------------------------------------------------------
def test_crosstalk_fit_recovers_a_known_misalignment_angle():
    """A tilted Bz sensor reads sin(theta) of the toroidal field.

    Yang et al. (Rev. Sci. Instrum. 85, 11D809) calibrate exactly this way:
    on a TF-only interval the true vertical field is negligible, so the Bz
    sensor's whole reading is toroidal bleed-through.
    """
    time = _time()
    i_tf = _tf_ramp(time)
    radii = 0.45 + np.arange(3) * 0.05
    b_toroidal = _synthetic_array(time, i_tf, radii, np.ones(3), noise=0.0)

    angles = np.array([2.0, -3.5, 0.5])
    b_z_raw = np.sin(np.radians(angles))[:, None] * b_toroidal

    window, _ = find_tf_calibration_window(time, i_tf)
    # bz_gain=1.0: the synthetic Bz "volts" are already tesla, so the slope
    # equals sin(angle) directly.
    fit = fit_impa_crosstalk(b_toroidal, b_z_raw, window, bz_gain=1.0)

    np.testing.assert_allclose(fit.slope_v_per_t, np.sin(np.radians(angles)), atol=1e-6)
    np.testing.assert_allclose(fit.angle_deg, angles, atol=1e-6)
    assert np.all(fit.r_squared > 0.999)
    assert not np.any(fit.bound_hit)


def test_crosstalk_removal_recovers_the_vertical_field_underneath():
    """Calibrate on the TF-only phase, then correct the phase with plasma.

    The correction is only unbiased if the calibration window really is free
    of vertical field -- fitting it against an interval that already contains
    the signal would absorb the signal into the crosstalk slope.
    """
    time = _time()
    i_tf = _tf_ramp(time)
    radii = 0.45 + np.arange(2) * 0.05
    b_toroidal = _synthetic_array(time, i_tf, radii, np.ones(2), noise=0.0)

    # A plasma exists only after 0.40 s, so the clean window sits before it.
    plasma = time >= 0.40
    ip = np.where(plasma, 50_000.0, 0.0)
    true_bz = np.zeros((2, time.size))
    true_bz[0, plasma] = 1.5e-3 * np.sin(2 * np.pi * 4 * time[plasma])
    true_bz[1, plasma] = -1.0e-3 * np.sin(2 * np.pi * 4 * time[plasma])

    angles = np.array([2.0, -3.0])
    b_z_raw = np.sin(np.radians(angles))[:, None] * b_toroidal + true_bz

    window, _ = find_tf_calibration_window(time, i_tf, ip)
    assert window.end_time <= 0.40
    fit = fit_impa_crosstalk(b_toroidal, b_z_raw, window, bz_gain=1.0)
    np.testing.assert_allclose(fit.angle_deg, angles, atol=1e-6)

    # Removal itself needs no gain -- it works in the sensor's native volts.
    fit_no_gain = fit_impa_crosstalk(b_toroidal, b_z_raw, window)
    assert np.all(np.isnan(fit_no_gain.angle_deg))
    recovered = remove_bz_crosstalk(b_z_raw, b_toroidal, fit_no_gain)
    np.testing.assert_allclose(recovered[:, plasma], true_bz[:, plasma], atol=1e-9)


def test_an_inactive_bz_sensor_is_flagged_rather_than_trusted():
    """A disconnected sensor shows no toroidal bleed at all."""
    time = _time()
    i_tf = _tf_ramp(time)
    radii = 0.45 + np.arange(2) * 0.05
    raw = _synthetic_array(time, i_tf, radii, np.ones(2)) / (2.0 / 15.0)
    dead = np.random.default_rng(1).normal(0.0, 1e-5, (2, time.size))

    result = process_impa(time, raw, i_tf, r=radii, b_z_raw=dead)

    assert result.quality.checks["bz_sensors"] == "warning"
    assert any("may be inactive" in reason for reason in result.quality.reasons)
    # No angle is reported without a configured Bz gain; the sensor's own
    # calibration is not established.
    assert np.all(np.isnan(result.crosstalk.angle_deg))
    assert np.all(np.isfinite(result.crosstalk.slope_v_per_t))


def test_incident_angle_is_recovered_from_the_rigid_pitch():
    """A probe pushed in at an angle projects its 5 cm spacing onto less."""
    time = _time()
    i_tf = _tf_ramp(time)
    incident = 30.0
    projected = 0.05 * np.cos(np.radians(incident))
    radii = 0.45 + np.arange(8) * projected
    raw = _synthetic_array(time, i_tf, radii, np.ones(8), noise=0.0) / (2.0 / 15.0)

    result = process_impa(time, raw, i_tf, pitch=0.05, fit_pitch=True)

    assert result.geometry.pitch == pytest.approx(projected, abs=1e-4)
    assert result.geometry.incident_angle_deg == pytest.approx(incident, abs=0.5)
    assert result.geometry.nominal_pitch == 0.05


def test_a_radial_insertion_reports_no_incident_angle():
    time = _time()
    i_tf = _tf_ramp(time)
    radii = 0.45 + np.arange(8) * 0.05
    raw = _synthetic_array(time, i_tf, radii, np.ones(8), noise=0.0) / (2.0 / 15.0)

    result = process_impa(time, raw, i_tf, pitch=0.05, fit_pitch=True)

    assert result.geometry.incident_angle_deg == pytest.approx(0.0, abs=1.0)


def test_geometry_fit_is_invariant_to_the_configured_gain_sign():
    """R0 must not depend on whether the Hall gain convention is +2/15 or -2/15.

    Confirmed by the DAQ wiring datasheet: the array's real Hall gain is
    negative (-2/15), opposite the sign this module used to assume by default.
    The geometry fit must detect the working polarity from the data rather
    than hard-coding alpha=+1.
    """
    time = _time()
    i_tf = _tf_ramp(time)
    radii = 0.4 + np.arange(8) * 0.05
    positive = _synthetic_array(time, i_tf, radii, np.ones(8), noise=0.0)
    negative = -positive

    window, _ = find_tf_calibration_window(time, i_tf)
    fit_pos = fit_impa_geometry(positive, i_tf, window, pitch=0.05)
    fit_neg = fit_impa_geometry(negative, i_tf, window, pitch=0.05)

    assert fit_pos.r0 == pytest.approx(fit_neg.r0, abs=1e-6)
    assert fit_pos.nrmse < 1e-6
    assert fit_neg.nrmse < 1e-6


def test_process_impa_rejects_a_reference_with_a_different_channel_count():
    """A caller that bypasses process_impa_shot's alignment gets a clear error.

    process_impa() has no field-code metadata to align by, so a mismatched
    reference must raise before broadcasting two differently-shaped arrays
    against each other -- not fail deep inside remove_tf_pickup with a
    confusing shape-mismatch traceback.
    """
    time = _time()
    i_tf = _tf_ramp(time)
    reference_radii = 0.4 + np.arange(7) * 0.05
    reference_raw = _synthetic_array(time, i_tf, reference_radii, np.ones(7)) / (2.0 / 15.0)
    reference = process_impa(time, reference_raw, i_tf, r=reference_radii)

    target_radii = 0.4 + np.arange(8) * 0.05
    target_raw = _synthetic_array(time, i_tf, target_radii, np.ones(8)) / (2.0 / 15.0)

    with pytest.raises(ValueError, match="7 channels but this shot has 8"):
        process_impa(time, target_raw, i_tf, reference=reference)
