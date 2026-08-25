"""IMPA mapping tests against archived VEST raw data.

Shot 39204 is the reference IMPA case: `Run_EFITlikeProfileFitting.m` in the
legacy VFIT repository keeps separate tuning for "39204 with impa".  Shot 44740
is the negative control -- fields 114-121 exist but carry no IMPA signal.
"""

from pathlib import Path

import numpy as np
import pytest
from omas import ODS

from vaft.database import raw as raw_db
from vaft.machine_mapping.impa import (
    HALL_PROBE_TYPE_INDEX,
    IMPA_IDENTIFIER_PREFIX,
    impa,
    impa_probe_indices,
    impa_probe_node,
    load_impa_inputs,
    process_impa_shot,
    resolve_impa_config,
)
from vaft.machine_mapping.magnetics import vfit_magnetics_static
from vaft.process.impa import (
    impa_calibrate_signals,
    legacy_impa_compensation,
    legacy_impa_position,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
IMPA_SAMPLE = str(REPOSITORY_ROOT / "vaft" / "data" / "sample" / "legacy" / "shot_{shot}_impa.json.gz")
NO_SIGNAL_SAMPLE = str(REPOSITORY_ROOT / "vaft" / "data" / "legacy" / "shot_{shot}.json.gz")

pytestmark = pytest.mark.skipif(
    not Path(IMPA_SAMPLE.format(shot=39204)).is_file(),
    reason="archived IMPA sample for shot 39204 is not available",
)

IMPA_FIELDS = (114, 115, 116, 117, 118, 119, 120, 121)

#: 2022-04-23 IMPA alignment block: the verified reference condition.  Seven
#: channels are wired and the rigid 5 cm array fits the TF 1/R profile.
REFERENCE_SHOT = 35376
REFERENCE_CHANNELS = 7


@pytest.fixture(scope="module")
def inputs_39204():
    return load_impa_inputs(39204, resolve_impa_config(39204), raw_source=IMPA_SAMPLE)


@pytest.fixture(scope="module")
def result_39204():
    result, _ = process_impa_shot(39204, raw_source=IMPA_SAMPLE)
    return result


# ---------------------------------------------------------------------------
# configuration and raw data
# ---------------------------------------------------------------------------
def test_configuration_lists_the_eight_canonical_raw_fields():
    config = resolve_impa_config(39204)

    fields = [int(channel["field"]) for channel in config["channels"].values()]
    assert fields == list(IMPA_FIELDS)
    assert config["calibration"]["gain"] == pytest.approx(2.0 / 15.0)
    assert config["tf_compensation"]["tf_turns"] == 24


def test_all_eight_raw_channels_and_the_tf_are_present_and_finite(inputs_39204):
    assert inputs_39204["raw"].shape[0] == len(IMPA_FIELDS)
    assert np.all(inputs_39204["channel_valid"])
    assert np.all(np.isfinite(inputs_39204["raw"]))
    assert np.all(np.diff(inputs_39204["time"]) > 0)
    assert np.max(np.abs(inputs_39204["i_tf"])) > 10_000.0
    assert inputs_39204["ip"] is not None
    assert inputs_39204["pf_currents"] is not None


def test_raw_impa_channels_decrease_monotonically_with_channel_index(inputs_39204):
    """Channel 1 sits innermost, so its TF pickup must be the largest."""
    calibrated = impa_calibrate_signals(
        inputs_39204["raw"], gain=2.0 / 15.0, cutoff_hz=250.0, sample_rate=25_000.0
    )
    index = int(np.argmax(np.abs(inputs_39204["i_tf"])))
    profile = calibrated[:, index]

    assert np.all(profile > 0)
    assert np.all(np.diff(profile) < 0)


# ---------------------------------------------------------------------------
# clean-window selection replaces the fixed legacy time
# ---------------------------------------------------------------------------
def test_calibration_window_is_found_from_signal_criteria_not_a_fixed_time(result_39204):
    window = result_39204.window

    assert window is not None
    # The legacy routines calibrated at 0.29-0.30 s, where the TF is flat.  The
    # automatic selection instead lands on the ramp, which actually constrains
    # a slope.
    assert not (window.start_time <= 0.29 <= window.end_time and window.metrics["tf_dynamic_range"] < 0.1)
    assert window.metrics["tf_dynamic_range"] >= 0.45
    assert window.n_samples > 1_000
    assert window.metrics["ip_peak"] <= 10_000.0


# ---------------------------------------------------------------------------
# the calibration verdict
# ---------------------------------------------------------------------------
def test_39204_does_not_meet_the_reference_conditions(result_39204):
    """39204 fails the rigid-array and alignment checks, and says why.

    Its two innermost channels return couplings far from unity and the rigid
    5 cm fit leaves a ~34% residual, against 9-12% on the 2022-04-23 alignment
    shots.  The shot is reported invalid rather than silently accepted.
    """
    assert result_39204.quality.status == "invalid"
    assert result_39204.quality.checks["tf_coupling"] == "invalid"
    assert any("toroidally aligned" in reason for reason in result_39204.quality.reasons)
    assert result_39204.geometry.nrmse > 0.2

    alpha = result_39204.coupling.alpha
    assert np.nanmax(alpha) > 0.9        # most channels do face the TF
    assert np.nanmin(alpha) < 0.5        # but the innermost ones do not


def test_only_the_ratio_of_coupling_to_radius_is_constrained(result_39204, inputs_39204):
    """A TF-only window fixes alpha/R, never alpha and R independently.

    Refitting against a deliberately shifted geometry moves every fitted
    coupling but leaves alpha/R untouched, which is why configured shot-era
    geometry always takes precedence over self-calibration.
    """
    from vaft.process.impa import fit_impa_tf_coupling

    baseline = result_39204.coupling
    np.testing.assert_allclose(
        baseline.coupling_ratio, baseline.alpha / result_39204.geometry.r, rtol=1e-9
    )

    shifted = fit_impa_tf_coupling(
        result_39204.b_measured,
        inputs_39204["i_tf"],
        result_39204.geometry.r + 0.1,
        result_39204.window,
    )

    assert np.all(np.abs(shifted.alpha - baseline.alpha) > 0.05)
    np.testing.assert_allclose(shifted.coupling_ratio, baseline.coupling_ratio, rtol=1e-6)


def test_uniform_pitch_geometry_does_not_describe_39204(result_39204):
    assert result_39204.geometry.method == "tf_profile_fit"
    assert result_39204.geometry.r0 == pytest.approx(0.341, abs=0.02)
    assert result_39204.geometry.nrmse > 0.2
    assert result_39204.quality.checks["geometry"] == "warning"


# ---------------------------------------------------------------------------
# legacy parity
# ---------------------------------------------------------------------------
def test_legacy_position_reproduces_its_documented_single_sample_fit(inputs_39204):
    tf_raw = raw_db.vest_load(39204, 1, sample_opt=IMPA_SAMPLE)[1]

    legacy = legacy_impa_position(inputs_39204["time"], inputs_39204["raw"], tf_raw)

    assert legacy["time"] == pytest.approx(0.30, abs=1e-4)
    assert legacy["r0"] == pytest.approx(0.343785, abs=1e-4)
    np.testing.assert_allclose(legacy["r"], legacy["r0"] + np.arange(8) * 0.05, atol=1e-9)
    assert not legacy["bound_hit"]


def test_legacy_tilt_fit_saturates_on_every_channel_for_this_shot(inputs_39204):
    """The legacy +/-10 degree tilt model cannot explain a toroidal probe.

    Saturating at the bound yields a "compensated Bz" of order 0.1 T, which is
    two orders of magnitude larger than VEST's vertical field -- the concrete
    reason this implementation gates on bound saturation.
    """
    tf_raw = raw_db.vest_load(39204, 1, sample_opt=IMPA_SAMPLE)[1]
    position = legacy_impa_position(inputs_39204["time"], inputs_39204["raw"], tf_raw)

    legacy = legacy_impa_compensation(
        inputs_39204["time"], inputs_39204["raw"], tf_raw, position["r"]
    )

    assert np.all(legacy["bound_hit"])
    np.testing.assert_allclose(legacy["tilt_deg"], 10.0, atol=1e-3)
    target = legacy["b_z"][:, legacy["index"]]
    assert np.all(np.abs(target) > 0.05)


def test_the_two_legacy_sign_conventions_differ_only_in_polarity(inputs_39204):
    """`VEST_IMPAProcessing` used -2/15 with +3e4, `vest_impa_position` +2/15 with -3e4."""
    raw = inputs_39204["raw"]
    processing = impa_calibrate_signals(raw, gain=-2.0 / 15.0, cutoff_hz=250.0, sample_rate=25_000.0)
    position = impa_calibrate_signals(raw, gain=+2.0 / 15.0, cutoff_hz=250.0, sample_rate=25_000.0)

    np.testing.assert_allclose(processing, -position, atol=1e-12)

    tf_raw = np.asarray(raw_db.vest_load(39204, 1, sample_opt=IMPA_SAMPLE)[1], dtype=float)
    np.testing.assert_allclose(tf_raw * 3.0e4, -(tf_raw * -3.0e4), atol=1e-9)


# ---------------------------------------------------------------------------
# ODS mapping
# ---------------------------------------------------------------------------
def test_impa_channels_are_appended_without_moving_existing_probe_indices():
    ods = ODS(consistency_check=False)
    vfit_magnetics_static(ods)
    pol_before = len(ods["magnetics.b_field_pol_probe"])
    names_before = [ods[f"magnetics.b_field_pol_probe.{i}.name"] for i in range(pol_before)]

    impa(ods, REFERENCE_SHOT, raw_source=IMPA_SAMPLE)

    # A toroidally mounted array lands in b_field_tor_probe, so the existing
    # poloidal probes are untouched.
    assert len(ods["magnetics.b_field_pol_probe"]) == pol_before
    assert [ods[f"magnetics.b_field_pol_probe.{i}.name"] for i in range(pol_before)] == names_before
    assert impa_probe_node(ods) == "magnetics.b_field_tor_probe"
    assert impa_probe_indices(ods) == list(range(REFERENCE_CHANNELS))


def test_mapped_impa_probes_carry_hall_metadata_and_geometry():
    ods = ODS(consistency_check=False)
    status = impa(ods, REFERENCE_SHOT, 0.10, 0.60, 4.0e-5, raw_source=IMPA_SAMPLE)

    node = status["ids_node"]
    indices = impa_probe_indices(ods)
    assert status["orientation"] == "toroidal"
    assert len(indices) == REFERENCE_CHANNELS
    for index in indices:
        prefix = f"{node}.{index}"
        assert ods[f"{prefix}.type.index"] == HALL_PROBE_TYPE_INDEX
        assert ods[f"{prefix}.type.name"] == "hall"
        assert str(ods[f"{prefix}.identifier"]).startswith(IMPA_IDENTIFIER_PREFIX)
        assert ods[f"{prefix}.position.z"] == pytest.approx(0.0)
        assert 0.1 <= ods[f"{prefix}.position.r"] <= 0.9
        assert ods[f"{prefix}.voltage.data"].size > 1_000

    radii = [ods[f"{node}.{i}.position.r"] for i in indices]
    assert radii == sorted(radii)
    # The array is rigid: the configured pitch must survive the fit.
    np.testing.assert_allclose(np.diff(radii), 0.05, atol=1e-9)
    assert status["provenance"]["reference_shot_used"] is False


def test_a_valid_shot_gets_a_field_waveform_and_a_rejected_one_does_not():
    good = ODS(consistency_check=False)
    good_status = impa(good, REFERENCE_SHOT, 0.10, 0.60, 4.0e-5, raw_source=IMPA_SAMPLE)
    assert good_status["status"] == "valid"
    for index in impa_probe_indices(good):
        prefix = f"{good_status['ids_node']}.{index}"
        assert good[f"{prefix}.field.validity"] >= 0
        assert good[f"{prefix}.field.data"].size > 1

    bad = ODS(consistency_check=False)
    bad_status = impa(bad, 39204, raw_source=IMPA_SAMPLE)
    assert bad_status["status"] == "invalid"
    for index in impa_probe_indices(bad):
        prefix = f"{bad_status['ids_node']}.{index}"
        assert bad[f"{prefix}.field.validity"] < 0
        # A zero-filled trace would be indistinguishable from a measurement.
        assert f"{prefix}.field.data" not in bad


def test_status_records_the_calibration_provenance():
    ods = ODS(consistency_check=False)
    status = impa(ods, REFERENCE_SHOT, 0.10, 0.60, 4.0e-5, raw_source=IMPA_SAMPLE)

    window = status["provenance"]["calibration_window"]
    assert window["n_samples"] > 1_000
    assert status["provenance"]["orientation"] == "toroidal"
    assert status["geometry_method"] == "tf_profile_fit"
    for channel in status["channels"].values():
        assert channel["field"] in IMPA_FIELDS
        assert np.isfinite(channel["implied_radius_alpha_unity"])


# ---------------------------------------------------------------------------
# negative control
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not Path(NO_SIGNAL_SAMPLE.format(shot=44740)).is_file(),
    reason="archived raw dump for shot 44740 is not available",
)
def test_a_shot_without_a_real_impa_signal_is_reported_invalid():
    """Shot 44740 has fields 114-121 wired but no IMPA installed."""
    result, _ = process_impa_shot(44740, raw_source=NO_SIGNAL_SAMPLE)

    assert result.quality.status == "invalid"
    assert result.quality.reasons


# ---------------------------------------------------------------------------
# cross-shot comparison
# ---------------------------------------------------------------------------
SECOND_REFERENCE = 39923


@pytest.mark.skipif(
    not Path(IMPA_SAMPLE.format(shot=SECOND_REFERENCE)).is_file(),
    reason=f"archived IMPA sample for shot {SECOND_REFERENCE} is not available",
)
def test_a_low_tf_shot_still_yields_a_calibration_window():
    """Shot 39923 peaks at 1.3 kA against 39204's 12.7 kA.

    The clean-window criterion is a fraction of each shot's own TF peak, so a
    legitimately weak shot is not silently rejected by a threshold tuned on a
    strong one.
    """
    result, inputs = process_impa_shot(SECOND_REFERENCE, raw_source=IMPA_SAMPLE)

    assert np.max(np.abs(inputs["i_tf"])) < 5_000.0  # would fail a fixed 5 kA cut
    assert result.window is not None
    assert result.window.metrics["tf_current_threshold"] < 1_000.0
    assert result.window.metrics["tf_dynamic_range"] > 0.4


@pytest.mark.skipif(
    not Path(IMPA_SAMPLE.format(shot=SECOND_REFERENCE)).is_file(),
    reason=f"archived IMPA sample for shot {SECOND_REFERENCE} is not available",
)
def test_channel_ordering_is_reproducible_across_both_reference_shots():
    """Wiring order is hardware, and it does reproduce: R increases with index."""
    for shot in (39204, SECOND_REFERENCE):
        result, inputs = process_impa_shot(shot, raw_source=IMPA_SAMPLE)
        peak = int(np.argmax(np.abs(inputs["i_tf"])))
        profile = result.b_measured[:, peak]

        assert np.all(profile > 0), shot
        assert np.all(np.diff(profile) < 0), shot


@pytest.mark.skipif(
    not Path(IMPA_SAMPLE.format(shot=SECOND_REFERENCE)).is_file(),
    reason=f"archived IMPA sample for shot {SECOND_REFERENCE} is not available",
)
def test_geometry_is_a_per_shot_quantity_for_this_insertable_array():
    """The array is inserted per shot, so fitted positions legitimately differ.

    Cross-shot agreement is therefore *not* a correctness criterion: the probe
    really did move between 39204 and 39923.  Under the ``alpha = 1`` reading
    the implied radii sit roughly 0.16-0.27 m further out on 39923, broadly a
    radial translation.  This test records that the difference is real and
    ordered, so nobody later mistakes it for a calibration bug.
    """
    implied, r0 = {}, {}
    for shot in (39204, SECOND_REFERENCE):
        result, inputs = process_impa_shot(shot, raw_source=IMPA_SAMPLE)
        peak = int(np.argmax(np.abs(inputs["i_tf"])))
        # Response per unit TF drive, free of any assumed radius; its
        # reciprocal is the radius implied by a toroidally aligned probe.
        kappa = result.b_measured[:, peak] / (2.0e-7 * 24 * inputs["i_tf"][peak])
        implied[shot] = 1.0 / kappa
        r0[shot] = result.geometry.r0

    shift = implied[SECOND_REFERENCE] - implied[39204]
    assert np.all(shift > 0.1)          # moved outward on every channel
    assert np.ptp(shift) < 0.2          # by a broadly similar amount
    assert r0[SECOND_REFERENCE] > r0[39204]


@pytest.mark.skipif(
    not Path(IMPA_SAMPLE.format(shot=SECOND_REFERENCE)).is_file(),
    reason=f"archived IMPA sample for shot {SECOND_REFERENCE} is not available",
)
def test_no_static_model_certifies_a_single_shot_yet():
    """Both reference shots are classified invalid, for reasons they state.

    The open problem is within one shot, not between shots: a static
    ``alpha_i * Bt(R_i) + beta_i`` model does not describe a whole clean-TF
    window, so neither shot can be certified.  The pipeline must say so rather
    than emit a plausible-looking waveform.
    """
    for shot in (39204, SECOND_REFERENCE):
        result, _ = process_impa_shot(shot, raw_source=IMPA_SAMPLE)

        assert result.quality.status == "invalid", shot
        assert result.quality.reasons, shot
        assert result.window is not None, shot


# ---------------------------------------------------------------------------
# the verified reference condition (2022-04-23 alignment block)
# ---------------------------------------------------------------------------
ALIGNMENT_SHOTS = (35376, 35385)
PROBE_OUT_SHOT = 35325


@pytest.mark.skipif(
    not all(Path(IMPA_SAMPLE.format(shot=s)).is_file() for s in ALIGNMENT_SHOTS),
    reason="archived 2022-04-23 IMPA alignment samples are not available",
)
@pytest.mark.parametrize("shot", ALIGNMENT_SHOTS)
def test_alignment_shots_classify_as_valid(shot):
    """The reference condition: a rigid 5 cm array facing the toroidal field.

    On these shots the fitted coupling is unity to within a few percent and the
    rigid-array fit lands well inside tolerance, so the shot is certified.
    """
    result, _ = process_impa_shot(shot, raw_source=IMPA_SAMPLE)

    assert result.quality.status == "valid", result.quality.reasons
    assert result.geometry.n_channels_fitted == REFERENCE_CHANNELS
    assert 0.45 <= result.geometry.r0 <= 0.52
    assert result.geometry.nrmse < 0.15
    np.testing.assert_allclose(np.abs(result.coupling.alpha[:REFERENCE_CHANNELS]), 1.0, atol=0.05)


@pytest.mark.skipif(
    not all(Path(IMPA_SAMPLE.format(shot=s)).is_file() for s in ALIGNMENT_SHOTS),
    reason="archived 2022-04-23 IMPA alignment samples are not available",
)
def test_the_rigid_five_centimetre_pitch_holds_on_the_alignment_shots():
    """Spacing is hardware; the fit must not need to bend it."""
    for shot in ALIGNMENT_SHOTS:
        result, _ = process_impa_shot(shot, raw_source=IMPA_SAMPLE)
        wired = result.geometry.r[result.channel_valid]
        np.testing.assert_allclose(np.diff(wired), 0.05, atol=1e-9)


@pytest.mark.skipif(
    not Path(IMPA_SAMPLE.format(shot=PROBE_OUT_SHOT)).is_file(),
    reason="archived TF-reference sample is not available",
)
def test_a_shot_with_the_probe_withdrawn_is_rejected():
    """35325 is a TF reference taken without the array inserted."""
    result, _ = process_impa_shot(PROBE_OUT_SHOT, raw_source=IMPA_SAMPLE)

    assert result.quality.status == "invalid"
    assert np.nanmedian(np.abs(result.coupling.alpha)) < 0.1
    assert any("toroidally aligned" in reason for reason in result.quality.reasons)


@pytest.mark.skipif(
    not all(Path(IMPA_SAMPLE.format(shot=s)).is_file() for s in ALIGNMENT_SHOTS),
    reason="archived 2022-04-23 IMPA alignment samples are not available",
)
def test_a_reference_shot_calibration_transfers_to_another_shot():
    """The legacy two-shot arrangement, kept optional.

    Calibrating on one alignment shot and applying it to another reproduces
    that shot's own geometry, because the array did not move between them.
    """
    own, _ = process_impa_shot(35385, raw_source=IMPA_SAMPLE)
    borrowed, _ = process_impa_shot(35385, raw_source=IMPA_SAMPLE, reference_shot=35376)

    assert borrowed.provenance["reference_shot_used"] is True
    assert borrowed.geometry.method == "reference_shot:35376"
    assert borrowed.quality.status == "valid"
    assert abs(borrowed.geometry.r0 - own.geometry.r0) < 0.02


@pytest.mark.skipif(
    not Path(IMPA_SAMPLE.format(shot=PROBE_OUT_SHOT)).is_file(),
    reason="archived TF-reference sample is not available",
)
def test_an_unusable_reference_shot_is_refused():
    """A reference with the probe withdrawn must not calibrate anything."""
    with pytest.raises(ValueError, match="not usable as a calibration"):
        process_impa_shot(35376, raw_source=IMPA_SAMPLE, reference_shot=PROBE_OUT_SHOT)
