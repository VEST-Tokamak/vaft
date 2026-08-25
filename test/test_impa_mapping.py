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
    assert window.metrics["tf_dynamic_range"] > 0.5
    assert window.n_samples > 1_000
    assert window.metrics["ip_peak"] <= 10_000.0


# ---------------------------------------------------------------------------
# the calibration verdict
# ---------------------------------------------------------------------------
def test_39204_self_calibration_is_rejected_rather_than_silently_accepted(result_39204):
    """The probes are toroidally aligned, so the legacy tilt model cannot hold.

    ``alpha`` near unity means a probe measures essentially the whole toroidal
    field; projecting that back onto the vertical axis is unbounded, so the
    shot must be reported as invalid instead of yielding a plausible number.
    """
    assert result_39204.quality.status == "invalid"
    assert result_39204.quality.checks["tf_coupling"] == "invalid"
    assert any("tilt bounds" in reason for reason in result_39204.quality.reasons)

    alpha = result_39204.coupling.alpha
    assert np.nanmax(alpha) > 0.9
    # Nothing physical is emitted for the saturated channels.
    assert not np.all(np.isfinite(result_39204.b_z))


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


def test_uniform_pitch_geometry_does_not_describe_this_array(result_39204):
    assert result_39204.geometry.method == "tf_profile_fit"
    assert result_39204.geometry.r0 == pytest.approx(0.341, abs=0.02)
    # A ~30% residual is why the fitted geometry is reported with its quality
    # rather than trusted.
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
    before = len(ods["magnetics.b_field_pol_probe"])
    names_before = [ods[f"magnetics.b_field_pol_probe.{i}.name"] for i in range(before)]

    impa(ods, 39204, raw_source=IMPA_SAMPLE)

    indices = impa_probe_indices(ods)
    assert indices == list(range(before, before + len(IMPA_FIELDS)))
    assert [ods[f"magnetics.b_field_pol_probe.{i}.name"] for i in range(before)] == names_before


def test_mapped_impa_probes_carry_hall_metadata_and_geometry():
    ods = ODS(consistency_check=False)
    status = impa(ods, 39204, raw_source=IMPA_SAMPLE)

    indices = impa_probe_indices(ods)
    assert len(indices) == len(IMPA_FIELDS)
    for index in indices:
        prefix = f"magnetics.b_field_pol_probe.{index}"
        assert ods[f"{prefix}.type.index"] == HALL_PROBE_TYPE_INDEX
        assert ods[f"{prefix}.type.name"] == "hall"
        assert str(ods[f"{prefix}.identifier"]).startswith(IMPA_IDENTIFIER_PREFIX)
        assert ods[f"{prefix}.position.z"] == pytest.approx(0.0)
        assert 0.1 <= ods[f"{prefix}.position.r"] <= 0.8
        assert ods[f"{prefix}.voltage.data"].size > 1_000

    radii = [ods[f"magnetics.b_field_pol_probe.{i}.position.r"] for i in indices]
    assert radii == sorted(radii)
    assert status["provenance"]["reference_shot_used"] is False


def test_a_rejected_channel_gets_no_field_waveform_at_all():
    ods = ODS(consistency_check=False)
    status = impa(ods, 39204, raw_source=IMPA_SAMPLE)

    assert status["status"] == "invalid"
    for index in impa_probe_indices(ods):
        prefix = f"magnetics.b_field_pol_probe.{index}"
        assert ods[f"{prefix}.field.validity"] < 0
        # A zero-filled trace would be indistinguishable from a measurement.
        assert f"{prefix}.field.data" not in ods


def test_status_records_the_calibration_provenance():
    ods = ODS(consistency_check=False)
    status = impa(ods, 39204, raw_source=IMPA_SAMPLE)

    window = status["provenance"]["calibration_window"]
    assert window["n_samples"] > 1_000
    assert window["tf_dynamic_range"] > 0.5
    assert status["provenance"]["sign_convention"].startswith("I_TF = raw * -3e4")
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
# cross-shot stability
# ---------------------------------------------------------------------------
def _archived_impa_shots() -> list[int]:
    """Every archived reduced IMPA sample, by shot number."""
    directory = REPOSITORY_ROOT / "vaft" / "data" / "sample" / "legacy"
    shots = []
    for path in sorted(directory.glob("shot_*_impa.json.gz")):
        try:
            shots.append(int(path.name.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return shots


@pytest.mark.skipif(
    len(_archived_impa_shots()) < 2,
    reason="cross-shot stability needs at least two archived IMPA samples",
)
def test_fitted_geometry_and_coupling_are_stable_across_shots_of_one_era():
    """Probe geometry is hardware, so it must not move shot to shot.

    The check activates as soon as a second reduced IMPA sample is added under
    ``vaft/data/sample/legacy``.
    """
    radii, ratios = [], []
    for shot in _archived_impa_shots():
        result, _ = process_impa_shot(shot, raw_source=IMPA_SAMPLE)
        assert result.window is not None
        radii.append(result.geometry.r)
        ratios.append(result.coupling.coupling_ratio)

    radii = np.vstack(radii)
    ratios = np.vstack(ratios)
    # Fitted positions within one hardware era should agree to a centimetre,
    # and the observable coupling ratio to a few percent.
    assert np.max(np.ptp(radii, axis=0)) < 0.01
    assert np.max(np.ptp(ratios, axis=0) / np.mean(np.abs(ratios), axis=0)) < 0.05
