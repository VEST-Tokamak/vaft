"""Magnetics signal-quality validation and its downstream consumers (issue #189).

Every synthetic case is built so the *right* answer is known by construction:
a waveform is assembled from a clean signal plus one named defect, and the test
asserts that exactly that defect is found, over exactly the samples it occupies.

The separation being defended is that a channel is usable *over an interval*,
not for a shot.  Most of these tests therefore check where the verdict changes,
not merely that it changed.
"""

from __future__ import annotations

import numpy as np
import pytest
from omas import ODS

from vaft.code.efit.kfile import apply_validity_exclusions
from vaft.omas.vacuum_magnetics import select_vacuum_channels
from vaft.validation.imas import (
    VALIDITY_CERTIFIED,
    VALIDITY_INVALID,
    VALIDITY_VALID,
    read_validity,
    read_validity_timed,
)
from vaft.validation.magnetics import (
    implausible_magnitude,
    population_peak_outliers,
    population_peak_ratios,
    MagneticsQualityConfig,
    channel_node,
    constant_runs,
    magnetics_quality_metrics,
    offset_jump_samples,
    project_validity,
    spike_intervals,
    unusable_channels_at,
    validate_magnetics_signals,
)
from vaft.validation.model import ValidationStatus

N_TIME = 500
TIME = np.linspace(0.26, 0.36, N_TIME)


def _clean(scale: float = 1.0e-2, seed: int = 20260901) -> np.ndarray:
    """A plausible probe waveform: a smooth swing plus a little noise.

    Noise matters -- a noiseless signal has no scale for a robust detector to
    measure against, and would make every test a statement about float error.
    """
    rng = np.random.default_rng(seed)
    signal = scale * np.sin(np.linspace(0.0, 3.0, N_TIME))
    return signal + 0.002 * scale * rng.standard_normal(N_TIME)


def _ods(
    probes: dict[int, np.ndarray] | None = None,
    loops: dict[int, np.ndarray] | None = None,
) -> ODS:
    """A magnetics IDS carrying the given waveforms and nothing else."""
    ods = ODS(consistency_check=False)
    ods["magnetics.time"] = TIME
    for index, data in (probes or {}).items():
        base = f"magnetics.b_field_pol_probe.{index}"
        ods[f"{base}.name"] = f"probe{index}"
        ods[f"{base}.position.r"] = 0.05 + 0.02 * index
        ods[f"{base}.position.z"] = 0.1
        ods[f"{base}.field.data"] = np.asarray(data, dtype=float)
    for index, data in (loops or {}).items():
        base = f"magnetics.flux_loop.{index}"
        ods[f"{base}.name"] = f"loop{index}"
        ods[f"{base}.position.0.r"] = 0.10 + 0.02 * index
        ods[f"{base}.position.0.z"] = 0.3
        ods[f"{base}.flux.data"] = np.asarray(data, dtype=float)
    return ods


def _only(ods: ODS, **kwargs) -> "object":
    report = validate_magnetics_signals(ods, **kwargs)
    assert len(report) == 1
    return report[0]


def _reasons(quality) -> set[str]:
    return {event.reason for event in quality.events}


# ---------------------------------------------------------------------------
# 1-7: one waveform, one known defect
# ---------------------------------------------------------------------------

def test_a_healthy_waveform_is_left_alone():
    quality = _only(_ods(probes={0: _clean()}))

    assert quality.status is ValidationStatus.PASS
    assert quality.validity == VALIDITY_VALID
    assert quality.valid_fraction == 1.0
    assert quality.events == ()


def test_an_all_zero_channel_is_a_flatline_not_a_quiet_signal():
    quality = _only(_ods(probes={0: np.zeros(N_TIME)}))

    assert quality.status is ValidationStatus.FAIL
    assert _reasons(quality) == {"flatline"}
    assert quality.valid_fraction == 0.0
    assert (quality.validity_timed == VALIDITY_INVALID).all()


def test_non_finite_samples_are_invalid_exactly_where_they_are():
    data = _clean()
    data[100:104] = np.nan
    quality = _only(_ods(probes={0: data}))

    assert "non_finite" in _reasons(quality)
    assert (quality.validity_timed[100:104] == VALIDITY_INVALID).all()
    assert (quality.validity_timed[:100] >= VALIDITY_VALID).all()
    assert (quality.validity_timed[104:] >= VALIDITY_VALID).all()
    assert quality.metrics["finite_fraction"] == pytest.approx(1.0 - 4 / N_TIME)


def test_saturation_partway_through_keeps_the_earlier_samples():
    """The case a shot-level boolean cannot express: the channel is good until
    the amplifier rails, and everything before that is still a measurement.
    """
    data = _clean()
    data[300:] = data[:300].max()
    quality = _only(_ods(probes={0: data}))

    assert quality.status is ValidationStatus.FAIL
    assert "held_tail" in _reasons(quality)
    assert (quality.validity_timed[:300] >= VALIDITY_VALID).all()
    assert (quality.validity_timed[300:] == VALIDITY_INVALID).all()
    assert quality.valid_fraction == pytest.approx(0.6)
    assert quality.metrics["last_valid_time"] == pytest.approx(TIME[299])


def test_an_interior_rail_is_named_saturation_and_a_mid_range_hold_a_dropout():
    """Both are unusable; the distinction says which fault to go looking for."""
    railed = _clean()
    railed[200:260] = railed.max()
    assert "saturation" in _reasons(_only(_ods(probes={0: railed})))

    stuck = _clean()
    stuck[200:260] = float(np.median(stuck))
    assert "dropout" in _reasons(_only(_ods(probes={0: stuck})))


def test_an_isolated_spike_is_flagged_but_never_rejects_the_channel():
    """A threshold that is not yet justified on a VEST population must not
    remove data, so a spike is a warning: the sample stays selectable.
    """
    data = _clean()
    data[250] += 50 * float(np.ptp(data))
    quality = _only(_ods(probes={0: data}))

    assert quality.status is ValidationStatus.WARN
    assert _reasons(quality) == {"spike"}
    assert quality.validity_timed[250] == VALIDITY_CERTIFIED
    assert quality.valid_fraction == 1.0
    assert quality.metrics["spike_count"] == 1


def test_a_step_is_an_offset_jump_rather_than_a_spike():
    data = _clean()
    data[300:] += 20 * float(np.ptp(data[:300]))
    quality = _only(_ods(probes={0: data}))

    assert "offset_jump" in _reasons(quality)
    assert quality.metrics["offset_jump_count"] == 1
    # Flagged from the step to the end -- everything after carries an unknown
    # offset -- and still usable, because the finding is a warning.
    assert (quality.validity_timed[300:] == VALIDITY_CERTIFIED).all()
    assert quality.valid_fraction == 1.0


def test_baseline_drift_is_reported_as_a_metric_and_never_as_a_verdict():
    """Deliberately measured, deliberately not judged.

    Drift is only diagnosable against a window where the signal *should* be
    flat, and VEST's magnetics record starts with the PF coils already ramping.
    A leading window that trends is therefore indistinguishable from the
    machine doing its job, so the trend is reported -- #189 asks for baseline
    drift as a metric -- and no event is raised on it.
    """
    walking = _clean()
    walking += 400.0 * float(np.ptp(walking)) * (TIME - TIME[0])
    drifting = _only(_ods(probes={0: walking}))

    assert drifting.metrics["leading_drift_per_second"] > 0.0
    assert "baseline_drift" not in _reasons(drifting)
    assert drifting.valid_fraction == 1.0

    # And a signal that merely swings during its leading window -- which most
    # real VEST channels do -- is not flagged for it either.
    swinging = _only(_ods(probes={0: _clean()}))
    assert swinging.status is ValidationStatus.PASS


def test_a_smooth_ramp_is_not_mistaken_for_a_spike_or_a_step():
    """Plasma breakdown is a fast rise in these signals.  A detector that fires
    on it would zero-weight every channel at exactly the interesting time.
    """
    data = _clean()
    onset = np.clip((TIME - 0.30) / 0.004, 0.0, 1.0)
    data = data + 30.0 * float(np.ptp(data)) * onset
    quality = _only(_ods(probes={0: data}))

    assert "spike" not in _reasons(quality)
    assert quality.valid_fraction == 1.0


def test_a_step_the_whole_array_shares_is_read_as_the_machine():
    """One waveform cannot tell an instrumentation step from breakdown; the
    rest of the array can, because breakdown happens on all of them at once.
    """
    step = np.zeros(N_TIME)
    step[300:] = 1.0
    coherent = {
        index: _clean(seed=index) + 20.0 * float(np.ptp(_clean(seed=index))) * step
        for index in range(6)
    }
    report = validate_magnetics_signals(_ods(probes=coherent))
    assert all("offset_jump" not in _reasons(quality) for quality in report)
    assert all(quality.metrics["coherent_jump_count"] >= 1 for quality in report)

    # The same step on one channel out of six is that channel's problem.
    lonely = {index: _clean(seed=index) for index in range(6)}
    lonely[0] = lonely[0] + 20.0 * float(np.ptp(lonely[0])) * step
    report = validate_magnetics_signals(_ods(probes=lonely))
    assert "offset_jump" in _reasons(report[0])


def test_a_channel_without_a_waveform_is_not_available_rather_than_invalid():
    ods = _ods(probes={0: _clean()})
    ods["magnetics.b_field_pol_probe.1.name"] = "unwired"
    ods["magnetics.b_field_pol_probe.1.field.data"] = np.array([], dtype=float)

    report = validate_magnetics_signals(ods)
    absent = report[1]
    assert absent.status is ValidationStatus.NOT_AVAILABLE
    assert absent.reason
    # Nothing is written for it: "no datum" is not "an invalid datum".
    project_validity(ods, report)
    assert read_validity(ods, channel_node("b_field_pol_probe", 1, "field")) is None


def test_an_invalid_raw_voltage_lowers_the_processed_verdict():
    """Raw and processed validity describe different data and are not assumed
    equal -- but an acquisition that never arrived cannot yield a trustworthy
    processed value either, so the raw verdict seeds the processed one.
    """
    ods = _ods(probes={0: _clean()})
    ods["magnetics.b_field_pol_probe.0.voltage.validity"] = VALIDITY_INVALID

    quality = _only(ods)
    assert quality.validity == VALIDITY_INVALID
    assert quality.reason == "seeded from an invalid raw voltage"


# ---------------------------------------------------------------------------
# 8-9: projection, and the per-time verdict downstream needs
# ---------------------------------------------------------------------------

def test_projection_writes_both_native_nodes_and_the_scalar_summarizes_the_timed():
    data = _clean()
    data[300:] = data[:300].max()
    ods = _ods(probes={0: data}, loops={0: _clean(scale=1.0e-3)})

    written = project_validity(ods, validate_magnetics_signals(ods))
    assert set(written) == {
        "magnetics.b_field_pol_probe.0.field",
        "magnetics.flux_loop.0.flux",
    }

    node = "magnetics.b_field_pol_probe.0.field"
    timed = read_validity_timed(ods, node)
    assert timed.size == N_TIME
    assert read_validity(ods, node) == VALIDITY_INVALID == timed.min()
    # The scalar is a summary, and reading it alone would discard 300 good
    # samples -- which is why the timed field is the authoritative one.
    assert (timed[:300] >= VALIDITY_VALID).all()
    assert read_validity(ods, "magnetics.flux_loop.0.flux") == VALIDITY_VALID


def test_a_channel_valid_early_and_invalid_late_is_excluded_only_late():
    data = _clean()
    data[300:] = data[:300].max()
    ods = _ods(probes={0: data})
    project_validity(ods, validate_magnetics_signals(ods))

    early, late = TIME[100], TIME[400]
    unusable = unusable_channels_at(ods, [early, late])
    assert unusable[("b_field_pol_probe", 0)].tolist() == [False, True]


def test_an_ods_without_validity_excludes_nothing():
    ods = _ods(probes={0: _clean()})
    assert unusable_channels_at(ods, TIME[::100]) == {}


# ---------------------------------------------------------------------------
# 10-11: the two downstream consumers
# ---------------------------------------------------------------------------

def test_eddy_channel_selection_honors_validity_over_its_own_window():
    """The eddy stage validates before breakdown, so a channel that fails
    afterwards is still a good witness for it.
    """
    late_failure = _clean()
    late_failure[400:] = late_failure[:400].max()
    dead = np.zeros(N_TIME)
    ods = _ods(probes={0: _clean(seed=1), 1: late_failure, 2: dead})
    project_validity(ods, validate_magnetics_signals(ods))

    pre_plasma = [
        (row["kind"], row["index"])
        for row in select_vacuum_channels(
            ods, per_family=10, window=(float("-inf"), TIME[350])
        )
    ]
    assert ("b_field_pol_probe", 0) in pre_plasma
    assert ("b_field_pol_probe", 1) in pre_plasma
    # The dead channel is gone whichever way it is read: invalid, and carrying
    # no information for the eddy stage's own precondition either.
    assert ("b_field_pol_probe", 2) not in pre_plasma

    whole_record = [
        (row["kind"], row["index"])
        for row in select_vacuum_channels(ods, per_family=10)
    ]
    assert ("b_field_pol_probe", 1) in whole_record, (
        "a channel with usable samples anywhere in the window stays selectable"
    )


def test_efit_zero_weights_a_channel_only_at_the_slices_it_is_unusable_for():
    data = _clean()
    data[300:] = data[:300].max()
    ods = _ods(probes={0: data, 1: _clean(seed=7)})
    project_validity(ods, validate_magnetics_signals(ods))

    equilibrium = ods["equilibrium"]
    equilibrium["time"] = np.array([TIME[100], TIME[400]])
    for slice_index in (0, 1):
        for channel in (0, 1):
            base = f"time_slice.{slice_index}.constraints.bpol_probe.{channel}"
            equilibrium[f"{base}.measured"] = 1.0e-3
            equilibrium[f"{base}.weight"] = 1.0

    excluded = apply_validity_exclusions(ods, equilibrium)

    assert excluded == {("b_field_pol_probe", 0): [1]}
    assert equilibrium["time_slice.0.constraints.bpol_probe.0.weight"] == 1.0
    assert equilibrium["time_slice.1.constraints.bpol_probe.0.weight"] == 0.0
    assert equilibrium["time_slice.1.constraints.bpol_probe.1.weight"] == 1.0


def test_a_warning_never_costs_a_channel_its_efit_weight():
    spiked = _clean()
    spiked[250] += 50 * float(np.ptp(spiked))
    ods = _ods(probes={0: spiked})
    project_validity(ods, validate_magnetics_signals(ods))

    equilibrium = ods["equilibrium"]
    equilibrium["time"] = np.array([TIME[250]])
    equilibrium["time_slice.0.constraints.bpol_probe.0.measured"] = 1.0e-3
    equilibrium["time_slice.0.constraints.bpol_probe.0.weight"] = 1.0

    assert apply_validity_exclusions(ods, equilibrium) == {}
    assert equilibrium["time_slice.0.constraints.bpol_probe.0.weight"] == 1.0


def test_no_constraint_is_invented_for_a_channel_efit_never_submitted():
    dead = np.zeros(N_TIME)
    ods = _ods(probes={0: _clean(), 1: dead})
    project_validity(ods, validate_magnetics_signals(ods))

    equilibrium = ods["equilibrium"]
    equilibrium["time"] = np.array([TIME[100]])
    equilibrium["time_slice.0.constraints.bpol_probe.0.measured"] = 1.0e-3
    equilibrium["time_slice.0.constraints.bpol_probe.0.weight"] = 1.0

    assert apply_validity_exclusions(ods, equilibrium) == {}
    assert "time_slice.0.constraints.bpol_probe.1.measured" not in equilibrium


# ---------------------------------------------------------------------------
# The detectors, driven directly
# ---------------------------------------------------------------------------

def test_constant_runs_need_bit_identical_samples():
    data = np.concatenate([np.zeros(20), np.arange(1, 21, dtype=float)])
    assert constant_runs(data, 16) == [(0, 20)]
    assert constant_runs(data, 21) == []
    assert constant_runs(np.arange(40, dtype=float), 4) == []


def test_a_spike_plateau_is_one_spike_not_two_steps():
    data = np.zeros(200)
    data[100:102] = 50.0
    assert spike_intervals(data, spike_sigma=8.0, max_spike_samples=3) == [(100, 102)]
    assert offset_jump_samples(data, offset_jump_sigma=10.0, max_spike_samples=3) == []


def test_an_excursion_wider_than_the_spike_budget_is_not_a_spike():
    data = np.zeros(200)
    data[100:130] = 50.0
    assert spike_intervals(data, spike_sigma=8.0, max_spike_samples=3) == []


def test_the_significance_floor_keeps_a_smooth_signal_off_the_detector():
    """These waveforms are integrated and low-pass filtered, so their residual
    against a local median is numerical texture with a minuscule robust scale.
    A purely relative threshold would call every ordinary wiggle an outlier.
    """
    smooth = np.sin(np.linspace(0.0, 6.0, 400))
    assert spike_intervals(smooth, spike_sigma=8.0, max_spike_samples=3) == []
    assert (
        spike_intervals(
            smooth, spike_sigma=8.0, max_spike_samples=3, significance_floor=0.01
        )
        == []
    )


# ---------------------------------------------------------------------------
# Real-data regression
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# A channel the rest of the array contradicts
# ---------------------------------------------------------------------------

def _waveforms(*scales: float) -> dict[int, np.ndarray]:
    return {index: _clean(scale, seed=index) for index, scale in enumerate(scales)}


def _one_family(ods: ODS) -> ODS:
    """Put every channel at one radius, so the whole array is one geometric
    family and the population vote is about amplitude alone."""
    for kind, prefix in (("b_field_pol_probe", "position"), ("flux_loop", "position.0")):
        for index in range(len(ods[f"magnetics.{kind}"]) if f"magnetics.{kind}" in ods else 0):
            ods[f"magnetics.{kind}.{index}.{prefix}.r"] = 0.05 if kind == "b_field_pol_probe" else 0.10
    return ods


def test_a_channel_far_above_its_family_is_condemned_for_the_whole_record():
    """Four probes swinging 0.05 T and one swinging 0.5 T: the loud one is not
    louder physics, it is a fault, and no sample of it is trustworthy."""
    report = validate_magnetics_signals(_one_family(_ods(probes=_waveforms(0.05, 0.05, 0.05, 0.05, 0.5))))
    loud, quiet = report[4], report[:4]
    assert loud.validity == VALIDITY_INVALID
    assert loud.valid_fraction == 0.0
    assert "population_outlier" in _reasons(loud)
    assert loud.metrics["amplitude_over_family_median"] == pytest.approx(10.0, rel=0.05)
    assert all(q.metrics["amplitude_over_family_median"] == pytest.approx(1.0, rel=0.05) for q in quiet)
    assert all("population_outlier" not in _reasons(q) for q in quiet)
    assert all(q.valid_fraction == 1.0 for q in quiet)


def test_a_merely_loud_channel_is_left_alone():
    """2.5x the family median is inside the spread healthy VEST probes show
    (they top out at 2.6x); the factor sits above that on purpose."""
    report = validate_magnetics_signals(_one_family(_ods(probes=_waveforms(0.05, 0.05, 0.05, 0.05, 0.125))))
    assert all("population_outlier" not in _reasons(q) for q in report)


def test_the_physical_ceiling_needs_no_family():
    """A lone probe reading 1.5 T cannot be voted on, and does not need to be:
    no VEST probe can see that field."""
    lone = _only(_ods(probes={0: _clean(1.5)}))
    assert lone.validity == VALIDITY_INVALID
    assert "implausible_magnitude" in _reasons(lone)
    assert "population_outlier" not in _reasons(lone)
    assert "physical ceiling" in lone.reason


def test_flux_loops_vote_but_have_no_ceiling():
    """The tesla ceiling is a probe fact.  A loop is judged only against the
    other loops, so a lone 1.5 Wb loop is simply a loop."""
    report = validate_magnetics_signals(_one_family(_ods(loops=_waveforms(0.05, 0.05, 0.05, 0.05, 1.5))))
    assert "population_outlier" in _reasons(report[4])
    assert "implausible_magnitude" not in _reasons(report[4])
    assert _only(_ods(loops={0: _clean(1.5)})).validity == VALIDITY_VALID


def test_two_channels_cannot_form_a_population():
    report = validate_magnetics_signals(_one_family(_ods(probes=_waveforms(0.05, 0.5))))
    assert all("population_outlier" not in _reasons(q) for q in report)


def test_both_detectors_can_be_switched_off():
    off = MagneticsQualityConfig(max_plausible_amplitude={}, population_peak_factor=None)
    report = validate_magnetics_signals(_one_family(_ods(probes=_waveforms(0.05, 0.05, 0.05, 0.05, 2.0))), config=off)
    assert all(q.validity == VALIDITY_VALID for q in report)


def test_the_pure_detectors_agree_with_their_definitions():
    assert implausible_magnitude(1.2, 1.0)
    assert not implausible_magnitude(0.9, 1.0)
    assert not implausible_magnitude(float("nan"), 1.0)
    assert not implausible_magnitude(5.0, None)
    amplitudes = {"a": 1.0, "b": 1.2, "c": 0.9, "d": 6.0}
    assert population_peak_ratios(amplitudes)["d"] == pytest.approx(6.0 / 1.1)
    assert population_peak_outliers(amplitudes, 4.0) == {"d"}
    assert population_peak_outliers({"a": 1.0, "b": 9.0}, 4.0) == set()
    assert population_peak_outliers(amplitudes, None) == set()


def test_a_dead_majority_cannot_condemn_the_living():
    """Six flatlined loops at a DC offset and five healthy ones: the dead do
    not vote, so the median is the healthy median and nobody is condemned."""
    dead = {index: np.full(N_TIME, 1.0e-4) for index in range(6)}
    alive = {index: _clean(0.03, seed=index) for index in range(6, 11)}
    report = validate_magnetics_signals(_one_family(_ods(loops={**dead, **alive})))
    assert all("population_outlier" not in _reasons(q) for q in report)
    assert all(report[i].valid_fraction == 1.0 for i in range(6, 11))
    assert all(report[i].valid_fraction == 0.0 for i in range(6))


def test_one_glitch_sample_does_not_condemn_a_record():
    """A single 6x sample is a spike -- reported, soft -- not a gain fault.
    The amplitude the population judges is a 99th percentile, not a max."""
    waveforms = _waveforms(0.05, 0.05, 0.05, 0.05, 0.05)
    waveforms[4] = waveforms[4].copy()
    waveforms[4][N_TIME // 2] = 0.3
    report = validate_magnetics_signals(_one_family(_ods(probes=waveforms)))
    assert "spike" in _reasons(report[4])
    assert "population_outlier" not in _reasons(report[4])
    assert report[4].validity == VALIDITY_VALID


def test_a_generator_of_kinds_still_gets_the_population_review():
    ods = _one_family(_ods(probes=_waveforms(0.05, 0.05, 0.05, 0.05, 0.5)))
    report = validate_magnetics_signals(ods, kinds=(k for k in ("b_field_pol_probe",)))
    assert "population_outlier" in _reasons(report[4])


def test_both_causes_and_the_seed_are_all_named():
    ods = _one_family(_ods(probes=_waveforms(0.05, 0.05, 0.05, 0.05, 2.0)))
    ods["magnetics.b_field_pol_probe.4.voltage.validity"] = VALIDITY_INVALID
    loud = validate_magnetics_signals(ods)[4]
    assert "physical ceiling" in loud.reason
    assert "median" in loud.reason
    assert "seeded from an invalid raw voltage" in loud.reason


@pytest.fixture(scope="module")
def packaged():
    from vaft.omas.sample import sample_ods

    ods = sample_ods()
    return ods, validate_magnetics_signals(ods)


def _measuring(report):
    """Channels that carry a processed waveform AND were not condemned outright."""
    return [
        q
        for q in report
        if q.status is not ValidationStatus.NOT_AVAILABLE and q.valid_fraction > 0.0
    ]


def test_the_detector_does_not_reject_known_usable_vest_magnetics(packaged):
    ods, report = packaged
    present = [q for q in report if q.status is not ValidationStatus.NOT_AVAILABLE]
    usable = _measuring(report)

    assert len(present) == 74, "63 B-probes and 11 flux loops carry a processed field"
    assert len(usable) == 73, "all but the one probe the population contradicts"
    assert all(quality.metrics["first_valid_time"] == pytest.approx(0.26) for quality in usable)


def test_the_one_probe_the_population_contradicts_is_rejected_outright(packaged):
    """A real finding.  Probe H3-08 reads 2.7 T on this shot against a
    B-probe population whose median peak is 0.04 T; it is never flatlined
    and never non-finite, so it passed every other detector here and every
    consumer's own filter.  Both the physical ceiling and the family vote
    condemn it, and the verdict is whole-record: a probe's gain does not
    come and go within a shot.
    """
    _ods_, report = packaged
    condemned = [
        q
        for q in report
        if q.status is not ValidationStatus.NOT_AVAILABLE and q.valid_fraction == 0.0
    ]
    assert [q.name for q in condemned] == ["MagneticFieldProbe_H3-08_Bz"]
    (probe,) = condemned
    assert probe.validity == VALIDITY_INVALID
    assert {"implausible_magnitude", "population_outlier"} <= _reasons(probe)
    assert probe.metrics["amplitude"] > 1.0
    assert probe.metrics["amplitude_over_family_median"] > 20.0
    assert "physical ceiling" in probe.reason


def test_the_population_margin_holds_on_the_packaged_shot(packaged):
    """The justification for a hard threshold, kept under test: within every
    geometric family the healthiest channel sits well below the factor and
    the condemned one far above it."""
    _ods_, report = packaged
    factor = MagneticsQualityConfig().population_peak_factor
    voters = [q for q in report if "amplitude_over_family_median" in q.metrics]
    healthy = [q.metrics["amplitude_over_family_median"] for q in voters if q.valid_fraction > 0.0]
    condemned = [q.metrics["amplitude_over_family_median"] for q in voters if q.valid_fraction == 0.0]
    assert max(healthy) < 0.75 * factor
    assert min(condemned) > 5.0 * factor


def test_the_packaged_shot_stops_measuring_at_0_34_seconds(packaged):
    """A real finding, not a detector artifact.

    The equilibrium magnetics are processed over `vest.yaml`'s index window,
    which ends at 0.34 s, and are then interpolated onto the diagnostics grid
    that runs to 0.36 s.  `np.interp` clamps, so the last 500 samples of every
    B-probe and flux loop are a held value rather than a measurement -- which
    is exactly the "which time interval is valid for downstream use?" question
    #189 exists to answer.
    """
    _ods_, report = packaged
    present = _measuring(report)

    assert all("held_tail" in _reasons(quality) for quality in present)
    assert all(
        quality.metrics["last_valid_time"] == pytest.approx(0.34) for quality in present
    )
    assert all(quality.valid_fraction == pytest.approx(0.8, abs=0.01) for quality in present)


def test_the_quality_layer_removes_exactly_the_condemned_probe_from_efit(packaged):
    """Every reconstructed slice of the packaged shot lies inside the measured
    span, so the held tail costs no constraint its weight.  What the quality
    layer does change is the one probe the population contradicts: EFIT
    submitted H3-08 at every slice, and it now loses its weight at every
    slice -- and nothing else does.
    """
    ods, report = packaged
    project_validity(ods, report)

    slice_times = np.asarray(ods["equilibrium.time"], dtype=float)
    assert slice_times.size and slice_times.max() < 0.34

    equilibrium = ods["equilibrium"]
    excluded = apply_validity_exclusions(ods, equilibrium)
    assert set(excluded) == {("b_field_pol_probe", 25)}
    assert excluded[("b_field_pol_probe", 25)] == list(range(slice_times.size))


def test_the_manifest_block_separates_present_usable_and_fully_usable(packaged):
    ods, report = packaged
    metrics = magnetics_quality_metrics(ods, report)

    summary = metrics["summary"]
    assert summary["expected"] == 87
    assert summary["present"] == 74
    # Every present channel but one is usable over part of the record; none is
    # usable over all of it, because they all share the held tail.  The one is
    # the probe the population contradicts, condemned for the whole record.
    assert summary["usable"] == 73
    assert summary["fully_usable"] == 0
    assert summary["events"]["held_tail"] == 74
    assert summary["events"]["implausible_magnitude"] == 1
    assert summary["events"]["population_outlier"] == 1
    assert set(metrics["families"]) >= {"inboard", "outboard", "side", "inboard_flux_loop"}
    assert metrics["configuration"]["significance_floor"] == (
        MagneticsQualityConfig().significance_floor
    )


def test_efit_only_asks_about_the_families_it_submits():
    """The validation layer's kind set is deliberately open; EFIT's is not.

    `unusable_channels_at` defaults to every kind in `QUANTITY_BY_KIND`, and
    that model is documented as ready to take native-rate Mirnov voltage next.
    If EFIT asked for all of them it would `KeyError` on a family its constraint
    tree has no name for -- a hard failure in the reconstruction pipeline caused
    by an edit in an unrelated module.
    """
    from vaft.code.efit.kfile import _EFIT_CONSTRAINT_FAMILY, apply_validity_exclusions

    data = _clean()
    data[300:] = data[:300].max()
    ods = _ods(probes={0: data})
    project_validity(ods, validate_magnetics_signals(ods))
    # A kind the validation layer knows about and EFIT does not.
    ods["magnetics.b_field_tor_probe.0.field.data"] = np.zeros(N_TIME)
    ods["magnetics.b_field_tor_probe.0.field.validity"] = VALIDITY_INVALID

    equilibrium = ods["equilibrium"]
    equilibrium["time"] = np.array([TIME[400]])
    equilibrium["time_slice.0.constraints.bpol_probe.0.measured"] = 1.0e-3
    equilibrium["time_slice.0.constraints.bpol_probe.0.weight"] = 1.0

    assert set(_EFIT_CONSTRAINT_FAMILY) == {"b_field_pol_probe", "flux_loop"}
    assert apply_validity_exclusions(ods, equilibrium) == {("b_field_pol_probe", 0): [0]}
    assert equilibrium["time_slice.0.constraints.bpol_probe.0.weight"] == 0.0


@pytest.mark.parametrize("shot", [41524, 41672])
def test_the_same_probe_is_rejected_on_the_other_packaged_shots(shot):
    """H3-08 is bad in all three packaged shots, at 21-69x the B-probe median;
    a few outboard probes join it on the higher-current shots.  No flux loop
    is ever implicated."""
    import vaft
    import vaft.omas

    try:
        path = vaft.data.sample(shot, "imas")
    except (ValueError, FileNotFoundError):  # repository-only artifact
        pytest.skip(f"sample {shot} is not available in this checkout")
    report = validate_magnetics_signals(vaft.omas.load(path))
    condemned = [
        q
        for q in report
        if q.status is not ValidationStatus.NOT_AVAILABLE and q.valid_fraction == 0.0
    ]
    assert "MagneticFieldProbe_H3-08_Bz" in {q.name for q in condemned}
    assert all(q.kind == "b_field_pol_probe" for q in condemned)
    assert 1 <= len(condemned) <= 8


def test_a_held_tail_does_not_make_a_probe_broken_for_efit(packaged):
    """The scalar validity is "worst state reached", so every probe on the
    packaged shot reads -2 there because of its held tail; the k-file writer
    must judge the time-resolved validity and condemn only the record the
    quality layer rejected outright."""
    from vaft.code.efit.kfile import _condemned_channels

    ods, report = packaged
    project_validity(ods, report)
    assert _condemned_channels(ods, nbprobe=76) == {25}
    from vaft.validation.imas import is_condemned_channel

    assert {
        i
        for i in range(len(ods["magnetics.b_field_pol_probe"]))
        if is_condemned_channel(ods, f"magnetics.b_field_pol_probe.{i}.field")
    } == {25}


def test_every_consumer_agrees_with_the_interpretation_layer(packaged):
    """One rule, one answer (#424): the k-file writer, plot discovery and the
    drawn series must condemn exactly the channels the interpretation layer
    condemns, for every magnetics channel of the packaged shot."""
    from vaft.code.efit.kfile import _condemned_channels
    from vaft.plot.backend.recipes import _validity_of
    from vaft.plot.models import Series
    from vaft.validation.imas import is_condemned_channel

    ods, report = packaged
    project_validity(ods, report)
    nbprobe = len(ods["magnetics.b_field_pol_probe"])
    kfile = _condemned_channels(ods, nbprobe=nbprobe)
    for kind, quantity, offset in (("b_field_pol_probe", "field", 0), ("flux_loop", "flux", nbprobe)):
        for index in range(len(ods[f"magnetics.{kind}"])):
            base = f"magnetics.{kind}.{index}.{quantity}"
            expected = is_condemned_channel(ods, base)
            assert ((index + offset) in kfile) == expected, base
            code, mask = _validity_of(ods, f"magnetics.{kind}.{{i}}.{quantity}.data", index)
            y = np.asarray(ods[f"{base}.data"], dtype=float)
            drawn = Series(x=np.arange(y.size, dtype=float), y=y, validity=code, valid_mask=mask)
            assert drawn.is_invalid_channel == expected, base
