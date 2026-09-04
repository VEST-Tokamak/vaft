"""Plasma-free magnetic-response benchmark for the VEST wall model (issue #190).

The synthetic cases are built so the answer is known by construction: the
"measured" signals are synthesized *from* the same forward model the benchmark
evaluates, so a residual that does not vanish means the benchmark reads the ODS
differently from how the builder wrote it -- not that the physics is hard.

The separation these tests defend is that this layer asks only "does the
measurement agree with the vacuum model?".  It reads the signal validity #189
established and never writes it, and it never re-fits anything to make its own
residual smaller.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
from omas import ODS

from vaft.formula import green_r
from vaft.formula.magnetics import project_poloidal_field
from vaft.machine_mapping.magnetics import POLOIDAL_ANGLE
from vaft.process.electromagnetics import compute_mutual_passive_active
from vaft.omas.process_wrapper import compute_point_response_matrices_ods
from vaft.omas.vacuum_magnetics import synthetic_vacuum_magnetics, vacuum_residual_metrics
from vaft.validation.imas import VALIDITY_INVALID, read_validity
from vaft.validation.vacuum_benchmark import (
    BenchmarkError,
    aggregate_benchmark,
    benchmark_wall_currents,
    plasma_free_interval,
    run_benchmark_case,
    solver_history_check,
    wall_time_constants,
)

N_TIME = 400
TIME = np.linspace(0.20, 0.34, N_TIME)
PROBES = (("inboard_probe", 0.05, 0.10), ("outboard_probe", 0.85, 0.10))
LOOPS = (("inboard_loop", 0.10, 0.30), ("outboard_loop", 0.70, 0.30))
#: Chosen so the wall's slowest mode is ~16 ms -- long enough that the
#: solver-history requirement genuinely bites, short enough that a usable
#: validation window is still left inside the record.
LOOP_RESISTANCE = 3.0e-4
#: Minor radius used for the filament self-inductance on the coupling matrix's
#: diagonal. The Green's function diverges at zero separation, so a passive
#: loop's self term needs a finite conductor size; the value only has to make
#: the matrix well conditioned and the decay physical.
LOOP_MINOR_RADIUS = 0.02


def _self_inductance(major: float, minor: float = LOOP_MINOR_RADIUS) -> float:
    """A circular filament's self-inductance, ``mu0 R (ln(8R/a) - 2)``."""
    return 4.0e-7 * np.pi * major * (np.log(8.0 * major / minor) - 2.0)


def _machine() -> ODS:
    """Two PF coils and two passive loops, driven by a real PF transient."""
    ods = ODS(consistency_check=False)
    ramp = np.clip((TIME - 0.22) / 0.02, 0.0, 1.0)
    currents = np.array([1.0e4 * ramp, -6.0e3 * ramp**2])
    for index, (r, z) in enumerate(((0.15, 0.6), (0.9, -0.4))):
        base = f"pf_active.coil.{index}"
        ods[f"{base}.name"] = f"PF{index}"
        ods[f"{base}.element.0.turns_with_sign"] = 10.0
        ods[f"{base}.element.0.geometry.rectangle.r"] = r
        ods[f"{base}.element.0.geometry.rectangle.z"] = z
        ods[f"{base}.current.data"] = currents[index]
        ods[f"{base}.current.time"] = TIME
    ods["pf_active.time"] = TIME

    for index, (r, z) in enumerate(((0.2, 0.9), (0.95, 0.0))):
        base = f"pf_passive.loop.{index}"
        ods[f"{base}.name"] = f"passive{index}"
        ods[f"{base}.resistance"] = LOOP_RESISTANCE
        ods[f"{base}.element.0.geometry.geometry_type"] = 2
        ods[f"{base}.element.0.geometry.rectangle.r"] = r
        ods[f"{base}.element.0.geometry.rectangle.z"] = z

    # The eddy solver needs the coupling matrices, and the packaged VEST mapper
    # only knows the real machine's ten named coils -- so this fixture supplies
    # its own, from the same Green's functions the machine mapping uses.
    passive = [
        (ods[f"pf_passive.loop.{i}.name"],
         ods[f"pf_passive.loop.{i}.element.0.geometry.rectangle.r"],
         ods[f"pf_passive.loop.{i}.element.0.geometry.rectangle.z"],
         1.04)
        for i in range(len(ods["pf_passive.loop"]))
    ]
    coil_geometry = [
        [(ods[f"pf_active.coil.{i}.element.0.geometry.rectangle.r"],
          ods[f"pf_active.coil.{i}.element.0.geometry.rectangle.z"],
          int(ods[f"pf_active.coil.{i}.element.0.turns_with_sign"]))]
        for i in range(len(ods["pf_active.coil"]))
    ]
    mutual_pp = np.array(
        [
            [
                _self_inductance(r_i) if i == j else green_r(r_i, z_i, r_j, z_j)
                for j, (_n_j, r_j, z_j, _c_j) in enumerate(passive)
            ]
            for i, (_n_i, r_i, z_i, _c_i) in enumerate(passive)
        ]
    )
    ods["em_coupling.mutual_passive_passive"] = mutual_pp
    ods["em_coupling.mutual_passive_active"] = compute_mutual_passive_active(
        passive, coil_geometry
    )

    ods["magnetics.time"] = TIME
    for index, (name, r, z) in enumerate(PROBES):
        base = f"magnetics.b_field_pol_probe.{index}"
        ods[f"{base}.name"] = name
        ods[f"{base}.position.r"] = r
        ods[f"{base}.position.z"] = z
        ods[f"{base}.poloidal_angle"] = POLOIDAL_ANGLE
    for index, (name, r, z) in enumerate(LOOPS):
        base = f"magnetics.flux_loop.{index}"
        ods[f"{base}.name"] = name
        ods[f"{base}.position.0.r"] = r
        ods[f"{base}.position.0.z"] = z
    return ods


def _synthesize(ods: ODS, *, scale: float = 1.0) -> ODS:
    """Fill in measured signals equal to the PF-only coil+wall forward model.

    ``scale`` multiplies the wall contribution only, so a value other than 1
    represents a machine whose wall responds differently from the declared
    model -- which is what the benchmark should then be able to see.
    """
    solved = benchmark_wall_currents(ods)
    n_coil = len(ods["pf_active.coil"])
    n_loop = len(ods["pf_passive.loop"])
    coil_currents = np.array(
        [np.asarray(ods[f"pf_active.coil.{i}.current.data"]) for i in range(n_coil)]
    )
    loop_currents = np.array(
        [np.asarray(solved[f"pf_passive.loop.{i}.current"]) for i in range(n_loop)]
    )

    positions = [(r, z) for _n, r, z in PROBES] + [(r, z) for _n, r, z in LOOPS]
    # The same response path the forward model takes (issue #239): a fixture
    # built on the other implementation would test the ~1e-7 gap between two
    # Green's-function paths rather than the model.
    psi, b_z, b_r = compute_point_response_matrices_ods(
        ods, [[r, z] for r, z in positions], components=("psi", "bz", "br")
    )
    for position, (r, z) in enumerate(positions):
        if position < len(PROBES):
            # The DD's own convention: `poloidal_angle` runs clockwise from +R,
            # so the sensitive axis is (cos, -sin) in (R, Z) -- issue #288. The
            # fixture has to synthesize with the projection the model uses, or
            # it is testing two conventions against each other rather than the
            # forward chain.
            response = (
                project_poloidal_field(b_r[position], b_z[position], POLOIDAL_ANGLE)
            )
            node = f"magnetics.b_field_pol_probe.{position}.field"
        else:
            response = psi[position]
            node = f"magnetics.flux_loop.{position - len(PROBES)}.flux"
        coil = response[:n_coil] @ coil_currents
        eddy = response[n_coil : n_coil + n_loop] @ loop_currents
        ods[f"{node}.data"] = coil + scale * eddy
        ods[f"{node}.time"] = TIME
    return ods


@pytest.fixture
def vacuum_shot() -> ODS:
    """A shot that never forms a plasma: no `magnetics.ip` at all."""
    return _synthesize(_machine())


@pytest.fixture
def plasma_shot(vacuum_shot) -> ODS:
    """The same machine and the same measured magnetics, plus a plasma current.

    The magnetics are deliberately left as the plasma-free ones: the benchmark
    must be unaffected by an Ip waveform outside the accepted interval, and
    keeping the fields identical is what makes that testable.
    """
    ods = copy.deepcopy(vacuum_shot)
    onset = 0.30
    # With a little Rogowski noise, as a real one has: a noiseless baseline
    # gives the sigma detector no band to measure a crossing against, and it
    # would report nothing at all.
    rng = np.random.default_rng(20260901)
    ods["magnetics.ip.0.time"] = TIME
    ods["magnetics.ip.0.data"] = 8.0e4 * np.clip(
        (TIME - onset) / 0.004, 0.0, 1.0
    ) + 30.0 * rng.standard_normal(N_TIME)
    return ods


# ---------------------------------------------------------------------------
# 1, 3: the forward chain
# ---------------------------------------------------------------------------

def test_a_known_coil_and_wall_transient_is_recovered(vacuum_shot):
    case = run_benchmark_case(vacuum_shot, shot=1)

    assert case["case_type"] == "vacuum"
    assert case["metrics"]["summary"]["excluded"] == 0
    # Measured *is* coil+eddy here, so the coil+eddy residual must vanish while
    # the coil-only residual does not -- that difference is the wall model.
    for row in case["metrics"]["channels"]:
        assert row["improvement"] > 0.99, row["name"]
        assert row["correlation"] > 0.999, row["name"]
        assert row["normalized_residual"] < 1.0e-6, row["name"]


def test_b_field_and_flux_observables_stay_in_their_own_units(vacuum_shot):
    case = run_benchmark_case(vacuum_shot, shot=1)
    units = {row["kind"]: row["unit"] for row in case["metrics"]["channels"]}

    assert units == {"b_field_pol_probe": "T", "flux_loop": "Wb"}
    assert {row["family"] for row in case["metrics"]["channels"]} == {
        "inboard",
        "outboard",
        "inboard_flux_loop",
        "outboard_flux_loop",
    }


def test_every_usable_channel_is_evaluated(vacuum_shot):
    case = run_benchmark_case(vacuum_shot, shot=1)
    assert len(case["channels"]["selected"]) == len(PROBES) + len(LOOPS)


# ---------------------------------------------------------------------------
# 2: sensitivity to the wall resistance
# ---------------------------------------------------------------------------

def test_the_wall_decay_time_scales_inversely_with_its_resistance(vacuum_shot):
    """The quantity #190 is most sensitive to, and #117 has not settled."""
    nominal = wall_time_constants(vacuum_shot)
    scaled = wall_time_constants(
        benchmark_wall_currents(vacuum_shot, resistance_scale=4.0)
    )

    assert np.allclose(scaled, nominal / 4.0, rtol=1.0e-8)
    assert nominal[0] > 1.0e-3, "the fixture's wall must be slow enough to matter"


def test_a_wrong_wall_resistance_shows_up_as_a_worse_residual(vacuum_shot):
    """The benchmark must be able to *see* a resistance error, not absorb it.

    That is the whole difference between qualifying a model and fitting one.
    """
    nominal = run_benchmark_case(vacuum_shot, shot=1)
    perturbed = run_benchmark_case(vacuum_shot, shot=1, resistance_scale=3.0)

    assert nominal["metrics"]["summary"]["improvement"]["median"] > (
        perturbed["metrics"]["summary"]["improvement"]["median"]
    )
    assert perturbed["static_model"]["resistance_scale"] == 3.0
    assert perturbed["static_model"]["wall_time_constants"]["slowest"] == pytest.approx(
        nominal["static_model"]["wall_time_constants"]["slowest"]
    ), "the recorded model is the declared one; the scale is reported separately"


# ---------------------------------------------------------------------------
# 4: the assumed initial state
# ---------------------------------------------------------------------------

def test_the_validation_window_opens_only_once_the_initial_state_has_decayed(
    vacuum_shot,
):
    case = run_benchmark_case(vacuum_shot, shot=1)
    solver = case["solver"]

    assert solver["sufficient"] is True
    assert solver["available_history"] >= solver["required_history"]
    assert case["validation_window"][0] > case["solver_input_window"][0]
    # exp(-3) of the assumed zero-current state survives to the window's start.
    assert solver["residual_initial_condition"] == pytest.approx(np.exp(-3.0), rel=0.05)


def test_a_case_without_enough_solver_history_is_rejected_with_the_reason(vacuum_shot):
    with pytest.raises(BenchmarkError, match="solver history"):
        run_benchmark_case(vacuum_shot, shot=1, n_tau=500.0)


def test_an_early_window_is_flagged_rather_than_silently_accepted(vacuum_shot):
    """Reported, not raised: an under-supported case is still informative, and
    a caller may legitimately want to look at one.
    """
    opened_too_early = solver_history_check(vacuum_shot, float(TIME[1]))

    assert opened_too_early["sufficient"] is False
    assert opened_too_early["residual_initial_condition"] > 0.9


# ---------------------------------------------------------------------------
# 5, 6: plasma independence
# ---------------------------------------------------------------------------

def test_a_shot_that_never_formed_a_plasma_is_a_first_class_case(vacuum_shot):
    """The routine eddy stage treats this shot as having nothing to validate;
    for a vacuum-model benchmark it is the cleanest case there is.
    """
    interval = plasma_free_interval(vacuum_shot)

    assert interval.case_type == "vacuum"
    assert (interval.start, interval.end) == (pytest.approx(TIME[0]), pytest.approx(TIME[-1]))
    assert run_benchmark_case(vacuum_shot, shot=1)["metrics"]["summary"]["evaluated"] == 4


def test_a_plasma_shot_contributes_the_stretch_before_breakdown(plasma_shot):
    interval = plasma_free_interval(plasma_shot)

    assert interval.case_type == "pre_plasma"
    assert interval.end < 0.301
    evidence = interval["plasma_free_evidence"]
    # The boundary is the plasma-free boundary of the shared timing policy;
    # this fixture carries no filterscope, so the current's principal pulse
    # answers, and the half-open interval ends exactly there.
    assert evidence["plasma_timing"]["source"] == "ip_principal"
    assert evidence["boundary_source"] == "ip_principal"
    assert evidence["boundary"] == pytest.approx(0.300, abs=1e-3)
    assert interval.end == evidence["boundary"]
    assert evidence["boundary_on_pf_grid"] is True
    # The retired detectors are still reported, for one release.
    assert np.isfinite(evidence["legacy"]["discharge_detector_onset"])
    assert np.isfinite(evidence["legacy"]["sigma_crossing_onset"])
    # And essentially no plasma current is left inside the accepted interval.
    assert evidence["max_abs_ip_in_interval"] < 5.0 * evidence["ip_reference_std"]


def test_the_benchmark_wall_solution_ignores_any_plasma_current(
    vacuum_shot, plasma_shot
):
    """The property that separates this from the routine eddy stage: measured
    Ip and plasma filaments must not drive the wall solve, or the plasma would
    partly explain the response being validated.
    """
    without = benchmark_wall_currents(vacuum_shot)
    with_plasma = benchmark_wall_currents(plasma_shot)

    for index in range(len(vacuum_shot["pf_passive.loop"])):
        path = f"pf_passive.loop.{index}.current"
        assert np.array_equal(
            np.asarray(without[path]), np.asarray(with_plasma[path])
        )


def test_the_source_ods_is_never_modified_by_a_benchmark_run(vacuum_shot, tmp_path):
    """Checked by saving, not by `flat()`.

    An ODS creates paths on access, and a leaf materialized by a bare probe --
    for the `magnetics.ip` this vacuum shot does not have, say -- is invisible
    to `flat()` while still being fatal to the next consistency check.  So the
    assertion has to be that the ODS still survives a round trip.
    """
    from omas import load_omas_json, save_omas_json

    before = set(vacuum_shot.flat())
    run_benchmark_case(vacuum_shot, shot=1)

    assert set(vacuum_shot.flat()) == before
    assert "pf_passive.loop.0.current" not in vacuum_shot.flat()
    assert "ip" not in vacuum_shot["magnetics"].keys()

    path = tmp_path / "unmodified.json"
    save_omas_json(vacuum_shot, str(path))
    load_omas_json(str(path), consistency_check=True)


# ---------------------------------------------------------------------------
# The layer boundary
# ---------------------------------------------------------------------------

def test_a_model_disagreement_never_invalidates_the_source_measurement(vacuum_shot):
    """#253 §10: native validity may only be written by validation whose subject
    is the datum itself.  A channel disagreeing with a forward model is
    evidence about the *model*.
    """
    # A wall whose response *opposes* the declared model: adding the modelled
    # eddy term then makes agreement strictly worse, which is the clearest
    # possible model disagreement.  (An under-scaled wall would still help, so
    # it would not test what this test is about.)
    ods = _synthesize(_machine(), scale=-1.0)
    case = run_benchmark_case(ods, shot=1)

    assert case["metrics"]["summary"]["improvement"]["median"] < 0.0, (
        "the fixture's wall response is deliberately mismodelled"
    )
    for index in range(len(PROBES)):
        assert read_validity(ods, f"magnetics.b_field_pol_probe.{index}.field") is None


def test_a_channel_the_diagnostics_stage_rejected_is_excluded_with_a_reason(
    vacuum_shot,
):
    node = "magnetics.b_field_pol_probe.0.field"
    vacuum_shot[f"{node}.validity"] = VALIDITY_INVALID

    case = run_benchmark_case(vacuum_shot, shot=1)

    assert PROBES[0][0] not in case["channels"]["selected"]
    assert len(case["channels"]["selected"]) == 3
    # Excluded, not zero-filled and not counted as a model failure.
    assert all(
        row["status"] == "evaluated" for row in case["metrics"]["channels"]
    )


def test_a_window_with_too_few_usable_samples_is_excluded_not_failed(vacuum_shot):
    channels = synthetic_vacuum_magnetics(benchmark_wall_currents(vacuum_shot))
    metrics = vacuum_residual_metrics(
        channels, window=(TIME[10], TIME[10]), min_samples=5
    )

    assert metrics["summary"]["evaluated"] == 0
    assert metrics["summary"]["excluded"] == len(channels)
    for row in metrics["channels"]:
        assert row["status"] == "excluded"
        assert "usable sample" in row["reason"]


# ---------------------------------------------------------------------------
# Across cases
# ---------------------------------------------------------------------------

def test_aggregation_exposes_the_axes_a_discrepancy_can_be_attributed_to(
    vacuum_shot, plasma_shot
):
    cases = [
        run_benchmark_case(vacuum_shot, shot=1, machine_era="A"),
        run_benchmark_case(plasma_shot, shot=2, machine_era="B"),
    ]
    aggregate = aggregate_benchmark(cases)

    assert aggregate["case_count"] == 2
    assert set(aggregate["by_case"]) == {"1", "2"}
    assert set(aggregate["by_machine_era"]) == {"A", "B"}
    assert set(aggregate["by_channel"]) == {name for name, _r, _z in PROBES + LOOPS}
    assert set(aggregate["by_excitation"]) == {"PF0,PF1"}
    assert aggregate["summary"]["improved_fraction"] == 1.0


def test_aggregating_nothing_says_so_rather_than_dividing_by_zero():
    assert aggregate_benchmark([])["status"] == "empty"


# ---------------------------------------------------------------------------
# Real-data regression
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def packaged_case():
    from vaft.omas.sample import sample_ods

    return run_benchmark_case(sample_ods(), shot=39915, machine_era="packaged")


def test_the_packaged_shot_reproduces_its_plasma_free_magnetic_response(packaged_case):
    """The real evidence #190 asks for, on the one shot the package ships.

    Not a tuned bound: the assertion is that the passive-wall term explains most
    of what the coils alone cannot, across the full usable channel set, driven
    by measured PF currents alone.
    """
    summary = packaged_case["metrics"]["summary"]

    # 73, not 74: the packaged artifact now carries the quality layer's
    # verdicts (#189), and H3-08 -- 2.7 T on a 0.04 T population -- is
    # rejected by the IDS validity before the benchmark ever sees it.
    assert summary["evaluated"] == 73, "every usable B-probe and flux loop"
    assert summary["excluded"] == 0
    # The window now runs to the plasma onset of the light (0.3063 s), not to
    # the PF-pickup crossing 14 ms earlier (#409), so it covers the strongly
    # driven stretch the old window left out: median improvement 0.78 and
    # median correlation 0.977 over it (0.8+ / 0.99+ over the short window).
    assert summary["improvement"]["median"] > 0.75
    assert summary["improved_fraction"] > 0.9
    assert summary["correlation"]["median"] > 0.97


def test_the_benchmark_evaluates_far_more_than_the_routine_qa_subset(packaged_case):
    """#139's compact per-family subset is right for per-shot QA and too small
    to qualify a machine model: one channel per family cannot separate a sensor
    problem from a wall-model one.
    """
    from vaft.omas.sample import sample_ods

    compact = synthetic_vacuum_magnetics(benchmark_wall_currents(sample_ods()))

    assert len(compact) < 12
    assert len(packaged_case["channels"]["selected"]) == 73
    assert "MagneticFieldProbe_H3-08_Bz" not in packaged_case["channels"]["selected"]


def test_the_packaged_case_records_the_model_revision_it_was_evaluated_against(
    packaged_case,
):
    """A residual is not reproducible without the model that produced it."""
    static = packaged_case["static_model"]

    assert static["passive_loop_count"] == 950
    assert static["pf_coil_count"] == 10
    assert static["resistance_scale"] == 1.0
    assert static["passive_resistance_sum"] > 0.0
    assert static["wall_time_constants"]["slowest"] > static["wall_time_constants"]["fastest"]


def test_the_packaged_case_needs_no_plasma_and_declares_its_evidence(packaged_case):
    assert packaged_case["case_type"] == "pre_plasma"
    evidence = packaged_case["plasma_free_evidence"]

    # The light bounds the interval (39915: 0.3063 s, the slow H-alpha line,
    # consistent with the current), the interval ends on the pf_active grid,
    # and the residual current left inside it is reported against the noise
    # band the detector called it consistent with -- evidence, not a verdict.
    from vaft.validation.vacuum_benchmark import PLASMA_FREE_EVIDENCE_SCHEMA

    assert evidence["schema_version"] == PLASMA_FREE_EVIDENCE_SCHEMA
    timing = evidence["plasma_timing"]
    assert timing["source"] == "h_alpha_primary"
    assert timing["agreement"] == "consistent"
    assert timing["onset"] == pytest.approx(0.3063, abs=5e-4)
    # The plasma-free boundary is the earliest evidence of plasma: here the
    # current's principal pulse, one sample before the light.
    assert evidence["boundary"] <= timing["onset"]
    assert evidence["boundary_source"] == "ip_principal"
    assert packaged_case["solver_input_window"][1] == pytest.approx(evidence["boundary"])
    # Both retired detectors fired on PF pickup, 5.5 and 14 ms before the light.
    legacy = evidence["legacy"]
    assert legacy["sigma_crossing_onset"] < legacy["discharge_detector_onset"] < evidence["boundary"]
    assert evidence["max_abs_ip_in_interval"] < 20.0 * evidence["ip_reference_std"]


def test_the_nominal_wall_resistance_beats_a_doubled_one_on_real_data(packaged_case):
    """Evidence for #117, which the benchmark exists to supply.

    #117 has shown that the stored `pf_passive.loop.*.resistance` values are not
    reproduced by a naive `rho * 2 pi R_mean / area`. This does not settle that,
    but it does say the stored values are approximately right rather than out by
    a factor: agreement is best at the declared resistance and degrades on both
    sides of it. A scan over the packaged shot reads

    ==========  =================  ======================
    ``s_R``     median improvement  median |residual|/range
    ==========  =================  ======================
    0.50        0.742              0.227
    0.75        0.892              0.089
    **1.00**    **0.912**          **0.085**
    1.50        0.713              0.255
    2.00        0.568              0.389
    ==========  =================  ======================

    Only the doubled point is asserted here, to keep the test to one extra
    solve. One shot is not a qualification of the VEST wall model -- #190 says
    so explicitly -- but a benchmark that could *not* see this would be
    absorbing the error rather than measuring it.
    """
    from vaft.omas.sample import sample_ods

    doubled = run_benchmark_case(sample_ods(), shot=39915, resistance_scale=2.0)

    assert doubled["metrics"]["summary"]["improvement"]["median"] < (
        packaged_case["metrics"]["summary"]["improvement"]["median"]
    )
    assert doubled["metrics"]["summary"]["normalized_residual"]["median"] > (
        packaged_case["metrics"]["summary"]["normalized_residual"]["median"]
    )


def test_the_packaged_case_surfaces_the_channels_the_wall_model_fails_on(packaged_case):
    """The benchmark's purpose: not a pass mark, but which channels to look at.

    A negative improvement means adding the wall response made agreement
    worse.  Two channels still show it on this shot -- an outboard probe and
    an inboard flux loop -- and #190 exists to make them visible instead of
    averaging them away.  What is no longer here is the anticorrelated
    channel this test used to find: that was H3-08, a probe reading 2.7 T on
    a 0.04 T population, and the quality layer (#189) now rejects it through
    the artifact's own validity before the benchmark sees it.  A sensor
    fault is not a wall-model finding, and the benchmark should not have to
    rediscover one.
    """
    rows = [r for r in packaged_case["metrics"]["channels"] if r["status"] == "evaluated"]
    names = {row["name"] for row in rows}
    worse = [row for row in rows if row["improvement"] < 0.0]
    anticorrelated = [row for row in rows if row["correlation"] < -0.5]

    assert "MagneticFieldProbe_H3-08_Bz" not in names
    assert 0 < len(worse) <= 5, "a handful, not a systematic failure"
    assert not anticorrelated, "nothing opposes the model once the broken sensor is out"
    assert packaged_case["metrics"]["summary"]["improvement"]["min"] < 0.0
    # Over the window to the light's onset one side probe, C4-04, whose wall
    # term is a third of what it reads, is the one channel the model does not
    # track (correlation -0.44); every other channel is above 0.85.  A finding
    # to keep visible, not a bound to tune away.
    below = sorted(row["name"] for row in rows if row["correlation"] <= 0.7)
    assert below == ["MagneticFieldProbe_C4-04"], below


def test_the_evaluation_window_excludes_its_own_upper_bound(vacuum_shot):
    """Half-open, and it has to be.

    Both callers derive the upper bound from a plasma-onset time, and that time
    is a grid sample -- `vfit_plasma_mgods_startend` returns
    `float(time[start_index])`. An inclusive bound folds the plasma's first
    sample into a nominally plasma-free window, and silently widens the
    pre-plasma statistics the eddy stage has always reported over
    `time < plasma_onset`.
    """
    from vaft.omas.vacuum_magnetics import VacuumChannel, evaluation_mask

    grid = np.linspace(0.0, 1.0, 11)
    channel = VacuumChannel(
        name="c", kind="b_field_pol_probe", family="inboard", index=0,
        r=0.5, z=0.0, unit="T", time=grid,
        measured=np.zeros(11), coil=np.zeros(11), coil_eddy=np.zeros(11),
    )
    onset = float(grid[5])

    assert evaluation_mask(channel, (float("-inf"), onset)).sum() == (grid < onset).sum()
    assert not evaluation_mask(channel, (float("-inf"), onset))[5]


# ---------------------------------------------------------------------------
# Coil drive: a precondition for reading a plasma-free score at all
# ---------------------------------------------------------------------------

def test_a_window_without_coil_drive_is_reported_as_undriven():
    from vaft.validation.vacuum_benchmark import MIN_COIL_DRIVE_FRACTION, coil_drive_check

    ods = ODS(consistency_check=False)
    time = np.linspace(0.0, 1.0, 101)
    ods["pf_active.time"] = time
    ods["pf_active.coil.0.name"] = "PF1"
    ods["pf_active.coil.0.current.data"] = np.where(time >= 0.5, 1000.0 * (time - 0.5), 0.0)

    quiet = coil_drive_check(ods, (0.0, 0.5))
    assert quiet["coil_drive_fraction"] == pytest.approx(0.0)
    assert quiet["sufficiently_driven"] is False
    assert quiet["min_coil_drive_fraction"] == MIN_COIL_DRIVE_FRACTION
    assert "ratio of noise" in quiet["reason"]

    driven = coil_drive_check(ods, (0.5, 1.01))
    assert driven["coil_drive_fraction"] == pytest.approx(1.0)
    assert driven["sufficiently_driven"] is True
    assert driven["reason"] == ""


def test_a_coil_off_the_time_grid_is_listed_not_zeroed():
    from vaft.validation.vacuum_benchmark import coil_drive_check

    ods = ODS(consistency_check=False)
    time = np.linspace(0.0, 1.0, 101)
    ods["pf_active.time"] = time
    ods["pf_active.coil.0.name"] = "PF1"
    ods["pf_active.coil.0.current.data"] = 1000.0 * np.ones(101)
    ods["pf_active.coil.1.name"] = "PF2"
    ods["pf_active.coil.1.current.data"] = 5000.0 * np.ones(57)  # its own grid

    report = coil_drive_check(ods, (0.0, 1.01))
    assert report["skipped_coils"] == ["PF2"]
    assert report["shot_peak_abs_current"] == pytest.approx(5000.0)
    assert report["window_peak_abs_current"] == pytest.approx(1000.0)
    assert "off the pf_active grid" in report["reason"]


def test_a_missing_time_grid_still_reports_every_key():
    from vaft.validation.vacuum_benchmark import coil_drive_check

    report = coil_drive_check(ODS(consistency_check=False), (0.0, 1.0))
    assert report["sufficiently_driven"] is False
    assert report["coil_drive_fraction"] is None
    assert "time grid" in report["reason"]


def test_the_packaged_shot_was_driven_through_its_validation_window(packaged_case):
    """39915's benchmark window opens after the solver-history requirement and
    closes at the plasma onset of the light; the coils reach their shot peak
    inside it (the window to the PF-pickup crossing used to see a third)."""
    from vaft.validation.vacuum_benchmark import MIN_COIL_DRIVE_FRACTION

    drive = packaged_case["coil_drive"]
    assert drive["window"] == packaged_case["validation_window"]
    assert MIN_COIL_DRIVE_FRACTION < drive["coil_drive_fraction"] <= 1.0
    assert drive["coil_drive_fraction"] > 0.9
    assert drive["sufficiently_driven"] is True


def test_a_shot_the_legacy_detector_cut_before_its_solenoid_is_driven_now():
    """41524 is the shot the drive gate was written for: the legacy Ip
    discharge detector fired on PF pickup ~0.8 ms *before* the solenoid, so
    the nominal plasma-free window carried 0.3 % of the shot's coil drive.
    With the interval ending at the light's onset (#409) the window holds the
    whole PF ramp and the retired detector's cut is visible in the evidence.
    The undriven branch itself is kept alive by the synthetic case above."""
    import vaft
    import vaft.omas
    from vaft.validation.vacuum_benchmark import (
        MIN_COIL_DRIVE_FRACTION,
        coil_drive_check,
        plasma_free_interval,
    )

    try:
        path = vaft.data.sample(41524, "imas")
    except (ValueError, FileNotFoundError):  # repository-only artifact
        pytest.skip("sample 41524 is not available in this checkout")
    ods = vaft.omas.load(path)
    interval = plasma_free_interval(ods)
    evidence = interval["plasma_free_evidence"]
    assert evidence["plasma_timing"]["source"] == "h_alpha_primary"
    assert evidence["boundary"] == pytest.approx(0.3146, abs=1e-3)
    assert evidence["legacy"]["discharge_detector_onset"] < evidence["boundary"] - 0.015
    drive = coil_drive_check(ods, (interval.start, interval.end))
    assert drive["coil_drive_fraction"] > MIN_COIL_DRIVE_FRACTION
    assert drive["sufficiently_driven"] is True


def test_the_aggregate_keeps_undriven_cases_out_of_its_spreads():
    from vaft.validation.vacuum_benchmark import aggregate_benchmark

    def case(shot, driven, improvement):
        return {
            "shot": shot,
            "machine_era": "packaged",
            "pf_excitation": {"active_coils": ["PF1"]},
            "coil_drive": {"sufficiently_driven": driven},
            "metrics": {"channels": [
                {"status": "evaluated", "name": "Bp1", "kind": "b_field_pol_probe",
                 "family": "inboard", "improvement": improvement,
                 "normalized_residual": 0.1, "correlation": 0.9, "wall_authority": 0.5},
            ]},
        }

    aggregate = aggregate_benchmark([case(1, True, 0.8), case(2, False, -12.0)])
    assert aggregate["undriven_cases"] == ["2"]
    assert aggregate["channel_rows"] == 2
    assert aggregate["summary"]["driven_channel_rows"] == 1
    assert aggregate["summary"]["median_improvement"] == pytest.approx(0.8)
    assert aggregate["summary"]["median_wall_authority"] == pytest.approx(0.5)


def test_a_current_the_product_marks_unusable_cannot_certify_a_vacuum_case(plasma_shot):
    """Review finding: an invalidated current used to make a plasma shot a
    whole-record vacuum case."""
    from vaft.validation.vacuum_benchmark import BenchmarkError

    plasma_shot["magnetics.ip.0.validity"] = -2
    with pytest.raises(BenchmarkError, match="unusable"):
        plasma_free_interval(plasma_shot)
