"""The array review (#977): a B-probe its own array contradicts is condemned.

Probes sharing a radius form an array ordered by height. A healthy array reads
a field that varies smoothly along it; a probe that departs from the
interpolation of its two neighbours by more than the array's own amplitude,
for most of its record, is instrumentation. These tests pin that the review
attributes the right probe -- not a neighbour it dragged -- and that it
survives several faults in one array, which the witness-scaled scorer the
vacuum benchmark uses does not.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omas import ODS

from vaft.validation.magnetics import (
    MagneticsQualityConfig,
    array_contradiction_fractions_of,
    array_departure_ratios,
    attribute_array_contradictions,
    validate_magnetics_signals,
)

N_TIME = 400
TIME = np.linspace(0.26, 0.34, N_TIME)
HEIGHTS = np.round(np.linspace(-0.42, 0.42, 15), 3)


def _profile(z: float) -> float:
    """A smooth, strongly varying field along the array (a factor of ~3 end to end)."""
    return -(1.0 + 2.5 * z**2)


def _array(faults: dict[int, float] | None = None, *, seed: int = 977, r: float = 0.796) -> ODS:
    """One vertical array of B-probes reading ``profile(z) * swing(t)`` plus noise.

    ``faults`` maps a probe index to the factor its reading is multiplied by
    (``-1`` a sign fault, ``5`` a gain fault).
    """
    rng = np.random.default_rng(seed)
    swing = 0.05 * np.sin(np.linspace(0.2, 2.8, N_TIME))
    ods = ODS(consistency_check=False)
    ods["magnetics.time"] = TIME
    for index, z in enumerate(HEIGHTS):
        base = f"magnetics.b_field_pol_probe.{index}"
        data = _profile(z) * swing * (faults or {}).get(index, 1.0)
        ods[f"{base}.name"] = f"probe{index}"
        ods[f"{base}.position.r"] = r
        ods[f"{base}.position.z"] = float(z)
        ods[f"{base}.field.data"] = data + 2.0e-4 * rng.standard_normal(N_TIME)
    return ods


def _condemned_by_array(report) -> list[int]:
    return [q.index for q in report if "array_contradiction" in {e.reason for e in q.events}]


def test_a_healthy_array_with_a_strong_smooth_gradient_is_left_alone():
    report = validate_magnetics_signals(_array(), known_faults={})

    assert _condemned_by_array(report) == []
    assert all(q.valid_fraction == 1.0 for q in report)
    fractions = [q.metrics["array_contradiction_fraction"] for q in report]
    assert max(fractions) < 0.05


def test_a_sign_fault_inside_the_family_is_condemned_and_its_neighbours_are_not():
    """C4-04's case: the reading has the family's size, the wrong sign."""
    report = validate_magnetics_signals(_array({7: -1.0}), known_faults={})

    assert _condemned_by_array(report) == [7]
    probe = report[7]
    assert probe.valid_fraction == 0.0
    assert "departs from its array neighbours" in probe.reason
    # Inside the family by amplitude, so the family review could not have seen it.
    assert "population_outlier" not in {e.reason for e in probe.events}
    assert all(report[i].valid_fraction == 1.0 for i in (6, 8))


def test_two_adjacent_faults_are_both_condemned():
    report = validate_magnetics_signals(_array({5: -1.0, 6: 3.0}), known_faults={})

    assert _condemned_by_array(report) == [5, 6]


def test_a_fault_at_the_end_of_the_array_is_blamed_not_the_probe_next_to_it():
    """An end probe is predicted by extrapolation from the two above it, and it
    drags that pair's departures up with it; leave-one-out must pick the end."""
    report = validate_magnetics_signals(_array({0: -1.0}), known_faults={})

    assert _condemned_by_array(report) == [0]


def test_several_faults_in_one_array_are_all_found():
    """41524's outboard case: with a fifth of the array broken, a scale made of
    the other probes' departures widens until nothing stands out; the array's
    own amplitude does not."""
    # Gains under the family review's 4x, so the array review is the one that
    # has to find them.
    faults = {2: 3.0, 6: -2.0, 10: -1.0}
    ods = _array(faults)
    report = validate_magnetics_signals(ods, known_faults={})
    assert _condemned_by_array(report) == sorted(faults)

    Y = np.array([np.asarray(ods[f"magnetics.b_field_pol_probe.{i}.field.data"]) for i in range(len(HEIGHTS))])
    witness = attribute_array_contradictions(
        HEIGHTS, Y, floor=0.0, sigma=20.0, fraction=0.3, min_members=7, min_scored=0.5, scale="witness",
    )
    assert {entry["index"] for entry in witness} != set(faults)


def test_an_array_shorter_than_the_minimum_is_not_judged():
    ods = _array({3: -1.0})
    keep = 6
    short = ODS(consistency_check=False)
    short["magnetics.time"] = TIME
    for index in range(keep):
        for leaf in ("name", "position.r", "position.z", "field.data"):
            short[f"magnetics.b_field_pol_probe.{index}.{leaf}"] = ods[f"magnetics.b_field_pol_probe.{index}.{leaf}"]

    report = validate_magnetics_signals(short, known_faults={})

    assert keep < MagneticsQualityConfig().min_array_members
    assert _condemned_by_array(report) == []
    assert all("array_contradiction_fraction" not in q.metrics for q in report)


def test_the_review_can_be_switched_off():
    config = MagneticsQualityConfig(array_departure_factor=None)

    report = validate_magnetics_signals(_array({7: -1.0}), known_faults={}, config=config)

    assert _condemned_by_array(report) == []


def test_samples_the_array_may_not_judge_are_not_judged():
    """A stretch masked as not a measurement neither condemns its probe nor
    predicts its neighbours."""
    Y = np.array([_profile(z) * np.linspace(0.01, 0.05, N_TIME) for z in HEIGHTS])
    Y[7] *= -1.0
    Y[7, : int(0.9 * N_TIME)] = np.nan  # scorable on 10 %, under min_scored

    fractions, _mask = array_contradiction_fractions_of(
        HEIGHTS, Y, floor=0.0, sigma=1.0, min_scored=0.5, scale="array_amplitude"
    )
    assert np.isnan(fractions[7])
    assert attribute_array_contradictions(
        HEIGHTS, Y, floor=0.0, sigma=1.0, fraction=0.3, min_members=7, min_scored=0.5,
        scale="array_amplitude",
    ) == []


def test_the_departure_ratio_is_in_units_of_the_array_amplitude():
    Y = np.array([np.full(4, _profile(z)) for z in HEIGHTS]) * 0.05
    Y[7] = -Y[7]

    ratios = array_departure_ratios(HEIGHTS, Y)

    # The probe reads -B where its evenly spaced neighbours predict their mean
    # (about +B): a departure of about 2|B| at the array's quietest point, in
    # units of the array's median |B| -- over one, which is the review's cut.
    predicted = 0.5 * (Y[6, 0] + Y[8, 0])
    assert ratios[7, 0] == pytest.approx(abs(Y[7, 0] - predicted) / np.median(np.abs(Y[:, 0])), rel=1e-9)
    assert ratios[7, 0] > MagneticsQualityConfig().array_departure_factor
    assert np.nanmax(np.delete(ratios, [6, 7, 8], axis=0)) < 0.05


# ---------------------------------------------------------------------------
# The reference shots: the justification for the thresholds, kept under test
# ---------------------------------------------------------------------------

REFERENCE_CONDEMNED = {
    39915: {"MagneticFieldProbe_C4-04"},
    41524: {
        "MagneticFieldProbe_C4-04", "MagneticFieldProbe_C4-02_Bz", "MagneticFieldProbe_C1-01_Bz",
        "MagneticFieldProbe_C3-02_Bz", "MagneticFieldProbe_C4-06", "MagneticFieldProbe_H3-01_Bz",
        "MagneticFieldProbe_L-07_Bz",
    },
    41672: {
        "MagneticFieldProbe_C4-04", "MagneticFieldProbe_C2-04_Bz", "MagneticFieldProbe_C3-03_Bz",
        "MagneticFieldProbe_C4-06", "MagneticFieldProbe_C4-03", "MagneticFieldProbe_H3-01_Bz",
        "MagneticFieldProbe_H3-03_Bz", "MagneticFieldProbe_H3-06_Bz", "MagneticFieldProbe_H3-07_Bz",
        "MagneticFieldProbe_H3-09_Bz", "MagneticFieldProbe_L-07_Bz",
    },
}


def _reference_ods(shot: int):
    import vaft

    if shot == 39915:
        from vaft.omas.sample import sample_ods

        return sample_ods()
    try:
        path = vaft.data.sample(shot, "imas")
    except (ValueError, FileNotFoundError):  # repository-only artifact
        pytest.skip(f"sample {shot} is not available in this checkout")
    return vaft.omas.load(path)


@pytest.mark.parametrize("shot", sorted(REFERENCE_CONDEMNED))
def test_the_array_margin_holds_on_the_reference_shots(shot):
    """Every probe the review condemns sits well above the fraction, every
    survivor well below it (#977: 0.35 and up against 0.24 and down)."""
    report = validate_magnetics_signals(_reference_ods(shot), known_faults={})
    fraction = MagneticsQualityConfig().array_contradiction_fraction
    probes = [q for q in report if q.kind == "b_field_pol_probe"]
    condemned = [q for q in probes if "array_contradiction" in {e.reason for e in q.events}]
    survivors = [
        q.metrics["array_contradiction_fraction"]
        for q in probes
        if q.valid_fraction > 0.0 and np.isfinite(q.metrics.get("array_contradiction_fraction", np.nan))
    ]

    assert {q.name for q in condemned} == REFERENCE_CONDEMNED[shot]
    assert min(q.metrics["array_contradiction_fraction"] for q in condemned) > 1.1 * fraction
    assert max(survivors) < 0.85 * fraction
