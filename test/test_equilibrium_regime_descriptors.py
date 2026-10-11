"""Regime descriptors, export and representative cases (#1891).

Bin boundaries, the unknown event state, the eddy descriptors and the
exported table, all on hand-built ODS -- no HSDS, no EFIT run.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest
from omas import ODS

from vaft.machine_mapping.equilibrium_regime import vest_equilibrium_regime_policy
from vaft.validation import classify_equilibrium_regimes
from vaft.validation.equilibrium_cohorts import (
    cohort_rows,
    join_regime_and_termination,
    representative_cases,
    summarize_by_cohort,
    write_cohort_table,
)

POLICY = vest_equilibrium_regime_policy()


def _ods(currents, times=None):
    ods = ODS(consistency_check=False)
    values = np.asarray(currents, dtype=float)
    grid = np.asarray(times if times is not None else np.arange(values.size) * 1e-3 + 0.30, dtype=float)
    ods["equilibrium.time"] = grid
    for index, value in enumerate(values):
        ods[f"equilibrium.time_slice.{index}.constraints.ip.measured"] = float(value)
    return ods


# --- the configured rule ------------------------------------------------------


def test_the_policy_states_measured_bins_and_a_version():
    assert POLICY.version == 2
    assert POLICY.current_edges == (15e3, 50e3, 100e3, 150e3)
    assert POLICY.rate_edges == (-1e7, -2e6, 2e6, 1e7)
    assert POLICY.status["current_bins"] == POLICY.status["ramp_rate_bins"] == "measured"
    # The first current edge is the vacuum threshold, so the two never disagree.
    assert POLICY.current_edges[0] == POLICY.vacuum_current_amperes


@pytest.mark.parametrize(
    "current, expected",
    [
        (14_999.9, "lt15kA"),
        (15_000.0, "15-50kA"),  # bins are [lower, upper)
        (-49_999.0, "15-50kA"),  # the magnitude, whatever the sign
        (50_000.0, "50-100kA"),
        (100_000.0, "100-150kA"),
        (150_000.0, "ge150kA"),
        (None, "unknown"),
        (float("nan"), "unknown"),
    ],
)
def test_current_bin_boundaries(current, expected):
    assert POLICY.current_bin(current) == expected


@pytest.mark.parametrize(
    "rate, expected",
    [
        (-1.0e7 - 1, "falling_fast"),
        (-1.0e7, "falling"),
        (-2.0e6, "steady"),
        (1.999e6, "steady"),
        (2.0e6, "rising"),
        (1.0e7, "rising_fast"),
        (None, "unknown"),
    ],
)
def test_ramp_rate_bin_boundaries(rate, expected):
    assert POLICY.rate_bin(rate) == expected


# --- the labels ---------------------------------------------------------------


def test_rate_bins_follow_the_magnitude_whatever_the_current_sign():
    rising = [20e3, 40e3, 60e3]  # +20 kA/ms = 20 MA/s
    for sign in (1.0, -1.0):
        labels = classify_equilibrium_regimes(_ods(np.multiply(sign, rising)), policy=POLICY)
        assert [label.rate_bin for label in labels] == ["rising_fast"] * 3
        assert labels[1].magnitude_rate == pytest.approx(2.0e7)


def test_event_is_unknown_without_annotations_and_none_outside_them():
    ods = _ods([50e3, 60e3, 70e3])
    assert {label.event for label in classify_equilibrium_regimes(ods, policy=POLICY)} == {"unknown"}
    annotated = classify_equilibrium_regimes(ods, policy=POLICY, events=[(0.3005, 0.3015, "vde")])
    assert [label.event for label in annotated] == ["none", "vde", "none"]
    # Annotated with no events at all is still an annotation: "none", not "unknown".
    empty = classify_equilibrium_regimes(ods, policy=POLICY, events=[])
    assert {label.event for label in empty} == {"none"}


def test_eddy_descriptors_are_read_at_the_slice_and_absent_outside_the_series():
    ods = _ods([100e3, 100e3, 100e3])
    labels = classify_equilibrium_regimes(
        ods, policy=POLICY, passive_current=([0.3005, 0.3015], [-10e3, -30e3])
    )
    assert labels[0].eddy_current is None  # before the series starts
    assert labels[1].eddy_current == pytest.approx(-20e3)
    assert labels[1].eddy_fraction == pytest.approx(0.2)
    assert labels[2].eddy_current is None


def test_era_comes_from_the_caller_and_defaults_to_unknown():
    ods = _ods([50e3])
    assert classify_equilibrium_regimes(ods, policy=POLICY)[0].era == "unknown"
    assert classify_equilibrium_regimes(ods, policy=POLICY, era="vest-43017-45967-pf1906")[0].era == (
        "vest-43017-45967-pf1906"
    )


def test_limloc_by_millisecond_fills_an_ods_that_has_none():
    labels = classify_equilibrium_regimes(_ods([50e3, 60e3]), policy=POLICY, limloc={300: "IN"})
    assert [label.topology for label in labels] == ["inboard_limited", "unknown"]


# --- the join, the export, the cases -------------------------------------------


def _log(times_ms, accepted):
    return [
        {"time_ms": t, "accepted": ok, "collapsed": False, "solver_errors": [], "warnings": [], "failures": []}
        for t, ok in zip(times_ms, accepted)
    ]


def _join(currents, accepted, statuses, shot):
    ods = _ods(currents)
    times = [300 + i for i in range(len(currents))]
    return join_regime_and_termination(
        ods,
        _log(times, accepted),
        policy=POLICY,
        slice_statuses=[{"time": t / 1e3, "overall_status": s} for t, s in zip(times, statuses)],
        shot=shot,
        source="test",
        era="era-a",
    )


def test_quality_decides_success_over_the_log():
    join = _join([60e3, 60e3, 60e3], [True, True, False], ["usable", "physical_failed", "output_failed"], 1)
    assert [item.succeeded for item in join] == [True, False, False]
    summary = summarize_by_cohort(join, group_by=("era", "current_bin"))
    assert summary["cohorts"]["era-a|50-100kA"]["quality"] == {
        "output_failed": 1,
        "physical_failed": 1,
        "usable": 1,
    }


def test_representative_cases_are_deterministic_and_near_the_cohort_median():
    first = list(_join([60e3, 70e3, 90e3], [True] * 3, ["usable"] * 3, 2))
    second = list(_join([70e3, 65e3], [True] * 2, ["usable", "output_failed"], 1))
    cases = representative_cases(first + second, group_by=("current_bin",))
    again = representative_cases(list(reversed(first + second)), group_by=("current_bin",))
    assert cases == again
    chosen = cases["50-100kA"]["successful"][0]
    # median of the usable currents is 70 kA: shot 1 and shot 2 both have one,
    # and the tie is broken by shot.
    assert (chosen["shot"], chosen["current"]) == (1, 70e3)
    assert cases["50-100kA"]["failed"][0]["quality"] == "output_failed"


def test_csv_export_carries_the_rule(tmp_path):
    rows = cohort_rows(_join([60e3, 120e3], [True, False], ["usable", "output_failed"], 7))
    path = write_cohort_table(rows, tmp_path / "cohorts.csv", policy=POLICY)
    with path.open(encoding="utf-8") as handle:
        read = list(csv.DictReader(handle))
    assert [row["current_bin"] for row in read] == ["50-100kA", "100-150kA"]
    assert {row["classification_version"] for row in read} == {"2"}
    meta = json.loads((tmp_path / "cohorts.csv.meta.json").read_text(encoding="utf-8"))
    assert meta["rows"] == 2
    assert meta["thresholds"]["current_bin_edges_amperes"] == [15e3, 50e3, 100e3, 150e3]


def test_parquet_export_round_trips(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    rows = cohort_rows(_join([60e3], [True], ["usable"], 7))
    path = write_cohort_table(rows, tmp_path / "cohorts.parquet", policy=POLICY)
    frame = pd.read_parquet(path)
    assert frame.loc[0, "quality"] == "usable"
    assert int(frame.loc[0, "classification_version"]) == 2


def test_an_unknown_suffix_is_refused(tmp_path):
    with pytest.raises(ValueError, match="csv or .parquet"):
        write_cohort_table([], tmp_path / "cohorts.xlsx", policy=POLICY)


def test_statuses_are_joined_by_time_and_collisions_refused():
    ods = _ods([60e3, 70e3])
    log = _log([300, 301], [True, True])
    join = join_regime_and_termination(
        ods, log, policy=POLICY,
        slice_statuses=[{"time": 0.301, "overall_status": "usable"}, {"time": None, "overall_status": "x"}],
    )
    assert [item.quality for item in join] == ["unknown", "usable"]
    with pytest.raises(ValueError, match="same millisecond"):
        join_regime_and_termination(
            ods, log, policy=POLICY,
            slice_statuses=[{"time": 0.3001, "overall_status": "a"}, {"time": 0.2999, "overall_status": "b"}],
        )


def test_group_by_accepts_every_real_field_and_refuses_none():
    join = list(_join([60e3], [True], ["usable"], 1))
    assert set(summarize_by_cohort(join, group_by=("mode",))["cohorts"]) == {"accepted"}
    assert set(representative_cases(join, group_by=("mode",))) == {"accepted"}
    with pytest.raises(ValueError, match="at least one"):
        representative_cases(join, group_by=())
    with pytest.raises(ValueError, match="no 'nope'"):
        summarize_by_cohort(join, group_by=("nope",))


def test_the_sidecar_must_describe_the_rule_the_rows_were_drawn_with(tmp_path):
    from dataclasses import replace

    rows = cohort_rows(_join([60e3], [True], ["usable"], 7))
    with pytest.raises(ValueError, match="version"):
        write_cohort_table(rows, tmp_path / "cohorts.csv", policy=replace(POLICY, version=3))
